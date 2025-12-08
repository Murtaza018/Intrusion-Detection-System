# sweep_ensemble_threshold.py
# Sweep ENSEMBLE_THRESH for your CNN+RF+XGB ensemble on X_val/y_val.

import os
import sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from model_loader import ModelLoader

# ---------- CONFIG ----------
DATA_PATH = "../Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/"
X_VAL_PATH = os.path.join(DATA_PATH, "X_val.npy")
Y_VAL_PATH = os.path.join(DATA_PATH, "y_val.npy")

W_CNN = 0.6
W_RF  = 0.25
W_XGB = 0.15

HIGH_CONF_OVERRIDE = 0.95
THRESHOLDS = np.linspace(0.1, 0.9, 17)  # 0.10, 0.15, ..., 0.90
BATCH_SIZE = 4096
# ----------------------------


def load_data():
    X_val = np.load(X_VAL_PATH)
    y_val = np.load(Y_VAL_PATH).astype(int)
    print(f"[+] Loaded X_val: {X_val.shape}, y_val: {y_val.shape}")
    return X_val, y_val


def load_models():
    ml = ModelLoader()
    if not ml.load_models():
        raise RuntimeError("Failed to load models via ModelLoader.")
    cnn = ml.get_main_model()
    rf  = ml.get_rf_model()
    xgb = ml.get_xgb_model()
    print("[+] CNN, RF, XGB loaded.")
    return cnn, rf, xgb


def compute_ensemble_scores_with_progress(cnn, rf, xgb, X, batch_size=BATCH_SIZE):
    """Apply your ensemble logic to all samples in X with a progress bar."""
    n = X.shape[0]
    cnn_all = []
    rf_all_probs = []
    xgb_all_probs = []

    print(f"[*] Computing model probabilities on {n} samples (batch_size={batch_size})...")
    num_batches = (n + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        batch = X[start:end]

        # CNN
        cnn_probs = cnn.predict(batch, verbose=0).reshape(-1)
        cnn_all.append(cnn_probs)

        # RF / XGB
        rf_probs_batch  = rf.predict_proba(batch)
        xgb_probs_batch = xgb.predict_proba(batch)
        rf_all_probs.append(rf_probs_batch)
        xgb_all_probs.append(xgb_probs_batch)

        # progress bar
        done = end
        frac = done / n
        bar_len = 30
        filled = int(bar_len * frac)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r    [{bar}] {done}/{n} ({frac*100:5.1f}%)")
        sys.stdout.flush()

    sys.stdout.write("\n")

    cnn_all = np.concatenate(cnn_all, axis=0)
    rf_all_probs = np.concatenate(rf_all_probs, axis=0)
    xgb_all_probs = np.concatenate(xgb_all_probs, axis=0)

    # Convert to attack probabilities
    rf_probs  = 1.0 - rf_all_probs[:, 0]
    xgb_probs = 1.0 - xgb_all_probs[:, 0]

    # High-conf override + weighted vote
    p_ens = np.zeros_like(cnn_all)
    high_conf = np.maximum.reduce([cnn_all, rf_probs, xgb_probs])

    override_mask = high_conf >= HIGH_CONF_OVERRIDE
    p_ens[override_mask] = high_conf[override_mask]

    normal_mask = ~override_mask
    p_ens[normal_mask] = (
        W_CNN * cnn_all[normal_mask] +
        W_RF  * rf_probs[normal_mask] +
        W_XGB * xgb_probs[normal_mask]
    )

    return p_ens, cnn_all, rf_probs, xgb_probs


def sweep_thresholds(y_true, p_ens):
    print("Threshold sweep (ensemble):")
    print("T\tPrecision\tRecall\t\tF1\t\tFPR")

    neg = (y_true == 0)
    neg_count = neg.sum()

    for T in THRESHOLDS:
        y_pred = (p_ens >= T).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        fp = (neg & (y_pred == 1)).sum()
        fpr = fp / neg_count if neg_count > 0 else 0.0

        print(f"{T:.2f}\t{precision:.3f}\t\t{recall:.3f}\t\t{f1:.3f}\t\t{fpr:.4f}")

    try:
        auc = roc_auc_score(y_true, p_ens)
        print(f"\nEnsemble ROC-AUC: {auc:.4f}")
    except Exception:
        pass


def main():
    X_val, y_val = load_data()
    cnn, rf, xgb = load_models()

    p_ens, cnn_probs, rf_probs, xgb_probs = compute_ensemble_scores_with_progress(
        cnn, rf, xgb, X_val, batch_size=BATCH_SIZE
    )
    print("\n[+] Finished computing ensemble scores. Sweeping thresholds...\n")

    sweep_thresholds(y_val, p_ens)


if __name__ == "__main__":
    main()
