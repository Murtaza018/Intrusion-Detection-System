# check_cnn_calibration.py
# Evaluate how well the CNN's probabilities match reality on X_val/y_val.

import os
import sys
import numpy as np
from sklearn.metrics import brier_score_loss

# Adjust these paths if needed
DATA_PATH = "../Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/"
X_VAL_PATH = os.path.join(DATA_PATH, "X_val.npy")
Y_VAL_PATH = os.path.join(DATA_PATH, "y_val.npy")

from model_loader import ModelLoader


def load_data():
    X_val = np.load(X_VAL_PATH)
    y_val = np.load(Y_VAL_PATH).astype(int)
    print(f"[+] Loaded X_val: {X_val.shape}, y_val: {y_val.shape}")
    return X_val, y_val


def load_cnn():
    ml = ModelLoader()
    if not ml.load_models():
        raise RuntimeError("Failed to load models via ModelLoader.")
    cnn = ml.get_main_model()
    print("[+] CNN model loaded.")
    return cnn


def predict_with_progress(model, X, batch_size=4096):
    """Run model.predict with a simple progress bar."""
    n = X.shape[0]
    probs_list = []

    print(f"[*] Getting CNN probabilities on {n} samples (batch_size={batch_size})...")
    num_batches = (n + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        batch = X[start:end]

        batch_probs = model.predict(batch, verbose=0).reshape(-1)
        probs_list.append(batch_probs)

        # progress bar
        done = end
        frac = done / n
        bar_len = 30
        filled = int(bar_len * frac)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r    [{bar}] {done}/{n} ({frac*100:5.1f}%)")
        sys.stdout.flush()

    sys.stdout.write("\n")
    return np.concatenate(probs_list, axis=0)


def main():
    X_val, y_val = load_data()
    cnn = load_cnn()

    probs = predict_with_progress(cnn, X_val, batch_size=4096)

    # 1) Bin-wise calibration table
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print("\nBin\tMean p\tEmpirical attack rate\tCount")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        count = mask.sum()
        if count == 0:
            continue
        mean_p = probs[mask].mean()
        emp_rate = y_val[mask].mean()
        print(f"[{lo:.1f},{hi:.1f})\t{mean_p:.3f}\t{emp_rate:.3f}\t{count}")

    # 2) Brier score
    brier = brier_score_loss(y_val, probs)
    print(f"\nBrier score (lower is better): {brier:.4f}")

    # 3) High-confidence region
    high_mask = probs >= 0.90
    high_count = high_mask.sum()
    if high_count > 0:
        high_emp_rate = y_val[high_mask].mean()
        print(f"\nHigh-confidence region (p >= 0.90):")
        print(f"  Samples: {high_count}")
        print(f"  Empirical attack rate: {high_emp_rate:.3f}")
    else:
        print("\nNo samples with p >= 0.90 in validation set.")


if __name__ == "__main__":
    main()
