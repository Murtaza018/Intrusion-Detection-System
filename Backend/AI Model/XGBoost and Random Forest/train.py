import os
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

# =====================================================
# Helpers
# =====================================================

def load_data():
    print("\n[*] Loading processed CIC-IDS-2017 data...")
    
    # Check if files exist to prevent obscure errors
    base_path = "../Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/"
    if not os.path.exists(base_path):
        print(f"[!] Error: Path {base_path} not found.")
        return None, None, None, None, None, None

    X_train = np.load(os.path.join(base_path, "X_train.npy"))
    y_train = np.load(os.path.join(base_path, "y_train.npy"))
    X_val   = np.load(os.path.join(base_path, "X_val.npy"))
    y_val   = np.load(os.path.join(base_path, "y_val.npy"))
    X_test  = np.load(os.path.join(base_path, "X_test.npy"))
    y_test  = np.load(os.path.join(base_path, "y_test.npy"))

    # ---------------------------------------------------------
    # [!] THE FIX: Enforce strictly 78 features for the Cascade
    # ---------------------------------------------------------
    if X_train.shape[1] > 78:
        print(f"[*] Notice: Truncating features from {X_train.shape[1]} down to exactly 78...")
        X_train = X_train[:, :78]
        X_val   = X_val[:, :78]
        X_test  = X_test[:, :78]
    elif X_train.shape[1] < 78:
        print(f"[!] Warning: Data only has {X_train.shape[1]} features, expected 78!")

    print(f"[+] Loaded train: X={X_train.shape}, y={y_train.shape}")
    print(f"[+] Loaded val:   X={X_val.shape}, y={y_val.shape}")
    print(f"[+] Loaded test:  X={X_test.shape}, y={y_test.shape}")
    print(f"[INFO] Number of classes: {len(np.unique(y_train))}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate(model, X, y, label="TEST"):
    preds = model.predict(X)

    # Use 'weighted' average to handle both Binary and Multiclass safely
    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, average='weighted', zero_division=0)
    rec  = recall_score(y, preds, average='weighted', zero_division=0)
    f1   = f1_score(y, preds, average='weighted', zero_division=0)

    print(f"[RESULT] {label}: Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")


def print_header(text):
    print("\n" + "="*40)
    print(text)
    print("="*40)


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    print("\n--- 🚀 Initializing Ensemble Training (78-Feature Cascade Mode) ---")

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    if X_train is None:
        exit(1)

    os.makedirs("models_ensemble", exist_ok=True)

    # Determine if we are doing Binary or Multiclass for XGBoost
    num_classes = len(np.unique(y_train))
    xgb_objective = 'binary:logistic' if num_classes == 2 else 'multi:softprob'

    # =====================================================
    # 🌲 TRAIN RANDOM FOREST
    # =====================================================
    print_header("🌲 Training Random Forest (78 Features)...")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,      # Sped up: uses all available CPU cores
        verbose=2 
    )

    start = time.time()
    rf.fit(X_train, y_train)
    end = time.time()

    print(f"\n[+] Random Forest trained in {end-start:.2f}s")

    print("[Random Forest Evaluation]")
    evaluate(rf, X_val, y_val, "VAL")
    evaluate(rf, X_test, y_test, "TEST")

    joblib.dump(rf, "./models_ensemble/rf_model.joblib")
    print("[+] RF model saved to: ./models_ensemble/rf_model.joblib")

    # =====================================================
    # 🚀 TRAIN XGBOOST
    # =====================================================
    print_header("🚀 Training XGBoost (78 Features)...")

    xgb = XGBClassifier(
        objective=xgb_objective, 
        eval_metric='mlogloss' if num_classes > 2 else 'logloss',
        num_class=num_classes if num_classes > 2 else None,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        n_jobs=-1,       # Sped up: uses all available CPU cores
    )

    start = time.time()
    
    xgb.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=False  
    )
    
    end = time.time()
    print(f"[+] XGBoost trained in {end-start:.2f}s\n")

    print("[XGBoost Evaluation]")
    evaluate(xgb, X_val, y_val, "VAL")
    evaluate(xgb, X_test, y_test, "TEST")

    joblib.dump(xgb, "./models_ensemble/xgb_model.joblib")
    print("[+] XGB model saved to: ./models_ensemble/xgb_model.joblib")

    print("\n🎉 TRAINING COMPLETE: Both 78-Feature RF + XGB models saved!\n")