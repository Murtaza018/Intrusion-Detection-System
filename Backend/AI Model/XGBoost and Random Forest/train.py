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

    print(f"[+] Loaded train: X={X_train.shape}, y={y_train.shape}")
    print(f"[+] Loaded val:   X={X_val.shape}, y={y_val.shape}")
    print(f"[+] Loaded test:  X={X_test.shape}, y={y_test.shape}")
    print(f"[INFO] Number of classes: {len(np.unique(y_train))}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate(model, X, y, label="TEST"):
    preds = model.predict(X)

    # Use 'weighted' average to handle both Binary and Multiclass safely
    # If strictly binary, 'binary' is default, but 'weighted' works for both without crashing
    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, average='weighted', zero_division=0)
    rec  = recall_score(y, preds, average='weighted', zero_division=0)
    f1   = f1_score(y, preds, average='weighted', zero_division=0)

    print(f"[RESULT] {label}: Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    
    # Optional: Print full report for detailed class breakdown
    # print(classification_report(y, preds))


def print_header(text):
    print("\n" + "="*40)
    print(text)
    print("="*40)


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    print("\n--- 🚀 Initializing Ensemble Training (RF + XGB on CIC-IDS-2017-Processed) ---")

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    if X_train is None:
        exit(1)

    os.makedirs("models_ensemble", exist_ok=True)

    # =====================================================
    # 🌲 TRAIN RANDOM FOREST
    # =====================================================
    print_header("🌲 Training Random Forest...")

    # Note: verbose=2 allows you to see progress inside the model training
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=1,       # Increase this (e.g., -1) for faster training if consistency isn't critical
        verbose=2       # Logic change: Built-in logging is better than a manual loop
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
    print_header("🚀 Training XGBoost...")

    xgb = XGBClassifier(
        objective='binary:logistic', # Ensure your labels are 0 and 1. If multiclass, use 'multi:softmax'
        eval_metric='logloss',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        n_jobs=1,
        # verbosity=1 will print warnings/info. 
        # XGBoost doesn't support a simple CLI progress bar easily without callback functions,
        # but 'verbose=True' in fit() prints evaluation metric history.
    )

    start = time.time()
    
    # We pass eval_set to monitor performance during training, ensuring it's actually learning
    xgb.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=False  # Set to True if you want to see logloss decrease every iteration
    )
    
    end = time.time()
    print(f"[+] XGBoost trained in {end-start:.2f}s\n")

    print("[XGBoost Evaluation]")
    evaluate(xgb, X_val, y_val, "VAL")
    evaluate(xgb, X_test, y_test, "TEST")

    joblib.dump(xgb, "./models_ensemble/xgb_model.joblib")
    print("[+] XGB model saved to: ./models_ensemble/xgb_model.joblib")

    print("\n🎉 TRAINING COMPLETE: Both RF + XGB models saved!\n")