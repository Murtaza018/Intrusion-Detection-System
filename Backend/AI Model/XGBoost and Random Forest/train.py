# train_ensemble.py
# Trains Random Forest and XGBoost models for the Hybrid Ensemble IDS

import numpy as np
import os
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

print("--- 🚀 Initializing Ensemble Training ---")

# --- CONFIGURATION ---
# Adjust this path to match where your .npy files are!
# Based on your previous messages, it seems to be here:
DATA_PATH = "../Preprocessing/CGAN/CGAN_preprocessed_data/" 

MODELS_DIR = "./models_ensemble"
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 1. LOAD DATA ---
print(f"\n[*] Loading dataset from {DATA_PATH}...")
try:
    X = np.load(os.path.join(DATA_PATH, 'X_full.npy'))
    y = np.load(os.path.join(DATA_PATH, 'y_full.npy'))
    print(f"[+] Dataset loaded: {len(X):,} samples.")
except FileNotFoundError:
    print(f"[!] CRITICAL ERROR: Could not find .npy files in {DATA_PATH}")
    exit()

# Split data for validation (important to check accuracy!)
print("[*] Splitting data 80/20...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. TRAIN RANDOM FOREST ---
print("\n" + "="*40)
print("🌲 Training Random Forest...")
print("="*40)
start_time = time.time()

# Optimizing for speed: n_jobs=-1 uses all CPU cores
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=20, 
    n_jobs=-1, 
    random_state=42,
    verbose=1
)
rf_model.fit(X_train, y_train)

print(f"[+] Random Forest trained in {time.time() - start_time:.2f}s")

# Evaluate
val_preds = rf_model.predict(X_val)
acc = accuracy_score(y_val, val_preds)
print(f"[RESULT] Random Forest Accuracy: {acc:.4f}")

# Save
rf_path = os.path.join(MODELS_DIR, "rf_model.joblib")
joblib.dump(rf_model, rf_path)
print(f"[+] Saved to: {rf_path}")

# --- 3. TRAIN XGBOOST ---
print("\n" + "="*40)
print("🚀 Training XGBoost...")
print("="*40)
start_time = time.time()

# Tree method 'hist' is much faster for large datasets
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    tree_method='hist', 
    device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu', # Use GPU if available
    random_state=42
)
xgb_model.fit(X_train, y_train)

print(f"[+] XGBoost trained in {time.time() - start_time:.2f}s")

# Evaluate
val_preds = xgb_model.predict(X_val)
acc = accuracy_score(y_val, val_preds)
print(f"[RESULT] XGBoost Accuracy: {acc:.4f}")

# Save
xgb_path = os.path.join(MODELS_DIR, "xgb_model.joblib")
joblib.dump(xgb_model, xgb_path)
print(f"[+] Saved to: {xgb_path}")

print("\n" + "="*40)
print("🎉 ENSEMBLE TRAINING COMPLETE!")
print("="*40)