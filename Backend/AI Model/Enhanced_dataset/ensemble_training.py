import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- CONFIG ---
X_ENHANCED_PATH = "X_train_enhanced_95.npy"
Y_TRAIN_PATH = "../Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/y_train.npy"

def retrain_ensemble():
    print("[*] Loading 95-dimension dataset...")
    X = np.load(X_ENHANCED_PATH)
    y = np.load(Y_TRAIN_PATH)

    # 1. Retrain Random Forest
    print("[*] Retraining Random Forest (Adaptive Context)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, "rf_model.joblib")
    print("[+] RF Saved.")

    # 2. Retrain XGBoost
    print("[*] Retraining XGBoost (Anomaly Optimized)...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, use_label_encoder=False)
    xgb.fit(X, y)
    joblib.dump(xgb, "xgb_model.joblib")
    print("[+] XGB Saved.")

    print("\n[***] ENSEMBLE RETRAINING COMPLETE!")
    print("      These models now use GNN and MAE signals for decision making.")

if __name__ == "__main__":
    retrain_ensemble()