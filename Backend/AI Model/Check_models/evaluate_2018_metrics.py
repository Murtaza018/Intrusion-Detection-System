import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

# --- 1. Define Paths ---
PROCESSED_DATA_DIR = "../Preprocessing/Processed_2018_Test"
CNN_PATH = "../Adversarial Attack and Defense/cicids_spatiotemporal_model_hardened.keras" # Update if different
RF_PATH = "../XGBoost and Random Forest/models_ensemble/rf_model.joblib" # Update if different
XGB_PATH = "../XGBoost and Random Forest/models_ensemble/xgb_model.joblib" # Update if different
AE_PATH = "../Autoencoder/cicids_autoencoder.keras" # Update if different

# The thresholds we calculated earlier
AWS_STANDARD_AE_THRESHOLD = 0.083687

def main():
    print("[!] INITIATING 2018 PIPELINE BENCHMARK")

    # --- 2. Load Data ---
    print("[*] Loading 2018 Preprocessed Data...")
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test_2018.npy"))
    y_test_strings = np.load(os.path.join(PROCESSED_DATA_DIR, "y_test_2018.npy"), allow_pickle=True)

    # Convert string labels to binary for global metrics (0 = Normal, 1 = Attack/Zero-Day)
    y_true = np.where(y_test_strings == 'BENIGN', 0, 1)

    # --- 3. Load Models ---
    print("[*] Loading Models (This may take a moment)...")
    cnn_model = tf.keras.models.load_model(CNN_PATH)
    rf_model = joblib.load(RF_PATH)
    xgb_model = joblib.load(XGB_PATH)
    ae_model = tf.keras.models.load_model(AE_PATH)

    # --- 4. Batch Processing (To prevent RAM crashes on 5.2M rows) ---
    print(f"[*] Starting Batch Evaluation on {len(X_test)} flows...")
    batch_size = 50000
    y_pred = []

    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i + batch_size]
        
        # A. Get Supervised Probabilities
        # CNN output
        cnn_probs = cnn_model.predict(X_batch, batch_size=4096, verbose=0).flatten()
        
        # RF/XGB outputs (assuming [:, 1] is the probability of an attack)
        rf_probs = rf_model.predict_proba(X_batch)[:, 1]
        xgb_probs = xgb_model.predict_proba(X_batch)[:, 1]

        # Max Ensemble Logic
        ensemble_probs = np.maximum.reduce([cnn_probs, rf_probs, xgb_probs])

        # B. Get Autoencoder Anomaly Scores (using Absolute Error to match your training)
        reconstructions = ae_model.predict(X_batch, batch_size=4096, verbose=0)
        ae_errors = np.mean(np.abs(X_batch - reconstructions), axis=1)

        # C. Pipeline Decision Logic
        batch_preds = []
        for ens_prob, ae_err in zip(ensemble_probs, ae_errors):
            if ens_prob > 0.40:
                batch_preds.append(1) # Known Attack
            elif ae_err > AWS_STANDARD_AE_THRESHOLD:
                batch_preds.append(1) # Zero-Day Anomaly
            else:
                batch_preds.append(0) # Normal

        y_pred.extend(batch_preds)
        print(f"    [-] Processed {min(i + batch_size, len(X_test))} / {len(X_test)} flows...")

    y_pred = np.array(y_pred)

    # --- 5. Output Metrics ---
    print("\n" + "="*50)
    print(" 2018 PIPELINE BENCHMARK RESULTS")
    print("="*50)
    
    print(f"\nOverall Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    
    print("\nClassification Report (0 = Normal, 1 = Attack/Zero-Day):")
    print(classification_report(y_true, y_pred, digits=4))
    
    print("\nConfusion Matrix:")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"True Negatives (Correctly Normal): {tn:,}")
    print(f"False Positives (Normal flagged as Attack): {fp:,}")
    print(f"False Negatives (Missed Attacks): {fn:,}")
    print(f"True Positives (Caught Attacks): {tp:,}")
    print("="*50)

if __name__ == "__main__":
    main()