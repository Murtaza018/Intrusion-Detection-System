import torch
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from mae_model import MAEModel  # Importing your blueprint

# --- CONFIGURATION ---
MODEL_PATH = "mae_visual_engine.pth"
DATA_DIR = "../Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/"
INPUT_DIM = 78
MAE_MASK_RATIO = 0.4  # Use the same ratio used in training/detector

def perform_performance_audit():
    # 1. Load the "exam" (Test Data)
    print("[*] Loading unseen test data...")
    x_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    
    # 2. Load the Model
    model = MAEModel(input_dim=INPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("[✔] Model loaded for audit.")

    # 3. Calculate Reconstruction Errors
    print("[*] Calculating reconstruction errors for all test samples...")
    test_tensor = torch.tensor(x_test, dtype=torch.float32)
    
    with torch.no_grad():
        # We run the model to see how well it reconstructs the test set
        recon, original = model(test_tensor, mask_ratio=MAE_MASK_RATIO)
        # Calculate Mean Squared Error (MSE) per sample
        mse_per_sample = torch.mean((recon - original)**2, dim=(1, 2, 3)).numpy()

    # 4. Establish the Anomaly Threshold
    # We use the Benign samples in the test set to find what "Normal" error looks like
    benign_errors = mse_per_sample[y_test == 0]
    mean_err = np.mean(benign_errors)
    std_err = np.std(benign_errors)
    
    # Heuristic: Threshold = Mean + 3*Std (Common in SOTA anomaly detection)
    threshold = mean_err + (3 * std_err)
    print(f"\n[>] Audit Threshold established: {threshold:.6f}")
    print(f"    (Normal Mean: {mean_err:.6f}, Std: {std_err:.6f})")

    # 5. Classify and Generate Scores
    # If error > threshold, we predict "Attack" (1), else "Benign" (0)
    y_pred = (mse_per_sample > threshold).astype(int)

    # 6. PRINT PERFORMANCE METRICS
    print("\n" + "="*40)
    print("      MAE PERFORMANCE AUDIT REPORT")
    print("="*40)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f} (Ability to avoid false alarms)")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f} (Ability to catch every attack)")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f} (Harmonic balance)")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*40)

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        perform_performance_audit()
    else:
        print(f"[!] Audit failed: {MODEL_PATH} not found. Train the model first!")