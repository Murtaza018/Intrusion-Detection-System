import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

# --- Paths ---
PROCESSED_DATA_DIR = "../Preprocessing/Processed_2018_Test"
CNN_PATH = "../Adversarial Attack and Defense/cicids_spatiotemporal_model_hardened.keras" 
RF_PATH = "../XGBoost and Random Forest/models_ensemble/rf_model.joblib" # Update if different
XGB_PATH = "../XGBoost and Random Forest/models_ensemble/xgb_model.joblib" # Update if different
AE_PATH = "../Autoencoder/cicids_autoencoder.keras" # Update if different
MASKED_AE_PATH = "../MAE/mae_visual_engine.pth"

# --- Thresholds & Config ---
AWS_STANDARD_AE_THRESHOLD = 0.083687
AWS_MASKED_AE_THRESHOLD = 0.063120
ENSEMBLE_CONFIDENCE_THRESHOLD = 0.75 # Raised to stop supervised models from panicking
SAMPLE_SIZE = 100000

# --- PyTorch MAE Class Definition (Required to load weights) ---
class MAEModel(nn.Module):
    def __init__(self, input_dim=78, grid_size=9):
        super(MAEModel, self).__init__()
        self.grid_size = grid_size
        self.total_pixels = grid_size * grid_size 
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 32 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=1, padding=0), 
            nn.Sigmoid()
        )

    def forward(self, x, mask_ratio=0.5):
        batch_size = x.shape[0]
        padding = torch.zeros((batch_size, self.total_pixels - self.input_dim)).to(x.device)
        x_padded = torch.cat([x, padding], dim=1)
        x_img = x_padded.view(-1, 1, self.grid_size, self.grid_size)
        if self.training or mask_ratio > 0:
            mask = torch.rand(x_img.shape).to(x.device) > mask_ratio
            x_masked = x_img * mask
        else:
            x_masked = x_img
        latent = self.encoder(x_masked)
        reconstruction = self.decoder(latent)
        return reconstruction, x_img

def main():
    print("[!] INITIATING DUAL-AE CASCADE BENCHMARK (SUBSET)")

    # 1. Load Data
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test_2018.npy"))
    y_test_strings = np.load(os.path.join(PROCESSED_DATA_DIR, "y_test_2018.npy"), allow_pickle=True)
    y_true_all = np.where(y_test_strings == 'BENIGN', 0, 1)

    # 2. Extract a random subset to save RAM and time
    print(f"[*] Extracting random subset of {SAMPLE_SIZE} flows...")
    np.random.seed(42)
    indices = np.random.choice(len(X_test), SAMPLE_SIZE, replace=False)
    X_subset = X_test[indices]
    y_subset = y_true_all[indices]

    # 3. Load Models
    print("[*] Loading Models...")
    cnn_model = tf.keras.models.load_model(CNN_PATH)
    rf_model = joblib.load(RF_PATH)
    xgb_model = joblib.load(XGB_PATH)
    ae_model = tf.keras.models.load_model(AE_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mae_model = MAEModel().to(device)
    mae_model.load_state_dict(torch.load(MASKED_AE_PATH, map_location=device, weights_only=True))
    mae_model.eval()

    y_pred = []

    print("[*] Running Cascade Detection Pipeline...")
    
    # We will evaluate in smaller chunks within the subset to handle PyTorch memory
    batch_size = 5000
    for i in range(0, len(X_subset), batch_size):
        X_batch = X_subset[i:i + batch_size]
        
        # --- A. Supervised Ensemble (The Fast Guard) ---
        cnn_probs = cnn_model.predict(X_batch, batch_size=1024, verbose=0).flatten()
        
        # NOTE: If your RF/XGB models expect 95 features (with GNN/MAE appended), 
        # we have to bypass them here since we only have 78 features and no GNN graph.
        # Assuming for this benchmark we rely on CNN for the supervised known attacks:
        ensemble_probs = cnn_probs 

        # --- B. Standard Autoencoder (MAE loss) ---
        ae_recon = ae_model.predict(X_batch, batch_size=1024, verbose=0)
        ae_errors = np.mean(np.abs(X_batch - ae_recon), axis=1)

        # --- C. Masked Autoencoder (MSE loss) ---
        X_tensor = torch.FloatTensor(X_batch).to(device)
        with torch.no_grad():
            mae_recon, mae_original = mae_model(X_tensor, mask_ratio=0.0)
            mae_errors = torch.mean(torch.pow(mae_original - mae_recon, 2), dim=[1, 2, 3]).cpu().numpy()

        # --- D. The High-Confidence Logic ---
        for ens_prob, ae_err, mae_err in zip(ensemble_probs, ae_errors, mae_errors):
            if ens_prob > ENSEMBLE_CONFIDENCE_THRESHOLD:
                y_pred.append(1) # High confidence known attack
            elif mae_err > AWS_MASKED_AE_THRESHOLD:
                y_pred.append(1) # MAE Structural Zero-Day
            elif ae_err > AWS_STANDARD_AE_THRESHOLD:
                y_pred.append(1) # Standard AE Anomaly
            else:
                y_pred.append(0) # Truly Normal

    y_pred = np.array(y_pred)

    print("\n" + "="*50)
    print(" DUAL-AE CASCADE BENCHMARK RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {accuracy_score(y_subset, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_subset, y_pred, digits=4))
    
    tn, fp, fn, tp = confusion_matrix(y_subset, y_pred).ravel()
    print("\nConfusion Matrix:")
    print(f"True Negatives:  {tn:,}")
    print(f"False Positives: {fp:,} (Normal flagged as attack)")
    print(f"False Negatives: {fn:,} (Missed attacks)")
    print(f"True Positives:  {tp:,}")
    print("="*50)

if __name__ == "__main__":
    main()