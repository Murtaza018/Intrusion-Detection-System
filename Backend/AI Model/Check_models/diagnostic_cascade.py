import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
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
ENSEMBLE_CONFIDENCE_THRESHOLD = 0.75 
SAMPLE_SIZE = 100000

# --- PyTorch MAE Class Definition ---
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
    print("[!] INITIATING DIAGNOSTIC WIRETAP ON CASCADE PIPELINE")

    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test_2018.npy"))
    y_test_strings = np.load(os.path.join(PROCESSED_DATA_DIR, "y_test_2018.npy"), allow_pickle=True)
    y_true_all = np.where(y_test_strings == 'BENIGN', 0, 1)

    print(f"[*] Extracting random subset of {SAMPLE_SIZE} flows...")
    np.random.seed(42)
    indices = np.random.choice(len(X_test), SAMPLE_SIZE, replace=False)
    X_subset = X_test[indices]
    y_subset = y_true_all[indices]

    print("[*] Loading Models (Silencing TF logs)...")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silence TF spam
    cnn_model = tf.keras.models.load_model(CNN_PATH)
    ae_model = tf.keras.models.load_model(AE_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mae_model = MAEModel().to(device)
    mae_model.load_state_dict(torch.load(MASKED_AE_PATH, map_location=device, weights_only=True))
    mae_model.eval()

    # --- Diagnostic Trackers ---
    fp_triggers = {'CNN': 0, 'MAE': 0, 'STD_AE': 0}
    tp_triggers = {'CNN': 0, 'MAE': 0, 'STD_AE': 0}
    y_pred = []

    print("[*] Running Diagnostic Pipeline...")
    
    batch_size = 5000
    for i in range(0, len(X_subset), batch_size):
        X_batch = X_subset[i:i + batch_size]
        y_batch = y_subset[i:i + batch_size]
        
        # Supervised
        cnn_probs = cnn_model.predict(X_batch, batch_size=1024, verbose=0).flatten()
        
        # Standard AE
        ae_recon = ae_model.predict(X_batch, batch_size=1024, verbose=0)
        ae_errors = np.mean(np.abs(X_batch - ae_recon), axis=1)

        # Masked AE
        X_tensor = torch.FloatTensor(X_batch).to(device)
        with torch.no_grad():
            mae_recon, mae_original = mae_model(X_tensor, mask_ratio=0.0)
            mae_errors = torch.mean(torch.pow(mae_original - mae_recon, 2), dim=[1, 2, 3]).cpu().numpy()

        # Wiretap Logic
        for ens_prob, ae_err, mae_err, true_label in zip(cnn_probs, ae_errors, mae_errors, y_batch):
            
            cnn_flag = ens_prob > ENSEMBLE_CONFIDENCE_THRESHOLD
            mae_flag = mae_err > AWS_MASKED_AE_THRESHOLD
            std_ae_flag = ae_err > AWS_STANDARD_AE_THRESHOLD
            
            pipeline_flag = cnn_flag or mae_flag or std_ae_flag
            y_pred.append(1 if pipeline_flag else 0)

            # Record exactly who did what
            if true_label == 0 and pipeline_flag: # False Positive
                if cnn_flag: fp_triggers['CNN'] += 1
                if mae_flag: fp_triggers['MAE'] += 1
                if std_ae_flag: fp_triggers['STD_AE'] += 1
                
            elif true_label == 1 and pipeline_flag: # True Positive
                if cnn_flag: tp_triggers['CNN'] += 1
                if mae_flag: tp_triggers['MAE'] += 1
                if std_ae_flag: tp_triggers['STD_AE'] += 1

    y_pred = np.array(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_subset, y_pred).ravel()

    print("\n" + "="*50)
    print(" 🚨 DIAGNOSTIC AUTOPSY REPORT 🚨")
    print("="*50)
    
    print("\n[ FALSE POSITIVES ] (Normal traffic flagged as attacks)")
    print(f"Total Pipeline FPs: {fp:,}")
    print(f"  -> Caused by CNN:    {fp_triggers['CNN']:,} times")
    print(f"  -> Caused by Std_AE: {fp_triggers['STD_AE']:,} times")
    print(f"  -> Caused by MAE:    {fp_triggers['MAE']:,} times")
    print("  *(Note: Multiple models can flag the same packet)*")

    print("\n[ TRUE POSITIVES ] (Attacks successfully caught)")
    print(f"Total Pipeline TPs: {tp:,}")
    print(f"  -> Caught by CNN:    {tp_triggers['CNN']:,} times")
    print(f"  -> Caught by Std_AE: {tp_triggers['STD_AE']:,} times")
    print(f"  -> Caught by MAE:    {tp_triggers['MAE']:,} times")
    print("="*50)

if __name__ == "__main__":
    main()