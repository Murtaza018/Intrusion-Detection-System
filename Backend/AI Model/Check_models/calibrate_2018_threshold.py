import numpy as np
import tensorflow as tf
import torch
import os

# --- 1. Define Paths ---
PROCESSED_DATA_DIR = "../Preprocessing/Processed_2018_Test"
STANDARD_AE_PATH = "../Autoencoder/cicids_autoencoder.keras" # Path based on your model.py output
MASKED_AE_PATH = "../MAE/mae_visual_engine.pth" # Assuming you saved the PyTorch model state_dict

# --- 2. PyTorch MAE Model Definition ---
# We must redefine the PyTorch model class here to load the weights
import torch.nn as nn

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

# --- 3. Calibration Functions ---

def calibrate_tf_ae(model, X):
    print("\n[*] Calibrating Standard Autoencoder (TensorFlow)...")
    print("    [-] Generating reconstructions...")
    reconstructions = model.predict(X, batch_size=4096)
    
    print("    [-] Calculating Mean Absolute Error (MAE)...")
    # Using MAE to match your training logic
    errors = np.mean(np.abs(X - reconstructions), axis=1)
    print_thresholds(errors, "STANDARD AE")

def calibrate_pt_mae(model_path, X_numpy):
    print("\n[*] Calibrating Masked Autoencoder (PyTorch)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    [-] Using device: {device}")
    
    # Initialize and load model
    model = MAEModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
         print(f"    [!] Error loading PyTorch weights: {e}")
         print("    [!] Make sure MASKED_AE_PATH points to a saved state_dict (.pth file).")
         return
         
    model.eval() # Set to evaluation mode
    
    # Convert Numpy data to PyTorch Tensor
    X_tensor = torch.FloatTensor(X_numpy).to(device)
    
    errors = []
    batch_size = 1024
    
    print("    [-] Generating reconstructions...")
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            # Set mask_ratio=0 for inference/calibration
            reconstructed, original_img = model(batch, mask_ratio=0.0)
            
            # Calculate MSE between reconstructed image and original padded image
            batch_error = torch.mean(torch.pow(original_img - reconstructed, 2), dim=[1, 2, 3])
            errors.extend(batch_error.cpu().numpy())
            
    errors = np.array(errors)
    print_thresholds(errors, "MASKED AE")

def print_thresholds(errors, model_name):
    print(f"\n[!] {model_name} CALIBRATION RESULTS:")
    print("-" * 40)
    print(f"Mean Error: {np.mean(errors):.6f}")
    print(f"Max Error:  {np.max(errors):.6f}")
    
    threshold_95 = np.percentile(errors, 95)
    threshold_99 = np.percentile(errors, 99)
    threshold_99_9 = np.percentile(errors, 99.9)
    
    print(f"95th Percentile Threshold:   {threshold_95:.6f}")
    print(f"99th Percentile Threshold:   {threshold_99:.6f}  <-- Recommended Baseline")
    print(f"99.9th Percentile Threshold: {threshold_99_9:.6f}")
    print("-" * 40)

def main():
    print("[!] INITIATING DUAL-AE ZERO-DAY THRESHOLD CALIBRATION")
    
    x_path = os.path.join(PROCESSED_DATA_DIR, "X_test_2018.npy")
    y_path = os.path.join(PROCESSED_DATA_DIR, "y_test_2018.npy")
    
    print(f"[*] Loading tensor data from {PROCESSED_DATA_DIR}...")
    X_test = np.load(x_path)
    # FIX: allow_pickle=True to handle string labels
    y_test = np.load(y_path, allow_pickle=True) 
    
    print("[*] Isolating BENIGN traffic for calibration...")
    benign_indices = np.where(y_test == 'BENIGN')[0]
    X_benign = X_test[benign_indices]
    print(f"    [-] Found {len(X_benign)} BENIGN flows.")
    
    sample_size = min(100000, len(X_benign))
    print(f"[*] Extracting random calibration sample of {sample_size} flows...")
    np.random.seed(42) 
    sample_indices = np.random.choice(len(X_benign), sample_size, replace=False)
    X_calibration = X_benign[sample_indices]

    # --- Calibrate Standard AE (TensorFlow) ---
    if os.path.exists(STANDARD_AE_PATH):
        try:
             std_ae = tf.keras.models.load_model(STANDARD_AE_PATH)
             calibrate_tf_ae(std_ae, X_calibration)
        except Exception as e:
             print(f"\n[!] Error loading TensorFlow model: {e}")
    else:
        print(f"\n[!] ERROR: Standard AE not found at {STANDARD_AE_PATH}")

    # --- Calibrate Masked AE (PyTorch) ---
    if os.path.exists(MASKED_AE_PATH):
         calibrate_pt_mae(MASKED_AE_PATH, X_calibration)
    else:
        print(f"\n[!] ERROR: Masked AE not found at {MASKED_AE_PATH}")

if __name__ == "__main__":
    main()