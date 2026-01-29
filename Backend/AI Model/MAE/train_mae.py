import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from mae_model import MAEModel  # Ensure mae_model.py is in the same directory

# --- CONFIGURATION ---
INPUT_DIM = 78
EPOCHS = 20
BATCH_SIZE = 64
LR = 0.001
SAVE_PATH = "mae_visual_engine.pth"

def train_mae(data_path, labels_path):
    # 1. Load data and labels
    print(f"[*] Loading features from {data_path}...")
    x_train = np.load(data_path)
    
    print(f"[*] Loading labels from {labels_path}...")
    y_train = np.load(labels_path)
    
    # 2. APPLY THE BENIGN FILTER
    # We only want the model to learn what 'Normal' looks like.
    # In your preprocessor, BENIGN is assigned the value 0.
    print("[*] Filtering for Benign-only samples...")
    x_benign = x_train[y_train == 0]
    
    print(f"[+] Filter complete. Training on {len(x_benign):,} benign samples.")

    # 3. Prepare DataLoader
    train_tensor = torch.tensor(x_benign, dtype=torch.float32)
    dataset = TensorDataset(train_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Initialize Model
    model = MAEModel(input_dim=INPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"\n[*] Starting MAE Visual Training...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            
            # Forward pass: 40% masking ratio
            reconstruction, original = model(x, mask_ratio=0.4)
            
            loss = criterion(reconstruction, original)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"    Epoch {epoch+1}/{EPOCHS} | Reconstruction Loss: {avg_loss:.6f}")

    # 5. Save the final model weights
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n[***] SUCCESS: MAE trained on Benign traffic and saved as {SAVE_PATH}")

if __name__ == "__main__":
    # Corrected paths based on your CIC_Preprocessing.py output
    DATA_DIR = "../Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/"
    
    x_path = os.path.join(DATA_DIR, 'X_train.npy')
    y_path = os.path.join(DATA_DIR, 'y_train.npy')
    
    if os.path.exists(x_path) and os.path.exists(y_path):
        train_mae(x_path, y_path)
    else:
        print(f"[!] Critical Error: Data files not found at {DATA_DIR}")