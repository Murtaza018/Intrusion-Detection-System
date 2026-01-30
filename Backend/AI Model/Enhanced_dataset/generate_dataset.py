import os
import sys
import numpy as np
import torch
import pickle
from tqdm import tqdm

# --- FIX: Add parent directory to sys.path ---
# This allows the script to see the 'GNN' and 'MAE' folders
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now these imports will work
from GNN.train_gnn import ContextSAGE
from MAE.mae_model import MAEModel

# --- CONFIG (Updated with absolute paths for safety) ---
DATA_DIR = os.path.join(parent_dir, "Preprocessing", "CIC-IDS-2017", "CIC-IDS-2017-Processed")
GNN_PATH = os.path.join(parent_dir, "GNN", "gnn_context_engine_final.pth")
MAE_PATH = os.path.join(parent_dir, "MAE", "mae_visual_engine.pth")
OUTPUT_FILE = "X_train_enhanced_95.npy"

def generate_enhanced_features():
    # 1. Load Models
    gnn = ContextSAGE(in_channels=36, embedding_dim=16)
    if not os.path.exists(GNN_PATH):
        print(f"[!] GNN Model not found at {GNN_PATH}")
        return
    gnn.load_state_dict(torch.load(GNN_PATH, map_location='cpu'))
    gnn.eval()

    mae = MAEModel(input_dim=78)
    if not os.path.exists(MAE_PATH):
        print(f"[!] MAE Model not found at {MAE_PATH}")
        return
    mae.load_state_dict(torch.load(MAE_PATH, map_location='cpu'))
    mae.eval()
    print("[✔] GNN and MAE loaded successfully.")

    # 2. Load Raw Features
    train_path = os.path.join(DATA_DIR, "X_train.npy")
    if not os.path.exists(train_path):
        print(f"[!] Training data not found at {train_path}")
        return
        
    x_train = np.load(train_path)
    print(f"[*] Processing {len(x_train):,} samples...")

    enhanced_data = []

    # 3. Process in batches
    batch_size = 1000
    with torch.no_grad():
        for i in tqdm(range(0, len(x_train), batch_size)):
            batch = x_train[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32)

            # GNN Feature Extraction
            gnn_input = batch_tensor[:, :36]
            dummy_edges = torch.zeros((2, 0), dtype=torch.long) 
            gnn_out = gnn.conv1(gnn_input, dummy_edges).relu()
            gnn_features = gnn.conv2(gnn_out, dummy_edges).numpy()

            # MAE Anomaly Scoring
            recon, original = mae(batch_tensor, mask_ratio=0.4)
            mae_errors = torch.mean((recon - original)**2, dim=(1, 2, 3)).numpy().reshape(-1, 1)

            # Combine: 78 + 16 + 1 = 95
            combined = np.hstack([batch, gnn_features, mae_errors])
            enhanced_data.append(combined)

    # 4. Save
    final_data = np.vstack(enhanced_data)
    np.save(OUTPUT_FILE, final_data)
    print(f"\n[***] SUCCESS: Enhanced dataset saved as {OUTPUT_FILE}")
    print(f"      New Shape: {final_data.shape}")

if __name__ == "__main__":
    generate_enhanced_features()