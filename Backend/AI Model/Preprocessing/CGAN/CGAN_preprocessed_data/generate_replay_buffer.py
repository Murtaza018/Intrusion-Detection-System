# generate_replay_buffer.py
# Creates a compact, balanced dataset (replay_buffer.npz)
# FIXED: Correct path traversal (Up 3 levels from script location)

import numpy as np
import os

# --- PATH SETUP ---
# 1. Get location of this script (.../CGAN_preprocessed_data)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go UP 3 levels to reach 'ids_pipeline'
# Level 1 Up: .../CGAN
# Level 2 Up: .../preprocessing
# Level 3 Up: .../ids_pipeline
IDS_PIPELINE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))

# 3. Define Output Path
OUTPUT_PATH = os.path.join(IDS_PIPELINE_DIR, "ids_pipeline","replay_buffer.npz")

# 4. Define Source Files (In the same folder as this script)
SOURCE_X_PATH = os.path.join(SCRIPT_DIR, "X_full.npy") 
SOURCE_Y_PATH = os.path.join(SCRIPT_DIR, "y_full.npy")

# How many samples to keep per class
SAMPLES_PER_CLASS = 200 

def create_buffer():
    print("--- 🧠 Generating Replay Buffer ---")
    print(f"[*] Script Location: {SCRIPT_DIR}")
    print(f"[*] Target Folder:   {IDS_PIPELINE_DIR}")
    print(f"[*] Output File:     {OUTPUT_PATH}")
    
    # 1. Verify Source Files
    if not os.path.exists(SOURCE_X_PATH) or not os.path.exists(SOURCE_Y_PATH):
        print(f"[!] Error: Source files not found in {SCRIPT_DIR}")
        return

    # 2. Load Data
    print(f"[*] Loading massive datasets (this may take a moment)...")
    try:
        # mmap_mode='r' saves RAM
        X_full = np.load(SOURCE_X_PATH, mmap_mode='r')
        y_full = np.load(SOURCE_Y_PATH, mmap_mode='r')
        
        print(f"    Original Shape: X={X_full.shape}, y={y_full.shape}")
    except Exception as e:
        print(f"[!] Error loading .npy files: {e}")
        return

    # 3. Stratified Sampling
    print(f"[*] Selecting {SAMPLES_PER_CLASS} samples per class...")
    
    unique_classes = np.unique(y_full)
    selected_indices = []
    stats = {}

    for cls in unique_classes:
        indices = np.where(y_full == cls)[0]
        total_available = len(indices)
        count_to_take = min(total_available, SAMPLES_PER_CLASS)
        
        if count_to_take > 0:
            # Random choice without replacement
            chosen = np.random.choice(indices, count_to_take, replace=False)
            selected_indices.extend(chosen)
        
        stats[cls] = count_to_take

    # 4. Extract & Save
    print(f"[*] Extracting {len(selected_indices)} total samples...")
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices) 
    
    # Load into memory
    X_buffer = X_full[selected_indices]
    y_buffer = y_full[selected_indices]
    
    print(f"[*] Saving to {OUTPUT_PATH}...")
    np.savez_compressed(OUTPUT_PATH, X=X_buffer, y=y_buffer)
    
    print("\n--- ✅ Success! ---")
    print(f"Buffer saved at: {OUTPUT_PATH}")
    print("Class Distribution in Buffer:")
    for cls, count in stats.items():
        print(f"  Class {cls}: {count}")

if __name__ == "__main__":
    create_buffer()