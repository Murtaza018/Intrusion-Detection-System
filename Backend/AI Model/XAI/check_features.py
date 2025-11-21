import numpy as np
import json
import os
import pandas as pd

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_MODEL_DIR = os.path.dirname(BASE_DIR)

# We try to find a raw CSV from the dataset to get the real column names.
# Adjust this path if your CSVs are in a different subfolder.
CSV_PATH = os.path.join(
    AI_MODEL_DIR, 
    "Preprocessing", 
    "Datasets", 
    "CIC-IDS-2017", 
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)
OUT_PATH = os.path.join(BASE_DIR, "feature_schema.json")

print(f"Attempting to read column names from: {CSV_PATH}")

try:
    # Read just the header (efficient)
    df = pd.read_csv(CSV_PATH, nrows=0)
    
    # Clean the column names (strip spaces) just like in preprocessing
    cols = df.columns.str.strip().tolist()
    
    # Remove the 'Label' column as it's not a feature
    if 'Label' in cols:
        cols.remove('Label')
        
    print(f"[*] Found {len(cols)} features from CSV.")
    
    # Save to JSON
    with open(OUT_PATH, "w") as f:
        json.dump(cols, f, indent=2)
        
    print(f"✅ Created 'feature_schema.json' with REAL feature names.")

except FileNotFoundError:
    print(f"[!] Warning: Could not find the CSV file to read headers.")
    print(f"    path checked: {CSV_PATH}")
    print(f"[*] Falling back to placeholder names (f0, f1, ...).")
    
    # Fallback: Load the numpy array to see how many features there are
    X_PATH = os.path.join(AI_MODEL_DIR, "Preprocessing", "CIC-IDS-2017-Processed", "X_train.npy")
    
    if os.path.exists(X_PATH):
        X = np.load(X_PATH)
        n_features = X.shape[1]
        features = [f"f{i}" for i in range(n_features)]
        
        with open(OUT_PATH, "w") as f:
            json.dump(features, f, indent=2)
            
        print(f"✅ Created 'feature_schema.json' with {n_features} PLACEHOLDER names.")
        print("    NOTE: Explanations will use 'f0', 'f1' etc. unless you fix the CSV path.")
    else:
        print("[!] Error: Could not find X_train.npy either. Cannot generate schema.")