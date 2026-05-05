import pandas as pd
import numpy as np
import joblib
import os
import glob

# 1. Define Paths (Adjust these if your raw 2018 CSVs are elsewhere)
RAW_2018_DIR = "./Datasets/CSE-CIC-IDS-2018/"
SCALER_PATH = "./CGAN/CGAN_preprocessed_data/scaler.pkl" # Using your saved 2017 scaler
OUTPUT_DIR = "Processed_2018_Test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. The Translation Dictionary
# 2018 researchers changed the names of the attacks. Your 2017 models will fail to recognize them.
# We must map 2018 strings -> 2017 strings.
LABEL_MAP = {
    'Benign': 'BENIGN',
    'FTP-BruteForce': 'FTP-Patator',
    'SSH-Bruteforce': 'SSH-Patator',
    'DoS attacks-GoldenEye': 'DoS GoldenEye',
    'DoS attacks-Slowloris': 'DoS slowloris',
    'DoS attacks-SlowHTTPTest': 'DoS Slowhttptest',
    'DoS attacks-Hulk': 'DoS Hulk',
    'Brute Force -Web': 'Web Attack - Brute Force',
    'Brute Force -XSS': 'Web Attack - XSS',
    'SQL Injection': 'Web Attack - Sql Injection'
    # Any attack not in this list (e.g., DDoS-HOIC) will remain as-is. 
    # Your supervised models will fail on them, but your Autoencoder should catch them as Zero-Days.
}

# 3. Features to Drop
# 2018 includes metadata columns that 2017 usually stripped before training.
METADATA_COLS = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp']

def clean_and_map_data(file_path, scaler, expected_features):
    print(f"[*] Processing: {os.path.basename(file_path)}")
    
    # Load dataset, dropping low_memory warning
    df = pd.read_csv(file_path, low_memory=False)
    
    # Clean column names (strip whitespace, lowercase to avoid case-sensitivity bugs)
    df.columns = df.columns.str.strip()

    # ---> FIX 1: PURGE REPEATED HEADERS <---
    # Drop any row where the value in the first column is literally the name of the first column
    first_col_name = df.columns[0]
    df = df[df.iloc[:, 0] != first_col_name].copy()
    
    # Drop metadata if it exists
    cols_to_drop = [col for col in METADATA_COLS if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Separate Features and Labels
    label_col = [col for col in df.columns if col.lower() == 'label'][0]
    y = df[label_col].map(LABEL_MAP).fillna(df[label_col]) 
    X = df.drop(columns=[label_col])

    # ---> FIX 2: FORCE NUMERIC CONVERSION <---
    # Because the ghost headers made Pandas treat columns as strings, 
    # we must violently force them back to floats. Anything that fails becomes NaN.
    print("    [-] Forcing numeric data types...")
    X = X.apply(pd.to_numeric, errors='coerce')

    # 4. Handle Infinity and NaN Landmines
    print("    [-] Neutralizing NaN and Infinity values...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop the NaNs (including any created by our forced numeric conversion)
    valid_indices = X.dropna().index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    # 5. Enforce Strict Feature Alignment
    print("    [-] Enforcing 2017 feature alignment...")
    for col in expected_features:
        if col not in X.columns:
            # If 2018 is missing a 2017 feature, inject zeros
            X[col] = 0.0 
            
    # Keep strictly the expected features in the EXACT order the scaler expects
    X = X[expected_features]

    # 6. Apply the 2017 Scaler (DO NOT FIT)
    print("    [-] Applying 2017 Scaler transformation...")
    X_scaled = scaler.transform(X)

    return X_scaled, y.values

def main():
    print("[!] INITIATING 2018 PREPROCESSING PROTOCOL")
    
    # Load the hardened 2017 Scaler
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"CRITICAL: Scaler not found at {SCALER_PATH}. Pipeline compromised.")
    scaler = joblib.load(SCALER_PATH)
    
    # We must extract the exact feature names the scaler expects.
    # If using scikit-learn >= 1.0, scalers save the feature names.
    if hasattr(scaler, 'feature_names_in_'):
        expected_features = list(scaler.feature_names_in_)
    else:
        raise AttributeError("CRITICAL: Scaler does not contain 'feature_names_in_'. You must manually define the 78 features list here.")

    all_X = []
    all_y = []

    csv_files = glob.glob(os.path.join(RAW_2018_DIR, "*.csv"))
    if not csv_files:
        print("No CSV files found in the specified directory. Check RAW_2018_DIR.")
        return

    for file in csv_files:
        X_scaled, y_mapped = clean_and_map_data(file, scaler, expected_features)
        all_X.append(X_scaled)
        all_y.append(y_mapped)

    # Concatenate all days into one massive test set
    final_X = np.vstack(all_X)
    final_y = np.concatenate(all_y)

    print(f"\n[+] Processing Complete. Final Tensor Shape: {final_X.shape}")
    
    # Save the strictly formatted data
    np.save(os.path.join(OUTPUT_DIR, "X_test_2018.npy"), final_X)
    np.save(os.path.join(OUTPUT_DIR, "y_test_2018.npy"), final_y)
    
    print(f"[+] Hardened 2018 test data saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()  