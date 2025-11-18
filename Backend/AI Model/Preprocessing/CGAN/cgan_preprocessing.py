# preprocess_cgan.py
# Description: A new preprocessing pipeline specifically for the Conditional GAN (CGAN).
# This script is nearly identical to our first preprocessor, but with one
# critical difference: it creates MULTI-CLASS labels (e.g., 0=BENIGN, 1=DDoS, etc.)
# instead of binary (0/1) labels. This is what allows the CGAN to be "conditional".

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

print("--- CGAN Preprocessing Pipeline (Multi-Class) ---")

# --- 1. Define File Paths ---
INPUT_DATA_PATH = "../Datasets/CIC-IDS-2017/"
# We will save to a new directory to keep our data separate.
OUTPUT_DATA_PATH = "CGAN_preprocessed_data/"

if not os.path.exists(OUTPUT_DATA_PATH):
    os.makedirs(OUTPUT_DATA_PATH)
    print(f"I've created a new folder for our CGAN data: '{OUTPUT_DATA_PATH}'")


# --- 2. Load and Combine the Dataset ---
print(f"\nI'm looking for the dataset files in '{INPUT_DATA_PATH}'...")
all_files = glob.glob(os.path.join(INPUT_DATA_PATH, "*.csv"))
if not all_files:
    print(f"\nUh oh! I couldn't find any CSV files in that folder.")
    exit()

print(f"Great! I found {len(all_files)} files. Merging them...")
df_list = [pd.read_csv(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)
print(f"Success! We have {len(df):,} total rows.")


# --- 3. Data Cleaning ---
print("\nCleaning up the data...")
df.columns = df.columns.str.strip()
print("    - Column names trimmed.")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"    - Replaced {df.isin([np.inf, -np.inf]).sum().sum()} infinite values.")
initial_rows = len(df)
df.dropna(inplace=True)
print(f"    - Dropped {initial_rows - len(df):,} rows with missing values.")


# --- 4. Feature and Label Separation ---
print("\nSeparating features (X) and labels (y)...")
X = df.drop(columns=['Label'])
y_text = df['Label']


# --- 5. CRITICAL STEP: Multi-Class Label Encoding ---
print("I'm converting the text labels into multi-class numeric labels...")
# We use LabelEncoder to turn text ('BENIGN', 'DDoS') into numbers (0, 1)
le = LabelEncoder()
y = le.fit_transform(y_text)

# Let's see what the classes are:
num_classes = len(le.classes_)
print(f"\nFound {num_classes} unique classes:")
for i, class_name in enumerate(le.classes_):
    print(f"    Label {i} -> {class_name} ({(y == i).sum():,} samples)")

# We will save this label encoder so we can reverse the process later.
import pickle
with open(os.path.join(OUTPUT_DATA_PATH, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)
print("\n[+] Label Encoder saved to 'label_encoder.pkl'")


# --- 6. Feature Scaling (Normalization) ---
print("\nScaling all features to be between 0 and 1...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("Features have been scaled successfully.")

# We will also save the scaler.
with open(os.path.join(OUTPUT_DATA_PATH, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print("[+] Scaler saved to 'scaler.pkl'")


# --- 7. Save the Processed Data ---
# For a GAN, we usually train on the *entire* dataset at once,
# so we'll save the full, processed X and y.
print(f"\nSaving processed data to: {OUTPUT_DATA_PATH}")
np.save(os.path.join(OUTPUT_DATA_PATH, 'X_full.npy'), X_scaled)
np.save(os.path.join(OUTPUT_DATA_PATH, 'y_full.npy'), y)

print("\n--- CGAN Preprocessing Complete! ---")