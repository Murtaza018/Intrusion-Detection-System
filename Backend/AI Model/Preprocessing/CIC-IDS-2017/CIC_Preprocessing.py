# preprocess_cicids2017.py
# Description: A friendly, step-by-step guide to preprocessing the CIC-IDS-2017 dataset.
# This script takes the raw, messy CSV files, cleans them up, and prepares them
# to be used for training a machine learning model.

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print("--- Let's Get the CIC-IDS-2017 Dataset Ready for AI! ---")

# --- Step 1: Setting Up Our File Paths ---
# I'll assume this script is located in 'AI Model/Preprocessing/'.
# The dataset should be in a subfolder called 'Datasets/CIC-IDS-2017/'.
INPUT_DATA_PATH = "../Datasets/CIC-IDS-2017/"
OUTPUT_DATA_PATH = "CIC-IDS-2017-Processed/"

# Let's create a new folder to save our clean, processed data.
if not os.path.exists(OUTPUT_DATA_PATH):
    os.makedirs(OUTPUT_DATA_PATH)
    print(f"I've created a new folder for our clean data: '{OUTPUT_DATA_PATH}'")


# --- Step 2: Finding and Combining All the CSV Files ---
print(f"\nI'm looking for the dataset files in '{INPUT_DATA_PATH}'...")
all_files = glob.glob(os.path.join(INPUT_DATA_PATH, "*.csv"))

if not all_files:
    print(f"\nUh oh! I couldn't find any CSV files in that folder.")
    print("Please make sure you've extracted the dataset there before we continue.")
    exit()

print(f"Great! I found {len(all_files)} files. Let's merge them into one big dataset.")
df_list = [pd.read_csv(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)
print(f"Success! We now have one large table with {len(df):,} rows of data.")


# --- Step 3: Cleaning Up the Messy, Real-World Data ---
print("\nAlright, time for some data cleaning. Real-world data is never perfect!")

# First, some of the column names have extra spaces. Let's trim those.
df.columns = df.columns.str.strip()
print("    - Column names have been trimmed.")

# Next, machine learning models can't handle 'infinity' values. Let's replace them.
df.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"    - Replaced {df.isin([np.inf, -np.inf]).sum().sum()} infinite values.")

# Finally, we'll remove any rows that have missing data (NaNs).
initial_rows = len(df)
df.dropna(inplace=True)
print(f"    - Dropped rows with missing values. We removed {initial_rows - len(df):,} rows.")


# --- Step 4: Separating Our Features from Our Labels ---
print("\nNow, let's separate our main data (the features, X) from the final answer (the labels, y).")
X = df.drop(columns=['Label'])
y_text = df['Label']


# --- Step 5: Translating Labels into Numbers ---
# Our autoencoder is an anomaly detector. It needs a simple, binary answer:
# Is this traffic normal (0) or is it an attack (1)?
print("I'm converting the text labels ('BENIGN', 'DDoS', etc.) into simple 0s and 1s.")
y = y_text.apply(lambda x: 0 if x == 'BENIGN' else 1)
print(f"    - We have {(y == 0).sum():,} normal samples and {(y == 1).sum():,} attack samples.")


# --- Step 6: Splitting Our Data for Training and Testing ---
# It's standard practice to split the data into three sets:
# 1. Training Set (70%): The model learns from this.
# 2. Validation Set (15%): We use this to fine-tune the model.
# 3. Test Set (15%): The model's final, unseen exam.
print("\nSplitting the data into training, validation, and testing sets...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
print(f"    - Training set ready with {len(X_train):,} samples.")
print(f"    - Validation set ready with {len(X_val):,} samples.")
print(f"    - Test set ready with {len(X_test):,} samples.")


# --- Step 7: Scaling Our Features ---
# The feature columns have wildly different scales. We need to normalize them
# so they are all consistently between 0 and 1. This helps the model learn better.
# Important: We only 'fit' the scaler on the training data to avoid peeking at the answers!
print("\nScaling all features to be between 0 and 1...")
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
print("Features have been scaled successfully.")


# --- Step 8: Saving Our Clean Data! ---
# The final step is to save our beautifully processed data as NumPy files.
print(f"\nSaving the final, clean data files to '{OUTPUT_DATA_PATH}'...")
np.save(os.path.join(OUTPUT_DATA_PATH, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_DATA_PATH, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_DATA_PATH, 'X_val.npy'), X_val)
np.save(os.path.join(OUTPUT_DATA_PATH, 'y_val.npy'), y_val)
np.save(os.path.join(OUTPUT_DATA_PATH, 'X_test.npy'), X_test)
np.save(os.path.join(OUTPUT_DATA_PATH, 'y_test.npy'), y_test)
print("All done! Your data is now clean, processed, and ready for the AI model.")
print("\n--- Preprocessing Complete! ---")

