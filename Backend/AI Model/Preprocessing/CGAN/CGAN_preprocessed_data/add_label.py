# add_new_label.py
# Description: This script allows an analyst to manually add a new attack label
# to the system.
# UPDATED: Automatically assigns the next sequential ID to prevent user error.

import pickle
import numpy as np
import os
import sys
import argparse

print("--- Add New Attack Label Tool ---")

# --- Configuration ---
DATA_PATH = "./"
ENCODER_FILE = os.path.join(DATA_PATH, 'label_encoder.pkl')

# --- Step 1: Load the Existing Encoder ---
print(f"\n[*] Loading Label Encoder from '{ENCODER_FILE}'...")

try:
    with open(ENCODER_FILE, 'rb') as f:
        le = pickle.load(f)
    print("[+] Encoder loaded successfully.")
except FileNotFoundError:
    print(f"[!] Error: Could not find '{ENCODER_FILE}'.")
    print("    Make sure you have run 'preprocess_cgan.py' first.")
    sys.exit(1)

# Get current classes
current_classes = list(le.classes_)
next_id = len(current_classes)

print(f"\n[*] Current Status:")
print(f"    Total Classes: {len(current_classes)}")
print(f"    Last ID Used:  {len(current_classes) - 1} ({current_classes[-1]})")
print(f"    Next Available ID: {next_id} (Auto-Calculated)")

# --- Step 2: Get Analyst Input ---
parser = argparse.ArgumentParser(description="Add a new attack label to the IDS.")
parser.add_argument("--name", type=str, help="The name of the new attack (e.g., 'ZeroDay-X')")

args = parser.parse_args()

# Only ask for the Name
if args.name is None:
    new_label_name = input(f"\nEnter the name for the new attack: ").strip()
else:
    new_label_name = args.name

# --- Step 3: Validation ---
if new_label_name in current_classes:
    print(f"[!] Error: The label '{new_label_name}' already exists (ID: {current_classes.index(new_label_name)}).")
    print("    Action cancelled.")
    sys.exit(1)

# --- Step 4: Update and Save ---
print(f"\n[*] Assigning '{new_label_name}' to ID {next_id}...")

# Append the new class to the internal classes_ array
new_classes = np.append(le.classes_, new_label_name)
le.classes_ = new_classes

# Save the updated encoder back to the file
try:
    with open(ENCODER_FILE, 'wb') as f:
        pickle.dump(le, f)
    print(f"[+] Successfully saved updated 'label_encoder.pkl'.")
except Exception as e:
    print(f"[!] Error saving file: {e}")

print("\n--- SUCCESS! ---")
print(f"IMPORTANT: Your new attack '{new_label_name}' is officially Label ID: {next_id}")
print(f"-> When using the GAN, set TARGET_LABEL = {next_id}")
print(f"-> Update 'train_cgan.py' config to set NUM_CLASSES = {len(le.classes_)}")