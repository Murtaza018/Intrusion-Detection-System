# evaluate_cgan_checkpoints.py
# Description: This script evaluates the saved CGAN generator checkpoints.
# UPDATED: Now checks performance across ALL labels (0-14) to find the most
# versatile and robust generator.

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import glob

print("--- Evaluating CGAN Checkpoints (All Labels) ---")

# --- Configuration ---
CHECKPOINT_DIR = "." 
CLASSIFIER_PATH = "../Adversarial Attack and Defense/cicids_spatiotemporal_model_hardened.keras"

# Evaluation settings
NUM_SAMPLES_PER_CLASS = 200 # Reduced slightly since we are checking 15 classes
LATENT_DIM = 128
NUM_CLASSES = 15

# --- Step 1: Load the Expert Classifier ---
print(f"\n[*] Loading the judge classifier from '{CLASSIFIER_PATH}'...")
try:
    classifier = keras.models.load_model(CLASSIFIER_PATH)
    print("[+] Classifier loaded successfully.")
except Exception as e:
    print(f"\n[!] Error loading classifier: {e}")
    # Fallback logic
    try:
        classifier = keras.models.load_model("cicids_spatiotemporal_model.keras")
        print("[+] Fallback classifier loaded.")
    except:
        print("[!] Could not load any classifier. Exiting.")
        exit()

# --- Step 2: Find Checkpoint Files ---
checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "cgan_generator_epoch_*.keras"))

if not checkpoint_files:
    print("[!] No checkpoint files found.")
    exit()

def get_epoch_num(filename):
    try:
        return int(filename.split('_')[-1].split('.')[0])
    except:
        return -1

checkpoint_files.sort(key=get_epoch_num)
# Check every 5th epoch to save time, plus the final one
selected_checkpoints = [f for f in checkpoint_files if get_epoch_num(f) % 5 == 0]

if "cgan_generator_final.keras" in os.listdir(CHECKPOINT_DIR):
    final_path = os.path.join(CHECKPOINT_DIR, "cgan_generator_final.keras")
    if final_path not in selected_checkpoints:
         selected_checkpoints.append(final_path)

print(f"[*] Selected {len(selected_checkpoints)} checkpoints. Analyzing comprehensive performance...")


# --- Step 3: Evaluation Loop ---
print("\n--- Summary of Epoch Performance ---")
print(f"{'Epoch/Model':<35} | {'Avg Attack Conf.':<18} | {'Normal Conf.':<15} | {'Verdict'}")
print("-" * 85)

best_overall_score = 0
best_model_file = ""
best_model_details = {}

for model_file in selected_checkpoints:
    try:
        # Load generator without compiling (faster)
        generator = keras.models.load_model(model_file, compile=False)
        
        total_attack_confidence = 0
        normal_confidence = 0
        
        # Store detailed scores for this model just in case it's the best one
        current_model_details = {}

        # Loop through ALL classes
        for label_idx in range(NUM_CLASSES):
            # Generate noise
            noise = np.random.normal(0, 1, (NUM_SAMPLES_PER_CLASS, LATENT_DIM))
            # Create label vector for this specific class
            labels = tf.keras.utils.to_categorical([label_idx] * NUM_SAMPLES_PER_CLASS, num_classes=NUM_CLASSES)
            
            # Generate synthetic data
            synthetic_data = generator.predict([noise, labels], verbose=0)
            # Reshape for classifier (samples, 78, 1)
            synthetic_data = np.expand_dims(synthetic_data, -1)
            
            # Get classifier predictions (Probability of being an ATTACK)
            preds = classifier.predict(synthetic_data, verbose=0)
            avg_conf = np.mean(preds)
            
            current_model_details[label_idx] = avg_conf

            if label_idx == 0:
                # For Label 0 (Normal), prediction should be close to 0.
                # We record the raw prediction to see if it looks like an attack (bad) or normal (good).
                normal_confidence = avg_conf
            else:
                # For Labels 1-14 (Attacks), prediction should be close to 1.
                total_attack_confidence += avg_conf

        # Calculate Average Attack Confidence (across 14 attack types)
        avg_attack_score = total_attack_confidence / (NUM_CLASSES - 1)
        
        # Determine Verdict
        verdict = "Poor"
        if avg_attack_score > 0.5: verdict = "Okay"
        if avg_attack_score > 0.7: verdict = "Good"
        if avg_attack_score > 0.9: verdict = "Excellent"

        print(f"{os.path.basename(model_file):<35} | {avg_attack_score:.2%}            | {normal_confidence:.2%}           | {verdict}")

        # Track the best model (highest attack confidence)
        if avg_attack_score > best_overall_score:
            best_overall_score = avg_attack_score
            best_model_file = model_file
            best_model_details = current_model_details

    except Exception as e:
        print(f"{os.path.basename(model_file):<35} | ERROR             | {e}")

# --- Step 4: Detailed Report for the Winner ---
print("-" * 85)
print("\n--- Detailed Analysis of the Best Generator ---")

if best_overall_score > 0.7:
    print(f"[*] Winner: {os.path.basename(best_model_file)}")
    print(f"[*] Overall Attack Generation Quality: {best_overall_score:.2%}")
    print("\nBreakdown by Label:")
    print(f"{'Label ID':<10} | {'Type':<25} | {'Classifier Confidence'}")
    print("-" * 60)
    
    # Hardcoded map based on your previous output for clarity
    label_map = {
        0: "BENIGN", 1: "Bot", 2: "DDoS", 3: "DoS GoldenEye", 4: "DoS Hulk",
        5: "DoS Slowhttptest", 6: "DoS slowloris", 7: "FTP-Patator", 8: "Heartbleed",
        9: "Infiltration", 10: "PortScan", 11: "SSH-Patator", 12: "Web Attack Brute Force",
        13: "Web Attack Sql Injection", 14: "Web Attack XSS"
    }

    for label_id in range(NUM_CLASSES):
        score = best_model_details.get(label_id, 0.0)
        name = label_map.get(label_id, "Unknown")
        
        # Visual indicator
        status = "✅" if (label_id > 0 and score > 0.7) or (label_id == 0 and score < 0.3) else "⚠️"
        if label_id > 0 and score < 0.5: status = "❌"
        
        print(f"{label_id:<10} | {name:<25} | {score:.2%} {status}")

    print("\n[Action] Rename this file to 'cgan_generator_best.keras' to complete your Training Academy.")
else:
    print("[FAILURE] Even the best model didn't perform well across all categories.")
    print("Recommendation: The model might need longer training or WGAN-GP architecture.")