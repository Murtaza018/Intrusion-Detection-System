# evaluate_cgan_checkpoints.py
# Description: This script loads the saved generator checkpoints from our CGAN training
# run. It generates synthetic attack samples from each checkpoint and uses our
# hardened classifier to judge their quality. This helps us find the "golden epoch"
# where the generator was producing the most realistic attacks.

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import glob

print("--- Evaluating CGAN Checkpoints ---")

# --- Configuration ---
# Path where the epoch checkpoints were saved (current directory by default)
CHECKPOINT_DIR = "." 
# Our expert classifier to judge the fakes
CLASSIFIER_PATH = "../Adversarial Attack and Defense/cicids_spatiotemporal_model_hardened.keras"

# Evaluation settings
NUM_SAMPLES = 1000
LATENT_DIM = 128
NUM_CLASSES = 15
# Let's test generating a specific attack type, e.g., DDoS (Label 1)
TARGET_LABEL = 1 

# --- Step 1: Load the Expert Classifier ---
print(f"\n[*] Loading the judge classifier from '{CLASSIFIER_PATH}'...")
try:
    classifier = keras.models.load_model(CLASSIFIER_PATH)
    print("[+] Classifier loaded successfully.")
except Exception as e:
    print(f"\n[!] Error loading classifier: {e}")
    exit()

# --- Step 2: Find Checkpoint Files ---
# We look for files named 'cgan_generator_epoch_*.keras'
checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "cgan_generator_epoch_*.keras"))
# Sort them by epoch number so we test in order
checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

if not checkpoint_files:
    print("[!] No checkpoint files found. Make sure you are running this in the correct directory.")
    exit()

print(f"[*] Found {len(checkpoint_files)} checkpoints to evaluate.")

# We'll select a subset to speed things up (every 10th epoch)
selected_checkpoints = [f for f in checkpoint_files if int(f.split('_')[-1].split('.')[0]) % 10 == 0]
# Also add the final one if it's not there
if "cgan_generator_final.keras" in os.listdir(CHECKPOINT_DIR):
    selected_checkpoints.append("cgan_generator_final.keras")

print(f"[*] Selected {len(selected_checkpoints)} checkpoints for detailed evaluation.")


# --- Step 3: Evaluation Loop ---
print("\n--- Starting Evaluation ---")
print(f"{'Model File':<35} | {'Conf. Score':<12} | {'Verdict'}")
print("-" * 65)

best_score = 0
best_model_file = ""

for model_file in selected_checkpoints:
    try:
        # Load the generator
        # We use compile=False because we only need to predict, not train
        generator = keras.models.load_model(model_file, compile=False)
        
        # Generate synthetic samples for our target label
        noise = np.random.normal(0, 1, (NUM_SAMPLES, LATENT_DIM))
        labels = tf.keras.utils.to_categorical([TARGET_LABEL] * NUM_SAMPLES, num_classes=NUM_CLASSES)
        
        synthetic_data = generator.predict([noise, labels], verbose=0)
        
        # Reshape for the classifier (samples, 78, 1)
        synthetic_data = np.expand_dims(synthetic_data, -1)
        
        # Ask the classifier: "Are these attacks?"
        predictions = classifier.predict(synthetic_data, verbose=0)
        
        # Calculate the score: What % were classified as attacks (label 1)?
        # Since our classifier outputs probability of being attack (1), 
        # we can just take the average probability or count > 0.5
        avg_confidence = np.mean(predictions)
        percent_attacks = np.mean(predictions > 0.5) * 100
        
        verdict = "Poor"
        if avg_confidence > 0.5: verdict = "Okay"
        if avg_confidence > 0.8: verdict = "Good"
        if avg_confidence > 0.95: verdict = "Excellent"
        
        print(f"{os.path.basename(model_file):<35} | {avg_confidence:.2%}      | {verdict}")
        
        if avg_confidence > best_score:
            best_score = avg_confidence
            best_model_file = model_file

    except Exception as e:
        print(f"{os.path.basename(model_file):<35} | ERROR        | {e}")

# --- Step 4: Final Recommendation ---
print("-" * 65)
print("\n--- Evaluation Complete ---")
if best_score > 0.8:
    print(f"[SUCCESS] Found a high-quality generator!")
    print(f"[*] Best Model: {best_model_file}")
    print(f"[*] Confidence Score: {best_score:.2%}")
    print("You should rename this file to 'cgan_generator_best.keras' and use it for your pipeline.")
else:
    print("[FAILURE] No generator produced convincing attacks (Score < 80%).")
    print("The 'Discriminator Overpowering' problem likely prevented proper training.")
    print("Recommendation: Re-train with Label Smoothing (Plan B).")