# check_gan.py
# Verifies that the Generator can load and produce valid traffic.

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

# --- CONFIGURATION ---
# Path to your BEST model (Epoch 3)
MODEL_PATH = "cgan_generator.keras" 
LATENT_DIM = 128
NUM_CLASSES = 15
NUM_SAMPLES = 5

def check_model():
    print(f"[*] Loading model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model not found at {MODEL_PATH}")
        return

    try:
        # Load the Generator
        # compile=False is important because we don't need the optimizer/loss to just generate
        generator = load_model(MODEL_PATH, compile=False)
        print("[+] Model loaded successfully.")
        
        # 1. Prepare Noise (Latent Vector)
        noise = tf.random.normal(shape=(NUM_SAMPLES, LATENT_DIM))
        
        # 2. Prepare Condition (Label)
        # Let's ask for class 1 (Assuming 1 = Attack, but depends on your encoding)
        # If your training data was one-hot encoded, we need a vector of 15 zeros with one 1.
        labels = np.zeros((NUM_SAMPLES, NUM_CLASSES))
        labels[:, 1] = 1.0  # Set Class 1
        
        print(f"[*] Generating {NUM_SAMPLES} samples for Class 1...")
        
        # 3. Generate
        generated_data = generator.predict([noise, labels], verbose=0)
        
        # 4. Analysis
        print("\n--- 📊 Generated Data Analysis ---")
        print(f"Shape: {generated_data.shape} (Should be [{NUM_SAMPLES}, 78])")
        
        # Convert to DataFrame for easier viewing
        df = pd.DataFrame(generated_data)
        
        print("\n[Sample 1 Features (First 10)]:")
        print(df.iloc[0, :10].values)
        
        print("\n[Statistics]:")
        print(f"Min Value: {df.min().min():.4f} (Should be close to 0)")
        print(f"Max Value: {df.max().max():.4f} (Should be close to 1)")
        print(f"Mean Value: {df.mean().mean():.4f}")
        
        # Check for Mode Collapse (Are all rows identical?)
        std_dev = df.std().mean()
        print(f"\n[Diversity Score]: {std_dev:.4f}")
        if std_dev < 0.01:
            print("[!] WARNING: Diversity is very low. Model might have collapsed.")
        else:
            print("[✅] Diversity looks healthy.")

    except Exception as e:
        print(f"[!] Error testing model: {e}")

if __name__ == "__main__":
    check_model()