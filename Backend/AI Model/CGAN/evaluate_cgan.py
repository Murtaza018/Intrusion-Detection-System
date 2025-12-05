# evaluate_cgan_local.py
# Script to evaluate CGAN checkpoints locally in VS Code

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import os

# --- CONFIGURATION ---
# Path to the model you want to test (Relative to this script)
# Matches the save path from train_cgan_local.py
MODEL_NAME = "./Models/cgan_generator_epoch_1.keras"
NUM_SAMPLES = 10

def evaluate_generator():
    # Verify file exists locally
    if not os.path.exists(MODEL_NAME):
        print(f"[!] Error: Could not find model file at: {os.path.abspath(MODEL_NAME)}")
        print("    Make sure the training script has finished at least 1 epoch.")
        return

    print(f"[*] Loading generator from {MODEL_NAME}...")
    try:
        generator = load_model(MODEL_NAME)
        print("[+] Model loaded successfully.")
    except Exception as e:
        print(f"[!] Failed to load model: {e}")
        return

    # --- AUTO-DETECT INPUT SHAPES ---
    print("\n[*] Inspecting Model Inputs...")
    try:
        input_shapes = generator.input_shape
        print(f"    Model expects: {input_shapes}")

        # Defaults
        latent_dim = 100
        label_dim = 1
        noise_idx = 0
        label_idx = 1

        # Logic to determine which input is noise vs label
        if isinstance(input_shapes, (list, tuple)) and len(input_shapes) == 2:
            shape1 = input_shapes[0] # e.g. (None, 128)
            shape2 = input_shapes[1] # e.g. (None, 15)
            
            # Handle None dimensions
            dim1 = shape1[1]
            dim2 = shape2[1]
            
            # The larger dimension is usually the latent noise
            if dim1 > dim2:
                latent_dim = dim1
                label_dim = dim2
                noise_idx = 0
                label_idx = 1
            else:
                latent_dim = dim2
                label_dim = dim1
                noise_idx = 1
                label_idx = 0
                
            print(f"    -> Detected Latent Dim: {latent_dim} (Input {noise_idx})")
            print(f"    -> Detected Label Dim:  {label_dim} (Input {label_idx})")
        else:
            print("[!] Could not auto-detect multiple inputs. Using default (100).")

    except Exception as e:
        print(f"[!] Inspection failed: {e}")
        return

    print(f"\n[*] Generating {NUM_SAMPLES} synthetic 'Normal' packets...")
    
    # 1. Prepare Noise
    random_latent_vectors = tf.random.normal(shape=(NUM_SAMPLES, latent_dim))
    
    # 2. Prepare Labels (One-Hot Encoded if dim > 1)
    if label_dim > 1:
        # Assuming Class 0 is "Normal"
        labels = np.zeros((NUM_SAMPLES, label_dim))
        labels[:, 0] = 1 
        print(f"    -> Created One-Hot Encoded labels for Class 0 (Normal)")
    else:
        # Binary classification
        labels = tf.zeros((NUM_SAMPLES, 1))
        print(f"    -> Created Binary labels (0)")

    # 3. Arrange Inputs correctly
    inputs = [None, None]
    inputs[noise_idx] = random_latent_vectors
    inputs[label_idx] = labels

    # 4. Generate
    try:
        generated_data = generator.predict(inputs, verbose=0)
    except Exception as e:
        print(f"\n[!] Generation failed!")
        print(f"    Error details: {e}")
        return

    # 5. Analyze Results
    print("\n" + "="*50)
    print("ANALYSIS OF GENERATED PACKETS")
    print("="*50)
    
    df = pd.DataFrame(generated_data)
    
    print(f"1. Value Range Check:")
    print(f"   Min Value: {df.min().min():.4f}")
    print(f"   Max Value: {df.max().max():.4f}")

    print(f"\n2. Variance Check (Mode Collapse detection):")
    std_dev = df.std().mean()
    print(f"   Avg Feature Std Dev: {std_dev:.4f}")
    
    if std_dev < 0.01:
        print("   [!] WARNING: Very low variance. Model suffers from MODE COLLAPSE.")
        print("       It is generating the same packet repeatedly.")
    else:
        print("   [OK] Variance looks healthy.")

    print(f"\n3. Sample Output (First 3 packets, first 5 features):")
    print(df.iloc[:3, :5].to_string())
    print("\n" + "="*50)

if __name__ == "__main__":
    evaluate_generator()