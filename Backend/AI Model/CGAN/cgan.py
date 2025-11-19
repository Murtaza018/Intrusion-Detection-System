# train_cgan.py
# Description: This script builds and trains a Conditional GAN (CGAN).
# UPDATE: Implements "One-Sided Label Smoothing" to fix the Discriminator Overpowering problem.
# We use 0.9 instead of 1.0 for real labels to stabilize training.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

print("--- Training a Conditional GAN (CGAN) with Label Smoothing ---")

# --- Configuration ---
DATA_PATH = "../Preprocessing/CGAN/CGAN_preprocessed_data/"
LATENT_DIM = 128    # Input dimension for the random noise
DATA_DIM = 78       # Number of features in our dataset
NUM_CLASSES = 15    # Number of classes
EPOCHS = 50         # Reduced to 50 for this attempt
BATCH_SIZE = 256

# --- Step 1: Load the Multi-Class Preprocessed Data ---
print(f"\n[*] Loading the CGAN-preprocessed dataset from '{DATA_PATH}'...")
try:
    X_train = np.load(DATA_PATH + 'X_full.npy')
    y_train = np.load(DATA_PATH + 'y_full.npy')
    print(f"[+] Dataset loaded: {len(X_train):,} samples.")
except FileNotFoundError:
    print(f"\n[!] Error: I couldn't find the processed data files at '{DATA_PATH}'.")
    print("[!] Please run 'preprocess_cgan.py' first.")
    exit()

# Convert labels to one-hot encoding for the model
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)

# --- Step 2: Build the Generator ---
print("[*] Building the Generator model...")
noise_input = layers.Input(shape=(LATENT_DIM,), name="noise_input")
label_input = layers.Input(shape=(NUM_CLASSES,), name="label_input")
merged_input = layers.concatenate([noise_input, label_input])

gen = layers.Dense(128, activation="relu")(merged_input)
gen = layers.Dense(256, activation="relu")(gen)
gen = layers.Dense(512, activation="relu")(gen)
gen = layers.Dense(DATA_DIM, activation="sigmoid")(gen)

generator = keras.Model([noise_input, label_input], gen, name="generator")
generator.summary()

# --- Step 3: Build the Discriminator ---
print("\n[*] Building the Discriminator model...")
data_input = layers.Input(shape=(DATA_DIM,), name="data_input")
label_input_d = layers.Input(shape=(NUM_CLASSES,), name="label_input_d")
merged_input_d = layers.concatenate([data_input, label_input_d])

disc = layers.Dense(512, activation="relu")(merged_input_d)
disc = layers.Dense(256, activation="relu")(disc)
disc = layers.Dense(128, activation="relu")(disc)
disc = layers.Dense(1, activation="sigmoid")(disc)

discriminator = keras.Model([data_input, label_input_d], disc, name="discriminator")
discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
discriminator.summary()

# --- Step 4: Build the Combined CGAN Model ---
discriminator.trainable = False
noise, label = generator.inputs
gen_output = generator.outputs[0]
gan_output = discriminator([gen_output, label])
cgan = keras.Model([noise, label], gan_output, name="cgan")
cgan.compile(optimizer="adam", loss="binary_crossentropy")
print("\n[*] Building the combined CGAN model...")
cgan.summary()

# --- Step 5: Train the CGAN ---
print("\n--- Starting CGAN Training (Stabilized) ---")

# ** THE FIX: Label Smoothing **
# Instead of 1.0, we use 0.9 for real labels. This prevents the discriminator
# from becoming too confident and killing the gradients.
real_labels_y = np.ones((BATCH_SIZE, 1)) * 0.9 
fake_labels_y = np.zeros((BATCH_SIZE, 1))

num_batches = X_train.shape[0] // BATCH_SIZE

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    
    for i in range(num_batches):
        # --- 1. Train the Discriminator ---
        idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
        real_samples = X_train[idx]
        real_labels = y_train_one_hot[idx]
        
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        fake_labels = tf.keras.utils.to_categorical(
            np.random.randint(0, NUM_CLASSES, BATCH_SIZE), num_classes=NUM_CLASSES
        )
        fake_samples = generator.predict([noise, fake_labels], verbose=0)
        
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch([real_samples, real_labels], real_labels_y)
        d_loss_fake = discriminator.train_on_batch([fake_samples, fake_labels], fake_labels_y)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # --- 2. Train the Generator ---
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        trick_labels = tf.keras.utils.to_categorical(
            np.random.randint(0, NUM_CLASSES, BATCH_SIZE), num_classes=NUM_CLASSES
        )
        
        discriminator.trainable = False
        # Ideally, we want the discriminator to think these are REAL (1.0, not 0.9)
        # when training the generator, to push it towards perfection.
        valid_y = np.ones((BATCH_SIZE, 1)) 
        g_loss = cgan.train_on_batch([noise, trick_labels], valid_y)
        
        if i % 200 == 0:
            print(f"    Batch {i}/{num_batches}  [D loss: {d_loss[0]:.4f}, acc: {d_loss[1]*100:.2f}%]  [G loss: {g_loss:.4f}]")

    # Save the generator model after each epoch
    generator.save(f"cgan_generator_epoch_{epoch+1}.keras")
    print(f"[+] Saved generator model for epoch {epoch+1}")

print("\n--- CGAN Training Complete! ---")
generator.save("cgan_generator_final.keras")
print("[*] Final generator model saved as 'cgan_generator_final.keras'")