# train_cgan.py
# Description: This script builds and trains a Conditional GAN (CGAN).
# The Generator will learn to create synthetic network traffic that matches
# a specific "condition" or label (e.g., 'Normal', 'DDoS', 'PortScan').
# This gives us a "vending machine" for creating new, specific types of
# training data on demand.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

print("--- Training a Conditional GAN (CGAN) on CIC-IDS-2017 ---")

# --- Configuration ---
DATA_PATH = "../Preprocessing/CGAN/CGAN_preprocessed_data/"
LATENT_DIM = 128    # Input dimension for the random noise
DATA_DIM = 78       # Number of features in our dataset (must match preprocessor)
NUM_CLASSES = 15    # Number of classes (must match preprocessor output)
EPOCHS = 100
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
# The Generator now takes TWO inputs: random noise and a class label.
print("[*] Building the Generator model...")

# Input 1: The random noise vector
noise_input = layers.Input(shape=(LATENT_DIM,), name="noise_input")
# Input 2: The conditional label
label_input = layers.Input(shape=(NUM_CLASSES,), name="label_input")

# Combine the two inputs
merged_input = layers.concatenate([noise_input, label_input])

# Upscale the merged input to create a fake data sample
gen = layers.Dense(128, activation="relu")(merged_input)
gen = layers.Dense(256, activation="relu")(gen)
gen = layers.Dense(512, activation="relu")(gen)
gen = layers.Dense(DATA_DIM, activation="sigmoid")(gen) # Output layer matches our data shape

generator = keras.Model([noise_input, label_input], gen, name="generator")
generator.summary()


# --- Step 3: Build the Discriminator ---
# The Discriminator also takes TWO inputs: a data sample and its label.
# It must decide if the sample is (a) real and (b) matches the provided label.
print("\n[*] Building the Discriminator model...")

# Input 1: The network traffic data
data_input = layers.Input(shape=(DATA_DIM,), name="data_input")
# Input 2: The conditional label
label_input_d = layers.Input(shape=(NUM_CLASSES,), name="label_input_d")

# Combine the two inputs
merged_input_d = layers.concatenate([data_input, label_input_d])

# Downscale the merged input to a single probability (Real or Fake)
disc = layers.Dense(512, activation="relu")(merged_input_d)
disc = layers.Dense(256, activation="relu")(disc)
disc = layers.Dense(128, activation="relu")(disc)
disc = layers.Dense(1, activation="sigmoid")(disc) # Output: 1 for Real, 0 for Fake

discriminator = keras.Model([data_input, label_input_d], disc, name="discriminator")
discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
discriminator.summary()


# --- Step 4: Build the Combined CGAN Model ---
# To train the generator, we "freeze" the discriminator and chain them together.
# We feed the CGAN (noise + label) and check if the discriminator is fooled.
discriminator.trainable = False

noise, label = generator.inputs
gen_output = generator.outputs[0]
gan_output = discriminator([gen_output, label])

cgan = keras.Model([noise, label], gan_output, name="cgan")
cgan.compile(optimizer="adam", loss="binary_crossentropy")
print("\n[*] Building the combined CGAN model...")
cgan.summary()


# --- Step 5: Train the CGAN ---
print("\n--- Starting CGAN Training ---")
# This is a complex training loop that alternates between training
# the discriminator and the generator.

real_labels_y = np.ones((BATCH_SIZE, 1))
fake_labels_y = np.zeros((BATCH_SIZE, 1))

num_batches = X_train.shape[0] // BATCH_SIZE

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    
    for i in range(num_batches):
        # --- 1. Train the Discriminator ---
        
        # Get a batch of real samples
        idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
        real_samples = X_train[idx]
        real_labels = y_train_one_hot[idx]
        
        # Generate a batch of fake samples
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        fake_labels = tf.keras.utils.to_categorical(
            np.random.randint(0, NUM_CLASSES, BATCH_SIZE), num_classes=NUM_CLASSES
        )
        fake_samples = generator.predict([noise, fake_labels], verbose=0)
        
        # Train the discriminator (on real and fake data)
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch([real_samples, real_labels], real_labels_y)
        d_loss_fake = discriminator.train_on_batch([fake_samples, fake_labels], fake_labels_y)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # --- 2. Train the Generator ---
        
        # Generate new noise and random labels to try and fool the discriminator
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        trick_labels = tf.keras.utils.to_categorical(
            np.random.randint(0, NUM_CLASSES, BATCH_SIZE), num_classes=NUM_CLASSES
        )
        
        # Train the generator (by training the combined CGAN model)
        discriminator.trainable = False
        g_loss = cgan.train_on_batch([noise, trick_labels], real_labels_y)
        
        if i % 200 == 0:
            print(f"    Batch {i}/{num_batches}  [D loss: {d_loss[0]:.4f}, acc: {d_loss[1]*100:.2f}%]  [G loss: {g_loss:.4f}]")

    # Save the generator model after each epoch
    generator.save(f"cgan_generator_epoch_{epoch+1}.keras")
    print(f"[+] Saved generator model for epoch {epoch+1}")

print("\n--- CGAN Training Complete! ---")
generator.save("cgan_generator_final.keras")
print("[*] Final generator model saved as 'cgan_generator_final.keras'")