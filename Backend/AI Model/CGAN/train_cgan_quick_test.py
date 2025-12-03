# train_cgan_quick_test.py
# Fast sanity check for CGAN architecture

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# --- QUICK TEST CONFIG ---
DATA_PATH = "../Preprocessing/CGAN/CGAN_preprocessed_data/"
LATENT_DIM = 128
DATA_DIM = 78
NUM_CLASSES = 15
EPOCHS = 5          # <--- ONLY 5 EPOCHS
BATCH_SIZE = 64     # Smaller batch for faster updates on small data
SUBSET_SIZE = 5000  # <--- Only use 5,000 samples for speed

print(f"[*] Starting Quick Test: {EPOCHS} Epochs on {SUBSET_SIZE} samples...")

# --- 1. Load & Subset Data ---
try:
    X_full = np.load(DATA_PATH + 'X_full.npy')
    y_full = np.load(DATA_PATH + 'y_full.npy')
    
    # Take a random subset
    idx = np.random.randint(0, X_full.shape[0], SUBSET_SIZE)
    X_train = X_full[idx]
    y_train = y_full[idx]
    print(f"[+] Loaded subset: {X_train.shape}")
except:
    print(f"[!] Error loading data from {DATA_PATH}")
    exit()

y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)

# --- 2. Build Models (Robust Architecture) ---
# Generator
noise = layers.Input(shape=(LATENT_DIM,))
label = layers.Input(shape=(NUM_CLASSES,))
x = layers.concatenate([noise, label])
x = layers.Dense(256)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dense(512)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(0.2)(x)
# IMPORTANT: Check your data scaling from Step 1! 
# Defaulting to sigmoid (0-1) here. Change to 'tanh' or 'linear' if Step 1 says so.
output = layers.Dense(DATA_DIM, activation='sigmoid')(x) 
generator = keras.Model([noise, label], output)

# Discriminator
img = layers.Input(shape=(DATA_DIM,))
label_d = layers.Input(shape=(NUM_CLASSES,))
x = layers.concatenate([img, label_d])
x = layers.Dense(512)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
valid = layers.Dense(1, activation='sigmoid')(x)
discriminator = keras.Model([img, label_d], valid)
discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

# Combined
discriminator.trainable = False
valid = discriminator([generator([noise, label]), label])
cgan = keras.Model([noise, label], valid)
cgan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))

# --- 3. Quick Training Loop ---
real_labels = np.ones((BATCH_SIZE, 1)) * 0.9
fake_labels = np.zeros((BATCH_SIZE, 1))

for epoch in range(EPOCHS):
    # Train D
    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
    imgs, labels = X_train[idx], y_train_one_hot[idx]
    
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    sampled_labels = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
    fake_labels_input = tf.keras.utils.to_categorical(sampled_labels, NUM_CLASSES)
    gen_imgs = generator.predict([noise, fake_labels_input], verbose=0)
    
    d_loss_real = discriminator.train_on_batch([imgs, labels], real_labels)
    d_loss_fake = discriminator.train_on_batch([gen_imgs, fake_labels_input], fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train G
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    sampled_labels = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
    valid_labels_input = tf.keras.utils.to_categorical(sampled_labels, NUM_CLASSES)
    g_loss = cgan.train_on_batch([noise, valid_labels_input], np.ones((BATCH_SIZE, 1)))
    
    print(f"Epoch {epoch+1}/{EPOCHS} [D loss: {d_loss[0]:.4f}] [G loss: {g_loss:.4f}]")

# Save Test Model
generator.save("cgan_generator.keras")
print("[+] Test model saved. Run evaluate_cgan.py now!")