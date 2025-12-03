# train_cgan.py
# Robust CGAN Training with LeakyReLU, BatchNormalization, and Dropout

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

print("--- Training a Robust Conditional GAN (CGAN) ---")

# --- Configuration ---
DATA_PATH = "../Preprocessing/CGAN/CGAN_preprocessed_data/"
LATENT_DIM = 128
DATA_DIM = 78
NUM_CLASSES = 15
EPOCHS = 100        # Full training duration
BATCH_SIZE = 128    # Batch size
LEARNING_RATE = 0.0002 
BETA_1 = 0.5        # Adam Beta1 (Crucial for GAN stability)

# --- Step 1: Load Data ---
print(f"\n[*] Loading dataset from '{DATA_PATH}'...")
try:
    X_train = np.load(DATA_PATH + 'X_full.npy')
    y_train = np.load(DATA_PATH + 'y_full.npy')
    print(f"[+] Dataset loaded: {len(X_train):,} samples.")
except FileNotFoundError:
    print(f"[!] Error: Data not found at '{DATA_PATH}'")
    exit()

# One-hot encode labels
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)

# --- Step 2: Build Generator (Robust Architecture) ---
def build_generator():
    noise = layers.Input(shape=(LATENT_DIM,))
    label = layers.Input(shape=(NUM_CLASSES,))
    
    # Merge noise and label
    x = layers.concatenate([noise, label])
    
    # Layer 1
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Layer 2
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Layer 3 (Deep layer for full dataset)
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Output Layer
    # We use 'sigmoid' because your data check confirmed it is [0, 1]
    output = layers.Dense(DATA_DIM, activation='sigmoid')(x) 
    
    model = keras.Model([noise, label], output, name="generator")
    return model

generator = build_generator()
print("\n[*] Generator Built:")
generator.summary()

# --- Step 3: Build Discriminator (Robust Architecture) ---
def build_discriminator():
    img = layers.Input(shape=(DATA_DIM,))
    label = layers.Input(shape=(NUM_CLASSES,))
    
    x = layers.concatenate([img, label])
    
    # Layer 1
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.4)(x) 
    
    # Layer 2
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.4)(x)
    
    # Layer 3
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Output
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model([img, label], output, name="discriminator")
    
    # Compile
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

discriminator = build_discriminator()
print("\n[*] Discriminator Built:")
discriminator.summary()

# --- Step 4: Build Combined Model ---
discriminator.trainable = False
noise = layers.Input(shape=(LATENT_DIM,))
label = layers.Input(shape=(NUM_CLASSES,))
img = generator([noise, label])
valid = discriminator([img, label])

cgan = keras.Model([noise, label], valid)
opt_gan = keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
cgan.compile(loss='binary_crossentropy', optimizer=opt_gan)

# --- Step 5: Training Loop ---
print("\n--- Starting Full CGAN Training ---")

# Labels for training (with smoothing for real labels)
real_labels = np.ones((BATCH_SIZE, 1)) * 0.9 
fake_labels = np.zeros((BATCH_SIZE, 1))

num_batches = X_train.shape[0] // BATCH_SIZE

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for i in range(num_batches):
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        # Select random real samples
        idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
        imgs, labels = X_train[idx], y_train_one_hot[idx]
        
        # Generate fake samples
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        # Randomly sample labels for the fake data
        sampled_labels_indices = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
        sampled_labels = tf.keras.utils.to_categorical(sampled_labels_indices, NUM_CLASSES)
        
        gen_imgs = generator.predict([noise, sampled_labels], verbose=0)
        
        # Train
        d_loss_real = discriminator.train_on_batch([imgs, labels], real_labels)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, sampled_labels], fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        # We want the discriminator to mistake these as real (label=1.0)
        valid_y = np.ones((BATCH_SIZE, 1))
        
        # Train
        g_loss = cgan.train_on_batch([noise, sampled_labels], valid_y)
        
        if i % 100 == 0:
             print(f"  Batch {i}/{num_batches} [D loss: {d_loss[0]:.4f}, acc: {d_loss[1]*100:.2f}%] [G loss: {g_loss:.4f}]")

    # Save model checkpoint
    generator.save(f"cgan_generator_epoch_{epoch+1}.keras")
    print(f"[+] Saved checkpoint: cgan_generator_epoch_{epoch+1}.keras")

print("\n--- Training Complete! ---")
generator.save("cgan_generator_final.keras")