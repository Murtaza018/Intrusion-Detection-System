# train_cgan_local.py
# Complete, High-Performance CGAN Training Script for Local VS Code execution.

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gc

# Enable GPU memory growth to prevent allocation errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[*] GPU Detected: {len(gpus)} device(s). Memory growth enabled.")
    except RuntimeError as e:
        print(e)

print("--- 🚀 Initializing High-Performance CGAN Training (Local) ---")

# --- 1. CONFIGURATION ---
# Adjust these paths to match your local folder structure
DATA_PATH = "../Preprocessing/CGAN/CGAN_preprocessed_data/" 
SAVE_PATH = "./Models/"

# Hyperparameters
LATENT_DIM = 128       
DATA_DIM = 78          
NUM_CLASSES = 15       
EPOCHS = 100           
BATCH_SIZE = 64        # Small batch size for stability
LEARNING_RATE = 0.0002 
BETA_1 = 0.5           

# --- SUBSET CONFIGURATION ---
# Set to False to train on the full dataset (will take longer)
USE_SUBSET = True
SUBSET_SIZE = 200000

# Ensure save directory exists
os.makedirs(SAVE_PATH, exist_ok=True)

# --- 2. LOAD AND SUBSET DATA ---
print(f"\n[*] Loading dataset from {DATA_PATH}...")
try:
    X_train_full = np.load(os.path.join(DATA_PATH, 'X_full.npy'))
    y_train_full = np.load(os.path.join(DATA_PATH, 'y_full.npy'))
    print(f"[+] Full Dataset loaded: {len(X_train_full):,} samples.")
except FileNotFoundError:
    print(f"[!] CRITICAL ERROR: Could not find .npy files in {DATA_PATH}")
    print("    Please check your path.")
    exit()

if USE_SUBSET and len(X_train_full) > SUBSET_SIZE:
    print(f"\n[*] ✂️ Subsetting data to {SUBSET_SIZE:,} random samples...")
    
    # Create random indices
    indices = np.random.permutation(len(X_train_full))[:SUBSET_SIZE]
    
    # Select subset
    X_train = X_train_full[indices]
    y_train = y_train_full[indices]
    
    # Free up memory
    del X_train_full
    del y_train_full
    gc.collect()
    
    print(f"[+] Subset ready: {len(X_train):,} samples.")
else:
    X_train = X_train_full
    y_train = y_train_full

# One-Hot Encode Labels
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)

# --- 3. BUILD GENERATOR ---
def build_generator():
    noise = layers.Input(shape=(LATENT_DIM,), name="noise_input")
    label = layers.Input(shape=(NUM_CLASSES,), name="label_input")
    x = layers.concatenate([noise, label])
    
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    output = layers.Dense(DATA_DIM, activation='sigmoid')(x) 
    
    model = keras.Model([noise, label], output, name="generator")
    return model

generator = build_generator()

# --- 4. BUILD DISCRIMINATOR ---
def build_discriminator():
    img = layers.Input(shape=(DATA_DIM,), name="data_input")
    label = layers.Input(shape=(NUM_CLASSES,), name="label_input")
    x = layers.concatenate([img, label])
    
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x) 
    
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model([img, label], output, name="discriminator")
    
    # 10x Slower Learning Rate for Discriminator
    opt = keras.optimizers.Adam(learning_rate=0.00002, beta_1=BETA_1)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

discriminator = build_discriminator()

# --- 5. BUILD COMBINED MODEL ---
discriminator.trainable = False
noise = layers.Input(shape=(LATENT_DIM,))
label = layers.Input(shape=(NUM_CLASSES,))
img = generator([noise, label])
valid = discriminator([img, label])

cgan = keras.Model([noise, label], valid)
opt_gan = keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
cgan.compile(loss='binary_crossentropy', optimizer=opt_gan)

# --- 6. TRAINING LOOP ---
print("\n" + "="*50)
print("🏁 STARTING TRAINING LOOP")
print(f"    Batch Size: {BATCH_SIZE}")
print(f"    Batches per Epoch: {len(X_train) // BATCH_SIZE}")
print("="*50)

real_labels = np.ones((BATCH_SIZE, 1)) * 0.95 
fake_labels = np.zeros((BATCH_SIZE, 1))
num_batches = X_train.shape[0] // BATCH_SIZE

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for i in range(num_batches):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
        imgs, labels = X_train[idx], y_train_one_hot[idx]
        
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        sampled_labels = tf.keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, BATCH_SIZE), num_classes=NUM_CLASSES)
        
        gen_imgs = generator.predict([noise, sampled_labels], verbose=0)
        
        d_loss_real = discriminator.train_on_batch([imgs, labels], real_labels)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, sampled_labels], fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        valid_y = np.ones((BATCH_SIZE, 1)) 
        g_loss = cgan.train_on_batch([noise, sampled_labels], valid_y)
        
        if i % 100 == 0:
             print(f"  Batch {i}/{num_batches} [D loss: {d_loss[0]:.4f}] [G loss: {g_loss:.4f}]")

    # Save Checkpoint
    save_loc = os.path.join(SAVE_PATH, f"cgan_generator_epoch_{epoch+1}.keras")
    generator.save(save_loc)
    print(f"[+] Saved checkpoint: {save_loc}")

print("\n" + "="*50)
print("🎉 TRAINING COMPLETE!")
generator.save(os.path.join(SAVE_PATH, "cgan_generator_final.keras"))
print("="*50)