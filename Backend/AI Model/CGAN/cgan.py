# train_wgan_gp_colab.py
# Robust WGAN-GP Implementation for Tabular Data
# FIXED: train_step data unpacking

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from google.colab import drive

print("--- 🚀 Initializing WGAN-GP Training ---")

# --- 1. MOUNT GOOGLE DRIVE ---
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# --- 2. CONFIGURATION ---
DRIVE_SOURCE_PATH = "/content/drive/My Drive/IDS_Project/"
SAVE_PATH = "/content/drive/My Drive/IDS_Project/Models/"
LOCAL_DATA_PATH = "/content/data_cache/"

# Hyperparameters
LATENT_DIM = 128
DATA_DIM = 78
NUM_CLASSES = 15
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
CRITIC_EXTRA_STEPS = 5
GP_WEIGHT = 10.0

# Subset for speed/stability
USE_SUBSET = True
SUBSET_SIZE = 100000

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOCAL_DATA_PATH, exist_ok=True)

# --- 3. DATA TRANSFER ---
x_src = os.path.join(DRIVE_SOURCE_PATH, 'X_full.npy')
y_src = os.path.join(DRIVE_SOURCE_PATH, 'y_full.npy')
x_dst = os.path.join(LOCAL_DATA_PATH, 'X_full.npy')
y_dst = os.path.join(LOCAL_DATA_PATH, 'y_full.npy')

if not os.path.exists(x_dst):
    try:
        shutil.copy(x_src, x_dst)
        shutil.copy(y_src, y_dst)
    except FileNotFoundError:
        print(f"[!] Critical Error: Files not found at {DRIVE_SOURCE_PATH}")
        raise

# --- 4. LOAD DATA ---
print(f"[*] Loading data...")
X_train_full = np.load(x_dst)
y_train_full = np.load(y_dst)

if USE_SUBSET and len(X_train_full) > SUBSET_SIZE:
    print(f"[*] Subsetting to {SUBSET_SIZE} samples...")
    indices = np.random.permutation(len(X_train_full))[:SUBSET_SIZE]
    X_train = X_train_full[indices]
    y_train = y_train_full[indices]
    del X_train_full, y_train_full
    import gc; gc.collect()
else:
    X_train = X_train_full
    y_train = y_train_full

# --- CRITICAL FIX: Cast to float32 ---
X_train = X_train.astype('float32')
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES).astype('float32')
# -------------------------------------

# --- 5. BUILD MODELS ---

def build_critic():
    img = layers.Input(shape=(DATA_DIM,))
    label = layers.Input(shape=(NUM_CLASSES,))
    x = layers.concatenate([img, label])

    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    output = layers.Dense(1)(x)

    return keras.Model([img, label], output, name="critic")

def build_generator():
    noise = layers.Input(shape=(LATENT_DIM,))
    label = layers.Input(shape=(NUM_CLASSES,))
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
    return keras.Model([noise, label], output, name="generator")

critic = build_critic()
generator = build_generator()

# --- 6. WGAN-GP MODEL CLASS ---
class WGAN_GP(keras.Model):
    def __init__(self, critic, generator, latent_dim, critic_extra_steps=5, gp_weight=10.0):
        super(WGAN_GP, self).__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = critic_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN_GP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images, labels):
        # Interpolate between real and fake
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the critic output for this interpolated image
            pred = self.critic([interpolated, labels], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        # 4. Penalty: distance from 1.0
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        # FIX: Correct unpacking of data tuple (X, y)
        real_images, labels = data
        batch_size = tf.shape(real_images)[0]

        # --- Train Critic ---
        for i in range(self.d_steps):
            noise = tf.random.normal([batch_size, self.latent_dim])

            with tf.GradientTape() as tape:
                fake_images = self.generator([noise, labels], training=True)
                fake_logits = self.critic([fake_images, labels], training=True)
                real_logits = self.critic([real_images, labels], training=True)

                d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images, labels)
                d_loss = d_cost + gp * self.gp_weight

            d_gradient = tape.gradient(d_loss, self.critic.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.critic.trainable_variables))

        # --- Train Generator ---
        noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as tape:
            fake_images = self.generator([noise, labels], training=True)
            gen_img_logits = self.critic([fake_images, labels], training=True)
            g_loss = -tf.reduce_mean(gen_img_logits)

        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# --- 7. COMPILE & TRAIN ---
wgan = WGAN_GP(critic=critic, generator=generator, latent_dim=LATENT_DIM)

wgan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.0, beta_2=0.9),
    g_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.0, beta_2=0.9),
    d_loss_fn=None,
    g_loss_fn=None
)

print("\n--- Starting WGAN-GP Training ---")

class SaveCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        save_loc = os.path.join(SAVE_PATH, f"cgan_generator_epoch_{epoch+1}.keras")
        self.model.generator.save(save_loc)
        print(f"\n[+] Saved checkpoint: {save_loc}")

# Pass data as a tuple (X, y) directly, not nested
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_one_hot))
dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

wgan.fit(dataset, epochs=EPOCHS, callbacks=[SaveCallback()])

print("\n🎉 WGAN-GP Training Complete!")
final_save_loc = os.path.join(SAVE_PATH, "cgan_generator_final.keras")
generator.save(final_save_loc)