# adversarial_training.py
# Description: This script performs Phase 2 of our adversarial process: the defense.
# It loads our best-performing model, combines the original training data with the
# newly generated adversarial samples, and then fine-tunes the model on this
# "hardened" dataset. The result is a more robust model that is resistant to
# the adversarial attacks we just created.

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle

print("--- Phase 2: Adversarial Training (Defense) ---")

# --- Step 1: Load the Model, Original Data, and Adversarial Data ---
print("\n[*] Loading the model and all necessary datasets...")
try:
    # Load the model we are going to harden.
    model = keras.models.load_model("../CNN+LSTM/cicids_spatiotemporal_model.keras")

    # Load the original training data.
    DATA_PATH = "../Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/"
    X_train = np.load(DATA_PATH + 'X_train.npy')
    y_train = np.load(DATA_PATH + 'y_train.npy')

    # Load the adversarial samples we created in Phase 1.
    ADVERSARIAL_PATH = "./Adversarial_Samples/"
    X_adversarial = np.load(ADVERSARIAL_PATH + "X_adversarial.npy")
    y_adversarial = np.load(ADVERSARIAL_PATH + "y_adversarial.npy")
    
    print("[+] All files loaded successfully.")
except Exception as e:
    print(f"\n[!] Error loading files: {e}")
    print("[!] Please ensure all required models and datasets are present.")
    exit()

# --- Step 2: Create the Hardened Training Dataset ---
print("\n[*] Creating the new, hardened training dataset...")

# **THE FIX**: We must ensure both arrays are 3D before we can combine them.
# The model expects a 3D input, so we'll reshape the original X_train.
if len(X_train.shape) == 2:
    X_train = np.expand_dims(X_train, -1)

# Combine the original training data (now 3D) with our adversarial samples (already 3D).
X_train_hardened = np.concatenate([X_train, X_adversarial])
y_train_hardened = np.concatenate([y_train, y_adversarial])

# It's very important to shuffle the combined dataset.
X_train_hardened, y_train_hardened = shuffle(X_train_hardened, y_train_hardened, random_state=42)

print(f"[+] New training set created with {len(X_train_hardened)} total samples.")
# The data is now correctly shaped, so no more reshaping is needed.


# --- Step 3: Fine-Tune (Retrain) the Model ---
# We don't need to train for many epochs. We are just "fine-tuning" the expert.
# We also use a lower learning rate to make small, careful adjustments.
print("\n[*] Fine-tuning the model on the hardened dataset...")

# Re-compile the model with a lower learning rate for fine-tuning.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # Lower LR
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    X_train_hardened, y_train_hardened,
    epochs=5, # Just a few epochs are needed for fine-tuning.
    batch_size=128,
    verbose=1
)
print("[+] Fine-tuning complete!")

# --- Step 4: Save the New, Hardened Model ---
print("\n[*] Saving the final, hardened model...")
model.save("cicids_spatiotemporal_model_hardened.keras")
print("[+] Hardened model saved as 'cicids_spatiotemporal_model_hardened.keras'")


# --- Step 5: Verify the Defense ---
# Now, let's see if our hardened model can resist the same attack that fooled the original.
print("\n[*] Verifying the defense by re-running the attack on the new model...")

# We need the original, un-hardened model to create the same adversarial sample again.
try:
    original_model = keras.models.load_model("../CNN+LSTM/cicids_spatiotemporal_model.keras")
except Exception:
    original_model = model # Fallback, though results may differ

def create_adversarial_pattern(input_data, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_data)
        # Use the ORIGINAL model to find the weakness
        prediction = original_model(input_data)
        loss = keras.losses.BinaryCrossentropy()(input_label, prediction)
    gradient = tape.gradient(loss, input_data)
    return tf.sign(gradient)

# Find the same attack sample from before to test.
X_test = np.load(DATA_PATH + 'X_test.npy')
y_test = np.load(DATA_PATH + 'y_test.npy')
if len(X_test.shape) == 2:
    X_test = np.expand_dims(X_test, -1)

for i in range(len(X_test)):
    if y_test[i] == 1:
        sample = X_test[i]
        sample_3d = np.reshape(sample, (1, sample.shape[0], sample.shape[1]))
        
        # Check that the original model is still fooled
        original_sample_tf = tf.convert_to_tensor(sample_3d)
        original_label_tf = tf.convert_to_tensor([[1.0]])
        perturbations = create_adversarial_pattern(original_sample_tf, original_label_tf)
        adversarial_sample = tf.clip_by_value(original_sample_tf + 0.02 * perturbations, 0, 1)

        original_pred_proba = original_model.predict(adversarial_sample, verbose=0)[0][0]
        if original_pred_proba < 0.5:
             print(f"\n[+] Found an adversarial sample that fools the original model (Prediction: {original_pred_proba:.2%}).")
             
             # Now, test the HARDENED model on that same tricky sample.
             hardened_pred_proba = model.predict(adversarial_sample, verbose=0)[0][0]
             print(f"    Prediction from our NEW hardened model: {hardened_pred_proba:.2%}")

             if hardened_pred_proba > 0.5:
                 print("\n[SUCCESS] The defense worked! The hardened model correctly identified the disguised attack.")
             else:
                 print("\n[FAILURE] The hardened model was still fooled, though it may be more confident.")
             
             break

