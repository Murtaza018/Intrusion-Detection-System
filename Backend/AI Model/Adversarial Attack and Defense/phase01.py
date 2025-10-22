# generate_adversarial_samples.py
# Description: This script performs an adversarial attack against our trained CNN+LSTM model.
# It uses the Fast Gradient Sign Method (FGSM) to create slightly perturbed attack
# samples that are designed to be misclassified by the model as "Normal".

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

print("--- Phase 1: Adversarial Attack Simulation ---")

# --- Step 1: Load the Model and Data ---
# We need our best classifier and some real attack data to work with.
print("\n[*] Loading our best-performing model and the test dataset...")
try:
    # Load the model we are going to attack.
    model = keras.models.load_model("../CNN+LSTM/cicids_spatiotemporal_model.keras")
    
    # Load the test data to find a sample to attack.
    DATA_PATH = "../Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/"
    X_test = np.load(DATA_PATH + 'X_test.npy')
    y_test = np.load(DATA_PATH + 'y_test.npy')
    
    print("[+] Model and data loaded successfully.")
except Exception as e:
    print(f"\n[!] Error loading files: {e}")
    print("[!] Please ensure 'cicids_spatiotemporal_model.keras' and the test data are present.")
    exit()

# The model expects a 3D input, so let's make sure our data is in the right shape.
if len(X_test.shape) == 2:
    X_test = np.expand_dims(X_test, -1)

# --- Step 2: Define the Adversarial Attack Function (FGSM) ---
# This function will create the disguised attack sample.
def create_adversarial_pattern(input_data, input_label):
    """
    Generates an adversarial perturbation using the Fast Gradient Sign Method.
    """
    # We need to use TensorFlow's GradientTape to find out how to change the
    # input data to fool the model.
    with tf.GradientTape() as tape:
        # We need to explicitly watch the input data so we can get gradients.
        tape.watch(input_data)
        prediction = model(input_data)
        loss_fn = keras.losses.BinaryCrossentropy()
        loss = loss_fn(input_label, prediction)

    # Get the gradients of the loss with respect to the input image.
    gradient = tape.gradient(loss, input_data)
    # Get the sign of the gradients. This tells us the direction to move in.
    signed_grad = tf.sign(gradient)
    return signed_grad

# --- Step 3: Perform the Attack on a Sample ---
print("\n[*] Selecting a sample to attack...")

# Find the first attack sample in the test set that our model currently gets right.
for i in range(len(X_test)):
    sample = X_test[i]
    label = y_test[i]
    
    # We only want to attack an 'attack' sample.
    if label == 1:
        sample_3d = np.reshape(sample, (1, sample.shape[0], sample.shape[1]))
        original_pred_proba = model.predict(sample_3d, verbose=0)[0][0]
        
        # Check if the model correctly identifies it as an attack.
        if original_pred_proba > 0.5:
            print(f"[+] Found a suitable attack sample at index {i}.")
            print(f"    Original model prediction: {original_pred_proba:.2%} (Correctly identified as Attack)")
            
            # Convert to TensorFlow tensors for the attack.
            original_sample_tf = tf.convert_to_tensor(sample_3d)
            original_label_tf = tf.convert_to_tensor([[float(label)]])
            
            # --- Generate the disguise ---
            perturbations = create_adversarial_pattern(original_sample_tf, original_label_tf)
            
            # --- Create the adversarial sample ---
            # Epsilon controls how "strong" the disguise is. A tiny epsilon is often enough.
            epsilon = 0.02 
            adversarial_sample = original_sample_tf + epsilon * perturbations
            # We clip the values to ensure they stay in the valid [0, 1] range.
            adversarial_sample = tf.clip_by_value(adversarial_sample, 0, 1)
            
            # --- Test the disguise ---
            adversarial_pred_proba = model.predict(adversarial_sample, verbose=0)[0][0]
            print(f"    Prediction on disguised sample: {adversarial_pred_proba:.2%} (Attack successful if < 50%)")

            if adversarial_pred_proba < 0.5:
                print("\n[SUCCESS] The adversarial attack worked! The model was fooled.")
            else:
                print("\n[FAILURE] The model resisted the attack. It may be very robust!")

            break # We only need to demonstrate on one sample.

# --- Step 4: Generate and Save a Batch of Adversarial Samples for Retraining ---
print("\n[*] Now generating a larger batch of adversarial samples for our defense phase...")
adversarial_samples = []
adversarial_labels = []
count = 0
# We'll generate 5,000 adversarial samples to add to our training data.
num_to_generate = 5000

for i in range(len(X_test)):
    if y_test[i] == 1: # Only use attack samples
        sample = X_test[i]
        sample_3d = np.reshape(sample, (1, sample.shape[0], sample.shape[1]))
        sample_tf = tf.convert_to_tensor(sample_3d)
        label_tf = tf.convert_to_tensor([[1.0]])

        perturbations = create_adversarial_pattern(sample_tf, label_tf)
        # Using a slightly different epsilon for batch generation can sometimes be effective.
        adv_sample = tf.clip_by_value(sample_tf + 0.025 * perturbations, 0, 1)
        
        # We only add it to our list if it successfully fools the model.
        if model.predict(adv_sample, verbose=0)[0][0] < 0.5:
            adversarial_samples.append(adv_sample.numpy().reshape(sample.shape))
            adversarial_labels.append(1) # It's still an attack, just disguised.
            count += 1
            if count % 100 == 0:
                print(f"    ... {count}/{num_to_generate} adversarial samples generated.")
            if count >= num_to_generate:
                break

if count > 0:
    print(f"\n[+] Successfully generated {count} adversarial samples.")
    OUTPUT_PATH = "Adversarial_Samples/"
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    np.save(os.path.join(OUTPUT_PATH, "X_adversarial.npy"), np.array(adversarial_samples))
    np.save(os.path.join(OUTPUT_PATH, "y_adversarial.npy"), np.array(adversarial_labels))
    print(f"[*] Saved adversarial samples to the '{OUTPUT_PATH}' directory.")
else:
    print("[!] Could not generate any successful adversarial samples. The model may be too robust for this epsilon.")
