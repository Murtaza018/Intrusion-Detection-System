import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# Phase 1: Adversarial Attack Simulation
print("Adversarial Attack Simulation started.")

# --- Step 1: Load the Model and Data ---
print("Loading model and test dataset...")
try:
    # Load the model we are going to attack.
    model = keras.models.load_model("../CNN+LSTM/cicids_spatiotemporal_model.keras")
    
    # Load the test data to find a sample to attack.
    DATA_PATH = "../Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/"
    X_test = np.load(DATA_PATH + 'X_test.npy')
    y_test = np.load(DATA_PATH + 'y_test.npy')
    
    print("Model and data loaded.")
except Exception as e:
    print(f"Error loading files: {e}")
    print("Ensure files are present.")
    exit()

# Reshape data if necessary for the 3D input expected by the model.
if len(X_test.shape) == 2:
    X_test = np.expand_dims(X_test, -1)

# --- Step 2: Define the Adversarial Attack Function (FGSM) ---
def create_adversarial_pattern(input_data, input_label):
    # Generates a perturbation using FGSM.
    with tf.GradientTape() as tape:
        tape.watch(input_data)
        prediction = model(input_data)
        loss_fn = keras.losses.BinaryCrossentropy()
        loss = loss_fn(input_label, prediction)

    # Get the sign of the gradients.
    gradient = tape.gradient(loss, input_data)
    signed_grad = tf.sign(gradient)
    return signed_grad

# --- Step 3: Perform the Attack on a Sample ---
print("Selecting a sample to attack...")

# Find the first attack sample that the model correctly classifies.
for i in range(len(X_test)):
    sample = X_test[i]
    label = y_test[i]
    
    if label == 1:
        sample_3d = np.reshape(sample, (1, sample.shape[0], sample.shape[1]))
        original_pred_proba = model.predict(sample_3d, verbose=0)[0][0]
        
        # Check if model correctly identifies it as an attack.
        if original_pred_proba > 0.5:
            print(f"Found suitable attack sample at index {i}.")
            print(f"Original prediction: {original_pred_proba:.2%}")
            
            # Convert to TensorFlow tensors.
            original_sample_tf = tf.convert_to_tensor(sample_3d)
            original_label_tf = tf.convert_to_tensor([[float(label)]])
            
            perturbations = create_adversarial_pattern(original_sample_tf, original_label_tf)
            
            # Create the adversarial sample.
            epsilon = 0.02 
            adversarial_sample = original_sample_tf + epsilon * perturbations
            adversarial_sample = tf.clip_by_value(adversarial_sample, 0, 1)
            
            # Test the adversarial sample.
            adversarial_pred_proba = model.predict(adversarial_sample, verbose=0)[0][0]
            print(f"Disguised sample prediction: {adversarial_pred_proba:.2%}")

            if adversarial_pred_proba < 0.5:
                print("Adversarial attack succeeded. Model was fooled.")
            else:
                print("Model resisted the attack.")

            break 

# --- Step 4: Generate and Save a Batch of Adversarial Samples for Retraining ---
print("Generating a batch of adversarial samples for defense...")
adversarial_samples = []
adversarial_labels = []
count = 0
num_to_generate = 5000

for i in range(len(X_test)):
    if y_test[i] == 1:
        sample = X_test[i]
        sample_3d = np.reshape(sample, (1, sample.shape[0], sample.shape[1]))
        sample_tf = tf.convert_to_tensor(sample_3d)
        label_tf = tf.convert_to_tensor([[1.0]])

        perturbations = create_adversarial_pattern(sample_tf, label_tf)
        adv_sample = tf.clip_by_value(sample_tf + 0.025 * perturbations, 0, 1)
        
        # Only keep samples that successfully fool the model.
        if model.predict(adv_sample, verbose=0)[0][0] < 0.5:
            adversarial_samples.append(adv_sample.numpy().reshape(sample.shape))
            adversarial_labels.append(1)
            count += 1
            if count % 100 == 0:
                print(f"... {count}/{num_to_generate} samples generated.")
            if count >= num_to_generate:
                break

if count > 0:
    print(f"Successfully generated {count} adversarial samples.")
    OUTPUT_PATH = "Adversarial_Samples/"
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    np.save(os.path.join(OUTPUT_PATH, "X_adversarial.npy"), np.array(adversarial_samples))
    np.save(os.path.join(OUTPUT_PATH, "y_adversarial.npy"), np.array(adversarial_labels))
    print(f"Saved adversarial samples to '{OUTPUT_PATH}' directory.")
else:
    print("Could not generate any successful adversarial samples.")