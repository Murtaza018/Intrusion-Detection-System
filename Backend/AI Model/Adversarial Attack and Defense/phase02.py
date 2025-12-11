import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle

# Phase 2: Adversarial Training (Defense)
print("Adversarial Training started.")

# --- Step 1: Load the Model, Original Data, and Adversarial Data ---
print("Loading model and all necessary datasets...")
try:
    # Load the model we are going to harden.
    model = keras.models.load_model("../CNN+LSTM/cicids_spatiotemporal_model.keras")

    # Load the original training data.
    DATA_PATH = "../Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/"
    X_train = np.load(DATA_PATH + 'X_train.npy')
    y_train = np.load(DATA_PATH + 'y_train.npy')

    # Load the adversarial samples from Phase 1.
    ADVERSARIAL_PATH = "./Adversarial_Samples/"
    X_adversarial = np.load(ADVERSARIAL_PATH + "X_adversarial.npy")
    y_adversarial = np.load(ADVERSARIAL_PATH + "y_adversarial.npy")
    
    print("All files loaded.")
except Exception as e:
    print(f"Error loading files: {e}")
    print("Ensure all required models and datasets are present.")
    exit()

# --- Step 2: Create the Hardened Training Dataset ---
print("Creating the new, hardened training dataset...")

# Ensure original data is 3D for concatenation.
if len(X_train.shape) == 2:
    X_train = np.expand_dims(X_train, -1)

# Combine original data with adversarial samples.
X_train_hardened = np.concatenate([X_train, X_adversarial])
y_train_hardened = np.concatenate([y_train, y_adversarial])

# Important: shuffle the combined dataset.
X_train_hardened, y_train_hardened = shuffle(X_train_hardened, y_train_hardened, random_state=42)

print(f"New training set created with {len(X_train_hardened)} total samples.")


# --- Step 3: Fine-Tune (Retrain) the Model ---
print("Fine-tuning the model on the hardened dataset...")

# Use a lower learning rate for fine-tuning.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) 
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    X_train_hardened, y_train_hardened,
    epochs=5, # Fine-tune for a few epochs.
    batch_size=128,
    verbose=1
)
print("Fine-tuning complete.")

# --- Step 4: Save the New, Hardened Model ---
print("Saving the final, hardened model...")
model.save("cicids_spatiotemporal_model_hardened.keras")
print("Hardened model saved as 'cicids_spatiotemporal_model_hardened.keras'")


# --- Step 5: Verify the Defense ---
print("Verifying the defense on a known adversarial sample...")

try:
    # Load the original model to create the perturbation.
    original_model = keras.models.load_model("../CNN+LSTM/cicids_spatiotemporal_model.keras")
except Exception:
    original_model = model

def create_adversarial_pattern(input_data, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_data)
        # Use the ORIGINAL model to find the weakness.
        prediction = original_model(input_data)
        loss = keras.losses.BinaryCrossentropy()(input_label, prediction)
    return tf.sign(tape.gradient(loss, input_data))

# Load test data to find a sample.
X_test = np.load(DATA_PATH + 'X_test.npy')
y_test = np.load(DATA_PATH + 'y_test.npy')
if len(X_test.shape) == 2:
    X_test = np.expand_dims(X_test, -1)

# Find an attack sample and test the hardened model.
for i in range(len(X_test)):
    if y_test[i] == 1:
        sample = X_test[i]
        sample_3d = np.reshape(sample, (1, sample.shape[0], sample.shape[1]))
        
        # Generate the adversarial sample using the original model's weakness.
        original_sample_tf = tf.convert_to_tensor(sample_3d)
        original_label_tf = tf.convert_to_tensor([[1.0]])
        perturbations = create_adversarial_pattern(original_sample_tf, original_label_tf)
        adversarial_sample = tf.clip_by_value(original_sample_tf + 0.02 * perturbations, 0, 1)

        original_pred_proba = original_model.predict(adversarial_sample, verbose=0)[0][0]
        if original_pred_proba < 0.5:
            print(f"Found adversarial sample that fools original model (Prediction: {original_pred_proba:.2%}).")
            
            # Test the HARDENED model.
            hardened_pred_proba = model.predict(adversarial_sample, verbose=0)[0][0]
            print(f"New hardened model prediction: {hardened_pred_proba:.2%}")

            if hardened_pred_proba > 0.5:
                print("Defense worked! Hardened model correctly identified the disguised attack.")
            else:
                print("Hardened model was still fooled.")
            
            break