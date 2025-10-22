# train_deep_autoencoder.py
# Description: This script trains a deep autoencoder on the CIC-IDS-2017 dataset.
# This was our first and most balanced unsupervised model, intended to act as
# a zero-day attack detector by identifying anomalies in network traffic.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

print("--- Training the Original Deep Autoencoder on the CIC-IDS-2017 Dataset ---")

# --- Step 1: Load Our Clean, Preprocessed Data ---
DATA_PATH = "../Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/"
print(f"\nI'm loading the clean dataset from '{DATA_PATH}'...")
try:
    X_train = np.load(DATA_PATH + 'X_train.npy')
    y_train = np.load(DATA_PATH + 'y_train.npy')
    X_val   = np.load(DATA_PATH + 'X_val.npy')
    y_val   = np.load(DATA_PATH + 'y_val.npy')
    X_test  = np.load(DATA_PATH + 'X_test.npy')
    y_test  = np.load(DATA_PATH + 'y_test.npy')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("\n[!] Error: I couldn't find the processed data files.")
    print("[!] Please make sure you've run the 'preprocess_cicids2017.py' script first.")
    exit()

# --- Step 2: Prepare the Training Data ---
# We only train the autoencoder on 'normal' data to learn its structure.
X_train_normal = X_train[y_train == 0]
X_val_normal   = X_val[y_val == 0]
print(f"I'll be training the model on {len(X_train_normal):,} normal samples.")


# --- Step 3: Build the Deep Autoencoder Model ---
input_dim = X_train.shape[1]
latent_dim = 16 # Our bottleneck layer

input_layer = keras.Input(shape=(input_dim,), name="input_layer")

# Encoder
encoded = layers.Dense(64, activation="relu")(input_layer)
encoded = layers.Dense(32, activation="relu")(encoded)
encoded = layers.Dense(latent_dim, activation="relu", name="latent_space")(encoded) # The compressed representation

# Decoder
decoded = layers.Dense(32, activation="relu")(encoded)
decoded = layers.Dense(64, activation="relu")(decoded)
decoded = layers.Dense(input_dim, activation="sigmoid", name="output_layer")(decoded)

autoencoder = keras.Model(input_layer, decoded, name="Deep_Autoencoder_CICIDS")

# We found Mean Absolute Error to be a good loss function for this task.
autoencoder.compile(optimizer="adam", loss="mae")

print("\nHere's a summary of the model architecture:")
autoencoder.summary()


# --- Step 4: Train the Model ---
print("\nStarting model training... this might take a little while.")

# Early stopping will monitor the validation loss and restore the best model.
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    restore_best_weights=True
)

history = autoencoder.fit(
    X_train_normal, X_train_normal, # It learns to reconstruct itself
    epochs=100,
    batch_size=256,
    validation_data=(X_val_normal, X_val_normal),
    callbacks=[early_stopping],
    verbose=2 # We'll show one line per epoch
)
print("Training is complete!")


# --- Step 5: Save the Trained Model ---
print("\nSaving the trained model to 'cicids_autoencoder.keras'...")
autoencoder.save("cicids_autoencoder.keras")
print("Model saved successfully!")


# --- Step 6: Find the Best Anomaly Threshold ---
print("\nNow, let's find the best threshold for detecting anomalies...")
# We get the model's reconstructions of the validation set
reconstructions_val = autoencoder.predict(X_val)
# The anomaly score is the reconstruction error
val_loss = np.mean(np.abs(X_val - reconstructions_val), axis=1)

# Find the threshold that best separates normal from attack
fpr, tpr, roc_thresholds = roc_curve(y_val, val_loss)
j_scores = tpr - fpr
best_threshold = roc_thresholds[np.argmax(j_scores)]
print(f"The best threshold for flagging an anomaly is a reconstruction error of: {best_threshold:.4f}")


# --- Step 7: Final Evaluation on the Test Set ---
print("\nTime for the final evaluation on the unseen test data...")
reconstructions_test = autoencoder.predict(X_test)
test_loss = np.mean(np.abs(X_test - reconstructions_test), axis=1)
y_pred = (test_loss > best_threshold).astype(int)


# --- Step 8: Show Me the Results! ---
print("\n--- Final Model Performance (Deep Autoencoder) ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, test_loss):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))
print("\nConfusion Matrix:")
print("                 Predicted Normal   Predicted Attack")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"Actual Normal    {tn:>10,} {fp:>18,}")
print(f"Actual Attack    {fn:>10,} {tp:>18,}")
