# train_supervised_classifier.py
# Description: This script trains a supervised Deep Neural Network (DNN) to act as a
# highly accurate classifier for known attack types. Unlike an autoencoder, this
# model is trained on both normal and attack data, allowing it to learn the specific
# patterns that distinguish them. This is expected to yield much higher precision and
# a better balance between catching attacks and avoiding false alarms.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)

print("--- Training a Supervised DNN Classifier on the CIC-IDS-2017 Dataset ---")

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

print(f"I'll be training this classifier on all {len(X_train):,} samples (both normal and attack).")

# --- Step 2: Build the Supervised DNN Model ---
# This is a standard classification architecture. It's deep enough to learn
# complex patterns but simple enough to train efficiently.
input_dim = X_train.shape[1]

model = keras.Sequential(
    [
        keras.Input(shape=(input_dim,), name="input_layer"),
        layers.Dense(128, activation="relu", name="dense_1"),
        layers.Dropout(0.2, name="dropout_1"), # Dropout helps prevent overfitting
        layers.Dense(64, activation="relu", name="dense_2"),
        layers.Dropout(0.2, name="dropout_2"),
        layers.Dense(32, activation="relu", name="dense_3"),
        # The final layer uses a 'sigmoid' activation because we have two classes (0 or 1).
        layers.Dense(1, activation="sigmoid", name="output_layer"),
    ],
    name="Supervised_DNN_Classifier"
)

# We use 'binary_crossentropy' for the loss because this is a binary (0/1) problem.
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nHere's a summary of our new Supervised Classifier:")
model.summary()


# --- Step 3: Train the Supervised Model ---
print("\nStarting model training...")

# Early stopping will monitor the accuracy on our validation set.
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    mode='max', # We want to maximize accuracy
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=256,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)
print("Training is complete!")


# --- Step 4: Save the Trained Model ---
print("\nSaving the trained classifier...")
model.save("cicids_supervised_classifier.keras")
print("Model saved successfully!")


# --- Step 5: Final Evaluation on the Test Set ---
# Now we'll see how well our specialist performs on data it has never seen.
print("\nEvaluating on the unseen test data...")

# For a classifier, we get probabilities. We'll set the threshold at 0.5.
y_pred_proba = model.predict(X_test).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)


# --- Step 6: Show Me the Results! ---
print("\n--- Final Supervised DNN Classifier Performance ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))
print("\nConfusion Matrix:")
print("                 Predicted Normal   Predicted Attack")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"Actual Normal    {tn:>10,} {fp:>18,}")
print(f"Actual Attack    {fn:>10,} {tp:>18,}")

