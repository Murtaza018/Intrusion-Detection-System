import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)

print("Training a Supervised DNN Classifier on the CIC-IDS-2017 Dataset.")

# --- Step 1: Load Our Clean, Preprocessed Data ---
DATA_PATH = "../Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/"
print(f"Loading dataset from '{DATA_PATH}'...")
try:
    X_train = np.load(DATA_PATH + 'X_train.npy')
    y_train = np.load(DATA_PATH + 'y_train.npy')
    X_val   = np.load(DATA_PATH + 'X_val.npy')
    y_val   = np.load(DATA_PATH + 'y_val.npy')
    X_test  = np.load(DATA_PATH + 'X_test.npy')
    y_test  = np.load(DATA_PATH + 'y_test.npy')
    print("Dataset loaded.")
except FileNotFoundError:
    print("Error: Could not find the processed data files.")
    print("Please run the preprocessing script first.")
    exit()

print(f"Training this classifier on all {len(X_train):,} samples.")

# --- Step 2: Build the Supervised DNN Model ---
input_dim = X_train.shape[1]

model = keras.Sequential(
    [
        keras.Input(shape=(input_dim,), name="input_layer"),
        layers.Dense(128, activation="relu", name="dense_1"),
        layers.Dropout(0.2, name="dropout_1"), # Dropout prevents overfitting
        layers.Dense(64, activation="relu", name="dense_2"),
        layers.Dropout(0.2, name="dropout_2"),
        layers.Dense(32, activation="relu", name="dense_3"),
        # Final layer uses 'sigmoid' for binary classification (0 or 1).
        layers.Dense(1, activation="sigmoid", name="output_layer"),
    ],
    name="Supervised_DNN_Classifier"
)

# Use 'binary_crossentropy' for binary classification.
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Summary of the Supervised Classifier:")
model.summary()


# --- Step 3: Train the Supervised Model ---
print("Starting model training...")

# Early stopping monitors validation accuracy.
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    mode='max',
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=256,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)
print("Training complete.")


# --- Step 4: Save the Trained Model ---
print("Saving the trained classifier...")
model.save("cicids_supervised_classifier.keras")
print("Model saved successfully.")


# --- Step 5: Final Evaluation on the Test Set ---
print("Evaluating on the unseen test data...")

y_pred_proba = model.predict(X_test).ravel()
# Set the classification threshold at 0.5.
y_pred = (y_pred_proba > 0.5).astype(int)


# --- Step 6: Show the Results! ---
print("\nFinal Supervised DNN Classifier Performance:")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))
print("\nConfusion Matrix:")
# Print confusion matrix in a readable format.
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
matrix_display = f"""
                 Predicted Normal   Predicted Attack
Actual Normal      {tn:>10,}         {fp:>18,}
Actual Attack      {fn:>10,}         {tp:>18,}
"""
print(matrix_display)