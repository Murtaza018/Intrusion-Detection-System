import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    roc_curve
)

print("Training the Deep Autoencoder on the CIC-IDS-2017 Dataset.")

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

# --- Step 2: Prepare the Training Data ---
# Train only on 'normal' data to learn its structure.
X_train_normal = X_train[y_train == 0]
X_val_normal   = X_val[y_val == 0]
print(f"Training on {len(X_train_normal):,} normal samples.")


# --- Step 3: Build the Deep Autoencoder Model ---
input_dim = X_train.shape[1]
latent_dim = 16 

input_layer = keras.Input(shape=(input_dim,), name="input_layer")

# Encoder
encoded = layers.Dense(64, activation="relu")(input_layer)
encoded = layers.Dense(32, activation="relu")(encoded)
encoded = layers.Dense(latent_dim, activation="relu", name="latent_space")(encoded) # Compressed representation

# Decoder
decoded = layers.Dense(32, activation="relu")(encoded)
decoded = layers.Dense(64, activation="relu")(decoded)
decoded = layers.Dense(input_dim, activation="sigmoid", name="output_layer")(decoded)

autoencoder = keras.Model(input_layer, decoded, name="Deep_Autoencoder_CICIDS")

# Use Mean Absolute Error (MAE) as the loss function.
autoencoder.compile(optimizer="adam", loss="mae")

print("Model architecture summary:")
autoencoder.summary()


# --- Step 4: Train the Model ---
print("Starting model training...")

# Early stopping monitors validation loss and restores the best model.
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    restore_best_weights=True
)

history = autoencoder.fit(
    X_train_normal, X_train_normal, # Model learns to reconstruct itself
    epochs=100,
    batch_size=256,
    validation_data=(X_val_normal, X_val_normal),
    callbacks=[early_stopping],
    verbose=2
)
print("Training complete.")


# --- Step 5: Save the Trained Model ---
print("Saving the trained model...")
autoencoder.save("cicids_autoencoder.keras")
print("Model saved as 'cicids_autoencoder.keras'.")


# --- Step 6: Find the Best Anomaly Threshold ---
print("Finding the best anomaly threshold using validation data...")
# Anomaly score is the reconstruction error.
reconstructions_val = autoencoder.predict(X_val)
val_loss = np.mean(np.abs(X_val - reconstructions_val), axis=1)

# Use Youden's J statistic (TPR - FPR) to find the optimal ROC threshold.
fpr, tpr, roc_thresholds = roc_curve(y_val, val_loss)
j_scores = tpr - fpr
best_threshold = roc_thresholds[np.argmax(j_scores)]
print(f"Best anomaly threshold: {best_threshold:.4f}")


# --- Step 7: Final Evaluation on the Test Set ---
print("Final evaluation on the test data...")
reconstructions_test = autoencoder.predict(X_test)
test_loss = np.mean(np.abs(X_test - reconstructions_test), axis=1)
y_pred = (test_loss > best_threshold).astype(int)


# --- Step 8: Show the Results! ---
print("\nFinal Model Performance (Deep Autoencoder):")
print(f"ROC-AUC Score: {roc_auc_score(y_test, test_loss):.4f}")
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