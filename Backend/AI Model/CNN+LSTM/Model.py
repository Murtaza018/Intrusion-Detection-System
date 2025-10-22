# train_spatiotemporal_model.py
# Description: This script trains a hybrid, spatio-temporal deep learning model (CNN + LSTM)
# on the CIC-IDS-2017 dataset. This is a highly advanced, supervised classifier designed
# to learn both spatial patterns within a single network flow and the sequential nature
# of those patterns. This is our most powerful model for detecting known attacks.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils import class_weight

print("--- Training a Hybrid Spatio-Temporal Model (CNN + LSTM) ---")

# --- Step 1: Load Our Clean, Preprocessed Data ---
DATA_PATH = "../Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/"
print(f"\nI'm loading our best dataset from '{DATA_PATH}'...")
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

# --- Step 2: Reshape Data for the CNN/LSTM Layers ---
# The Conv1D and LSTM layers expect a 3D input shape: (samples, timesteps, features).
# We will treat the features of each flow as a sequence.
if len(X_train.shape) == 2:
    X_train = np.expand_dims(X_train, -1)
    X_val   = np.expand_dims(X_val, -1)
    X_test  = np.expand_dims(X_test, -1)
    print(f"I've reshaped the data to the required 3D format: {X_train.shape}")

# --- Step 3: Handle Class Imbalance ---
# The dataset has many more normal samples than attack samples.
# We'll calculate class weights to make sure the model pays equal attention to both.
print("\nThis dataset is imbalanced, so I'm calculating class weights to help the model learn better...")
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print(f"Calculated weights: {class_weights}")


# --- Step 4: Build the Spatio-Temporal Model ---
print("\nBuilding the hybrid CNN + LSTM model architecture...")
model = Sequential([
    # The CNN part learns spatial features from the flow data.
    Conv1D(128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    # The LSTM part learns the sequential patterns from the features found by the CNN.
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),

    # A standard classifier head.
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
], name="SpatioTemporal_CNN_LSTM")

# Compile the model with a fine-tuned optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Step 5: Train the Model ---
# We'll use callbacks to stop training early if it's not improving and to reduce the learning rate.
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)

print("\nStarting the final model training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30, # Max epochs, but early stopping will likely finish sooner.
    batch_size=128,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
print("Training is complete!")

# --- Step 6: Save the Final Model ---
print("\nSaving our new, powerful model...")
model.save("cicids_spatiotemporal_model.keras")
print("Model saved successfully!")

# --- Step 7: Evaluate the Model on the Test Set ---
print("\nTime for the final evaluation on the unseen test data...")
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype("int32")

# --- Step 8: Show Me the Results! ---
print("\n--- Final Spatio-Temporal Model Performance ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))
print("\nConfusion Matrix:")
print("                 Predicted Normal   Predicted Attack")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"Actual Normal    {tn:>10,} {fp:>18,}")
print(f"Actual Attack    {fn:>10,} {tp:>18,}")
