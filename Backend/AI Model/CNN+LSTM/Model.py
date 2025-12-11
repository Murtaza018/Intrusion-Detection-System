import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils import class_weight

print("Training a Hybrid Spatio-Temporal Model (CNN + LSTM)")

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

# --- Step 2: Reshape Data for the CNN/LSTM Layers ---
# Data must be 3D: (samples, timesteps, features).
if len(X_train.shape) == 2:
    X_train = np.expand_dims(X_train, -1)
    X_val   = np.expand_dims(X_val, -1)
    X_test  = np.expand_dims(X_test, -1)
    print(f"Reshaped data to 3D format: {X_train.shape}")

# --- Step 3: Handle Class Imbalance ---
print("Calculating class weights for imbalanced dataset...")
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print(f"Calculated weights: {class_weights}")


# --- Step 4: Build the Spatio-Temporal Model ---
print("Building the hybrid CNN + LSTM model architecture...")
model = Sequential([
    # CNN part: Learns spatial features from the flow data.
    Conv1D(128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    # LSTM part: Learns sequential patterns from the features.
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),

    # Classifier head.
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
], name="SpatioTemporal_CNN_LSTM")

# Compile the model.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Step 5: Train the Model ---
# Callbacks for robust training.
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)

print("Starting model training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30, 
    batch_size=128,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
print("Training complete.")

# --- Step 6: Save the Final Model ---
print("Saving the model...")
model.save("cicids_spatiotemporal_model.keras")
print("Model saved as 'cicids_spatiotemporal_model.keras'.")

# --- Step 7: Evaluate the Model on the Test Set ---
print("Final evaluation on the unseen test data...")
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype("int32")

# --- Step 8: Show the Results! ---
print("\nFinal Spatio-Temporal Model Performance:")
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