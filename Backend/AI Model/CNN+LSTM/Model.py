import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Load preprocessed data
X_train = np.load("../Preprocessing/KDD/X_train.npy")
y_train = np.load("../Preprocessing/KDD/y_train.npy")
X_val   = np.load("../Preprocessing/KDD/X_val.npy")
y_val   = np.load("../Preprocessing/KDD/y_val.npy")
X_test  = np.load("../Preprocessing/KDD/X_test.npy")
y_test  = np.load("../Preprocessing/KDD/y_test.npy")

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Reshape if needed
if len(X_train.shape) == 2:
    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    print(f"Expanded input shape to: {X_train.shape}")

num_classes = len(np.unique(y_train))

# Handle imbalance
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    layers.Conv1D(64, 3, activation="relu", padding="same"),
    layers.Dropout(0.3),
    layers.Conv1D(128, 3, activation="relu", padding="same"),
    layers.Dropout(0.3),
    layers.GRU(64, return_sequences=False),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Accuracy: {acc:.4f}")

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Reports
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
