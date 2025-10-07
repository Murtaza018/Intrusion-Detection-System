import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load preprocessed data
X_train = np.load("../Preprocessing/KDD/X_train.npy")
y_train = np.load("../Preprocessing/KDD/y_train.npy")
X_val   = np.load("../Preprocessing/KDD/X_val.npy")
y_val   = np.load("../Preprocessing/KDD/y_val.npy")
X_test  = np.load("../Preprocessing/KDD/X_test.npy")
y_test  = np.load("../Preprocessing/KDD/y_test.npy")

num_classes = len(np.unique(y_train))

# Compute class weights for imbalance
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Define CNN + LSTM model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    layers.Conv1D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling1D(2),
    layers.Dropout(0.5),
    layers.LSTM(32, dropout=0.3, recurrent_dropout=0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Early stopping to prevent overfitting
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate on test data
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
