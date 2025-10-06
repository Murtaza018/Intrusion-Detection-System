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

# Reshape to 3D (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val   = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

num_classes = len(np.unique(y_train))

# Class weights (handle imbalance)
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Model definition
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    layers.Conv1D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),
    layers.LSTM(64),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Training
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val),
    class_weight=class_weights
)

# Evaluation
y_pred = np.argmax(model.predict(X_test), axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
