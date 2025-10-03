import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 1. Load preprocessed data
X_train = np.load("../Preprocessing/")
y_train = np.load("y_train.npy")
X_val   = np.load("X_val.npy")
y_val   = np.load("y_val.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

# 2. Use only NORMAL data for training
X_train_normal = X_train[y_train == 0]
X_val_normal   = X_val[y_val == 0]

# 3. Build Autoencoder
input_dim = X_train.shape[1]
encoding_dim = 32

input_layer = keras.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation="relu")(input_layer)
decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)

autoencoder = keras.Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# 4. Train on normal data
autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(X_val_normal, X_val_normal)
)

# 5. Compute reconstruction error threshold
reconstructions = autoencoder.predict(X_train_normal)
train_loss = np.mean(np.square(X_train_normal - reconstructions), axis=1)
threshold = np.mean(train_loss) + 3*np.std(train_loss)

print("Threshold for anomaly detection:", threshold)

# 6. Test phase
reconstructions_test = autoencoder.predict(X_test)
test_loss = np.mean(np.square(X_test - reconstructions_test), axis=1)
y_pred = (test_loss > threshold).astype(int)

# 7. Evaluation
print("ROC-AUC:", roc_auc_score(y_test, test_loss))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
