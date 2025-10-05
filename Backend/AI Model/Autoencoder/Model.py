import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

#  Load preprocessed data
X_train = np.load("../Preprocessing/KDD/X_train.npy")
y_train = np.load("../Preprocessing/KDD/y_train.npy")
X_val   = np.load("../Preprocessing/KDD/X_val.npy")
y_val   = np.load("../Preprocessing/KDD/y_val.npy")
X_test  = np.load("../Preprocessing/KDD/X_test.npy")
y_test  = np.load("../Preprocessing/KDD/y_test.npy")

#  Use only NORMAL data for training
X_train_normal = X_train[y_train == 0]
X_val_normal   = X_val[y_val == 0]

#  Build Autoencoder
input_dim = X_train.shape[1]
encoding_dim = 32

input_layer = keras.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation="relu")(input_layer)
decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)

autoencoder = keras.Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

#  Train on normal data
autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(X_val_normal, X_val_normal)
)

#  Validation losses
reconstructions_val = autoencoder.predict(X_val)
val_loss = np.mean(np.square(X_val - reconstructions_val), axis=1)

#  ROC method: maximize Youden’s J (TPR - FPR)
fpr, tpr, roc_thresholds = roc_curve(y_val, val_loss)
j_scores = tpr - fpr
best_threshold_roc = roc_thresholds[np.argmax(j_scores)]
print("Best ROC threshold:", best_threshold_roc)

#  Precision-Recall method: maximize F1
precisions, recalls, pr_thresholds = precision_recall_curve(y_val, val_loss)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
best_threshold_pr = pr_thresholds[np.argmax(f1_scores)]
print("Best PR threshold:", best_threshold_pr)

# Final threshold: average of both
threshold = (best_threshold_roc + best_threshold_pr) / 2
print("Final chosen threshold:", threshold)

# Test phase
reconstructions_test = autoencoder.predict(X_test)
test_loss = np.mean(np.square(X_test - reconstructions_test), axis=1)
y_pred = (test_loss > threshold).astype(int)

# Evaluation
print("ROC-AUC:", roc_auc_score(y_test, test_loss))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
