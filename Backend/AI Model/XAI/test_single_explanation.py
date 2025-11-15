import os
import sys
import numpy as np

# ---------------------------------------------------
# FIX PYTHON PATHS (IMPORTANT)
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # XAI folder
AI_MODEL_DIR = os.path.dirname(BASE_DIR)                       # Backend/AI Model

# Add both paths to sys.path so imports work
if AI_MODEL_DIR not in sys.path:
    sys.path.insert(0, AI_MODEL_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Now imports will work
from inference import predict
from explanation_inference import explain_alert

# ---------------------------------------------------
# Load test data
# ---------------------------------------------------
DATA_PATH = os.path.join(
    AI_MODEL_DIR,
    "Preprocessing", "CIC-IDS-2017", "CIC-IDS-2017-Processed"
)

X = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y = np.load(os.path.join(DATA_PATH, "y_test.npy"))

# pick an ATTACK sample
attack_indices = np.where(y == 1)[0]
sample_index = int(attack_indices[0])  # use the first attack; you can change later

sample = X[sample_index]
true_label = y[sample_index]

print("Sample index:", sample_index)
print("True label:", true_label)


# ---------------------------------------------------
# Predict using your real model
# ---------------------------------------------------
pred = predict(sample, "cnn_lstm_hardened")
print("Predicted:", pred)

# ---------------------------------------------------
# XAI Explanation
# ---------------------------------------------------
facts = explain_alert(sample, "cnn_lstm_hardened", attack_type=pred["label"])

print("\n==================== EXPLANATION ====================\n")
print(facts["explanation"])

print("\nTop Concepts:", facts["top_concepts"])
print("Top Features:", facts["top_features"])
