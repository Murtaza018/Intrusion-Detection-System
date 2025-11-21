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
# ** THE FIX: Use the correct key 'hardened_classifier' **
pred = predict(sample, "hardened_classifier")
print("Predicted:", pred)

# ---------------------------------------------------
# XAI Explanation
# ---------------------------------------------------
# Only explain if it was detected as an attack (which we expect)
if pred["label"] == "Attack":
    print("\n--- Generating Explanation ---")
    # We pass the same model key here
    explanation = explain_alert(sample, "hardened_classifier", attack_type="Attack")
    
    print("\n[FACTS]")
    print(explanation["facts"])
    
    print("\n[EXPLANATION TEXT]")
    print(explanation["explanation"])
else:
    print("Prediction was Normal, so skipping explanation.")