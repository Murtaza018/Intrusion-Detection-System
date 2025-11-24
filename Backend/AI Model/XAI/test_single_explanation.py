import os
import sys
import numpy as np

# ---------------------------------------------------
# FIX PYTHON PATHS (IMPORTANT)
# ---------------------------------------------------
# Get the directory where this script is located (Backend/AI Model/XAI)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (Backend/AI Model)
AI_MODEL_DIR = os.path.dirname(CURRENT_DIR)

# Add these paths to sys.path so Python can find the modules
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if AI_MODEL_DIR not in sys.path:
    sys.path.insert(0, AI_MODEL_DIR)

# Now imports will work
# We import from the filename (explanation_inference) directly
try:
    from inference import predict
    from explanation_inference import explain_alert
except ImportError as e:
    print(f"[!] Import Error: {e}")
    print(f"    Current Dir: {CURRENT_DIR}")
    print(f"    AI Model Dir: {AI_MODEL_DIR}")
    sys.exit(1)

# ---------------------------------------------------
# Load test data
# ---------------------------------------------------
DATA_PATH = os.path.join(
    AI_MODEL_DIR,
    "Preprocessing", "CIC-IDS-2017-Processed" # UPDATED PATH based on your logs
)

# Fallback if folder name is different
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(
        AI_MODEL_DIR,
        "Preprocessing", "CIC-IDS-2017", "CIC-IDS-2017-Processed"
    )

try:
    X = np.load(os.path.join(DATA_PATH, "X_test.npy"))
    y = np.load(os.path.join(DATA_PATH, "y_test.npy"))
except FileNotFoundError:
    print(f"[!] Error: Could not find test data at '{DATA_PATH}'")
    sys.exit(1)

# pick an ATTACK sample
attack_indices = np.where(y == 1)[0]
if len(attack_indices) == 0:
    print("[!] No attack samples found in test data.")
    sys.exit(1)

sample_index = int(attack_indices[0])
sample = X[sample_index]
true_label = y[sample_index]

print("Sample index:", sample_index)
print("True label:", true_label)


# ---------------------------------------------------
# Predict using your real model
# ---------------------------------------------------
try:
    # ** THE FIX: Use the correct key 'hardened_classifier' **
    pred = predict(sample, "hardened_classifier")
    print("Predicted:", pred)
except ValueError as e:
    print(f"[!] Prediction Error: {e}")
    sys.exit(1)

# ---------------------------------------------------
# XAI Explanation
# ---------------------------------------------------
if pred["label"] == "Attack":
    print("\n--- Generating Explanation ---")
    explanation = explain_alert(sample, "hardened_classifier", attack_type="Attack")
    
    print("\n[FACTS]")
    print(explanation.get("facts", "No facts returned"))
    
    print("\n[EXPLANATION TEXT]")
    print(explanation.get("explanation", "No explanation text returned"))
else:
    print("Prediction was Normal, so skipping explanation.")