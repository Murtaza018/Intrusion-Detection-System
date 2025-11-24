import os
import sys
import json
import numpy as np
import shap
import tensorflow as tf

# ----- Path setup -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../Backend/AI Model/XAI
AI_MODEL_DIR = os.path.dirname(BASE_DIR)                   # .../Backend/AI Model

# So we can import inference.py even though the folder has a space ("AI Model")
if AI_MODEL_DIR not in sys.path:
    sys.path.insert(0, AI_MODEL_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference import load_model   # your existing model loading function

FEATURE_SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")
BACKGROUND_PATH = os.path.join(BASE_DIR, "background.npy")

# Load feature names (order must match how you built X_train/X_test)
with open(FEATURE_SCHEMA_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

if not os.path.exists(BACKGROUND_PATH):
    raise FileNotFoundError(
        f"background.npy not found at {BACKGROUND_PATH}. Run build_background.py first."
    )

BACKGROUND = np.load(BACKGROUND_PATH)

# One explainer per model
_EXPLAINERS = {}

def _make_kernel_explainer(model_key):
    model = load_model(model_key)

    # Prediction function wrapper for SHAP
    def f(X):
        # Check if model expects 3D input (CNN+LSTM) or 2D (Autoencoder/DNN)
        if model_key in ["hardened_classifier", "cnn_lstm", "cnn_lstm_hardened"]:
            # Reshape to (batch, features, 1)
            X_tf = tf.convert_to_tensor(
                X.reshape(X.shape[0], X.shape[1], 1),
                dtype=tf.float32
            )
        else:
            # Autoencoder/DNN expects 2D
            X_tf = tf.convert_to_tensor(X, dtype=tf.float32)

        # Get predictions
        preds = model(X_tf, training=False).numpy()
        
        # If it's the classifier, we want the probability (index 0 if output is shape (N,1))
        if preds.shape[1] == 1:
            return preds.reshape(-1)
        
        # If it's the autoencoder, SHAP usually explains reconstruction error.
        # This is complex for KernelExplainer. For simplicity, we might just explain
        # the reconstruction of the *features themselves*, but standard SHAP
        # on autoencoders for anomaly detection often requires a custom wrapper
        # that calculates Mean Absolute Error.
        # For now, let's assume this is mostly for the Classifier.
        return preds

    # Keep SHAP background small to avoid OOM (Out of Memory)
    bg = BACKGROUND
    if bg.shape[0] > 50:
        bg = bg[:50]

    return shap.KernelExplainer(f, bg)


def get_explainer(model_key: str):
    if model_key not in _EXPLAINERS:
        print(f"[*] Initializing SHAP explainer for {model_key}...")
        _EXPLAINERS[model_key] = _make_kernel_explainer(model_key)
    return _EXPLAINERS[model_key]


def explain_with_shap(sample: np.ndarray, model_key: str, top_k: int = 6):
    """
    sample: 1D numpy array matching FEATURE_NAMES order.
    returns: list[{feature, shap_value}] for top_k features by |shap_value|.
    """
    # SHAP expects 2D input (1 sample)
    x = np.asarray(sample, dtype=float).reshape(1, -1)

    explainer = get_explainer(model_key)

    # Limit nsamples so SHAP doesn't try huge batches (prevents OOM)
    # nsamples='auto' is usually fine, but a fixed number is safer for speed.
    shap_values = explainer.shap_values(x)

    # KernelExplainer returns a list for each output. 
    # Our classifier has 1 output, so we take index 0.
    if isinstance(shap_values, list):
        arr = shap_values[0]
    else:
        arr = shap_values

    # arr may be (1, F) or (F,)
    if arr.ndim == 2:
        vals = arr[0]        # take first sample
    else:
        vals = arr


        # Create list of dicts: {"feature": name, "shap_value": val}
    explanations = []
    for i, val in enumerate(vals):
        feat_name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"Feature {i}"

        # FIX: If autoencoder SHAP returns vectors, reduce to scalar
        if isinstance(val, (list, np.ndarray)):
            scalar_val = float(np.mean(np.abs(val)))
        else:
            scalar_val = float(val)

        explanations.append({"feature": feat_name, "shap_value": scalar_val})

    # Sort by absolute impact (magnitude)
    explanations.sort(key=lambda k: abs(k["shap_value"]), reverse=True)

    return explanations[:top_k]
