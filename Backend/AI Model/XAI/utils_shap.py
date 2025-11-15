import os
import sys
import json
import numpy as np
import shap

# ----- Path setup -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../Backend/AI Model/XAI
AI_MODEL_DIR = os.path.dirname(BASE_DIR)                   # .../Backend/AI Model

# So we can import inference.py even though the folder has a space ("AI Model")
if AI_MODEL_DIR not in sys.path:
    sys.path.insert(0, AI_MODEL_DIR)

# Make sure this XAI folder is also on the path for cross-imports if needed
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

# One explainer per model_key, cached
_EXPLAINERS = {}


def _make_kernel_explainer(model_key: str):
    """
    Create a SHAP KernelExplainer for the given model.
    Assumes the model outputs a single probability for 'Attack'
    or can be reshaped to (batch,).
    """
    model = load_model(model_key)

    def f(X):
        # X: (n_samples, n_features)
        import tensorflow as tf

        X = np.asarray(X, dtype=float)
        if model_key in ["cnn_lstm", "cnn_lstm_hardened"]:
            # reshape to (batch, timesteps, channels) for CNN+LSTM
            X_tf = tf.convert_to_tensor(
                X.reshape(X.shape[0], X.shape[1], 1),
                dtype=tf.float32
            )
        else:
            X_tf = tf.convert_to_tensor(X, dtype=tf.float32)

        preds = model(X_tf, training=False).numpy().reshape(-1)
        return preds

    # Keep SHAP background small to avoid OOM
    bg = BACKGROUND
    if bg.shape[0] > 50:
        bg = bg[:50]

    return shap.KernelExplainer(f, bg)


def get_explainer(model_key: str):
    if model_key not in _EXPLAINERS:
        _EXPLAINERS[model_key] = _make_kernel_explainer(model_key)
    return _EXPLAINERS[model_key]


def explain_with_shap(sample: np.ndarray, model_key: str, top_k: int = 6):
    """
    sample: 1D numpy array matching FEATURE_NAMES order.
    returns: list[{feature, shap_value}] for top_k features by |shap_value|.
    """
    x = np.asarray(sample, dtype=float).reshape(1, -1)

    explainer = get_explainer(model_key)

    # Limit nsamples so SHAP doesn't try huge batches (prevents OOM)
    shap_values = explainer.shap_values(x, nsamples=100)

    # shap_values may be [class0, class1]; assume last index = Attack
    if isinstance(shap_values, list):
        sv = np.array(shap_values[-1][0])
    else:
        sv = np.array(shap_values[0])

    feats = [
        {"feature": name, "shap_value": float(val)}
        for name, val in zip(FEATURE_NAMES, sv)
    ]
    feats_sorted = sorted(feats, key=lambda v: abs(v["shap_value"]), reverse=True)
    return feats_sorted[:top_k]
