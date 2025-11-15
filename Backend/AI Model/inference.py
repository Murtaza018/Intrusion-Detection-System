import os
import numpy as np
from tensorflow import keras

# ---------------------------------------------------------
# Path setup
# ---------------------------------------------------------
# This file lives in: Backend/AI Model/inference.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Backend/AI Model

# Model files are stored in subfolders of this directory:
#   Backend/AI Model/DNN/
#   Backend/AI Model/CNN+LSTM/
#   Backend/AI Model/Adversarial Attack and Defense/
#   Backend/AI Model/Autoencoder/
#
# Using BASE_DIR makes paths work no matter where you run Python from.
MODEL_PATHS = {
    "dnn": os.path.join(
        BASE_DIR, "DNN", "cicids_supervised_classifier.keras"
    ),
    "cnn_lstm": os.path.join(
        BASE_DIR, "CNN+LSTM", "cicids_spatiotemporal_model.keras"
    ),
    "cnn_lstm_hardened": os.path.join(
        BASE_DIR, "CNN+LSTM",
        "cicids_spatiotemporal_model_hardened.keras"
    ),
    "autoencoder": os.path.join(
        BASE_DIR, "Autoencoder", "cicids_autoencoder.keras"
    ),
}

# Cache loaded models so we don't reload from disk every time
_LOADED_MODELS = {}


def load_model(model_key: str):
    """
    Load (or retrieve cached) Keras model for the given key.
    model_key must be one of MODEL_PATHS keys.
    """
    if model_key not in MODEL_PATHS:
        raise ValueError(f"Unknown model_key: {model_key}")

    if model_key not in _LOADED_MODELS:
        path = MODEL_PATHS[model_key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        _LOADED_MODELS[model_key] = keras.models.load_model(path)
    return _LOADED_MODELS[model_key]

def _prepare_input(sample: np.ndarray, model_key: str) -> np.ndarray:
    """
    Prepare input shape for different model types.
    - DNN / autoencoder: (1, n_features)
    - CNN+LSTM variants: (1, n_features, 1)
    """
    x = np.asarray(sample, dtype=float)

    if model_key in ["cnn_lstm", "cnn_lstm_hardened"]:
        # CNN+LSTM expects (batch, timesteps, channels)
        x = x.reshape(1, -1, 1)
    else:
        x = x.reshape(1, -1)

    return x

def predict(sample: np.ndarray, model_key: str):
    """
    sample: 1D numpy array of features (already preprocessed, correct order).
    returns: dict with label, score, model_key
    """
    model = load_model(model_key)
    x = _prepare_input(sample, model_key)

    proba = float(model.predict(x, verbose=0)[0][0])
    label = "Attack" if proba >= 0.5 else "Normal"

    return {
        "label": label,
        "score": proba,
        "model": model_key,
    }

