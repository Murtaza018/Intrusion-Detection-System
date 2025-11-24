import os
import numpy as np
from tensorflow import keras

# ---------------------------------------------------------
# Path setup
# ---------------------------------------------------------
# Assumes this file is in: Backend/AI Model/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# We only keep the two essential models for the final pipeline.
MODEL_PATHS = {
    # 1. The Frontline Defense (Hardened Spatio-Temporal Model)
    # Note: Ensure the .keras file is moved to this folder or update the path below.
    "hardened_classifier": os.path.join(
        BASE_DIR, "Adversarial Attack and Defense", "cicids_spatiotemporal_model_hardened.keras"
    ),
    
    # 2. The Zero-Day Hunter (Autoencoder)
    "zero_day_hunter": os.path.join(
        BASE_DIR, "Autoencoder", "cicids_autoencoder.keras"
    ),
}

# Threshold for the Autoencoder (from your training results)
AUTOENCODER_THRESHOLD = 0.01

# Cache loaded models
_LOADED_MODELS = {}

def load_model(model_key: str):
    """
    Load (or retrieve cached) Keras model for the given key.
    """
    if model_key not in MODEL_PATHS:
        raise ValueError(f"Unknown model_key: {model_key}. Available: {list(MODEL_PATHS.keys())}")

    if model_key not in _LOADED_MODELS:
        path = MODEL_PATHS[model_key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        
        print(f"[*] Loading model '{model_key}' from disk...")
        # compile=False makes loading faster since we only need inference, not training
        _LOADED_MODELS[model_key] = keras.models.load_model(path, compile=False)
        
    return _LOADED_MODELS[model_key]

def _prepare_input(sample: np.ndarray, model_key: str) -> np.ndarray:
    """
    Prepare input shape based on specific model requirements.
    """
    x = np.asarray(sample, dtype=float)

    # The Hardened CNN+LSTM expects 3D input: (batch, features, 1)
    if model_key == "hardened_classifier":
        x = x.reshape(1, -1, 1)
    
    # The Autoencoder expects 2D input: (batch, features)
    elif model_key == "zero_day_hunter":
        x = x.reshape(1, -1)

    return x

def predict(sample: np.ndarray, model_key: str):
    """
    Run inference on a single sample.
    Handles logic differences between Classifier (Probability) and Autoencoder (Reconstruction Error).
    """
    model = load_model(model_key)
    x = _prepare_input(sample, model_key)

    # --- Logic for Frontline Defense (Classifier) ---
    if model_key == "hardened_classifier":
        # Output is a probability (0 to 1)
        proba = float(model.predict(x, verbose=0)[0][0])
        label = "Attack" if proba >= 0.5 else "Normal"
        return {
            "label": label,
            "score": proba, # Confidence score
            "type": "classifier_result"
        }

    # --- Logic for Zero-Day Hunter (Autoencoder) ---
    elif model_key == "zero_day_hunter":
        # Output is a reconstruction of the input
        reconstruction = model.predict(x, verbose=0)
        # Calculate Mean Absolute Error between Input and Reconstruction
        error = np.mean((x - reconstruction) ** 2)

        
        is_anomaly = error > AUTOENCODER_THRESHOLD
        label = "Anomaly (Zero-Day)" if is_anomaly else "Normal"
        
        return {
            "label": label,
            "score": float(error), # Reconstruction error
            "threshold": AUTOENCODER_THRESHOLD,
            "type": "anomaly_result"
        }