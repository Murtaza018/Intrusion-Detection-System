# config.py
# Configuration settings for the IDS pipeline

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NETWORK_INTERFACE = "Wi-Fi 2"

# Model paths
MAIN_MODEL_REL_PATH = os.path.join("Adversarial Attack and Defense", "cicids_spatiotemporal_model_hardened.keras")
AUTOENCODER_REL_PATH = os.path.join("Autoencoder", "cicids_autoencoder.keras")
RF_MODEL_PATH = os.path.join("XGBoost and Random Forest","models_ensemble", "rf_model.joblib")
XGB_MODEL_PATH = os.path.join("XGBoost and Random Forest","models_ensemble", "xgb_model.joblib")

# Resolve absolute paths for them
RF_MODEL_ABS_PATH = os.path.join(BASE_DIR, RF_MODEL_PATH) # Adjust based on where you put the folder
XGB_MODEL_ABS_PATH = os.path.join(BASE_DIR, XGB_MODEL_PATH)
MAIN_MODEL_ABS_PATH = os.path.join(BASE_DIR, MAIN_MODEL_REL_PATH)
AUTOENCODER_ABS_PATH = os.path.join(BASE_DIR, AUTOENCODER_REL_PATH)

# API Configuration
API_KEY = "MySuperSecretKey12345!"
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5001

# Feature Configuration
NUM_FEATURES = 78
FLOW_TIMEOUT = 300
DEBUG = True
WARMUP_SAMPLES = 50
BACKGROUND_SUMMARY_SIZE = 20

# Memory Configuration
MAX_MEMORY_MB = 1500
XAI_QUEUE_MAXSIZE = 5
MEMORY_CHECK_INTERVAL = 100

# Threshold Configuration
ERROR_WINDOW = 200
THRESHOLD_K = 3.0
MIN_SAMPLES_FOR_THRESHOLD = 50

# XAI Configuration
XAI_DIR = os.path.join(BASE_DIR, "XAI")  # XAI folder location
USE_PROPER_XAI = True  # Set to False to use fallback