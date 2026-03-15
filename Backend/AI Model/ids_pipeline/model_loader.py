import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import joblib
import torch
import sys

# --- [SYSTEM PATH PROTECTION] ---
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from GNN.train_gnn import ContextSAGE
from MAE.mae_model import MAEModel
from config import (
    MAIN_MODEL_ABS_PATH,
    AUTOENCODER_ABS_PATH,
    RF_MODEL_ABS_PATH,
    XGB_MODEL_ABS_PATH,
    GNN_EMBEDDING_DIM,
    GNN_IN_CHANNELS,
    GNN_MODEL_PATH,
    MAE_MODEL_PATH,
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Canonical key → absolute save path.
# ContinualRetrainer calls get_path(key) to know where to write updated models.
_MODEL_PATHS = {
    "cnn": MAIN_MODEL_ABS_PATH,
    "ae":  AUTOENCODER_ABS_PATH,
    "rf":  RF_MODEL_ABS_PATH,
    "xgb": XGB_MODEL_ABS_PATH,
    "gnn": GNN_MODEL_PATH,
    "mae": MAE_MODEL_PATH,
}


class ModelLoader:
    def __init__(self):
        # All live model objects stored here so ContinualRetrainer can
        # swap in a restored backup via:  self.model_loader._models[key] = restored
        self._models = {
            "cnn": None,
            "ae":  None,
            "rf":  None,
            "xgb": None,
            "gnn": None,
            "mae": None,
        }

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_models(self) -> bool:
        """Load all 6 models for the Hybrid Pipeline."""
        try:
            print("\n" + "=" * 40)
            print("🏗️  LOADING HYBRID DETECTION BRAIN")
            print("=" * 40)

            # 1. Keras models (78-dim)
            print(f"[*] Loading CNN: {os.path.basename(MAIN_MODEL_ABS_PATH)}...")
            self._models["cnn"] = load_model(MAIN_MODEL_ABS_PATH, compile=False)

            print(f"[*] Loading Autoencoder: {os.path.basename(AUTOENCODER_ABS_PATH)}...")
            self._models["ae"] = load_model(AUTOENCODER_ABS_PATH, compile=False)

            # 2. Sklearn models (95-dim)
            print(f"[*] Loading Random Forest: {os.path.basename(RF_MODEL_ABS_PATH)}...")
            self._models["rf"] = joblib.load(RF_MODEL_ABS_PATH)

            print(f"[*] Loading XGBoost: {os.path.basename(XGB_MODEL_ABS_PATH)}...")
            self._models["xgb"] = joblib.load(XGB_MODEL_ABS_PATH)

            # 3. GNN (PyTorch)
            print(f"[*] Loading GNN: {os.path.basename(GNN_MODEL_PATH)}...")
            gnn = ContextSAGE(in_channels=GNN_IN_CHANNELS, embedding_dim=GNN_EMBEDDING_DIM)
            gnn.load_state_dict(torch.load(GNN_MODEL_PATH, map_location="cpu"))
            gnn.eval()
            self._models["gnn"] = gnn

            # 4. MAE (PyTorch)
            print(f"[*] Loading MAE: {os.path.basename(MAE_MODEL_PATH)}...")
            mae = MAEModel(input_dim=78)
            mae.load_state_dict(torch.load(MAE_MODEL_PATH, map_location="cpu"))
            mae.eval()
            self._models["mae"] = mae

            print("[+] All models loaded.")

            # Warmup
            dummy_78 = np.zeros((1, 78), dtype=np.float32)
            self._models["cnn"].predict(dummy_78, verbose=0)
            self._models["ae"].predict(dummy_78, verbose=0)

            dummy_95 = np.zeros((1, 95), dtype=np.float32)
            self._models["rf"].predict_proba(dummy_95)
            self._models["xgb"].predict_proba(dummy_95)

            print("[+] Warmup complete (78 and 95-dim paths verified).")
            return True

        except Exception as e:
            print(f"\n[!] CRITICAL ERROR loading models: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Path lookup  (used by ContinualRetrainer to save updated models)
    # ------------------------------------------------------------------

    def get_path(self, key: str) -> str:
        """
        Returns the absolute file path for the given model key.
        Keys: 'cnn', 'ae', 'rf', 'xgb', 'gnn', 'mae'
        """
        path = _MODEL_PATHS.get(key)
        if path is None:
            raise KeyError(f"ModelLoader: unknown model key '{key}'. Valid keys: {list(_MODEL_PATHS)}")
        return path

    # ------------------------------------------------------------------
    # Getters  (rest of codebase uses these — unchanged API)
    # ------------------------------------------------------------------

    def get_main_model(self):        return self._models["cnn"]
    def get_autoencoder_model(self): return self._models["ae"]
    def get_rf_model(self):          return self._models["rf"]
    def get_xgb_model(self):         return self._models["xgb"]
    def get_gnn_model(self):         return self._models["gnn"]
    def get_mae_model(self):         return self._models["mae"]