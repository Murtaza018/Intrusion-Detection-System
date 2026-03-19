import warnings
import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import joblib
import torch
import sys

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
        self._models = {
            "cnn": None,
            "ae":  None,
            "rf":  None,
            "xgb": None,
            "gnn": None,
            "mae": None,
        }
        # RLock so the detection thread can safely read models while
        # ContinualRetrainer swaps them in after retraining.
        # Detection uses _lock in non-blocking read mode (just grab the
        # current reference). Retrainer acquires it exclusively when
        # writing a new model reference so the detection worker never
        # sees a half-trained model.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_models(self) -> bool:
        try:
            print("\n" + "=" * 40)
            print("🏗️  LOADING HYBRID DETECTION BRAIN")
            print("=" * 40)

            print(f"[*] Loading CNN: {os.path.basename(MAIN_MODEL_ABS_PATH)}...")
            self._models["cnn"] = load_model(MAIN_MODEL_ABS_PATH, compile=False)

            print(f"[*] Loading Autoencoder: {os.path.basename(AUTOENCODER_ABS_PATH)}...")
            self._models["ae"] = load_model(AUTOENCODER_ABS_PATH, compile=False)

            print(f"[*] Loading Random Forest: {os.path.basename(RF_MODEL_ABS_PATH)}...")
            self._models["rf"] = joblib.load(RF_MODEL_ABS_PATH)

            print(f"[*] Loading XGBoost: {os.path.basename(XGB_MODEL_ABS_PATH)}...")
            self._models["xgb"] = joblib.load(XGB_MODEL_ABS_PATH)

            print(f"[*] Loading GNN: {os.path.basename(GNN_MODEL_PATH)}...")
            gnn = ContextSAGE(in_channels=GNN_IN_CHANNELS, embedding_dim=GNN_EMBEDDING_DIM)
            gnn.load_state_dict(torch.load(GNN_MODEL_PATH, map_location="cpu"))
            gnn.eval()
            self._models["gnn"] = gnn

            print(f"[*] Loading MAE: {os.path.basename(MAE_MODEL_PATH)}...")
            mae = MAEModel(input_dim=78)
            mae.load_state_dict(torch.load(MAE_MODEL_PATH, map_location="cpu"))
            mae.eval()
            self._models["mae"] = mae

            print("[+] All models loaded.")

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
    # Path lookup
    # ------------------------------------------------------------------

    def get_path(self, key: str) -> str:
        path = _MODEL_PATHS.get(key)
        if path is None:
            raise KeyError(
                f"ModelLoader: unknown model key '{key}'. "
                f"Valid keys: {list(_MODEL_PATHS)}"
            )
        return path

    # ------------------------------------------------------------------
    # Thread-safe model swap (called by ContinualRetrainer after training)
    #
    # The retrain thread calls swap_model() instead of writing to
    # _models directly. This holds the lock while replacing the
    # reference so the detection thread never reads a half-trained model.
    # ------------------------------------------------------------------

    def swap_model(self, key: str, new_model):
        """
        Atomically replace a live model reference.
        ContinualRetrainer must call this instead of writing _models[key] directly.
        """
        with self._lock:
            self._models[key] = new_model
            print(f"[ModelLoader] Swapped '{key}' model safely.")

    # ------------------------------------------------------------------
    # Thread-safe getters
    # The detection thread grabs a local reference under the lock.
    # It then uses that local reference for inference — even if the
    # retrainer swaps the model mid-packet, the detection worker
    # finishes with the old (fully valid) model object.
    # ------------------------------------------------------------------

    def get_main_model(self):
        with self._lock:
            return self._models["cnn"]

    def get_autoencoder_model(self):
        with self._lock:
            return self._models["ae"]

    def get_rf_model(self):
        with self._lock:
            return self._models["rf"]

    def get_xgb_model(self):
        with self._lock:
            return self._models["xgb"]

    def get_gnn_model(self):
        with self._lock:
            return self._models["gnn"]

    def get_mae_model(self):
        with self._lock:
            return self._models["mae"]