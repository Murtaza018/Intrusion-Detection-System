import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import joblib
import torch
import sys

# --- [SYSTEM PATH PROTECTION] ---
# Ensures the script can see the GNN and MAE folders in the parent directory
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
    MAE_MODEL_PATH
)

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class ModelLoader:
    def __init__(self):
        self.loaded_model = None   # CNN (78 Features)
        self.zero_day_model = None # AE (78 Features)
        self.rf_model = None       # RF (95 Features)
        self.xgb_model = None      # XGB (95 Features)
        self.gnn_model = None      # GNN Engine
        self.mae_model = None      # MAE Engine
        
    def load_models(self):
        """Load all 6 models for the Hybrid Pipeline"""
        try:
            print("\n" + "="*40)
            print("🏗️  LOADING HYBRID DETECTION BRAIN")
            print("="*40)
            
            # 1. Load Keras Models (78-dim)
            print(f"[*] Loading CNN Classifier: {os.path.basename(MAIN_MODEL_ABS_PATH)}...")
            self.loaded_model = load_model(MAIN_MODEL_ABS_PATH, compile=False)
            
            print(f"[*] Loading Autoencoder: {os.path.basename(AUTOENCODER_ABS_PATH)}...")
            self.zero_day_model = load_model(AUTOENCODER_ABS_PATH, compile=False)

            # 2. Load Retrained Ensemble Models (95-dim)
            print(f"[*] Loading 95-dim Random Forest: {os.path.basename(RF_MODEL_ABS_PATH)}...")
            self.rf_model = joblib.load(RF_MODEL_ABS_PATH)

            print(f"[*] Loading 95-dim XGBoost: {os.path.basename(XGB_MODEL_ABS_PATH)}...")
            self.xgb_model = joblib.load(XGB_MODEL_ABS_PATH)

            # 3. Load GNN Context Engine
            print(f"[*] Loading GNN Context Engine: {os.path.basename(GNN_MODEL_PATH)}...")
            self.gnn_model = ContextSAGE(in_channels=GNN_IN_CHANNELS, embedding_dim=GNN_EMBEDDING_DIM)
            self.gnn_model.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=torch.device('cpu')))
            self.gnn_model.eval()

            # 4. Load MAE Visual Engine
            print(f"[*] Loading MAE Visual Engine: {os.path.basename(MAE_MODEL_PATH)}...")
            self.mae_model = MAEModel(input_dim=78)
            self.mae_model.load_state_dict(torch.load(MAE_MODEL_PATH, map_location=torch.device('cpu')))
            self.mae_model.eval()

            print("[+] All models loaded successfully.")
            
            # --- [WARMUP LOGIC] ---
            # CNN and AE use 78 features
            dummy_78 = np.zeros((1, 78), dtype=np.float32)
            self.loaded_model.predict(dummy_78, verbose=0)
            self.zero_day_model.predict(dummy_78, verbose=0)
            
            # RF and XGB use 95 features (Raw 78 + GNN 16 + MAE 1)
            dummy_95 = np.zeros((1, 95), dtype=np.float32)
            self.rf_model.predict_proba(dummy_95)
            self.xgb_model.predict_proba(dummy_95)
            
            print("[+] Model warmup complete (78 and 95-dim paths verified).")
            return True
            
        except Exception as e:
            print(f"\n[!] CRITICAL ERROR loading hybrid brain: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Getters
    def get_main_model(self): return self.loaded_model
    def get_autoencoder_model(self): return self.zero_day_model
    def get_rf_model(self): return self.rf_model    
    def get_xgb_model(self): return self.xgb_model
    def get_gnn_model(self): return self.gnn_model
    def get_mae_model(self): return self.mae_model