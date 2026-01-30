import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import joblib
import torch
from GNN.train_gnn import ContextSAGE
from MAE.mae_model import MAEModel

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from config import MAIN_MODEL_ABS_PATH, AUTOENCODER_ABS_PATH, RF_MODEL_ABS_PATH, XGB_MODEL_ABS_PATH

class ModelLoader:
    def __init__(self):
        self.loaded_model = None
        self.zero_day_model = None
        self.rf_model = None   
        self.xgb_model = None  
        self.gnn_model = None
        self.mae_model = None
        
    def load_models(self):
        """Load both Keras models"""
        try:
            print("[*] Loading Keras models...")
            
            # Load main classifier
            if not os.path.exists(MAIN_MODEL_ABS_PATH):
                raise FileNotFoundError(f"Main model not found at: {MAIN_MODEL_ABS_PATH}")
            
            print(f"[*] Loading main model from: {MAIN_MODEL_ABS_PATH}...")
            self.loaded_model = load_model(MAIN_MODEL_ABS_PATH, compile=False)
            
            # Load autoencoder
            if not os.path.exists(AUTOENCODER_ABS_PATH):
                raise FileNotFoundError(f"Autoencoder model not found at: {AUTOENCODER_ABS_PATH}")
                
            print(f"[*] Loading autoencoder from: {AUTOENCODER_ABS_PATH}...")
            self.zero_day_model = load_model(AUTOENCODER_ABS_PATH, compile=False)

            print(f"[*] Loading Random Forest: {RF_MODEL_ABS_PATH}...")
            if not os.path.exists(RF_MODEL_ABS_PATH):
                 raise FileNotFoundError(f"RF model not found: {RF_MODEL_ABS_PATH}")
            self.rf_model = joblib.load(RF_MODEL_ABS_PATH)

            print(f"[*] Loading XGBoost: {XGB_MODEL_ABS_PATH}...")
            if not os.path.exists(XGB_MODEL_ABS_PATH):
                 raise FileNotFoundError(f"XGB model not found: {XGB_MODEL_ABS_PATH}")
            self.xgb_model = joblib.load(XGB_MODEL_ABS_PATH)

            print("[+] All models loaded successfully.")
            
            # Warmup models
            dummy_input = np.zeros((1, 78), dtype=np.float32)
            self.loaded_model.predict(dummy_input, verbose=0)
            self.zero_day_model.predict(dummy_input, verbose=0)
            print("[+] Model warmup complete.")

            print(f"[*] Loading GNN Context Engine: {GNN_MODEL_PATH}...")
            self.gnn_model = ContextSAGE(in_channels=GNN_IN_CHANNELS, embedding_dim=GNN_EMBEDDING_DIM)
            self.gnn_model.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=torch.device('cpu')))
            self.gnn_model.eval()
            print("[+] GNN loaded successfully.")

            print(f"[*] Loading MAE Visual Engine: {MAE_MODEL_PATH}...")
            self.mae_model = MAEModel()
            self.mae_model.load_state_dict(torch.load(MAE_MODEL_PATH, map_location=torch.device('cpu')))
            self.mae_model.eval()
            print("[+] MAE loaded successfully.")
            
            return True
            
        except Exception as e:
            print(f"\n[!] CRITICAL ERROR loading Keras models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_main_model(self):
        return self.loaded_model
    
    def get_autoencoder_model(self):
        return self.zero_day_model
    
    def get_rf_model(self): 
        return self.rf_model    
    
    def get_xgb_model(self): 
        return self.xgb_model
    
    def get_gnn_model(self):
        return self.gnn_model
    
    def get_mae_model(self):
        return self.mae_model