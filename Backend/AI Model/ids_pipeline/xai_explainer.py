import sys
import os
import numpy as np
import shap
import threading
import json
from collections import deque
from datetime import datetime

# Add XAI directory to path
from config import XAI_DIR, USE_PROPER_XAI

class XAIExplainer:
    """Hybrid XAI System: Explains 95-feature Ensemble (Raw + GNN + MAE)"""
    
    def __init__(self):
        self.background_data = deque(maxlen=100)
        self.shap_explainer = None
        self.initialized = False
        self.lock = threading.Lock()
        # Load the base 78 feature names and append the 17 new sensory features
        self.feature_names = self._load_extended_feature_names()
    
    def _load_extended_feature_names(self):
        """Loads 78 base features + 16 GNN features + 1 MAE feature = 95 Total"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        xai_folder = os.path.join(project_root, "XAI")
        
        base_names = []
        schema_path = os.path.join(xai_folder, "feature_schema.json")
        
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                base_names = json.load(f)
        
        # Fallback if schema is missing or wrong size
        if len(base_names) != 78:
            base_names = [f"Raw_Feature_{i}" for i in range(78)]
            
        # POINT 3 ALIGNMENT: Append Sensory Feature Names
        gnn_names = [f"GNN_Context_{i}" for i in range(16)]
        mae_name = ["MAE_Visual_Anomaly_Score"]
        
        return base_names + gnn_names + mae_name

    def initialize_shap(self, model_predict_func, num_samples=20):
        """Initialize SHAP for the 95-dimensional input space"""
        with self.lock:
            if self.initialized: return True
            try:
                if len(self.background_data) < num_samples: return False
                
                background_array = np.array(list(self.background_data))
                # Summarize 95-dim background data
                bg_summary = shap.kmeans(background_array, min(num_samples, len(background_array)))
                
                self.shap_explainer = shap.KernelExplainer(
                    model=model_predict_func,
                    data=bg_summary
                )
                self.initialized = True
                print(f"[XAI] ✅ SHAP initialized for 95-feature Hybrid Ensemble")
                return True
            except Exception as e:
                print(f"[!] SHAP Init Failed: {e}")
                return False
    
    
    def generate_explanation(self, features, model_predict_func, confidence, packet_info, attack_type="Attack"):
        """Generate explanation accounting for Raw, Topological, and Visual features"""
        try:
            if not self.initialized:
                if not self.initialize_shap(model_predict_func):
                    return self._generate_fallback_explanation(features, confidence, packet_info, attack_type)
            
            # features is (1, 95)
            shap_values = self.shap_explainer.shap_values(features.reshape(1, -1), nsamples=50, silent=True)
            
            # --- FIXED ROBUST EXTRACTION (REMOVED target_array variable) ---
            if isinstance(shap_values, list):
                # For binary/multi-class, take positive class (index 1) or first (index 0)
                vals_to_flatten = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                shap_vals = vals_to_flatten.flatten()
            else:
                shap_vals = shap_values.flatten()
            
            # FORCE SLICE TO 95 (Roadmap Point 2 Alignment)
            shap_vals = shap_vals[:95] 
            
            # Get top contributors using the sliced array
            top_features = self._get_top_features(shap_vals, features.flatten()[:95])
            
            # Construct the Explanation Object
            explanation = {
                "type": "HYBRID_ENSEMBLE_XAI",
                "title": f"🚨 {attack_type.replace('_', ' ').title()} Detected",
                "risk_level": "CRITICAL" if attack_type == "zero_day" else "HIGH",
                "confidence": f"{confidence:.1%}",
                "top_contributing_factors": top_features,
                "sensory_analysis": {
                    "topological_shift": "Detected" if float(np.abs(np.sum(shap_vals[78:94]))) > 0.05 else "Stable",
                    "visual_anomaly": "Detected" if float(shap_vals[94]) > 0.05 else "Stable"
                },
                "recommended_actions": self._get_recommended_actions(attack_type),
                "timestamp": datetime.now().isoformat()
            }
            return explanation
            
        except Exception as e:
            print(f"[!] XAI Error: {e}")
            import traceback
            traceback.print_exc() 
            return self._generate_fallback_explanation(features, confidence, packet_info, attack_type)
    
    def _get_top_features(self, shap_values, features, top_n=5):
        contributions = []
        for i, contrib in enumerate(shap_values):
            name = self.feature_names[i] if i < len(self.feature_names) else f"Feature_{i}"
            
            # --- FIX: Convert contrib to a float to avoid array ambiguity ---
            c_val = float(contrib)
            contributions.append({
                "factor": name,
                "impact": "Increased Risk" if c_val > 0 else "Decreased Risk",
                "magnitude": f"{abs(c_val):.4f}",
                "observed_value": f"{float(features[i]):.4f}"
            })
        contributions.sort(key=lambda x: float(x["magnitude"]), reverse=True)
        return contributions[:top_n]

    def _get_recommended_actions(self, attack_type):
        if attack_type == "zero_day":
            return ["Isolate host", "Inspect MAE reconstruction grid", "Review GNN topological graph"]
        return ["Block Source IP", "Rate-limit Port", "Update Firewall Rules"]

    def _generate_fallback_explanation(self, features, confidence, packet_info, attack_type):
        return {"title": "Detection Alert", "risk_level": "HIGH", "confidence": f"{confidence:.1%}"}