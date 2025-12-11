
import sys
import os
import numpy as np
import shap
import threading
import json
from collections import deque

# Add XAI directory to path
from config import XAI_DIR, USE_PROPER_XAI

if XAI_DIR not in sys.path:
    sys.path.insert(0, XAI_DIR)

# Try to import proper XAI
XAI_AVAILABLE = False
if USE_PROPER_XAI:
    try:
        from ..XAI.explanation_inference import explain_alert
        XAI_AVAILABLE = True
        print("[+] ✅ Loaded proper XAI system")
    except ImportError as e:
        print(f"[!] Could not load XAI system: {e}")
        XAI_AVAILABLE = False

class XAIExplainer:
    """Complete XAI explanation system"""
    
    def __init__(self):
        self.background_data = deque(maxlen=100)
        self.shap_explainer = None
        self.initialized = False
        self.lock = threading.Lock()
        # Load feature names during initialization
        self.feature_names = self._load_feature_names()
    
    def _load_feature_names(self):
        """Load feature names relative to this script location"""
        # Get the directory containing this script (ids_pipeline)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Go up one level to the project root
        project_root = os.path.dirname(current_dir)
        
        # Construct path to XAI folder
        xai_folder = os.path.join(project_root, "XAI")
        
        print(f"[DEBUG] Looking for schema in: {xai_folder}")

        # List of potential filenames to try (prioritizing the one you confirmed)
        possible_filenames = [
            "feature_schema.json",
            "features schema.json", 
            "features_schema.json",
            "schema.json"
        ]
        
        for fname in possible_filenames:
            schema_path = os.path.join(xai_folder, fname)
            if os.path.exists(schema_path):
                try:
                    print(f"[*] Found feature schema at: {schema_path}")
                    with open(schema_path, 'r') as f:
                        feature_names = json.load(f)
                    
                    if isinstance(feature_names, list) and len(feature_names) == 78:
                        print(f"[+] ✅ Successfully loaded {len(feature_names)} feature names from {fname}")
                        return feature_names
                    else:
                        print(f"[!] Invalid schema format in {fname}. Expected list of 78 items.")
                except Exception as e:
                    print(f"[!] Error reading {fname}: {e}")

        print(f"[!] CRITICAL: Could not find valid feature schema in {xai_folder}")
        # Fallback to generic names
        return [f"Feature_{i}" for i in range(78)]

    def add_background_sample(self, features):
        """Add sample to background data for SHAP"""
        with self.lock:
            self.background_data.append(features)
    
    def initialize_shap(self, model_predict_func, num_samples=20):
        """Initialize SHAP explainer"""
        with self.lock:
            if self.initialized:
                return True
            
            try:
                print("[XAI] Initializing SHAP explainer...")
                
                if len(self.background_data) < num_samples:
                    print(f"[XAI] Need at least {num_samples} background samples, have {len(self.background_data)}")
                    return False
                
                # Convert to numpy array
                background_array = np.array(list(self.background_data))
                
                # Use K-means to summarize background data
                bg_summary = shap.kmeans(background_array, min(num_samples, len(background_array)))
                
                # Initialize SHAP KernelExplainer
                self.shap_explainer = shap.KernelExplainer(
                    model=model_predict_func,
                    data=bg_summary
                )
                
                self.initialized = True
                print(f"[XAI] ✅ SHAP initialized with {len(background_array)} background samples")
                return True
                
            except Exception as e:
                print(f"[!] SHAP initialization failed: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def generate_explanation(self, features, model_predict_func, confidence, packet_info, attack_type="Attack"):
        """Generate complete XAI explanation"""
        
        # Try to use proper XAI system first
        if XAI_AVAILABLE and attack_type != "zero_day":
            try:
                explanation = explain_alert(features, "hardened_classifier", attack_type=attack_type)
                return {
                    "type": "PROPER_XAI",
                    "title": "🔍 AI-Powered Threat Analysis",
                    "facts": explanation.get("facts", {}),
                    "explanation": explanation.get("explanation", "No explanation generated"),
                    "risk_level": self._extract_risk_level(explanation.get("explanation", "")),
                    "confidence": f"{confidence:.1%}",
                    "detection_method": "Ensemble Classifier + SHAP XAI",
                    "computation_method": "explain_alert()"
                }
            except Exception as e:
                print(f"[!] Proper XAI failed, falling back to SHAP: {e}")
        
        # Use SHAP for explanation
        try:
            if not self.initialized:
                if not self.initialize_shap(model_predict_func):
                    return self._generate_fallback_explanation(features, confidence, packet_info, attack_type)
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(
                features.reshape(1, -1), 
                nsamples=100, 
                silent=True
            )
            
            # Process SHAP output
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]  # For binary classification
            else:
                shap_vals = shap_values[0]
            
            # Get top contributing features
            top_features = self._get_top_features(shap_vals, features)
            
            # Generate rich explanation
            explanation = self._generate_shap_based_explanation(
                shap_vals, features, confidence, packet_info, attack_type, top_features
            )
            
            explanation["computation_method"] = "SHAP KernelExplainer"
            return explanation
            
        except Exception as e:
            print(f"[!] SHAP explanation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback_explanation(features, confidence, packet_info, attack_type)
    
    def _get_top_features(self, shap_values, features, top_n=5):
        """Get top contributing features from SHAP values"""
        contributions = []
        for i, contrib in enumerate(shap_values):
            # Use the feature name from the loaded list
            # Safety check: ensure index is within bounds of feature_names
            if i < len(self.feature_names):
                feature_name = self.feature_names[i]
            else:
                feature_name = f"Feature_{i}"
                
            contributions.append({
                "feature": feature_name,
                "contribution": float(contrib),
                "value": float(features[i])
            })
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions[:top_n]
    
    def _generate_shap_based_explanation(self, shap_values, features, confidence, packet_info, attack_type, top_features):
        """Generate rich explanation based on SHAP values"""
        
        if attack_type == "zero_day":
            return {
                "type": "ZERO_DAY_ANOMALY",
                "title": "🆕 Zero-Day Anomaly Detected",
                "description": "Unprecedented network behavior detected that doesn't match any known attack patterns.",
                "risk_level": "HIGH",
                "confidence": f"{confidence:.1%}",
                "key_indicators": [
                    "Behavior deviates significantly from normal patterns",
                    "Autoencoder reconstruction error elevated",
                    "Pattern doesn't match known attack signatures"
                ],
                "top_contributing_factors": [
                    {
                        "factor": f"{feat['feature']}",
                        "impact": "Increased risk" if feat['contribution'] > 0 else "Decreased risk",
                        "magnitude": f"{abs(feat['contribution']):.3f}",
                        "observed_value": f"{feat['value']:.2f}"
                    }
                    for feat in top_features
                ],
                "recommended_actions": [
                    "Investigate source IP for compromise",
                    "Check for unusual process executions", 
                    "Review system and application logs",
                    "Monitor for similar patterns"
                ]
            }
        else:
            # Determine attack subtype
            dst_port = int(features[0])
            attack_subtype = self._determine_attack_subtype(dst_port, top_features)
            
            explanations = {
                "brute_force": {
                    "title": "🚨 Brute Force Attack Detected",
                    "description": f"Multiple rapid connection attempts detected on port {dst_port} indicating potential credential brute forcing.",
                    "indicators": ["High packet frequency", "Multiple failed connections", "Suspicious source behavior"]
                },
                "web_attack": {
                    "title": "🌐 Web Application Attack",
                    "description": "Suspicious HTTP/HTTPS traffic patterns indicating potential SQL injection, XSS, or directory traversal attempts.",
                    "indicators": ["Abnormal request patterns", "Suspicious payload signatures", "Anomalous header structures"]
                },
                "ddos": {
                    "title": "⚠️ DDoS/Flood Attack",
                    "description": "High volume traffic patterns suggesting distributed denial of service or flood attack.",
                    "indicators": ["Extremely high packet rate", "Multiple source IPs", "Sustained traffic volume"]
                },
                "general": {
                    "title": "⚠️ Suspicious Network Activity",
                    "description": "Unusual network traffic patterns detected that match known attack signatures.",
                    "indicators": ["Anomalous flow characteristics", "Suspicious timing patterns", "Irregular packet sizes"]
                }
            }
            
            attack_info = explanations.get(attack_subtype, explanations["general"])
            
            return {
                "type": "KNOWN_ATTACK",
                "title": attack_info["title"],
                "description": attack_info["description"],
                "risk_level": "HIGH" if confidence > 0.8 else "MEDIUM",
                "confidence": f"{confidence:.1%}",
                "attack_classification": attack_subtype.replace('_', ' ').title(),
                "key_indicators": attack_info["indicators"],
                "top_contributing_factors": [
                    {
                        "factor": f"{feat['feature']}",
                        "impact": "Increased risk" if feat['contribution'] > 0 else "Decreased risk",
                        "magnitude": f"{abs(feat['contribution']):.3f}",
                        "observed_value": f"{feat['value']:.2f}"
                    }
                    for feat in top_features
                ],
                "recommended_actions": self._get_recommended_actions(attack_subtype),
                "shap_summary": {
                    "base_value": float(np.mean(shap_values)) if len(shap_values) > 0 else 0.0,
                    "total_impact": float(np.sum(np.abs(shap_values))) if len(shap_values) > 0 else 0.0
                }
            }
    
    def _determine_attack_subtype(self, dst_port, top_features):
        """Determine attack subtype based on port and features"""
        if dst_port in [21, 22, 23, 3389]:
            return "brute_force"
        elif dst_port in [80, 443, 8080]:
            return "web_attack"
        
        # Check top features for DDoS indicators
        ddos_indicators = ["Total Fwd Packets", "Total Backward Packets", "Flow Packets/s", "Flow Bytes/s"]
        if any(feat["feature"] in ddos_indicators and feat["contribution"] > 0.1 for feat in top_features[:3]):
            return "ddos"
        else:
            return "general"
    
    def _get_recommended_actions(self, attack_type):
        """Get recommended actions based on attack type"""
        actions = {
            "brute_force": [
                "Block source IP temporarily",
                "Implement rate limiting",
                "Enable account lockout policies",
                "Review SSH/RDP/FTP logs",
                "Check for compromised accounts"
            ],
            "web_attack": [
                "Block source IP temporarily",
                "Review web server logs",
                "Check for SQL injection patterns",
                "Validate input sanitization",
                "Update web application firewall rules"
            ],
            "ddos": [
                "Enable DDoS protection",
                "Block offending IP ranges",
                "Implement rate limiting",
                "Contact ISP if attack persists",
                "Monitor network bandwidth"
            ],
            "general": [
                "Block source IP temporarily",
                "Investigate source for compromise",
                "Check authentication logs",
                "Monitor for similar patterns",
                "Update firewall rules if pattern persists"
            ]
        }
        return actions.get(attack_type, actions["general"])
    
    def _generate_fallback_explanation(self, features, confidence, packet_info, attack_type):
        """Generate fallback explanation when SHAP fails"""
        if attack_type == "zero_day":
            return {
                "type": "ZERO_DAY_FALLBACK",
                "title": "🆕 Anomaly Detected",
                "description": f"Unusual network pattern detected with confidence {confidence:.1%}.",
                "risk_level": "MEDIUM",
                "confidence": f"{confidence:.1%}"
            }
        else:
            return {
                "type": "ATTACK_FALLBACK",
                "title": "⚠️ Attack Detected",
                "description": f"Known attack pattern detected with confidence {confidence:.1%}.",
                "risk_level": "HIGH" if confidence > 0.8 else "MEDIUM",
                "confidence": f"{confidence:.1%}",
                "recommended_actions": ["Investigate immediately", "Check logs", "Monitor network"]
            }

    def _extract_risk_level(self, explanation_text):
        """Extract risk level from explanation text"""
        if "Risk: High" in explanation_text:
            return "HIGH"
        elif "Risk: Medium" in explanation_text:
            return "MEDIUM"
        elif "Risk: Low" in explanation_text:
            return "LOW"
        else:
            return "MEDIUM"