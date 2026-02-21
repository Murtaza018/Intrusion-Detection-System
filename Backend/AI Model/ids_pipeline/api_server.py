from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
from functools import wraps
import psutil
import traceback
import sys
import os
import joblib
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- ECC CRYPTOGRAPHY IMPORTS ---
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

# --- CONFIG ---
from config import API_KEY

# --- RETRAINER IMPORTS ---
from gan_retrainer import GanRetrainer

# --- HELPER FUNCTIONS ---
def calculate_group_consistency(feature_list):
    """Calculates Cosine Similarity for consistency checks."""
    if not feature_list or len(feature_list) < 2: return 1.0
    matrix = np.array(feature_list)
    try:
        sim_matrix = cosine_similarity(matrix)
        iu = np.triu_indices(len(sim_matrix), k=1)
        if len(iu[0]) == 0: return 1.0
        avg_similarity = np.mean(sim_matrix[iu])
        return float(avg_similarity)
    except: return 0.0

class APIServer:
    """Flask API server for the Hybrid IDS with ECC Signing."""
    
    def __init__(self, packet_storage, feature_extractor, pipeline_manager, model_loader):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Store Dependencies
        self.packet_storage = packet_storage
        self.feature_extractor = feature_extractor
        self.pipeline_manager = pipeline_manager
        self.model_loader = model_loader
        
        # Initialize Retrainers
        self.gan_retrainer = GanRetrainer(packet_storage, feature_extractor)

        # --- ECC INITIALIZATION ---
        self._initialize_ecc()

        # Setup Routes
        self._register_routes()

    def _initialize_ecc(self):
        """Loads the ECC Private Key using absolute pathing."""
        try:
            # 1. Get the directory where api_server.py is located
            # Path: .../Backend/AI Model/ids_pipeline/
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # 2. Go up two levels to reach the 'Backend' folder
            # Path: .../Backend/
            backend_base = os.path.abspath(os.path.join(script_dir, "..", ".."))

            # 3. Target the ECC folder inside Backend
            key_path = os.path.join(backend_base, "ECC", "key.pem")

            if not os.path.exists(key_path):
                # Final fallback: check the current working directory
                key_path = os.path.abspath("Backend/ECC/key.pem")

            with open(key_path, "rb") as key_file:
                self.private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None, 
                )
            print(f"[*] ECC Security Layer: {key_path} loaded successfully.")
            
        except Exception as e:
            print(f"[!] ECC Initialization Error: {e}")
            print(f"[!] Attempted path: {key_path if 'key_path' in locals() else 'Unknown'}")
            self.private_key = None

    def _generate_signature(self, data_dict):
        """Creates a digital signature for a JSON response body."""
        if not self.private_key:
            return "NO_KEY_LOADED"
        
        try:
            # Serialize dict to a consistent JSON string
            json_str = json.dumps(data_dict, sort_keys=True)
            signature = self.private_key.sign(
                json_str.encode(),
                ec.ECDSA(hashes.SHA256())
            )
            return signature.hex()
        except:
            return "SIGNING_FAILED"

    def _secure_response(self, data, status=200):
        """Helper to wrap all responses with a digital signature."""
        response_body = {
            "payload": data,
            "signature": self._generate_signature(data),
            "signature_type": "ECDSA_SHA256",
            "server_time": datetime.now().isoformat()
        }
        return jsonify(response_body), status

    def _register_routes(self):
        """Setup Flask routes with API key authentication."""
        
        def require_api_key(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # GET THE KEY FROM HEADERS
                sent_key = request.headers.get('X-API-Key')
                
               
                
                if sent_key != API_KEY:
                    return jsonify({"error": "Invalid API key"}), 401
                return f(*args, **kwargs)
            return decorated_function
        
        # --- 1. PIPELINE CONTROL ---
        @self.app.route("/api/pipeline/start", methods=['POST'])
        @require_api_key
        def start_pipeline():
            try:
                sys.stdout.flush()
                success = self.pipeline_manager.start()
                if success:
                    data = {
                        "status": "started", 
                        "message": "IDS pipeline started",
                        "start_time": datetime.now().isoformat()
                    }
                    return self._secure_response(data)
                return self._secure_response({"error": "Failed to start pipeline"}, 500)
            except Exception as e:
                traceback.print_exc()
                return self._secure_response({"error": str(e)}, 500)
        
        @self.app.route("/api/pipeline/stop", methods=['POST'])
        @require_api_key
        def stop_pipeline():
            try:
                self.pipeline_manager.stop()
                return self._secure_response({"status": "stopped", "message": "IDS pipeline stopped"})
            except Exception as e:
                return self._secure_response({"error": str(e)}, 500)

        @self.app.route("/api/pipeline/status", methods=['GET'])
        @require_api_key
        def get_pipeline_status():
            data = {
                "running": self.pipeline_manager.is_running(),
                "start_time": self.packet_storage.get_stats().get("start_time"),
                "packets_processed": self.packet_storage.get_stats()["total_packets"],
            }
            return self._secure_response(data)

        # --- 2. DATA FETCHING ---
        @self.app.route("/api/packets/recent", methods=['GET'])
        @require_api_key
        def get_recent_packets():
            limit = request.args.get('limit', default=10, type=int)
            offset = request.args.get('offset', default=0, type=int)
            status = request.args.get('status', default=None, type=str)
            
            packets = self.packet_storage.get_packets(limit=limit, offset=offset, status_filter=status)
            data = {
                "packets": packets,
                "count": len(packets),
            }
            return self._secure_response(data)

        @self.app.route("/api/sensory/live", methods=['GET'])
        @require_api_key
        def get_live_sensory():
            recent = self.packet_storage.get_packets(limit=1)
            if recent:
                p = recent[0]
                expl = p.get('explanation', {})
                data = {
                    "gnn_anomaly": expl.get('gnn_anomaly', 0.0),
                    "mae_anomaly": expl.get('mae_anomaly', 0.0),
                    "status": p.get('status', 'unknown'),
                }
                return self._secure_response(data)
            return self._secure_response({"gnn_anomaly": 0.0, "mae_anomaly": 0.0})

        @self.app.route("/api/stats", methods=['GET'])
        @require_api_key
        def get_stats():
            stats = self.packet_storage.get_stats()
            recent_packets = self.packet_storage.get_packets(limit=50)
            if recent_packets:
                mae_vals = [p.get('explanation', {}).get('mae_anomaly', 0) for p in recent_packets]
                stats["avg_visual_anomaly"] = round(float(np.mean(mae_vals)), 4)
            else:
                stats["avg_visual_anomaly"] = 0.0

            try:
                mem = psutil.Process().memory_info().rss / 1024 / 1024
                stats["memory_usage_mb"] = round(mem, 1)
            except: 
                stats["memory_usage_mb"] = 0.0
            
            return self._secure_response(stats)

        # --- 3. LABELS ---
        @self.app.route("/api/labels", methods=['GET'])
        @require_api_key
        def get_labels():
            try:
                possible_paths = ["ids_pipeline/label_encoder.pkl", "label_encoder.pkl"]
                target_path = next((p for p in possible_paths if os.path.exists(p)), None)
                if target_path:
                    encoder = joblib.load(target_path)
                    return self._secure_response({"labels": list(encoder.classes_)})
            except: pass
            return self._secure_response({"labels": ["BENIGN", "DDoS", "PortScan", "Bot", "WebAttack"]})

        # --- 4. CONTINUAL LEARNING (GAN ANALYSIS) ---
        @self.app.route("/api/analyze_selection", methods=['POST'])
        @require_api_key
        def analyze_selection():
            data = request.json
            gan_ids = [p['id'] for p in data.get('gan_queue', [])]
            resp_payload = {"gan_score": 0.0, "gan_status": "Insufficient Data"}

            if gan_ids:
                raw_feats = self.packet_storage.get_features_for_training(gan_ids)
                feats = [f for f in raw_feats if f is not None and len(f) > 0]
                if len(feats) > 1:
                    score = calculate_group_consistency(feats)
                    resp_payload["gan_score"] = round(score, 4)
                    if score > 0.9: resp_payload["gan_status"] = "Excellent (Homogeneous)"
                    elif score > 0.6: resp_payload["gan_status"] = "Mixed (Caution)"
                    else: resp_payload["gan_status"] = "Poor (Too Diverse)"
                elif len(feats) == 1:
                    resp_payload["gan_score"] = 1.0
                    resp_payload["gan_status"] = "Single Item"

            return self._secure_response(resp_payload)

        @self.app.route("/api/retrain", methods=['POST'])
        @require_api_key
        def trigger_retrain():
            data = request.json
            if not data: return self._secure_response({"error": "No data provided"}, 400)
            
            gan_packets = data.get('gan_queue', [])
            target_label = data.get('target_label', 'Unknown_Attack')
            is_new_label = data.get('is_new_label', False)
            
            if not gan_packets:
                return self._secure_response({"message": "No packets to retrain on."})

            gan_ids = [p['id'] for p in gan_packets]
            result = self.gan_retrainer.retrain(gan_ids, target_label, is_new_label)

            final_data = {
                "status": "success" if result['status'] == 'success' else "error", 
                "message": result['message'],
                "gan_count": len(gan_packets)
            }
            return self._secure_response(final_data)

    def run(self, host="0.0.0.0", port=5001):
        """Run the Flask server."""
        print(f"[*] Starting Secure Production API with ECC Signing on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=True, use_reloader=False)