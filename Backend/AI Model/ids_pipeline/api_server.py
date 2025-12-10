# api_server.py
# Full Version: Includes GAN Retraining + Detailed Debug Logs + All Utility Routes

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
from functools import wraps
import psutil
import traceback
import sys
import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
from config import API_KEY

# --- RETRAINER IMPORTS ---
from gan_retrainer import GanRetrainer
# from jitter_retrainer import JitterRetrainer # Uncomment when file is ready

# --- HELPER FUNCTIONS ---
def calculate_group_consistency(feature_list):
    """Calculates Cosine Similarity for consistency checks"""
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
    """Flask API server for Flutter"""
    
    def __init__(self, packet_storage, feature_extractor, pipeline_manager, model_loader):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # 1. Store Dependencies
        self.packet_storage = packet_storage
        self.feature_extractor = feature_extractor
        self.pipeline_manager = pipeline_manager
        self.model_loader = model_loader
        
        # 2. Initialize Retrainers
        self.gan_retrainer = GanRetrainer(packet_storage, feature_extractor)
        # self.jitter_retrainer = JitterRetrainer(packet_storage, feature_extractor, model_loader) 

        # 3. Setup Routes
        self._register_routes()
    
    def _register_routes(self):
        """Setup Flask routes with authentication"""
        
        def require_api_key(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if request.headers.get('X-API-Key') != API_KEY:
                    return jsonify({"error": "Invalid API key"}), 401
                return f(*args, **kwargs)
            return decorated_function
        
        # --- 1. PIPELINE CONTROL ---
        @self.app.route("/api/pipeline/start", methods=['POST'])
        @require_api_key
        def start_pipeline():
            try:
                print("[DEBUG] Request to START pipeline received")
                sys.stdout.flush()
                success = self.pipeline_manager.start()
                if success:
                    return jsonify({
                        "status": "started", 
                        "message": "IDS pipeline started successfully",
                        "start_time": datetime.now().isoformat()
                    })
                return jsonify({"error": "Failed to start pipeline"}), 500
            except Exception as e:
                print(f"[!] Exception in start_pipeline: {e}")
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/pipeline/stop", methods=['POST'])
        @require_api_key
        def stop_pipeline():
            try:
                print("[DEBUG] Request to STOP pipeline received")
                self.pipeline_manager.stop()
                return jsonify({"status": "stopped", "message": "IDS pipeline stopped successfully"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/pipeline/status", methods=['GET'])
        @require_api_key
        def get_pipeline_status():
            return jsonify({
                "running": self.pipeline_manager.is_running(),
                "start_time": self.packet_storage.get_stats().get("start_time"),
                "packets_processed": self.packet_storage.get_stats()["total_packets"],
                "current_time": datetime.now().isoformat()
            })

        # --- 2. DATA FETCHING ---
        @self.app.route("/api/packets/recent", methods=['GET'])
        @require_api_key
        def get_recent_packets():
            limit = request.args.get('limit', default=10, type=int)
            offset = request.args.get('offset', default=0, type=int)
            status = request.args.get('status', default=None, type=str)
            
            packets = self.packet_storage.get_packets(limit=limit, offset=offset, status_filter=status)
            
            return jsonify({
                "packets": packets,
                "count": len(packets),
                "filter": status,
                "timestamp": datetime.now().isoformat()
            })

        @self.app.route("/api/stats", methods=['GET'])
        @require_api_key
        def get_stats():
            stats = self.packet_storage.get_stats()
            try:
                mem = psutil.Process().memory_info().rss / 1024 / 1024
                stats["memory_usage_mb"] = round(mem, 1)
            except: 
                stats["memory_usage_mb"] = 0.0
            stats["current_time"] = datetime.now().isoformat()
            return jsonify(stats)

        @self.app.route("/api/system/health", methods=['GET'])
        @require_api_key
        def get_health():
            try:
                mem = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
            except: mem = 0.0
            
            return jsonify({
                "memory_mb": mem,
                "total_packets_processed": self.packet_storage.get_stats()["total_packets"],
                "pipeline_running": self.pipeline_manager.is_running(),
                "timestamp": datetime.now().isoformat()
            })

        # --- 3. LABELS (ROBUST PATH FINDING) ---
        @self.app.route("/api/labels", methods=['GET'])
        @require_api_key
        def get_labels():
            try:
                possible_paths = [
                    "ids_pipeline/label_encoder.pkl",
                    "label_encoder.pkl",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_encoder.pkl")
                ]
                target_path = next((p for p in possible_paths if os.path.exists(p)), None)
                
                if target_path:
                    encoder = joblib.load(target_path)
                    return jsonify({"labels": list(encoder.classes_)})
            except Exception as e:
                print(f"[!] Error loading labels: {e}")
            
            return jsonify({"labels": ["BENIGN", "DDoS", "PortScan", "Bot"]})

        # --- 4. ANALYSIS (CONSISTENCY CHECK) ---
        @self.app.route("/api/analyze_selection", methods=['POST'])
        @require_api_key
        def analyze_selection():
            data = request.json
            gan_ids = [p['id'] for p in data.get('gan_queue', [])]
            jitter_ids = [p['id'] for p in data.get('jitter_queue', [])]
            
            response = {
                "gan_score": 0.0, "gan_status": "Insufficient Data",
                "jitter_score": 0.0, "jitter_status": "Insufficient Data"
            }

            def get_valid_features(ids):
                raw_feats = self.packet_storage.get_features_for_training(ids)
                return [f for f in raw_feats if f and len(f) > 0]

            # GAN Queue Analysis
            if gan_ids:
                feats = get_valid_features(gan_ids)
                if len(feats) > 1:
                    try:
                        score = calculate_group_consistency(feats)
                        response["gan_score"] = round(score, 4)
                        if score > 0.9: response["gan_status"] = "Excellent (Homogeneous)"
                        elif score > 0.6: response["gan_status"] = "Mixed (Caution)"
                        else: response["gan_status"] = "Poor (Too Diverse)"
                    except: response["gan_status"] = "Error"
                elif len(feats) == 1:
                    response["gan_score"] = 1.0
                    response["gan_status"] = "Single Item"

            # Jitter Queue Analysis
            if jitter_ids:
                feats = get_valid_features(jitter_ids)
                if len(feats) > 1:
                    try:
                        score = calculate_group_consistency(feats)
                        response["jitter_score"] = round(score, 4)
                        response["jitter_status"] = "Consistent" if score > 0.85 else "Varied"
                    except: response["jitter_status"] = "Error"
                elif len(feats) == 1:
                    response["jitter_score"] = 1.0
                    response["jitter_status"] = "Single Item"

            return jsonify(response)

        # --- 5. RETRAINING LOGIC (GAN + JITTER) ---
        @self.app.route("/api/retrain", methods=['POST'])
        @require_api_key
        def trigger_retrain():
            data = request.json
            if not data: return jsonify({"error": "No data provided"}), 400
            
            gan_packets = data.get('gan_queue', [])
            jitter_packets = data.get('jitter_queue', [])
            target_label = data.get('target_label', 'Unknown_Attack')
            is_new_label = data.get('is_new_label', False)
            
            if not gan_packets and not jitter_packets:
                return jsonify({"message": "No packets to retrain on."}), 200

            messages = []
            print(f"\n[API] === Received Retrain Request ===")
            print(f"   - GAN Queue: {len(gan_packets)} packets")
            print(f"   - Label: {target_label} (New: {is_new_label})")

            # A. GAN Retraining
            if gan_packets:
                gan_ids = [p['id'] for p in gan_packets]
                print(f"[API] 🚀 Triggering GAN Pipeline on {len(gan_ids)} IDs...")
                
                result = self.gan_retrainer.retrain(gan_ids, target_label, is_new_label)
                messages.append(f"GAN: {result['message']}")
                
                if result['status'] == 'error':
                    print(f"[API] ❌ GAN Error: {result['message']}")

            # B. Jitter Retraining (Commented out until file is ready)
            if jitter_packets:
                messages.append("Jitter: Skipped (Disabled in Code)")
                # jitter_ids = [p['id'] for p in jitter_packets]
                # result = self.jitter_retrainer.retrain(jitter_ids)
                # messages.append(f"Jitter: {result['message']}")

            full_message = " | ".join(messages)
            print(f"[API] ✅ Retraining Cycle Complete.\n")

            return jsonify({
                "status": "success", 
                "message": full_message,
                "gan_count": len(gan_packets),
                "jitter_count": len(jitter_packets)
            })

    def run(self, host="0.0.0.0", port=5001):
        """Run the Flask server"""
        print(f"[FLUTTER] Starting Flask server on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=True, use_reloader=False)