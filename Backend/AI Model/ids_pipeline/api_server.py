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
    """Flask API server for the Hybrid IDS frontend."""
    
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

        # Setup Routes
        self._register_routes()
    
    def _register_routes(self):
        """Setup Flask routes with API key authentication."""
        
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
                sys.stdout.flush()
                success = self.pipeline_manager.start()
                if success:
                    return jsonify({
                        "status": "started", 
                        "message": "IDS pipeline started",
                        "start_time": datetime.now().isoformat()
                    })
                return jsonify({"error": "Failed to start pipeline"}), 500
            except Exception as e:
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/pipeline/stop", methods=['POST'])
        @require_api_key
        def stop_pipeline():
            try:
                self.pipeline_manager.stop()
                return jsonify({"status": "stopped", "message": "IDS pipeline stopped"})
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
                "timestamp": datetime.now().isoformat()
            })

        # --- NEW: REAL-TIME SENSORY DATA (For Flutter Dashboard Gauges) ---
        @self.app.route("/api/sensory/live", methods=['GET'])
        @require_api_key
        def get_live_sensory():
            """
            Fetches the latest GNN and MAE scores from the most recent packet.
            This satisfies the Point 3 roadmap for real-time sensor visualization.
            """
            recent = self.packet_storage.get_packets(limit=1)
            if recent:
                p = recent[0]
                expl = p.get('explanation', {})
                # Extract the sensory metrics we injected in detector.py
                return jsonify({
                    "gnn_anomaly": expl.get('gnn_anomaly', 0.0),
                    "mae_anomaly": expl.get('mae_anomaly', 0.0),
                    "status": p.get('status', 'unknown'),
                    "timestamp": datetime.now().isoformat()
                })
            return jsonify({"gnn_anomaly": 0.0, "mae_anomaly": 0.0})

        @self.app.route("/api/stats", methods=['GET'])
        @require_api_key
        def get_stats():
            stats = self.packet_storage.get_stats()
            
            # Enrich stats with current sensory health (avg of last 50 packets)
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
            
            stats["current_time"] = datetime.now().isoformat()
            return jsonify(stats)

        # --- 3. LABELS ---
        @self.app.route("/api/labels", methods=['GET'])
        @require_api_key
        def get_labels():
            try:
                # Search for the label encoder to provide dynamic classes to Flutter
                possible_paths = [
                    "ids_pipeline/label_encoder.pkl",
                    "label_encoder.pkl",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "label_encoder.pkl")
                ]
                target_path = next((p for p in possible_paths if os.path.exists(p)), None)
                
                if target_path:
                    encoder = joblib.load(target_path)
                    return jsonify({"labels": list(encoder.classes_)})
            except: pass
            return jsonify({"labels": ["BENIGN", "DDoS", "PortScan", "Bot", "WebAttack"]})

        # --- 4. CONTINUAL LEARNING (GAN ANALYSIS) ---
        @self.app.route("/api/analyze_selection", methods=['POST'])
        @require_api_key
        def analyze_selection():
            data = request.json
            gan_ids = [p['id'] for p in data.get('gan_queue', [])]
            
            response = {"gan_score": 0.0, "gan_status": "Insufficient Data"}

            if gan_ids:
                raw_feats = self.packet_storage.get_features_for_training(gan_ids)
                feats = [f for f in raw_feats if f is not None and len(f) > 0]
                
                if len(feats) > 1:
                    score = calculate_group_consistency(feats)
                    response["gan_score"] = round(score, 4)
                    if score > 0.9: response["gan_status"] = "Excellent (Homogeneous)"
                    elif score > 0.6: response["gan_status"] = "Mixed (Caution)"
                    else: response["gan_status"] = "Poor (Too Diverse)"
                elif len(feats) == 1:
                    response["gan_score"] = 1.0
                    response["gan_status"] = "Single Item"

            return jsonify(response)

        @self.app.route("/api/retrain", methods=['POST'])
        @require_api_key
        def trigger_retrain():
            """Triggers the WGAN-GP retrainer for Point 1 & 2 Roadmap alignment."""
            data = request.json
            if not data: return jsonify({"error": "No data provided"}), 400
            
            gan_packets = data.get('gan_queue', [])
            target_label = data.get('target_label', 'Unknown_Attack')
            is_new_label = data.get('is_new_label', False)
            
            if not gan_packets:
                return jsonify({"message": "No packets to retrain on."}), 200

            gan_ids = [p['id'] for p in gan_packets]
            result = self.gan_retrainer.retrain(gan_ids, target_label, is_new_label)

            return jsonify({
                "status": "success" if result['status'] == 'success' else "error", 
                "message": result['message'],
                "gan_count": len(gan_packets)
            })

    def run(self, host="0.0.0.0", port=5001):
        """Run the Flask server."""
        print(f"[*] Starting Production API on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=True, use_reloader=False)