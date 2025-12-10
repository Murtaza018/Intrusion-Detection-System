# api_server.py
# Flask API server for Flutter integration

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import psutil
import traceback
import sys

from config import API_KEY
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_group_consistency(feature_list):
    """
    Calculates how similar a group of feature vectors are.
    Returns a score between 0.0 (Different) and 1.0 (Identical).
    """
    if not feature_list or len(feature_list) < 2:
        return 1.0 # 1 packet is always consistent with itself
    
    # Convert to numpy array
    matrix = np.array(feature_list)
    
    # Compute Cosine Similarity between all pairs
    # Result is a N x N matrix where [i,j] is similarity between packet i and j
    sim_matrix = cosine_similarity(matrix)
    
    # We want the average similarity of the upper triangle (excluding self-similarity diagonal)
    # Get upper triangle indices
    iu = np.triu_indices(len(sim_matrix), k=1)
    
    if len(iu[0]) == 0:
        return 1.0
        
    avg_similarity = np.mean(sim_matrix[iu])
    return float(avg_similarity)


class APIServer:
    """Flask API server for Flutter"""
    
    def __init__(self, packet_storage, pipeline_manager):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.packet_storage = packet_storage
        self.pipeline_manager = pipeline_manager
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        def require_api_key(f):
            def decorated_function(*args, **kwargs):
                api_key = request.headers.get('X-API-Key')
                if api_key != API_KEY:
                    return jsonify({"error": "Invalid API key"}), 401
                return f(*args, **kwargs)
            decorated_function.__name__ = f.__name__
            return decorated_function
        
        @self.app.route("/api/pipeline/start", methods=['POST'])
        @require_api_key
        def start_pipeline():
            try:
                print("[DEBUG] start_pipeline called")
                sys.stdout.flush()
                
                print("[DEBUG] Calling pipeline_manager.start()")
                sys.stdout.flush()
                success = self.pipeline_manager.start()
                
                print(f"[DEBUG] pipeline_manager.start() returned: {success}")
                sys.stdout.flush()
                
                if success:
                    return jsonify({
                        "status": "started", 
                        "message": "IDS pipeline started successfully",
                        "start_time": datetime.now().isoformat()
                    })
                else:
                    return jsonify({"error": "Failed to start pipeline"}), 500
            except Exception as e:
                print(f"\n[!] CAUGHT EXCEPTION in start_pipeline: {e}")
                traceback.print_exc()
                sys.stdout.flush()
                return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500
        
        @self.app.route("/api/analyze_selection", methods=['POST'])
        @require_api_key
        def analyze_selection():
            data = request.json
            gan_ids = [p['id'] for p in data.get('gan_queue', [])]
            jitter_ids = [p['id'] for p in data.get('jitter_queue', [])]
            
            response = {
                "gan_score": 0.0,
                "jitter_score": 0.0,
                "gan_status": "Insufficient Data",
                "jitter_status": "Insufficient Data"
            }

            # Helper to filter valid features
            def get_valid_features(ids):
                raw_feats = self.packet_storage.get_features_for_training(ids)
                # Only keep features that are not empty lists
                return [f for f in raw_feats if f and len(f) > 0]

            # --- ANALYZE GAN QUEUE ---
            if gan_ids:
                feats = get_valid_features(gan_ids)
                if len(feats) > 1: # Need at least 2 to compare
                    try:
                        score = calculate_group_consistency(feats)
                        response["gan_score"] = round(score, 4)
                        
                        if score > 0.90: response["gan_status"] = "Excellent (Homogeneous)"
                        elif score > 0.80: response["gan_status"] = "Good"
                        elif score > 0.60: response["gan_status"] = "Mixed (Caution)"
                        else: response["gan_status"] = "Poor (Too Diverse)"
                    except Exception as e:
                        print(f"[!] Analysis Error (GAN): {e}")
                        response["gan_status"] = "Error in Calculation"
                elif len(feats) == 1:
                    response["gan_score"] = 1.0
                    response["gan_status"] = "Single Item (Perfect)"
                else:
                    response["gan_status"] = "No Features Found"

            # --- ANALYZE JITTER QUEUE ---
            if jitter_ids:
                feats = get_valid_features(jitter_ids)
                if len(feats) > 1:
                    try:
                        score = calculate_group_consistency(feats)
                        response["jitter_score"] = round(score, 4)
                        if score > 0.85: response["jitter_status"] = "Consistent"
                        else: response["jitter_status"] = "Varied"
                    except Exception as e:
                        print(f"[!] Analysis Error (Jitter): {e}")
                        response["jitter_status"] = "Error"
                elif len(feats) == 1:
                    response["jitter_score"] = 1.0
                    response["jitter_status"] = "Single Item"
                else:
                    response["jitter_status"] = "No Features Found"

            return jsonify(response)

        @self.app.route("/api/labels", methods=['GET'])
        @require_api_key
        def get_labels():
            try:
                import joblib
                import os
                
                # Robust Path Finding logic
                possible_paths = [
                    "ids_pipeline/label_encoder.pkl",
                    "label_encoder.pkl",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_encoder.pkl")
                ]
                
                target_path = None
                for p in possible_paths:
                    if os.path.exists(p):
                        target_path = p
                        break
                
                if target_path:
                    print(f"[API] Loading labels from: {target_path}") # Debug print
                    encoder = joblib.load(target_path)
                    labels = list(encoder.classes_)
                    return jsonify({"labels": labels})
                else:
                    print("[API] Error: label_encoder.pkl not found!")
            except Exception as e:
                print(f"[!] Error loading labels: {e}")
            
            return jsonify({"labels": ["BENIGN", "DDoS", "PortScan", "Bot"]})
        
        
        @self.app.route("/api/pipeline/stop", methods=['POST'])
        @require_api_key
        def stop_pipeline():
            self.pipeline_manager.stop()
            return jsonify({"status": "stopped", "message": "IDS pipeline stopped successfully"})
        
        @self.app.route("/api/pipeline/status", methods=['GET'])
        @require_api_key
        def get_pipeline_status():
            return jsonify({
                "running": self.pipeline_manager.is_running(),
                "start_time": self.packet_storage.get_stats().get("start_time"),
                "current_time": datetime.now().isoformat(),
                "packets_processed": self.packet_storage.get_stats()["total_packets"]
            })
        

        @self.app.route("/api/packets/recent", methods=['GET'])
        @require_api_key
        def get_recent_packets():
            limit = request.args.get('limit', default=10, type=int)
            offset = request.args.get('offset', default=0, type=int)
            status = request.args.get('status', default=None, type=str) # NEW
            
            # Pass status to storage
            packets = self.packet_storage.get_packets(limit=limit, offset=offset, status_filter=status)
            
            return jsonify({
                "packets": packets,
                "count": len(packets),
                "limit": limit,
                "offset": offset,
                "filter": status,
                "last_updated": datetime.now().isoformat()
            })
        
        # Add this new route
        @self.app.route("/api/retrain", methods=['POST'])
        @require_api_key
        def trigger_retrain():
            data = request.json
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
            gan_packets = data.get('gan_queue', [])
            jitter_packets = data.get('jitter_queue', [])
            
            if not gan_packets and not jitter_packets:
                return jsonify({"message": "No packets to retrain on."}), 200

            # 1. Extract IDs
            gan_ids = [p['id'] for p in gan_packets]
            jitter_ids = [p['id'] for p in jitter_packets]
            
            print(f"[API] Received Retrain Request:")
            print(f"   - GAN Queue: {len(gan_ids)} packets")
            print(f"   - Jitter Queue: {len(jitter_ids)} packets")

            # 2. Fetch Features from DB (Crucial Step!)
            # We need the math (features), not just the text summary
            gan_features = self.packet_storage.get_features_for_training(gan_ids)
            jitter_features = self.packet_storage.get_features_for_training(jitter_ids)
            
            # 3. Trigger Training (Placeholder for now)
            # This is where we will hook up your actual GAN/Jitter scripts later
            # For now, we confirm we got the data.
            success = True 
            message = f"Queued {len(gan_features)} samples for GAN and {len(jitter_features)} for Jittering."

            return jsonify({
                "status": "success", 
                "message": message,
                "gan_count": len(gan_features),
                "jitter_count": len(jitter_features)
            })


        @self.app.route("/api/stats", methods=['GET'])
        @require_api_key
        def get_stats():
            stats = self.packet_storage.get_stats()
            stats["memory_usage_mb"] = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
            stats["current_time"] = datetime.now().isoformat()
            return jsonify(stats)
        
        @self.app.route("/api/system/health", methods=['GET'])
        @require_api_key
        def get_health():
            health_info = {
                "memory_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
                "total_packets_processed": self.packet_storage.get_stats()["total_packets"],
                "pipeline_running": self.pipeline_manager.is_running(),
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(health_info)
    
    def run(self, host="127.0.0.1", port=5001):
        """Run the Flask server"""
        print(f"[FLUTTER] Starting Flask server on http://{host}:{port}")
        print(f"[FLUTTER] API Key: {API_KEY}")
        self.app.run(host=host, port=port, debug=True, use_reloader=False)