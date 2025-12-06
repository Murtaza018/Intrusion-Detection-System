# api_server.py
# Flask API server for Flutter integration

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import psutil
import traceback
import sys

from config import API_KEY

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
            limit = request.args.get('limit', default=10000, type=int)
            packets = self.packet_storage.get_packets(limit=limit)
            
            return jsonify({
                "packets": packets,
                "count": len(packets),
                "limit": limit,
                "last_updated": datetime.now().isoformat()
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