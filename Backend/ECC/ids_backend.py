# ids_backend.py
# Integrated backend with Flutter app support + existing security features

from flask import Flask, jsonify, request, Response
from flask_cors import CORS  # ADD THIS
from datetime import datetime, timedelta
import ssl
import json
import threading
import time
from collections import deque
import random

# --- Configuration ---
app = Flask(__name__)
CORS(app)
SECRET_API_KEY = "MySuperSecretKey12345!"

# --- Global State for Pipeline Control ---
pipeline_running = False
packets_captured = deque(maxlen=1000)  # Keep last 1000 packets
packet_id_counter = 1
stats = {
    "total_packets": 0,
    "normal_count": 0, 
    "attack_count": 0,
    "zero_day_count": 0,
    "start_time": None
}

# --- Mock Data ---
mock_alerts = [
    {
        "id": "1a2b3c",
        "timestamp": datetime.now().isoformat(),
        "type": "Nmap Xmas Scan",
        "source_ip": "192.168.1.101",
        "dest_port": 23,
        "severity": "High",
        "details": "A suspicious scan consistent with Nmap was detected."
    },
    {
        "id": "4d5e6f",
        "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat(),
        "type": "SQL Injection Attempt",
        "source_ip": "10.0.2.15",
        "dest_port": 80,
        "severity": "Critical",
        "details": "Detected a potential SQL injection pattern in an HTTP request payload."
    }
]

# --- Helper Functions ---
def authenticate_request():
    """Check if request has valid API key"""
    provided_key = request.headers.get("X-API-Key")
    if not provided_key or provided_key != SECRET_API_KEY:
        return False
    return True

def generate_mock_packet():
    """Generate realistic mock packet data"""
    global packet_id_counter
    
    protocols = ["TCP", "UDP", "ICMP"]
    statuses = ["normal", "normal", "normal", "normal", "known_attack", "zero_day"]  # Weighted
    src_ips = ["192.168.1.11", "192.168.1.12", "192.168.1.13", "192.168.1.14"]
    dst_ips = ["8.8.8.8", "1.1.1.1", "140.82.114.22", "104.18.39.21", "172.217.16.206"]
    
    protocol = random.choice(protocols)
    status = random.choice(statuses)
    src_ip = random.choice(src_ips)
    dst_ip = random.choice(dst_ips)
    src_port = random.randint(1000, 65000)
    dst_port = random.choice([80, 443, 53, 22, 3389])
    length = random.randint(64, 1500)
    
    packet = {
        "id": packet_id_counter,
        "summary": f"{protocol} {src_ip}:{src_port} → {dst_ip}:{dst_port}",
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "protocol": protocol,
        "src_port": src_port,
        "dst_port": dst_port,
        "length": length,
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "confidence": random.uniform(0.5, 0.95) if status != "normal" else 0.0,
    }
    
    # Add explanation for attacks
    if status == "known_attack":
        packet["explanation"] = {
            "type": "Known Attack",
            "risk": random.choice(["Medium", "High"]),
            "pattern": random.choice(["Port Scanning", "DDoS", "Brute Force"]),
            "features": ["Flow Duration", "Packet Rate", "TCP Flags"]
        }
    elif status == "zero_day":
        packet["explanation"] = {
            "type": "Zero-Day Anomaly", 
            "error": round(random.uniform(1.5, 8.7), 2),
            "threshold": 1.0,
            "pattern": "Unusual Protocol Behavior"
        }
    
    packet_id_counter += 1
    return packet

def pipeline_simulator():
    """Simulate packet capture when pipeline is running"""
    global pipeline_running, packets_captured, stats
    
    while True:
        if pipeline_running:
            # Generate 1-3 packets per second
            packet_count = random.randint(1, 3)
            for _ in range(packet_count):
                packet = generate_mock_packet()
                packets_captured.appendleft(packet)  # Add to beginning (latest first)
                
                # Update stats
                stats["total_packets"] += 1
                if packet["status"] == "normal":
                    stats["normal_count"] += 1
                elif packet["status"] == "known_attack":
                    stats["attack_count"] += 1
                elif packet["status"] == "zero_day":
                    stats["zero_day_count"] += 1
                
                print(f"[PACKET] {packet['summary']} - {packet['status'].upper()}")
            
            time.sleep(1)  # Wait 1 second
        else:
            time.sleep(0.5)  # Check more frequently when not running

# Start pipeline simulator in background thread
pipeline_thread = threading.Thread(target=pipeline_simulator, daemon=True)
pipeline_thread.start()

# --- API Endpoints ---

@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Get security alerts (existing endpoint)"""
    if not authenticate_request():
        return jsonify({"error": "Unauthorized"}), 401
    
    print("[LOG] Successful API access. /api/alerts endpoint was accessed.")
    return jsonify(mock_alerts)

@app.route("/api/pipeline/start", methods=["POST"])
def start_pipeline():
    """Start the IDS pipeline"""
    global pipeline_running, stats
    
    if not authenticate_request():
        return jsonify({"error": "Unauthorized"}), 401
    
    if not pipeline_running:
        pipeline_running = True
        stats["start_time"] = datetime.now().isoformat()
        print("[LOG] Pipeline started")
        return jsonify({"status": "started", "message": "IDS pipeline started successfully"})
    else:
        return jsonify({"status": "already_running", "message": "Pipeline is already running"})

@app.route("/api/pipeline/stop", methods=["POST"])
def stop_pipeline():
    """Stop the IDS pipeline"""
    global pipeline_running
    
    if not authenticate_request():
        return jsonify({"error": "Unauthorized"}), 401
    
    if pipeline_running:
        pipeline_running = False
        print("[LOG] Pipeline stopped")
        return jsonify({"status": "stopped", "message": "IDS pipeline stopped successfully"})
    else:
        return jsonify({"status": "already_stopped", "message": "Pipeline is already stopped"})

@app.route("/api/pipeline/status", methods=["GET"])
def pipeline_status():
    """Get current pipeline status"""
    if not authenticate_request():
        return jsonify({"error": "Unauthorized"}), 401
    
    return jsonify({
        "running": pipeline_running,
        "stats": stats,
        "uptime": str(datetime.now() - datetime.fromisoformat(stats["start_time"])) if stats["start_time"] else "0:00:00"
    })

@app.route("/api/packets/recent", methods=["GET"])
def get_recent_packets():
    """Get recent captured packets"""
    if not authenticate_request():
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get limit from query parameter, default to 50
    limit = min(int(request.args.get('limit', 50)), 200)
    
    recent_packets = list(packets_captured)[:limit]
    return jsonify({
        "packets": recent_packets,
        "count": len(recent_packets),
        "total_captured": stats["total_packets"]
    })

@app.route("/api/packets/stream")
def packet_stream():
    """Server-Sent Events stream for real-time packets"""
    if not authenticate_request():
        return jsonify({"error": "Unauthorized"}), 401
    
    def generate():
        last_count = 0
        while True:
            # Check for new packets
            if len(packets_captured) > last_count:
                new_packets = list(packets_captured)[last_count:len(packets_captured)]
                for packet in reversed(new_packets):  # Send oldest first
                    yield f"data: {json.dumps(packet)}\n\n"
                last_count = len(packets_captured)
            
            # Also send stats updates periodically
            yield f"data: {json.dumps({'type': 'stats', 'stats': stats})}\n\n"
            time.sleep(2)  # Send updates every 2 seconds
    
    return Response(generate(), mimetype="text/plain")

@app.route("/api/packets/<int:packet_id>", methods=["GET"])
def get_packet_details(packet_id):
    """Get detailed information for a specific packet"""
    if not authenticate_request():
        return jsonify({"error": "Unauthorized"}), 401
    
    # Find packet by ID
    for packet in packets_captured:
        if packet["id"] == packet_id:
            # Add more detailed information
            detailed_packet = packet.copy()
            detailed_packet.update({
                "detailed_info": {
                    "ttl": random.randint(50, 128),
                    "window_size": random.randint(1000, 65000),
                    "flags": "ACK" if packet["protocol"] == "TCP" else "N/A",
                    "service": "HTTP" if packet["dst_port"] == 80 else 
                               "HTTPS" if packet["dst_port"] == 443 else 
                               "DNS" if packet["dst_port"] == 53 else "Unknown"
                },
                "geo_info": {
                    "src_country": "Local" if packet["src_ip"].startswith("192.168") else "Unknown",
                    "dst_country": "USA" if packet["dst_ip"] in ["8.8.8.8", "1.1.1.1"] else "Unknown"
                }
            })
            return jsonify(detailed_packet)
    
    return jsonify({"error": "Packet not found"}), 404

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get current statistics"""
    if not authenticate_request():
        return jsonify({"error": "Unauthorized"}), 401
    
    return jsonify(stats)

@app.route("/api/system/info", methods=["GET"])
def system_info():
    """Get system information"""
    if not authenticate_request():
        return jsonify({"error": "Unauthorized"}), 401
    
    return jsonify({
        "version": "1.0.0",
        "models_loaded": ["hardened_classifier", "zero_day_hunter"],
        "scaling_enabled": True,
        "warmup_complete": True,
        "interface": "\\Device\\NPF_{2AD3C549-645F-4004-8232-782E6D2E2A91}",
        "features_extracted": 78
    })

# --- Run the Server ---
if __name__ == "__main__":
    print("--- Starting IDS Backend Server (HTTP - Development) ---")
    print("[*] Available Endpoints:")
    print("    GET  /api/alerts - Get security alerts")
    print("    POST /api/pipeline/start - Start IDS pipeline") 
    print("    POST /api/pipeline/stop - Stop IDS pipeline")
    print("    GET  /api/pipeline/status - Get pipeline status")
    print("    GET  /api/packets/recent - Get recent packets")
    print("    GET  /api/packets/stream - Real-time packet stream")
    print("    GET  /api/packets/<id> - Get packet details")
    print("    GET  /api/stats - Get statistics")
    print("    GET  /api/system/info - Get system info")
    
    print("\n[*] Server is running on: http://127.0.0.1:5000")  # CHANGED TO http
    print("[!] Use X-API-Key: MySuperSecretKey12345! for authentication")
    
    app.run(
        debug=True,
        host='0.0.0.0', 
        port=5000
        # REMOVED ssl_context parameter
    )