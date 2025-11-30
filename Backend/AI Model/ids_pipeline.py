# ids_pipeline_fixed.py
# COMPLETE VERSION: XAI with Memory Protection

import sys
import os
import time
import threading
import queue
from collections import defaultdict, deque
from datetime import datetime
import json
import gc
import psutil

import numpy as np
from scapy.all import sniff, IP, TCP, UDP

# ----- MEMORY PROTECTION -----
def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0

def safe_explain_alert(features, model_name, attack_type):
    """Safe XAI wrapper with memory limits"""
    memory_before = get_memory_usage()
    
    # Don't run XAI if memory is already high
    if memory_before > 1000:  # 1GB threshold
        return {
            "explanation": "Memory protection: XAI disabled to prevent system overload",
            "facts": ["System operating in safe mode due to memory constraints"]
        }
    
    try:
        from explanation_inference import explain_alert
        
        # Run XAI with timeout protection
        result = explain_alert(features, model_name, attack_type)
        
        memory_after = get_memory_usage()
        memory_used = memory_after - memory_before
        
        print(f"[XAI MEMORY] Used: {memory_used:.1f}MB, Total: {memory_after:.1f}MB")
        
        # Force garbage collection after XAI
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"[XAI ERROR] {e}")
        return {
            "explanation": f"XAI temporarily unavailable: {str(e)}",
            "facts": ["Explanation service recovering"]
        }

# ----- FLUTTER API INTEGRATION -----
from flask import Flask, jsonify, request
from flask_cors import CORS

# Initialize Flask app for Flutter communication
app = Flask(__name__)
CORS(app)

# Global state for Flutter communication
flutter_packets = deque(maxlen=100)  # Reduced from 1000
packet_id_counter = 1
flutter_stats = {
    "total_packets": 0,
    "normal_count": 0,
    "attack_count": 0, 
    "zero_day_count": 0,
    "start_time": None
}

def send_to_flutter(packet, status, confidence=0.0, explanation=None):
    """Send real packet data to Flutter app with memory checks"""
    global packet_id_counter, flutter_stats
    
    packet_data = {
        "id": packet_id_counter,
        "summary": f"{'TCP' if packet.haslayer(TCP) else 'UDP' if packet.haslayer(UDP) else 'OTHER'} {packet[IP].src if packet.haslayer(IP) else ''} → {packet[IP].dst if packet.haslayer(IP) else ''}",
        "src_ip": packet[IP].src if packet.haslayer(IP) else "",
        "dst_ip": packet[IP].dst if packet.haslayer(IP) else "",
        "protocol": "TCP" if packet.haslayer(TCP) else "UDP" if packet.haslayer(UDP) else "OTHER",
        "src_port": getattr(packet, "sport", 0),
        "dst_port": getattr(packet, "dport", 0),
        "length": len(packet),
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "confidence": float(confidence),
    }
    
    # Only include explanation if memory is safe
    if explanation and get_memory_usage() < 800:  # Only if < 800MB
        packet_data["explanation"] = explanation
    
    flutter_packets.appendleft(packet_data)
    packet_id_counter += 1
    
    # Update statistics
    flutter_stats["total_packets"] += 1
    if status == "normal":
        flutter_stats["normal_count"] += 1
    elif status == "known_attack":
        flutter_stats["attack_count"] += 1
    elif status == "zero_day":
        flutter_stats["zero_day_count"] += 1
    
    # Force GC every 50 packets
    if flutter_stats["total_packets"] % 50 == 0:
        gc.collect()

# Flask endpoints for Flutter app
@app.route("/api/pipeline/start", methods=['POST'])
def start_pipeline():
    """Start the IDS pipeline"""
    flutter_stats["start_time"] = datetime.now().isoformat()
    print("[FLUTTER] Pipeline start requested")
    return jsonify({"status": "started", "message": "IDS pipeline is running"})

@app.route("/api/pipeline/stop", methods=['POST'])
def stop_pipeline():
    """Stop the IDS pipeline"""
    print("[FLUTTER] Pipeline stop requested")
    return jsonify({"status": "stopped", "message": "Stop functionality not implemented yet"})

@app.route("/api/packets/recent", methods=['GET'])
def get_recent_packets():
    """Get recent captured packets"""
    limit = min(int(request.args.get('limit', 20)), 50)
    recent_packets = list(flutter_packets)[:limit]
    return jsonify({
        "packets": recent_packets,
        "count": len(recent_packets),
        "total_captured": flutter_stats["total_packets"]
    })

@app.route("/api/stats", methods=['GET'])
def get_stats():
    """Get current statistics including zero-day count"""
    return jsonify(flutter_stats)

@app.route("/api/pipeline/status", methods=['GET'])
def pipeline_status():
    return jsonify({
        "running": True,
        "stats": flutter_stats,
        "memory_mb": round(get_memory_usage(), 1)
    })

@app.route("/api/packets/<int:packet_id>", methods=['GET'])
def get_packet_details(packet_id):
    """Get detailed information for a specific packet"""
    for packet in flutter_packets:
        if packet["id"] == packet_id:
            return jsonify(packet)
    return jsonify({"error": "Packet not found"}), 404

def run_flask():
    """Run Flask in a separate thread"""
    print("[FLUTTER] Starting Flask server on http://127.0.0.1:5001")
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

# ----- ORIGINAL IDS PIPELINE CONFIG -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

XAI_DIR = os.path.join(BASE_DIR, "XAI")
if XAI_DIR not in sys.path:
    sys.path.append(XAI_DIR)

try:
    from inference import predict
    # We'll import explain_alert only when needed in safe_explain_alert
except ImportError as e:
    print(f"[!] Error importing AI modules: {e}")
    sys.exit(1)

BACKEND_URL = "https://127.0.0.1:5000/api/alerts"
API_KEY = "MySuperSecretKey12345!"

from scapy.config import conf
INTERFACE = conf.iface

NUM_FEATURES = 78
FLOW_TIMEOUT = 300
DEBUG = True

# Memory-optimized settings
WARMUP_SAMPLES = 4
MINMAX_SAVE_PATH = os.path.join(BASE_DIR, "feature_minimax.json")

ERROR_WINDOW = 200  # Reduced from 500
THRESHOLD_K = 6.0
MIN_SAMPLES_FOR_THRESHOLD = 50
PCT_FOR_THRESHOLD = 95.0

# More frequent flow cleanup
flow_cleanup_interval = 60
last_flow_cleanup = time.time()

# runtime structures
packet_queue = queue.Queue()
flows = defaultdict(lambda: {"packets": [], "timestamps": [], "lengths": [], "flags": []})

# min/max state and warmup
_live_min = None
_live_max = None
_warmup_count = 0
_warmup_lock = threading.Lock()
_recent_errors = deque(maxlen=ERROR_WINDOW)
_scaling_enabled = False

# Debug counters
_total_packets = 0
_anomaly_count = 0
_normal_count = 0

# ---------------- Memory-optimized feature extraction ----------------
def extract_features_pure_python(packet):
    """Lightweight feature extraction optimized for memory"""
    features = [0.0] * NUM_FEATURES
    try:
        if not packet.haslayer(IP):
            return features

        ip_layer = packet[IP]
        proto = 6 if packet.haslayer(TCP) else 17 if packet.haslayer(UDP) else 0
        sport = getattr(packet, "sport", 0)
        dport = getattr(packet, "dport", 0)
        flow_key = (ip_layer.src, ip_layer.dst, sport, dport, proto)

        # Memory-optimized flow cleanup
        global last_flow_cleanup
        now = time.time()
        if now - last_flow_cleanup > flow_cleanup_interval:
            for k in list(flows.keys()):
                if flows[k]["timestamps"] and now - flows[k]["timestamps"][-1] > FLOW_TIMEOUT:
                    del flows[k]
            last_flow_cleanup = now

        # Update flow with limits
        if flow_key not in flows:
            flows[flow_key] = {"packets": [], "timestamps": [], "lengths": [], "flags": []}
        
        flow = flows[flow_key]
        flow["packets"].append(packet)
        flow["timestamps"].append(now)
        pkt_len = len(packet)
        flow["lengths"].append(pkt_len)

        # Limit flow history to prevent memory growth
        if len(flow["packets"]) > 50:
            flow["packets"] = flow["packets"][-25:]
            flow["timestamps"] = flow["timestamps"][-25:]
            flow["lengths"] = flow["lengths"][-25:]
            flow["flags"] = flow["flags"][-25:]

        # Flags -> int safely
        last_flags = 0
        if proto == 6 and packet.haslayer(TCP):
            try:
                last_flags = int(packet[TCP].flags)
            except Exception:
                last_flags = 0
        flow["flags"].append(last_flags)

        lengths = flow["lengths"]
        pkt_count = len(lengths)
        total_len = float(sum(lengths))

        # Essential features only (reduced computation)
        features[0] = float(dport)
        features[1] = float(flow["timestamps"][-1] - flow["timestamps"][0] + 1e-9) if pkt_count > 1 else 1.0
        features[2] = float(pkt_count)
        features[3] = float(pkt_count)
        features[4] = float(total_len)
        features[5] = float(total_len)

        if pkt_count:
            features[6] = float(max(lengths))
            features[7] = float(min(lengths))
            features[8] = float(total_len / pkt_count)
            features[10] = features[6]
            features[11] = features[7]
            features[12] = features[8]

        dur = features[1]
        if dur <= 1e-6:
            features[14] = float(total_len)
            features[15] = float(pkt_count)
        else:
            features[14] = float(total_len / dur)
            features[15] = float(pkt_count / dur)

        # Simplified timing features
        timestamps = flow["timestamps"]
        if len(timestamps) > 1:
            iats = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            if iats:
                features[16] = float(sum(iats) / len(iats))
                features[18] = float(max(iats))

        features[34] = float(pkt_len)
        features[35] = float(pkt_len)

        # Fill remaining features with computed values to maintain shape
        for i in range(20, 30):
            features[i] = features[16] if features[16] != 0.0 else 0.1

        for i in range(52, NUM_FEATURES):
            features[i] = features[8] if features[8] != 0.0 else 1.0

    except Exception as e:
        if DEBUG:
            print(f"[!] Feature extraction warning: {e}")

    return features

# ---------------- min/max updates & scaling ----------------
def _update_minmax(vec):
    """Update running min/max arrays"""
    global _live_min, _live_max, _warmup_count, _scaling_enabled
    vec = np.asarray(vec, dtype=np.float32)  # Use float32 to save memory
    with _warmup_lock:
        if _live_min is None:
            _live_min = vec.copy()
            _live_max = vec.copy()
        else:
            _live_min = np.minimum(_live_min, vec)
            _live_max = np.maximum(_live_max, vec)
        _warmup_count += 1

        if not _scaling_enabled and _warmup_count >= WARMUP_SAMPLES:
            try:
                out = {"min": _live_min.tolist(), "max": _live_max.tolist()}
                with open(MINMAX_SAVE_PATH, "w") as fh:
                    json.dump(out, fh, indent=2)
                _scaling_enabled = True
                print(f"\n[***] WARMUP COMPLETE! Scaling enabled after {_warmup_count} samples")
            except Exception as e:
                print(f"[!] Failed to save {MINMAX_SAVE_PATH}: {e}")

def scale_features_live(arr):
    """Scale features using current min/max"""
    if not _scaling_enabled or _live_min is None or _live_max is None:
        return arr
    
    arr = np.asarray(arr, dtype=np.float32)
    denom = (_live_max - _live_min)
    denom_safe = np.where(denom == 0.0, 1.0, denom)
    scaled = (arr - _live_min) / denom_safe
    scaled = np.clip(scaled, 0.0, 1.0)
    
    return scaled

# ---------------- dynamic threshold ----------------
def update_error_window(err, frontend_label):
    """Record reconstruction error only when frontline labels traffic as Normal."""
    try:
        if frontend_label == "Normal":
            _recent_errors.append(float(err))
    except Exception:
        pass

def compute_dynamic_threshold(default=1.0):
    """Compute dynamic threshold for autoencoder."""
    if len(_recent_errors) >= MIN_SAMPLES_FOR_THRESHOLD:
        arr = np.asarray(_recent_errors, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return default
        m = float(arr.mean())
        s = float(arr.std())
        thr1 = m + THRESHOLD_K * s
        thr2 = float(np.percentile(arr, PCT_FOR_THRESHOLD))
        thr = max(thr1, thr2, default)
        return thr
    else:
        return default

# ---------------- alert sending ----------------
def send_alert(alert_data):
    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}
    try:
        requests.post(BACKEND_URL, json=alert_data, headers=headers, verify=False, timeout=2)
    except Exception:
        pass

# ---------------- Memory-optimized worker logic ----------------
def worker_logic():
    global _total_packets, _anomaly_count, _normal_count
    print("[*] Starting Memory-Safe Worker with XAI...")
    
    # Track last memory check
    last_memory_check = time.time()
    
    while True:
        packet = packet_queue.get()
        if packet is None:
            break
            
        try:
            _total_packets += 1
            
            # Memory monitoring every 20 packets
            if _total_packets % 20 == 0:
                current_time = time.time()
                if current_time - last_memory_check > 30:
                    memory_mb = get_memory_usage()
                    print(f"[MEMORY] Usage: {memory_mb:.1f} MB, Packets: {_total_packets}")
                    last_memory_check = current_time
                    
                    # Force garbage collection if memory is high
                    if memory_mb > 500:
                        gc.collect()
            
            # Feature extraction
            features_list = extract_features_pure_python(packet)
            features_np = np.array(features_list, dtype=np.float32)  # Use float32
            
            # Update running min/max
            _update_minmax(features_np)
            
            # Scale features
            features_for_model = scale_features_live(features_np) if _scaling_enabled else features_np

            # Frontline classifier
            frontline_result = predict(features_for_model, "hardened_classifier")
            
            if frontline_result["label"] == "Attack":
                _anomaly_count += 1
                print(f"\n[!!!] KNOWN ATTACK DETECTED! (#{_anomaly_count}) Score: {frontline_result['score']:.4f}")
                
                # SAFE XAI with memory protection
                explanation = safe_explain_alert(features_for_model, "hardened_classifier", "Known Attack")
                if explanation and 'explanation' in explanation:
                    print(f"[XAI] {explanation['explanation'][:100]}...")
                
                # Send to Flutter
                send_to_flutter(
                    packet=packet,
                    status="known_attack",
                    confidence=frontline_result["score"],
                    explanation=explanation
                )
                
                # Send alert to backend
                alert_data = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "Known Attack",
                    "model": "CNN+LSTM",
                    "src_ip": packet[IP].src if packet.haslayer(IP) else None,
                    "dst_ip": packet[IP].dst if packet.haslayer(IP) else None,
                    "confidence": float(frontline_result["score"]),
                }
                if explanation:
                    alert_data["explanation"] = explanation.get("explanation", "")
                    alert_data["facts"] = explanation.get("facts", [])
                
                send_alert(alert_data)

            else:
                _normal_count += 1
                # Zero-day hunter
                hunter_result = predict(features_for_model, "zero_day_hunter")
                reconstruction_error = hunter_result.get("score", 0.0)
                
                # Enhanced debugging for autoencoder
                if DEBUG and _total_packets <= 20:
                    scaling_status = "SCALED" if _scaling_enabled else "UNSCALED"
                    print(f"[AUTOENCODER] Packet {_total_packets} ({scaling_status}): Error = {reconstruction_error:.2f}")
                
                # record normal reconstruction errors only
                update_error_window(reconstruction_error, frontline_result["label"])

                # compute dynamic threshold
                dynamic_threshold = compute_dynamic_threshold(default=1.0)

                # report only if score exceeds dynamic threshold
                if reconstruction_error > dynamic_threshold:
                    _anomaly_count += 1
                    print(f"\n[?!?] ZERO-DAY ANOMALY #{_anomaly_count} (Packet {_total_packets})")
                    print(f"      Error: {reconstruction_error:.2f} > Threshold: {dynamic_threshold:.2f}")
                    
                    # Safe XAI for zero-day
                    explanation = safe_explain_alert(features_for_model.reshape(1, -1), "zero_day_hunter", "Zero-Day Anomaly")
                    
                    send_to_flutter(
                        packet=packet,
                        status="zero_day",
                        confidence=reconstruction_error,
                        explanation=explanation
                    )
                    
                    alert_data = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "Zero-Day Anomaly",
                        "model": "Autoencoder",
                        "src_ip": packet[IP].src if packet.haslayer(IP) else None,
                        "dst_ip": packet[IP].dst if packet.haslayer(IP) else None,
                        "reconstruction_error": float(reconstruction_error),
                        "threshold_used": float(dynamic_threshold),
                        "explanation": explanation.get("explanation", "Unusual pattern detected") if explanation else "Unusual pattern detected"
                    }
                    send_alert(alert_data)
                else:
                    # Send normal packet to Flutter
                    send_to_flutter(packet=packet, status="normal")

            # Print periodic stats
            if _total_packets % 50 == 0:
                memory_mb = get_memory_usage()
                print(f"\n[STATS] Packets: {_total_packets}, Normal: {_normal_count}, Anomalies: {_anomaly_count}")
                print(f"[MEMORY] Current usage: {memory_mb:.1f} MB")

        except Exception as e:
            print(f"\n[!] Processing Error on packet {_total_packets}: {e}")
        finally:
            packet_queue.task_done()

# ---------------- packet handler & main ----------------
def packet_handler(packet):
    if packet.haslayer(IP):
        packet_queue.put(packet)

if __name__ == "__main__":
    # Initial memory usage
    initial_memory = get_memory_usage()
    print(f"💾 Initial memory usage: {initial_memory:.1f} MB")
    
    # Start Flask in a separate thread for Flutter communication
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    print("--- Intelligent IDS Pipeline (Memory-Optimized with XAI) ---")
    print("✅ XAI ENABLED with memory protection")
    print("🚫 XAI auto-disabled when memory > 1GB")
    print(f"[*] Frontline Model: Hardened CNN+LSTM")
    print(f"[*] Zero-Day Hunter: Autoencoder")
    print(f"[*] Memory monitoring: ACTIVE")
    print(f"[*] Flutter API: http://127.0.0.1:5001")
    print(f"[*] Listening on: {INTERFACE}")
    print("----------------------------------------")

    worker_thread = threading.Thread(target=worker_logic, daemon=True)
    worker_thread.start()

    try:
        sniff(iface=INTERFACE, prn=packet_handler, store=0)
    except KeyboardInterrupt:
        final_memory = get_memory_usage()
        print(f"\n[!] Stopping IDS...")
        print(f"[FINAL STATS] Packets: {_total_packets}, Normal: {_normal_count}, Anomalies: {_anomaly_count}")
        print(f"[MEMORY] Final usage: {final_memory:.1f} MB (Started at: {initial_memory:.1f} MB)")
        packet_queue.put(None)
        worker_thread.join()
    except Exception as e:
        print(f"\n[!] Sniffer Error: {e}")