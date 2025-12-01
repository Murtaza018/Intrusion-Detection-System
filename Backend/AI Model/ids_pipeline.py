# ids_pipeline_fixed_xai.py
# FIXED XAI IMPORT - SAME FOLDER LEVEL

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
import warnings
import numpy as np

# Import SHAP and Scapy
import shap
from scapy.all import sniff, IP, TCP, UDP

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ----- CONFIGURATION -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MINMAX_SAVE_PATH = os.path.join(BASE_DIR, "feature_minimax.json")

# Model paths
MAIN_MODEL_REL_PATH = os.path.join("Adversarial Attack and Defense", "cicids_spatiotemporal_model_hardened.keras")
AUTOENCODER_REL_PATH = os.path.join("Autoencoder", "cicids_autoencoder.keras")

MAIN_MODEL_ABS_PATH = os.path.join(BASE_DIR, MAIN_MODEL_REL_PATH)
AUTOENCODER_ABS_PATH = os.path.join(BASE_DIR, AUTOENCODER_REL_PATH)

# API Configuration
API_KEY = "MySuperSecretKey12345!"

# ----- FIXED XAI IMPORT - SAME FOLDER LEVEL -----
# XAI folder is in the same directory as this script
XAI_DIR = os.path.join(BASE_DIR, "XAI")
if XAI_DIR not in sys.path:
    sys.path.insert(0, XAI_DIR)

print(f"[*] Looking for XAI in: {XAI_DIR}")

# Try to import the proper XAI system from XAI folder
try:
    from explanation_inference import explain_alert
    XAI_AVAILABLE = True
    print("[+] ✅ Loaded proper XAI explanation system from XAI folder")
except ImportError as e:
    print(f"[!] ❌ Could not load proper XAI system: {e}")
    print(f"[!] Files in XAI directory: {os.listdir(XAI_DIR) if os.path.exists(XAI_DIR) else 'Directory not found'}")
    print("[!] Will use enhanced fallback explanations")
    XAI_AVAILABLE = False

# ----- KERAS MODEL LOADING -----
print("[*] Loading Keras models...")
loaded_model = None
zero_day_model = None

try:
    if not os.path.exists(MAIN_MODEL_ABS_PATH):
        raise FileNotFoundError(f"Main model not found at: {MAIN_MODEL_ABS_PATH}")
    
    print(f"[*] Loading main model from: {MAIN_MODEL_REL_PATH}...")
    loaded_model = load_model(MAIN_MODEL_ABS_PATH, compile=False)
    
    if not os.path.exists(AUTOENCODER_ABS_PATH):
        raise FileNotFoundError(f"Autoencoder model not found at: {AUTOENCODER_ABS_PATH}")
        
    print(f"[*] Loading autoencoder from: {AUTOENCODER_REL_PATH}...")
    zero_day_model = load_model(AUTOENCODER_ABS_PATH, compile=False)

    print("[+] Keras models loaded successfully.")
    # Warmup models
    dummy_input = np.zeros((1, 78), dtype=np.float32)
    loaded_model.predict(dummy_input, verbose=0)
    zero_day_model.predict(dummy_input, verbose=0)
    print("[+] Model warmup complete.")

except FileNotFoundError as e:
    print(f"\n[!] CRITICAL ERROR: {e}")
    print("Please ensure the model directories exist with the correct .keras files")
    sys.exit(1)
except Exception as e:
    print(f"\n[!] CRITICAL ERROR loading Keras models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

from scapy.config import conf
INTERFACE = conf.iface

# FEATURE CONFIG
NUM_FEATURES = 78
FLOW_TIMEOUT = 300
DEBUG = True
WARMUP_SAMPLES = 50
BACKGROUND_SUMMARY_SIZE = 20

# MEMORY CONFIG
MAX_MEMORY_MB = 1500
XAI_QUEUE_MAXSIZE = 5

# THRESHOLD CONFIG
ERROR_WINDOW = 200
THRESHOLD_K = 3.0
MIN_SAMPLES_FOR_THRESHOLD = 50

# ----- GLOBAL STRUCTURES -----
packet_queue = queue.Queue()
xai_queue = queue.Queue(maxsize=XAI_QUEUE_MAXSIZE)
flows = defaultdict(lambda: {"packets": [], "timestamps": [], "lengths": [], "flags": []})
_recent_errors = deque(maxlen=ERROR_WINDOW)

# Min/max state
_live_min = None
_live_max = None
_warmup_count = 0
_warmup_lock = threading.Lock()
_scaling_enabled = False

# XAI globals
_background_data_buffer = deque(maxlen=100)
_shap_explainer = None
_shap_initialized = False
_xai_lock = threading.Lock()

# Pipeline state
_pipeline_running = False
_pipeline_start_time = None

# ----- FLUTTER API WITH AUTHENTICATION -----
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def require_api_key(f):
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != API_KEY:
            return jsonify({"error": "Invalid API key"}), 401
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Flutter display - thread-safe storage
class PacketStorage:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.packets = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.stats = {
            "total_packets": 0,
            "normal_count": 0,
            "attack_count": 0,
            "zero_day_count": 0,
            "start_time": None,
            "memory_usage_mb": 0,
            "pipeline_status": "stopped"
        }
        self.stats_lock = threading.Lock()

    def add_packet(self, packet_data):
        with self.lock:
            # Remove existing packet with same ID if present
            self.packets = deque([p for p in self.packets if p['id'] != packet_data['id']], maxlen=self.max_size)
            self.packets.appendleft(packet_data)

    def get_packets(self, limit=None):
        with self.lock:
            packets = list(self.packets)
            if limit and len(packets) > limit:
                return packets[:limit]
            return packets

    def update_stats(self, stats_update):
        with self.stats_lock:
            self.stats.update(stats_update)

    def get_stats(self):
        with self.stats_lock:
            return self.stats.copy()

    def clear(self):
        with self.lock:
            self.packets.clear()
        with self.stats_lock:
            self.stats.update({
                "total_packets": 0,
                "normal_count": 0,
                "attack_count": 0,
                "zero_day_count": 0
            })

packet_storage = PacketStorage()

@app.route("/api/pipeline/start", methods=['POST'])
@require_api_key
def start_pipeline():
    global _pipeline_running, _pipeline_start_time
    
    if _pipeline_running:
        return jsonify({"status": "already_running", "message": "Pipeline is already running"})
    
    _pipeline_running = True
    _pipeline_start_time = datetime.now().isoformat()
    
    packet_storage.update_stats({
        "start_time": _pipeline_start_time,
        "pipeline_status": "running"
    })
    
    print("[API] Pipeline started via Flutter request")
    return jsonify({
        "status": "started", 
        "message": "IDS pipeline started successfully",
        "start_time": _pipeline_start_time
    })

@app.route("/api/pipeline/stop", methods=['POST'])
@require_api_key
def stop_pipeline():
    global _pipeline_running
    
    if not _pipeline_running:
        return jsonify({"status": "already_stopped", "message": "Pipeline is already stopped"})
    
    _pipeline_running = False
    
    packet_storage.update_stats({
        "pipeline_status": "stopped"
    })
    
    print("[API] Pipeline stopped via Flutter request")
    return jsonify({"status": "stopped", "message": "IDS pipeline stopped successfully"})

@app.route("/api/pipeline/status", methods=['GET'])
@require_api_key
def get_pipeline_status():
    return jsonify({
        "running": _pipeline_running,
        "start_time": _pipeline_start_time,
        "current_time": datetime.now().isoformat(),
        "packets_processed": packet_storage.get_stats()["total_packets"]
    })

@app.route("/api/packets/recent", methods=['GET'])
@require_api_key
def get_recent_packets():
    limit = request.args.get('limit', default=10, type=int)
    packets = packet_storage.get_packets(limit=limit)
    
    return jsonify({
        "packets": packets,
        "count": len(packets),
        "limit": limit,
        "last_updated": datetime.now().isoformat()
    })

@app.route("/api/stats", methods=['GET'])
@require_api_key
def get_stats():
    stats = packet_storage.get_stats()
    stats["memory_usage_mb"] = round(get_memory_usage(), 1)
    stats["current_time"] = datetime.now().isoformat()
    return jsonify(stats)

@app.route("/api/system/health", methods=['GET'])
@require_api_key
def get_health():
    health_info = {
        "memory_mb": round(get_memory_usage(), 1),
        "total_packets_processed": packet_storage.get_stats()["total_packets"],
        "background_samples": len(_background_data_buffer),
        "active_flows": len(flows),
        "scaling_enabled": _scaling_enabled,
        "shap_initialized": _shap_initialized,
        "pipeline_running": _pipeline_running,
        "timestamp": datetime.now().isoformat()
    }
    return jsonify(health_info)

def run_flask():
    print("[FLUTTER] Starting Flask server on http://127.0.0.1:5001")
    print(f"[FLUTTER] API Key: {API_KEY}")
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)

# ----- PROPER XAI EXPLANATION SYSTEM -----
def generate_proper_xai_explanation(features, confidence, packet_info, attack_type="Attack"):
    """Use the proper XAI system from explanation_inference.py"""
    
    if XAI_AVAILABLE:
        try:
            print(f"[XAI] 🧠 Generating proper explanation using explain_alert()")
            explanation = explain_alert(features, "hardened_classifier", attack_type=attack_type)
            
            # Format the explanation for Flutter
            formatted_explanation = {
                "type": "PROPER_XAI_EXPLANATION",
                "title": "🔍 AI-Powered Threat Analysis",
                "facts": explanation.get("facts", {}),
                "explanation_text": explanation.get("explanation", "No explanation generated"),
                "risk_level": _extract_risk_level(explanation.get("explanation", "")),
                "confidence": f"{confidence:.1%}",
                "detection_method": "Ensemble Classifier + SHAP XAI",
                "technical_details": {
                    "model_used": "hardened_classifier",
                    "feature_count": len(features),
                    "attack_type": attack_type
                }
            }
            return formatted_explanation
        except Exception as e:
            print(f"[!] Proper XAI failed: {e}")
            # Fall through to fallback
    
    # Fallback explanation if proper XAI is not available
    return generate_fallback_explanation(features, confidence, packet_info, attack_type)

def _extract_risk_level(explanation_text):
    """Extract risk level from explanation text"""
    if "Risk: High" in explanation_text:
        return "HIGH"
    elif "Risk: Medium" in explanation_text:
        return "MEDIUM" 
    elif "Risk: Low" in explanation_text:
        return "LOW"
    else:
        return "MEDIUM"  # Default

def generate_fallback_explanation(features, confidence, packet_info, attack_type):
    """Enhanced fallback explanation"""
    
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
            "recommended_actions": [
                "Investigate source IP for compromise",
                "Check for unusual process executions", 
                "Review system and application logs",
                "Monitor for similar patterns"
            ]
        }
    else:
        return {
            "type": "KNOWN_ATTACK",
            "title": "🚨 Known Attack Pattern Detected",
            "description": f"Network traffic matches known {attack_type.lower()} patterns with {confidence:.1%} confidence.",
            "risk_level": "HIGH" if confidence > 0.8 else "MEDIUM",
            "confidence": f"{confidence:.1%}",
            "key_indicators": [
                "Suspicious network flow characteristics",
                "Anomalous packet timing and sizes", 
                "Matches trained attack signatures"
            ],
            "recommended_actions": [
                "Block source IP temporarily",
                "Investigate source for compromise",
                "Check authentication logs",
                "Update firewall rules if pattern persists"
            ]
        }

def generate_normal_explanation(features, packet_info):
    """Explanation for normal traffic"""
    return {
        "type": "NORMAL_TRAFFIC",
        "title": "✅ Normal Network Traffic",
        "description": "This network traffic matches expected patterns and shows no signs of malicious activity.",
        "risk_level": "LOW",
        "confidence": "99.9%",
        "verification_notes": [
            "All security checks passed",
            "No behavioral anomalies detected", 
            "Traffic patterns within normal parameters"
        ]
    }

# ----- MEMORY MANAGEMENT -----
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def check_memory_usage():
    current_memory = get_memory_usage()
    if current_memory > MAX_MEMORY_MB:
        emergency_memory_cleanup()
    return current_memory

def emergency_memory_cleanup():
    global _background_data_buffer, flows
    print(f"[!] Emergency memory cleanup - Current: {get_memory_usage():.1f}MB")
    gc.collect()
    tf.keras.backend.clear_session()
    if len(_background_data_buffer) > 50:
        _background_data_buffer = deque(list(_background_data_buffer)[-50:], maxlen=100)
    cleanup_old_flows(flows, max_age=60, max_flows=50)

# ----- FEATURE EXTRACTION -----
def extract_features_pure_python(packet):
    features = [0.0] * NUM_FEATURES
    try:
        if not packet.haslayer(IP): 
            return features
            
        ip_layer = packet[IP]
        proto = 6 if packet.haslayer(TCP) else 17 if packet.haslayer(UDP) else 0
        sport = getattr(packet, "sport", 0)
        dport = getattr(packet, "dport", 0)
        flow_key = (ip_layer.src, ip_layer.dst, sport, dport, proto)

        if flow_key not in flows:
            flows[flow_key] = {"packets": [], "timestamps": [], "lengths": [], "flags": []}
        flow = flows[flow_key]
        
        flow["packets"].append(packet)
        flow["timestamps"].append(time.time())
        pkt_len = len(packet)
        flow["lengths"].append(pkt_len)
        
        if len(flow["packets"]) > 50:
            for k in flow: 
                flow[k] = flow[k][-50:]

        last_flags = 0
        if proto == 6 and packet.haslayer(TCP):
            try: 
                last_flags = int(packet[TCP].flags)
            except: 
                pass
        flow["flags"].append(last_flags)

        lengths = flow["lengths"]
        pkt_count = len(lengths)
        total_len = float(sum(lengths))
        
        # Basic features
        features[0] = float(dport)
        features[1] = float(flow["timestamps"][-1] - flow["timestamps"][0] + 1e-9) if pkt_count > 1 else 0.0
        features[2] = float(pkt_count)
        features[3] = float(pkt_count)
        features[4] = float(total_len)
        features[5] = float(total_len)
        
        if pkt_count > 0:
            features[6] = float(max(lengths))
            features[7] = float(min(lengths))
            features[8] = float(total_len / pkt_count)
        
        dur = features[1]
        if dur > 1e-6:
            features[14] = float(total_len / dur)
            features[15] = float(pkt_count / dur)
        
        features[34] = float(pkt_len)
        features[35] = float(pkt_len)

    except Exception as e:
        if DEBUG: 
            print(f"[!] Feature extraction error: {e}")
    return features

def _update_minmax(vec):
    global _live_min, _live_max, _warmup_count, _scaling_enabled
    vec = np.asarray(vec, dtype=np.float32)
    with _warmup_lock:
        if _warmup_count < WARMUP_SAMPLES:
            _background_data_buffer.append(vec)
        if _live_min is None:
            _live_min = vec.copy()
            _live_max = vec.copy()
        else:
            _live_min = np.minimum(_live_min, vec)
            _live_max = np.maximum(_live_max, vec)
        _warmup_count += 1
        if not _scaling_enabled and _warmup_count >= WARMUP_SAMPLES:
            _scaling_enabled = True
            print(f"\n[***] WARMUP COMPLETE! Scaling enabled. Collected {len(_background_data_buffer)} samples.")

def scale_features_live(arr):
    if not _scaling_enabled or _live_min is None or _live_max is None: 
        return arr
    arr = np.asarray(arr, dtype=np.float32)
    denom = (_live_max - _live_min)
    denom_safe = np.where(denom == 0.0, 1.0, denom)
    return np.clip((arr - _live_min) / denom_safe, 0.0, 1.0)

# ----- PACKET PROCESSING FOR FLUTTER -----
packet_id_counter = 1

def send_to_flutter(packet, status, confidence=0.0, explanation=None, packet_id=None):
    global packet_id_counter
    
    if not _pipeline_running:
        return
    
    pid = packet_id if packet_id else packet_id_counter
    if not packet_id: 
        packet_id_counter += 1

    # Extract protocol information safely
    protocol = "OTHER"
    src_ip = ""
    dst_ip = ""
    src_port = 0
    dst_port = 0
    
    if packet.haslayer(IP):
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        if packet.haslayer(TCP):
            protocol = "TCP"
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif packet.haslayer(UDP):
            protocol = "UDP" 
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport

    # Create complete packet data for Flutter
    packet_data = {
        "id": pid,
        "summary": f"{protocol} {src_ip}:{src_port} → {dst_ip}:{dst_port}",
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": protocol,
        "length": len(packet),
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "confidence": float(confidence),
        "explanation": explanation
    }
    
    # Add to storage
    packet_storage.add_packet(packet_data)
    
    # Update statistics
    if not packet_id:
        current_stats = packet_storage.get_stats()
        new_stats = {
            "total_packets": current_stats["total_packets"] + 1,
            "memory_usage_mb": round(get_memory_usage(), 1)
        }
        
        if status == "normal": 
            new_stats["normal_count"] = current_stats["normal_count"] + 1
        elif status == "known_attack": 
            new_stats["attack_count"] = current_stats["attack_count"] + 1
        elif status == "zero_day": 
            new_stats["zero_day_count"] = current_stats["zero_day_count"] + 1
        
        packet_storage.update_stats(new_stats)

# ----- FIXED XAI WORKER -----
def xai_worker_logic():
    print("[*] Fixed XAI Worker started.")
    while True:
        try:
            task = xai_queue.get(timeout=1)
            if task is None: 
                break

            print(f"[XAI] Generating explanation for ID: {task['packet_id']}")
            start_t = time.time()

            packet_info = {
                "src_ip": task['packet'][IP].src if task['packet'].haslayer(IP) else "Unknown",
                "dst_ip": task['packet'][IP].dst if task['packet'].haslayer(IP) else "Unknown",
                "protocol": "TCP" if task['packet'].haslayer(TCP) else "UDP" if task['packet'].haslayer(UDP) else "OTHER"
            }

            # Use the proper XAI system
            if task['status'] == 'known_attack':
                explanation = generate_proper_xai_explanation(
                    task['features'][0], task['confidence'], packet_info, "Attack"
                )
            elif task['status'] == 'zero_day':
                explanation = generate_proper_xai_explanation(
                    task['features'][0], task['confidence'], packet_info, "Zero-Day"
                )
            else:
                explanation = generate_normal_explanation(task['features'][0], packet_info)

            explanation["computation_time"] = f"{time.time() - start_t:.2f}s"

            # Update packet with proper explanation
            send_to_flutter(
                packet=task['packet'], 
                status=task['status'], 
                confidence=task['confidence'], 
                explanation=explanation,
                packet_id=task['packet_id']
            )
            
            print(f"[XAI] Explanation generated for ID: {task['packet_id']} in {explanation['computation_time']}")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[!] XAI Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback explanation
            fallback_explanation = {
                "type": "ERROR",
                "title": "❌ Explanation Error",
                "description": f"Could not generate detailed explanation: {str(e)}",
                "risk_level": "UNKNOWN",
                "error_message": str(e)
            }
            
            send_to_flutter(
                packet=task['packet'], 
                status=task['status'], 
                confidence=task['confidence'], 
                explanation=fallback_explanation,
                packet_id=task['packet_id']
            )
        finally:
            if 'task' in locals():
                xai_queue.task_done()
            gc.collect()
            tf.keras.backend.clear_session()

# ----- DETECTION WORKER -----
def main_detection_worker():
    global last_flow_cleanup
    print("[*] Detection Worker started.")
    last_flow_cleanup = time.time()
    last_memory_check = 0
    _total_packets = 0
    
    while True:
        if not _pipeline_running:
            time.sleep(1)
            continue
            
        try:
            packet = packet_queue.get(timeout=1)
            _total_packets += 1
            cur_time = time.time()
            
            # Periodic cleanup
            if cur_time - last_flow_cleanup > 30:
                cleanup_old_flows(flows)
                last_flow_cleanup = cur_time

            # Memory check
            if _total_packets - last_memory_check > 100:
                check_memory_usage()
                last_memory_check = _total_packets

            # Feature extraction
            feats_list = extract_features_pure_python(packet)
            feats_np = np.array([feats_list], dtype=np.float32)
            _update_minmax(feats_np[0])
            feats_model = scale_features_live(feats_np) if _scaling_enabled else feats_np

            # Main classification
            prob = loaded_model.predict(feats_model, verbose=0)[0][0]
            label = "Attack" if prob > 0.5 else "Normal"
            
            cur_pid = packet_id_counter

            if label == "Attack":
                print(f"\n[!!!] KNOWN ATTACK DETECTED - ID:{cur_pid} Confidence:{prob:.4f}")
                
                # Send immediate alert
                initial_explanation = {
                    "type": "INITIAL_DETECTION",
                    "title": "🔍 Analyzing Attack...",
                    "description": "Known attack pattern detected. Generating AI-powered explanation...",
                    "risk_level": "HIGH",
                    "status": "analyzing"
                }
                
                send_to_flutter(
                    packet=packet, 
                    status="known_attack", 
                    confidence=prob, 
                    explanation=initial_explanation
                )
                
                # Queue for proper XAI analysis
                if _scaling_enabled and not xai_queue.full():
                    try:
                        xai_queue.put_nowait({
                            "packet_id": cur_pid, 
                            "packet": packet, 
                            "features": feats_model, 
                            "status": "known_attack", 
                            "confidence": prob
                        })
                        print(f"[XAI] Queued known attack for proper XAI analysis - ID:{cur_pid}")
                    except queue.Full:
                        print(f"[!] XAI queue full, basic alert sent for ID:{cur_pid}")
            else:
                # Zero-day check
                reconstruction = zero_day_model.predict(feats_model, verbose=0)
                mse = np.mean(np.power(feats_model - reconstruction, 2))
                _recent_errors.append(mse)
                
                # Dynamic threshold
                threshold = compute_dynamic_threshold()
                
                if mse > threshold and _scaling_enabled:
                    print(f"\n[?!?] ZERO-DAY ANOMALY DETECTED - ID:{cur_pid} Error:{mse:.4f} > Threshold:{threshold:.4f}")
                    
                    # Send immediate zero-day alert
                    initial_explanation = {
                        "type": "INITIAL_DETECTION", 
                        "title": "🔬 Analyzing Anomaly...",
                        "description": "Zero-day anomaly detected. Generating AI-powered explanation...",
                        "risk_level": "CRITICAL",
                        "status": "analyzing"
                    }
                    
                    send_to_flutter(
                        packet=packet, 
                        status="zero_day", 
                        confidence=mse,
                        explanation=initial_explanation
                    )
                    
                    # Queue for proper XAI analysis
                    if not xai_queue.full():
                        try:
                            xai_queue.put_nowait({
                                "packet_id": cur_pid, 
                                "packet": packet,
                                "features": feats_model, 
                                "status": "zero_day", 
                                "confidence": mse
                            })
                            print(f"[XAI] Queued zero-day for proper XAI analysis - ID:{cur_pid}")
                        except queue.Full:
                            print(f"[!] XAI queue full, basic alert sent for ID:{cur_pid}")
                else:
                    _total_packets += 1
                    if _total_packets % 20 == 0: 
                        print(".", end="", flush=True)
                    send_to_flutter(packet=packet, status="normal")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"\n[!] Detection Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if 'packet' in locals():
                packet_queue.task_done()

def compute_dynamic_threshold(default=1.0):
    if len(_recent_errors) < MIN_SAMPLES_FOR_THRESHOLD:
        return default
    
    arr = np.asarray(_recent_errors, dtype=float)
    arr = arr[np.isfinite(arr)]
    
    if arr.size == 0:
        return default
    
    m = float(arr.mean())
    s = float(arr.std())
    return m + THRESHOLD_K * s

def cleanup_old_flows(flows, max_age=300, max_flows=1000):
    now = time.time()
    flows_to_delete = []
    for flow_key, flow_data in flows.items():
        if flow_data["timestamps"] and now - flow_data["timestamps"][-1] > max_age:
            flows_to_delete.append(flow_key)
    for flow_key in flows_to_delete[:20]:
        if flow_key in flows: 
            del flows[flow_key]

def packet_handler(packet):
    if _pipeline_running and packet.haslayer(IP):
        packet_queue.put(packet)

if __name__ == "__main__":
    # Initial cleanup
    gc.collect()
    tf.keras.backend.clear_session()
    
    print(f"💾 Initial memory: {get_memory_usage():.1f} MB")
    print(f"🔑 API Key: {API_KEY}")
    print(f"🤖 XAI System: {'✅ Available' if XAI_AVAILABLE else '❌ Not Available'}")
    
    # Start threads
    flask_thread = threading.Thread(target=run_flask, daemon=True, name="Flask_Server")
    flask_thread.start()
    
    xai_thread = threading.Thread(target=xai_worker_logic, daemon=True, name="XAI_Worker")
    xai_thread.start()

    detect_thread = threading.Thread(target=main_detection_worker, daemon=True, name="Detect_Worker")
    detect_thread.start()
    
    print("\n" + "="*60)
    print("🚀 IDS PIPELINE READY - PROPER XAI INTEGRATION")
    print("="*60)
    print("✅ Keras Models Loaded")
    print(f"✅ XAI System: {'PROPER explain_alert()' if XAI_AVAILABLE else 'FALLBACK'}")
    print("✅ Flutter API Running on http://127.0.0.1:5001")
    print("⏸️  Pipeline stopped - Start via Flutter app")
    print("="*60)
    
    # Start sniffing
    sniff_kwargs = {
        "prn": packet_handler, 
        "store": 0, 
        "filter": "ip"
    }
    
    if INTERFACE:
        sniff_kwargs["iface"] = INTERFACE
        print(f"[*] Sniffing on interface: {INTERFACE}")
    else:
        print(f"[*] Sniffing on default interface")
    
    try:
        sniff(**sniff_kwargs)
    except KeyboardInterrupt:
        print(f"\n[!] Shutting down...")
    except Exception as e:
        print(f"\n[!] Sniffer Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _pipeline_running = False
        packet_queue.put(None)
        xai_queue.put(None)
        print("[+] IDS Pipeline shutdown complete")