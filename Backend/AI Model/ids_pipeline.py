# ids_pipeline.py
# Threaded Intelligent IDS with:
#  - flow-aware feature extraction
#  - live min/max collection + saving to feature_minimax.json
#  - live scaling of features to [0,1] after warmup
#  - dynamic autoencoder threshold calibration from recent normal errors
#  - compact debug output

import sys
import os
import time
import threading
import queue
from collections import defaultdict, deque
from datetime import datetime
import json

import numpy as np
import requests
from scapy.all import sniff, IP, TCP, UDP

# ----- CONFIG -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

XAI_DIR = os.path.join(BASE_DIR, "XAI")
if XAI_DIR not in sys.path:
    sys.path.append(XAI_DIR)

# Import AI entrypoints (must exist)
try:
    from inference import predict  # prediction function used before
    from explanation_inference import explain_alert
except ImportError as e:
    print(f"[!] Error importing AI modules: {e}")
    sys.exit(1)

BACKEND_URL = "https://127.0.0.1:5000/api/alerts"
API_KEY = "MySuperSecretKey12345!"

from scapy.config import conf
INTERFACE = conf.iface

NUM_FEATURES = 78
FLOW_TIMEOUT = 300        # seconds, expire flows older than this
DEBUG = True              # set False to reduce console output

# Warmup and calibration
WARMUP_SAMPLES = 300             # collect this many feature vectors before enabling live scaling
MINMAX_SAVE_PATH = os.path.join(BASE_DIR, "feature_minimax.json")
ERROR_WINDOW = 500               # number of recent reconstruction errors to track
THRESHOLD_K = 6.0                # threshold = mean + THRESHOLD_K * std (calibrated from normal errors)
MIN_SAMPLES_FOR_THRESHOLD = 100  # require this many normal errors before using dynamic threshold

# Worker queue + flows
packet_queue = queue.Queue()
flows = defaultdict(lambda: {"packets": [], "timestamps": [], "lengths": [], "flags": []})

# Running stats for min/max
_live_min = None   # numpy array shape (NUM_FEATURES,)
_live_max = None   # numpy array shape (NUM_FEATURES,)
_warmup_count = 0
_warmup_lock = threading.Lock()

# Recent reconstruction errors (assume normal when frontline classifier says Normal)
_recent_errors = deque(maxlen=ERROR_WINDOW)

# Scaling enabled flag
_scaling_enabled = False

# Try to load previously saved min/max (if present)
if os.path.exists(MINMAX_SAVE_PATH):
    try:
        with open(MINMAX_SAVE_PATH, "r") as fh:
            mm = json.load(fh)
            _live_min = np.asarray(mm.get("min", []), dtype=float)
            _live_max = np.asarray(mm.get("max", []), dtype=float)
            if _live_min.shape[0] != NUM_FEATURES or _live_max.shape[0] != NUM_FEATURES:
                if DEBUG:
                    print("[!] feature_minimax.json shape mismatch -> ignoring file")
                _live_min = _live_max = None
            else:
                _scaling_enabled = True
                if DEBUG:
                    print("[*] Loaded existing feature_minimax.json -> live scaling enabled")
    except Exception as e:
        if DEBUG:
            print(f"[!] Failed to load {MINMAX_SAVE_PATH}: {e}")
        _live_min = _live_max = None
else:
    # Temporary fallback: estimate min/max from first N packets
    FEATURE_MIN = np.zeros(NUM_FEATURES)
    FEATURE_MAX = np.ones(NUM_FEATURES) * 1000  # rough guess
    print("[*] Using fallback min/max for live scaling")
    
# ---------------- Feature Extraction ----------------
def extract_features_pure_python(packet):
    """
    Flow-aware extractor. Returns a Python list of length NUM_FEATURES (floats).
    Defensive: converts Scapy FlagValue -> int, guards tiny durations, and uses deterministic placeholders.
    """
    features = [0.0] * NUM_FEATURES
    try:
        if not packet.haslayer(IP):
            return features

        ip_layer = packet[IP]
        proto = 6 if packet.haslayer(TCP) else 17 if packet.haslayer(UDP) else 0
        sport = getattr(packet, "sport", 0)
        dport = getattr(packet, "dport", 0)
        flow_key = (ip_layer.src, ip_layer.dst, sport, dport, proto)

        # Cleanup old flows
        now = time.time()
        for key in list(flows.keys()):
            if flows[key]["timestamps"] and now - flows[key]["timestamps"][-1] > FLOW_TIMEOUT:
                del flows[key]

        # Update flow data
        flows[flow_key]["packets"].append(packet)
        flows[flow_key]["timestamps"].append(now)
        pkt_len = len(packet)
        flows[flow_key]["lengths"].append(pkt_len)

        # Flags -> int (safe)
        last_flags = 0
        if proto == 6 and packet.haslayer(TCP):
            try:
                last_flags = int(packet[TCP].flags)
            except Exception:
                f = getattr(packet[TCP], "flags", 0)
                try:
                    last_flags = int(str(f))
                except Exception:
                    last_flags = 0
        flows[flow_key]["flags"].append(last_flags)

        # Basic stats
        lengths = flows[flow_key]["lengths"]
        pkt_count = len(lengths)
        total_len = float(sum(lengths))
        mean_len = float(total_len / pkt_count) if pkt_count else 0.0
        std_len = float((sum((l - mean_len) ** 2 for l in lengths) / pkt_count) ** 0.5) if pkt_count else 0.0

        # Fill feature slots defensively (mapping corresponds to your feature list)
        features[0] = float(dport)                                     # Destination Port
        features[1] = float(flows[flow_key]["timestamps"][-1] - flows[flow_key]["timestamps"][0] + 1e-9)  # Flow Duration
        features[2] = float(pkt_count)                                # Total Fwd Packets
        features[3] = float(pkt_count)                                # Total Backward Packets
        features[4] = float(total_len)                                # Total Length Fwd
        features[5] = float(total_len)                                # Total Length Bwd

        if pkt_count:
            features[6] = float(max(lengths))
            features[7] = float(min(lengths))
            features[8] = float(mean_len)
            features[9] = float(std_len)
            features[10] = features[6]
            features[11] = features[7]
            features[12] = features[8]
            features[13] = features[9]

        dur = features[1]
        if dur <= 1e-6:
            features[14] = float(total_len)
            features[15] = float(pkt_count)
        else:
            features[14] = float(total_len / dur)
            features[15] = float(pkt_count / dur)

        # Inter-arrival times
        timestamps = flows[flow_key]["timestamps"]
        iats = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        if iats:
            mean_iat = float(sum(iats) / len(iats))
            std_iat = float((sum((i - mean_iat) ** 2 for i in iats) / len(iats)) ** 0.5)
            features[16] = mean_iat
            features[17] = std_iat
            features[18] = float(max(iats))
            features[19] = float(min(iats))
        else:
            features[16:20] = [0.0, 0.0, 0.0, 0.0]

        # 20..29 placeholders (repeat mean_iat for determinism)
        for i in range(20, 30):
            features[i] = float(features[16])

        # Flags simplified
        features[30] = float(bool(last_flags & 0x08))  # PSH
        features[31] = float(bool(last_flags & 0x08))
        features[32] = float(bool(last_flags & 0x20))  # URG
        features[33] = float(bool(last_flags & 0x20))

        # Header length heuristics
        features[34] = float(pkt_len)
        features[35] = float(pkt_len)

        features[36] = float(features[15])
        features[37] = float(features[15])

        if pkt_count:
            features[38] = float(min(lengths))
            features[39] = float(max(lengths))
            features[40] = float(mean_len)
            features[41] = float(std_len)

        # flags counts and placeholders - use bit masks
        flags_map = {44: 0x01, 45: 0x02, 46: 0x04, 47: 0x08, 48: 0x10, 49: 0x20, 50: 0x40, 51: 0x80}
        for idx, mask in flags_map.items():
            features[idx] = float(bool(last_flags & mask))

        # Rest -> mean packet size as stable placeholder
        for i in range(52, NUM_FEATURES):
            features[i] = float(mean_len)

    except Exception as e:
        if DEBUG:
            print(f"[!] Feature extraction warning: {e}")

    return features

# ---------------- Live min/max updates & scaling ----------------
def _update_minmax(vec):
    """Update running min/max arrays with 1D numpy vec."""
    global _live_min, _live_max, _warmup_count, _scaling_enabled
    with _warmup_lock:
        if _live_min is None:
            _live_min = vec.copy()
            _live_max = vec.copy()
        else:
            _live_min = np.minimum(_live_min, vec)
            _live_max = np.maximum(_live_max, vec)
        _warmup_count += 1

        # when we reach warmup threshold, save file and enable scaling
        if not _scaling_enabled and _warmup_count >= WARMUP_SAMPLES:
            try:
                out = {"min": _live_min.tolist(), "max": _live_max.tolist()}
                with open(MINMAX_SAVE_PATH, "w") as fh:
                    json.dump(out, fh, indent=2)
                _scaling_enabled = True
                if DEBUG:
                    print(f"[*] Warmup complete ({_warmup_count} samples). Saved {MINMAX_SAVE_PATH}. Live scaling enabled.")
            except Exception as e:
                if DEBUG:
                    print(f"[!] Failed to save {MINMAX_SAVE_PATH}: {e}")

def scale_features_live(arr):
    """Scale arr (1D numpy) using current _live_min/_live_max if enabled."""
    if not _scaling_enabled or _live_min is None or _live_max is None:
        return arr
    denom = (_live_max - _live_min)
    denom_safe = np.where(denom == 0.0, 1.0, denom)
    scaled = (arr - _live_min) / denom_safe
    scaled = np.clip(scaled, 0.0, 1.0)
    return scaled

# ---------------- Dynamic threshold helper ----------------
def update_error_window(err, frontend_label):
    """
    Append reconstruction error if frontend labelled Normal.
    Use these errors to compute dynamic threshold.
    """
    if frontend_label == "Normal":
        _recent_errors.append(float(err))

def compute_dynamic_threshold(default=0.0001):
    """
    Compute threshold = mean + THRESHOLD_K * std from recent normal errors,
    only if enough samples exist; otherwise return default.
    """
    if len(_recent_errors) >= MIN_SAMPLES_FOR_THRESHOLD:
        arr = np.asarray(_recent_errors, dtype=float)
        m = float(arr.mean())
        s = float(arr.std())
        thr = m + THRESHOLD_K * s
        # ensure threshold not smaller than tiny default
        return max(thr, default)
    else:
        return default

# ---------------- Alert sending ----------------
def send_alert(alert_data):
    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}
    try:
        requests.post(BACKEND_URL, json=alert_data, headers=headers, verify=False, timeout=2)
    except Exception:
        pass

# ---------------- Worker logic ----------------
def worker_logic():
    print("[*] Processing Worker Started...")
    while True:
        packet = packet_queue.get()
        if packet is None:
            break

        try:
            # Debug: packet summary
            if DEBUG:
                print(f"[DEBUG] Packet: {packet.summary()}")

            # 1) feature extraction (pure python list)
            features_list = extract_features_pure_python(packet)
            features_np = np.array(features_list, dtype=np.float64)

            # 2) update min/max during warmup (always update to gather live min/max)
            _update_minmax(features_np)

            # 3) scale if scaling enabled
            features_for_model = scale_features_live(features_np) if _scaling_enabled else features_np

            if DEBUG:
                # print compact stats instead of full arrays
                try:
                    print(f"[DEBUG] Features shape: {features_for_model.shape}, mean={features_for_model.mean():.2f}, max={features_for_model.max():.2f}")
                except Exception:
                    print("[DEBUG] Features (unprintable)")

            # 4) Frontline classifier (expects appropriate shape inside predict)
            frontline_result = predict(features_for_model, "hardened_classifier")
            alert_data = None

            # If classifier flags attack
            if frontline_result["label"] == "Attack":
                print(f"\n[!!!] KNOWN ATTACK DETECTED! Score: {frontline_result['score']:.4f}")
                explanation = explain_alert(features_for_model, "hardened_classifier", attack_type="Known Attack")
                alert_data = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "Known Attack",
                    "model": "CNN+LSTM",
                    "src_ip": packet[IP].src if packet.haslayer(IP) else None,
                    "dst_ip": packet[IP].dst if packet.haslayer(IP) else None,
                    "confidence": float(frontline_result["score"]),
                    "explanation": explanation["explanation"],
                    "facts": explanation["facts"]
                }
                # For attacks, do not add to normal error window

            else:
                # 5) Zero-day (autoencoder) detection
                hunter_result = predict(features_for_model, "zero_day_hunter")
                # keep a record of normal reconstruction errors for dynamic threshold calibration
                update_error_window(hunter_result["score"], frontline_result["label"])

                # compute dynamic threshold from recent normal errors
                dynamic_threshold = compute_dynamic_threshold(default=hunter_result.get("threshold", 0.0001))

                # only report anomaly if error exceeds dynamic_threshold
                if hunter_result["score"] > dynamic_threshold:
                    print(f"\n[?!?] ZERO-DAY ANOMALY DETECTED! Error: {hunter_result['score']:.6f} (thr={dynamic_threshold:.6f})")
                    # pass a 2D array for explanation inference if required
                    explanation = explain_alert(features_for_model.reshape(1, -1), "zero_day_hunter", attack_type="Zero-Day Anomaly")
                    alert_data = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "Zero-Day Anomaly",
                        "model": "Autoencoder",
                        "src_ip": packet[IP].src if packet.haslayer(IP) else None,
                        "dst_ip": packet[IP].dst if packet.haslayer(IP) else None,
                        "reconstruction_error": float(hunter_result["score"]),
                        "threshold_used": float(dynamic_threshold),
                        "explanation": explanation["explanation"],
                        "facts": explanation["facts"]
                    }

            # send alert if present
            if alert_data:
                send_alert(alert_data)
            else:
                # heartbeat for normal packets
                print(".", end="", flush=True)

        except Exception as e:
            print(f"\n[!] Processing Error: {e}")
        finally:
            packet_queue.task_done()

# ---------------- Packet handler ----------------
def packet_handler(packet):
    if packet.haslayer(IP):
        packet_queue.put(packet)

# ---------------- Main ----------------
if __name__ == "__main__":
    print("--- Intelligent IDS Pipeline Started (Threaded) ---")
    print(f"[*] Frontline Model: Hardened CNN+LSTM")
    print(f"[*] Zero-Day Hunter: Autoencoder")
    print(f"[*] Warmup samples for live scaling: {WARMUP_SAMPLES}")
    print(f"[*] Dynamic threshold uses last {ERROR_WINDOW} normal errors (min samples {MIN_SAMPLES_FOR_THRESHOLD})")
    print(f"[*] XAI Engine: Active")
    print(f"[*] Secure Backend: {BACKEND_URL}")
    print(f"[*] Listening on: {INTERFACE}")
    print("----------------------------------------")

    worker_thread = threading.Thread(target=worker_logic, daemon=True)
    worker_thread.start()

    try:
        sniff(iface=INTERFACE, prn=packet_handler, store=0)
    except KeyboardInterrupt:
        print("\n[!] Stopping IDS...")
        packet_queue.put(None)
        worker_thread.join()
    except Exception as e:
        print(f"\n[!] Sniffer Error: {e}")
