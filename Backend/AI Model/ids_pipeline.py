# ids_pipeline.py
# UPDATED: Reduced warmup samples and added scaling debug

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

try:
    from inference import predict
    from explanation_inference import explain_alert
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

# Warmup / calibration / thresholding - REDUCED FOR TESTING
WARMUP_SAMPLES = 4  # REDUCED from 300 to 10 for immediate testing
MINMAX_SAVE_PATH = os.path.join(BASE_DIR, "feature_minimax.json")
PROCESSED_X_PATH = os.path.join(BASE_DIR, "Preprocessing", "CIC-IDS-2017-Processed", "X_train.npy")

ERROR_WINDOW = 500
THRESHOLD_K = 6.0
MIN_SAMPLES_FOR_THRESHOLD = 100
PCT_FOR_THRESHOLD = 95.0

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

# ---------------- Helpers to load/compute minmax ----------------
def _try_load_minmax():
    """Try to load feature_minimax.json. Return True if loaded and enabled."""
    global _live_min, _live_max, _scaling_enabled
    if os.path.exists(MINMAX_SAVE_PATH):
        try:
            with open(MINMAX_SAVE_PATH, "r") as fh:
                mm = json.load(fh)
            mn = np.asarray(mm.get("min", []), dtype=float)
            mx = np.asarray(mm.get("max", []), dtype=float)
            if mn.size == NUM_FEATURES and mx.size == NUM_FEATURES:
                _live_min = mn
                _live_max = mx
                _scaling_enabled = True
                if DEBUG:
                    print("[*] Loaded feature_minimax.json -> live scaling enabled")
                return True
            else:
                if DEBUG:
                    print("[!] feature_minimax.json shape mismatch; ignoring file")
        except Exception as e:
            if DEBUG:
                print(f"[!] Failed to read {MINMAX_SAVE_PATH}: {e}")
    return False

_loaded = _try_load_minmax()

# If not present, attempt to compute from processed training data
if not _loaded and os.path.exists(PROCESSED_X_PATH):
    try:
        if DEBUG:
            print(f"[*] feature_minimax.json not found — attempting to compute min/max from {PROCESSED_X_PATH}")
        Xtrain = np.load(PROCESSED_X_PATH)
        if Xtrain.ndim == 2 and Xtrain.shape[1] == NUM_FEATURES:
            mn = np.nanmin(Xtrain, axis=0)
            mx = np.nanmax(Xtrain, axis=0)
            mn = np.where(np.isfinite(mn), mn, 0.0)
            mx = np.where(np.isfinite(mx), mx, mn + 1.0)
            _live_min = mn.astype(float)
            _live_max = mx.astype(float)
            try:
                out = {"min": _live_min.tolist(), "max": _live_max.tolist()}
                with open(MINMAX_SAVE_PATH, "w") as fh:
                    json.dump(out, fh, indent=2)
                _scaling_enabled = True
                if DEBUG:
                    print(f"[*] Computed and saved {MINMAX_SAVE_PATH}. Live scaling enabled")
            except Exception as e:
                if DEBUG:
                    print(f"[!] Failed to save computed minmax to {MINMAX_SAVE_PATH}: {e}")
        else:
            if DEBUG:
                print(f"[!] {PROCESSED_X_PATH} has unexpected shape (expected Nx{NUM_FEATURES})")
    except Exception as e:
        if DEBUG:
            print(f"[!] Error loading {PROCESSED_X_PATH}: {e}")

if not _scaling_enabled and DEBUG:
    print("[*] feature_minimax.json not found — warmup will collect live min/max before enabling scaling")
    print(f"[*] Warmup samples reduced to {WARMUP_SAMPLES} for testing")

# ---------------- Feature extraction (unchanged) ----------------
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

        # Cleanup old flows
        now = time.time()
        for k in list(flows.keys()):
            if flows[k]["timestamps"] and now - flows[k]["timestamps"][-1] > FLOW_TIMEOUT:
                del flows[k]

        # Update flow
        flows[flow_key]["packets"].append(packet)
        flows[flow_key]["timestamps"].append(now)
        pkt_len = len(packet)
        flows[flow_key]["lengths"].append(pkt_len)

        # Flags -> int safely
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

        lengths = flows[flow_key]["lengths"]
        pkt_count = len(lengths)
        total_len = float(sum(lengths))
        mean_len = float(total_len / pkt_count) if pkt_count else 0.0
        std_len = float((sum((l - mean_len) ** 2 for l in lengths) / pkt_count) ** 0.5) if pkt_count else 0.0

        # Fill features
        features[0] = float(dport)
        features[1] = float(flows[flow_key]["timestamps"][-1] - flows[flow_key]["timestamps"][0] + 1e-9)
        features[2] = float(pkt_count)
        features[3] = float(pkt_count)
        features[4] = float(total_len)
        features[5] = float(total_len)

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

        timestamps = flows[flow_key]["timestamps"]
        iats = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        if iats:
            mean_iat = float(sum(iats) / len(iats))
            std_iat = float((sum((i - mean_iat) ** 2 for i in iats) / len(iats) ) ** 0.5)
            features[16] = mean_iat
            features[17] = std_iat
            features[18] = float(max(iats))
            features[19] = float(min(iats))
        else:
            features[16:20] = [0.0, 0.0, 0.0, 0.0]

        for i in range(20, 30):
            features[i] = float(features[16])

        features[30] = float(bool(last_flags & 0x08))
        features[31] = float(bool(last_flags & 0x08))
        features[32] = float(bool(last_flags & 0x20))
        features[33] = float(bool(last_flags & 0x20))

        features[34] = float(pkt_len)
        features[35] = float(pkt_len)
        features[36] = float(features[15])
        features[37] = float(features[15])

        if pkt_count:
            features[38] = float(min(lengths))
            features[39] = float(max(lengths))
            features[40] = float(mean_len)
            features[41] = float(std_len)

        flags_map = {44: 0x01, 45: 0x02, 46: 0x04, 47: 0x08, 48: 0x10, 49: 0x20, 50: 0x40, 51: 0x80}
        for idx, mask in flags_map.items():
            features[idx] = float(bool(last_flags & mask))

        for i in range(52, NUM_FEATURES):
            features[i] = float(mean_len)

    except Exception as e:
        if DEBUG:
            print(f"[!] Feature extraction warning: {e}")

    return features

# ---------------- min/max updates & scaling ----------------
def _update_minmax(vec):
    """Update running min/max arrays and enable scaling once warmup threshold reached."""
    global _live_min, _live_max, _warmup_count, _scaling_enabled
    vec = np.asarray(vec, dtype=float)
    with _warmup_lock:
        if _live_min is None:
            _live_min = vec.copy()
            _live_max = vec.copy()
        else:
            _live_min = np.minimum(_live_min, vec)
            _live_max = np.maximum(_live_max, vec)
        _warmup_count += 1

        # Save and enable scaling when warmup completes
        if not _scaling_enabled and _warmup_count >= WARMUP_SAMPLES:
            try:
                out = {"min": _live_min.tolist(), "max": _live_max.tolist()}
                with open(MINMAX_SAVE_PATH, "w") as fh:
                    json.dump(out, fh, indent=2)
                _scaling_enabled = True
                print(f"\n[***] WARMUP COMPLETE! Scaling enabled after {_warmup_count} samples")
                print(f"[***] Min range: {_live_min.min():.2f} to {_live_min.max():.2f}")
                print(f"[***] Max range: {_live_max.min():.2f} to {_live_max.max():.2f}")
            except Exception as e:
                print(f"[!] Failed to save {MINMAX_SAVE_PATH}: {e}")

def scale_features_live(arr):
    """Scale 1D numpy arr using the current live min/max, clip to [0,1]."""
    if not _scaling_enabled or _live_min is None or _live_max is None:
        if DEBUG and _total_packets <= 15:  # Only show first few packets
            print(f"[SCALING] ⚠️  Scaling DISABLED - using raw features")
            print(f"[SCALING] Raw features - min: {arr.min():.2f}, max: {arr.max():.2f}, mean: {arr.mean():.2f}")
        return arr
    
    arr = np.asarray(arr, dtype=float)
    denom = (_live_max - _live_min)
    denom_safe = np.where(denom == 0.0, 1.0, denom)
    scaled = (arr - _live_min) / denom_safe
    scaled = np.clip(scaled, 0.0, 1.0)
    
    if DEBUG and _total_packets <= 15:  # Only show first few packets after scaling enabled
        print(f"[SCALING] ✅ Scaling ENABLED")
        print(f"[SCALING] Scaled features - min: {scaled.min():.4f}, max: {scaled.max():.4f}, mean: {scaled.mean():.4f}")
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

# ---------------- worker logic ----------------
def worker_logic():
    global _total_packets, _anomaly_count, _normal_count
    print("[*] Processing Worker Started...")
    
    while True:
        packet = packet_queue.get()
        if packet is None:
            break
        try:
            _total_packets += 1
            
            if DEBUG and _total_packets <= 15:
                print(f"\n[DEBUG] Packet {_total_packets}: {packet.summary()}")

            # 1) features
            features_list = extract_features_pure_python(packet)
            features_np = np.array(features_list, dtype=np.float64)

            # 2) update running min/max to support warmup
            _update_minmax(features_np)

            # 3) scale if enabled
            features_for_model = scale_features_live(features_np) if _scaling_enabled else features_np

            # 4) frontline classifier
            frontline_result = predict(features_for_model, "hardened_classifier")
            alert_data = None

            if frontline_result["label"] == "Attack":
                _anomaly_count += 1
                print(f"\n[!!!] KNOWN ATTACK DETECTED! (#{_anomaly_count}) Score: {frontline_result['score']:.4f}")
                explanation = explain_alert(features_for_model, "hardened_classifier", attack_type="Known Attack")
                print(f"[XAI] Explanation: {explanation['explanation']}")
                print(f"[XAI] Key Facts: {explanation['facts']}")
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

            else:
                _normal_count += 1
                # 5) zero-day (autoencoder)
                hunter_result = predict(features_for_model, "zero_day_hunter")
                reconstruction_error = hunter_result.get("score", 0.0)
                
                # Enhanced debugging for autoencoder
                if DEBUG and _total_packets <= 20:
                    scaling_status = "SCALED" if _scaling_enabled else "UNSCALED"
                    print(f"[AUTOENCODER] Packet {_total_packets} ({scaling_status}): Error = {reconstruction_error:.2f}")
                    if not _scaling_enabled:
                        print(f"[AUTOENCODER] ⚠️  WARNING: Using unscaled features (max value: {features_for_model.max():.1f})")
                
                # record normal reconstruction errors only
                update_error_window(reconstruction_error, frontline_result["label"])

                # compute dynamic threshold
                dynamic_threshold = compute_dynamic_threshold(default=1.0)

                # report only if score exceeds dynamic threshold
                if reconstruction_error > dynamic_threshold:
                    _anomaly_count += 1
                    print(f"\n[?!?] ZERO-DAY ANOMALY #{_anomaly_count} (Packet {_total_packets})")
                    print(f"      Error: {reconstruction_error:.2f} > Threshold: {dynamic_threshold:.2f}")
                    print(f"      Scaling: {'ENABLED' if _scaling_enabled else 'DISABLED'}")
                    
                    explanation = explain_alert(features_for_model.reshape(1, -1), "zero_day_hunter", attack_type="Zero-Day Anomaly")
                    print(f"[XAI] Explanation: {explanation['explanation']}")
                    print(f"[XAI] Key Facts: {explanation['facts']}")
                    alert_data = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "Zero-Day Anomaly",
                        "model": "Autoencoder",
                        "src_ip": packet[IP].src if packet.haslayer(IP) else None,
                        "dst_ip": packet[IP].dst if packet.haslayer(IP) else None,
                        "reconstruction_error": float(reconstruction_error),
                        "threshold_used": float(dynamic_threshold),
                        "explanation": explanation["explanation"],
                        "facts": explanation["facts"]
                    }

            # Print periodic stats
            if _total_packets % 50 == 0:
                print(f"\n[STATS] Packets: {_total_packets}, Normal: {_normal_count}, Anomalies: {_anomaly_count}")
                print(f"[STATS] Scaling: {_scaling_enabled}, Warmup: {_warmup_count}/{WARMUP_SAMPLES}")

            if alert_data:
                send_alert(alert_data)
            else:
                # light heartbeat for normal packets
                if _total_packets % 10 == 0:
                    print(".", end="", flush=True)

        except Exception as e:
            print(f"\n[!] Processing Error on packet {_total_packets}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            packet_queue.task_done()

# ---------------- packet handler & main ----------------
def packet_handler(packet):
    if packet.haslayer(IP):
        packet_queue.put(packet)

if __name__ == "__main__":
    print("--- Intelligent IDS Pipeline Started (Threaded) ---")
    print(f"[*] Frontline Model: Hardened CNN+LSTM")
    print(f"[*] Zero-Day Hunter: Autoencoder")
    print(f"[*] Warmup samples for live scaling: {WARMUP_SAMPLES} ⚡ REDUCED FOR TESTING")
    print(f"[*] Dynamic threshold uses last {ERROR_WINDOW} normal errors (min samples {MIN_SAMPLES_FOR_THRESHOLD})")
    print(f"[*] Default threshold: 1.0 (increased to reduce false positives)")
    print(f"[*] XAI Engine: Active (SHAP runs only for confirmed anomalies)")
    print(f"[*] Secure Backend: {BACKEND_URL}")
    print(f"[*] Listening on: {INTERFACE}")
    print("----------------------------------------")

    worker_thread = threading.Thread(target=worker_logic, daemon=True)
    worker_thread.start()

    try:
        sniff(iface=INTERFACE, prn=packet_handler, store=0)
    except KeyboardInterrupt:
        print(f"\n[!] Stopping IDS...")
        print(f"[FINAL STATS] Total packets: {_total_packets}, Anomalies: {_anomaly_count}")
        packet_queue.put(None)
        worker_thread.join()
    except Exception as e:
        print(f"\n[!] Sniffer Error: {e}")