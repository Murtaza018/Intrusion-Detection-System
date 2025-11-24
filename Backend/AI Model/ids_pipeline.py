# ids_main_pipeline.py
# Threaded Intelligent IDS with Frontline Classifier and Autoencoder Zero-Day Hunter

import sys
import os
import time
import random
import threading
import queue
from collections import defaultdict
from datetime import datetime

import numpy as np
import requests
from scapy.all import sniff, IP, TCP, UDP

# ----- Path Configuration -----
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

# ----- Config -----
BACKEND_URL = "https://127.0.0.1:5000/api/alerts"
API_KEY = "MySuperSecretKey12345!"
from scapy.config import conf
INTERFACE = conf.iface
NUM_FEATURES = 78
FLOW_TIMEOUT = 300  # seconds to remove old flows

# Thread-safe packet queue
packet_queue = queue.Queue()

# Flow tracking: key = (src_ip, dst_ip, sport, dport, proto)
flows = defaultdict(lambda: {
    "packets": [],
    "timestamps": [],
    "lengths": [],
    "flags": []
})

# ---------------- Feature Extraction ----------------
def extract_features_pure_python(packet):
    features = [0.0] * NUM_FEATURES

    try:
        if not packet.haslayer(IP):
            return features

        ip_layer = packet[IP]
        proto = 6 if packet.haslayer(TCP) else 17 if packet.haslayer(UDP) else 0
        sport = getattr(packet, 'sport', 0)
        dport = getattr(packet, 'dport', 0)
        flow_key = (ip_layer.src, ip_layer.dst, sport, dport, proto)

        # Cleanup old flows
        now = time.time()
        for key in list(flows.keys()):
            if flows[key]["timestamps"] and now - flows[key]["timestamps"][-1] > FLOW_TIMEOUT:
                del flows[key]

        # Update flow
        flows[flow_key]["packets"].append(packet)
        flows[flow_key]["timestamps"].append(now)
        flows[flow_key]["lengths"].append(len(packet))
        flags = getattr(packet[TCP], "flags", 0) if proto == 6 else 0
        flows[flow_key]["flags"].append(flags)

        # Packet statistics
        lengths = flows[flow_key]["lengths"]
        pkt_count = len(lengths)
        total_len = sum(lengths)
        mean_len = total_len / pkt_count if pkt_count else 0
        std_len = (sum((l - mean_len) ** 2 for l in lengths) / pkt_count) ** 0.5 if pkt_count else 0

        # --- Fill features ---
        # Destination port
        features[0] = float(dport)
        # Flow duration
        features[1] = float(flows[flow_key]["timestamps"][-1] - flows[flow_key]["timestamps"][0] + 1e-6)
        # Total Fwd/Bwd packets
        features[2] = float(pkt_count)
        features[3] = float(pkt_count)
        # Total length of Fwd/Bwd packets
        features[4] = float(total_len)
        features[5] = float(total_len)
        # Fwd/Bwd packet length max/min/mean/std
        features[6] = max(lengths)
        features[7] = min(lengths)
        features[8] = mean_len
        features[9] = std_len
        features[10] = features[6]
        features[11] = features[7]
        features[12] = features[8]
        features[13] = features[9]
        # Flow bytes/s and packets/s
        dur = features[1]
        features[14] = total_len / dur
        features[15] = pkt_count / dur
        # Flow IAT mean/std/max/min
        iats = [t2 - t1 for t1, t2 in zip(flows[flow_key]["timestamps"][:-1], flows[flow_key]["timestamps"][1:])]
        if iats:
            mean_iat = sum(iats) / len(iats)
            std_iat = (sum((i - mean_iat) ** 2 for i in iats) / len(iats)) ** 0.5
            features[16:20] = [mean_iat, std_iat, max(iats), min(iats)]
        else:
            features[16:20] = [0, 0, 0, 0]

        # Fwd/Bwd IAT placeholders (same as flow IAT)
        features[20:30] = features[16:26][:10] if len(features) >= 30 else [0]*10

        # TCP flags
        last_flags = flags
        features[30] = float(last_flags & 0x08)  # Fwd PSH
        features[31] = float(last_flags & 0x08)  # Bwd PSH
        features[32] = float(last_flags & 0x20)  # Fwd URG
        features[33] = float(last_flags & 0x20)  # Bwd URG
        features[44] = float(last_flags & 0x01)  # FIN
        features[45] = float(last_flags & 0x02)  # SYN
        features[46] = float(last_flags & 0x04)  # RST
        features[47] = float(last_flags & 0x08)  # PSH
        features[48] = float(last_flags & 0x10)  # ACK
        features[49] = float(last_flags & 0x20)  # URG
        features[50] = float(last_flags & 0x40)  # CWR
        features[51] = float(last_flags & 0x80)  # ECE

        # Remaining features: average packet size placeholders
        for i in range(52, NUM_FEATURES):
            features[i] = mean_len

    except Exception as e:
        print(f"[!] Feature extraction warning: {e}")

    return features

# ---------------- Alert Sending ----------------
def send_alert(alert_data):
    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}
    try:
        requests.post(BACKEND_URL, json=alert_data, headers=headers, verify=False, timeout=2)
    except Exception:
        pass

# ---------------- Worker Thread ----------------
def worker_logic():
    print("[*] Processing Worker Started...")
    while True:
        packet = packet_queue.get()
        if packet is None:
            break

        try:
            # 1. Extract features
            features_list = extract_features_pure_python(packet)
            features_np = np.array(features_list, dtype=np.float64)

            # 2. Frontline classifier
            frontline_result = predict(features_np, "hardened_classifier")
            alert_data = None

            if frontline_result["label"] == "Attack":
                print(f"\n[!!!] KNOWN ATTACK DETECTED! Score: {frontline_result['score']:.4f}")
                explanation = explain_alert(features_np, "hardened_classifier", attack_type="Known Attack")
                alert_data = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "Known Attack",
                    "model": "CNN+LSTM",
                    "src_ip": packet[IP].src,
                    "dst_ip": packet[IP].dst,
                    "confidence": float(frontline_result["score"]),
                    "explanation": explanation["explanation"],
                    "facts": explanation["facts"]
                }
            else:
                # 3. Zero-day detection
                hunter_result = predict(features_np, "zero_day_hunter")
                if hunter_result["label"] != "Normal":
                    print(f"\n[?!?] ZERO-DAY ANOMALY DETECTED! Error: {hunter_result['score']:.6f}")
                    explanation = explain_alert(features_np.reshape(1, -1), "zero_day_hunter", attack_type="Zero-Day Anomaly")
                    alert_data = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "Zero-Day Anomaly",
                        "model": "Autoencoder",
                        "src_ip": packet[IP].src,
                        "dst_ip": packet[IP].dst,
                        "reconstruction_error": float(hunter_result["score"]),
                        "explanation": explanation["explanation"],
                        "facts": explanation["facts"]
                    }

            if alert_data:
                send_alert(alert_data)
            else:
                print(".", end="", flush=True)

        except Exception as e:
            print(f"\n[!] Processing Error: {e}")
        finally:
            packet_queue.task_done()

# ---------------- Packet Handler ----------------
def packet_handler(packet):
    if packet.haslayer(IP):
        packet_queue.put(packet)

# ---------------- Main ----------------
if __name__ == "__main__":
    print("--- Intelligent IDS Pipeline Started (Threaded) ---")
    print(f"[*] Frontline Model: Hardened CNN+LSTM")
    print(f"[*] Zero-Day Hunter: Autoencoder")
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
