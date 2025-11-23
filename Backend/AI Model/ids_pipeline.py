# ids_main_pipeline.py
# The core engine of the IDS. It connects Scapy sniffing, AI Detection,
# XAI Explanation, and Secure Reporting into one loop.

import sys
import os
import numpy as np
import requests
import json
from datetime import datetime
from scapy.all import sniff, IP, TCP, UDP

# ----- Path Configuration -----
# Ensure we can import from our AI Model folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_MODEL_DIR = os.path.join(BASE_DIR, "AI Model")
XAI_DIR = os.path.join(AI_MODEL_DIR, "XAI")

sys.path.append(AI_MODEL_DIR)
sys.path.append(XAI_DIR)

# Import our modules
try:
    from inference import predict
    from explanation_inference import explain_alert
except ImportError as e:
    print(f"[!] Error importing AI modules: {e}")
    print(f"    Ensure 'inference.py' is in '{AI_MODEL_DIR}'")
    print(f"    Ensure 'explanation_inference.py' is in '{XAI_DIR}'")
    sys.exit(1)

# ----- Config -----
BACKEND_URL = "https://127.0.0.1:5000/api/alerts"
API_KEY = "MySuperSecretKey12345!"
INTERFACE = "Wi-Fi" # Change this to your actual interface name (from get_if_list)

# Feature list size for CIC-IDS-2017
NUM_FEATURES = 78 

def extract_features(packet):
    """
    Converts a raw Scapy packet into a 78-feature numpy array.
    NOTE: This is a simplified extractor for demonstration.
    Real CIC-IDS-2017 features require tracking 'Flows' over time.
    """
    features = np.zeros(NUM_FEATURES)
    
    if packet.haslayer(IP):
        # Basic extraction logic (simplified mapping)
        # In a real deployment, use cicflowmeter logic here.
        
        # Example mapping (indices based on typical CIC-IDS columns):
        # 0: Destination Port
        if packet.haslayer(TCP):
            features[0] = packet[TCP].dport
        elif packet.haslayer(UDP):
            features[0] = packet[UDP].dport
            
        # 1: Flow Duration (Placeholder: random small jitter for realism)
        features[1] = np.random.randint(10, 1000) 
        
        # 2: Total Fwd Packets (We see 1 packet, so 1)
        features[2] = 1
        
        # 3: Total Length of Fwd Packets
        features[4] = len(packet)
        
        # ... (Other 70+ statistical features would be calculated here) ...
        
    return features

def send_alert(alert_data):
    """
    Sends the alert JSON to the secure backend API.
    """
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    try:
        # verify=False is needed because we are using a self-signed cert locally.
        # In production, you would remove verify=False.
        response = requests.post(BACKEND_URL, json=alert_data, headers=headers, verify=False)
        if response.status_code == 200:
            print(f"[+] Alert sent successfully!")
        else:
            print(f"[!] Failed to send alert. Status: {response.status_code}")
    except Exception as e:
        print(f"[!] Connection error sending alert: {e}")

def process_packet(packet):
    """
    The main logic loop for every packet.
    1. Extract Features -> 2. Frontline Check -> 3. Zero-Day Check -> 4. XAI -> 5. Report
    """
    if not packet.haslayer(IP):
        return

    # 1. Extract Features
    # Reshape to (1, 78) for the model
    features = extract_features(packet) 
    
    # 2. Frontline Defense (Hardened CNN+LSTM)
    frontline_result = predict(features, "hardened_classifier")
    
    alert_data = None
    
    if frontline_result["label"] == "Attack":
        print(f"\n[!!!] KNOWN ATTACK DETECTED! Score: {frontline_result['score']:.4f}")
        
        # 4. XAI Explanation
        print("      Generating explanation...")
        explanation = explain_alert(features, "hardened_classifier", attack_type="Known Attack")
        
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
        # 3. Zero-Day Hunter (Autoencoder)
        # Only runs if Frontline says "Normal"
        hunter_result = predict(features, "zero_day_hunter")
        
        if hunter_result["label"] != "Normal":
            print(f"\n[?!?] ZERO-DAY ANOMALY DETECTED! Error: {hunter_result['score']:.6f}")
            
            # 4. XAI Explanation (Why is it anomalous?)
            print("      Generating explanation...")
            explanation = explain_alert(features, "zero_day_hunter", attack_type="Zero-Day Anomaly")
            
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

    # 5. Send Alert (if any)
    if alert_data:
        send_alert(alert_data)
    else:
        # Optional: Print a dot for normal traffic just to show it's working
        print(".", end="", flush=True)

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Intelligent IDS Pipeline Started ---")
    print(f"[*] Frontline Model: Hardened CNN+LSTM")
    print(f"[*] Zero-Day Hunter: Autoencoder")
    print(f"[*] XAI Engine: Active")
    print(f"[*] Secure Backend: {BACKEND_URL}")
    print(f"[*] Listening on: {INTERFACE}")
    print("----------------------------------------")
    
    # Start Sniffing
    try:
        # store=0 prevents memory leaks by not keeping packets in RAM
        sniff(iface=INTERFACE, prn=process_packet, store=0)
    except KeyboardInterrupt:
        print("\n[!] Stopping IDS...")
    except Exception as e:
        print(f"\n[!] Error: {e}")