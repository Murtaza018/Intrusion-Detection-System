

import os
import pickle
import time
from scapy.all import sniff
from feature_extractor import FeatureExtractor

# --- CONFIGURATION ---
# Update these paths to match your system
PCAPS_TO_PROCESS = {
    "Wednesday-workingHours.pcap": 3000,
    "Thursday-WorkingHours.pcap": 3000
}
PCAP_DIR = "D:/CIC-IDS-PCAP/" 
SAVE_PATH = "./training_data/gnn_snapshots/"
SNAPSHOT_INTERVAL = 1000 
os.makedirs(SAVE_PATH, exist_ok=True)

extractor = FeatureExtractor()

def process_partial():
    for pcap_name, target_snapshots in PCAPS_TO_PROCESS.items():
        pcap_path = os.path.join(PCAP_DIR, pcap_name)
        if not os.path.exists(pcap_path):
            print(f"[!] File not found: {pcap_path}")
            continue

        print(f"\n[*] Targeted Capture: {pcap_name}")
        current_snapshots = 0
        packet_count = 0
        
        # We use a manual loop with PcapReader for better control over large files
        from scapy.all import PcapReader
        try:
            with PcapReader(pcap_path) as reader:
                for pkt in reader:
                    features, flow_key = extractor.extract_features(pkt)
                    packet_count += 1
                    
                    if packet_count % SNAPSHOT_INTERVAL == 0:
                        edge_index, edge_attr = extractor.graph_builder.get_graph_data()
                        
                        if edge_index is not None:
                            timestamp = int(time.time())
                            filename = f"{SAVE_PATH}partial_{pcap_name[:3]}_{packet_count}.pkl"
                            
                            snapshot = {
                                "edge_index": edge_index,
                                "edge_attr": edge_attr,
                                "node_count": extractor.graph_builder.id_counter
                            }
                            
                            with open(filename, 'wb') as f:
                                pickle.dump(snapshot, f)
                            
                            current_snapshots += 1
                            print(f"    [+] Progress: {current_snapshots}/{target_snapshots} snapshots", end="\r")
                            
                            if current_snapshots >= target_snapshots:
                                print(f"\n[+] Target reached for {pcap_name}.")
                                break
        except Exception as e:
            print(f"\n[!] Error: {e}")

if __name__ == "__main__":
    process_partial()