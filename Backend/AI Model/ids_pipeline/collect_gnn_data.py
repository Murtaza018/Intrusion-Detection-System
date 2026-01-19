import os
import pickle
import time
from scapy.all import sniff
from feature_extractor import FeatureExtractor

# Configuration
SAVE_PATH = "./training_data/gnn_snapshots/"
SNAPSHOT_INTERVAL = 100 # Save a graph every 100 packets
os.makedirs(SAVE_PATH, exist_ok=True)

extractor = FeatureExtractor()
packet_count = 0

def packet_callback(pkt):
    global packet_count
    features, flow_key = extractor.extract_features(pkt)
    packet_count += 1
    
    if packet_count % SNAPSHOT_INTERVAL == 0:
        # Get the current graph state
        edge_index, edge_attr = extractor.graph_builder.get_graph_data()
        
        if edge_index is not None:
            timestamp = int(time.time())
            filename = f"{SAVE_PATH}snapshot_{timestamp}_{packet_count}.pkl"
            
            snapshot = {
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "node_count": extractor.graph_builder.id_counter
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(snapshot, f)
            print(f"[*] Saved Graph Snapshot: {filename} ({len(edge_attr)} edges)")

print("[*] Starting Phase 1 Data Collection...")
sniff(iface="eth0", prn=packet_callback, store=0) # Change eth0 to your interface