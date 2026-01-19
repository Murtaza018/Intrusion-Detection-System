import numpy as np
from scapy.all import IP, TCP, Ether, conf
from feature_extractor import FeatureExtractor

# Suppress the MAC address warnings to keep the console clean
conf.verb = 0 

def verify_phase1():
    print("[*] Starting Phase 1 Integration Test...")
    extractor = FeatureExtractor()
    
    # Simulating 15 packets to ensure we exceed any internal buffers
    attacker_ip = "192.168.1.10"
    targets = [f"192.168.1.{i}" for i in range(101, 116)] 
    
    print(f"[*] Simulating scan from {attacker_ip} to {len(targets)} targets...")
    
    for target in targets:
        pkt = Ether()/IP(src=attacker_ip, dst=target)/TCP(dport=80, flags="S")
        extractor.extract_features(pkt)
        
    gb = extractor.graph_builder
    edge_index, edge_attr = gb.get_graph_data()
    
    print("-" * 30)
    print(f"[TEST] Node Count: {gb.id_counter}")
    
    if edge_attr is not None:
        print(f"[TEST] Edge Count: {len(edge_attr)}")
        
        # Check Topology
        attacker_id = gb.ip_to_id[attacker_ip]
        sources = edge_index[0]
        success = all(s == attacker_id for s in sources)
        
        if success:
            print("[+] SUCCESS: GraphBuilder correctly mapped the scanning topology!")
        else:
            print("[!] FAILED: Source mapping is incorrect.")
    else:
        print("[!] FAILED: get_graph_data() returned None. Check your buffer thresholds.")

if __name__ == "__main__":
    verify_phase1()