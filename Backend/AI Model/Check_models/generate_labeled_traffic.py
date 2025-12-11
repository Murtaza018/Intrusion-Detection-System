import sys
import csv
from scapy.all import IP, TCP, UDP, Raw, Ether, wrpcap

# --- Configuration ---
PCAP_OUTPUT_FILE = "synthetic_attacks.pcap"
CSV_OUTPUT_FILE = "synthetic_labels.csv"
PACKET_COUNT = 1000 

# Attack types and parameters
ATTACKS = {
    "NORMAL": {"count": 700, "label": "BENIGN", "attack_type": "Normal"},
    "PortScan": {"count": 100, "label": "MALICIOUS", "attack_type": "PortScan"},
    "SSH_BruteForce": {"count": 50, "label": "MALICIOUS", "attack_type": "SSH Brute Force"},
    "SQLi": {"count": 50, "label": "MALICIOUS", "attack_type": "SQL Injection"},
    "XSS": {"count": 50, "label": "MALICIOUS", "attack_type": "XSS"},
    "DDoS_UDP": {"count": 50, "label": "MALICIOUS", "attack_type": "DDoS UDP Flood"},
}
# --- End Configuration ---

def create_packet(src_ip, dst_ip, src_port, dst_port, payload=None, protocol=TCP, tcp_flags="S"):
    # Creates a basic IP/TCP or IP/UDP packet.
    layer3 = IP(src=src_ip, dst=dst_ip)
    
    if protocol == TCP:
        layer4 = TCP(sport=src_port, dport=dst_port, flags=tcp_flags) 
    elif protocol == UDP:
        layer4 = UDP(sport=src_port, dport=dst_port)
    else:
        return None

    if payload:
        return Ether()/layer3/layer4/Raw(load=payload)
    else:
        return Ether()/layer3/layer4

def generate_traffic(attack_type, params, packet_index_start):
    # Generates packets and labels for a specific attack type.
    packets = []
    labels = []
    
    current_index = packet_index_start

    # Common source/destination.
    src_ip = "192.168.1.10"
    dst_ip = "10.0.0.5"

    for i in range(params["count"]):
        
        # --- Packet Generation Logic ---
        if attack_type == "NORMAL":
            pkt = create_packet(src_ip, dst_ip, 50000 + i, 80, payload=f"Normal Data {i}")
        
        elif attack_type == "PortScan":
            # Rapidly changing destination port.
            pkt = create_packet(src_ip, dst_ip, 50001, 1000 + i % 100) 
        
        elif attack_type == "SSH_BruteForce":
            # Repeated attempts to port 22.
            pkt = create_packet(src_ip, dst_ip, 50000 + i, 22, payload=f"ssh attempt {i}")
        
        elif attack_type == "SQLi":
            # HTTP-like traffic with suspicious payload.
            sql_payload = f"GET /index.php?id={i}' OR 1=1 --" 
            pkt = create_packet(src_ip, dst_ip, 50000 + i, 80, payload=sql_payload)
            
        elif attack_type == "XSS":
            # HTTP-like traffic with XSS payload.
            xss_payload = f"GET /search.php?q=<script>alert({i})</script>"
            pkt = create_packet(src_ip, dst_ip, 50000 + i, 80, payload=xss_payload)
            
        elif attack_type == "DDoS_UDP":
            # High volume UDP traffic.
            pkt = create_packet(src_ip, dst_ip, 50000 + i, 53, payload=f"Junk data {i}", protocol=UDP)

        else:
            continue
        
        if pkt: 
            packets.append(pkt)
            
            # The protocol field from the IP header.
            protocol_number = pkt[IP].proto 
            
            labels.append({
                "packet_index": current_index,
                "label": params["label"],
                "attack_type": params["attack_type"],
                "protocol": protocol_number, 
                "src_ip": src_ip,
                "dst_ip": dst_ip
            })
            current_index += 1

    return packets, labels, current_index

def main():
    all_packets = []
    all_labels = []
    current_packet_index = 1
    
    # 1. Generate Packets and Labels
    print("Generating traffic...")
    for attack, params in ATTACKS.items():
        print(f"-> Generating {params['count']} packets for {attack}...")
        
        new_packets, new_labels, next_index = generate_traffic(attack, params, current_packet_index)
        
        all_packets.extend(new_packets)
        all_labels.extend(new_labels)
        current_packet_index = next_index
        
    print(f"\nTotal packets generated: {len(all_packets)}")
    
    # 2. Write PCAP file
    print(f"Writing PCAP file to {PCAP_OUTPUT_FILE}...")
    wrpcap(PCAP_OUTPUT_FILE, all_packets)
    print("PCAP file written.")

    # 3. Write CSV file
    print(f"Writing CSV file to {CSV_OUTPUT_FILE}...")
    fieldnames = ["packet_index", "label", "attack_type", "protocol", "src_ip", "dst_ip"]
    with open(CSV_OUTPUT_FILE, 'w', newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_labels)
        
    print("CSV file written.")
    
    # 4. Success Message
    print("\nGeneration Complete!")

if __name__ == "__main__":
    main()