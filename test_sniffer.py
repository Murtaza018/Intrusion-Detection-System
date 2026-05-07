from scapy.all import sniff, IP

# Replace this with the exact Windows interface name or Index from Step 1
IFACE = "VMware Virtual Ethernet Adapter for VMnet2" 

def process_packet(packet):
    # We only care about IP packets for the frontend dashboard
    if IP in packet:
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        
        # This is exactly what you will eventually send to your React/Material UI frontend
        print(f"[LIVE TRAFFIC] Source: {src_ip} --> Destination: {dst_ip}")

print(f"[*] Starting raw sniffer on {IFACE}... Waiting for packets.")
# Sniff continuously, don't store in memory (prevents the 18GB RAM crash)
sniff(iface=IFACE, prn=process_packet, store=False)