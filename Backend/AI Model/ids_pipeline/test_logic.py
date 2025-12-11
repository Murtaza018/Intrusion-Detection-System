from scapy.all import rdpcap, send, IP, TCP, UDP, conf
import os

PCAP_FILE = "D:/Wednesday-WorkingHours.pcap"
TARGET_IP = "8.8.8.8" # Route to internet so adapter accepts it

if not os.path.exists(PCAP_FILE):
    print(f"[!] Error: {PCAP_FILE} not found.")
    exit()

print(f"[*] Reading {PCAP_FILE}...")
packets = rdpcap(PCAP_FILE, count=300) # Load 5000 packets
print(f"[+] Loaded {len(packets)} packets. Preparing to blast...")

clean_packets = []
for pkt in packets:
    if IP in pkt:
        ip_pkt = pkt[IP]
        ip_pkt.dst = TARGET_IP
        
        # Clear checksums
        del ip_pkt.chksum
        if TCP in ip_pkt: del ip_pkt[TCP].chksum
        if UDP in ip_pkt: del ip_pkt[UDP].chksum
        
        clean_packets.append(ip_pkt)

print(f"[*] Sending {len(clean_packets)} packets via Layer 3...")
send(clean_packets, verbose=1, inter=0)
print("[+] Done!")