# trigger_real_zeroday.py
from scapy.all import send, IP, TCP, Raw, UDP
import time

# Target an external IP to ensure it goes through the interface
TARGET_IP = "8.8.8.8" 

print(f"[*] Crafting Stealthy Zero-Day packet for {TARGET_IP}...")

# Strategy: Send a packet that looks "technically" valid (so CNN ignores it)
# but has statistical anomalies in the payload size/content (so Autoencoder catches it).

# Attempt 1: UDP packet to high port with unusual but low-entropy payload
# CNNs often ignore high UDP ports unless they match specific DDOS patterns.
# The Autoencoder might flag the unusual payload length/content relationship.
packet1 = IP(dst=TARGET_IP)/UDP(dport=55555, sport=44444)/Raw(load="A"*500 + "1"*100)

print(f"[*] Sending Packet 1 (UDP High Port)...")
send(packet1, verbose=0)
time.sleep(1)

# Attempt 2: TCP packet with slightly wrong flags and weird window size
# This mimics a subtle protocol anomaly.
packet2 = IP(dst=TARGET_IP)/TCP(dport=80, sport=44445, flags="FPU", window=1234)/Raw(load="GET / HTTP/1.1\r\nHost: google.com\r\n\r\n" + "X"*200)

print(f"[*] Sending Packet 2 (TCP Weird Flags)...")
send(packet2, verbose=0)
time.sleep(1)

# Attempt 3: "Slow" anomaly - Standard HTTP but massive header
# This looks like normal HTTP to a simple classifier, but the length distribution is off.
payload = "GET / HTTP/1.1\r\n" + "X-Custom-Header: " + "A"*800 + "\r\n\r\n"
packet3 = IP(dst=TARGET_IP)/TCP(dport=80, sport=44446, flags="PA")/Raw(load=payload)

print(f"[*] Sending Packet 3 (HTTP Large Header)...")
send(packet3, verbose=0)

print(f"[+] Packets sent! Check backend for [?!?] ZERO-DAY detection.")