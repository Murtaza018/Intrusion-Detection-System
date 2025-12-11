from scapy.all import send, IP, TCP, Raw, UDP
import time

TARGET_IP = "8.8.8.8" 

print(f"Crafting stealthy anomaly packet for {TARGET_IP}...")

packet1 = IP(dst=TARGET_IP)/UDP(dport=55555, sport=44444)/Raw(load="A"*500 + "1"*100)

print(f"Sending Packet 1 (UDP High Port)...")
send(packet1, verbose=0)
time.sleep(1)

packet2 = IP(dst=TARGET_IP)/TCP(dport=80, sport=44445, flags="FPU", window=1234)/Raw(load="GET / HTTP/1.1\r\nHost: google.com\r\n\r\n" + "X"*200)

print(f"Sending Packet 2 (TCP Weird Flags)...")
send(packet2, verbose=0)
time.sleep(1)

payload = "GET / HTTP/1.1\r\n" + "X-Custom-Header: " + "A"*800 + "\r\n\r\n"
packet3 = IP(dst=TARGET_IP)/TCP(dport=80, sport=44446, flags="PA")/Raw(load=payload)

print(f"Sending Packet 3 (HTTP Large Header)...")
send(packet3, verbose=0)

print(f"Packets sent! Check detection logs.")