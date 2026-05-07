from scapy.all import sendp, Ether, IP, TCP

# Ensure this matches your sniffer interface exactly
IFACE = "VMware Network Adapter VMnet2" 

print("[*] Simulating compromised DMZ Web-Server pivoting to Internal PC...")

# We wrap the IP packet in a raw Ethernet() frame. 
# This bypasses the Windows kernel routing table and forces the packet onto the wire.
malicious_frame = Ether()/IP(src="192.168.50.10", dst="192.168.10.10")/TCP(dport=445, flags="S")

print("[*] Launching 500-packet Layer-2 SYN burst...")

# sendp() operates at Layer 2. It will not be blocked by Windows.
sendp(malicious_frame, iface=IFACE, count=500, inter=0.001, verbose=False)

print("[*] Burst complete. Check the React Dashboard.")