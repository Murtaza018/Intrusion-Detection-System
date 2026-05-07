from scapy.all import send, IP, TCP

# Replace with the exact Windows interface string you used in your sniffer
IFACE = "VMware Network Adapter VMnet2" 

print("[*] Simulating compromised DMZ Web-Server pivoting to Internal PC...")

# Spoof the Web-Server IP, attacking the Inside PC on port 445 (SMB)
malicious_packet = IP(src="192.168.50.10", dst="192.168.10.10")/TCP(dport=445, flags="S")

# Blast it to create an anomalous flow
send(malicious_packet, iface=IFACE, loop=1, inter=0.01)