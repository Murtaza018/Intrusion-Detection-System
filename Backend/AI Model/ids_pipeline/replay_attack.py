from scapy.all import *
import time

pcap_file = "2021-10-01-TR-Qakbot-infection-wtih-spambot-acitvity.pcap" 
# Make sure it looks exactly like this line:
target_interface = r"\Device\NPF_{98FACC85-B69E-4177-BBBF-0F143020C5D2}"

print(f"[*] Loading packets from {pcap_file}...")
packets = rdpcap(pcap_file)
print(f"[*] Loaded {len(packets)} packets.")

print(f"[*] Starting replay on {target_interface}...")
# We modify destination MAC to broadcast so your local OS stack picks it up for sniffing
for pkt in packets:
    if Ether in pkt:
        pkt[Ether].dst = "ff:ff:ff:ff:ff:ff"
    # Send packet at layer 2
    sendp(pkt, iface=target_interface, verbose=0)
 
print("[*] Replay finished.")