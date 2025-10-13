# ids_core.py
# A foundational Intrusion Detection System (IDS) core using Scapy.
# This script sniffs network traffic and applies basic detection rules.

# Make sure you have Scapy installed:
# pip install scapy

import sys
from datetime import datetime
from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw, get_if_list

# --- Configuration ---
# You can add IP addresses to a whitelist to prevent them from triggering alerts.
IP_WHITELIST = ["127.0.0.1", "localhost"]
PACKET_COUNT = 0

# --- SQL Injection Signatures ---
# A list of common, basic SQL injection patterns to look for.
SQL_INJECTION_PATTERNS = [
    "' OR '1'='1'",
    "UNION SELECT",
    "DROP TABLE",
    "--",
    "';"
]


def log_alert(message):
    """
    Prints a formatted alert message with a timestamp.
    In a real application, this function would send alerts to a dashboard,
    log to a file, or send a notification.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n🚨 ALERT [{timestamp}]: {message}")


def packet_analysis_engine(packet):
    """
    This is the heart of the IDS. It's called for every packet captured.
    It inspects the packet and applies various detection rules.
    """
    global PACKET_COUNT
    PACKET_COUNT += 1
    if PACKET_COUNT % 200 == 0:
        # Print a status update every 200 packets to show the IDS is alive.
        print(f"[*] {PACKET_COUNT} packets captured...", end='\r')

    try:
        # --- Rule 1: IP Layer Analysis ---
        if packet.haslayer(IP):
            source_ip = packet[IP].src
            dest_ip = packet[IP].dst

            # Skip packets from whitelisted IPs
            if source_ip in IP_WHITELIST or dest_ip in IP_WHITELIST:
                return

            # --- Rule 2: TCP Protocol Analysis ---
            if packet.haslayer(TCP):
                source_port = packet[TCP].sport
                dest_port = packet[TCP].dport
                flags = packet[TCP].flags

                # Signature Rule: Detect Telnet connection attempt (unencrypted, insecure)
                if dest_port == 23:
                    log_alert(f"Insecure Telnet connection detected from {source_ip}:{source_port} to {dest_ip}:{dest_port}")

                # Signature Rule: Detect potential Nmap Xmas Scan
                # This scan sets FIN, PSH, and URG flags, which is highly unusual.
                if flags == 0x29:  # FIN/PSH/URG flags
                    log_alert(f"Possible Nmap Xmas Scan detected from {source_ip} to {dest_ip}:{dest_port}")

                # --- NEW RULE: Basic SQL Injection Detection in HTTP ---
                if dest_port == 80 and packet.haslayer(Raw):
                    payload = packet[Raw].load.decode('utf-8', errors='ignore').upper()
                    for pattern in SQL_INJECTION_PATTERNS:
                        if pattern in payload:
                            log_alert(f"Possible SQL Injection attempt detected from {source_ip} to {dest_ip}. Pattern: {pattern}")
                            # Break after first match to avoid multiple alerts for one packet
                            break
            
            # --- Rule 3: UDP Protocol Analysis ---
            elif packet.haslayer(UDP):
                source_port = packet[UDP].sport
                dest_port = packet[UDP].dport
                
                # Signature Rule: Detect potential DNS Zone Transfer attempt (AXFR)
                # While often legitimate, unsolicited zone transfers can leak network info.
                if dest_port == 53 and packet.haslayer(Raw):
                    payload = bytes(packet[Raw].load).lower()
                    if b'axfr' in payload:
                        log_alert(f"Potential DNS Zone Transfer attempt from {source_ip} to {dest_ip}")


            # --- Rule 4: ICMP Protocol Analysis (Ping) ---
            elif packet.haslayer(ICMP):
                # Anomaly Rule: Detect "Ping of Death" attempt (oversized ICMP packet)
                # A normal ping packet is small. A very large one can be malicious.
                if len(packet[ICIP].payload) > 1024:
                    log_alert(f"Oversized ICMP (Ping) packet detected from {source_ip} to {dest_ip}. Potential Ping of Death.")

    except Exception as e:
        # In a production environment, you would log errors more robustly.
        # print(f"Error processing packet: {e}")
        pass


def start_monitoring(interface):
    """
    Starts the continuous packet sniffing process on a given network interface.
    """
    if not interface:
        print("[!] Error: No network interface provided. Exiting.")
        sys.exit(1)

    print(f"[*] Starting IDS monitoring on interface: {interface}")
    print("[*] Press Ctrl+C to stop.")

    # The sniff() function runs continuously.
    # 'prn' specifies the callback function to run on each packet.
    # 'store=0' prevents Scapy from storing all packets in memory, which is crucial for continuous monitoring.
    # 'iface' specifies the network interface to sniff on.
    try:
        sniff(iface=interface, prn=packet_analysis_engine, store=0)
    except OSError as e:
        print(f"\n[!] Error starting sniffer: {e}")
        print("[!] Please make sure you are running this script with root/administrator privileges.")
        print(f"[!] Also, ensure the interface name '{interface}' is correct.")
    except Exception as e:
        print(f"\n[!] An unexpected error occurred: {e}")


if __name__ == "__main__":
    print("--- Simple Python IDS with Scapy ---")
    
    # IMPORTANT: You must run this script with root/administrator privileges
    # for it to access network packets. (e.g., `sudo python ids_core.py`)
    
    # List available network interfaces to help the user
    print("\nAvailable network interfaces:")
    try:
        if sys.platform == "win32":
            import winreg
            import re

            guid_to_friendly_name = {}
            reg_path = r'SYSTEM\CurrentControlSet\Control\Network\{4d36e972-e325-11ce-bfc1-08002be10318}'
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as adapters_key:
                    for i in range(winreg.QueryInfoKey(adapters_key)[0]):
                        guid_with_braces = winreg.EnumKey(adapters_key, i)
                        try:
                            connection_reg_path = fr'{reg_path}\{guid_with_braces}\Connection'
                            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, connection_reg_path) as key:
                                friendly_name, _ = winreg.QueryValueEx(key, 'Name')
                                guid_without_braces = guid_with_braces.strip('{}')
                                guid_to_friendly_name[guid_without_braces.upper()] = friendly_name
                        except FileNotFoundError:
                            continue
                        except Exception:
                            continue
            except Exception as e:
                 print(f"[!] Could not query Windows Registry. Error: {e}")

            interface_map = {}
            scapy_interfaces = get_if_list()

            for scapy_name in scapy_interfaces:
                guid_match = re.search(r'\{(.+?)\}', scapy_name)
                if not guid_match:
                    interface_map[scapy_name] = scapy_name
                    continue
                
                guid_from_scapy = guid_match.group(1).upper()
                friendly_name = guid_to_friendly_name.get(guid_from_scapy, scapy_name)
                interface_map[scapy_name] = friendly_name
            
            print("-----------------------------------------------------------------")
            for scapy_name, friendly_name in interface_map.items():
                print(f"  Friendly Name: {friendly_name}")
                print(f"  Name to use:   {scapy_name}")
                print("-----------------------------------------------------------------")

        else:
            # For Linux/macOS
            interfaces = get_if_list()
            for iface in interfaces:
                print(f"- {iface}")

    except Exception as e:
        print(f"\n[!] An unexpected error occurred during interface discovery: {e}")

    
    network_interface = input("\nCopy and paste the 'Name to use' from the list above: ")
    start_monitoring(network_interface)

