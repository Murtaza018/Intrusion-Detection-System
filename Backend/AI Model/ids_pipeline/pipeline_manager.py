# pipeline_manager.py
# Pipeline coordination and management

import threading
from scapy.all import sniff
from scapy.config import conf
from datetime import datetime
import sys

from config import FLASK_HOST, FLASK_PORT

class PipelineManager:
    """Manage the complete IDS pipeline"""
    
    # Update __init__ to accept interface
    def __init__(self, model_loader, feature_extractor, xai_explainer, 
                 packet_storage, detector, api_server, interface=None):
        self.model_loader = model_loader
        self.feature_extractor = feature_extractor
        self.xai_explainer = xai_explainer
        self.packet_storage = packet_storage
        self.detector = detector
        self.api_server = api_server
        self.interface = interface # Store the interface
        
        self.running = False
        self.flask_thread = None
        self.sniffing = False
        
    def start(self):
        """Start the complete pipeline"""
        if self.running:
            return False
        
        print("\n" + "="*60)
        print("🚀 STARTING IDS PIPELINE")
        print("="*60)
        
        # Load models
        if not self.model_loader.load_models():
            print("[!] Failed to load models")
            return False
        
        # NOTE: Flask server is already started in main.py
        
        # Start detector
        self.detector.start()
        
        # Update storage stats
        self.packet_storage.update_stats({
            "pipeline_status": "running",
            "start_time": datetime.now().isoformat()
        })
        
        self.running = True # Set running to True BEFORE starting sniffing
        
        # Start sniffing
        self.start_sniffing(self.interface)
        
        print("[+] ✅ Pipeline started successfully")
        print("="*60)
        
        return True
    
    def stop(self):
        """Stop the pipeline"""
        if not self.running:
            return
        
        print("\n[!] Stopping pipeline...")
        
        # Stop detector
        self.detector.stop()
        
        # Update storage stats
        self.packet_storage.update_stats({
            "pipeline_status": "stopped"
        })
        
        self.running = False
        # Sniffing thread will stop automatically because it checks self.running
        print("[+] Pipeline stopped")
    
    def is_running(self):
        """Check if pipeline is running"""
        return self.running
    
    def start_sniffing(self, interface=None):
        """Start packet sniffing"""
        # This check is now redundant if called from start(), but good for safety
        if not self.running:
            print("[!] Cannot start sniffing - pipeline not running")
            return
        
        print(f"[*] Starting packet capture...")
        
        # Get interface (use stored one if argument is None)
        if interface is None:
            interface = self.interface

        if interface is None:
            interface = conf.iface
            print(f"[*] Using default interface: {interface}")
        else:
            print(f"[*] Using configured interface: {interface}")
        
        # Packet handler
        def packet_handler(packet):
            if self.running:
                # print(".", end="", flush=True) # Uncomment to see every packet
                self.detector.process_packet(packet)
        
        # Start sniffing in background thread
        self.sniffing = True
        sniff_thread = threading.Thread(
            target=self._sniff_loop,
            args=(packet_handler, interface),
            daemon=True,
            name="Sniffer"
        )
        sniff_thread.start()
    
    def _sniff_loop(self, packet_handler, interface):
        """Sniffing loop"""
        try:
            print(f"[DEBUG] _sniff_loop started. Interface argument: {interface}")
            sys.stdout.flush()

            if interface:
                conf.iface = interface
                print(f"[DEBUG] Scapy conf.iface set to: {conf.iface}")
            else:
                print("[DEBUG] No interface provided, using current conf.iface")
            
            sys.stdout.flush()

            sniff_kwargs = {
                "prn": packet_handler,
                "store": 0,
                "filter": "ip"
            }
            
            print(f"[DEBUG] Calling sniff() with conf.iface={conf.iface}")
            sys.stdout.flush()
            
            sniff(**sniff_kwargs)
            
            print("[DEBUG] sniff() finished (unexpected)")
            sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n[!] Sniffing stopped by user")
        except Exception as e:
            print(f"\n[!] Sniffing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.sniffing = False
            print("[DEBUG] _sniff_loop finishing")
            sys.stdout.flush()