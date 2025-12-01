# pipeline_manager.py
# Pipeline coordination and management

import threading
from scapy.all import sniff
from scapy.config import conf
# --- FIX: Import datetime ---
from datetime import datetime
# ----------------------------

from config import FLASK_HOST, FLASK_PORT

class PipelineManager:
    """Manage the complete IDS pipeline"""
    
    def __init__(self, model_loader, feature_extractor, xai_explainer, 
                 packet_storage, detector, api_server):
        self.model_loader = model_loader
        self.feature_extractor = feature_extractor
        self.xai_explainer = xai_explainer
        self.packet_storage = packet_storage
        self.detector = detector
        self.api_server = api_server
        
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
        
        # Start Flask API
        self.flask_thread = threading.Thread(
            target=self.api_server.run,
            args=(FLASK_HOST, FLASK_PORT),
            daemon=True,
            name="Flask_Server"
        )
        self.flask_thread.start()
        
        # Start detector
        self.detector.start()
        
        # Update storage stats
        self.packet_storage.update_stats({
            "pipeline_status": "running",
            "start_time": datetime.now().isoformat()
        })
        
        self.running = True
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
        print("[+] Pipeline stopped")
    
    def is_running(self):
        """Check if pipeline is running"""
        return self.running
    
    def start_sniffing(self, interface=None):
        """Start packet sniffing"""
        if not self.running:
            print("[!] Cannot start sniffing - pipeline not running")
            return
        
        print(f"[*] Starting packet capture...")
        
        # Get interface
        if interface is None:
            interface = conf.iface
        
        # Packet handler
        def packet_handler(packet):
            if self.running:
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
        
        print(f"[+] Sniffing started on interface: {interface}")
    
    def _sniff_loop(self, packet_handler, interface):
        """Sniffing loop"""
        try:
            sniff_kwargs = {
                "prn": packet_handler,
                "store": 0,
                "filter": "ip"
            }
            
            if interface:
                sniff_kwargs["iface"] = interface
            
            sniff(**sniff_kwargs)
        except KeyboardInterrupt:
            print("\n[!] Sniffing stopped by user")
        except Exception as e:
            print(f"\n[!] Sniffing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.sniffing = False