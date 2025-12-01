# main.py
# Main entry point for the IDS pipeline

import sys
import os
import signal
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASE_DIR
from model_loader import ModelLoader
from feature_extractor import FeatureExtractor
from xai_explainer import XAIExplainer
from packet_storage import PacketStorage
from detector import Detector
from api_server import APIServer
from pipeline_manager import PipelineManager

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🔐 INTRUSION DETECTION SYSTEM - PRODUCTION READY")
    print("="*60)
    print(f"📁 Base Directory: {BASE_DIR}")
    
    # Initialize components
    model_loader = ModelLoader()
    feature_extractor = FeatureExtractor()
    xai_explainer = XAIExplainer()
    packet_storage = PacketStorage(max_size=50)
    
    # Create detector
    detector = Detector(model_loader, feature_extractor, xai_explainer, packet_storage)
    
    # Create pipeline manager
    pipeline_manager = PipelineManager(
        model_loader=model_loader,
        feature_extractor=feature_extractor,
        xai_explainer=xai_explainer,
        packet_storage=packet_storage,
        detector=detector,
        api_server=None  # Will be set after creation
    )
    
    # Create API server
    api_server = APIServer(packet_storage, pipeline_manager)
    
    # Update pipeline manager with API server
    pipeline_manager.api_server = api_server
    
    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print(f"\n[!] Received shutdown signal...")
        pipeline_manager.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n📋 COMPONENTS INITIALIZED:")
    print("  ✅ Model Loader")
    print("  ✅ Feature Extractor")
    print("  ✅ XAI Explainer")
    print("  ✅ Packet Storage")
    print("  ✅ Detector")
    print("  ✅ Pipeline Manager")
    print("  ✅ API Server")
    print("\n⚠️  IMPORTANT: Pipeline will start when Flutter app sends start command")
    print("="*60)
    
    # --- REMOVED AUTOMATIC START ---
    # pipeline_manager.start()
    # -------------------------------
    
    # Start sniffing (packets will be processed when pipeline is running)
    # This is okay to start now, as it checks self.running
    pipeline_manager.start_sniffing()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Shutting down...")
        pipeline_manager.stop()

if __name__ == "__main__":
    main()