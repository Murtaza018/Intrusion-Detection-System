
import sys
import os
import signal
import time
import threading

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASE_DIR, FLASK_HOST, FLASK_PORT, NETWORK_INTERFACE
from model_loader import ModelLoader
from feature_extractor import FeatureExtractor
from xai_explainer import XAIExplainer
from packet_storage import PacketStorage
from detector import Detector
from api_server import APIServer
from pipeline_manager import PipelineManager

import firebase_admin
from firebase_admin import credentials, messaging

CERT_PATH = "service-account.json" 

def initialize_firebase():
    try:
        # 2. Check if already initialized to avoid "app already exists" error
        if not firebase_admin._apps:
            cred = credentials.Certificate(CERT_PATH)
            firebase_admin.initialize_app(cred)
            print("[+] Firebase Admin SDK initialized successfully.")
    except Exception as e:
        print(f"[!] Firebase Initialization Error: {e}")

# 3. Call this function before starting the pipeline


def main():
    """Main function"""
    print("\n" + "="*60)
    print("🔐 INTRUSION DETECTION SYSTEM - PRODUCTION READY")
    print("="*60)
    print(f"📁 Base Directory: {BASE_DIR}")

    
    # 1. Initialize Core 
    initialize_firebase()
    model_loader = ModelLoader()
    feature_extractor = FeatureExtractor()
    xai_explainer = XAIExplainer()
    packet_storage = PacketStorage(max_size=100000)
    
    # 2. Create Detector
    detector = Detector(model_loader, feature_extractor, xai_explainer, packet_storage)
    
    # 3. Create Pipeline Manager
    pipeline_manager = PipelineManager(
        model_loader=model_loader,
        feature_extractor=feature_extractor,
        xai_explainer=xai_explainer,
        packet_storage=packet_storage,
        detector=detector,
        api_server=None,  # Will be set next
        interface=NETWORK_INTERFACE
    )
    
    # 4. Create API Server (CORRECTED LINE)
    # Passing all 4 dependencies required for GAN and Jitter retraining
    api_server = APIServer(packet_storage, feature_extractor, pipeline_manager, model_loader)
    
    # 5. Link API Server back to Pipeline Manager
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
    print("  ✅ API Server (GAN Ready)")
    print("\n⚠️  IMPORTANT: Pipeline will start when Flutter app sends start command")
    print("="*60)
    
    # Start Flask server explicitly in a daemon thread
    flask_thread = threading.Thread(
        target=api_server.run,
        args=(FLASK_HOST, FLASK_PORT),
        daemon=True,
        name="Flask_Server"
    )
    flask_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Shutting down...")
        pipeline_manager.stop()

if __name__ == "__main__":
    main()