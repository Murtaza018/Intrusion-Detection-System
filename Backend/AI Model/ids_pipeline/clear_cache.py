"""
test_detection_db.py

Use your existing PacketStorage class to:
 - Load features + metadata of 3 specific packets from the DB,
 - Run them through the same detection pipeline as live IDS,
 - Print label + confidence to terminal.
"""

import os
import sys
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your pipeline modules
from model_loader import ModelLoader
from feature_extractor import FeatureExtractor
from xai_explainer import XAIExplainer
from packet_storage import PacketStorage
from detector import Detector


def main():
    print("\n=== Loading models... ===")
    model_loader = ModelLoader()
    if not model_loader.load_models():
        print("[!] Failed to load models. Exiting.")
        return

    feature_extractor = FeatureExtractor()
    xai_explainer = XAIExplainer()
    packet_storage = PacketStorage(max_size=1000)
    detector = Detector(
        model_loader=model_loader,
        feature_extractor=feature_extractor,
        xai_explainer=xai_explainer,
        packet_storage=packet_storage,
    )

    # ------------------------------------------------------------------
    # 1. Choose the 3 zero‑day packet IDs from the DB
    # ------------------------------------------------------------------

    # Replace these with real packet_id_backend values
    test_packet_ids = [
        38929,  # <-- real zero‑day packet ID
        39624,
        39660,
    ]

    # Load features from DB
    raw_features_list = packet_storage.get_features_for_training(test_packet_ids)
    if len(raw_features_list) != 3:
        print(f"[!] Expected 3 packets, got {len(raw_features_list)}.")
        return

    # Convert to numpy array (N, 78)
    X = np.array(raw_features_list, dtype=np.float32)
    print(f"[+] Loaded {len(X)} packets from DB, shape {X.shape}.")

    # Load packet metadata (from DB) for each packet
    metadata_list = [
        packet_storage.get_packet_by_id(pid).to_dict()
        for pid in test_packet_ids
    ]

    # ------------------------------------------------------------------
    # 2. Run each packet through the same detection pipeline,
    #    using REAL packet info from DB
    # ------------------------------------------------------------------

    print("\n=== Running detection on 3 DB packets (with GNN‑context info) ===")
    for i in range(len(X)):
        raw_packet = X[i]  # 78‑dim feature vector from DB
        meta = metadata_list[i]  # packet metadata from DB

        # Use real packet info from database
        extra_info = {
            "src_ip": meta["src_ip"],
            "dst_ip": meta["dst_ip"],
            "protocol": meta["protocol"],
            "src_port": meta["src_port"],
            "dst_port": meta["dst_port"],
        }

        # Call the full‑pipeline method
        result = detector.classify_features_pipeline(raw_packet, extra_info=extra_info)

        label = result["label"]
        confidence = result["confidence"]
        scores = result.get("scores", {})

        print(f"\nPacket {i} (DB ID={meta['id']} | status={meta['status']}):")
        print(f"  Raw features shape: {raw_packet.shape}")
        print(f"  Prediction: {label} (conf={confidence:.4f})")
        if scores:
            print(f"  Scores: {scores}")


if __name__ == "__main__":
    main()
