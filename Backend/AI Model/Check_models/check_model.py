import pandas as pd
import numpy as np
import os
import sys
import joblib
from scapy.all import PcapReader, IP, TCP, UDP
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports from IDS components
from ids_pipeline.feature_extractor import FeatureExtractor
from ids_pipeline.model_loader import ModelLoader

# --- CONFIGURATION ---
PCAP_FILE = "synthetic_attacks.pcap"
LABELS_FILE = "synthetic_labels.csv"
ENCODER_PATH = "../ids_pipeline/label_encoder.pkl"

def load_label_encoder():
    # Try multiple paths to find the encoder.
    paths = [
        ENCODER_PATH,
        "label_encoder.pkl",
        os.path.join("ids_pipeline", "label_encoder.pkl")
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"Loaded Label Encoder from: {p}")
            return joblib.load(p)
    print("Warning: Label Encoder not found.")
    return None

def evaluate_system():
    print("\n" + "="*50)
    print("IDS PERFORMANCE EVALUATION")
    print("="*50)

    # 1. Load Ground Truth
    if not os.path.exists(LABELS_FILE) or not os.path.exists(PCAP_FILE):
        print(f"Error: Missing files. Looking for: {PCAP_FILE} and {LABELS_FILE}")
        return

    print(f"Loading Ground Truth from {LABELS_FILE}...")
    try:
        df_labels = pd.read_csv(LABELS_FILE)
        # Look for 'label' or 'Label' column.
        label_col = next((c for c in df_labels.columns if c.lower() == 'label'), df_labels.columns[-1])
        true_labels = df_labels[label_col].astype(str).values
        print(f"- Found {len(true_labels)} labeled entries.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. Initialize System Components
    print("Initializing IDS Components...")
    extractor = FeatureExtractor()
    model_loader = ModelLoader()
    label_encoder = load_label_encoder()
    
    if not model_loader.rf_model and not model_loader.xgb_model:
        print("Error: No models loaded. Train them first!")
        return

    # 3. Process PCAP
    print(f"Processing Packets from {PCAP_FILE}...")
    
    predicted_labels = []
    processed_count = 0
    
    try:
        # Use PcapReader for memory efficiency.
        with PcapReader(PCAP_FILE) as pcap_reader:
            for i, packet in enumerate(pcap_reader):
                if i >= len(true_labels):
                    print(f"Warning: PCAP has more packets than CSV. Stopping at {i}.")
                    break
                
                # A. Extract Features
                features, _ = extractor.extract_features(packet)
                
                # B. Scale Features
                extractor.update_minmax(features) # Update live scaler
                scaled_features = extractor.scale_features(features)

                # C. Predict
                pred_id = model_loader.predict(scaled_features) 
                
                # D. Convert ID to String Label
                pred_str = str(pred_id)
                if label_encoder:
                    try:
                        if isinstance(pred_id, (np.ndarray, list)):
                            pred_id = pred_id[0]
                        pred_str = label_encoder.inverse_transform([int(pred_id)])[0]
                    except Exception:
                        pred_str = str(pred_id)

                predicted_labels.append(pred_str)
                processed_count += 1
                
                if processed_count % 500 == 0:
                    print(f"... processed {processed_count} packets", end="\r")

    except Exception as e:
        print(f"\nError during processing: {e}")
        # Continue with available data.
    
    print(f"\nProcessing Complete. {processed_count} packets analyzed.")

    # 4. Alignment & Scoring
    limit = min(len(true_labels), len(predicted_labels))
    y_true = true_labels[:limit]
    y_pred = predicted_labels[:limit]

    print("\n" + "="*50)
    print("FINAL METRICS REPORT")
    print("="*50)
    
    try:
        # Calculate Accuracy
        acc = accuracy_score(y_true, y_pred)
        print(f"Overall Accuracy: {acc * 100:.2f}%")
        print("-" * 50)
        
        # Detailed Metrics
        print("Classification Report (Precision, Recall, F1-Score):")
        # Explain the metrics briefly for clarity. 
        print(classification_report(y_true, y_pred, zero_division=0))
        
        print("-" * 50)
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")

if __name__ == "__main__":
    evaluate_system()