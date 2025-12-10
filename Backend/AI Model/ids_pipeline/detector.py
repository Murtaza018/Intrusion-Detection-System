# detector.py
# Detection logic and classification (Complete Version)

import numpy as np
import threading
import queue
import traceback

from packet_storage import Packet
from config import BACKGROUND_SUMMARY_SIZE

class Detector:
    """Main detection logic"""
    
    def __init__(self, model_loader, feature_extractor, xai_explainer, packet_storage):
        self.model_loader = model_loader
        self.feature_extractor = feature_extractor
        self.xai_explainer = xai_explainer
        self.packet_storage = packet_storage
        
        self.packet_queue = queue.Queue()
        self.xai_queue = queue.Queue(maxsize=5)
        
        self.running = False
        self.thread = None
        self.xai_thread = None
        
    def start(self):
        """Start detection threads"""
        # Clear queues to prevent "poison pill" issues on restart
        self.packet_queue = queue.Queue()
        self.xai_queue = queue.Queue(maxsize=5)
        
        self.running = True
        
        # Start XAI worker thread
        self.xai_thread = threading.Thread(target=self._xai_worker, daemon=True, name="XAI_Worker")
        self.xai_thread.start()
        
        # Start detection worker thread
        self.thread = threading.Thread(target=self._detection_worker, daemon=True, name="Detection_Worker")
        self.thread.start()
        
        print("[*] Detection system started")
    
    def stop(self):
        """Stop detection threads"""
        self.running = False
        self.packet_queue.put(None)
        self.xai_queue.put(None)
        
        if self.thread:
            self.thread.join(timeout=2)
        if self.xai_thread:
            self.xai_thread.join(timeout=2)
        
        print("[*] Detection system stopped")
    
    def process_packet(self, packet):
        """Queue a packet for processing"""
        if self.running:
            self.packet_queue.put(packet)
    
    def _detection_worker(self):
        """Main detection worker thread"""
        print("[*] Detection worker started")
        
        while self.running:
            try:
                packet = self.packet_queue.get(timeout=1)
                if packet is None:
                    break
                
                # Clean up old flows periodically
                self.feature_extractor.cleanup_old_flows()
                
                # Extract features (Raw List of floats)
                features, flow_key = self.feature_extractor.extract_features(packet)
                
                # Update min/max and get scaled features
                self.feature_extractor.update_minmax(features)
                scaled_features = self.feature_extractor.scale_features(features)
                
                # Get packet info for Flutter
                packet_info = self._get_packet_info(packet)
                packet_id = self.packet_storage.get_next_packet_id()
                
                # Main classification
                if self.feature_extractor.is_scaling_enabled():
                    # Get models
                    main_model = self.model_loader.get_main_model()
                    rf_model = self.model_loader.get_rf_model()
                    xgb_model = self.model_loader.get_xgb_model()
                    autoencoder = self.model_loader.get_autoencoder_model()
                    
                    # 1. CNN Prediction (Binary)
                    cnn_prob = main_model.predict(scaled_features, verbose=0)[0][0]
                    
                    # 2. RF Prediction (Multi-Class)
                    # We assume class 0 is Normal, rest are Attack
                    rf_all_probs = rf_model.predict_proba(scaled_features)[0]
                    rf_prob = 1.0 - rf_all_probs[0]
                    
                    # 3. XGB Prediction (Multi-Class)
                    xgb_all_probs = xgb_model.predict_proba(scaled_features)[0]
                    xgb_prob = 1.0 - xgb_all_probs[0]
                    
                    # 4. Ensemble Voting (Max for Sensitivity)
                    # Using MAX ensures if ANY model sees an attack, we flag it.
                    ensemble_prob = max(cnn_prob, rf_prob, xgb_prob)
                    
                    # Debug logging
                    if ensemble_prob > 0.1 or packet_id % 50 == 0:
                        print(
                            f"[VOTE] ID:{packet_id} | "
                            f"CNN:{cnn_prob:.2f} RF:{rf_prob:.2f} XGB:{xgb_prob:.2f} | "
                            f"FINAL:{ensemble_prob:.2f}"
                        )

                    if ensemble_prob > 0.40:
                        # Known attack
                        print(f"\n[!!!] ENSEMBLE ATTACK - ID:{packet_id} Score:{ensemble_prob:.4f}")
                        self._handle_known_attack(packet, packet_id, scaled_features, ensemble_prob, packet_info, features)
                    else:
                        # Check for zero-day
                        reconstruction = autoencoder.predict(scaled_features, verbose=0)
                        mse = np.mean(np.power(scaled_features - reconstruction, 2))
                        self.feature_extractor.add_reconstruction_error(mse)
                        
                        threshold = self.feature_extractor.compute_dynamic_threshold()
                        
                        if mse > threshold:
                            # Zero-day anomaly
                            print(f"\n[?!?] ZERO-DAY - ID:{packet_id} Error:{mse:.4f} > Thr:{threshold:.4f}")
                            self._handle_zero_day(packet, packet_id, scaled_features, mse, packet_info, features)
                        else:
                            # Normal traffic
                            if self.packet_storage.get_stats()["total_packets"] % 20 == 0:
                                print(".", end="", flush=True)
                            self._handle_normal(packet, packet_id, packet_info, features)
                else:
                    # Still in warmup, just record as normal
                    self._handle_normal(packet, packet_id, packet_info, features)
                
                self.packet_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n[!] Detection error: {e}")
                traceback.print_exc()
    
    def _xai_worker(self):
        """XAI explanation worker thread"""
        print("[*] XAI worker started")
        
        main_model = self.model_loader.get_main_model()
        
        while self.running:
            try:
                task = self.xai_queue.get(timeout=1)
                if task is None:
                    break
                
                print(f"[XAI] Processing ID:{task['packet_id']}")
                
                # Attempt to initialize SHAP if needed
                if not self.xai_explainer.initialized:
                    bg_samples_unscaled = self.feature_extractor.get_background_samples()
                    
                    if len(bg_samples_unscaled) >= BACKGROUND_SUMMARY_SIZE:
                        print(f"[XAI] Populating background data...")
                        
                        with self.xai_explainer.lock:
                            self.xai_explainer.background_data.clear()
                            for sample in bg_samples_unscaled:
                                scaled_sample = self.feature_extractor.scale_features(sample)
                                self.xai_explainer.background_data.append(scaled_sample.flatten())
                        
                        self.xai_explainer.initialize_shap(main_model.predict, num_samples=BACKGROUND_SUMMARY_SIZE)

                # Generate explanation
                explanation = self.xai_explainer.generate_explanation(
                    features=task['features'].flatten(),
                    model_predict_func=main_model.predict,
                    confidence=task['confidence'],
                    packet_info=task['packet_info'],
                    attack_type=task['attack_type']
                )
                
                # Update DB with explanation
                # We don't need to pass 'features' here because they were already saved
                # when the packet was first created in _handle_... methods
                packet_obj = self._create_packet_object(
                    packet=task['packet'],
                    packet_id=task['packet_id'],
                    status=task['status'],
                    confidence=task['confidence'],
                    explanation=explanation,
                    features=None 
                )
                
                # Update in storage (this will perform an SQL UPDATE)
                self.packet_storage.add_packet(packet_obj)
                
                print(f"[XAI] ✅ Explanation generated for ID:{task['packet_id']}")
                
                self.xai_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[!] XAI error: {e}")
                traceback.print_exc()
    
    def _handle_known_attack(self, packet, packet_id, features, confidence, packet_info, raw_features):
        """Handle known attack detection"""
        # Send initial alert
        initial_explanation = {
            "type": "INITIAL_DETECTION",
            "title": "🔍 Analyzing Attack...",
            "description": "Known attack pattern detected. Generating AI-powered explanation...",
            "risk_level": "HIGH",
            "status": "analyzing"
        }
        
        packet_obj = self._create_packet_object(
            packet=packet,
            packet_id=packet_id,
            status="known_attack",
            confidence=confidence,
            explanation=initial_explanation,
            features=raw_features
        )
        
        self.packet_storage.add_packet(packet_obj)
        
        # Queue for XAI analysis
        if not self.xai_queue.full():
            try:
                self.xai_queue.put_nowait({
                    "packet_id": packet_id,
                    "packet": packet,
                    "features": features,
                    "confidence": confidence,
                    "packet_info": packet_info,
                    "status": "known_attack",
                    "attack_type": "Attack"
                })
                print(f"[XAI] 📤 Queued attack for analysis - ID:{packet_id}")
            except queue.Full:
                print(f"[!] XAI queue full for ID:{packet_id}")
    
    def _handle_zero_day(self, packet, packet_id, features, error, packet_info, raw_features):
        """Handle zero-day anomaly detection"""
        # Send initial alert
        initial_explanation = {
            "type": "INITIAL_DETECTION",
            "title": "🔬 Analyzing Anomaly...",
            "description": "Zero-day anomaly detected. Generating AI-powered explanation...",
            "risk_level": "CRITICAL",
            "status": "analyzing"
        }
        
        packet_obj = self._create_packet_object(
            packet=packet,
            packet_id=packet_id,
            status="zero_day",
            confidence=error,
            explanation=initial_explanation,
            features=raw_features
        )
        
        self.packet_storage.add_packet(packet_obj)
        
        # Queue for XAI analysis
        if not self.xai_queue.full():
            try:
                self.xai_queue.put_nowait({
                    "packet_id": packet_id,
                    "packet": packet,
                    "features": features,
                    "confidence": error,
                    "packet_info": packet_info,
                    "status": "zero_day",
                    "attack_type": "zero_day"
                })
                print(f"[XAI] 📤 Queued zero-day for analysis - ID:{packet_id}")
            except queue.Full:
                print(f"[!] XAI queue full for ID:{packet_id}")
    
    def _handle_normal(self, packet, packet_id, packet_info, raw_features):
        """Handle normal traffic"""
        explanation = {
            "type": "NORMAL_TRAFFIC",
            "title": "✅ Normal Network Traffic",
            "description": "This network traffic matches expected patterns and shows no signs of malicious activity.",
            "risk_level": "LOW",
            "confidence": "99.9%"
        }
        
        packet_obj = self._create_packet_object(
            packet=packet,
            packet_id=packet_id,
            status="normal",
            confidence=0.0,
            explanation=explanation,
            features=raw_features
        )
        
        self.packet_storage.add_packet(packet_obj)
    
    def _get_packet_info(self, packet):
        """Extract packet information for XAI"""
        from scapy.all import IP, TCP, UDP
        
        packet_info = {
            "src_ip": "",
            "dst_ip": "",
            "protocol": "OTHER",
            "src_port": 0,
            "dst_port": 0
        }
        
        if packet.haslayer(IP):
            packet_info["src_ip"] = packet[IP].src
            packet_info["dst_ip"] = packet[IP].dst
            
            if packet.haslayer(TCP):
                packet_info["protocol"] = "TCP"
                packet_info["src_port"] = packet[TCP].sport
                packet_info["dst_port"] = packet[TCP].dport
            elif packet.haslayer(UDP):
                packet_info["protocol"] = "UDP"
                packet_info["src_port"] = packet[UDP].sport
                packet_info["dst_port"] = packet[UDP].dport
        
        return packet_info
    
    def _create_packet_object(self, packet, packet_id, status, confidence, explanation, features=None):
        """Create Packet object for storage"""
        from datetime import datetime
        from scapy.all import IP, TCP, UDP
        
        # Extract packet details
        protocol = "OTHER"
        src_ip = ""
        dst_ip = ""
        src_port = 0
        dst_port = 0
        
        if packet.haslayer(IP):
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            
            if packet.haslayer(TCP):
                protocol = "TCP"
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
            elif packet.haslayer(UDP):
                protocol = "UDP"
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
        
        summary = f"{protocol} {src_ip}:{src_port} → {dst_ip}:{dst_port}"
        
        return Packet(
            id=packet_id,
            summary=summary,
            src_ip=src_ip,
            dst_ip=dst_ip,
            protocol=protocol,
            src_port=src_port,
            dst_port=dst_port,
            length=len(packet),
            timestamp=datetime.now(),
            status=status,
            confidence=confidence,
            explanation=explanation,
            features=features
        )