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
        """Main detection worker thread: GNN + MAE + Ensemble Integration"""
        import torch
        import numpy as np
        from config import GNN_IN_CHANNELS, MAE_MASK_RATIO
        
        print("[*] Detection worker started")
        gnn_model = self.model_loader.get_gnn_model()
        mae_model = self.model_loader.get_mae_model()
        
        while self.running:
            try:
                packet = self.packet_queue.get(timeout=1)
                if packet is None: break
                
                self.feature_extractor.cleanup_old_flows()
                features, flow_key = self.feature_extractor.extract_features(packet)
                self.feature_extractor.update_minmax(features)
                scaled_features = self.feature_extractor.scale_features(features)
                
                packet_info = self._get_packet_info(packet)
                packet_id = self.packet_storage.get_next_packet_id()
                
                if self.feature_extractor.is_scaling_enabled():
                    # --- [SENSORY LAYER 1: GNN CONTEXT] ---
                    edge_index_np, edge_attr_np = self.feature_extractor.graph_builder.get_graph_data()
                    context_vector = None
                    if edge_index_np is not None and gnn_model is not None:
                        # (GNN Inference Logic from Step 1)
                        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
                        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)
                        x_gnn = torch.zeros((self.feature_extractor.graph_builder.id_counter, GNN_IN_CHANNELS))
                        x_gnn.index_add_(0, edge_index[0], edge_attr[:, :GNN_IN_CHANNELS])
                        with torch.no_grad():
                            z = gnn_model(x_gnn, edge_index)
                            src_id = self.feature_extractor.graph_builder.ip_to_id.get(packet_info['src_ip'])
                            if src_id is not None: context_vector = z[src_id].cpu().numpy()

                    # --- [SENSORY LAYER 2: MAE VISUAL ANOMALY] ---
                    mae_loss = 0.0
                    if mae_model is not None:
                        with torch.no_grad():
                            feat_tensor = torch.tensor(scaled_features, dtype=torch.float)
                            recon, original = mae_model(feat_tensor, mask_ratio=MAE_MASK_RATIO)
                            mae_loss = torch.mean((recon - original)**2).item()

                    # --- [ENSEMBLE CLASSIFICATION] ---
                    main_model = self.model_loader.get_main_model()
                    rf_model = self.model_loader.get_rf_model()
                    xgb_model = self.model_loader.get_xgb_model()
                    autoencoder = self.model_loader.get_autoencoder_model()
                    
                    cnn_prob = main_model.predict(scaled_features, verbose=0)[0][0]
                    rf_prob = 1.0 - rf_model.predict_proba(scaled_features)[0][0]
                    xgb_prob = 1.0 - xgb_model.predict_proba(scaled_features)[0][0]
                    
                    ensemble_prob = max(cnn_prob, rf_prob, xgb_prob)
                    
                    # Log Sensory Results
                    if packet_id % 50 == 0:
                        print(f"[SENSE] ID:{packet_id} | GNN:{'YES' if context_vector is not None else 'NO'} | MAE_ERR:{mae_loss:.4f}")

                    if ensemble_prob > 0.40:
                        self._handle_known_attack(packet, packet_id, scaled_features, ensemble_prob, packet_info, features)
                    else:
                        # Final Zero-Day Check (Standard AE + MAE Insight)
                        reconstruction = autoencoder.predict(scaled_features, verbose=0)
                        mse = np.mean(np.power(scaled_features - reconstruction, 2))
                        self.feature_extractor.add_reconstruction_error(mse)
                        threshold = self.feature_extractor.compute_dynamic_threshold()
                        
                        # Trigger alert if either the standard AE or the Visual MAE sees a massive spike
                        if mse > threshold or mae_loss > 0.15: # 0.15 is a heuristic threshold
                            self._handle_zero_day(packet, packet_id, scaled_features, mse, packet_info, features)
                        else:
                            self._handle_normal(packet, packet_id, packet_info, features)
                else:
                    self._handle_normal(packet, packet_id, packet_info, features)
                
                self.packet_queue.task_done()
            except Exception as e:
                print(f"[!] Detection error: {e}")
                
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