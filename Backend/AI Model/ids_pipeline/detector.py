import numpy as np
import threading
import queue
import traceback
import torch
from datetime import datetime

from packet_storage import Packet
from config import (
    BACKGROUND_SUMMARY_SIZE, 
    GNN_IN_CHANNELS, 
    MAE_MASK_RATIO, 
    GNN_EMBEDDING_DIM
)

class Detector:
    """
    Hybrid Detection Logic: GNN + MAE + 95-feature Ensemble.
    Fully integrated with SHAP XAI for hybrid explanations.
    """
    
    def __init__(self, model_loader, feature_extractor, xai_explainer, packet_storage):
        self.model_loader = model_loader
        self.feature_extractor = feature_extractor
        self.xai_explainer = xai_explainer
        self.packet_storage = packet_storage
        
        self.packet_queue = queue.Queue()
        self.xai_queue = queue.Queue(maxsize=50)
        
        self.running = False
        self.thread = None
        self.xai_thread = None
        
    def start(self):
        """Start detection threads and RESUME unfinished XAI tasks with full metadata"""
        self.packet_queue = queue.Queue()
        self.xai_queue = queue.Queue(maxsize=50)
        self.running = True
        
        print("[*] Priming XAI background from database...")
        # Prime SHAP background
        past_normal = self.packet_storage.get_packets(limit=50, status_filter='normal')
        for p_dict in past_normal:
            feat_list = self.packet_storage.get_features_for_training([p_dict['id']])
            if feat_list:
                self.feature_extractor.update_minmax(feat_list[0])

        # Find packets stuck in 'analyzing'
        unfinished = self.packet_storage.get_packets(limit=10, status_filter='analyzing')
        for p_dict in unfinished:
            print(f"[+] Resuming XAI analysis for Packet ID: {p_dict['id']}")
            raw_feats = self.packet_storage.get_features_for_training([p_dict['id']])
            
            if raw_feats:
                scaled_78 = self.feature_extractor.scale_features(raw_feats[0])
                enhanced = np.hstack([scaled_78, np.zeros((1, 17))]) 
                
                # --- FIX: Pass the EXISTING metadata from p_dict ---
                self.xai_queue.put({
                    "packet_id": p_dict['id'], 
                    "packet": None,
                    "features": enhanced,
                    "confidence": p_dict.get('confidence', 0.5),
                    # RE-INSERTING IP/PORT DATA FROM THE DB FETCH
                    "packet_info": {
                        "src_ip": p_dict.get('src_ip', ''), 
                        "dst_ip": p_dict.get('dst_ip', ''),
                        "protocol": p_dict.get('protocol', 'OTHER'),
                        "src_port": p_dict.get('src_port', 0),
                        "dst_port": p_dict.get('dst_port', 0)
                    },
                    "status": p_dict['status'],
                    "attack_type": "zero_day" if p_dict['status'] == 'zero_day' else "Attack"
                })

        self.xai_thread = threading.Thread(target=self._xai_worker, daemon=True, name="XAI_Worker")
        self.xai_thread.start()
        
        self.thread = threading.Thread(target=self._detection_worker, daemon=True, name="Detection_Worker")
        self.thread.start()
        
        print("[*] Hybrid Detection System Started (Metadata Preserved)")

    def stop(self):
        """Stop detection threads safely"""
        self.running = False
        self.packet_queue.put(None)
        self.xai_queue.put(None)
        
        if self.thread: self.thread.join(timeout=2)
        if self.xai_thread: self.xai_thread.join(timeout=2)
        
        print("[*] Detection system stopped")
    
    def process_packet(self, packet):
        """Queue a packet for background processing"""
        if self.running:
            self.packet_queue.put(packet)

    def _detection_worker(self):
        """Main detection loop incorporating sensory fusion (95 features)"""
        print("[*] Detection worker started")
        gnn_model = self.model_loader.get_gnn_model()
        mae_model = self.model_loader.get_mae_model()
        
        while self.running:
            try:
                # SILENTLY handle empty queue during low traffic
                try:
                    packet = self.packet_queue.get(timeout=1)
                except queue.Empty:
                    continue 
                
                if packet is None: break
                
                # 1. Feature Extraction & Scaling (Raw 78)
                self.feature_extractor.cleanup_old_flows()
                features, flow_key = self.feature_extractor.extract_features(packet)
                self.feature_extractor.update_minmax(features)
                scaled_features = self.feature_extractor.scale_features(features) # (1, 78)
                
                packet_info = self._get_packet_info(packet)
                packet_id = self.packet_storage.get_next_packet_id()
                
                if self.feature_extractor.is_scaling_enabled():
                    # --- [SENSORY LAYER 1: GNN TOPOLOGY] ---
                    edge_index_np, edge_attr_np = self.feature_extractor.graph_builder.get_graph_data()
                    gnn_vec = np.zeros((1, GNN_EMBEDDING_DIM))
                    
                    if edge_index_np is not None and gnn_model is not None:
                        x_gnn = torch.zeros((self.feature_extractor.graph_builder.id_counter, GNN_IN_CHANNELS))
                        x_gnn.index_add_(0, torch.tensor(edge_index_np[0], dtype=torch.long), 
                                         torch.tensor(edge_attr_np[:, :GNN_IN_CHANNELS], dtype=torch.float))
                        
                        with torch.no_grad():
                            z = gnn_model(x_gnn, torch.tensor(edge_index_np, dtype=torch.long))
                            src_id = self.feature_extractor.graph_builder.ip_to_id.get(packet_info['src_ip'])
                            if src_id is not None:
                                gnn_vec = z[src_id].cpu().numpy().reshape(1, GNN_EMBEDDING_DIM)

                    # --- [SENSORY LAYER 2: MAE VISUAL ANOMALY] ---
                    mae_err = 0.0
                    if mae_model is not None:
                        with torch.no_grad():
                            feat_tensor = torch.tensor(scaled_features, dtype=torch.float)
                            recon, original = mae_model(feat_tensor, mask_ratio=MAE_MASK_RATIO)
                            mae_err = torch.mean((recon - original)**2).item()

                    # --- [FEATURE FUSION: 78 + 16 + 1 = 95] ---
                    # Defining it early here ensures correct scope for handlers
                    enhanced_features = np.hstack([scaled_features, gnn_vec, [[mae_err]]])
                    print(f"DEBUG: Raw GNN Vec Mean: {np.mean(gnn_vec):.4f}")
                    raw_gnn_val = float(np.mean(np.abs(gnn_vec)))

                    # LOG-NORMALIZATION: log1p(x) handles the explosion gracefully.
                    # This maps: 
                    # 100 -> ~0.46 (46%)
                    # 1,000 -> ~0.69 (69%)
                    # 10,000 -> ~0.92 (92%)
                    # 32,000 -> ~0.99 (99%)
                    normalized_gnn = float(np.tanh(np.log1p(raw_gnn_val) / 10.0))

                    extra_metrics = {
                        "gnn_anomaly": normalized_gnn,
                        "mae_anomaly": mae_err
                    }

                    # --- [ENSEMBLE CLASSIFICATION] ---
                    cnn_prob = self.model_loader.get_main_model().predict(scaled_features, verbose=0)[0][0]
                    rf_prob = 1.0 - self.model_loader.get_rf_model().predict_proba(enhanced_features)[0][0]
                    xgb_prob = 1.0 - self.model_loader.get_xgb_model().predict_proba(enhanced_features)[0][0]
                    
                    ensemble_prob = max(cnn_prob, rf_prob, xgb_prob)
                    
                    if packet_id % 50 == 0:
                        print(f"[SENSE] ID:{packet_id} | MAE:{mae_err:.4f} | ENSEMBLE:{ensemble_prob:.2f}")

                    if ensemble_prob > 0.40:
                        # Passing enhanced_features (95-dim) to XAI queue
                        self._handle_known_attack(packet, packet_id, enhanced_features, ensemble_prob, packet_info, features, extra_metrics)
                    else:
                        # Final Zero-Day Check
                        reconstruction = self.model_loader.get_autoencoder_model().predict(scaled_features, verbose=0)
                        mse = np.mean(np.power(scaled_features - reconstruction, 2))
                        self.feature_extractor.add_reconstruction_error(mse)
                        threshold = self.feature_extractor.compute_dynamic_threshold()
                        
                        if mse > threshold or mae_err > 0.15:
                            self._handle_zero_day(packet, packet_id, enhanced_features, mse, packet_info, features, extra_metrics)
                        else:
                            self._handle_normal(packet, packet_id, packet_info, features, extra_metrics)
                else:
                    self._handle_normal(packet, packet_id, packet_info, features, {})
                
                self.packet_queue.task_done()
            except Exception as e:
                print(f"[!] Detection error: {e}")
                traceback.print_exc()

    def _xai_worker(self):
        """
        SHAP XAI worker thread with Metadata Preservation.
        Uses specialized database updates to ensure network info is not overwritten.
        """
        print("[*] XAI worker started")
        target_model = self.model_loader.get_xgb_model()
        
        while self.running:
            try:
                task = self.xai_queue.get(timeout=1)
                if task is None: break
                
                # 1. SHAP INITIALIZATION (Roadmap Point 3)
                # Build background context if not already initialized
                if not self.xai_explainer.initialized:
                    bg_samples_unscaled = self.feature_extractor.get_background_samples()
                    if len(bg_samples_unscaled) >= BACKGROUND_SUMMARY_SIZE:
                        with self.xai_explainer.lock:
                            self.xai_explainer.background_data.clear()
                            for sample in bg_samples_unscaled:
                                scaled_78 = self.feature_extractor.scale_features(sample)
                                # Pad to 95 dimensions for the Hybrid Ensemble Explainer
                                enhanced_bg = np.hstack([scaled_78, np.zeros((1, 17))]) 
                                self.xai_explainer.background_data.append(enhanced_bg.flatten())
                        
                        self.xai_explainer.initialize_shap(
                            target_model.predict_proba, 
                            num_samples=BACKGROUND_SUMMARY_SIZE
                        )

                # 2. GENERATE EXPLANATION
                # Generate robust SHAP-based factors for the 95-feature vector
                explanation = self.xai_explainer.generate_explanation(
                    features=task['features'], # The 95-dim enhanced vector
                    model_predict_func=target_model.predict_proba,
                    confidence=task['confidence'],
                    packet_info=task['packet_info'],
                    attack_type=task['attack_type']
                )
                
                # Mark as 'done' so Flutter stops showing the loading spinner
                explanation['status'] = 'done'
                
                # Merge GNN and MAE scores into the explanation for the visual gauges
                if 'extra_metrics' in task: 
                    explanation.update(task['extra_metrics'])

                # 3. THE CRITICAL FIX: METADATA PRESERVATION
                # Call the specialized update method that ONLY touches status/expl columns.
                # This ensures src_ip, dst_ip, and protocol are NEVER modified here.
                self.packet_storage.update_packet_xai_results(
                    packet_id=task['packet_id'],
                    explanation=explanation,
                    status=task['status'],
                    confidence=task['confidence']
                )
                
                self.xai_queue.task_done()
                
            except queue.Empty: 
                continue
            except Exception as e:
                print(f"[!] XAI worker error: {e}")
                import traceback
                traceback.print_exc()


    def _handle_known_attack(self, packet, packet_id, enhanced_features, confidence, packet_info, raw_features, extra_metrics):
        """Handle alert and queue 95-dim features for XAI"""
        initial_expl = {
            "title": "🚨 Known Attack Detected",
            "description": f"Ensemble AI flagged this traffic with {confidence:.1%} confidence. SHAP analysis in progress...",
            "risk_level": "HIGH", 
            "status": "analyzing", 
            **extra_metrics
        }
        
        packet_obj = self._create_packet_object(packet, packet_id, "known_attack", confidence, initial_expl, raw_features)
        self.packet_storage.add_packet(packet_obj)
        
        if not self.xai_queue.full():
            self.xai_queue.put_nowait({
                "packet_id": packet_id, "packet": packet, "features": enhanced_features,
                "confidence": confidence, "packet_info": packet_info,
                "status": "known_attack", "attack_type": "Attack", "extra_metrics": extra_metrics
            })

    def _handle_zero_day(self, packet, packet_id, enhanced_features, error, packet_info, raw_features, extra_metrics):
        """Handle zero-day and queue 95-dim features for XAI"""
        initial_expl = {
            "title": "🔬 Novelty/Zero-Day Alert", 
            "description": f"MAE/Autoencoder detected structural deviation (MSE: {error:.4f}). Validating novelty...",
            "risk_level": "CRITICAL", 
            "status": "analyzing", 
            **extra_metrics
        }
        
        packet_obj = self._create_packet_object(packet, packet_id, "zero_day", error, initial_expl, raw_features)
        self.packet_storage.add_packet(packet_obj)
        
        if not self.xai_queue.full():
            self.xai_queue.put_nowait({
                "packet_id": packet_id, "packet": packet, "features": enhanced_features,
                "confidence": error, "packet_info": packet_info,
                "status": "zero_day", "attack_type": "zero_day", "extra_metrics": extra_metrics
            })

    def _handle_normal(self, packet, packet_id, packet_info, raw_features, extra_metrics):
        
        expl = {
            "type": "NORMAL", 
            "title": "✅ Normal Traffic", 
            "description": "Traffic aligns with learned network baseline. No threats detected.",
            "risk_level": "LOW", 
            **extra_metrics
        }
        packet_obj = self._create_packet_object(packet, packet_id, "normal", 0.0, expl, raw_features)
        self.packet_storage.add_packet(packet_obj)

    def _get_packet_info(self, packet):
        from scapy.all import IP, TCP, UDP
        info = {"src_ip": "", "dst_ip": "", "protocol": "OTHER", "src_port": 0, "dst_port": 0}
        if packet.haslayer(IP):
            info["src_ip"], info["dst_ip"] = packet[IP].src, packet[IP].dst
            if packet.haslayer(TCP):
                info["protocol"], info["src_port"], info["dst_port"] = "TCP", packet[TCP].sport, packet[TCP].dport
            elif packet.haslayer(UDP):
                info["protocol"], info["src_port"], info["dst_port"] = "UDP", packet[UDP].sport, packet[UDP].dport
        return info

    def _create_packet_object(self, packet, packet_id, status, confidence, explanation, features=None):
        info = self._get_packet_info(packet)
        summary = f"{info['protocol']} {info['src_ip']}:{info['src_port']} → {info['dst_ip']}:{info['dst_port']}"
        return Packet(
            id=packet_id, summary=summary, src_ip=info['src_ip'], dst_ip=info['dst_ip'],
            protocol=info['protocol'], src_port=info['src_port'], dst_port=info['dst_port'],
            length=len(packet), timestamp=datetime.now(), status=status,
            confidence=confidence, explanation=explanation, features=features
        )