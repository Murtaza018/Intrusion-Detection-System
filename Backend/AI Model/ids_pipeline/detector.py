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

import firebase_admin
from firebase_admin import credentials, messaging


class Detector:

    
    def __init__(self, model_loader, feature_extractor, xai_explainer, packet_storage):
        self.model_loader = model_loader
        self.feature_extractor = feature_extractor
        self.xai_explainer = xai_explainer
        self.packet_storage = packet_storage

        self.registered_tokens = set()
        
        self.packet_queue = queue.Queue()
        self.xai_queue = queue.Queue(maxsize=50)
        
        self.running = False
        self.thread = None
        self.xai_thread = None
        
    def send_push_notification(self, anomaly_type, details):
        """Sends a notification to all registered mobile devices."""
        if not self.registered_tokens:
            return

        try:
            message = messaging.MulticastMessage(
                notification=messaging.Notification(
                    title=f"🚨 IDS Alert: {anomaly_type}!",
                    body=details,
                ),
                tokens=list(self.registered_tokens),
            )
            response = messaging.send_multicast(message)
            print(f"[*] FCM: Successfully sent {response.success_count} alerts.")
        except Exception as e:
            print(f"[!] FCM Notification Error: {e}")


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
        """Main detection loop using True Cascade Architecture for max speed"""
        print("[*] Detection worker started")
        gnn_model = self.model_loader.get_gnn_model()
        mae_model = self.model_loader.get_mae_model()
        
        while self.running:
            try:
                try:
                    packet = self.packet_queue.get(timeout=1)
                except queue.Empty:
                    continue 
                
                if packet is None: break
                
                # 1. Feature Extraction (Raw 78)
                self.feature_extractor.cleanup_old_flows()
                features, flow_key = self.feature_extractor.extract_features(packet)
                self.feature_extractor.update_minmax(features)
                scaled_features = self.feature_extractor.scale_features(features) # (1, 78)
                
                packet_info = self._get_packet_info(packet)
                packet_id = self.packet_storage.get_next_packet_id()
                
                if self.feature_extractor.is_scaling_enabled():
                    
                    main_model = self.model_loader.get_main_model()
                    cnn_prob = float(main_model(scaled_features, training=False)[0][0])
                    
                    rf_prob = self.model_loader.get_rf_model().predict_proba(scaled_features)[0][1] 
                    xgb_prob = self.model_loader.get_xgb_model().predict_proba(scaled_features)[0][1] 

                    ensemble_prob = max(cnn_prob, rf_prob, xgb_prob)

                    if ensemble_prob > 0.40:
                        extra_metrics = {"gnn_anomaly": 1.0, "mae_anomaly": 1.0}
                        # We pass scaled_features (78) instead of enhanced_features (95)
                        self._handle_known_attack(packet, packet_id, scaled_features, ensemble_prob, packet_info, features, extra_metrics)
                        
                        # Keep the graph updated with the attack, but bypass heavy GNN inference
                        if self.feature_extractor.graph_builder is not None:
                            self.feature_extractor.graph_builder.add_packet(
                                packet_info=packet_info, features=scaled_features[0],
                                gnn_anomaly=1.0, ae_mse=1.0, mae_err=1.0
                            )
                    else:
                        
                        gnn_vec = np.zeros((1, GNN_EMBEDDING_DIM))
                        normalized_gnn = 0.0
                        edge_index_np, edge_attr_np, node_anomaly = self.feature_extractor.graph_builder.get_graph_data()
                        
                        if edge_index_np is not None and gnn_model is not None:
                            x_gnn = torch.zeros((self.feature_extractor.graph_builder.id_counter, GNN_IN_CHANNELS))
                            x_gnn.index_add_(0, torch.tensor(edge_index_np[0], dtype=torch.long), 
                                             torch.tensor(edge_attr_np[:, :GNN_IN_CHANNELS], dtype=torch.float))
                            with torch.no_grad():
                                z = gnn_model(x_gnn, torch.tensor(edge_index_np, dtype=torch.long))
                                src_id = self.feature_extractor.graph_builder.ip_to_id.get(packet_info['src_ip'])
                                if src_id is not None:
                                    gnn_vec = z[src_id].cpu().numpy().reshape(1, GNN_EMBEDDING_DIM)
                                    raw_gnn_val = float(np.mean(np.abs(gnn_vec)))
                                    normalized_gnn = float(np.tanh(np.log1p(raw_gnn_val) / 10.0))

                      
                        mae_err = 0.0
                        if mae_model is not None:
                            with torch.no_grad():
                                feat_tensor = torch.tensor(scaled_features, dtype=torch.float)
                                recon, original = mae_model(feat_tensor, mask_ratio=MAE_MASK_RATIO)
                                mae_err = torch.mean((recon - original)**2).item()

                       
                        autoencoder = self.model_loader.get_autoencoder_model()
                        reconstruction = autoencoder(scaled_features, training=False).numpy()
                        mse = np.mean(np.power(scaled_features - reconstruction, 2))
                        
                        self.feature_extractor.add_reconstruction_error(mse)
                        threshold = self.feature_extractor.compute_dynamic_threshold()
                        
                        extra_metrics = {"gnn_anomaly": normalized_gnn, "mae_anomaly": mae_err}
                        
                        if mse > threshold or mae_err > 0.15:
                            self._handle_zero_day(packet, packet_id, scaled_features, mse, packet_info, features, extra_metrics)
                        else:
                            self._handle_normal(packet, packet_id, packet_info, features, extra_metrics)

                        # Update Graph with actual calculated values
                        if self.feature_extractor.graph_builder is not None:
                            self.feature_extractor.graph_builder.add_packet(
                                packet_info=packet_info, features=scaled_features[0],
                                gnn_anomaly=normalized_gnn, ae_mse=mse, mae_err=mae_err
                            )
                else:
                    self._handle_normal(packet, packet_id, packet_info, features, {})
                
                self.packet_queue.task_done()
            except Exception as e:
                print(f"[!] Detection error: {e}")
                traceback.print_exc()

    def _xai_worker(self):
        """Main XAI loop for 78-feature fast guards"""
        print("[*] XAI worker started")
        target_model = self.model_loader.get_xgb_model()
        
        while self.running:
            try:
                task = self.xai_queue.get(timeout=1)
                if task is None: break
                
                if not self.xai_explainer.initialized:
                    bg_samples_unscaled = self.feature_extractor.get_background_samples()
                    if len(bg_samples_unscaled) >= BACKGROUND_SUMMARY_SIZE:
                        with self.xai_explainer.lock:
                            self.xai_explainer.background_data.clear()
                            for sample in bg_samples_unscaled:
                                scaled_78 = self.feature_extractor.scale_features(sample)
                                # THE FIX: No more padding. Just append the raw 78 features!
                                self.xai_explainer.background_data.append(scaled_78.flatten())
                        
                        self.xai_explainer.initialize_shap(
                            target_model.predict_proba, 
                            num_samples=BACKGROUND_SUMMARY_SIZE
                        )

                # 2. GENERATE EXPLANATION
                explanation = self.xai_explainer.generate_explanation(
                    features=task['features'], # This is now strictly 78 dims
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

                # 3. METADATA PRESERVATION
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

        self.send_push_notification(
            "Known Attack", 
            f"Detected {confidence:.1%} threat from {packet_info['src_ip']} ({packet_info['protocol']})"
        )
        
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
        
        self.send_push_notification(
            "Zero-Day Anomaly", 
            f"Structural deviation detected from {packet_info['src_ip']}. MSE: {error:.4f}"
        )

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
    
    def get_graph_snapshot(self):
        """
        Used by API server to expose current graph topology + anomaly.
        """
        if not hasattr(self.feature_extractor, 'graph_builder'):
            return None

        graph_builder = self.feature_extractor.graph_builder
        edge_index, edge_attr, node_anomaly = graph_builder.get_graph_data()
        if edge_index is None:
            return None

        ip_to_id = graph_builder.ip_to_id
        id_to_ip = {v: k for k, v in ip_to_id.items()}

        nodes = []
        for ip, nid in ip_to_id.items():
            anomaly = node_anomaly.get(nid, 0.0)
            nodes.append({
                "id": nid,
                "ip": ip,
                "anomaly": float(anomaly),
            })

        edges = []
        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i])
            dst = int(edge_index[1, i])
            edges.append({
                "source": src,
                "target": dst,
                "weight": 1.0,
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "timestamp": datetime.now().timestamp(),
        }



    def classify_features_pipeline(self, raw_features: np.ndarray, extra_info=None):
       
        if raw_features.ndim == 1:
            raw_features = raw_features.reshape(1, -1)  # (1, 78)

        # 1. Get models
        cnn_model = self.model_loader.get_main_model()
        rf_model = self.model_loader.get_rf_model()
        xgb_model = self.model_loader.get_xgb_model()
        ae_model = self.model_loader.get_autoencoder_model()
        gnn_model = self.model_loader.get_gnn_model()
        mae_model = self.model_loader.get_mae_model()

        # 2. Scale features (same as in _detection_worker)
        scaled_features = self.feature_extractor.scale_features(raw_features)  # maybe 3D

        # Ensure scaled_features is 2D for stacking
        scaled_features = scaled_features.reshape(scaled_features.shape[0], -1)  # (1, 78)

        # 3. GNN vector: try to build graph from extra_info
        gnn_vec = np.zeros((1, GNN_EMBEDDING_DIM), dtype=np.float32)
        gnn_anomaly_val = 0.0

        if extra_info and gnn_model is not None:
            try:
                edge_index_np = None
                edge_attr_np = None

                if hasattr(self.feature_extractor, "graph_builder"):
                    edge_index_np, edge_attr_np,node_anomaly = self.feature_extractor.graph_builder.get_graph_data()

                if edge_index_np is not None:
                    x_gnn = torch.zeros(
                        (self.feature_extractor.graph_builder.id_counter, GNN_IN_CHANNELS),
                        dtype=torch.float
                    )
                    x_gnn.index_add_(0, torch.tensor(edge_index_np[0], dtype=torch.long),
                                        torch.tensor(edge_attr_np[:, :GNN_IN_CHANNELS], dtype=torch.float))
                    with torch.no_grad():
                        z = gnn_model(x_gnn, torch.tensor(edge_index_np, dtype=torch.long))
                        src_id = self.feature_extractor.graph_builder.ip_to_id.get(
                            extra_info.get("src_ip")
                        )
                        if src_id is not None:
                            gnn_vec = z[src_id].cpu().numpy().reshape(1, GNN_EMBEDDING_DIM)
                            gnn_anomaly_val = float(np.mean(np.abs(gnn_vec)))
            except Exception as e:
                print(f"[!] GNN setup failed, using zeros: {e}")
                gnn_vec = np.zeros((1, GNN_EMBEDDING_DIM), dtype=np.float32)

        # 4. Normalize GNN anomaly (same as in _detection_worker)
        normalized_gnn = float(np.tanh(np.log1p(gnn_anomaly_val) / 10.0))

        # 5. MAE error (commented out to avoid shape error for now)
        mae_err = 0.0

        # if mae_model is not None:
        #     with torch.no_grad():
        #         feat_tensor = torch.tensor(scaled_features, dtype=torch.float)
        #         recon, original = mae_model(feat_tensor, mask_ratio=MAE_MASK_RATIO)
        #         mae_err = torch.mean((recon - original)**2).item()

        # 6. Build 95‑dim enhanced features (all 2D)
        enhanced_features = np.hstack([scaled_features, gnn_vec, np.array([[mae_err]])])  # (1, 95)

        # 7. CNN prediction (78‑dim)
        cnn_prob = cnn_model.predict(scaled_features, verbose=0)[0][0]  # prob of attack

        # 8. RF prediction (95‑dim)
        rf_prob = 1.0 - rf_model.predict_proba(enhanced_features)[0][0]  # prob of attack

        # 9. XGBoost prediction (95‑dim)
        xgb_prob = 1.0 - xgb_model.predict_proba(enhanced_features)[0][0]  # prob of attack

        # 10. Ensemble probability
        ensemble_prob = max(cnn_prob, rf_prob, xgb_prob)

        # 11. Autoencoder reconstruction (78‑dim) → zero‑day check
        reconstruction = ae_model.predict(scaled_features, verbose=0)
        ae_mse = np.mean(np.power(scaled_features - reconstruction, 2))

        # Threshold logic
        try:
            if self.feature_extractor.reconstruction_errors:
                threshold = self.feature_extractor.compute_dynamic_threshold()
            else:
                threshold = 0.1
        except:
            threshold = 0.1

        # 12. Apply labeling logic
        if ensemble_prob > 0.40:
            label = "known_attack"
            confidence = ensemble_prob
        elif ae_mse > threshold or mae_err > 0.15:
            label = "zero_day"
            confidence = ae_mse if ae_mse > mae_err else mae_err
        else:
            label = "normal"
            confidence = 0.0

        # 13. Return results
        return {
            "label": label,
            "confidence": confidence,
            "scores": {
                "cnn_prob": cnn_prob,
                "rf_prob": 1.0 - rf_prob,
                "xgb_prob": 1.0 - xgb_prob,
                "ae_mse": ae_mse,
                "mae_err": mae_err,
                "gnn_anomaly": normalized_gnn,
                "threshold": threshold,
            },
            "extra_metrics": {
                "gnn_anomaly": normalized_gnn,
                "mae_anomaly": mae_err,
            },
        }