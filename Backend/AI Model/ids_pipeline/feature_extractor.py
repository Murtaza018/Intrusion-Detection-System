# feature_extractor.py
# Feature extraction from network packets

import time
import numpy as np
from collections import defaultdict, deque
from scapy.all import IP, TCP, UDP

from config import NUM_FEATURES, FLOW_TIMEOUT

class FeatureExtractor:
    """Extract features from network packets"""
    
    def __init__(self):
        self.flows = defaultdict(lambda: {"packets": [], "timestamps": [], "lengths": [], "flags": []})
        self.flow_timeout = FLOW_TIMEOUT
        
        # For min/max scaling
        self.live_min = None
        self.live_max = None
        self.warmup_count = 0
        self.scaling_enabled = False
        self.background_buffer = deque(maxlen=100)
        
        # For dynamic threshold
        self.recent_errors = deque(maxlen=200)
    
    def extract_features(self, packet):
        """Extract features from a single packet"""
        features = [0.0] * NUM_FEATURES
        
        try:
            if not packet.haslayer(IP):
                return features, None
            
            ip_layer = packet[IP]
            proto = 6 if packet.haslayer(TCP) else 17 if packet.haslayer(UDP) else 0
            sport = getattr(packet, "sport", 0)
            dport = getattr(packet, "dport", 0)
            flow_key = (ip_layer.src, ip_layer.dst, sport, dport, proto)
            
            # Get or create flow
            if flow_key not in self.flows:
                self.flows[flow_key] = {"packets": [], "timestamps": [], "lengths": [], "flags": []}
            flow = self.flows[flow_key]
            
            # Add packet to flow
            flow["packets"].append(packet)
            flow["timestamps"].append(time.time())
            pkt_len = len(packet)
            flow["lengths"].append(pkt_len)
            
            # Limit flow history
            if len(flow["packets"]) > 50:
                for k in flow:
                    flow[k] = flow[k][-50:]
            
            # Extract TCP flags
            last_flags = 0
            if proto == 6 and packet.haslayer(TCP):
                try:
                    last_flags = int(packet[TCP].flags)
                except:
                    pass
            flow["flags"].append(last_flags)
            
            # Calculate flow statistics
            lengths = flow["lengths"]
            pkt_count = len(lengths)
            total_len = float(sum(lengths))
            
            # Basic flow features
            features[0] = float(dport)
            features[1] = float(flow["timestamps"][-1] - flow["timestamps"][0] + 1e-9) if pkt_count > 1 else 0.0
            features[2] = float(pkt_count)
            features[3] = float(pkt_count)
            features[4] = float(total_len)
            features[5] = float(total_len)
            
            if pkt_count > 0:
                features[6] = float(max(lengths))
                features[7] = float(min(lengths))
                features[8] = float(total_len / pkt_count)
            
            # Rate calculations
            dur = features[1]
            if dur > 1e-6:
                features[14] = float(total_len / dur)
                features[15] = float(pkt_count / dur)
            
            # Current packet features
            features[34] = float(pkt_len)
            features[35] = float(pkt_len)
            
            return features, flow_key
            
        except Exception as e:
            print(f"[!] Feature extraction error: {e}")
            return features, None
    
    def update_minmax(self, features):
        """Update min/max scaling parameters"""
        features_np = np.array(features, dtype=np.float32)
        
        # Add to background buffer for XAI
        self.background_buffer.append(features_np)
        
        # Update min/max
        if self.live_min is None:
            self.live_min = features_np.copy()
            self.live_max = features_np.copy()
        else:
            self.live_min = np.minimum(self.live_min, features_np)
            self.live_max = np.maximum(self.live_max, features_np)
        
        self.warmup_count += 1
        
        # Enable scaling after warmup
        if not self.scaling_enabled and self.warmup_count >= 50:
            self.scaling_enabled = True
            print(f"\n[***] WARMUP COMPLETE! Scaling enabled. Collected {len(self.background_buffer)} samples.")
    
    def scale_features(self, features):
        """Scale features using min/max"""
        if not self.scaling_enabled or self.live_min is None or self.live_max is None:
            return np.array([features], dtype=np.float32)
        
        features_np = np.array(features, dtype=np.float32)
        denom = (self.live_max - self.live_min)
        denom_safe = np.where(denom == 0.0, 1.0, denom)
        
        scaled = np.clip((features_np - self.live_min) / denom_safe, 0.0, 1.0)
        return scaled.reshape(1, -1)
    
    def add_reconstruction_error(self, error):
        """Add reconstruction error for dynamic threshold"""
        self.recent_errors.append(error)
    
    def compute_dynamic_threshold(self, default=1.0):
        """Compute dynamic threshold for zero-day detection"""
        if len(self.recent_errors) < 50:
            return default
        
        arr = np.asarray(self.recent_errors, dtype=float)
        arr = arr[np.isfinite(arr)]
        
        if arr.size == 0:
            return default
        
        m = float(arr.mean())
        s = float(arr.std())
        return m + 3.0 * s
    
    def cleanup_old_flows(self, max_age=300):
        """Clean up old flows"""
        now = time.time()
        flows_to_delete = []
        
        for flow_key, flow_data in self.flows.items():
            if flow_data["timestamps"] and now - flow_data["timestamps"][-1] > max_age:
                flows_to_delete.append(flow_key)
        
        for flow_key in flows_to_delete[:20]:
            if flow_key in self.flows:
                del self.flows[flow_key]
    
    def get_background_samples(self):
        """Get background samples for XAI"""
        return list(self.background_buffer)
    
    def is_scaling_enabled(self):
        """Check if scaling is enabled"""
        return self.scaling_enabled