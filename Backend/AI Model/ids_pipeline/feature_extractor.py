import time
import numpy as np
from collections import defaultdict, deque
from scapy.all import IP, TCP, UDP
from graph_builder import GraphBuilder
from config import NUM_FEATURES, FLOW_TIMEOUT, GRAPH_WINDOW_SIZE

class FeatureExtractor:
    """
    Stateful Feature Extractor for real-time network traffic.
    Includes support for GNN topological construction and live MinMax scaling.
    """
    
    def __init__(self):
        # Initialize flow storage
        self.flows = defaultdict(lambda: {
            "packets": [], 
            "timestamps": [], 
            "lengths": [], 
            "flags": []
        })
        self.flow_timeout = FLOW_TIMEOUT
        
        # Initialize GNN Graph Builder (Roadmap Point 2 & 3)
        self.graph_builder = GraphBuilder(window_size=GRAPH_WINDOW_SIZE)
        self.node_stats = defaultdict(lambda: {"conn_count": 0, "unique_dst": set()})
        
        # Scaling parameters
        self.live_min = None
        self.live_max = None
        self.warmup_count = 0
        self.scaling_enabled = False
        
        # --- FIXED: Initialization of buffers for XAI and Zero-Day Logic ---
        self.background_buffer = deque(maxlen=100) # For SHAP XAI background data
        self.recent_errors = deque(maxlen=200)    # For dynamic MSE thresholding
        # -------------------------------------------------------------------

    def extract_features(self, packet):
        """Extracts 78 raw features from a packet and updates flow state."""
        features = [0.0] * NUM_FEATURES
        try:
            if not packet.haslayer(IP):
                return features, None
            
            ip_layer = packet[IP]
            src_ip, dst_ip = ip_layer.src, ip_layer.dst
            proto = 6 if packet.haslayer(TCP) else 17 if packet.haslayer(UDP) else 0
            sport = getattr(packet, "sport", 0)
            dport = getattr(packet, "dport", 0)
            flow_key = (src_ip, dst_ip, sport, dport, proto)
            
            # --- [TOPOLOGICAL LOGIC (POINT 3)] ---
            self.node_stats[src_ip]["conn_count"] += 1
            self.node_stats[src_ip]["unique_dst"].add(dst_ip)
            
            # --- [FLOW STATE MANAGEMENT] ---
            if flow_key not in self.flows:
                self.flows[flow_key] = {"packets": [], "timestamps": [], "lengths": [], "flags": []}
            flow = self.flows[flow_key]
            
            flow["packets"].append(packet)
            current_ts = time.time()
            flow["timestamps"].append(current_ts)
            pkt_len = len(packet)
            flow["lengths"].append(pkt_len)
            
            # Extract TCP flags
            last_flags = 0
            if proto == 6 and packet.haslayer(TCP):
                try: last_flags = int(packet[TCP].flags)
                except: pass
            flow["flags"].append(last_flags)
            
            # Keep rolling window of last 50 packets for memory efficiency
            if len(flow["packets"]) > 50:
                for k in flow: flow[k] = flow[k][-50:]
            
            lengths = flow["lengths"]
            pkt_count = len(lengths)
            total_len = float(sum(lengths))
            
            # --- FEATURE MAPPING (Aligned with CIC-IDS-2017 schema) ---
            features[0] = float(dport)
            features[1] = float(flow["timestamps"][-1] - flow["timestamps"][0] + 1e-9) if pkt_count > 1 else 0.0
            features[2] = float(pkt_count)
            features[3] = float(pkt_count)  # Fwd Packets
            features[4] = float(total_len)  # Total Fwd Length
            features[5] = float(total_len)
            
            if pkt_count > 0:
                features[6] = float(max(lengths))
                features[7] = float(min(lengths))
                features[8] = float(total_len / pkt_count)
            
            # Timing/IAT Features
            if pkt_count > 1:
                iats = np.diff(flow["timestamps"])
                features[9] = float(np.mean(iats))  # Mean IAT
                features[10] = float(np.std(iats))  # IAT Std Dev
            
            # Flow Rates
            dur = features[1]
            if dur > 1e-6:
                features[14] = float(total_len / dur) # Flow Bytes/s
                features[15] = float(pkt_count / dur) # Flow Packets/s
            
            # Packet Length metrics
            features[34] = float(pkt_len)
            features[35] = float(pkt_len)
            
            # Update GNN Graph Builder with new packet
            packet_info = {"src_ip": src_ip, "dst_ip": dst_ip, "src_port": sport, "dst_port": dport, "protocol": proto}
            self.graph_builder.add_packet(packet_info, features)
            
            return features, flow_key
        except Exception as e:
            print(f"[!] Feature extraction error: {e}")
            return features, None

    def update_minmax(self, features):
        """Update live scaling parameters and populate the XAI background buffer."""
        features_np = np.array(features, dtype=np.float32)
        
        # Correctly append to the initialized buffer
        self.background_buffer.append(features_np)
        
        # Update min/max bounds for online normalization
        if self.live_min is None:
            self.live_min = features_np.copy()
            self.live_max = features_np.copy()
        else:
            self.live_min = np.minimum(self.live_min, features_np)
            self.live_max = np.maximum(self.live_max, features_np)
        
        self.warmup_count += 1
        
        # Enable scaling after seeing enough samples to establish a baseline
        if not self.scaling_enabled and self.warmup_count >= 50:
            self.scaling_enabled = True
            print(f"\n[***] WARMUP COMPLETE! Scaling enabled. Buffer size: {len(self.background_buffer)}")
    
    def scale_features(self, features):
        """Scale features into [0,1] range using live min/max statistics."""
        if not self.scaling_enabled or self.live_min is None or self.live_max is None:
            return np.array([features], dtype=np.float32)
        
        features_np = np.array(features, dtype=np.float32)
        denom = (self.live_max - self.live_min)
        denom_safe = np.where(denom == 0.0, 1.0, denom)
        
        scaled = np.clip((features_np - self.live_min) / denom_safe, 0.0, 1.0)
        return scaled.reshape(1, -1)

    def inverse_scale_features(self, scaled_features):
        """Crucial for GAN support: Reconstruct raw values for training CSVs."""
        if not self.scaling_enabled or self.live_min is None or self.live_max is None:
            return scaled_features

        features_np = np.array(scaled_features, dtype=np.float32)
        denom = (self.live_max - self.live_min)
        raw = features_np * denom + self.live_min
        return raw

    def add_reconstruction_error(self, error):
        """Stores reconstruction error for dynamic threshold calculation (Point 1)."""
        self.recent_errors.append(error)
    
    def compute_dynamic_threshold(self, default=1.0):
        """
        Uses statistics from recent packets to determine if a packet is a 'Zero-Day'.
        Aligns with SAFE/CND-IDS research (Roadmap Point 1).
        """
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
        """Periodically cleans up inactive network flows to save memory."""
        now = time.time()
        flows_to_delete = []
        
        for flow_key, flow_data in self.flows.items():
            if flow_data["timestamps"] and now - flow_data["timestamps"][-1] > max_age:
                flows_to_delete.append(flow_key)
        
        # Incremental cleanup
        for flow_key in flows_to_delete[:20]:
            if flow_key in self.flows:
                del self.flows[flow_key]
    
    def get_background_samples(self):
        """Returns the rolling window of samples for SHAP XAI background logic."""
        return list(self.background_buffer)
    
    def is_scaling_enabled(self):
        return self.scaling_enabled