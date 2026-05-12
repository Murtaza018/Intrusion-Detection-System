import numpy as np
from collections import deque

class GraphBuilder:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.packet_buffer = deque(maxlen=window_size)
        self.ip_to_id = {}
        self.id_to_ip = {}      # Reverse lookup for cleanup
        self.ip_ref_count = {}  # Tracks active packets per IP
        self.id_counter = 0

    def _get_node_id(self, ip):
        if ip not in self.ip_to_id:
            self.ip_to_id[ip] = self.id_counter
            self.id_to_ip[self.id_counter] = ip
            self.ip_ref_count[ip] = 0
            self.id_counter += 1
        return self.ip_to_id[ip]

    def _decrement_ref(self, node_id):
        # Safely purges IPs when they fall out of the sliding window
        ip = self.id_to_ip.get(node_id)
        if ip and ip in self.ip_ref_count:
            self.ip_ref_count[ip] -= 1
            if self.ip_ref_count[ip] <= 0:
                del self.ip_ref_count[ip]
                del self.ip_to_id[ip]
                del self.id_to_ip[node_id]

    def add_packet(self, packet_info, features, ae_mse=None, mae_err=None, gnn_anomaly=None):
        src_ip = packet_info['src_ip']
        dst_ip = packet_info['dst_ip']

        src_id = self._get_node_id(src_ip)
        dst_id = self._get_node_id(dst_ip)

        # Catch the oldest packet right before the deque pushes it out
        if len(self.packet_buffer) == self.window_size:
            old_src_id, old_dst_id, _, _ = self.packet_buffer[0]
            self._decrement_ref(old_src_id)
            self._decrement_ref(old_dst_id)

        # Increment reference count for the new packet
        self.ip_ref_count[src_ip] += 1
        self.ip_ref_count[dst_ip] += 1

        anomaly = gnn_anomaly or ae_mse or mae_err or 0.0
        self.packet_buffer.append((src_id, dst_id, features, float(anomaly)))

    def get_graph_data(self):
        if len(self.packet_buffer) < 10:
            return None, None, {}

        edge_index = []
        edge_attr = []
        node_anomaly = {}
        node_counts = {} # NEW: Track exactly how many connections this node has in the window

        for src, dst, feat, anom in self.packet_buffer:
            edge_index.append([src, dst])
            edge_attr.append(feat)
            
            # Sum the anomaly scores
            node_anomaly[src] = node_anomaly.get(src, 0.0) + anom
            node_anomaly[dst] = node_anomaly.get(dst, 0.0) + anom
            
            # Increment the packet count for these specific nodes
            node_counts[src] = node_counts.get(src, 0) + 1
            node_counts[dst] = node_counts.get(dst, 0) + 1

        # Normalize node anomalies by their ACTUAL packet count to get the true average
        for nid in node_anomaly:
            node_anomaly[nid] /= node_counts[nid]

        return (
            np.array(edge_index).T,
            np.array(edge_attr),
            node_anomaly,
        )