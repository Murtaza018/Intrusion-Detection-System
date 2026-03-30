import numpy as np
from collections import deque


class GraphBuilder:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.packet_buffer = deque(maxlen=window_size)
        self.ip_to_id = {}
        self.id_counter = 0

    def _get_node_id(self, ip):
        if ip not in self.ip_to_id:
            self.ip_to_id[ip] = self.id_counter
            self.id_counter += 1
        return self.ip_to_id[ip]

    def add_packet(self, packet_info, features, ae_mse=None, mae_err=None, gnn_anomaly=None):
        """
        packet_info: dict with src_ip, dst_ip
        features: 78‑dim scaled feature vector
        ae_mse / mae_err / gnn_anomaly: optional anomaly scores
        """
        src_id = self._get_node_id(packet_info['src_ip'])
        dst_id = self._get_node_id(packet_info['dst_ip'])

        # Anomaly score for this edge (could be one of several; pick one or composite)
        anomaly = gnn_anomaly or ae_mse or mae_err or 0.0

        self.packet_buffer.append((src_id, dst_id, features, float(anomaly)))

    def get_graph_data(self):
        """
        Returns:
            edge_index: (2, N_edges)
            edge_attr: (N_edges, N_feature_dims)
            node_anomaly: dict {node_id: mean anomaly}
        """
        if len(self.packet_buffer) < 10:
            return None, None, {}

        edge_index = []
        edge_attr = []
        node_anomaly = {}

        for src, dst, feat, anom in self.packet_buffer:
            edge_index.append([src, dst])
            edge_attr.append(feat)
            node_anomaly[src] = node_anomaly.get(src, 0.0) + anom
            node_anomaly[dst] = node_anomaly.get(dst, 0.0) + anom

        # Normalize node anomalies
        for nid in node_anomaly:
            node_anomaly[nid] /= len(node_anomaly)

        return (
            np.array(edge_index).T,
            np.array(edge_attr),
            node_anomaly,
        )
