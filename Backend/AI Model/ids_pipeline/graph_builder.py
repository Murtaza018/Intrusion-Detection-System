import numpy as np
import networkx as nx
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

    def add_packet(self, packet_info, features):
        """
        packet_info: dict from your _get_packet_info (src_ip, dst_ip, etc.)
        features: the scaled feature vector from your extractor
        """
        src_id = self._get_node_id(packet_info['src_ip'])
        dst_id = self._get_node_id(packet_info['dst_ip'])
        
        # Store edge data: (source, destination, feature_vector)
        self.packet_buffer.append((src_id, dst_id, features))

    def get_graph_data(self):
        """Returns the data structure needed for GNN training/inference"""
        if len(self.packet_buffer) < 10:
            return None, None

        # Create edge list for GNN (e.g., PyTorch Geometric format)
        edge_index = []
        edge_attr = []
        
        for src, dst, feat in self.packet_buffer:
            edge_index.append([src, dst])
            edge_attr.append(feat)

        return np.array(edge_index).T, np.array(edge_attr)