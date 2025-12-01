# packet_storage.py
# Thread-safe packet storage for Flutter

from collections import deque
import threading
from datetime import datetime
import json

class Packet:
    """Packet data structure"""
    def __init__(self, id, summary, src_ip, dst_ip, protocol, src_port, dst_port, 
                 length, timestamp, status, confidence=0.0, explanation=None):
        self.id = id
        self.summary = summary
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.protocol = protocol
        self.src_port = src_port
        self.dst_port = dst_port
        self.length = length
        self.timestamp = timestamp
        self.status = status
        self.confidence = confidence
        self.explanation = explanation
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "summary": self.summary,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "protocol": self.protocol,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "length": self.length,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "status": self.status,
            "confidence": float(self.confidence),
            "explanation": self.explanation
        }

class PacketStorage:
    """Thread-safe storage for packets"""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.packets = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.stats = {
            "total_packets": 0,
            "normal_count": 0,
            "attack_count": 0,
            "zero_day_count": 0,
            "start_time": None,
            "memory_usage_mb": 0,
            "pipeline_status": "stopped"
        }
        self.stats_lock = threading.Lock()
        self.packet_id_counter = 1

    def add_packet(self, packet_data):
        """Add a packet to storage"""
        with self.lock:
            # Remove existing packet with same ID if present
            self.packets = deque([p for p in self.packets if p.id != packet_data.id], maxlen=self.max_size)
            self.packets.appendleft(packet_data)

    def get_packets(self, limit=None):
        """Get packets with optional limit"""
        with self.lock:
            packets = list(self.packets)
            if limit and len(packets) > limit:
                return [p.to_dict() for p in packets[:limit]]
            return [p.to_dict() for p in packets]

    def update_stats(self, stats_update):
        """Update statistics"""
        with self.stats_lock:
            self.stats.update(stats_update)

    def get_stats(self):
        """Get current statistics"""
        with self.stats_lock:
            return self.stats.copy()

    def clear(self):
        """Clear all packets and reset stats"""
        with self.lock:
            self.packets.clear()
        with self.stats_lock:
            self.stats.update({
                "total_packets": 0,
                "normal_count": 0,
                "attack_count": 0,
                "zero_day_count": 0
            })
    
    def get_next_packet_id(self):
        """Get next packet ID"""
        with self.stats_lock:
            pid = self.packet_id_counter
            self.packet_id_counter += 1
            return pid
    
    def format_explanation_for_flutter(self, explanation):
        """Format explanation for Flutter display"""
        if isinstance(explanation, dict):
            return explanation
        elif isinstance(explanation, str):
            return {"text": explanation}
        else:
            return {"raw": str(explanation)}