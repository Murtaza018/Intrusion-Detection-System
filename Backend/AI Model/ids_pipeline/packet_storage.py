# packet_storage.py
# Persistent PostgreSQL storage (Now with Features!)

import psycopg2
import threading
import json
from datetime import datetime
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

class Packet:
    def __init__(self, id, summary, src_ip, dst_ip, protocol, src_port, dst_port, 
                 length, timestamp, status, confidence=0.0, explanation=None, features=None):
        self.id = id
        self.summary = summary
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.protocol = protocol
        self.src_port = src_port
        self.dst_port = dst_port
        self.length = length
        
        if isinstance(timestamp, str):
            try:
                self.timestamp = datetime.fromisoformat(timestamp)
            except:
                self.timestamp = datetime.now()
        else:
            self.timestamp = timestamp
            
        self.status = status
        self.confidence = float(confidence)
        
        if isinstance(explanation, str):
            try:
                self.explanation = json.loads(explanation)
            except:
                self.explanation = {"text": explanation}
        else:
            self.explanation = explanation

        # --- NEW: Store Features ---
        # Can be a list of floats or a JSON string
        if isinstance(features, str):
            try:
                self.features = json.loads(features)
            except:
                self.features = []
        else:
            self.features = features if features else []
    
    def to_dict(self):
        return {
            "id": self.id,
            "summary": self.summary,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "protocol": self.protocol,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "length": self.length,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "confidence": self.confidence,
            "explanation": self.explanation,
            # We generally DO NOT send features to frontend to save bandwidth
            # "features": self.features 
        }

class PacketStorage:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.db_lock = threading.Lock()
        self.packet_id_counter = 1
        
        self.stats = {
            "total_packets": 0, "normal_count": 0, "attack_count": 0, "zero_day_count": 0,
            "start_time": None, "pipeline_status": "stopped"
        }
        self._refresh_stats()

    def _get_conn(self):
        try:
            return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
        except Exception as e:
            print(f"[!] Database Connection Error: {e}")
            return None

    def _refresh_stats(self):
        with self.db_lock:
            conn = self._get_conn()
            if not conn: return
            try:
                cursor = conn.cursor()
                cursor.execute('SELECT count(*) FROM packets')
                self.stats["total_packets"] = cursor.fetchone()[0]
                cursor.execute("SELECT count(*) FROM packets WHERE status='normal'")
                self.stats["normal_count"] = cursor.fetchone()[0]
                cursor.execute("SELECT count(*) FROM packets WHERE status='known_attack'")
                self.stats["attack_count"] = cursor.fetchone()[0]
                cursor.execute("SELECT count(*) FROM packets WHERE status='zero_day'")
                self.stats["zero_day_count"] = cursor.fetchone()[0]
                
                cursor.execute('SELECT MAX(packet_id_backend) FROM packets')
                last_id = cursor.fetchone()[0]
                if last_id is not None:
                    self.packet_id_counter = last_id + 1
            except Exception as e:
                print(f"[!] Stats Refresh Error: {e}")
            finally:
                conn.close()


    def add_packet(self, packet_data):
        """Insert new packet or Update existing packet (Smart Update)"""
        with self.db_lock:
            conn = self._get_conn()
            if not conn: return
            
            try:
                cursor = conn.cursor()
                expl_json = json.dumps(packet_data.explanation) if packet_data.explanation else "{}"
                
                # Check if features are provided in this specific update
                features_provided = packet_data.features is not None and len(packet_data.features) > 0
                features_json = json.dumps(packet_data.features) if features_provided else "[]"
                
                # Check if packet already exists
                cursor.execute('SELECT id FROM packets WHERE packet_id_backend = %s', (packet_data.id,))
                exists = cursor.fetchone()
                
                if exists:
                    # --- UPDATE LOGIC ---
                    if features_provided:
                        # Case A: We have new features (rare, but safe to overwrite)
                        cursor.execute('''
                            UPDATE packets SET 
                                summary=%s, src_ip=%s, dst_ip=%s, protocol=%s, src_port=%s, 
                                dst_port=%s, length=%s, timestamp=%s, status=%s, confidence=%s, explanation=%s, features=%s
                            WHERE packet_id_backend=%s
                        ''', (
                            packet_data.summary, packet_data.src_ip, packet_data.dst_ip, packet_data.protocol,
                            packet_data.src_port, packet_data.dst_port, packet_data.length,
                            packet_data.timestamp, packet_data.status, packet_data.confidence,
                            expl_json, features_json, packet_data.id
                        ))
                    else:
                        # Case B: No features provided in this update (e.g., XAI update). 
                        # CRITICAL: Do NOT include 'features' in the SET clause. Preserve existing data!
                        cursor.execute('''
                            UPDATE packets SET 
                                summary=%s, src_ip=%s, dst_ip=%s, protocol=%s, src_port=%s, 
                                dst_port=%s, length=%s, timestamp=%s, status=%s, confidence=%s, explanation=%s
                            WHERE packet_id_backend=%s
                        ''', (
                            packet_data.summary, packet_data.src_ip, packet_data.dst_ip, packet_data.protocol,
                            packet_data.src_port, packet_data.dst_port, packet_data.length,
                            packet_data.timestamp, packet_data.status, packet_data.confidence,
                            expl_json, packet_data.id
                        ))
                else:
                    # --- INSERT LOGIC (Always include features) ---
                    # If features missing on insert, we default to empty list
                    cursor.execute('''
                        INSERT INTO packets (packet_id_backend, summary, src_ip, dst_ip, protocol, src_port, dst_port, length, timestamp, status, confidence, explanation, features)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        packet_data.id, packet_data.summary, packet_data.src_ip, packet_data.dst_ip,
                        packet_data.protocol, packet_data.src_port, packet_data.dst_port, packet_data.length,
                        packet_data.timestamp, packet_data.status, packet_data.confidence, expl_json, features_json
                    ))
                    
                    # Update Stats
                    self.stats["total_packets"] += 1
                    if packet_data.status == 'normal': self.stats["normal_count"] += 1
                    elif packet_data.status == 'known_attack': self.stats["attack_count"] += 1
                    elif packet_data.status == 'zero_day': self.stats["zero_day_count"] += 1
                
                conn.commit()
            except Exception as e:
                print(f"[!] DB Write Error: {e}")
            finally:
                conn.close()

    def get_packets(self, limit=100, offset=0, status_filter=None):
        """Get recent packets (Metadata only, for Frontend)"""
        with self.db_lock:
            conn = self._get_conn()
            if not conn: return []
            try:
                cursor = conn.cursor()
                query = "SELECT packet_id_backend, summary, src_ip, dst_ip, protocol, src_port, dst_port, length, timestamp, status, confidence, explanation FROM packets"
                params = []
                
                if status_filter and status_filter != 'all':
                    query += " WHERE status = %s"
                    params.append(status_filter)
                
                query += " ORDER BY packet_id_backend DESC LIMIT %s OFFSET %s"
                params.extend([limit, offset])
                
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                
                packets = []
                for row in rows:
                    p = Packet(
                        id=row[0], summary=row[1], src_ip=row[2], dst_ip=row[3],
                        protocol=row[4], src_port=row[5], dst_port=row[6],
                        length=row[7], timestamp=str(row[8]), status=row[9],
                        confidence=row[10], explanation=row[11]
                    )
                    packets.append(p.to_dict())
                return packets
            except Exception as e:
                print(f"[!] DB Read Error: {e}")
                return []
            finally:
                conn.close()

    def get_features_for_training(self, packet_ids):
        """NEW: Retrieve full feature vectors for specific IDs"""
        with self.db_lock:
            conn = self._get_conn()
            if not conn: return []
            try:
                cursor = conn.cursor()
                # Convert list of IDs to tuple for SQL IN clause
                ids_tuple = tuple(packet_ids)
                if not ids_tuple: return []
                
                query = "SELECT features FROM packets WHERE packet_id_backend IN %s"
                cursor.execute(query, (ids_tuple,))
                rows = cursor.fetchall()
                
                feature_list = []
                for row in rows:
                    if row[0]: # If features exist
                        feature_list.append(json.loads(row[0]))
                return feature_list
            except Exception as e:
                print(f"[!] Fetch Features Error: {e}")
                return []
            finally:
                conn.close()

    # ... (get_packet_by_id, update_stats, get_stats, get_next_packet_id, clear remain same) ...
    # Be sure to include them or copy them from the previous version if needed!
    def get_packet_by_id(self, packet_id):
        with self.db_lock:
            conn = self._get_conn()
            if not conn: return None
            try:
                cursor = conn.cursor()
                cursor.execute('SELECT packet_id_backend, summary, src_ip, dst_ip, protocol, src_port, dst_port, length, timestamp, status, confidence, explanation FROM packets WHERE packet_id_backend = %s', (packet_id,))
                row = cursor.fetchone()
                if row:
                    return Packet(id=row[0], summary=row[1], src_ip=row[2], dst_ip=row[3], protocol=row[4], src_port=row[5], dst_port=row[6], length=row[7], timestamp=str(row[8]), status=row[9], confidence=row[10], explanation=row[11])
                return None
            finally:
                conn.close()

    def update_stats(self, stats_update): self.stats.update(stats_update)
    def get_stats(self): return self.stats.copy()
    def get_next_packet_id(self):
        with self.db_lock:
            pid = self.packet_id_counter
            self.packet_id_counter += 1
            return pid
    def clear(self):
        with self.db_lock:
            conn = self._get_conn()
            if not conn: return
            try:
                cursor = conn.cursor()
                cursor.execute('TRUNCATE TABLE packets')
                conn.commit()
                self._refresh_stats()
            finally:
                conn.close()