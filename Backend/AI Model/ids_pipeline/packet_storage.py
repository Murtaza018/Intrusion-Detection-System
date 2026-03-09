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
        }

class PacketStorage:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.db_lock = threading.Lock()
        self.packet_id_counter = 1
        
        # In-memory stats are only used as a fallback if the DB is unreachable.
        # get_stats() always queries the DB directly — never trust this dict for counts.
        self._meta = {
            "start_time": None,
            "pipeline_status": "stopped",
        }
        
        # Set the ID counter from DB on startup
        self._init_id_counter()

    def _get_conn(self):
        try:
            return psycopg2.connect(
                host=DB_HOST, database=DB_NAME, user=DB_USER,
                password=DB_PASSWORD, port=DB_PORT
            )
        except Exception as e:
            print(f"[!] Database Connection Error: {e}")
            return None

    def _init_id_counter(self):
        """Set packet_id_counter from the DB max ID on startup."""
        with self.db_lock:
            conn = self._get_conn()
            if not conn:
                return
            try:
                cursor = conn.cursor()
                cursor.execute('SELECT MAX(packet_id_backend) FROM packets')
                last_id = cursor.fetchone()[0]
                self.packet_id_counter = (last_id + 1) if last_id is not None else 1
            except Exception as e:
                print(f"[!] ID Counter Init Error: {e}")
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Stats — always read live from DB so truncates / external changes
    # are reflected immediately without a server restart.
    # ------------------------------------------------------------------

    def get_stats(self):
        """
        Always queries the DB for fresh counts.
        Never returns stale in-memory values.
        """
        conn = self._get_conn()
        if not conn:
            # Graceful degradation: return zeros rather than stale numbers
            return {
                "total_packets": 0,
                "normal_count": 0,
                "attack_count": 0,
                "zero_day_count": 0,
                "start_time": self._meta.get("start_time"),
                "pipeline_status": self._meta.get("pipeline_status", "stopped"),
            }
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*)                                            AS total,
                    COUNT(*) FILTER (WHERE status = 'normal')          AS normal,
                    COUNT(*) FILTER (WHERE status = 'known_attack')    AS attack,
                    COUNT(*) FILTER (WHERE status = 'zero_day')        AS zero_day
                FROM packets
            """)
            row = cursor.fetchone()
            return {
                "total_packets":  row[0],
                "normal_count":   row[1],
                "attack_count":   row[2],
                "zero_day_count": row[3],
                "start_time":     self._meta.get("start_time"),
                "pipeline_status": self._meta.get("pipeline_status", "stopped"),
            }
        except Exception as e:
            print(f"[!] get_stats DB Error: {e}")
            return {
                "total_packets": 0, "normal_count": 0,
                "attack_count": 0,  "zero_day_count": 0,
                "start_time": self._meta.get("start_time"),
                "pipeline_status": self._meta.get("pipeline_status", "stopped"),
            }
        finally:
            conn.close()

    def update_stats(self, stats_update):
        """Only used for non-count metadata like start_time and pipeline_status."""
        self._meta.update({
            k: v for k, v in stats_update.items()
            if k in ("start_time", "pipeline_status")
        })

    # ------------------------------------------------------------------
    # Packet writes
    # ------------------------------------------------------------------

    def update_packet_xai_results(self, packet_id, explanation, status, confidence):
        """Update XAI results only — does not touch network metadata or features."""
        with self.db_lock:
            conn = self._get_conn()
            if not conn:
                return
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE packets SET status=%s, confidence=%s, explanation=%s
                    WHERE packet_id_backend=%s
                ''', (status, float(confidence), json.dumps(explanation), packet_id))
                conn.commit()
            except Exception as e:
                print(f"[!] DB XAI Update Error: {e}")
            finally:
                conn.close()

    def add_packet(self, packet_data):
        """Insert new packet or update existing one (smart upsert)."""
        with self.db_lock:
            conn = self._get_conn()
            if not conn:
                return
            try:
                cursor = conn.cursor()
                expl_json = json.dumps(packet_data.explanation) if packet_data.explanation else "{}"
                features_provided = packet_data.features is not None and len(packet_data.features) > 0
                features_json = json.dumps(packet_data.features) if features_provided else "[]"

                cursor.execute(
                    'SELECT id FROM packets WHERE packet_id_backend = %s',
                    (packet_data.id,)
                )
                exists = cursor.fetchone()

                if exists:
                    if features_provided:
                        cursor.execute('''
                            UPDATE packets SET
                                summary=%s, src_ip=%s, dst_ip=%s, protocol=%s, src_port=%s,
                                dst_port=%s, length=%s, timestamp=%s, status=%s,
                                confidence=%s, explanation=%s, features=%s
                            WHERE packet_id_backend=%s
                        ''', (
                            packet_data.summary, packet_data.src_ip, packet_data.dst_ip,
                            packet_data.protocol, packet_data.src_port, packet_data.dst_port,
                            packet_data.length, packet_data.timestamp, packet_data.status,
                            packet_data.confidence, expl_json, features_json, packet_data.id
                        ))
                    else:
                        cursor.execute('''
                            UPDATE packets SET
                                summary=%s, src_ip=%s, dst_ip=%s, protocol=%s, src_port=%s,
                                dst_port=%s, length=%s, timestamp=%s, status=%s,
                                confidence=%s, explanation=%s
                            WHERE packet_id_backend=%s
                        ''', (
                            packet_data.summary, packet_data.src_ip, packet_data.dst_ip,
                            packet_data.protocol, packet_data.src_port, packet_data.dst_port,
                            packet_data.length, packet_data.timestamp, packet_data.status,
                            packet_data.confidence, expl_json, packet_data.id
                        ))
                else:
                    cursor.execute('''
                        INSERT INTO packets (
                            packet_id_backend, summary, src_ip, dst_ip, protocol,
                            src_port, dst_port, length, timestamp, status,
                            confidence, explanation, features
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ''', (
                        packet_data.id, packet_data.summary, packet_data.src_ip,
                        packet_data.dst_ip, packet_data.protocol, packet_data.src_port,
                        packet_data.dst_port, packet_data.length, packet_data.timestamp,
                        packet_data.status, packet_data.confidence, expl_json, features_json
                    ))

                conn.commit()
            except Exception as e:
                print(f"[!] DB Write Error: {e}")
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Packet reads
    # ------------------------------------------------------------------

    def get_packets(self, limit=100, offset=0, status_filter=None):
        """Fetch recent packets for the frontend (metadata only, no features)."""
        with self.db_lock:
            conn = self._get_conn()
            if not conn:
                return []
            try:
                cursor = conn.cursor()
                query = """
                    SELECT packet_id_backend, summary, src_ip, dst_ip, protocol,
                           src_port, dst_port, length, timestamp, status,
                           confidence, explanation
                    FROM packets
                """
                params = []
                if status_filter and status_filter != 'all':
                    query += " WHERE status = %s"
                    params.append(status_filter)

                query += " ORDER BY packet_id_backend DESC LIMIT %s OFFSET %s"
                params.extend([limit, offset])

                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()

                return [
                    Packet(
                        id=r[0], summary=r[1], src_ip=r[2], dst_ip=r[3],
                        protocol=r[4], src_port=r[5], dst_port=r[6],
                        length=r[7], timestamp=str(r[8]), status=r[9],
                        confidence=r[10], explanation=r[11]
                    ).to_dict()
                    for r in rows
                ]
            except Exception as e:
                print(f"[!] DB Read Error: {e}")
                return []
            finally:
                conn.close()

    def get_packet_by_id(self, packet_id):
        with self.db_lock:
            conn = self._get_conn()
            if not conn:
                return None
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT packet_id_backend, summary, src_ip, dst_ip, protocol,
                           src_port, dst_port, length, timestamp, status,
                           confidence, explanation
                    FROM packets WHERE packet_id_backend = %s
                ''', (packet_id,))
                row = cursor.fetchone()
                if row:
                    return Packet(
                        id=row[0], summary=row[1], src_ip=row[2], dst_ip=row[3],
                        protocol=row[4], src_port=row[5], dst_port=row[6],
                        length=row[7], timestamp=str(row[8]), status=row[9],
                        confidence=row[10], explanation=row[11]
                    )
                return None
            finally:
                conn.close()

    def get_features_for_training(self, packet_ids):
        """Retrieve full feature vectors for specific packet IDs."""
        with self.db_lock:
            conn = self._get_conn()
            if not conn:
                return []
            try:
                cursor = conn.cursor()
                ids_tuple = tuple(packet_ids)
                if not ids_tuple:
                    return []
                cursor.execute(
                    "SELECT features FROM packets WHERE packet_id_backend IN %s",
                    (ids_tuple,)
                )
                return [
                    json.loads(row[0])
                    for row in cursor.fetchall()
                    if row[0]
                ]
            except Exception as e:
                print(f"[!] Fetch Features Error: {e}")
                return []
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def get_next_packet_id(self):
        with self.db_lock:
            pid = self.packet_id_counter
            self.packet_id_counter += 1
            return pid

    def clear(self):
        """Truncate the packets table and reset all state."""
        with self.db_lock:
            conn = self._get_conn()
            if not conn:
                return
            try:
                cursor = conn.cursor()
                cursor.execute('TRUNCATE TABLE packets')
                conn.commit()
                self.packet_id_counter = 1
                print("[*] PacketStorage cleared.")
            except Exception as e:
                print(f"[!] Clear Error: {e}")
            finally:
                conn.close()