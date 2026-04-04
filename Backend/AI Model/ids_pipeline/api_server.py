from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
from functools import wraps
import psutil
import traceback
import sys
import os
import joblib
import json
import numpy as np
import logging
from cryptography.hazmat.primitives.asymmetric import utils
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from retrain_manager import RetrainManager, JobStatus
from gan_retrainer import GanRetrainer, JitterRetrainer
from config import API_KEY
from detector import Detector
from datetime import datetime,timedelta,timezone




def _ts_to_epoch(obj):
    """
    Convert a dict with:
      "timestamp" (ISO‑string) to int epoch.
    """
    utc = datetime.now().astimezone()  # fallback to now
    ts = obj.get("timestamp")

    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except Exception as e:
            print(f"ts parse error: {e}")
    return int(utc.timestamp())


def calculate_group_consistency(feature_list):
    if not feature_list or len(feature_list) < 2: return 1.0
    matrix = np.array(feature_list)
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(matrix)
        iu = np.triu_indices(len(sim_matrix), k=1)
        if len(iu[0]) == 0: return 1.0
        return float(np.mean(sim_matrix[iu]))
    except:
        return 0.0


class APIServer:

    def __init__(self, packet_storage, feature_extractor, pipeline_manager, model_loader):
        self.app = Flask(__name__)
        CORS(self.app)

        self.packet_storage    = packet_storage
        self.feature_extractor = feature_extractor
        self.pipeline_manager  = pipeline_manager
        self.model_loader      = model_loader

        # GAN retrainer (for attack packets + new zero-days)
        self.gan_retrainer = GanRetrainer(packet_storage, feature_extractor, model_loader)

        # Jitter retrainer (for false positives — normal packets flagged as attack)
        self.jitter_retrainer = JitterRetrainer(
            packet_storage    = packet_storage,
            feature_extractor = feature_extractor,
            model_loader      = model_loader,
            replay_path       = self.gan_retrainer.replay_path,
            encoder_path      = self.gan_retrainer.encoder_path,
        )

        # Manager knows about both
        self.retrain_manager = RetrainManager(self.gan_retrainer, self.jitter_retrainer)

        self._initialize_ecc()
        self._register_routes()

    # ------------------------------------------------------------------
    # ECC
    # ------------------------------------------------------------------

    def _initialize_ecc(self):
        try:
            script_dir   = os.path.dirname(os.path.abspath(__file__))
            backend_base = os.path.abspath(os.path.join(script_dir, "..", ".."))
            key_path     = os.path.join(backend_base, "ECC", "key.pem")
            if not os.path.exists(key_path):
                key_path = os.path.abspath("Backend/ECC/key.pem")
            with open(key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(f.read(), password=None)
            print(f"[*] ECC loaded successfully.")
        except Exception as e:
            print(f"[!] ECC Init Error: {e}")
            self.private_key = None

    def _generate_signature(self, data_dict):
        if not self.private_key:
            return "NO_KEY_LOADED"
        try:
            json_str = json.dumps(data_dict, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            sig_der  = self.private_key.sign(json_str.encode('utf-8'), ec.ECDSA(hashes.SHA256()))
            r, s     = utils.decode_dss_signature(sig_der)
            return (r.to_bytes(32, 'big') + s.to_bytes(32, 'big')).hex()
        except Exception as e:
            print(f"[!] Signing Error: {e}")
            return "SIGNING_FAILED"

    def _secure_response(self, data, status=200):
        body = {
            "payload":        data,
            "signature":      self._generate_signature(data),
            "signature_type": "ECDSA_SHA256",
            "server_time":    datetime.now().isoformat(),
        }
        return jsonify(body), status

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _register_routes(self):

        def _require_api_key(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                if request.headers.get('X-API-Key') != API_KEY:
                    return jsonify({"error": "Invalid API key"}), 401
                return f(*args, **kwargs)
            return decorated

        self._require_api_key = _require_api_key

        def _slot_of(ts: int, bucket: int) -> int:
            return ts - (ts % bucket)


        def _window_to_timedelta(window: str) -> timedelta:
            if window == "1d":
                return timedelta(days=1)
            elif window == "1w":
                return timedelta(weeks=1)
            else:
                raise ValueError("Unsupported window")


        @self.app.route("/api/report/<string:window>", methods=['GET'])
        @self._require_api_key
        def get_summary_report(window):
            """
            window: "1d" or "1w"
            """
            try:
                now = datetime.now(timezone.utc)
                end = now
                start = end - _window_to_timedelta(window)

                # For demo, just use same query as history but with more details
                all_packets = self.packet_storage.get_packets(limit=100_000, status_filter=None)

                # Filter by time window
                packets=all_packets
                # packets = []
                for p in all_packets:
                    ts_str = p.get("timestamp")
                    if not ts_str:
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

                        # Ensure timezone-aware UTC
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        else:
                            ts = ts.astimezone(timezone.utc)

                    except Exception:
                        continue
                    if start <= ts <= end:
                        packets.append(p)

                # Simple stats
                counts = {"normal": 0, "attack": 0, "zero_day": 0}
                mae_values = []

                for p in packets:
                    status = p.get("status", "unknown")
                    expl = p.get("explanation", {}) or {}

                    mae = float(expl.get("mae_anomaly", 0.0))
                    if 0.0 < mae <= 1.0:
                        mae_values.append(mae)

                    if status == "normal":
                        counts["normal"] += 1
                    elif status == "known_attack":
                        counts["attack"] += 1
                    elif status == "zero_day":
                        counts["zero_day"] += 1

                # 1. Explainable insight 1: MAE clusters
                if mae_values:
                    avg_mae = sum(mae_values) / len(mae_values)
                    high_mae = [m for m in mae_values if m > 0.3]
                else:
                    avg_mae = 0.0
                    high_mae = []

                # 2. Simple executive‑style text
                total = counts["normal"] + counts["attack"] + counts["zero_day"]
                if total == 0:
                    text = "No traffic observed in the selected period."
                else:
                    if counts["zero_day"] > 0:
                        text = (
                            f"During the last {window}, the system observed {total} packets, "
                            f"with {counts['attack'] + counts['zero_day']} classified as malicious "
                            f"({counts['zero_day']} classified as zero‑day). "
                            f"This indicates the presence of known attack patterns and at least "
                            f"one potentially novel intrusion pattern. "
                        )
                    else:
                        text = (
                            f"During the last {window}, the system observed {total} packets, "
                            f"with {counts['attack']} classified as attacks. "
                            f"Most traffic is benign, but there are repeated attack attempts."
                        )

                payload = {
                    "window": window,
                    "start_time": start.isoformat(),
                    "end_time": end.isoformat(),
                    "summary_text": text,
                    "stats": {
                        "total_packets": total,
                        "normal": counts["normal"],
                        "attack": counts["attack"],
                        "zero_day": counts["zero_day"],
                        "detection_rate": round(
                            (counts["attack"] + counts["zero_day"]) / max(1, total),
                            4,
                        ),
                    },
                    "explainables": {
                        "mae_anomaly": {
                            "avg": round(avg_mae, 4),
                            "high_mae_count": len(high_mae),
                            "high_mae_example": (high_mae[0] if high_mae else 0.0),
                        },
                    },
                }
                return self._secure_response(payload)

            except Exception as e:
                traceback.print_exc()
                return self._secure_response({"error": str(e)}, 500)

        # ── 1. Pipeline control ───────────────────────────────────────

        @self.app.route("/api/pipeline/start", methods=['POST'])
        @_require_api_key
        def start_pipeline():
            try:
                success = self.pipeline_manager.start()
                if success:
                    return self._secure_response({
                        "status": "started",
                        "message": "IDS pipeline started",
                        "start_time": datetime.now().isoformat(),
                    })
                return self._secure_response({"error": "Failed to start pipeline"}, 500)
            except Exception as e:
                traceback.print_exc()
                return self._secure_response({"error": str(e)}, 500)

        @self.app.route("/api/pipeline/stop", methods=['POST'])
        @_require_api_key
        def stop_pipeline():
            try:
                self.pipeline_manager.stop()
                return self._secure_response({"status": "stopped", "message": "IDS pipeline stopped"})
            except Exception as e:
                return self._secure_response({"error": str(e)}, 500)

        @self.app.route("/api/pipeline/status", methods=['GET'])
        @_require_api_key
        def get_pipeline_status():
            stats = self.packet_storage.get_stats()
            return self._secure_response({
                "running":           self.pipeline_manager.is_running(),
                "start_time":        stats.get("start_time"),
                "packets_processed": stats["total_packets"],
            })

        # ── 2. Data fetching ──────────────────────────────────────────

        @self.app.route("/api/packets/recent", methods=['GET'])
        @_require_api_key
        def get_recent_packets():
            limit  = request.args.get('limit',  default=10,   type=int)
            offset = request.args.get('offset', default=0,    type=int)
            status = request.args.get('status', default=None, type=str)
            packets = self.packet_storage.get_packets(limit=limit, offset=offset, status_filter=status)
            for p in packets:
                p['confidence'] = "{:.4f}".format(float(p.get('confidence', 0.0)))
                if isinstance(p.get('explanation'), dict):
                    exp = p['explanation']
                    for key in ('gnn_anomaly', 'mae_anomaly'):
                        if key in exp:
                            exp[key] = "{:.4f}".format(float(exp.get(key, 0.0)))
            return self._secure_response({"packets": packets, "count": len(packets)})

        @self.app.route("/api/sensory/live", methods=['GET'])
        @_require_api_key
        def get_live_sensory():
            recent = self.packet_storage.get_packets(limit=1)
            if recent:
                expl = recent[0].get('explanation', {})
                return self._secure_response({
                    "gnn_anomaly": "{:.4f}".format(float(expl.get('gnn_anomaly', 0.0))),
                    "mae_anomaly": "{:.4f}".format(float(expl.get('mae_anomaly', 0.0))),
                    "status":      recent[0].get('status', 'unknown'),
                })
            return self._secure_response({"gnn_anomaly": 0.0, "mae_anomaly": 0.0})

        @self.app.route("/api/stats", methods=['GET'])
        @_require_api_key
        def get_stats():
            stats = self.packet_storage.get_stats()
            recent = self.packet_storage.get_packets(limit=50)
            mae_vals = [p.get('explanation', {}).get('mae_anomaly', 0) for p in recent]
            stats["avg_visual_anomaly"] = round(float(np.mean(mae_vals)), 4) if mae_vals else 0.0
            try:
                stats["memory_usage_mb"] = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
            except:
                stats["memory_usage_mb"] = 0.0
            return self._secure_response(stats)

        # ── 3. Labels ─────────────────────────────────────────────────

        @self.app.route("/api/labels", methods=['GET'])
        @_require_api_key
        def get_labels():
            try:
                for path in ["ids_pipeline/label_encoder.pkl", "label_encoder.pkl"]:
                    if os.path.exists(path):
                        encoder = joblib.load(path)
                        return self._secure_response({"labels": list(encoder.classes_)})
            except:
                pass
            return self._secure_response({"labels": ["BENIGN", "DDoS", "PortScan", "Bot", "WebAttack"]})

        # ── 4. Continual learning — consistency analysis ──────────────

        @self.app.route("/api/analyze_selection", methods=['POST'])
        @_require_api_key
        def analyze_selection():
            data    = request.json
            gan_ids = [p['id'] for p in data.get('gan_queue', [])]
            payload = {"gan_score": 0.0, "gan_status": "Insufficient Data"}

            if gan_ids:
                feats = [f for f in self.packet_storage.get_features_for_training(gan_ids)
                         if f is not None and len(f) > 0]
                if len(feats) > 1:
                    score = calculate_group_consistency(feats)
                    payload["gan_score"] = round(score, 4)
                    payload["gan_status"] = (
                        "Excellent (Homogeneous)" if score > 0.9 else
                        "Mixed (Caution)"         if score > 0.6 else
                        "Poor (Too Diverse)"
                    )
                elif len(feats) == 1:
                    payload["gan_score"]  = 1.0
                    payload["gan_status"] = "Single Item"

            return self._secure_response(payload)

        # ── 5. Retrain — async, routes to GAN or Jitter ──────────────
        #
        # Routing logic:
        #   - jitter_queue has packets AND gan_queue is empty → jitter pipeline
        #   - gan_queue has packets (with or without jitter)  → gan pipeline
        #   - both queues have packets                        → gan pipeline
        #     (GAN is the heavier correction; jitter packets
        #      are included in the packet_ids for context)

        @self.app.route("/api/retrain", methods=['POST'])
        @_require_api_key
        def trigger_retrain():
            data = request.json
            if not data:
                return self._secure_response({"error": "No data provided"}, 400)

            gan_packets    = data.get('gan_queue', [])
            jitter_packets = data.get('jitter_queue', [])
            target_label   = data.get('target_label', 'BENIGN')
            is_new_label   = data.get('is_new_label', False)

            if not gan_packets and not jitter_packets:
                return self._secure_response({"error": "No packets provided"}, 400)

            # Decide pipeline
            if jitter_packets and not gan_packets:
                pipeline   = 'jitter'
                packet_ids = [p['id'] for p in jitter_packets]
                # Jitter always corrects toward a known label — never a new class
                is_new_label = False
            else:
                pipeline   = 'gan'
                # Include both queues so the GAN has more seed data
                packet_ids = [p['id'] for p in gan_packets] + \
                             [p['id'] for p in jitter_packets]

            job, accepted = self.retrain_manager.submit(
                packet_ids = packet_ids,
                label      = target_label,
                is_new     = is_new_label,
                pipeline   = pipeline,
            )

            if not accepted:
                return self._secure_response({
                    "error":   "A retrain job is already running",
                    "current": self.retrain_manager.get_status(),
                }, 409)

            return self._secure_response({
                "job_id":   job.job_id,
                "status":   job.status.value,
                "pipeline": job.pipeline,
                "label":    job.label,
                "message":  f"Retrain job '{job.job_id}' queued "
                            f"[{pipeline}] for label '{target_label}'.",
            }, 202)

        @self.app.route("/api/retrain/status", methods=['GET'])
        @_require_api_key
        def retrain_status():
            status = self.retrain_manager.get_status()
            if status is None:
                return self._secure_response({"status": "idle", "message": "No retrain job has run yet."})
            return self._secure_response(status)

        @self.app.route("/api/retrain/cancel", methods=['POST'])
        @_require_api_key
        def retrain_cancel():
            job = self.retrain_manager.get_job()
            if job is None or job.status.value in (
                JobStatus.DONE.value, JobStatus.FAILED.value, JobStatus.ROLLED_BACK.value
            ):
                return self._secure_response({"message": "No active job to cancel."})
            job.status = JobStatus.FAILED
            job.phase  = "cancelled"
            job.error  = "Cancelled by user"
            return self._secure_response({"message": f"Job {job.job_id} marked as cancelled."})
        

        # @self.app.route("/api/graph", methods=['GET'])
        # @self._require_api_key
        # def get_graph():
        #     """
        #     Expose live network graph topology + anomaly hot‑spots.
        #     Requires X-API-Key (ECC‑signed JSON response).
        #     """
        #     try:
        #         # Get detector from pipeline_manager
        #         if not hasattr(self.pipeline_manager, 'detector'):
        #             return self._secure_response(
        #                 {"error": "Detector not initialized"}, 500
        #             )

        #         detector = self.pipeline_manager.detector
        #         if not hasattr(detector, "get_graph_snapshot"):
        #             return self._secure_response(
        #                 {"error": "Detector missing get_graph_snapshot method"}, 500
        #             )

        #         # 1. Get snapshot
        #         data = detector.get_graph_snapshot()
        #         if data is None:
        #             return self._secure_response(
        #                 {"nodes": [], "edges": [], "message": "Graph empty or not initialized."},
        #             )

        #         # 2. Add metadata (optional, for UI)
        #         stats = self.packet_storage.get_stats()
        #         data["total_packets"] = stats.get("total_packets", 0)
        #         data["threats"] = stats.get("attack_count", 0)
        #         data["zero_days"] = stats.get("zero_day_count", 0)

        #         return self._secure_response(data)

        #     except Exception as e:
        #         traceback.print_exc()
        #         return self._secure_response(
        #             {"error": str(e)},
        #             500
        #         )

        @self.app.route("/api/graph", methods=['GET'])
        @self._require_api_key
        def get_graph():
            """
            Expose live network graph topology + anomaly hot‑spots.
            Requires X-API-Key (ECC‑signed JSON response).
            """
            try:
                # Get detector from pipeline_manager
                if not hasattr(self.pipeline_manager, 'detector'):
                    return self._secure_response(
                        {"error": "Detector not initialized"}, 500
                    )

                detector = self.pipeline_manager.detector
                if not hasattr(detector, "get_graph_snapshot"):
                    return self._secure_response(
                        {"error": "Detector missing get_graph_snapshot method"}, 500
                    )

                # 1. Get snapshot
                data = detector.get_graph_snapshot()
                if data is None:
                    return self._secure_response(
                        {"nodes": [], "edges": [], "message": "Graph empty or not initialized."},
                    )

                # --- NEW LOGIC: ADD SUBNET, DMZ, & GATEWAY DATA ---
                if "nodes" in data:
                    for node in data["nodes"]:
                        ip = node.get("ip", "")
                        
                        # 1. Gateway Detection (typically ends in .1 or .254)
                        if ip.endswith(".1") or ip.endswith(".254"):
                            node["isGateway"] = True
                        else:
                            node["isGateway"] = False
                            
                        # 2. DMZ Detection (Example: you can customize these IPs based on your dataset)
                        if ip.startswith("172.16.") or ip in ["192.168.1.10", "192.168.1.11", "10.0.0.5"]:
                            node["isDmz"] = True
                            node["subnet"] = "DMZ-Zone"
                        else:
                            node["isDmz"] = False
                            
                        # 3. Standard Subnet Mapping
                        if not node["isDmz"]:
                            parts = ip.split('.')
                            if len(parts) == 4:
                                node["subnet"] = f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
                            else:
                                node["subnet"] = "External / Unknown"
                # --------------------------------------------------

                # 2. Add metadata (optional, for UI)
                stats = self.packet_storage.get_stats()
                data["total_packets"] = stats.get("total_packets", 0)
                data["threats"] = stats.get("attack_count", 0)
                data["zero_days"] = stats.get("zero_day_count", 0)

                return self._secure_response(data)

            except Exception as e:
                traceback.print_exc()
                return self._secure_response(
                    {"error": str(e)},
                    500
                )

        # inside class APIServer
# ── 6. Historical Alert Analytics ──────────────────────────────────

        # @self.app.route("/api/history", methods=['GET'])
        # @self._require_api_key
        # def get_alert_history():
        #     try:
        #         window = request.args.get('window', '24h', type=str)
        #         limit  = request.args.get('limit',  10_000, type=int)

        #         all_packets = self.packet_storage.get_packets(limit=limit, status_filter=None)

        #         BUCKET = 3600  # 1‑hour bucket
        #         by_time = {}
        #         by_type = {}
        #         total_anomaly = 0.0
        #         count_anomaly = 0

        #         for p in all_packets:
        #             status = p.get("status", "unknown")
        #             expl   = p.get("explanation", {}) or {}

        #             mae = float(expl.get("mae_anomaly", 0.0))
        #             if 0.0 < mae < 1.0:
        #                 total_anomaly += mae
        #                 count_anomaly += 1

        #             # 1. your three labels: normal, attack, zero_day
        #             if status == "normal":
        #                 label = "normal"
        #             elif status == "known_attack":
        #                 label = "attack"
        #             elif status == "zero_day":
        #                 label = "zero_day"
        #             else:
        #                 label = "unknown"

        #             by_type[label] = by_type.get(label, 0) + 1

        #             # 2. use _ts_to_epoch with your ISO timestamp field
        #             ts = _ts_to_epoch(p)  # <-- from above
        #             if ts == 0:
        #                 continue
        #             bucket = ts - (ts % BUCKET)

        #             by_time[bucket] = by_time.get(bucket, 0) + 1

        #         # 3. time series
        #         sorted_time = sorted(by_time.items())
        #         alerts_volume = [
        #             {"timestamp": k, "count": v}
        #             for k, v in sorted_time
        #         ]

        #         by_type_items = [
        #             {"label": k, "count": v}
        #             for k, v in by_type.items()
        #         ]

        #         avg_anomaly = (total_anomaly / count_anomaly) if count_anomaly else 0.0

        #         payload = {
        #             "window": window,
        #             "alerts_volume": alerts_volume,
        #             "by_type": by_type_items,
        #             "performance": {
        #                 "avg_anomaly": round(avg_anomaly, 4),
        #                 "total_alerts": len(by_time),
        #                 "total_packets": len(all_packets),
        #                 "detection_rate": round(
        #                     sum(by_type.get(l, 0) for l in ["attack", "zero_day"]) / max(len(all_packets), 1),
        #                     4
        #                 ),
        #                 "fp_estimate": 0.0,
        #             }
        #         }
        #         return self._secure_response(payload)

        #     except Exception as e:
        #         traceback.print_exc()
        #         return self._secure_response({"error": str(e)}, 500)   


        @self.app.route("/api/history", methods=['GET'])
        @self._require_api_key
        def get_alert_history():
            try:
                # 1. We increase the default limit to catch more traffic history
                window = request.args.get('window', '24h', type=str)
                limit  = request.args.get('limit',  50_000, type=int)

                all_packets = self.packet_storage.get_packets(limit=limit, status_filter=None)
                
                if not all_packets:
                    return self._secure_response({"alerts_volume": [], "by_type": [], "performance": {}})

                cutoff_seconds = 86400
                if window == '1h':
                    cutoff_seconds = 3600
                elif window == '24h':
                    cutoff_seconds = 86400
                elif window == '7d':
                    cutoff_seconds = 86400 * 7
                elif window == '30d':
                    cutoff_seconds = 86400 * 30

                # --- THE FIX: Anchor time to the newest packet ---
                # This guarantees accuracy even if timezones drift or you replay old PCAPs
                latest_ts = 0
                for p in all_packets:
                    ts = _ts_to_epoch(p)
                    if ts > latest_ts:
                        latest_ts = ts
                
                start_epoch = latest_ts - cutoff_seconds

                if window == '1h':
                    BUCKET = 300       
                elif window in ['7d', '30d']:
                    BUCKET = 86400     
                else:
                    BUCKET = 3600      

                by_time = {}
                by_type = {}
                total_anomaly = 0.0
                count_anomaly = 0
                valid_packets_count = 0

                total_attack_confidence = 0.0
                attack_count = 0

                for p in all_packets:
                    ts = _ts_to_epoch(p)
                    if ts == 0 or ts < start_epoch:
                        continue

                    valid_packets_count += 1
                    
                    status = p.get("status", "unknown")
                    expl   = p.get("explanation", {}) or {}

                    conf = float(p.get("confidence", 0.85))

                    if status in ["known_attack", "zero_day"]:
                        total_attack_confidence += conf
                        attack_count += 1

                    mae = float(expl.get("mae_anomaly", 0.0))
                    if 0.0 < mae < 1.0:
                        total_anomaly += mae
                        count_anomaly += 1

                    if status == "normal":
                        label = "normal"
                    elif status == "known_attack":
                        label = "attack"
                    elif status == "zero_day":
                        label = "zero_day"
                    else:
                        label = "unknown"

                    by_type[label] = by_type.get(label, 0) + 1
                    bucket = ts - (ts % BUCKET)
                    by_time[bucket] = by_time.get(bucket, 0) + 1

                sorted_time = sorted(by_time.items())
                alerts_volume = [{"timestamp": k, "count": v} for k, v in sorted_time]
                by_type_items = [{"label": k, "count": v} for k, v in by_type.items()]

                avg_anomaly = (total_anomaly / count_anomaly) if count_anomaly else 0.0

                # 1. Calculate the real total of malicious alerts first
                real_total_alerts = sum(by_type.get(l, 0) for l in ["attack", "zero_day"])

                fp_estimate = 0.0
                if attack_count > 0:
                    avg_confidence = total_attack_confidence / attack_count
                    fp_estimate = 1.0 - avg_confidence  # If 80% confident, 20% FP chance

                payload = {
                    "window": window,
                    "alerts_volume": alerts_volume,
                    "by_type": by_type_items,
                    "performance": {
                        "avg_anomaly": round(avg_anomaly, 4),
                        
                        # 2. FIX: Use the actual count instead of len(by_time)
                        "total_alerts": real_total_alerts, 
                        
                        "total_packets": valid_packets_count,
                        "detection_rate": round(
                            real_total_alerts / max(valid_packets_count, 1), 4
                        ),
                        "fp_estimate": round(fp_estimate, 4),
                    }
                }
                return self._secure_response(payload)

            except Exception as e:
                traceback.print_exc()
                return self._secure_response({"error": str(e)}, 500)
            
    def _get_require_api_key_decorator(self):
        return self._require_api_key

    def run(self, host="0.0.0.0", port=5001):
        print(f"[*] Starting Secure API on http://{host}:{port}")
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        self.app.run(host=host, port=port, debug=True, use_reloader=False)