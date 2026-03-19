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

    def _get_require_api_key_decorator(self):
        return self._require_api_key

    def run(self, host="0.0.0.0", port=5001):
        print(f"[*] Starting Secure API on http://{host}:{port}")
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        self.app.run(host=host, port=port, debug=True, use_reloader=False)