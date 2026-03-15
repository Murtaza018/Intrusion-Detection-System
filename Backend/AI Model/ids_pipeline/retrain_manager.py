"""
retrain_manager.py

Runs GAN retraining in a background thread so the Flask request returns
immediately. Flutter polls /api/retrain/status to watch progress.

Job lifecycle
─────────────
  queued  →  running  →  done
                      →  failed
                      →  rolled_back   (GAN health check failed)

Only ONE job runs at a time. A second retrain request while one is running
returns HTTP 409 with the current job's status.
"""

import threading
import traceback
import uuid
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    QUEUED       = "queued"
    RUNNING      = "running"
    DONE         = "done"
    FAILED       = "failed"
    ROLLED_BACK  = "rolled_back"


class RetrainJob:
    def __init__(self, job_id: str, label: str, is_new: bool, packet_count: int):
        self.job_id        = job_id
        self.label         = label
        self.is_new        = is_new
        self.packet_count  = packet_count

        self.status        = JobStatus.QUEUED
        self.phase         = "queued"        # human-readable current step
        self.progress      = 0              # 0–100
        self.started_at    = None
        self.finished_at   = None
        self.error         = None
        self.retrain_results = {}           # per-model outcomes from ContinualRetrainer


class RetrainManager:
    """
    Owns a single background worker thread.
    GanRetrainer is called from inside that thread, never from the request thread.
    """

    # Phase → approximate progress percentage when that phase starts
    _PHASE_PROGRESS = {
        "backing_up":        5,
        "label_encoding":   10,
        "architecture":     15,
        "data_prep":        20,
        "wgan_training":    25,
        "health_check":     70,
        "synthesis":        75,
        "replay_update":    80,
        "model_finetuning": 85,
        "done":            100,
    }

    def __init__(self, gan_retrainer):
        self._gan_retrainer = gan_retrainer
        self._lock          = threading.Lock()
        self._current_job: RetrainJob | None = None

    # ------------------------------------------------------------------
    # Public API (called from Flask routes)
    # ------------------------------------------------------------------

    def submit(self, packet_ids: list, label: str, is_new: bool) -> tuple[RetrainJob, bool]:
        """
        Try to enqueue a new retrain job.

        Returns (job, accepted):
          accepted=True  → job was queued, worker thread started
          accepted=False → another job is already running; job is the active one
        """
        with self._lock:
            if self._current_job and self._current_job.status in (
                JobStatus.QUEUED, JobStatus.RUNNING
            ):
                return self._current_job, False

            job = RetrainJob(
                job_id        = str(uuid.uuid4())[:8],
                label         = label,
                is_new        = is_new,
                packet_count  = len(packet_ids),
            )
            self._current_job = job

        # Start background thread (daemon=True so it doesn't block server shutdown)
        t = threading.Thread(
            target   = self._run_job,
            args     = (job, packet_ids),
            daemon   = True,
            name     = f"retrain-{job.job_id}",
        )
        t.start()
        return job, True

    def get_status(self) -> dict | None:
        """Returns a serialisable status dict, or None if no job has ever run."""
        with self._lock:
            job = self._current_job
        if job is None:
            return None
        return self._job_to_dict(job)

    def get_job(self) -> RetrainJob | None:
        with self._lock:
            return self._current_job

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _run_job(self, job: RetrainJob, packet_ids: list):
        job.status     = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()

        try:
            # We monkey-patch the GanRetrainer's internal methods to report
            # progress back to the job object without touching its logic.
            self._patch_progress_hooks(job)

            result = self._gan_retrainer.retrain(
                packet_ids   = packet_ids,
                target_label = job.label,
                is_new_label = job.is_new,
            )

            if result.get("status") == "success":
                job.status          = JobStatus.DONE
                job.phase           = "done"
                job.progress        = 100
                job.retrain_results = result.get("retrain_results", {})
            elif "rolled_back" in result.get("message", "").lower():
                job.status  = JobStatus.ROLLED_BACK
                job.phase   = "rolled_back"
                job.error   = result.get("message")
            else:
                job.status = JobStatus.FAILED
                job.phase  = "failed"
                job.error  = result.get("message", "Unknown error")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.phase  = "failed"
            job.error  = str(e)
            traceback.print_exc()

        finally:
            job.finished_at = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Progress hooks
    # ------------------------------------------------------------------

    def _patch_progress_hooks(self, job: RetrainJob):
        """
        Wraps key GanRetrainer methods to update job.phase and job.progress.
        This is done by replacing the method on the instance (not the class)
        so other instances are unaffected.
        """
        retrainer = self._gan_retrainer

        # Helper that wraps a bound method and sets phase before calling it
        def make_hook(original_method, phase_name):
            def hooked(*args, **kwargs):
                self._set_phase(job, phase_name)
                return original_method(*args, **kwargs)
            return hooked

        retrainer._handle_label_encoder = make_hook(
            retrainer._handle_label_encoder, "label_encoding"
        )
        retrainer._expand_model_classes = make_hook(
            retrainer._expand_model_classes, "architecture"
        )
        retrainer._prepare_training_batch = make_hook(
            retrainer._prepare_training_batch, "data_prep"
        )
        retrainer._generate_and_save_packets = make_hook(
            retrainer._generate_and_save_packets, "synthesis"
        )
        retrainer._update_replay_buffer = make_hook(
            retrainer._update_replay_buffer, "replay_update"
        )

        # WGAN training is the longest phase — we set it just before fit()
        # by wrapping retrain() itself isn't practical here, so we rely on
        # the fact that data_prep finishes just before wgan_training begins.
        # The phase is set to wgan_training after data_prep completes.
        original_prepare = retrainer._prepare_training_batch
        def prepare_then_set_wgan(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            self._set_phase(job, "wgan_training")
            return result
        retrainer._prepare_training_batch = prepare_then_set_wgan

        # ContinualRetrainer phase
        original_run = retrainer.continual_retrainer.run
        def run_with_progress(*args, **kwargs):
            self._set_phase(job, "model_finetuning")
            return original_run(*args, **kwargs)
        retrainer.continual_retrainer.run = run_with_progress

    def _set_phase(self, job: RetrainJob, phase: str):
        job.phase    = phase
        job.progress = self._PHASE_PROGRESS.get(phase, job.progress)
        print(f"[RetrainManager] Job {job.job_id}: {phase} ({job.progress}%)")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _job_to_dict(job: RetrainJob) -> dict:
        return {
            "job_id":          job.job_id,
            "label":           job.label,
            "is_new":          job.is_new,
            "packet_count":    job.packet_count,
            "status":          job.status.value,
            "phase":           job.phase,
            "progress":        job.progress,
            "started_at":      job.started_at,
            "finished_at":     job.finished_at,
            "error":           job.error,
            "retrain_results": job.retrain_results,
        }