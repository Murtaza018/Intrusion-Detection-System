import threading
import traceback
import uuid
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    QUEUED      = "queued"
    RUNNING     = "running"
    DONE        = "done"
    FAILED      = "failed"
    ROLLED_BACK = "rolled_back"


class RetrainJob:
    def __init__(self, job_id: str, label: str, is_new: bool, packet_count: int):
        self.job_id          = job_id
        self.label           = label
        self.is_new          = is_new
        self.packet_count    = packet_count
        self.status          = JobStatus.QUEUED
        self.phase           = "queued"
        self.progress        = 0
        self.started_at      = None
        self.finished_at     = None
        self.error           = None
        self.retrain_results = {}


class RetrainManager:

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
        self._gan_retrainer  = gan_retrainer
        self._lock           = threading.Lock()
        self._current_job    = None
        # Saved once on first job — always wrap from these, never from
        # previously patched versions, so hooks never stack across jobs.
        self._original_methods = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, packet_ids, label, is_new):
        with self._lock:
            if self._current_job and self._current_job.status in (
                JobStatus.QUEUED, JobStatus.RUNNING
            ):
                return self._current_job, False

            job = RetrainJob(
                job_id       = str(uuid.uuid4())[:8],
                label        = label,
                is_new       = is_new,
                packet_count = len(packet_ids),
            )
            self._current_job = job

        threading.Thread(
            target=self._run_job, args=(job, packet_ids),
            daemon=True, name=f"retrain-{job.job_id}"
        ).start()
        return job, True

    def get_status(self):
        with self._lock:
            job = self._current_job
        return self._job_to_dict(job) if job else None

    def get_job(self):
        with self._lock:
            return self._current_job

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _run_job(self, job, packet_ids):
        job.status     = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()
        try:
            self._patch_progress_hooks(job)
            result = self._gan_retrainer.retrain(
                packet_ids=packet_ids, target_label=job.label, is_new_label=job.is_new
            )
            if result.get("status") == "success":
                job.status          = JobStatus.DONE
                job.phase           = "done"
                job.progress        = 100
                job.retrain_results = result.get("retrain_results", {})
            elif "rolled_back" in result.get("message", "").lower():
                job.status = JobStatus.ROLLED_BACK
                job.phase  = "rolled_back"
                job.error  = result.get("message")
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

    def _save_originals(self):
        """Capture true originals once. Never overwritten after first call."""
        if self._original_methods is not None:
            return
        r  = self._gan_retrainer
        cr = r.continual_retrainer
        self._original_methods = {
            "handle_label_encoder": r._handle_label_encoder,
            "expand_model_classes": r._expand_model_classes,
            "prepare_training_batch": r._prepare_training_batch,
            "generate_and_save": r._generate_and_save_packets,
            "update_replay_buffer": r._update_replay_buffer,
            "continual_run": cr.run,
        }

    def _patch_progress_hooks(self, job):
        self._save_originals()
        orig = self._original_methods
        r    = self._gan_retrainer
        cr   = r.continual_retrainer

        def make_hook(original, phase):
            def hooked(*args, **kwargs):
                self._set_phase(job, phase)
                return original(*args, **kwargs)
            return hooked

        r._handle_label_encoder    = make_hook(orig["handle_label_encoder"],  "label_encoding")
        r._expand_model_classes    = make_hook(orig["expand_model_classes"],   "architecture")
        r._generate_and_save_packets = make_hook(orig["generate_and_save"],    "synthesis")
        r._update_replay_buffer    = make_hook(orig["update_replay_buffer"],   "replay_update")

        # prepare_training_batch: set data_prep on entry, wgan_training on exit
        orig_prepare = orig["prepare_training_batch"]
        def prepare_hook(*args, **kwargs):
            self._set_phase(job, "data_prep")
            result = orig_prepare(*args, **kwargs)
            self._set_phase(job, "wgan_training")
            return result
        r._prepare_training_batch = prepare_hook

        orig_cr_run = orig["continual_run"]
        def cr_run_hook(*args, **kwargs):
            self._set_phase(job, "model_finetuning")
            return orig_cr_run(*args, **kwargs)
        cr.run = cr_run_hook

    def _set_phase(self, job, phase):
        job.phase    = phase
        job.progress = self._PHASE_PROGRESS.get(phase, job.progress)
        print(f"[RetrainManager] Job {job.job_id}: {phase} ({job.progress}%)")

    @staticmethod
    def _job_to_dict(job):
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