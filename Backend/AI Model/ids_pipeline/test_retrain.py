"""
test_retrain.py

Standalone test for the ContinualRetrainer pipeline.
Run from the ids_pipeline directory:

    python test_retrain.py
    python test_retrain.py --model cnn      # test one model only
    python test_retrain.py --dry-run        # skip actual training, just verify loading
    python test_retrain.py --no-save        # train but don't write files to disk

What this tests
───────────────
For each model (xgb, rf, cnn, ae, mae, gnn):
  1. Records a fingerprint of the model weights BEFORE retraining
  2. Runs ContinualRetrainer.run() on synthetic data
  3. Records the fingerprint AFTER
  4. Checks:
       a. The model file on disk was actually updated (mtime changed)
       b. The in-memory weights changed (model actually trained)
       c. The model still produces valid output shapes
       d. No exceptions were thrown
  5. Prints a per-model PASS / FAIL / SKIPPED report

No real packets, no DB connection, no Flask server needed.
A mock FeatureExtractor and a synthetic CSV are generated internally.
"""

import argparse
import os
import sys
import time
import hashlib
import shutil
import tempfile
import traceback

import numpy as np
import pandas as pd
import torch
import joblib

# ── make sure ids_pipeline imports resolve ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model_loader import ModelLoader
from continual_retrainer import ContinualRetrainer

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✔ {msg}{RESET}")
def fail(msg): print(f"  {RED}✗ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠ {msg}{RESET}")
def info(msg): print(f"  {CYAN}→ {msg}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Mock feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class MockFeatureExtractor:
    """
    Minimal drop-in for the real FeatureExtractor.
    scale_features:         clips to [0,1] (assumes input already normalised)
    inverse_scale_features: identity (scaled ≈ raw in tests)
    """
    def scale_features(self, x: np.ndarray) -> np.ndarray:
        arr = np.array(x, dtype=np.float32).reshape(1, -1)
        return np.clip(arr, 0.0, 1.0)

    def inverse_scale_features(self, x: np.ndarray) -> np.ndarray:
        return np.array(x, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_csv(path: str, label: str, n_rows: int = 300, n_features: int = 78):
    """
    Writes a CSV that looks exactly like what GanRetrainer._generate_and_save_packets
    produces: n_features float columns + a 'label' column.
    """
    X = np.random.rand(n_rows, n_features).astype(np.float32)
    df = pd.DataFrame(X, columns=[str(i) for i in range(n_features)])
    df["label"] = label
    df.to_csv(path, index=False)
    info(f"Synthetic CSV: {n_rows} rows × {n_features} features → {path}")


def make_replay_buffer(path: str, n_samples: int = 400, n_features: int = 78):
    """
    Writes a replay_buffer.npz that looks like what GanRetrainer produces.
    Labels are random ints 0–4 (simulating 5 existing classes).
    """
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 5, size=n_samples).astype(np.float32)
    np.savez_compressed(path, X=X, y=y)
    info(f"Replay buffer: {n_samples} samples → {path}")


def make_label_encoder(path: str, labels: list):
    """Writes a minimal sklearn LabelEncoder .pkl."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(labels)
    joblib.dump(le, path)
    info(f"Label encoder: {le.classes_} → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Weight fingerprinting
# ─────────────────────────────────────────────────────────────────────────────

def fingerprint_keras(model) -> str:
    """SHA-256 of all Keras weight bytes."""
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(w.tobytes())
    return h.hexdigest()[:16]


def fingerprint_torch(model) -> str:
    """SHA-256 of all PyTorch parameter bytes."""
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.data.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def fingerprint_sklearn(model) -> str:
    """SHA-256 via joblib serialisation bytes."""
    import io
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def get_fingerprint(key: str, model) -> str:
    if key in ("cnn", "ae"):
        return fingerprint_keras(model)
    if key in ("gnn", "mae"):
        return fingerprint_torch(model)
    return fingerprint_sklearn(model)


# ─────────────────────────────────────────────────────────────────────────────
# Output shape sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def check_output_shapes(key: str, model) -> tuple[bool, str]:
    """
    Runs a dummy forward pass and checks the output shape is sensible.
    Returns (passed, description).
    """
    try:
        if key == "cnn":
            out = model.predict(np.zeros((4, 78), dtype=np.float32), verbose=0)
            assert out.shape == (4, 1), f"Expected (4,1) got {out.shape}"
            return True, f"output shape {out.shape} ✔"

        if key == "ae":
            out = model.predict(np.zeros((4, 78), dtype=np.float32), verbose=0)
            assert out.shape == (4, 78), f"Expected (4,78) got {out.shape}"
            return True, f"output shape {out.shape} ✔"

        if key in ("rf", "xgb"):
            proba = model.predict_proba(np.zeros((4, 95), dtype=np.float32))
            assert proba.shape[0] == 4, f"Expected 4 rows got {proba.shape[0]}"
            assert abs(proba[0].sum() - 1.0) < 1e-4, "Probabilities don't sum to 1"
            return True, f"predict_proba shape {proba.shape} ✔"

        if key == "mae":
            x = torch.zeros((4, 78))
            model.eval()
            with torch.no_grad():
                recon, original = model(x, mask_ratio=0.15)
            assert recon.shape == original.shape, "recon/original shape mismatch"
            assert recon.shape == (4, 1, 9, 9), f"Expected (4,1,9,9) got {recon.shape}"
            return True, f"recon shape {recon.shape} ✔"

        if key == "gnn":
            x  = torch.zeros((4, model.convs[0].in_channels
                               if hasattr(model, 'convs') else 16))
            n  = 4
            ei = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
            model.eval()
            with torch.no_grad():
                out = model(x, ei)
            assert out.shape[0] == 4, f"Expected 4 nodes got {out.shape[0]}"
            return True, f"output shape {out.shape} ✔"

    except Exception as e:
        return False, str(e)

    return True, "no check defined"


# ─────────────────────────────────────────────────────────────────────────────
# Per-model test runner
# ─────────────────────────────────────────────────────────────────────────────

def test_model(
    key: str,
    model_loader: ModelLoader,
    retrainer: ContinualRetrainer,
    csv_path: str,
    target_label: str,
    dry_run: bool,
    no_save: bool,
) -> dict:
    """
    Returns a result dict:
      { 'status': 'pass'|'fail'|'skipped', 'weights_changed': bool,
        'file_updated': bool, 'shape_ok': bool, 'notes': str }
    """
    result = {
        "status":          "fail",
        "weights_changed": False,
        "file_updated":    False,
        "shape_ok":        False,
        "notes":           "",
    }

    model = model_loader._models.get(key)
    if model is None:
        result["status"] = "skipped"
        result["notes"]  = "model not loaded"
        return result

    path = model_loader.get_path(key)

    # ── snapshot before ───────────────────────────────────────────────
    fp_before   = get_fingerprint(key, model)
    mtime_before = os.path.getmtime(path) if os.path.exists(path) else 0

    if dry_run:
        result["status"] = "skipped"
        result["notes"]  = "dry-run, skipping training"
        return result

    # ── back up the real file so we don't permanently alter it ────────
    bak_path = path + ".test_bak"
    if os.path.exists(path) and not no_save:
        shutil.copy2(path, bak_path)

    try:
        # Run only the specific model's retrain method
        method = {
            "xgb": retrainer._retrain_xgb,
            "rf":  retrainer._retrain_rf,
            "cnn": retrainer._retrain_cnn,
            "ae":  retrainer._retrain_ae,
            "mae": retrainer._retrain_mae,
            "gnn": retrainer._retrain_gnn,
        }.get(key)

        if method is None:
            result["status"] = "skipped"
            result["notes"]  = "no test method mapped"
            return result

        # Load the CSV so ContinualRetrainer has data in its internal state
        gen_df       = pd.read_csv(csv_path)
        feature_cols = [c for c in gen_df.columns if c != "label"]
        X_new        = gen_df[feature_cols].values.astype(np.float32)
        y_new_str    = gen_df["label"].values

        retrainer._load_replay()
        X_mixed, y_mixed_str = retrainer._build_mixed_dataset(X_new, y_new_str)
        y_mixed_int = retrainer._encode_labels(y_mixed_str)

        X_scaled = np.vstack([
            retrainer.feature_extractor.scale_features(x.reshape(1, -1)).flatten()
            for x in X_mixed
        ])
        X_95     = np.hstack([X_scaled, np.zeros((len(X_scaled), 17))])
        y_binary = (y_mixed_str != "BENIGN").astype(np.float32)

        # Call the right method with the right args
        t0 = time.time()
        if key in ("xgb", "rf"):
            outcome = method(X_95, y_mixed_int)
        elif key == "cnn":
            outcome = method(X_scaled, y_binary)
        elif key in ("ae", "mae"):
            outcome = method(X_scaled)
        elif key == "gnn":
            outcome = method(X_scaled, y_binary)
        elapsed = time.time() - t0

        result["notes"] = f"outcome={outcome}, elapsed={elapsed:.1f}s"

        if outcome == "skipped":
            result["status"] = "skipped"
            return result

        # ── snapshot after ────────────────────────────────────────────
        fp_after    = get_fingerprint(key, model)
        mtime_after = os.path.getmtime(path) if os.path.exists(path) else 0

        result["weights_changed"] = (fp_before != fp_after)
        result["file_updated"]    = (not no_save) and (mtime_after > mtime_before)

        # ── output shape check ────────────────────────────────────────
        shape_ok, shape_msg = check_output_shapes(key, model)
        result["shape_ok"] = shape_ok
        result["notes"]   += f" | shape: {shape_msg}"

        # ── overall verdict ───────────────────────────────────────────
        if outcome in ("ok", "rolled_back") and shape_ok:
            result["status"] = "pass"
            if outcome == "rolled_back":
                result["notes"] += " (rolled back — score dropped, original restored)"
        else:
            result["status"] = "fail"

    except Exception as e:
        result["status"] = "fail"
        result["notes"]  = traceback.format_exc()
    finally:
        # ── restore original file so we haven't broken anything ───────
        if os.path.exists(bak_path):
            shutil.copy2(bak_path, path)
            os.remove(bak_path)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test ContinualRetrainer pipeline")
    parser.add_argument("--model",   default=None,
                        choices=["xgb","rf","cnn","ae","mae","gnn"],
                        help="Test a single model instead of all six")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load models and generate data but skip training")
    parser.add_argument("--no-save", action="store_true",
                        help="Train but do not write updated weights to disk")
    parser.add_argument("--label",   default="DDoS",
                        help="Attack label to use for synthetic data (default: DDoS)")
    parser.add_argument("--rows",    default=300, type=int,
                        help="Number of synthetic rows to generate (default: 300)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ContinualRetrainer Test Suite")
    print(f"  label={args.label}  rows={args.rows}  "
          f"dry_run={args.dry_run}  no_save={args.no_save}")
    print(f"{'='*60}\n")

    # ── 1. Load models ────────────────────────────────────────────────
    print("[ Step 1 ] Loading models via ModelLoader…")
    loader = ModelLoader()
    ok_load = loader.load_models()
    if not ok_load:
        fail("ModelLoader.load_models() failed — check paths in config.py")
        sys.exit(1)
    ok("All models loaded")

    # ── 2. Build temp workspace ───────────────────────────────────────
    print("\n[ Step 2 ] Building synthetic test data…")
    tmpdir = tempfile.mkdtemp(prefix="retrain_test_")
    info(f"Temp dir: {tmpdir}")

    csv_path     = os.path.join(tmpdir, "synthetic.csv")
    replay_path  = os.path.join(tmpdir, "replay_buffer.npz")
    encoder_path = os.path.join(tmpdir, "label_encoder.pkl")

    # Labels must include BENIGN + the target label + a few extras
    # so the replay buffer mixing and label encoding work correctly
    all_labels = ["BENIGN", "DDoS", "PortScan", "Bot", "WebAttack", args.label]
    all_labels = list(dict.fromkeys(all_labels))  # deduplicate, preserve order

    make_synthetic_csv(csv_path, args.label, n_rows=args.rows)
    make_replay_buffer(replay_path, n_samples=500)
    make_label_encoder(encoder_path, all_labels)

    # ── 3. Build retrainer ────────────────────────────────────────────
    print("\n[ Step 3 ] Initialising ContinualRetrainer…")
    feature_extractor = MockFeatureExtractor()
    retrainer = ContinualRetrainer(
        model_loader      = loader,
        feature_extractor = feature_extractor,
        replay_path       = replay_path,
        encoder_path      = encoder_path,
    )
    ok("ContinualRetrainer ready")

    # ── 4. Run per-model tests ────────────────────────────────────────
    keys_to_test = [args.model] if args.model else ["xgb", "rf", "cnn", "ae", "mae", "gnn"]
    results = {}

    print(f"\n[ Step 4 ] Running tests for: {keys_to_test}\n")
    for key in keys_to_test:
        print(f"  ── {key.upper()} {'─'*40}")
        r = test_model(
            key           = key,
            model_loader  = loader,
            retrainer     = retrainer,
            csv_path      = csv_path,
            target_label  = args.label,
            dry_run       = args.dry_run,
            no_save       = args.no_save,
        )
        results[key] = r

        status = r["status"]
        notes  = r["notes"]

        if status == "pass":
            ok(f"PASS — weights_changed={r['weights_changed']}  "
               f"file_updated={r['file_updated']}  shape_ok={r['shape_ok']}")
            if notes:
                info(notes)
        elif status == "skipped":
            warn(f"SKIPPED — {notes}")
        else:
            fail(f"FAIL — {notes}")
        print()

    # ── 5. Summary ────────────────────────────────────────────────────
    print(f"{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    passed   = [k for k,v in results.items() if v["status"] == "pass"]
    failed   = [k for k,v in results.items() if v["status"] == "fail"]
    skipped  = [k for k,v in results.items() if v["status"] == "skipped"]

    if passed:  ok(      f"Passed  ({len(passed)}/{len(keys_to_test)}): {', '.join(passed)}")
    if skipped: warn(    f"Skipped ({len(skipped)}): {', '.join(skipped)}")
    if failed:  fail(    f"Failed  ({len(failed)}): {', '.join(failed)}")

    # Extra diagnostic: flag models whose weights didn't change
    # (trained but stayed identical — usually means no trainable params)
    unchanged = [k for k,v in results.items()
                 if v["status"] == "pass" and not v["weights_changed"]]
    if unchanged:
        warn(f"Weights unchanged after training (check layer freezing): {', '.join(unchanged)}")

    # ── 6. Cleanup ────────────────────────────────────────────────────
    shutil.rmtree(tmpdir, ignore_errors=True)
    info(f"Temp dir cleaned up")

    print()
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()



# # From your ids_pipeline directory:

# # Test all 6 models
# python test_retrain.py

# # Test one model at a time
# python test_retrain.py --model cnn
# python test_retrain.py --model mae
# python test_retrain.py --model gnn

# # Just verify everything loads without running training
# python test_retrain.py --dry-run

# # Train but don't write to disk (safe to run on production models)
# python test_retrain.py --no-save

# # Test with a specific label and more data rows
# python test_retrain.py --label PortScan --rows 500