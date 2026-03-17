"""
test_retrain.py

Standalone test for the ContinualRetrainer pipeline.
Run from the ids_pipeline directory:

    python test_retrain.py                        # test all 6 models, restore files after
    python test_retrain.py --model cnn            # test one model only
    python test_retrain.py --dry-run              # load + build data, skip training
    python test_retrain.py --no-save              # train in memory, score before AND after,
                                                  # never touch disk

--no-save flow (what you asked for):
    1. Score model on held-out synthetic data  (before)
    2. Train model in memory
    3. Score model on the same data            (after)
    4. Print before → after comparison
    5. Restore original weights from memory    (disk untouched)
"""

import argparse
import os
import sys
import time
import hashlib
import shutil
import tempfile
import copy
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model_loader import ModelLoader
from continual_retrainer import ContinualRetrainer

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✔ {msg}{RESET}")
def fail(msg):  print(f"  {RED}✗ {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠ {msg}{RESET}")
def info(msg):  print(f"  {CYAN}→ {msg}{RESET}")
def score_line(label, before, after):
    arrow = "↑" if after > before else ("↓" if after < before else "=")
    colour = GREEN if after >= before else YELLOW
    print(f"  {colour}{label:30s}  before={before:.4f}  after={after:.4f}  {arrow}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Mock feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class MockFeatureExtractor:
    def scale_features(self, x):
        return np.clip(np.array(x, dtype=np.float32).reshape(1, -1), 0.0, 1.0)

    def inverse_scale_features(self, x):
        return np.array(x, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_csv(path, label, n_rows=400, n_features=78):
    X  = np.random.rand(n_rows, n_features).astype(np.float32)
    df = pd.DataFrame(X, columns=[str(i) for i in range(n_features)])
    df["label"] = label
    df.to_csv(path, index=False)
    info(f"Synthetic CSV: {n_rows} rows × {n_features} features  label={label}")


def make_replay_buffer(path, n_samples=500, n_features=78):
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 5, size=n_samples).astype(np.float32)
    np.savez_compressed(path, X=X, y=y)
    info(f"Replay buffer: {n_samples} samples")


def make_label_encoder(path, labels):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(labels)
    joblib.dump(le, path)
    info(f"Label encoder: {list(le.classes_)}")


def make_held_out_data(label, all_labels, n_rows=200, n_features=78):
    """
    Separate held-out set used for before/after scoring.
    50% target label, 50% BENIGN so both classes are represented.
    """
    half   = n_rows // 2
    X      = np.random.rand(n_rows, n_features).astype(np.float32)
    y_str  = np.array([label] * half + ["BENIGN"] * half)
    y_bin  = (y_str != "BENIGN").astype(np.float32)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(all_labels)
    y_int = le.transform(y_str)

    # Scale
    X_scaled = np.clip(X, 0.0, 1.0)
    X_95     = np.hstack([X_scaled, np.zeros((len(X_scaled), 17))])
    return X_scaled, X_95, y_bin, y_int


# ─────────────────────────────────────────────────────────────────────────────
# Weight snapshot / restore  (used by --no-save)
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_weights(key, model):
    """Returns a deep copy of the model weights."""
    if key in ("cnn", "ae"):
        return [w.copy() for w in model.get_weights()]
    if key in ("gnn", "mae"):
        return copy.deepcopy(model.state_dict())
    # sklearn: pickle round-trip
    import io
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    return buf

def restore_weights(key, model, snapshot):
    """Puts the snapshot back into the model in-place."""
    if key in ("cnn", "ae"):
        model.set_weights(snapshot)
    elif key in ("gnn", "mae"):
        model.load_state_dict(snapshot)
        model.eval()
    else:
        # sklearn: reload from buffer, update loader reference later
        snapshot.seek(0)
        return joblib.load(snapshot)   # caller must re-assign
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Before / after scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_model(key, model, X_scaled, X_95, y_bin, y_int):
    """
    Returns a dict of metric_name → float for the given model.
    Uses the held-out data that was NOT used for training.
    """
    try:
        if key == "cnn":
            preds = (model.predict(X_scaled, verbose=0).flatten() > 0.5).astype(int)
            f1    = f1_score(y_bin.astype(int), preds, average="macro", zero_division=0)
            return {"f1_macro": round(f1, 4)}

        if key == "ae":
            recon = model.predict(X_scaled, verbose=0)
            mse   = float(np.mean((X_scaled - recon) ** 2))
            return {"recon_mse": round(mse, 6)}

        if key in ("rf", "xgb"):
            preds = model.predict(X_95)
            f1    = f1_score(y_int, preds, average="macro", zero_division=0)
            return {"f1_macro": round(f1, 4)}

        if key == "mae":
            model.eval()
            with torch.no_grad():
                t = torch.tensor(X_scaled, dtype=torch.float)
                recon, original = model(t, mask_ratio=0.15)
                mse = float(F.mse_loss(recon, original).item())
            return {"recon_mse": round(mse, 6)}

        if key == "gnn":
            # GNN is skipped during retraining — just verify inference still works
            # by checking it produces output without crashing
            from config import GNN_IN_CHANNELS
            model.eval()
            with torch.no_grad():
                n  = len(X_scaled)
                # GNN needs GNN_IN_CHANNELS-dim node features, not 78-dim packet features
                # Use zeros as a smoke test — we're just checking it doesn't crash
                x  = torch.zeros((min(n, 64), GNN_IN_CHANNELS))
                ei = torch.stack([torch.arange(len(x)), torch.arange(len(x))], dim=0)
                out = model(x, ei)
            return {"inference_ok": 1.0}

    except Exception as e:
        return {"error": str(e)}

    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Output shape sanity check
# ─────────────────────────────────────────────────────────────────────────────

def check_output_shapes(key, model):
    try:
        if key == "cnn":
            out = model.predict(np.zeros((4, 78), dtype=np.float32), verbose=0)
            assert out.shape == (4, 1), f"Expected (4,1) got {out.shape}"
            return True, f"{out.shape}"

        if key == "ae":
            out = model.predict(np.zeros((4, 78), dtype=np.float32), verbose=0)
            assert out.shape == (4, 78), f"Expected (4,78) got {out.shape}"
            return True, f"{out.shape}"

        if key in ("rf", "xgb"):
            p = model.predict_proba(np.zeros((4, 95), dtype=np.float32))
            assert p.shape[0] == 4
            assert abs(p[0].sum() - 1.0) < 1e-4, "proba doesn't sum to 1"
            return True, f"{p.shape}"

        if key == "mae":
            model.eval()
            with torch.no_grad():
                recon, orig = model(torch.zeros((4, 78)), mask_ratio=0.15)
            assert recon.shape == (4, 1, 9, 9), f"Got {recon.shape}"
            return True, f"{recon.shape}"

        if key == "gnn":
            # GNN is skipped in continual retraining — just verify it still
            # loads and produces output with its correct input dimensions
            from config import GNN_IN_CHANNELS
            n  = 4
            x  = torch.zeros((n, GNN_IN_CHANNELS))
            ei = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
            model.eval()
            with torch.no_grad():
                out = model(x, ei)
            assert out.shape[0] == n, f"Expected {n} nodes got {out.shape[0]}"
            return True, f"{out.shape}"

    except Exception as e:
        return False, str(e)

    return True, "no check"


# ─────────────────────────────────────────────────────────────────────────────
# Per-model test runner
# ─────────────────────────────────────────────────────────────────────────────

def test_model(key, model_loader, retrainer, csv_path, target_label,
               held_out, dry_run, no_save):

    result = {
        "status":          "fail",
        "weights_changed": False,
        "file_updated":    False,
        "shape_ok":        False,
        "scores_before":   {},
        "scores_after":    {},
        "outcome":         "",
        "notes":           "",
    }

    model = model_loader._models.get(key)
    if model is None:
        result.update(status="skipped", notes="model not loaded")
        return result

    path = model_loader.get_path(key)

    # ── fingerprint + score BEFORE ────────────────────────────────────
    X_scaled, X_95, y_bin, y_int = held_out
    result["scores_before"] = score_model(key, model, X_scaled, X_95, y_bin, y_int)

    if dry_run:
        result.update(status="skipped", notes="dry-run")
        return result

    # ── snapshot weights so we can restore them after (--no-save) ────
    snap     = snapshot_weights(key, model)
    bak_path = path + ".test_bak"

    # Only back up the file if we're going to write to disk
    if not no_save and os.path.exists(path):
        shutil.copy2(path, bak_path)

    mtime_before = os.path.getmtime(path) if os.path.exists(path) else 0

    try:
        # ── prepare shared data arrays ────────────────────────────────
        gen_df       = pd.read_csv(csv_path)
        feature_cols = [c for c in gen_df.columns if c != "label"]
        X_new        = gen_df[feature_cols].values.astype(np.float32)
        y_new_str    = gen_df["label"].values

        retrainer._load_replay()
        X_mix, y_mix_str = retrainer._build_mixed_dataset(X_new, y_new_str)
        y_mix_int        = retrainer._encode_labels(y_mix_str)

        X_sc = np.vstack([
            retrainer.feature_extractor.scale_features(x.reshape(1, -1)).flatten()
            for x in X_mix
        ])
        X_95_train = np.hstack([X_sc, np.zeros((len(X_sc), 17))])
        y_bin_train = (y_mix_str != "BENIGN").astype(np.float32)

        # ── if no_save: monkey-patch model path to /dev/null equivalent
        #    so ContinualRetrainer's joblib.dump / model.save / torch.save
        #    write to a temp file we discard ─────────────────────────────
        if no_save:
            tmp_model_path = os.path.join(tempfile.gettempdir(), f"test_{key}_tmp")
            # Override get_path for this key temporarily
            original_get_path = model_loader.get_path
            def patched_get_path(k):
                return tmp_model_path if k == key else original_get_path(k)
            model_loader.get_path = patched_get_path

        # ── call the right retrain method ─────────────────────────────
        t0 = time.time()
        method_map = {
            "xgb": lambda: retrainer._retrain_xgb(X_95_train, y_mix_int),
            "rf":  lambda: retrainer._retrain_rf(X_95_train, y_mix_int),
            "cnn": lambda: retrainer._retrain_cnn(X_sc, y_bin_train),
            "ae":  lambda: retrainer._retrain_ae(X_sc),
            "mae": lambda: retrainer._retrain_mae(X_sc),
            "gnn": lambda: retrainer._retrain_gnn(X_sc, y_bin_train),
        }
        outcome = method_map[key]()
        elapsed = time.time() - t0
        result["outcome"] = outcome
        result["notes"]   = f"elapsed={elapsed:.1f}s"

        # ── restore get_path ──────────────────────────────────────────
        if no_save:
            model_loader.get_path = original_get_path

        if outcome == "skipped":
            result.update(status="skipped", notes="retrainer returned skipped")
            return result

        # ── score AFTER ───────────────────────────────────────────────
        # For sklearn models that were rolled back, model_loader._models[key]
        # holds the restored model — score that
        live_model = model_loader._models.get(key) or model
        result["scores_after"] = score_model(
            key, live_model, X_scaled, X_95, y_bin, y_int
        )

        # ── weights changed? ──────────────────────────────────────────
        # For rolled_back, weights should match original (that's correct behaviour)
        if key in ("cnn", "ae"):
            after_fp  = hashlib.sha256(
                b"".join(w.tobytes() for w in live_model.get_weights())
            ).hexdigest()[:16]
            before_fp = hashlib.sha256(
                b"".join(w.tobytes() for w in snap)
            ).hexdigest()[:16]
        elif key in ("gnn", "mae"):
            after_fp  = hashlib.sha256(
                b"".join(p.data.cpu().numpy().tobytes()
                         for p in live_model.parameters())
            ).hexdigest()[:16]
            before_fp = hashlib.sha256(
                b"".join(v.cpu().numpy().tobytes() for v in snap.values())
            ).hexdigest()[:16]
        else:
            import io
            buf = io.BytesIO(); joblib.dump(live_model, buf)
            after_fp  = hashlib.sha256(buf.getvalue()).hexdigest()[:16]
            snap.seek(0)
            before_fp = hashlib.sha256(snap.read()).hexdigest()[:16]

        result["weights_changed"] = (before_fp != after_fp)

        # ── file updated? (only meaningful when no_save=False) ────────
        mtime_after = os.path.getmtime(path) if os.path.exists(path) else 0
        result["file_updated"] = (not no_save) and (mtime_after > mtime_before)

        # ── shape check ───────────────────────────────────────────────
        shape_ok, shape_msg = check_output_shapes(key, live_model)
        result["shape_ok"] = shape_ok
        result["notes"]   += f"  shape={shape_msg}"

        # ── overall pass/fail ─────────────────────────────────────────
        if outcome in ("ok", "rolled_back") and shape_ok:
            result["status"] = "pass"
        else:
            result["status"] = "fail"

    except Exception as e:
        result["status"] = "fail"
        result["notes"]  = traceback.format_exc()

    finally:
        # ── restore original weights in memory ────────────────────────
        if no_save:
            restored = restore_weights(key, model, snap)
            if key in ("rf", "xgb"):
                model_loader._models[key] = restored

        # ── restore original file on disk ─────────────────────────────
        if os.path.exists(bak_path):
            shutil.copy2(bak_path, path)
            os.remove(bak_path)

        # Clean up temp model file if no_save
        if no_save:
            tmp = os.path.join(tempfile.gettempdir(), f"test_{key}_tmp")
            for ext in ("", ".bak"):
                if os.path.exists(tmp + ext):
                    os.remove(tmp + ext)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=None,
                        choices=["xgb","rf","cnn","ae","mae","gnn"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save", action="store_true",
                        help="Train in memory, score before+after, restore original weights")
    parser.add_argument("--label",   default="DDoS")
    parser.add_argument("--rows",    default=400, type=int)
    args = parser.parse_args()

    print(f"\n{'='*62}")
    print(f"  {BOLD}ContinualRetrainer Test Suite{RESET}")
    print(f"  label={args.label}  rows={args.rows}  "
          f"dry_run={args.dry_run}  no_save={args.no_save}")
    print(f"{'='*62}\n")

    # 1. Load models
    print("[ 1 ] Loading models…")
    loader = ModelLoader()
    if not loader.load_models():
        fail("ModelLoader failed — check config.py paths")
        sys.exit(1)
    ok("Models loaded")

    # 2. Build temp data
    print("\n[ 2 ] Building synthetic data…")
    tmpdir       = tempfile.mkdtemp(prefix="retrain_test_")
    csv_path     = os.path.join(tmpdir, "synthetic.csv")
    replay_path  = os.path.join(tmpdir, "replay_buffer.npz")
    encoder_path = os.path.join(tmpdir, "label_encoder.pkl")

    all_labels = list(dict.fromkeys(
        ["BENIGN", "DDoS", "PortScan", "Bot", "WebAttack", args.label]
    ))
    make_synthetic_csv(csv_path, args.label, n_rows=args.rows)
    make_replay_buffer(replay_path)
    make_label_encoder(encoder_path, all_labels)

    # Held-out data for before/after scoring
    held_out = make_held_out_data(args.label, all_labels)
    info(f"Held-out set: {len(held_out[0])} samples (50% {args.label} / 50% BENIGN)")

    # 3. Build retrainer
    print("\n[ 3 ] Initialising ContinualRetrainer…")
    retrainer = ContinualRetrainer(
        model_loader      = loader,
        feature_extractor = MockFeatureExtractor(),
        replay_path       = replay_path,
        encoder_path      = encoder_path,
    )
    ok("ContinualRetrainer ready")

    # 4. Run tests
    keys = [args.model] if args.model else ["xgb", "rf", "cnn", "ae", "mae", "gnn"]
    results = {}

    print(f"\n[ 4 ] Running tests…\n")
    for key in keys:
        print(f"  {'─'*8} {key.upper()} {'─'*40}")
        r = test_model(
            key          = key,
            model_loader = loader,
            retrainer    = retrainer,
            csv_path     = csv_path,
            target_label = args.label,
            held_out     = held_out,
            dry_run      = args.dry_run,
            no_save      = args.no_save,
        )
        results[key] = r

        if r["status"] == "skipped":
            warn(f"SKIPPED — {r['notes']}")
        elif r["status"] == "fail":
            fail(f"FAIL — {r['notes']}")
        else:
            ok(f"PASS  outcome={r['outcome']}  "
               f"weights_changed={r['weights_changed']}  "
               f"shape_ok={r['shape_ok']}")

        # Print before/after scores
        sb = r["scores_before"]
        sa = r["scores_after"]
        if sb and sa and "error" not in sb and "error" not in sa:
            for metric in sb:
                score_line(f"  {metric}", sb[metric], sa.get(metric, 0))
        elif sb and not sa:
            info(f"  before={sb}  (no after score — skipped or dry-run)")

        if r["outcome"] == "rolled_back":
            warn(f"  Rolled back — score dropped below threshold, original restored")

        print()

    # 5. Summary
    print(f"{'='*62}")
    print(f"  {BOLD}SUMMARY{RESET}")
    print(f"{'='*62}")

    passed  = [k for k,v in results.items() if v["status"] == "pass"]
    failed  = [k for k,v in results.items() if v["status"] == "fail"]
    skipped = [k for k,v in results.items() if v["status"] == "skipped"]

    if passed:  ok(  f"Passed  ({len(passed)}/{len(keys)}): {', '.join(passed)}")
    if skipped: warn(f"Skipped ({len(skipped)}): {', '.join(skipped)}")
    if failed:  fail(f"Failed  ({len(failed)}): {', '.join(failed)}")

    unchanged = [k for k,v in results.items()
                 if v["status"] == "pass" and not v["weights_changed"]
                 and v["outcome"] != "rolled_back"]
    if unchanged:
        warn(f"Weights unchanged after training: {', '.join(unchanged)} "
             f"— check layer freezing / learning rate")

    if args.no_save:
        ok("--no-save: all original weights restored in memory and on disk")

    shutil.rmtree(tmpdir, ignore_errors=True)
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





# What each result means
# PASS — model trained, output shapes are correct, no exceptions thrown.
# PASS + weights_changed=False — this is a warning you should investigate. 
#     It means training ran but the weights came out identical, which usually means all layers were frozen 
#     and nothing actually updated. Check the layer freezing logic for that model.
# PASS + rolled_back=True in notes — training ran but the F1/loss score dropped more than the threshold, 
#     so the original weights were restored. This is the safety net working correctly, not a failure. It just means the 
#     synthetic data wasn't good enough to improve the model — which is expected on the first retrain with very few seed packets.
# FAIL — an exception was thrown. The full traceback is in the notes column. Most likely causes: 
#     wrong input dimensions, a missing dependency, or a shape mismatch in the MAE or GNN forward pass.
# SKIPPED — the model wasn't loaded (path missing in config.py) or --dry-run was passed.