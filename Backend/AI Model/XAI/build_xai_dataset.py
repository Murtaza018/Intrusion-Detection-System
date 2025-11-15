import os
import sys
import json
import numpy as np
import pandas as pd

# ----- Path setup -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../Backend/AI Model/XAI
AI_MODEL_DIR = os.path.dirname(BASE_DIR)                   # .../Backend/AI Model

if AI_MODEL_DIR not in sys.path:
    sys.path.insert(0, AI_MODEL_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Now imports work regardless of the space in "AI Model"
from inference import predict
from utils_shap import explain_with_shap
from templates import render_explanation

# ----- Load concepts -----
CONCEPTS_PATH = os.path.join(BASE_DIR, "concepts.json")

with open(CONCEPTS_PATH, "r") as f:
    CONCEPTS = json.load(f)


def map_top_features_to_concepts(top_features):
    """
    Map raw feature attributions into high-level concepts defined in concepts.json.
    top_features: list of dicts [{"feature": name, "shap_value": float}, ...]
    """
    scores = {c: 0.0 for c in CONCEPTS.keys()}
    for item in top_features:
        fname = item["feature"]
        contrib = abs(item["shap_value"])
        for cname, flist in CONCEPTS.items():
            # only count if feature name matches one of the concept's features
            if fname in flist:
                scores[cname] += contrib

    # keep only positive scores and sort
    scores = {c: s for (c, s) in scores.items() if s > 0}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:3]]


def guess_where():
    """
    Placeholder for contextual info (network segment, host group, etc.).
    In the future you can use metadata (dst_ip, subnet, etc.).
    """
    return "your network"


def build_dataset(X_path, Y_path, out_csv, model_key="cnn_lstm", max_samples=3000):
    if not os.path.exists(X_path) or not os.path.exists(Y_path):
        raise FileNotFoundError(f"Check X_path/Y_path:\n{X_path}\n{Y_path}")

    X = np.load(X_path)
    y = np.load(Y_path)

    rows = []
    n = min(max_samples, X.shape[0])

    for i in range(n):
        sample = X[i]
        label_val = y[i]

        # model inference
        pred = predict(sample, model_key)

        # simple downsampling for Normal traffic
        if pred["label"] == "Normal" and label_val == 0:
            if i % 80 != 0:
                continue
            attack_type = "Normal"
        else:
            # TODO (optional): map label_val to specific attack type if using multi-class labels
            attack_type = "Attack"

        # Local SHAP explanation
        top_feats = explain_with_shap(sample, model_key, top_k=6)
        top_concepts = map_top_features_to_concepts(top_feats)

        facts = {
            "attack_type": attack_type,
            "where": guess_where(),
            "top_concepts": top_concepts,
            "top_features": top_feats,
        }

        explanation = render_explanation(facts)

        rows.append(
            {
                "facts": json.dumps(facts),
                "explanation": explanation,
            }
        )

    if not rows:
        raise RuntimeError("No rows generated for XAI dataset. Check logic / thresholds.")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    # Use your actual CIC-IDS2017 processed test files relative to Backend/AI Model
    X_PATH = os.path.join(
        AI_MODEL_DIR,
        "Preprocessing", "CIC-IDS-2017", "CIC-IDS-2017-Processed", "X_test.npy"
    )
    Y_PATH = os.path.join(
        AI_MODEL_DIR,
        "Preprocessing", "CIC-IDS-2017", "CIC-IDS-2017-Processed", "y_test.npy"
    )

    OUT = os.path.join(BASE_DIR, "xai_explainer_dataset.csv")

    build_dataset(X_PATH, Y_PATH, OUT, model_key="cnn_lstm_hardened", max_samples=3000)

