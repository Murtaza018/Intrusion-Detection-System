import os
import sys
from typing import Dict

# ----- Path setup -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../Backend/AI Model/XAI
AI_MODEL_DIR = os.path.dirname(BASE_DIR)                   # .../Backend/AI Model

if AI_MODEL_DIR not in sys.path:
    sys.path.insert(0, AI_MODEL_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils_shap import explain_with_shap
from build_xai_dataset import map_top_features_to_concepts, guess_where
from templates import render_explanation


def explain_alert(sample, model_key: str, attack_type: str = "Attack", meta: Dict = None) -> Dict:
    """
    End-to-end explanation for a single alert:
      1) Compute SHAP for this sample
      2) Map features -> high-level concepts
      3) Build 'facts' dict
      4) Generate human-readable explanation text

    sample: 1D feature vector (numpy-like)
    model_key: which IDS model raised this alert (e.g., "cnn_lstm")
    attack_type: detected label (e.g., "DDoS", "PortScan", "Normal")
    meta: optional extra context in future (src_ip, dst_ip, etc.)

    returns: dict with facts + "explanation" string
    """
    # 1) Local explanation with SHAP
    top_feats = explain_with_shap(sample, model_key, top_k=6)

    # 2) Concept-level mapping
    top_concepts = map_top_features_to_concepts(top_feats)

    # 3) Context: where in the network
    where = guess_where()  # later: use meta to refine (e.g. subnet, host group)

    facts = {
        "attack_type": attack_type,
        "where": where,
        "top_concepts": top_concepts,
        "top_features": top_feats,
    }

    # 4) Narrative explanation for non-technical users
    explanation = render_explanation(facts)
    facts["explanation"] = explanation
    return facts
