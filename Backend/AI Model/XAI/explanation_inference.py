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
    model_key: which IDS model raised this alert (e.g., "hardened_classifier")
    attack_type: detected label (e.g., "DDoS", "PortScan", "Normal")
    meta: optional extra context in future (src_ip, dst_ip, etc.)

    returns: dict with "facts" (dict) and "explanation" (str)
    """
    # 1) Local explanation with SHAP
    # Note: We define top_k=6 features to keep the explanation focused
    top_feats = explain_with_shap(sample, model_key, top_k=6)

    # 2) Concept-level mapping
    # Maps raw features (e.g., 'Flow Duration') to concepts (e.g., 'Long_Lived_Connections')
    top_concepts = map_top_features_to_concepts(top_feats)

    # 3) Context: where in the network
    # Currently a placeholder, but could use 'meta' to say "Finance Server" etc.
    where = guess_where() 

    facts = {
        "attack_type": attack_type,
        "where": where,
        "top_concepts": top_concepts,
        "top_features": top_feats,
    }

    # 4) Render text
    text = render_explanation(facts)

    return {
        "facts": facts,
        "explanation": text
    }