import os
import sys
import json
import numpy as np
import pandas as pd

# ----- Path setup -----
# This ensures we can import from parent directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../Backend/AI Model/XAI
AI_MODEL_DIR = os.path.dirname(BASE_DIR)                   # .../Backend/AI Model

if AI_MODEL_DIR not in sys.path:
    sys.path.insert(0, AI_MODEL_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Imports
# We wrap these in try/except so this script can also be run standalone to build datasets
try:
    from inference import predict
    from utils_shap import explain_with_shap
    from templates import render_explanation
except ImportError:
    # Fallback if running directly or if dependencies are missing
    pass

# ----- Load concepts -----
CONCEPTS_PATH = os.path.join(BASE_DIR, "concepts.json")

try:
    with open(CONCEPTS_PATH, "r") as f:
        CONCEPTS = json.load(f)
except FileNotFoundError:
    print(f"[!] Warning: concepts.json not found at {CONCEPTS_PATH}")
    CONCEPTS = {}


def map_top_features_to_concepts(top_features):
    """
    Map raw feature attributions into high-level concepts defined in concepts.json.
    top_features: list of dicts [{"feature": name, "shap_value": float}, ...]
    
    Returns:
        A list of concept names (strings) sorted by importance.
    """
    scores = {c: 0.0 for c in CONCEPTS.keys()}
    
    # Iterate through the top features identified by SHAP
    for item in top_features:
        fname = item["feature"]
        # We use the absolute value because a strong negative contribution 
        # is just as important as a strong positive one.
        contrib = abs(item["shap_value"])
        
        # Find which concept this feature belongs to
        found = False
        for concept_name, feat_list in CONCEPTS.items():
            if fname in feat_list:
                scores[concept_name] += contrib
                found = True
        
        # Optional: If feature not in any concept, you could track it as "Other"
    
    # Sort concepts by score (highest contribution first)
    sorted_concepts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return just the names of concepts that have a non-zero score
    return [c for c, s in sorted_concepts if s > 0]


def guess_where():
    """
    Placeholder for contextual awareness.
    In a real app, this might look at dest_ip to say 'Database Server' vs 'User Laptop'.
    """
    return "your network"


def main():
    print("--- Building XAI Explainer Dataset (Offline) ---")
    # This block is for generating a CSV dataset of explanations for offline analysis.
    # It requires loading data and running predictions in a loop.
    
    # Example placeholder logic:
    print("This script contains helper functions for XAI generation.")
    print("To generate a full dataset, add your data loading logic here.")

if __name__ == "__main__":
    main()