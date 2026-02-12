from xai_explainer import IDSExplainer
import numpy as np

def test_explanation_faithfulness():
    """Tests if SHAP correctly identifies a forced feature anomaly."""
    explainer = IDSExplainer()
    
    # 1. Create a packet and spike a specific feature (index 4: Fwd Pkt Len Max)
    features = np.zeros((1, 95))
    features[0, 4] = 10.0 # Extreme value
    
    # Mock prediction: if index 4 > 5, then it's an attack
    def mock_predict(x):
        return np.array([[0.0, 1.0] if v[4] > 5 else [1.0, 0.0] for v in x])
    
    explainer.initialize_shap(mock_predict)
    explanation = explainer.generate_explanation(features, mock_predict, 1.0, {}, "Novelty")
    
    top_factor = explanation['top_contributing_factors'][0]['factor']
    print(f"[*] Simulated Anomaly Feature: Fwd Pkt Len Max")
    print(f"[*] XAI Identified Feature: {top_factor}")
    
    if "Fwd Packet Length Max" in top_factor or "index 4" in top_factor:
        print("✅ PASS: XAI is 'Faithful' to the model logic.")
    else:
        print("❌ FAIL: XAI is pointing to irrelevant features.")

if __name__ == "__main__":
    test_explanation_faithfulness()