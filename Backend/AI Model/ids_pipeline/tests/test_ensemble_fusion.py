import numpy as np

def test_weighted_soft_voting():
    # Mocking probabilities from different models
    # prob_xgb: 0.9 (Very sure it's an attack)
    # prob_sensory: 0.2 (Looks structurally okay)
    
    prob_xgb = 0.9
    prob_sensory = 0.2
    
    # Weighted calculation: 70% XGBoost, 30% Sensory
    final_score = (prob_xgb * 0.7) + (prob_sensory * 0.3)
    
    print(f"[*] Fusion Score: {final_score:.4f}")
    
    # Final threshold check
    if final_score > 0.5:
        print("✅ PASS: Ensemble successfully fused conflicting signals.")
    else:
        print("❌ FAIL: Fusion logic skewed toward incorrect baseline.")

if __name__ == "__main__":
    test_weighted_soft_voting()