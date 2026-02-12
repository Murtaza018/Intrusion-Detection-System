import numpy as np
from model_loader import ModelLoader

def test_adversarial_robustness():
    """Verify if the ensemble can be fooled by small input perturbations."""
    loader = ModelLoader()
    xgb = loader.get_xgb_model()
    
    # 1. Start with a known attack packet (95 features)
    attack_packet = np.zeros((1, 95))
    attack_packet[0, 0:10] = 5.0 # Strong signature
    
    # 2. Add adversarial noise (ε = 0.05)
    noise = np.random.uniform(-0.05, 0.05, (1, 95))
    perturbed_packet = attack_packet + noise
    
    original_prob = xgb.predict_proba(attack_packet)[0][1]
    perturbed_prob = xgb.predict_proba(perturbed_packet)[0][1]
    
    print(f"[*] Original Attack Probability: {original_prob:.2%}")
    print(f"[*] Perturbed Attack Probability: {perturbed_prob:.2%}")
    
    # If the drop is less than 15%, the model is considered robust
    if (original_prob - perturbed_prob) < 0.15:
        print("✅ PASS: Model is robust against small adversarial perturbations.")
    else:
        print("❌ FAIL: Model is vulnerable to evasion via noise.")

if __name__ == "__main__":
    test_adversarial_robustness()