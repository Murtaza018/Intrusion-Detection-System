import numpy as np
import torch
from model_loader import ModelLoader

def test_mae_structural_integrity():
    loader = ModelLoader()
    mae_model = loader.get_mae_model()
    
    # 1. Baseline: Normal distribution (Simulated BENIGN)
    normal_data = torch.randn(1, 78) * 0.1 
    
    # 2. Anomaly: Out-of-distribution (Simulated ZERO-DAY)
    # We create a structural "gap" by spiking specific feature indices
    anomaly_data = normal_data.clone()
    anomaly_data[0, 20:30] += 5.0 

    with torch.no_grad():
        rec_normal = mae_model(normal_data)
        rec_anomaly = mae_model(anomaly_data)
        
        mse_normal = torch.mean((normal_data - rec_normal)**2).item()
        mse_anomaly = torch.mean((anomaly_data - rec_anomaly)**2).item()

    print(f"[*] Normal MSE: {mse_normal:.6f}")
    print(f"[*] Anomaly MSE: {mse_anomaly:.6f}")

    if mse_anomaly > mse_normal * 10:
        print("✅ PASS: MAE isolated structural anomaly correctly.")
    else:
        print("❌ FAIL: MAE is not sensitive enough to structural novelty.")

if __name__ == "__main__":
    test_mae_structural_integrity()