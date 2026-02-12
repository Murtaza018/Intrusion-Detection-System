import time
import numpy as np
import torch
from model_loader import ModelLoader

def test_throughput_benchmark():
    """Measures the inference speed of the full hybrid pipeline."""
    loader = ModelLoader()
    xgb = loader.get_xgb_model()
    mae = loader.get_mae_model()
    
    # Simulate a batch of 1000 packets
    test_batch = np.random.randn(1000, 95) 
    test_batch_torch = torch.FloatTensor(test_batch[:, :78])
    
    start_time = time.time()
    
    # Run Hybrid Inference
    with torch.no_grad():
        _ = mae(test_batch_torch) # MAE Logic
        _ = xgb.predict_proba(test_batch) # Ensemble Logic
        
    duration = time.time() - start_time
    pps = 1000 / duration
    
    print(f"[*] Total Time for 1000 packets: {duration:.4f}s")
    print(f"[*] Throughput: {pps:.2f} Packets Per Second")
    
    if pps > 100:
        print("✅ PASS: Pipeline meets real-time processing requirements.")
    else:
        print("⚠️ WARNING: Latency may be too high for high-speed networks.")

if __name__ == "__main__":
    test_throughput_benchmark()