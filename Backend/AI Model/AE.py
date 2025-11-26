# test_autoencoder.py
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from inference import predict

print("=== Testing Autoencoder Model ===")

# Test with different input patterns
test_cases = [
    ("Random small", np.random.random(78) * 0.1),
    ("Random medium", np.random.random(78) * 1.0),
    ("Random large", np.random.random(78) * 10.0),
    ("Zeros", np.zeros(78)),
    ("Ones", np.ones(78)),
]

for name, features in test_cases:
    print(f"\n--- Test: {name} ---")
    print(f"Input - min: {features.min():.4f}, max: {features.max():.4f}, mean: {features.mean():.4f}")
    
    try:
        result = predict(features, "zero_day_hunter")
        print(f"Reconstruction error: {result.get('score', 'N/A')}")
        print(f"Threshold: {result.get('threshold', 'N/A')}")
    except Exception as e:
        print(f"Error: {e}")