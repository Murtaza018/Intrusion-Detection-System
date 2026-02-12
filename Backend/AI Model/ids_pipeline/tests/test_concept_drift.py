import numpy as np
from detector import IDSDetector

def test_drift_trigger():
    """Verify if a sudden statistical shift triggers a novelty alert."""
    detector = IDSDetector()
    
    # 1. Establish a normal baseline distribution
    baseline_batch = [np.random.normal(0, 0.1, 78) for _ in range(50)]
    
    # 2. Simulate a sudden shift (Concept Drift)
    # All new packets suddenly have a mean of 2.0 instead of 0.0
    drifted_packet = np.random.normal(2.0, 0.1, 78)
    
    # The MAE should see this high 'Reconstruction Error'
    result = detector.analyze_packet({'features': drifted_packet})
    
    if result['status'] == 'zero_day':
        print("✅ PASS: Concept drift correctly identified as Zero-Day.")
    else:
        print("❌ FAIL: System failed to detect major statistical drift.")

if __name__ == "__main__":
    test_drift_trigger()