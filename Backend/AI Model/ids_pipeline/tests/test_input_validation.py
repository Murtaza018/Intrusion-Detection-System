import numpy as np
from detector import IDSDetector

def test_fuzzing_resilience():
    """Ensure the pipeline handles malformed or out-of-bounds data."""
    detector = IDSDetector()
    
    # Simulate junk data: NaN values and ports that don't exist
    junk_packet = {
        'src_port': 999999, # Impossible port
        'dst_port': -1,     # Negative port
        'length': np.nan,   # Missing data
        'protocol': 'INVALID_PROTO'
    }
    
    try:
        # The detector should catch these or map them to a safe default
        result = detector.analyze_packet(junk_packet)
        print("✅ PASS: Pipeline successfully handled malformed packet.")
    except Exception as e:
        print(f"❌ FAIL: Pipeline crashed on malformed input: {e}")

if __name__ == "__main__":
    test_fuzzing_resilience()