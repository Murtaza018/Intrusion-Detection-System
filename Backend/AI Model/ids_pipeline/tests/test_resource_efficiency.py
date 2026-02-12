import psutil
import os
import time
from detector import IDSDetector

def test_resource_footprint():
    """Monitor RAM and CPU usage during peak processing."""
    process = psutil.Process(os.getpid())
    detector = IDSDetector()
    
    start_mem = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()
    
    # Simulate processing 5000 packets in a burst
    for _ in range(5000):
        detector.analyze_packet({'features': np.random.rand(78)})
        
    end_mem = process.memory_info().rss / (1024 * 1024)
    duration = time.time() - start_time
    
    print(f"[*] Memory Usage Peak: {end_mem:.2f} MB")
    print(f"[*] RAM Growth: {end_mem - start_mem:.2f} MB")
    print(f"[*] CPU Load during burst: {psutil.cpu_percent()}%")
    
    # Standard NIDS requirement: < 500MB RAM for edge deployment
    assert end_mem < 500, "❌ System is too heavy for edge deployment!"
    print("✅ PASS: Resource efficiency within operational limits.")

if __name__ == "__main__":
    test_resource_footprint()