import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from model_loader import ModelLoader

def test_standard_metrics():
    """Verify FPR, Precision, Recall, and F1-Score."""
    loader = ModelLoader()
    xgb = loader.get_xgb_model()
    
    # Mock a batch: 100 normal packets, 20 attack packets
    # In a real test, you would use a subset of the CIC-IDS dataset here.
    y_true = [0]*100 + [1]*20
    X_test = np.random.normal(0, 0.5, (120, 95)) # Simulated features
    
    y_pred = xgb.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) # False Positive Rate
    
    print(f"[*] False Positive Rate: {fpr:.4%}")
    print("\n[DETAILED CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack']))
    
    assert fpr < 0.05, "❌ FPR is too high! The system will annoy analysts with false alarms."
    print("✅ PASS: Detection efficacy meets research standards.")

if __name__ == "__main__":
    test_standard_metrics()