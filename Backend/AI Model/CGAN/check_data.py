import numpy as np

# Path to your data
DATA_PATH = "../Preprocessing/CGAN/CGAN_preprocessed_data/"

try:
    print(f"[*] Loading data from {DATA_PATH}...")
    X_train = np.load(DATA_PATH + 'X_full.npy')
    
    min_val = X_train.min()
    max_val = X_train.max()
    
    print("\n" + "="*40)
    print(f"DATA STATISTICS")
    print(f"Min Value: {min_val:.4f}")
    print(f"Max Value: {max_val:.4f}")
    print("="*40)
    
    print("\nVERDICT:")
    if 0 <= min_val and max_val <= 1.0:
        print("✅ Data is [0, 1]. Use 'sigmoid' activation in Generator.")
    elif -1.0 <= min_val and max_val <= 1.0:
        print("✅ Data is [-1, 1]. Use 'tanh' activation in Generator.")
    else:
        print("⚠️ Data is Unbounded (StandardScaler?). Use 'linear' activation.")
        print("   (Note: GANs are much harder to train with unbounded data.)")

except FileNotFoundError:
    print(f"[!] Could not find files in {DATA_PATH}")