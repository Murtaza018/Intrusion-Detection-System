import os
import sys
import numpy as np

# ----- Path setup -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../Backend/AI Model/XAI
AI_MODEL_DIR = os.path.dirname(BASE_DIR)                   # .../Backend/AI Model

# Make sure Backend/AI Model is importable if needed later
if AI_MODEL_DIR not in sys.path:
    sys.path.insert(0, AI_MODEL_DIR)

# ✅ Real CIC-IDS-2017 train files, relative to Backend/AI Model
X_PATH = os.path.join(
    AI_MODEL_DIR,
    "Preprocessing", "CIC-IDS-2017", "CIC-IDS-2017-Processed", "X_train.npy"
)
Y_PATH = os.path.join(
    AI_MODEL_DIR,
    "Preprocessing", "CIC-IDS-2017", "CIC-IDS-2017-Processed", "y_train.npy"
)

# ✅ Save background inside the XAI folder
OUT_PATH = os.path.join(BASE_DIR, "background.npy")


def main():
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise FileNotFoundError(f"Check X_PATH/Y_PATH. Not found:\n{X_PATH}\n{Y_PATH}")

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    # assuming 0 = Normal (change if labels differ)
    normal = X[y == 0]
    if normal.size == 0:
        raise ValueError("No normal samples found. Verify that label '0' is Normal in y_train.npy.")

    # limit for performance
    if normal.shape[0] > 5000:
        normal = normal[:5000]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, normal)
    print(f"✅ Saved background.npy with shape {normal.shape} at {OUT_PATH}")


if __name__ == "__main__":
    main()
