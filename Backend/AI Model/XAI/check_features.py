
import numpy as np, json, os

X_PATH = "Backend/AI Model/Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/X_train.npy"
OUT_PATH = "Backend/AI Model/XAI/feature_schema.json"

X = np.load(X_PATH)
n_features = X.shape[1]

# make placeholder names
features = [f"f{i}" for i in range(n_features)]

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(features, f, indent=2)

print(f"✅ Created feature_schema.json with {n_features} features.")
