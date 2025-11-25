# make_minmax.py  (run where Preprocessing/CIC-IDS-2017-Processed/X_train.npy exists)
import numpy as np, json, os

base = os.path.dirname(__file__)  # adjust if running from different dir
xpath = os.path.join(base, "X_train.npy")

X = np.load(xpath)
mins = X.min(axis=0).tolist()
maxs = X.max(axis=0).tolist()

out = {"min": mins, "max": maxs}
with open(os.path.join(base, "feature_minmax.json"), "w") as f:
    json.dump(out, f, indent=2)

print("Saved feature_minmax.json (len mins=", len(mins), ")")
