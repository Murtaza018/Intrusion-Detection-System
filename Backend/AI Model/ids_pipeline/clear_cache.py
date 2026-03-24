import numpy as np
import joblib

# Load replay buffer
data = np.load("replay_buffer.npz")
X_rep, y_rep = data["X"], data["y"]

# Load some real DDoS from the original CGAN data
X_full = np.load("../Preprocessing/CGAN/CGAN_preprocessed_data/X_full.npy")
y_full = np.load("../Preprocessing/CGAN/CGAN_preprocessed_data/y_full.npy")
ddos_mask = y_full == 2
X_ddos = X_full[ddos_mask][:50]  # first 50 DDoS samples
y_ddos = y_full[ddos_mask][:50]  # should be all 2

# Append to replay
X_new = np.concatenate([X_rep, X_ddos], axis=0)
y_new = np.concatenate([y_rep, y_ddos], axis=0)
print("New y unique:", np.unique(y_new))

np.savez_compressed("replay_buffer.npz", X=X_new, y=y_new)
print("DDoS re‑added to replay buffer.")
