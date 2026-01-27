import os
import pickle

folder = "./training_data/gnn_snapshots/"
files = [f for f in os.listdir(folder) if f.endswith('.pkl')]
deleted = 0

for filename in files:
    file_path = os.path.join(folder, filename)
    try:
        with open(file_path, 'rb') as f:
            pickle.load(f)
    except:
        os.remove(file_path)
        deleted += 1
        print(f"[!] Deleted corrupted file: {filename}")

print(f"[*] Pruning complete. {deleted} files removed.")