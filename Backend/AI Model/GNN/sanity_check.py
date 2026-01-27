import os
import pickle
from tqdm import tqdm  # pip install tqdm for a nice progress bar

def run_full_integrity_check():
    folder = "./training_data/gnn_snapshots/"
    files = [f for f in os.listdir(folder) if f.endswith('.pkl')]
    
    valid_count = 0
    invalid_count = 0
    total_files = len(files)
    
    print(f"[*] Starting Integrity Check on {total_files} files...")
    
    # We use tqdm to see a progress bar for all 15,000+ files
    for filename in tqdm(files, desc="Verifying"):
        file_path = os.path.join(folder, filename)
        try:
            with open(file_path, 'rb') as f:
                # Attempt a full load to check for structural integrity
                data = pickle.load(f)
                
                # Check for required GNN keys
                if all(key in data for key in ['edge_index', 'edge_attr', 'node_count']):
                    valid_count += 1
                else:
                    invalid_count += 1
                    
        except (pickle.UnpicklingError, EOFError, MemoryError, Exception):
            # Any error during load means the file is corrupted
            invalid_count += 1
            
    print("\n" + "="*30)
    print(f"INTEGRITY REPORT")
    print("="*30)
    print(f"Total Files:   {total_files}")
    print(f"Valid Files:   {valid_count} ({(valid_count/total_files)*100:.2f}%)")
    print(f"Invalid Files: {invalid_count} ({(invalid_count/total_files)*100:.2f}%)")
    print("="*30)

    if invalid_count == 0:
        print("[+] ALL GREEN: Data is healthy for GNN training.")
    else:
        print("[!] WARNING: Some snapshots are corrupted and should be deleted.")

if __name__ == "__main__":
    run_full_integrity_check()