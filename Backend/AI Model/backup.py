import shutil
import os
import datetime

# --- CONFIGURATION ---
# Define the files that are critical for the system state
CRITICAL_FILES = [
    "ids_pipeline/label_encoder.pkl",
    "ids_pipeline/replay_buffer.npz",
    "CGAN/cgan_generator.keras" 
]

BACKUP_ROOT = "Backups"

def create_backup():
    # 1. Create Timestamped Folder
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_dir = os.path.join(BACKUP_ROOT, f"State_{timestamp}")
    
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"[*] Created backup directory: {backup_dir}")

    # 2. Copy Files
    success_count = 0
    for file_path in CRITICAL_FILES:
        if os.path.exists(file_path):
            try:
                # Maintain folder structure in backup (optional, but flat is easier for restore)
                filename = os.path.basename(file_path)
                dest_path = os.path.join(backup_dir, filename)
                shutil.copy2(file_path, dest_path)
                print(f"    [+] Backed up: {filename}")
                success_count += 1
            except Exception as e:
                print(f"    [!] Failed to copy {file_path}: {e}")
        else:
            print(f"    [!] Warning: Source file not found: {file_path}")

    print("-" * 40)
    if success_count == len(CRITICAL_FILES):
        print(f"✅ Full System Backup Complete! ({success_count}/{len(CRITICAL_FILES)} files)")
        print(f"📍 Location: {backup_dir}")
    else:
        print(f"⚠️ Partial Backup Completed ({success_count}/{len(CRITICAL_FILES)} files)")

if __name__ == "__main__":
    create_backup()