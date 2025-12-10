# upgrade_db.py
# Run this ONCE to add the 'features' column to your existing database.

import psycopg2
import os
from dotenv import load_dotenv

def get_env_path():
    """Finds the .env file by looking up the directory tree."""
    # Start where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check locations: Current, Up 1 (Backend), Up 2 (Root), Up 3 (Just in case)
    possible_paths = [
        os.path.join(current_dir, ".env"),
        os.path.join(os.path.dirname(current_dir), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(current_dir)), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), ".env")
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# 1. Load .env
env_path = get_env_path()

if env_path:
    print(f"[*] Loading .env from: {env_path}")
    load_dotenv(env_path)
else:
    print("[!] CRITICAL: Could not find .env file!")
    print("    Please make sure .env exists in the project root.")

# 2. Get Credentials
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD") 
DB_NAME = os.getenv("DB_NAME")

def upgrade():
    # Sanity Check
    if not DB_PASSWORD:
        print("[!] Error: DB_PASSWORD is missing or empty.")
        print("    Check your .env file content.")
        return

    print(f"[*] Connecting to database '{DB_NAME}' on {DB_HOST}...")

    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
        )
        conn.autocommit = True
        cursor = conn.cursor()

        print("[*] Adding 'features' column to 'packets' table...")
        
        try:
            # We use TEXT to store the JSON string of features
            cursor.execute("ALTER TABLE packets ADD COLUMN features TEXT;")
            print("[+] Column 'features' added successfully.")
        except psycopg2.errors.DuplicateColumn:
            print("[!] Column 'features' already exists (Skipping).")

        conn.close()
        print("✅ Database Upgrade Complete.")

    except Exception as e:
        print(f"[!] Database Error: {e}")

if __name__ == "__main__":
    upgrade()