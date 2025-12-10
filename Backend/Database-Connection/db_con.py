# db_setup.py
# Run this script ONCE to initialize the PostgreSQL database and tables.
# It reads credentials from your .env file.

import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
# Load environment variables
load_dotenv()

# Configuration
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
TARGET_DB_NAME = os.getenv("DB_NAME")

def setup_database():
    print(f"---  Initializing Database: {TARGET_DB_NAME} ---")
    
    try:
        # 1. Connect to default 'postgres' DB to create the new DB
        print(f"[*] Connecting to PostgreSQL system...")
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname="postgres"
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # 2. Check if DB exists, create if not
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{TARGET_DB_NAME}'")
        exists = cursor.fetchone()
        
        if not exists:
            print(f"[*] Creating database '{TARGET_DB_NAME}'...")
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TARGET_DB_NAME)))
            print(f"[+] Database created successfully.")
        else:
            print(f"[!] Database '{TARGET_DB_NAME}' already exists. Skipping creation.")
        
        cursor.close()
        conn.close()

        # 3. Connect to the NEW DB to create Tables
        print(f"[*] Connecting to '{TARGET_DB_NAME}' to create schema...")
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=TARGET_DB_NAME
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # 4. Create Packets Table
        print("[*] Creating/Verifying 'packets' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS packets (
                id SERIAL PRIMARY KEY,
                packet_id_backend INTEGER UNIQUE NOT NULL, 
                summary TEXT,
                src_ip VARCHAR(50),
                dst_ip VARCHAR(50),
                protocol VARCHAR(10),
                src_port INTEGER,
                dst_port INTEGER,
                length INTEGER,
                timestamp TIMESTAMP,
                status VARCHAR(20),
                confidence REAL,
                explanation TEXT
            );
        """)
        
        # 5. Create Indices (For Speed)
        print("[*] Creating indices...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_packet_id ON packets (packet_id_backend);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON packets (status);")
        
        print("[+] Schema setup complete.")
        print("\n✅ SUCCESS: Database is ready for the IDS Pipeline.")

    except Exception as e:
        print(f"\n[!] Error setting up database: {e}")

if __name__ == "__main__":
    setup_database()