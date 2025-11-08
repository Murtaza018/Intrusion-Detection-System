# ids_backend.py
# This script now implements Step 3 of our roadmap.
# We are enabling SSL/TLS (HTTPS) on our Flask server
# using the ECC certificate and key we generated.

from flask import Flask, jsonify
from datetime import datetime, timedelta
import ssl  # <-- We import the ssl module

# Initialize the Flask application
app = Flask(__name__)

# --- Mock Data ---
# This remains the same as before.
mock_alerts = [
    {
        "id": "1a2b3c",
        "timestamp": datetime.now().isoformat(),
        "type": "Nmap Xmas Scan",
        "source_ip": "192.168.1.101",
        "dest_port": 23,
        "severity": "High",
        "details": "A suspicious scan consistent with Nmap was detected."
    },
    {
        "id": "4d5e6f",
        "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat(),
        "type": "SQL Injection Attempt",
        "source_ip": "10.0.2.15",
        "dest_port": 80,
        "severity": "Critical",
        "details": "Detected a potential SQL injection pattern in an HTTP request payload."
    }
]

# --- API Endpoint Definition ---
# The endpoint logic is exactly the same.
@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """
    This function is called when a user makes a GET request
    to the /api/alerts endpoint.
    """
    print("[LOG] /api/alerts endpoint was accessed.")
    return jsonify(mock_alerts)

# --- Run the Server (Now with HTTPS) ---
if __name__ == "__main__":
    print("--- Starting Secure IDS Backend Server (HTTPS) ---")
    
    # ** THE UPGRADE **
    # We define the 'ssl_context' to point to our generated
    # certificate and private key.
    # Flask will automatically use these to serve traffic over HTTPS.
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        context.load_cert_chain('cert.pem', 'key.pem')
    except FileNotFoundError:
        print("\n[!] Error: 'cert.pem' or 'key.pem' not found.")
        print("[!] Please make sure you have run the 'openssl' commands to generate them first.")
        exit()

    print("[*] To test, open your browser and go to: https://127.0.0.1:5000/api/alerts")
    print("[!] Your browser will show a security warning. This is NORMAL.")
    print("[!] Click 'Advanced' and 'Proceed to 127.0.0.1' to accept your self-signed certificate.")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        ssl_context=context  # <-- This is the line that enables HTTPS
    )
