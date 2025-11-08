# ids_backend.py
# This script now implements Step 5: API Key Authentication.
# The channel is already encrypted (HTTPS), and now we add a check
# to ensure that only clients with a secret key can access the data.

from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import ssl

# --- Configuration ---
app = Flask(__name__)
# THIS IS YOUR SECRET KEY.
# In a real app, this would be stored securely (e.g., in an environment variable).
# The mobile app must also have this key.
SECRET_API_KEY = "MySuperSecretKey12345!"

# --- Mock Data ---
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

# --- API Endpoint Definition (Now with Security) ---
@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """
    This function is called when a user makes a GET request
    to the /api/alerts endpoint.
    It now checks for a valid API key.
    """
    
    # ** THE UPGRADE **
    # Check if the 'X-API-Key' header was sent in the request.
    provided_key = request.headers.get("X-API-Key")
    
    if not provided_key or provided_key != SECRET_API_KEY:
        # If the key is missing or incorrect, deny access.
        print(f"[LOG] Failed API access attempt. Key: {provided_key}")
        # '401 Unauthorized' is the standard HTTP error for this.
        return jsonify({"error": "Unauthorized"}), 401
    
    # If the key is correct, proceed as normal.
    print("[LOG] Successful API access. /api/alerts endpoint was accessed.")
    return jsonify(mock_alerts)

# --- Run the Server (Still with HTTPS) ---
if __name__ == "__main__":
    print("--- Starting Secure IDS Backend Server (HTTPS + API Key) ---")
    
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        context.load_cert_chain('cert.pem', 'key.pem')
    except FileNotFoundError:
        print("\n[!] Error: 'cert.pem' or 'key.pem' not found.")
        print("[!] Please make sure they are in the same folder.")
        exit()

    print("[*] Server is running on: https://127.0.0.1:5000/api/alerts")
    print("[!] Test with a tool like Postman or curl, as your browser can't easily send the 'X-API-Key' header.")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        ssl_context=context
    )

