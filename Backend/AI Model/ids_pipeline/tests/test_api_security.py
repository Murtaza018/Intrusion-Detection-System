import requests

def test_api_security_gate():
    """Verify that unauthorized requests are rejected."""
    url = "http://127.0.0.1:5001/api/stats"
    
    # 1. Try without key
    res_fail = requests.get(url)
    
    # 2. Try with key
    res_pass = requests.get(url, headers={'X-API-Key': 'MySuperSecretKey12345!'})
    
    if res_fail.status_code == 401 and res_pass.status_code == 200:
        print("✅ PASS: API Security Layer is correctly enforcing authentication.")
    else:
        print("❌ FAIL: API Security is either too permissive or misconfigured.")

if __name__ == "__main__":
    test_api_security_gate()