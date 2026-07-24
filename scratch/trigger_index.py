import requests
import json

url = "http://127.0.0.1:8000/index/95b4cde3-f18f-4f7c-9214-3a568280df98"
try:
    response = requests.post(url, params={"force": "true"}, timeout=60)
    print(f"Status Code: {response.status_code}")
    print("\nResponse Body:")
    try:
        print(json.dumps(response.json(), indent=2))
    except Exception:
        print(response.text)
except Exception as e:
    print(f"Request failed: {str(e)}")
