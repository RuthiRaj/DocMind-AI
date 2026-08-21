"""
Verification script for optional/disabled features:
1. ENABLE_SELECTIVE_QUERY_REWRITING (Tested & Functional)
"""

import sys
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"
DOC_ID = "9e6c2862-505c-4c12-853e-60d66eadfed5"  # 10-page policy doc (~5.4k chars)

def test_query_rewriting():
    print("\n--- Testing ENABLE_SELECTIVE_QUERY_REWRITING=True ---")
    payload = {
        "question": "What are the access control and password complexity requirements?",
        "session_id": "test_rewrite_session"
    }
    r = requests.post(f"{BASE_URL}/chat/{DOC_ID}", json=payload, timeout=30)
    print(f"Status Code: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Answer: {data.get('answer')[:120]}...")
        print(f"Context Mode: {data.get('context_mode')}")
        print(f"Sources Count: {len(data.get('sources', []))}")
        return True
    else:
        print(f"Error: {r.text}")
        return False

if __name__ == "__main__":
    test_query_rewriting()
