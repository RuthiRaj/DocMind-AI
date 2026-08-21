import requests
import json
import uuid
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"
doc_id = "c40b6da5-1bdd-42e8-a231-6e4212d0b478"
session_id = str(uuid.uuid4())

questions = [
    "What is the candidate's email address and phone number?",
    "What educational degree did they earn and from which college?",
    "What certification did they get from Oracle, and when?"
]

all_passed = True

for i, q in enumerate(questions, 1):
    print(f"\n==================== [QUESTION {i}] ====================")
    print(f"Query: {q}")
    payload = {"question": q, "session_id": session_id}
    res = requests.post(f"{BASE_URL}/chat/{doc_id}", json=payload)
    print(f"HTTP Status: {res.status_code}")
    
    if res.status_code == 200:
        data = res.json()
        print(f"Answer: {data.get('answer')}")
        sources = data.get("sources", [])
        print(f"Sources Count: {len(sources)}")
        for s in sources:
            print(f"  - Page {s.get('start_page')}: Chunk ID {s.get('chunk_id')} (Score: {s.get('score')})")
        if len(sources) == 0:
            print("WARNING: No sources returned!")
            all_passed = False
    else:
        print(f"FAILED: {res.text}")
        all_passed = False

if all_passed:
    print("\n>>> ALL 3 SEQUENTIAL QUESTIONS COMPLETED SUCCESSFULLY WITH 200 OK AND GROUNDED CITATIONS! <<<")
else:
    print("\n>>> ONE OR MORE QUESTIONS FAILED! <<<")
