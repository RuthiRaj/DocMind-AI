"""
Concurrent Multi-User Stress Test - Run B: Repeated Questions (Tests In-Memory Response Caching in isolation).
8 simultaneous sessions x 3 questions = 24 requests.
Multiple sessions ask identical / normalized variations of popular questions about documents.
"""

import sys
import time
import uuid
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"

DOCS = [
    "ffd778ef-9d57-4feb-8d73-4343c3f79d1f",  # Resume
    "9e6c2862-505c-4c12-853e-60d66eadfed5",  # Acme Infrastructure Policy
    "14854346-dc56-4d06-96a4-8dcc7b174605",  # Enterprise Architecture Report
]

SESSION_CONFIGS = [
    {"session_num": 1, "doc_id": DOCS[0], "ip": "192.168.1.101", "questions": [
        "What is the candidate's email address and phone number?",
        "What educational degree did they earn and from which college?",
        "What AI certifications or projects are mentioned?"
    ]},
    {"session_num": 2, "doc_id": DOCS[1], "ip": "192.168.1.102", "questions": [
        "What is the server retention and backup policy?",
        "What are the access control and password requirements?",
        "What are the disaster recovery protocols?"
    ]},
    {"session_num": 3, "doc_id": DOCS[2], "ip": "192.168.1.103", "questions": [
        "What is the enterprise architecture vision and core pillars?",
        "What microservices and cloud infrastructure standards are outlined?",
        "What database engines and partitioning strategies are recommended?"
    ]},
    # Session 4 asks identical questions to Session 1 on Doc 0 (Tests Exact Cache Hits)
    {"session_num": 4, "doc_id": DOCS[0], "ip": "192.168.1.104", "questions": [
        "What is the candidate's email address and phone number?",
        "What educational degree did they earn and from which college?",
        "What AI certifications or projects are mentioned?"
    ]},
    # Session 5 asks normalized punctuation/casing variations of Session 2 on Doc 1 (Tests Normalized Cache Hits)
    {"session_num": 5, "doc_id": DOCS[1], "ip": "192.168.1.105", "questions": [
        "what is the server retention & backup policy???",
        "what are the access control and password requirements",
        "what are the disaster recovery protocols?"
    ]},
    # Session 6 asks normalized variations of Session 3 on Doc 2
    {"session_num": 6, "doc_id": DOCS[2], "ip": "192.168.1.106", "questions": [
        "What is the enterprise architecture vision and core pillars",
        "What microservices and cloud infrastructure standards are outlined?",
        "What database engines and partitioning strategies are recommended?"
    ]},
    # Session 7 asks identical questions to Session 1
    {"session_num": 7, "doc_id": DOCS[0], "ip": "192.168.1.107", "questions": [
        "What is the candidate's email address and phone number?",
        "What educational degree did they earn and from which college?",
        "What AI certifications or projects are mentioned?"
    ]},
    # Session 8 asks identical questions to Session 2
    {"session_num": 8, "doc_id": DOCS[1], "ip": "192.168.1.108", "questions": [
        "What is the server retention and backup policy?",
        "What are the access control and password requirements?",
        "What are the disaster recovery protocols?"
    ]},
]

barrier = threading.Barrier(8)
results_lock = threading.Lock()
all_results = []

def run_user_session(config):
    session_num = config["session_num"]
    doc_id = config["doc_id"]
    ip = config["ip"]
    questions = config["questions"]
    session_id = f"sess_{session_num}_{uuid.uuid4().hex[:6]}"
    
    barrier.wait()
    
    for q_idx, q in enumerate(questions, 1):
        headers = {
            "Content-Type": "application/json",
            "X-Forwarded-For": ip
        }
        payload = {
            "question": q,
            "session_id": session_id
        }
        
        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"{BASE_URL}/chat/{doc_id}",
                json=payload,
                headers=headers,
                timeout=60
            )
            latency = time.perf_counter() - t0
            status_code = r.status_code
            
            if status_code == 200:
                data = r.json()
                raw_ans = data.get("answer", "").strip()
                ans = raw_ans.replace("\n", " ")[:120]
                sources = len(data.get("sources", []))
                gen_time = data.get("generation_time_ms", 0)
                if latency < 0.20 or gen_time == 0:
                    outcome = "CACHE_HIT (200 OK)"
                elif "busy with other visitors" in raw_ans:
                    outcome = "SOFT_CAP_BUSY (200 OK)"
                else:
                    outcome = "SUCCESS_200 (LLM)"
                err_msg = ""
            elif status_code == 429:
                outcome = "RATE_LIMITED_429"
                ans = ""
                sources = 0
                err_msg = r.json().get("message", r.text)[:100]
            else:
                outcome = f"HTTP_{status_code}"
                ans = ""
                sources = 0
                err_msg = r.text[:100]
                
        except Exception as e:
            latency = time.perf_counter() - t0
            status_code = 0
            outcome = "EXCEPTION"
            ans = ""
            sources = 0
            err_msg = str(e)[:100]
            
        record = {
            "session_num": session_num,
            "session_id": session_id,
            "turn": q_idx,
            "doc_id": doc_id[:8],
            "ip": ip,
            "question": q[:50],
            "status_code": status_code,
            "outcome": outcome,
            "latency": latency,
            "sources": sources,
            "answer_preview": ans,
            "error_detail": err_msg,
            "timestamp": time.strftime("%H:%M:%S")
        }
        
        with results_lock:
            all_results.append(record)
            print(f"[{record['timestamp']}] [SESS {session_num} T{q_idx}] Status: {status_code} ({outcome}) | Latency: {latency:.2f}s | Sources: {sources} | Doc: {doc_id[:8]}")
            if ans:
                print(f"   Ans: {ans}...")
            if err_msg:
                print(f"   Err: {err_msg}")
        
        time.sleep(3.0)
        
    return session_num

def main():
    print("================================================================================")
    print("RUN B: 8 CONCURRENT SESSIONS WITH REPEATED QUESTIONS (TESTING RESPONSE CACHING)")
    print("================================================================================\n")

    start_all = time.perf_counter()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_user_session, cfg) for cfg in SESSION_CONFIGS]
        for future in as_completed(futures):
            future.result()

    total_duration = time.perf_counter() - start_all

    print("\n" + "="*80)
    print(f"RUN B COMPLETED IN {total_duration:.2f}s")
    print("="*80)

    total_reqs = len(all_results)
    success_reqs = sum(1 for r in all_results if r["status_code"] == 200)
    cache_hit_reqs = sum(1 for r in all_results if "CACHE_HIT" in r["outcome"])
    llm_success_reqs = sum(1 for r in all_results if "SUCCESS_200" in r["outcome"])
    soft_cap_reqs = sum(1 for r in all_results if "SOFT_CAP" in r["outcome"])
    rate_limited_429 = sum(1 for r in all_results if r["status_code"] == 429)
    failed_reqs = sum(1 for r in all_results if r["status_code"] not in (200, 429))
    avg_latency = sum(r["latency"] for r in all_results) / max(1, total_reqs)

    print(f"\nTotal Requests Sent: {total_reqs}")
    print(f"Total 200 OK Responses: {success_reqs} ({success_reqs/total_reqs*100:.1f}%)")
    print(f"  - Cached Instant Hits (<0.1s latency): {cache_hit_reqs} ({cache_hit_reqs/total_reqs*100:.1f}%)")
    print(f"  - Fresh Grounded RAG Generations: {llm_success_reqs}")
    print(f"  - Soft Concurrency Cap (Graceful Busy): {soft_cap_reqs}")
    print(f"Hard 429 Ejections: {rate_limited_429}")
    print(f"Server Errors / Timeouts: {failed_reqs}")
    print(f"Average Request Latency: {avg_latency:.2f}s")

if __name__ == "__main__":
    main()
