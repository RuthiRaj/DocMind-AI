"""
Concurrent Multi-User Stress Test - Run A: Unique Questions (Tests Soft Concurrency Cap in isolation).
8 simultaneous sessions x 3 questions = 24 requests.
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
        "What are the scalability and latency SLAs?"
    ]},
    {"session_num": 4, "doc_id": DOCS[0], "ip": "192.168.1.104", "questions": [
        "What diploma did they earn and what was the grade?",
        "What frontend and backend technologies are listed in technical skills?",
        "What is their career summary or objective?"
    ]},
    {"session_num": 5, "doc_id": DOCS[1], "ip": "192.168.1.105", "questions": [
        "What encryption standards are enforced for data at rest and in transit?",
        "How often are security audits and penetration testing conducted?",
        "Who is authorized to approve emergency infrastructure changes?"
    ]},
    {"session_num": 6, "doc_id": DOCS[2], "ip": "192.168.1.106", "questions": [
        "What database engines and partitioning strategies are recommended?",
        "How is identity and access management handled across regions?",
        "What monitoring and observability tools are integrated?"
    ]},
    {"session_num": 7, "doc_id": DOCS[0], "ip": "192.168.1.107", "questions": [
        "What Oracle certification is listed, and when was it obtained?",
        "What is the Smart Task Engine project about?",
        "What programming languages does the candidate know?"
    ]},
    {"session_num": 8, "doc_id": DOCS[1], "ip": "192.168.1.108", "questions": [
        "What are the incident response escalation steps?",
        "What is the maximum allowed downtime under the SLA?",
        "What physical security controls are required for data centers?"
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
                if "busy with other visitors" in raw_ans:
                    outcome = "SOFT_CAP_BUSY (200 OK)"
                else:
                    outcome = "SUCCESS_200"
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
    print("RUN A: 8 CONCURRENT SESSIONS WITH UNIQUE QUESTIONS (TESTING SOFT CAP)")
    print("================================================================================\n")

    start_all = time.perf_counter()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_user_session, cfg) for cfg in SESSION_CONFIGS]
        for future in as_completed(futures):
            future.result()

    total_duration = time.perf_counter() - start_all

    print("\n" + "="*80)
    print(f"RUN A COMPLETED IN {total_duration:.2f}s")
    print("="*80)

    total_reqs = len(all_results)
    success_reqs = sum(1 for r in all_results if r["status_code"] == 200)
    soft_cap_reqs = sum(1 for r in all_results if "SOFT_CAP" in r["outcome"])
    normal_success_reqs = sum(1 for r in all_results if r["outcome"] == "SUCCESS_200")
    rate_limited_429 = sum(1 for r in all_results if r["status_code"] == 429)
    failed_reqs = sum(1 for r in all_results if r["status_code"] not in (200, 429))
    avg_latency = sum(r["latency"] for r in all_results) / max(1, total_reqs)

    print(f"\nTotal Requests Sent: {total_reqs}")
    print(f"Total 200 OK Responses: {success_reqs} ({success_reqs/total_reqs*100:.1f}%)")
    print(f"  - Grounded RAG Generations: {normal_success_reqs}")
    print(f"  - Soft Concurrency Cap (Graceful 200 Busy): {soft_cap_reqs}")
    print(f"Hard 429 Ejections: {rate_limited_429}")
    print(f"Server Errors (500s / Socket Timeouts): {failed_reqs}")
    print(f"Average Request Latency: {avg_latency:.2f}s")

if __name__ == "__main__":
    main()
