"""
Generalization Verification Suite:
Tests Metadata/Label Grounding and Multi-Match Handling across two distinct document schemas:
1. acme_infrastructure_policy.pdf (10 pages - ACME Security Standards)
2. orion_40_page_spec.pdf (40 pages - ORION Control Specifications)
"""
import requests
import json
import time
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def upload_and_process_pdf(pdf_path: Path) -> str:
    print(f"\n=======================================================")
    print(f"Uploading & Processing: {pdf_path.name}")
    print(f"=======================================================")
    
    with open(pdf_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/upload", files={"file": (pdf_path.name, f, "application/pdf")})
    assert r.status_code == 200, f"Upload failed: {r.text}"
    doc_id = r.json()["document_id"]
    print(f"  [+] Uploaded -> doc_id: {doc_id}")
    
    stages = ["process", "chunk", "embed", "index"]
    for st in stages:
        r = requests.post(f"{BASE_URL}/{st}/{doc_id}")
        assert r.status_code == 200, f"Stage {st} failed: {r.text}"
        print(f"  [+] Stage {st} -> 200 OK")
        
    r_status = requests.get(f"{BASE_URL}/documents/{doc_id}/status")
    assert r_status.json().get("chat_ready") is True, f"Document not chat ready"
    print(f"  [+] Document {doc_id} chat_ready=True\n")
    return doc_id

def ask(doc_id: str, question: str, top_k: int = 10) -> dict:
    print(f"  --> Asking: {question}")
    r = requests.post(
        f"{BASE_URL}/chat/{doc_id}",
        json={"question": question, "top_k": top_k},
        timeout=60
    )
    if r.status_code == 429:
        wait = int(r.headers.get("Retry-After", 45)) + 2
        print(f"      [429 Rate Limit] Sleeping {wait}s...")
        time.sleep(wait)
        r = requests.post(
            f"{BASE_URL}/chat/{doc_id}",
            json={"question": question, "top_k": top_k},
            timeout=60
        )
    assert r.status_code == 200, f"Chat failed: {r.text}"
    return r.json()

def main():
    script_dir = Path(__file__).parent.resolve()
    
    # 1. Ingest ACME Policy Document
    acme_pdf = script_dir / "test_artifacts" / "acme_infrastructure_policy.pdf"
    acme_doc_id = upload_and_process_pdf(acme_pdf)
    
    print("="*70)
    print("DOC 1: ACME Infrastructure Policy Tests (New Schema)")
    print("="*70)
    
    # Test 1A: Grounding on unstated fields (POL-SEC-03 has no Escalation Lead or Department Code)
    q1a = "What is the Escalation Lead and Department Code for POL-SEC-03 Edge Firewall Filter Rules on Page 3?"
    res1a = ask(acme_doc_id, q1a)
    ans1a = res1a.get("answer", "")
    sources1a = res1a.get("sources", [])
    print(f"\n[RAW ANSWER 1A - ACME Unstated Field Grounding]:\n{ans1a}\n")
    print(f"[RAW SOURCES 1A]: {[{'chunk_id': s['chunk_id'], 'pages': [s['start_page'], s['end_page']]} for s in sources1a]}\n")
    
    # Test 1B: Multi-match query (Tier-1 Immediate matches POL-SEC-01 on Page 1, POL-SEC-04 on Page 4, POL-SEC-07 on Page 7)
    time.sleep(4)
    q1b = "Which policies in this document have an Enforcement Tier of Tier-1 Immediate?"
    res1b = ask(acme_doc_id, q1b)
    ans1b = res1b.get("answer", "")
    sources1b = res1b.get("sources", [])
    print(f"\n[RAW ANSWER 1B - ACME Multi-Match Filter]:\n{ans1b}\n")
    print(f"[RAW SOURCES 1B]: {[{'chunk_id': s['chunk_id'], 'pages': [s['start_page'], s['end_page']]} for s in sources1b]}\n")
    
    time.sleep(4)
    # 2. Test on ORION Document (doc_id = 70b032c4-c01a-41cd-b6df-a9d968be5269)
    orion_doc_id = "70b032c4-c01a-41cd-b6df-a9d968be5269"
    # Check if doc exists on server
    r_check = requests.get(f"{BASE_URL}/documents/{orion_doc_id}/status")
    if r_check.status_code != 200:
        orion_pdf = script_dir / "test_artifacts" / "orion_40_page_spec.pdf"
        orion_doc_id = upload_and_process_pdf(orion_pdf)
        
    print("="*70)
    print("DOC 2: ORION 40-Page Specification Tests (Original Schema)")
    print("="*70)
    
    # Test 2A: Grounding on unstated fields (ORION-006 has no Escalation Hotline or Lead Architect)
    q2a = "What is the Escalation Hotline and Lead Architect for ORION-006 on Page 6?"
    res2a = ask(orion_doc_id, q2a)
    ans2a = res2a.get("answer", "")
    sources2a = res2a.get("sources", [])
    print(f"\n[RAW ANSWER 2A - ORION Unstated Field Grounding]:\n{ans2a}\n")
    print(f"[RAW SOURCES 2A]: {[{'chunk_id': s['chunk_id'], 'pages': [s['start_page'], s['end_page']]} for s in sources2a]}\n")
    
    time.sleep(4)
    # Test 2B: Multi-match query (ORION-RETENTION control family on Pages 6, 16, 26, 36)
    q2b = "Which document segments and pages in this document belong to the ORION-RETENTION control family?"
    res2b = ask(orion_doc_id, q2b)
    ans2b = res2b.get("answer", "")
    sources2b = res2b.get("sources", [])
    print(f"\n[RAW ANSWER 2B - ORION Multi-Match Filter]:\n{ans2b}\n")
    print(f"[RAW SOURCES 2B]: {[{'chunk_id': s['chunk_id'], 'pages': [s['start_page'], s['end_page']]} for s in sources2b]}\n")

if __name__ == "__main__":
    main()
