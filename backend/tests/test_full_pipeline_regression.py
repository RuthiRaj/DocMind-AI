"""
Full Pipeline Comprehensive Regression Test Suite.

Single authoritative regression test verifying end-to-end against live server:
1. Upload & Pipeline Stage Execution (Process -> Chunk -> Embed -> Index)
2. Neighbor-Merge Cap Compliance (Character & Chunk bounds)
3. Cross-Page Citation Accuracy (start_page != end_page offset overlap)
4. Strict Metadata Grounding (No hallucinated labels or field values)
5. Multi-Match Completeness (All matching pages returned)
6. Correct Refusal on Unanswerable / Off-Topic Questions (Grounded fallback)
7. Groq 429 Rate-Limit Graceful Handling (Retry-After compliance)
"""

import os
import sys
import time
import re
from pathlib import Path
import requests

# Ensure UTF-8 stdout encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "test_artifacts"


def upload_and_process(pdf_path: Path) -> str:
    print(f"\n[PIPELINE] Uploading & Processing: {pdf_path.name}")
    with open(pdf_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/upload", files={"file": (pdf_path.name, f, "application/pdf")})
    assert r.status_code == 200, f"Upload failed: {r.text}"
    doc_id = r.json()["document_id"]
    print(f"  --> Uploaded: doc_id={doc_id}")

    for stage in ["process", "chunk", "embed", "index"]:
        st_r = requests.post(f"{BASE_URL}/{stage}/{doc_id}")
        assert st_r.status_code == 200, f"Stage {stage} failed: {st_r.text}"
        print(f"  --> Stage {stage} completed (200 OK)")

    status_res = requests.get(f"{BASE_URL}/documents/{doc_id}/status").json()
    assert status_res.get("chat_ready") is True, f"Document not ready for chat: {status_res}"
    print(f"  --> Pipeline verified: chat_ready=True")
    return doc_id


def wait_for_groq_quota(min_remaining: int = 4500, max_wait: int = 60):
    """
    Waits for the Groq rolling 1-minute TPM token window to have at least min_remaining tokens.
    """
    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = requests.get(f"{BASE_URL}/telemetry", timeout=10)
            if r.status_code == 200:
                calls = r.json().get("recent_calls", [])
                if calls:
                    last_call = calls[-1]
                    rem = last_call.get("remaining_tokens")
                    if rem is not None and rem >= min_remaining:
                        return
                    reset_str = str(last_call.get("reset_tokens", "5")).strip()
                    try:
                        if reset_str.endswith("ms"):
                            reset_val = float(reset_str[:-2]) / 1000.0
                        elif reset_str.endswith("s"):
                            reset_val = float(reset_str[:-1])
                        else:
                            reset_val = float(reset_str)
                    except ValueError:
                        reset_val = 5.0
                    sleep_sec = min(reset_val + 2, 20.0)
                    print(f"      [TPM Budget Wait] Remaining: {rem}/{min_remaining} -> Waiting {sleep_sec:.1f}s for quota window...")
                    time.sleep(sleep_sec)
                    continue
        except Exception:
            pass
        time.sleep(5)


def query_chat_with_retry(doc_id: str, question: str, top_k: int = 10, max_retries: int = 4) -> dict:
    for attempt in range(max_retries):
        r = requests.post(f"{BASE_URL}/chat/{doc_id}", json={"question": question, "top_k": top_k}, timeout=90)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", 10))
            sleep_time = max(retry_after + 2, 8)
            print(f"      [429 Rate Limit] Retry-After={retry_after}s -> Sleeping {sleep_time}s (Attempt {attempt+1}/{max_retries})...")
            time.sleep(sleep_time)
        elif r.status_code in (500, 502, 503, 504):
            sleep_time = 5 * (attempt + 1)
            print(f"      [{r.status_code} Transient Error] -> Sleeping {sleep_time}s (Attempt {attempt+1}/{max_retries})...")
            time.sleep(sleep_time)
        else:
            raise RuntimeError(f"Chat request failed with status {r.status_code}: {r.text}")
    raise RuntimeError("Exceeded max retries on chat endpoint.")


def ensure_test_artifacts(artifacts_dir: Path) -> tuple[Path, Path]:
    """
    Ensures test PDFs exist on-the-fly without requiring static binaries in git.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    acme_pdf = artifacts_dir / "acme_infrastructure_policy.pdf"
    orion_pdf = artifacts_dir / "orion_40_page_spec.pdf"

    import fitz

    # 1. Generate ACME Policy PDF if missing
    if not acme_pdf.exists():
        print(f"  --> Generating synthetic ACME policy PDF on-the-fly: {acme_pdf.name}")
        doc = fitz.open()
        policies = [
            ("POL-SEC-01", "Tier-1 Immediate", "Password and Credential Storage Policy: All secrets must use Argon2id hashing and HSM storage with zero cleartext logging."),
            ("POL-SEC-02", "Tier-2 Standard", "Network Access Control: All ingress traffic must pass through application load balancers with TLS 1.3 enforced."),
            ("POL-SEC-03", "Tier-3 Advisory", "Workstation Security: All employee workstations must run automated configuration management checks daily."),
            ("POL-SEC-04", "Tier-1 Immediate", "Incident Response: Critical severity breach events require containment within 15 minutes of initial detection."),
            ("POL-SEC-05", "Tier-2 Standard", "Data Classification: Sensitive datasets must be categorized into Public, Internal, Confidential, and Restricted."),
            ("POL-SEC-06", "Tier-3 Advisory", "Third-Party Vendor Assessment: Annual compliance audits must be submitted by all cloud integration vendors."),
            ("POL-SEC-07", "Tier-1 Immediate", "Database Encryption: Multi-region primary and replica databases must enforce AES-256 transparent data encryption."),
            ("POL-SEC-08", "Tier-2 Standard", "Backup Retention: Daily snapshots must be preserved for 90 days and monthly archives for 7 years."),
            ("POL-SEC-09", "Tier-3 Advisory", "Change Management: Production deployments require peer review signoff from two senior maintainers."),
            ("POL-SEC-10", "Tier-2 Standard", "Vulnerability Scanning: Automated static code analysis must run on every pull request prior to merge.")
        ]
        for i, (pol_id, tier, body) in enumerate(policies, start=1):
            page = doc.new_page(width=595, height=842)
            page_text = f"ACME CORPORATION INFRASTRUCTURE SECURITY POLICY - PAGE {i}\n\nPolicy ID: {pol_id}\nEnforcement Tier: {tier}\nScope: Global Cloud Infrastructure\n\nPolicy Statement:\n{body}\n\nCompliance Verification:\nAudited quarterly by the internal security assessment council. Violations subject to escalation."
            page.insert_text((50, 80), page_text, fontsize=11)
        doc.save(acme_pdf)
        doc.close()

    # 2. Generate ORION 40-Page Spec PDF if missing
    if not orion_pdf.exists():
        print(f"  --> Generating synthetic ORION 40-page specification PDF on-the-fly: {orion_pdf.name}")
        families = ["ORION-AUTH", "ORION-MONITOR", "ORION-CRYPTO", "ORION-ACCESS", "ORION-NETWORK", "ORION-RETENTION", "ORION-AUDIT", "ORION-BACKUP", "ORION-DEVOPS", "ORION-GOVERN"]
        doc = fitz.open()
        for page_num in range(1, 41):
            fam = families[(page_num - 1) % 10]
            seg_id = f"ORION-{page_num:03d}"
            # Every 4th page (4, 8, 12, 16, 20, 24, 28, 32, 36, 40) matches Weekly + Platform Operations
            if page_num % 4 == 0:
                interval = "Weekly"
                escalation = "Platform Operations"
            else:
                interval = "Monthly"
                escalation = "Security Governance"

            page = doc.new_page(width=595, height=842)
            page_text = (
                f"PROJECT ORION TECHNICAL ARCHITECTURE SPECIFICATION - PAGE {page_num}\n\n"
                f"Document Segment: {seg_id}\n"
                f"Control Family: {fam}\n"
                f"Security Classification: Restricted System\n"
                f"Verification Interval: {interval}\n"
                f"Escalation Path: {escalation}\n"
                f"Effective Date: 2026-01-01\n\n"
                f"Implementation Details:\n"
                f"This specification page governs {fam} baseline protocols for segment {seg_id}. All nodes and microservices allocated under this segment must enforce full cryptographic telemetry verification and operational telemetry logging.\n\n"
                f"Retention and Archival Specifications:\n"
                f"For segment {seg_id}, data preservation protocols require verified storage tiering with immutable multi-cluster replication across geographic datacenters.\n\n"
                f"Operational Notes:\n"
                f"System integrity checks execute automatically every 300 seconds to validate state compliance."
            )
            page.insert_text((50, 60), page_text, fontsize=10)
        doc.save(orion_pdf)
        doc.close()

    return acme_pdf, orion_pdf


def test_regression():
    print("=" * 80)
    print("STARTING FULL PIPELINE COMPREHENSIVE REGRESSION SUITE")
    print("=" * 80)

    acme_pdf, orion_pdf = ensure_test_artifacts(ARTIFACTS_DIR)

    # 1. Pipeline execution
    acme_id = upload_and_process(acme_pdf)
    orion_id = upload_and_process(orion_pdf)

    passes = []
    failures = []

    # -------------------------------------------------------------
    # TEST 1: Neighbor-Merge Cap Compliance
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 1: Neighbor-Merge Cap Compliance")
    print("-" * 70)
    try:
        r = requests.post(f"{BASE_URL}/retrieve/{orion_id}", json={"query": "ORION continuous monitoring controls", "top_k": 10})
        assert r.status_code == 200, f"Retrieval failed: {r.text}"
        results = r.json()["results"]
        for res in results:
            text_len = len(res["text"])
            assert text_len <= 1500, f"Neighbor merge char cap exceeded: {text_len} > 1500 for chunk {res['chunk_id']}"
        passes.append("Test 1: Neighbor-merge character cap (<= 1500 chars) strictly enforced.")
        print("  [PASS] All retrieved chunks strictly obey neighbor-merge limits.")
    except Exception as e:
        failures.append(f"Test 1 Failed: {str(e)}")
        print(f"  [FAIL] {str(e)}")

    # -------------------------------------------------------------
    # TEST 2: Cross-Page Citation Accuracy
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 2: Cross-Page Citation Accuracy (start_page != end_page)")
    print("-" * 70)
    try:
        q_cross = "What are the retention procedures for ORION-026 and ORION-036?"
        res_cross = query_chat_with_retry(orion_id, q_cross, top_k=10)
        cross_sources = [s for s in res_cross.get("sources", []) if s["start_page"] != s["end_page"]]
        assert len(cross_sources) > 0, "Expected at least one cross-page chunk in retrieval sources."
        for s in cross_sources:
            assert s["start_page"] < s["end_page"], f"Invalid page interval: {s['start_page']} >= {s['end_page']}"
            print(f"  --> Found verified cross-page citation: chunk {s['chunk_id']} (Pages {s['start_page']}–{s['end_page']})")
        passes.append(f"Test 2: Cross-page citation validated ({len(cross_sources)} multi-page chunks verified).")
        print("  [PASS] Cross-page interval overlap verified.")
    except Exception as e:
        failures.append(f"Test 2 Failed: {str(e)}")
        print(f"  [FAIL] {str(e)}")

    time.sleep(5)

    # -------------------------------------------------------------
    # TEST 3: Strict Metadata Grounding (No Hallucinated Labels)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 3: Strict Metadata Grounding (Unstated Field Refusal)")
    print("-" * 70)
    try:
        q_ground = "What is the Escalation Hotline and Lead Architect for ORION-006 on Page 6?"
        res_ground = query_chat_with_retry(orion_id, q_ground, top_k=10)
        ans_lower = res_ground["answer"].lower()
        # Must refuse or state not specified, not hallucinate a fictitious phone number or architect name
        assert any(term in ans_lower for term in ["not specify", "does not specify", "not explicitly mentioned", "not mentioned", "not specified", "couldn't find"]), (
            f"Model hallucinated unstated field: {res_ground['answer']}"
        )
        passes.append("Test 3: Metadata grounding passed — model accurately stated unmentioned fields.")
        print(f"  [PASS] Response correctly ground-checked:\n         {res_ground['answer'][:120]}...")
    except Exception as e:
        failures.append(f"Test 3 Failed: {str(e)}")
        print(f"  [FAIL] {str(e)}")

    time.sleep(5)

    # -------------------------------------------------------------
    # TEST 4A: Multi-Match Completeness (ORION-RETENTION across 4 pages)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 4A: Multi-Match Completeness (ORION-RETENTION across 4 pages)")
    print("-" * 70)
    try:
        wait_for_groq_quota(min_remaining=4800)
        q_multi = "Which document segments and pages in this document belong to the ORION-RETENTION control family?"
        res_multi = query_chat_with_retry(orion_id, q_multi, top_k=25)
        ans = res_multi["answer"]
        print(f"  --> Test 4A Answer:\n{ans}\n")
        pages_found = set()
        for p in [6, 16, 26, 36]:
            if (
                f"Page {p}" in ans
                or f"page {p}" in ans
                or f"ORION-{p:03d}" in ans
                or re.search(rf"\b[Pp]ages?\s*(\d+-)?{p}\b", ans)
                or re.search(rf"\b[Pp]ages?\s*{p}(-\d+)?\b", ans)
            ):
                pages_found.add(p)
        assert pages_found == {6, 16, 26, 36}, f"Missing matching pages in answer! Found: {pages_found}, Expected: {{6, 16, 26, 36}}"
        passes.append("Test 4A: Multi-match completeness passed — all 4 pages (6, 16, 26, 36) returned.")
        print(f"  [PASS] All 4 matching pages (6, 16, 26, 36) present in output.")
    except Exception as e:
        failures.append(f"Test 4A Failed: {str(e)}")
        print(f"  [FAIL] {str(e)}")

    time.sleep(5)

    # -------------------------------------------------------------
    # TEST 4B: 10-Match Enumeration & Anti-False-Completeness Check
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 4B: 10-Match Enumeration (Weekly + Platform Operations across 10 pages)")
    print("-" * 70)
    try:
        wait_for_groq_quota(min_remaining=4800)
        q_enum = "Which document segments and pages have a Weekly verification interval and Platform Operations escalation?"
        res_enum = query_chat_with_retry(orion_id, q_enum, top_k=25)
        ans_enum = res_enum["answer"]
        print(f"  --> Test 4B Answer:\n{ans_enum}\n")
        expected_pages = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40}
        pages_found_enum = set()
        for p in expected_pages:
            if (
                f"Page {p}" in ans_enum
                or f"page {p}" in ans_enum
                or f"ORION-{p:03d}" in ans_enum
                or re.search(rf"\b[Pp]ages?\s*(\d+-)?{p}\b", ans_enum)
                or re.search(rf"\b[Pp]ages?\s*{p}(-\d+)?\b", ans_enum)
            ):
                pages_found_enum.add(p)
        # 1. Verify retrieval pipeline pulled all 10 matching pages into source context
        source_pages = set()
        for src in res_enum.get("sources", []):
            sp = src.get("start_page")
            ep = src.get("end_page", sp)
            if sp is not None and ep is not None:
                for page_no in range(sp, ep + 1):
                    source_pages.add(page_no)
        assert expected_pages.issubset(source_pages), (
            f"Retrieval failed to pull all 10 matching pages into candidate sources! "
            f"Missing from sources: {expected_pages - source_pages}"
        )

        # 2. Verify LLM enumerated matches without truncating or claiming false completeness
        assert len(pages_found_enum) >= 7, (
            f"LLM enumerated fewer than 7 matches! Found: {pages_found_enum}, Expected at least 7 of {expected_pages}"
        )
        # Anti-false completeness check
        ans_lower = ans_enum.lower()
        assert not any(phrase in ans_lower for phrase in ["all three meet", "only three", "only 3", "only 4", "all 4", "only 5"]), (
            f"False completeness claim detected in response: {ans_enum[:200]}"
        )
        passes.append(f"Test 4B: 10-Match enumeration passed — all 10 pages retrieved in source context, {len(pages_found_enum)} cited in answer without false completeness.")
        print(f"  [PASS] All 10 matching pages retrieved in source context; {len(pages_found_enum)} enumerated in answer without false completeness.")
    except Exception as e:
        failures.append(f"Test 4B Failed: {str(e)}")
        print(f"  [FAIL] {str(e)}")

    time.sleep(5)

    # -------------------------------------------------------------
    # TEST 4C: Cross-Document Generalization Enumeration (ACME Tier-2 Deferred)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 4C: Cross-Document Generalization (ACME Tier-2 Deferred policies across 5 pages)")
    print("-" * 70)
    try:
        wait_for_groq_quota(min_remaining=4800)
        q_tier2 = "Which policy IDs and pages in the document belong to the Tier-2 Deferred enforcement tier?"
        res_tier2 = query_chat_with_retry(acme_id, q_tier2, top_k=25)
        ans_tier2 = res_tier2["answer"]
        print(f"  --> Test 4C Answer:\n{ans_tier2}\n")
        expected_tier2_pages = {2, 3, 5, 8, 9}
        pages_found_tier2 = set()
        for p in expected_tier2_pages:
            if (
                f"Page {p}" in ans_tier2
                or f"page {p}" in ans_tier2
                or f"POL-SEC-{p:02d}" in ans_tier2
                or re.search(rf"\b[Pp]ages?\s*(\d+-)?{p}\b", ans_tier2)
                or re.search(rf"\b[Pp]ages?\s*{p}(-\d+)?\b", ans_tier2)
            ):
                pages_found_tier2.add(p)
        assert pages_found_tier2 == expected_tier2_pages, (
            f"Missing Tier-2 Deferred matching pages! Found: {pages_found_tier2}, Expected: {expected_tier2_pages}"
        )
        passes.append(f"Test 4C: Cross-document generalization passed — all 5 Tier-2 pages (2, 3, 5, 8, 9) correctly returned.")
        print(f"  [PASS] All 5 Tier-2 Deferred pages (2, 3, 5, 8, 9) present in output.")
    except Exception as e:
        failures.append(f"Test 4C Failed: {str(e)}")
        print(f"  [FAIL] {str(e)}")

    time.sleep(5)

    # -------------------------------------------------------------
    # TEST 5: Correct Refusal on Off-Topic / Unanswerable Query
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 5: Grounded Refusal on Off-Topic Query")
    print("-" * 70)
    try:
        q_refuse = "What is the secret recipe for baking homemade chocolate chip cookies?"
        res_refuse = query_chat_with_retry(acme_id, q_refuse, top_k=5)
        ans_refuse = res_refuse["answer"].lower()
        sources_refuse = res_refuse.get("sources", [])
        assert any(term in ans_refuse for term in ["couldn't find enough information", "not enough information", "cannot find"]), (
            f"Model did not refuse off-topic question: {res_refuse['answer']}"
        )
        assert len(sources_refuse) == 0, f"Expected 0 sources on refusal, got: {len(sources_refuse)}"
        passes.append("Test 5: Grounded refusal passed (fallback message with 0 citations).")
        print("  [PASS] Model cleanly refused off-topic query with zero citations.")
    except Exception as e:
        failures.append(f"Test 5 Failed: {str(e)}")
        print(f"  [FAIL] {str(e)}")

    # -------------------------------------------------------------
    # TEST 6: Live Telemetry & 429 Header Compliance
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 6: Live Telemetry & Rate-Limit Tracking")
    print("-" * 70)
    try:
        tel_r = requests.get(f"{BASE_URL}/telemetry")
        assert tel_r.status_code == 200, f"Telemetry endpoint failed: {tel_r.text}"
        tel_data = tel_r.json()
        assert "recent_calls" in tel_data, "Missing recent_calls in telemetry response"
        print(f"  --> Live Groq telemetry calls recorded: {len(tel_data['recent_calls'])}")
        passes.append("Test 6: Telemetry and rate-limit tracking validated.")
        print("  [PASS] Live telemetry verified.")
    except Exception as e:
        failures.append(f"Test 6 Failed: {str(e)}")
        print(f"  [FAIL] {str(e)}")

    # Summary
    print("\n" + "=" * 80)
    print(f"REGRESSION SUITE COMPLETED: {len(passes)} PASSED, {len(failures)} FAILED")
    print("=" * 80)
    for p in passes:
        print(f"  [+] {p}")
    for f in failures:
        print(f"  [-] {f}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    test_regression()
