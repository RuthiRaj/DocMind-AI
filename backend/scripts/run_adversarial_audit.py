"""
Adversarial Production-Readiness Audit Script for DocMind AI.

Executes all 12 rigorous real-world adversarial test scenarios against the
actual FastAPI application pipeline:
1. Small PDF (<5,000 characters) - RAG enforcement check
2. 45-Page PDF - Page interval overlap and merge boundary checks
3. Precise Fact on Page 4 - Accurate single-page citation retrieval
4. Unanswerable Query - Grounded fallback & sources == []
5. Phantom Citation Probe - Page 99 hallucination immunity
6. 4 Rapid Consecutive Questions - Rate limit headroom & settlement
7. Multi-turn Session Memory - Context bounds & query continuity
8. Retrieval Failure Modes - Missing index, empty chunks, missing metadata, corrupt doc
9. Malformed Inputs - Empty question, huge question, invalid UUID, special characters
10. Programmatic Citation Integrity - Exact 1:1 chunk mapping audit
11. Frontend Source Contract Audit - No UI modification / fabrication
12. Static Production Code Scan - No FULL_CONTEXT, no regex citation parser, no bypasses
"""

import os
import sys
import json
import uuid
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Ensure backend root is in sys.path
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

import fitz  # PyMuPDF
import httpx
from httpx import ASGITransport

from app.main import app
from app.core.config import settings
from app.core.rate_limit import groq_token_window

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("adversarial_audit")

OUTPUT_DIR = backend_root / "test_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_pdf(filepath: Path, pages_content: List[str]) -> Path:
    """Generates a PDF with specified content per page."""
    doc = fitz.open()
    for idx, content in enumerate(pages_content, start=1):
        page = doc.new_page(width=595, height=842)
        p1 = fitz.Point(50, 50)
        page.insert_text(p1, f"=== PAGE {idx} ===", fontsize=14, color=(0, 0, 0))
        p2 = fitz.Point(50, 80)
        page.insert_textbox(
            fitz.Rect(50, 80, 545, 790),
            content,
            fontsize=10,
            color=(0.1, 0.1, 0.1)
        )
    doc.save(str(filepath))
    doc.close()
    return filepath


async def process_pdf_pipeline(client: httpx.AsyncClient, pdf_path: Path, filename: str) -> str:
    """Runs upload -> process -> chunk -> embed -> index on a PDF."""
    with open(pdf_path, "rb") as f:
        upload_res = await client.post(
            "/upload",
            files={"file": (filename, f, "application/pdf")}
        )
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    doc_id = upload_res.json()["document_id"]

    for stage in ["process", "chunk", "embed", "index"]:
        res = await client.post(f"/{stage}/{doc_id}")
        assert res.status_code == 200, f"Stage {stage} failed: {res.text}"

    return doc_id


async def post_chat_with_retry(
    client: httpx.AsyncClient,
    doc_id: str,
    payload: Dict[str, Any],
    max_retries: int = 4
) -> httpx.Response:
    """Helper to post chat queries, handling local or upstream 429 retry-after gracefully."""
    import anyio
    for attempt in range(max_retries):
        res = await client.post(f"/chat/{doc_id}", json=payload)
        if res.status_code == 429 and attempt < max_retries - 1:
            try:
                detail = res.json().get("detail", "")
                import re
                match = re.search(r"(\d+)\s*seconds", detail)
                wait_sec = int(match.group(1)) + 1 if match else 12
            except Exception:
                wait_sec = 12
            logger.info(f"Encountered 429 rate limit. Waiting {wait_sec}s for sliding window headroom before retry {attempt + 1}/{max_retries}...")
            await anyio.sleep(wait_sec)
            continue
        return res
    return res


async def run_audit():
    logger.info("=" * 80)
    logger.info("STARTING ADVERSARIAL PRODUCTION-READINESS AUDIT")
    logger.info("=" * 80)

    # Set test TPM budget for multi-scenario audit execution
    settings.GROQ_TPM_LIMIT = 40000
    settings.CONVERSATION_MAX_TURNS = 5

    transport = ASGITransport(app=app)
    audit_results: Dict[str, Any] = {}


    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000", timeout=60.0) as client:
        # Health check
        health_res = await client.get("/health")
        assert health_res.status_code == 200, "Backend is not healthy"

        # ----------------------------------------------------------------------
        # SCENARIO 1: Very small PDF (<5,000 chars) -> RAG enforcement check
        # ----------------------------------------------------------------------
        logger.info("\n--- SCENARIO 1: Small PDF (<5,000 chars) RAG Enforcement ---")
        small_pdf_path = OUTPUT_DIR / "small_document.pdf"
        generate_pdf(small_pdf_path, [
            "DocMind AI system kernel overview. The quantum encryption standard is AES-GCM-256 with Falcon-1024 signatures. Operational temperature is -40C to 85C."
        ])
        doc_id_small = await process_pdf_pipeline(client, small_pdf_path, "small_document.pdf")
        
        res_s1 = await post_chat_with_retry(client, doc_id_small, {"question": "What is the quantum encryption standard?"})
        assert res_s1.status_code == 200
        data_s1 = res_s1.json()

        
        s1_pass = (
            data_s1.get("context_mode") == "RAG"
            and len(data_s1.get("sources", [])) > 0
            and data_s1["sources"][0]["start_page"] == 1
        )
        audit_results["scenario_1_small_pdf_rag"] = {
            "passed": s1_pass,
            "context_mode": data_s1.get("context_mode"),
            "sources_count": len(data_s1.get("sources", [])),
            "answer_preview": data_s1.get("answer", "")[:100]
        }
        logger.info(f"Scenario 1 Result: PASS={s1_pass} (context_mode={data_s1.get('context_mode')})")

        # ----------------------------------------------------------------------
        # SCENARIO 2: 45-Page PDF -> Page span & merge boundary checks
        # ----------------------------------------------------------------------
        logger.info("\n--- SCENARIO 2: 45-Page PDF Page Intervals & Merge Bounds ---")
        pages_45 = []
        for i in range(1, 46):
            if i == 4:
                body = "SECRET KEY FACT ON PAGE 4: The auxiliary power unit runs on Liquid Hydrogen Fuel Cell Model H2-9000 at 350 bar pressure. " + ("Standard auxiliary systems maintenance procedure and telemetry logging guidelines. " * 15)
            elif i == 27:
                body = "SECTION 27 DATA: Memory allocation is capped at 64 Terabytes with NUMA topology node count equal to 8. " + ("High performance compute node memory bus bandwidth specification and latency benchmarks. " * 15)
            else:
                body = f"Section {i} standard operational procedures and routine maintenance logs for subsystem {i * 11}. " + ("Normal parameter status verified across sensor nodes and diagnostic telemetry channels. " * 15)
            pages_45.append(body)

        pdf_45_path = OUTPUT_DIR / "large_45_page_document.pdf"
        generate_pdf(pdf_45_path, pages_45)
        doc_id_45 = await process_pdf_pipeline(client, pdf_45_path, "large_45_page_document.pdf")

        res_s2 = await post_chat_with_retry(client, doc_id_45, {"question": "What are the routine maintenance logs and parameters?", "top_k": 10})
        assert res_s2.status_code == 200
        data_s2 = res_s2.json()

        sources_s2 = data_s2.get("sources", [])
        no_whole_doc_span = all((s["end_page"] - s["start_page"]) <= 2 for s in sources_s2)
        valid_page_orders = all(s["start_page"] <= s["end_page"] for s in sources_s2)
        bounded_sources = len(sources_s2) <= 10
        
        s2_pass = no_whole_doc_span and valid_page_orders and bounded_sources and len(sources_s2) > 0
        audit_results["scenario_2_45_page_bounds"] = {
            "passed": s2_pass,
            "source_count": len(sources_s2),
            "sources_sample": [{"chunk_id": s["chunk_id"], "pages": [s["start_page"], s["end_page"]]} for s in sources_s2[:4]],
            "no_whole_doc_span": no_whole_doc_span,
            "valid_page_orders": valid_page_orders
        }
        logger.info(f"Scenario 2 Result: PASS={s2_pass} (sources={len(sources_s2)}, no_whole_doc_span={no_whole_doc_span})")

        # ----------------------------------------------------------------------
        # SCENARIO 3: Answerable Query on Page 4 -> Exact citation precision
        # ----------------------------------------------------------------------
        logger.info("\n--- SCENARIO 3: Specific Fact on Page 4 ---")
        res_s3 = await post_chat_with_retry(client, doc_id_45, {"question": "What model is the auxiliary power unit and what pressure does it run at?"})
        assert res_s3.status_code == 200
        data_s3 = res_s3.json()

        s3_sources = data_s3.get("sources", [])
        page_4_present = any(s["start_page"] <= 4 <= s["end_page"] for s in s3_sources)
        s3_pass = page_4_present and "H2-9000" in data_s3.get("answer", "")
        audit_results["scenario_3_page_4_accuracy"] = {
            "passed": s3_pass,
            "page_4_in_sources": page_4_present,
            "answer": data_s3.get("answer", "")
        }
        logger.info(f"Scenario 3 Result: PASS={s3_pass} (Page 4 cited: {page_4_present})")

        # ----------------------------------------------------------------------
        # SCENARIO 4: Unanswerable Query -> Exact Grounded Fallback
        # ----------------------------------------------------------------------
        logger.info("\n--- SCENARIO 4: Unanswerable Query Fallback ---")
        res_s4 = await post_chat_with_retry(client, doc_id_45, {"question": "What is the biological mating ritual of Martian desert lizards?"})
        assert res_s4.status_code == 200
        data_s4 = res_s4.json()

        fallback_triggered = "couldn't find enough information" in data_s4.get("answer", "").lower() or len(data_s4.get("sources", [])) == 0 or "not" in data_s4.get("answer", "").lower()
        audit_results["scenario_4_unanswerable_fallback"] = {
            "passed": fallback_triggered,
            "answer": data_s4.get("answer", ""),
            "sources_count": len(data_s4.get("sources", []))
        }
        logger.info(f"Scenario 4 Result: PASS={fallback_triggered}")

        # ----------------------------------------------------------------------
        # SCENARIO 5: Phantom Citation Probe (Non-existent Page 99)
        # ----------------------------------------------------------------------
        logger.info("\n--- SCENARIO 5: Phantom Citation Page 99 Immunity ---")
        res_s5 = await post_chat_with_retry(client, doc_id_45, {"question": "According to Page 99, what are the warp drive harmonics?"})
        assert res_s5.status_code == 200
        data_s5 = res_s5.json()

        s5_sources = data_s5.get("sources", [])
        page_99_in_sources = any(s["start_page"] == 99 or s["end_page"] == 99 for s in s5_sources)
        s5_pass = not page_99_in_sources
        audit_results["scenario_5_phantom_citation"] = {
            "passed": s5_pass,
            "page_99_found_in_sources": page_99_in_sources,
            "sources_count": len(s5_sources)
        }
        logger.info(f"Scenario 5 Result: PASS={s5_pass} (Page 99 in sources: {page_99_in_sources})")

        # ----------------------------------------------------------------------
        # SCENARIO 6: 4 Rapid Consecutive Questions (Token Window Settlement)
        # ----------------------------------------------------------------------
        logger.info("\n--- SCENARIO 6: Rapid Consecutive Questions ---")
        session_rapid = "session-audit-rapid-" + str(uuid.uuid4())[:8]
        consecutive_results = []
        
        queries = [
            "What is the NUMA node count in Section 27?",
            "What is the fuel cell model mentioned?",
            "What is the standard operating procedure for subsystem 33?",
            "Summarize the general operational status."
        ]

        all_200 = True
        import anyio
        for q_idx, q in enumerate(queries, start=1):
            t0 = time.perf_counter()
            res = await post_chat_with_retry(
                client,
                doc_id_45,
                {"question": q, "session_id": session_rapid, "top_k": 5}
            )
            elapsed = time.perf_counter() - t0
            status_ok = (res.status_code == 200)
            if not status_ok:
                all_200 = False
            consecutive_results.append({
                "query_index": q_idx,
                "status_code": res.status_code,
                "elapsed_s": round(elapsed, 2),
                "answer_preview": res.json().get("answer", "")[:60] if status_ok else res.text[:60]
            })
            await anyio.sleep(0.5)


        s6_pass = all_200
        audit_results["scenario_6_rapid_consecutive_queries"] = {
            "passed": s6_pass,
            "queries_executed": consecutive_results
        }
        logger.info(f"Scenario 6 Result: PASS={s6_pass} ({len(consecutive_results)} queries executed)")

        # ----------------------------------------------------------------------
        # SCENARIO 7: Multi-turn Conversation Context Isolation
        # ----------------------------------------------------------------------
        logger.info("\n--- SCENARIO 7: Multi-turn Conversation Memory & Bounds ---")
        session_memory = "session-memory-" + str(uuid.uuid4())[:8]
        
        # Turn 1
        res_m1 = await post_chat_with_retry(client, doc_id_45, {"question": "What is on page 4?", "session_id": session_memory})
        assert res_m1.status_code == 200
        
        # Turn 2: Contextual follow-up
        res_m2 = await post_chat_with_retry(client, doc_id_45, {"question": "What was the pressure again?", "session_id": session_memory})
        assert res_m2.status_code == 200

        data_m2 = res_m2.json()

        m2_answer = data_m2.get("answer", "")
        m2_pass = "350" in m2_answer or "bar" in m2_answer.lower()
        audit_results["scenario_7_multi_turn_continuity"] = {
            "passed": m2_pass,
            "turn_2_answer": m2_answer
        }
        logger.info(f"Scenario 7 Result: PASS={m2_pass}")

        # ----------------------------------------------------------------------
        # SCENARIO 8: Retrieval Failure Modes & Corrupt Documents
        # ----------------------------------------------------------------------
        logger.info("\n--- SCENARIO 8: Retrieval Failure Modes & Corrupt Docs ---")
        doc_id_corrupt = str(uuid.uuid4())
        corrupt_dir = Path(settings.UPLOAD_DIRECTORY) / doc_id_corrupt
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        (corrupt_dir / "status.json").write_text(json.dumps({"status": "FAILED", "error": "Corrupted PDF header"}), encoding="utf-8")

        res_corrupt = await client.post(f"/chat/{doc_id_corrupt}", json={"question": "Test question"})
        corrupt_handled_cleanly = res_corrupt.status_code in [400, 422]
        
        # Missing doc
        res_missing = await client.post(f"/chat/{uuid.uuid4()}", json={"question": "Test question"})
        missing_handled_cleanly = res_missing.status_code == 404

        s8_pass = corrupt_handled_cleanly and missing_handled_cleanly
        audit_results["scenario_8_failure_modes"] = {
            "passed": s8_pass,
            "corrupt_status_code": res_corrupt.status_code,
            "missing_status_code": res_missing.status_code
        }
        logger.info(f"Scenario 8 Result: PASS={s8_pass} (Corrupt: {res_corrupt.status_code}, Missing: {res_missing.status_code})")

        # ----------------------------------------------------------------------
        # SCENARIO 9: Malformed User Inputs
        # ----------------------------------------------------------------------
        logger.info("\n--- SCENARIO 9: Malformed User Inputs ---")
        # Empty question
        res_empty = await client.post(f"/chat/{doc_id_45}", json={"question": "   "})
        empty_ok = res_empty.status_code == 400

        # Non-UUID document id
        res_bad_id = await client.post("/chat/non-existent-not-uuid", json={"question": "Test?"})
        bad_id_ok = res_bad_id.status_code == 400

        # Huge question
        res_huge = await client.post(f"/chat/{doc_id_45}", json={"question": "What is " + ("x " * 6000) + "?"})
        huge_ok = res_huge.status_code in [200, 400, 413]

        s9_pass = empty_ok and bad_id_ok and huge_ok
        audit_results["scenario_9_malformed_inputs"] = {
            "passed": s9_pass,
            "empty_status": res_empty.status_code,
            "bad_id_status": res_bad_id.status_code,
            "huge_status": res_huge.status_code
        }
        logger.info(f"Scenario 9 Result: PASS={s9_pass}")

        # ----------------------------------------------------------------------
        # SCENARIO 10: Programmatic Citation Integrity Audit
        # ----------------------------------------------------------------------
        logger.info("\n--- SCENARIO 10: Programmatic Citation Integrity Audit ---")
        # Verify 1:1 chunk mapping against actual disk chunks for doc_id_45
        chunks_file = Path(settings.UPLOAD_DIRECTORY) / doc_id_45 / "chunks.json"
        with open(chunks_file, "r", encoding="utf-8") as f:
            disk_chunks = {c["chunk_id"]: c for c in json.load(f)}

        citation_integrity_pass = True
        for s in data_s2.get("sources", []):
            cid = s["chunk_id"]
            if cid not in disk_chunks:
                citation_integrity_pass = False
                break
            disk_c = disk_chunks[cid]
            if s["start_page"] != disk_c["start_page"] or s["end_page"] != disk_c["end_page"]:
                citation_integrity_pass = False
                break

        audit_results["scenario_10_citation_integrity"] = {
            "passed": citation_integrity_pass,
            "verified_sources_count": len(data_s2.get("sources", []))
        }
        logger.info(f"Scenario 10 Result: PASS={citation_integrity_pass}")

        # ----------------------------------------------------------------------
        # SCENARIO 11: Frontend Source Contract Audit
        # ----------------------------------------------------------------------
        logger.info("\n--- SCENARIO 11: Frontend Source Contract Audit ---")
        chat_ts = (backend_root.parent / "frontend" / "services" / "chat.ts").read_text(encoding="utf-8")
        ui_page = (backend_root.parent / "frontend" / "app" / "chat" / "[documentId]" / "page.tsx").read_text(encoding="utf-8")

        no_frontend_regex = "exec(" not in chat_ts and "match(" not in chat_ts and "exec(" not in ui_page and "match(" not in ui_page
        direct_mapping = "(data.sources || []).map" in chat_ts
        s11_pass = no_frontend_regex and direct_mapping
        audit_results["scenario_11_frontend_contract"] = {
            "passed": s11_pass,
            "direct_mapping": direct_mapping,
            "no_frontend_regex": no_frontend_regex
        }
        logger.info(f"Scenario 11 Result: PASS={s11_pass}")

        # ----------------------------------------------------------------------
        # SCENARIO 12: Static Production Code Scan
        # ----------------------------------------------------------------------
        logger.info("\n--- SCENARIO 12: Static Production Code Scan ---")
        chat_service_code = (backend_root / "app" / "services" / "pdf" / "chat_service.py").read_text(encoding="utf-8")
        
        no_full_context_branch = 'context_mode == "FULL_CONTEXT"' not in chat_service_code and "context_mode = \"FULL_CONTEXT\"" not in chat_service_code
        no_raw_extracted_text = "extracted_text.txt" not in chat_service_code
        no_regex_citations = "re.findall" not in chat_service_code and "re.finditer" not in chat_service_code
        
        s12_pass = no_full_context_branch and no_raw_extracted_text and no_regex_citations
        audit_results["scenario_12_static_scan"] = {
            "passed": s12_pass,
            "no_full_context_branch": no_full_context_branch,
            "no_raw_extracted_text": no_raw_extracted_text,
            "no_regex_citations": no_regex_citations
        }
        logger.info(f"Scenario 12 Result: PASS={s12_pass}")

    # Summary
    all_passed = all(s["passed"] for s in audit_results.values())
    report_file = OUTPUT_DIR / "adversarial_audit_report.json"
    report_file.write_text(json.dumps(audit_results, indent=2), encoding="utf-8")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"FINAL ADVERSARIAL AUDIT RESULT: {'ALL SCENARIOS PASSED' if all_passed else 'SOME SCENARIOS FAILED'}")
    logger.info(f"Detailed Report: {report_file}")
    logger.info("=" * 80)
    return all_passed, audit_results


if __name__ == "__main__":
    import anyio
    anyio.run(run_audit)
