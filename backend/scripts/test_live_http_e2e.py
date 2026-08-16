"""
DocMind AI - Authoritative Live HTTP End-to-End API Integration Test Suite.

Executes real HTTP requests against the live running FastAPI/uvicorn server on http://127.0.0.1:8000:
1. Generates a fresh 40-page technical PDF with repeated headers, dense paragraphs, and verifiable facts.
2. Ingests via POST /upload.
3. Processes via POST /process/{id}, POST /chunk/{id}, POST /embed/{id}, POST /index/{id}.
4. Executes N real Chat questions via POST /chat/{id} (including multi-turn session and fallback).
5. Checks citations (start_page/end_page accuracy, no collapse).
6. Checks neighbor-merge caps (all chunk sizes <= 1500 chars, <= 2 merged chunks).
7. Fetches and logs live Groq telemetry (token counts, quota headers: remaining_tokens, limit_tokens, reset_tokens).
8. Prints raw JSON request/response payloads.
"""

import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("live_http_e2e")

BASE_URL = "http://127.0.0.1:8000"
BACKEND_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BACKEND_ROOT / "test_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_40_page_pdf(filepath: Path) -> Path:
    """
    Generates a 40-page technical PDF with repeated headers, dense paragraphs,
    and distinct verifiable facts on specific pages to test page citation resolution.
    """
    doc = fitz.open()

    # Define verifiable anchor facts on specific pages
    anchor_facts = {
        4: (
            "Section 4.2 - Quantum Encryption Subsystem\n\n"
            "The Quantum Encryption Core (QEC-9982) is active on port 8443 with a key length of 4096 bits. "
            "It enforces asymmetric ephemeral key rotation every 300 seconds across all primary egress routes. "
            "Hardware security modules (HSM) maintain isolation of master seed keys."
        ),
        17: (
            "Section 17.8 - Thermal Dissipation & Environmental Controls\n\n"
            "Thermal Dissipation Matrix: The liquid cooling threshold is strictly calibrated to 42.5 degrees Celsius. "
            "If rack temperatures exceed 48.0 degrees Celsius, auxiliary coolant pumps activate immediately. "
            "Sensor telemetry reports ambient temperature every 500 milliseconds to the central controller."
        ),
        27: (
            "Section 27.3 - Distributed Consensus Specification\n\n"
            "Consensus Mechanism: The Byzantine Fault Tolerant quorum requires exactly 29 validator signatures out of 40. "
            "Block proposal rounds execute in 250-millisecond epochs with optimistic pipelining. "
            "Slashing conditions penalize equivocation by 100% of bonded stake."
        ),
        39: (
            "Section 39.1 - Disaster Recovery & Failover Protocol\n\n"
            "Emergency Failover Procedure: Code Black emergency shutdown is triggered by executing sequence EPS-DELTA-9. "
            "Upon execution, all persistent database replicas transition to read-only mode within 15 milliseconds. "
            "Traffic is re-routed to the secondary disaster recovery site in Region West."
        )
    }

    for page_num in range(1, 41):
        page = doc.new_page(width=595, height=842)
        
        # Header (repeated across pages to test repeated text immunity)
        header_text = f"DOCMIND ENTERPRISE ARCHITECTURE SPECIFICATION - REVISION 9.4 (PAGE {page_num}/40)"
        page.insert_text(fitz.Point(50, 40), header_text, fontsize=9, color=(0.3, 0.3, 0.3))

        # Title
        title_text = f"Chapter {page_num}: Technical Systems Architecture & Operations"
        page.insert_text(fitz.Point(50, 70), title_text, fontsize=12, color=(0, 0, 0))

        # Content
        if page_num in anchor_facts:
            body = anchor_facts[page_num] + "\n\n" + (
                f"Standard operational telemetry for Chapter {page_num} remains within baseline tolerances. "
                f"Engineering teams must review subsystem metrics according to quarterly operational procedures. "
                f"All access logs are retained in encrypted cold storage for compliance auditing."
            )
        else:
            body = (
                f"This is section {page_num}.1 describing the operational parameters for system module {page_num}. "
                f"The subsystem maintains continuous event streaming with buffer capacity configured for standard workload queues. "
                f"Inter-service dependencies adhere to microservice boundary policies and distributed tracing standards. "
                f"Load balancers distribute ingress traffic evenly across worker nodes in availability zone {page_num % 3 + 1}. "
                f"\n\n"
                f"Operational health checks execute every 10 seconds via synthetic HTTP probes. "
                f"Failure thresholds initiate automated pod replacement within 30 seconds of persistent unresponsiveness. "
                f"All metrics are published to Prometheus scrape targets on port 9090."
            )

        page.insert_textbox(
            fitz.Rect(50, 95, 545, 780),
            body,
            fontsize=10,
            color=(0.1, 0.1, 0.1)
        )

        # Footer
        footer_text = "CONFIDENTIAL & PROPRIETARY - DOCMIND SYSTEMS INC."
        page.insert_text(fitz.Point(50, 810), footer_text, fontsize=8, color=(0.4, 0.4, 0.4))

    doc.save(str(filepath))
    doc.close()
    logger.info("Generated 40-page test PDF at '%s' (%d bytes)", filepath.name, filepath.stat().st_size)
    return filepath


async def run_live_e2e_test():
    """Executes the complete live HTTP API integration test against the running server."""
    pdf_path = OUTPUT_DIR / "docmind_40_page_spec.pdf"
    generate_40_page_pdf(pdf_path)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # Step 0: Verify Server Liveness
        logger.info("=== STEP 0: Checking Server Health ===")
        health_res = await client.get("/health")
        assert health_res.status_code == 200, f"Server health check failed: {health_res.text}"
        logger.info("Server is healthy: %s", health_res.json())

        # Step 1: Upload 40-Page PDF
        logger.info("\n=== STEP 1: POST /upload ===")
        with open(pdf_path, "rb") as f:
            upload_res = await client.post(
                "/upload",
                files={"file": ("docmind_40_page_spec.pdf", f, "application/pdf")}
            )
        assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
        upload_data = upload_res.json()
        doc_id = upload_data["document_id"]
        logger.info("Upload response: status=%d doc_id=%s filename=%s", upload_res.status_code, doc_id, upload_data.get("filename"))

        # Step 2: Ingestion Pipeline (Process -> Chunk -> Embed -> Index)
        for stage in ["process", "chunk", "embed", "index"]:
            logger.info("\n=== STEP 2: POST /%s/%s ===", stage, doc_id)
            t0 = time.perf_counter()
            stage_res = await client.post(f"/{stage}/{doc_id}")
            t1 = time.perf_counter()
            assert stage_res.status_code == 200, f"Stage {stage} failed: {stage_res.text}"
            logger.info("Stage '%s' succeeded in %.2fs: %s", stage, (t1 - t0), json.dumps(stage_res.json(), indent=2))

        # Step 3: Check Ingested Chunks & Page Offsets
        logger.info("\n=== STEP 3: Verifying Ingestion Artifacts & Page Offsets ===")
        doc_details_res = await client.get(f"/documents/{doc_id}")
        assert doc_details_res.status_code == 200, f"Get document failed: {doc_details_res.text}"
        doc_details = doc_details_res.json()
        total_pages = doc_details.get("metadata", {}).get("total_pages", 0)
        total_chunks = doc_details.get("chunk_statistics", {}).get("total_chunks", 0)
        logger.info("Document details: total_pages=%d total_chunks=%d", total_pages, total_chunks)
        assert total_pages == 40, f"Expected 40 pages, got {total_pages}"
        assert total_chunks >= 30, f"Expected >= 30 chunks, got {total_chunks}"

        # Step 4: Execute Real Chat Questions Against Live Server
        session_id = str(uuid.uuid4())
        chat_scenarios = [
            {
                "name": "Fact on Page 4 (Quantum Encryption)",
                "question": "What is the key length and active port for the Quantum Encryption Core QEC-9982?",
                "expected_page": 4,
                "expected_keywords": ["8443", "4096"],
            },
            {
                "name": "Fact on Page 17 (Thermal Dissipation)",
                "question": "What is the liquid cooling temperature threshold for the thermal dissipation matrix?",
                "expected_page": 17,
                "expected_keywords": ["42.5"],
            },
            {
                "name": "Fact on Page 27 (Consensus Quorum)",
                "question": "How many validator signatures are required for the Byzantine Fault Tolerant quorum?",
                "expected_page": 27,
                "expected_keywords": ["29", "40"],
            },
            {
                "name": "Multi-turn Conversational Turn (Follow-up on Page 4)",
                "question": "Regarding that encryption core we discussed, how often does its key rotation occur?",
                "expected_page": 4,
                "expected_keywords": ["300"],
            },
            {
                "name": "Fact on Page 39 (Emergency Failover)",
                "question": "What sequence triggers the emergency shutdown failover procedure?",
                "expected_page": 39,
                "expected_keywords": ["EPS-DELTA-9"],
            },
            {
                "name": "Unanswerable Query (Grounded Fallback)",
                "question": "What is the secret recipe for homemade chocolate chip cookies?",
                "expected_page": None,
                "expected_keywords": ["couldn't find enough information"],
            }
        ]

        chat_results = []
        for idx, scenario in enumerate(chat_scenarios, start=1):
            logger.info("\n=======================================================")
            logger.info("CHAT QUERY %d/6: %s", idx, scenario["name"])
            logger.info("Question: '%s'", scenario["question"])
            logger.info("=======================================================")

            payload = {
                "question": scenario["question"],
                "session_id": session_id,
                "top_k": 10
            }

            t0 = time.perf_counter()
            # Retry loop for upstream rate limits if needed
            for attempt in range(4):
                chat_res = await client.post(f"/chat/{doc_id}", json=payload)
                if chat_res.status_code == 429 and attempt < 3:
                    retry_after = int(chat_res.headers.get("Retry-After", "3"))
                    logger.warning("Local/Upstream rate limit (429). Sleeping %d seconds before retry...", retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                break
            t1 = time.perf_counter()

            assert chat_res.status_code == 200, f"Chat query {idx} failed with {chat_res.status_code}: {chat_res.text}"
            data = chat_res.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])

            logger.info("Response Status: %d (took %.2fs)", chat_res.status_code, (t1 - t0))
            logger.info("Answer: %s", answer)
            logger.info("Sources Count: %d", len(sources))

            for s_idx, s in enumerate(sources, start=1):
                logger.info(
                    "  Source %d: chunk_id=%s pages=%d-%d chars=%d score=%.4f",
                    s_idx, s.get("chunk_id"), s.get("start_page"), s.get("end_page"), len(s.get("text", "")), s.get("score")
                )

            # Verification Checks
            # 1. Check keyword presence
            for kw in scenario["expected_keywords"]:
                assert kw.lower() in answer.lower(), (
                    f"Query {idx} answer missing expected keyword '{kw}'. Answer: '{answer}'"
                )

            # 2. Check citation page correctness
            if scenario["expected_page"] is not None:
                assert len(sources) > 0, f"Query {idx} expected sources, but got 0 sources."
                source_pages = set()
                for s in sources:
                    source_pages.update(range(s["start_page"], s["end_page"] + 1))
                assert scenario["expected_page"] in source_pages, (
                    f"Query {idx} expected citation on page {scenario['expected_page']}, but sources cited pages: {sorted(source_pages)}"
                )
            else:
                # Fallback query
                assert len(sources) == 0, f"Fallback query expected 0 sources, but got {len(sources)}"

            # 3. Check neighbor-merge limits (<= 1500 chars, <= 2 merged chunks)
            for s in sources:
                chunk_len = len(s.get("text", ""))
                assert chunk_len <= 1500, (
                    f"Mega-chunk violation: source chunk {s.get('chunk_id')} has {chunk_len} chars (> 1500 limit)!"
                )
                page_span = s.get("end_page") - s.get("start_page")
                assert page_span <= 2, (
                    f"Mega-chunk page span violation: source chunk spans {page_span} pages ({s.get('start_page')}-{s.get('end_page')})!"
                )

            chat_results.append({
                "scenario": scenario["name"],
                "question": scenario["question"],
                "answer": answer,
                "sources": sources,
                "time_s": round(t1 - t0, 3)
            })

            # Small breather between queries to remain within free tier TPM window
            await asyncio.sleep(2)

        # Step 5: Query Live Telemetry Endpoint
        logger.info("\n=== STEP 5: Live Groq API Telemetry & Rate-Limit Quota Metrics ===")
        telem_res = await client.get("/telemetry/recent?count=10")
        assert telem_res.status_code == 200, f"Telemetry query failed: {telem_res.text}"
        telem_data = telem_res.json()
        telemetry_entries = telem_data.get("telemetry", [])

        logger.info("Total Recorded Groq Telemetry Calls: %d", len(telemetry_entries))
        for t_idx, entry in enumerate(telemetry_entries[-5:], start=1):
            logger.info("----------------------------------------------------------------")
            logger.info("REAL GROQ REQUEST #%d:", t_idx)
            logger.info("  Timestamp          : %s", entry.get("timestamp"))
            logger.info("  Call Type          : %s", entry.get("call_type"))
            logger.info("  Query              : %s", entry.get("query"))
            logger.info("  Prompt Tokens      : %s", entry.get("prompt_tokens"))
            logger.info("  Completion Tokens  : %s", entry.get("completion_tokens"))
            logger.info("  Total Tokens       : %s", entry.get("total_tokens"))
            logger.info("  Remaining Tokens   : %s", entry.get("remaining_tokens"))
            logger.info("  Limit Tokens       : %s", entry.get("limit_tokens"))
            logger.info("  Reset Tokens Time  : %s", entry.get("reset_tokens"))
            logger.info("  Remaining Requests : %s", entry.get("remaining_requests"))
            logger.info("----------------------------------------------------------------")

        logger.info("\n=======================================================")
        logger.info("ALL 5 VERIFICATION STEPS PASSED AGAINST LIVE HTTP API!")
        logger.info("=======================================================\n")
        return {
            "document_id": doc_id,
            "chat_results": chat_results,
            "telemetry": telemetry_entries[-5:]
        }


if __name__ == "__main__":
    result = asyncio.run(run_live_e2e_test())
    print("\n--- JSON OUTPUT SUMMARY ---")
    print(json.dumps(result, indent=2))
