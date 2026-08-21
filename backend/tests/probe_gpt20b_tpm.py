"""
Probe real TPM / RPM limits for openai/gpt-oss-20b by sending rapid burst requests without retries.
Captures live HTTP response headers and actual token limits from Groq upstream.
"""

import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq, RateLimitError

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("GROQ_API_KEY is not set. Please configure it in backend/.env")
    sys.exit(1)

client = Groq(api_key=api_key, max_retries=0)
model = "openai/gpt-oss-20b"

def main():
    print(f"Probing real TPM / RPM limits on {model}...")

    # Large payload of ~1,800 tokens to test headroom accurately
    payload_text = "Provide a comprehensive, highly exhaustive treatise on vector indexing, HNSW algorithms, inverted file indexing, and BM25 token weighting mathematics. " * 30
    messages = [
        {"role": "system", "content": "You are an AI research assistant."},
        {"role": "user", "content": payload_text}
    ]

    hit_429 = False
    for i in range(1, 25):
        try:
            t0 = time.perf_counter()
            raw = client.chat.completions.with_raw_response.create(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=0.0
            )
            elapsed = time.perf_counter() - t0
            headers = dict(raw.headers)
            comp = raw.parse()
            rem_tok = headers.get("x-ratelimit-remaining-tokens")
            lim_tok = headers.get("x-ratelimit-limit-tokens")
            res_tok = headers.get("x-ratelimit-reset-tokens")
            print(f"Request {i} -> Status: 200 (took {elapsed:.2f}s) | Tokens: {comp.usage.total_tokens} | Rem: {rem_tok} / Lim: {lim_tok} | Reset: {res_tok}")
        except RateLimitError as e:
            print(f"\n=======================================================")
            print(f">>> TRIGGERED 429 ON REQUEST {i}!")
            print(f"=======================================================")
            print("Error String:", str(e))
            print("\nError Body JSON:")
            print(json.dumps(getattr(e, "body", {}), indent=2))
            if hasattr(e, "response") and e.response is not None:
                print("\n429 HTTP Response Headers:")
                for k, v in e.response.headers.items():
                    if "ratelimit" in k.lower() or "retry-after" in k.lower():
                        print(f"  {k}: {v}")
            hit_429 = True
            break

    if not hit_429:
        print("\nAll 24 requests completed without hitting a 429 limit.")

if __name__ == "__main__":
    main()
