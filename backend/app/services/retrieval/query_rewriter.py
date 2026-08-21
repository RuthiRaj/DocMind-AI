"""
Query Rewriter Module.

Generates alternate phrasings of a user query using the Groq LLM to improve
semantic retrieval coverage across varied document vocabulary.
"""

import logging
import re
from typing import List, Tuple

from app.core.config import settings
from app.core.rate_limit import groq_token_window

logger = logging.getLogger(__name__)

# Compact system prompt for query expansion — optimized for low token usage
_REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's question into exactly 2 alternative technical search phrasings. "
    "Map colloquial or indirect terms to standard technical documentation terms (e.g. 'RAM' -> 'memory allocation limit', 'execution thread' -> 'worker thread', 'failover port' -> 'routing port'). "
    "Output ONLY the 2 rewrites, one per line, no numbering, no extra text."
)


def should_rewrite(query: str) -> Tuple[bool, str]:
    """
    Evaluates whether a query requires LLM query expansion based on heuristics.
    """
    if not getattr(settings, "ENABLE_SELECTIVE_QUERY_REWRITING", False):
        return False, "Selective query rewriting disabled in config."

    words = query.strip().split()
    min_words = getattr(settings, "REWRITE_MIN_WORD_COUNT", 5)
    if len(words) < min_words:
        return False, f"Query word count ({len(words)}) is below threshold ({min_words})."

    # Skip exact technical codes/keys (e.g. DB-PROD-9982, port 443)
    if re.search(r"\b[A-Z]{2,}-\w+-\d+\b", query) or re.search(r"\b(port|key|id|code|v2|ip)\s*[:=]?\s*\w+\b", query, re.IGNORECASE):
        return False, "Query contains exact technical entity or code identifier."

    return True, "Query qualified for LLM expansion."


def rewrite_query(original_query: str) -> List[str]:
    """
    Generates 2 alternate phrasings of the user's query using the Groq LLM if qualified.

    Falls back gracefully to returning only the original query if skipped or failed.
    """
    queries = [original_query]

    # If query rewriting is globally disabled, return immediately with zero overhead
    if not getattr(settings, "ENABLE_SELECTIVE_QUERY_REWRITING", False):
        return queries

    # Evaluate selective query rewriting heuristic filter
    qualified, reason = should_rewrite(original_query)
    if not qualified:
        logger.info("Query rewriter skipped for query '%s': %s", original_query[:50], reason)
        return queries

    try:
        # Check if Groq token window has sufficient headroom for query expansion without starving main completions
        current_tokens = groq_token_window.current_usage(window=60)
        safety_headroom = 6000  # Preserve at least 6,000 tokens for chat completion answer generation
        if current_tokens > (settings.GROQ_TPM_LIMIT - safety_headroom):
            logger.info(
                "Query rewriter skipped to prioritize Groq TPM quota for chat completions (usage=%d/%d).",
                current_tokens,
                settings.GROQ_TPM_LIMIT
            )
            return queries

        from app.services.chat.groq_provider import GroqProvider

        provider = GroqProvider()
        client = provider._get_client()

        rewrite_tokens = (
            (len(_REWRITE_SYSTEM_PROMPT) + len(original_query) + settings.TOKEN_ESTIMATION_RATIO - 1)
            // settings.TOKEN_ESTIMATION_RATIO
            + 150
        )
        allowed, retry_after, res_id = groq_token_window.reserve(
            tokens=rewrite_tokens,
            limit=settings.GROQ_TPM_LIMIT,
            window=60
        )
        if not allowed:
            logger.warning(
                "Query rewriter skipped because Groq token window is exhausted. Retry after %ds.",
                retry_after,
            )
            return queries

        try:
            raw_response = client.chat.completions.with_raw_response.create(
                messages=[
                    {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": original_query},
                ],
                model=settings.LLM_MODEL,
                temperature=settings.QUERY_REWRITE_TEMPERATURE,
                max_tokens=settings.QUERY_REWRITE_MAX_TOKENS,
                timeout=settings.QUERY_REWRITE_TIMEOUT_SECONDS,
            )
            response = raw_response.parse()
            headers = dict(raw_response.headers)

            if not response.choices or not response.choices[0].message.content:
                groq_token_window.settle(res_id, actual_tokens=0)
                logger.warning("Query rewriter received empty LLM response. Using original query only.")
                return queries

            usage_obj = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0)) if usage_obj else (rewrite_tokens - 150)
            completion_tokens = int(getattr(usage_obj, "completion_tokens", 0)) if usage_obj else 30
            actual_total = getattr(usage_obj, "total_tokens", None) if usage_obj else None

            if actual_total is not None and actual_total > 0:
                total_tokens = int(actual_total)
                groq_token_window.settle(res_id, actual_tokens=total_tokens)
            else:
                total_tokens = prompt_tokens + completion_tokens
                groq_token_window.settle(res_id, actual_tokens=total_tokens)

            rl_headers = {
                "remaining_tokens": headers.get("x-ratelimit-remaining-tokens"),
                "limit_tokens": headers.get("x-ratelimit-limit-tokens"),
                "reset_tokens": headers.get("x-ratelimit-reset-tokens"),
                "remaining_requests": headers.get("x-ratelimit-remaining-requests"),
                "limit_requests": headers.get("x-ratelimit-limit-requests"),
                "reset_requests": headers.get("x-ratelimit-reset-requests"),
            }

            from app.core.telemetry import groq_telemetry
            groq_telemetry.record_call(
                call_type="query_rewrite",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                ratelimit_headers=rl_headers,
                query=original_query,
                extra={"reason": reason}
            )

            logger.info(
                "[GROQ_TELEMETRY] call=query_rewrite prompt_tokens=%d completion_tokens=%d total_tokens=%d "
                "remaining_tokens=%s limit_tokens=%s reset_tokens=%s remaining_requests=%s",
                prompt_tokens,
                completion_tokens,
                total_tokens,
                rl_headers.get("remaining_tokens"),
                rl_headers.get("limit_tokens"),
                rl_headers.get("reset_tokens"),
                rl_headers.get("remaining_requests"),
            )

            raw_output = response.choices[0].message.content.strip()
            rewrites = [
                line.strip().lstrip("0123456789.-) ")
                for line in raw_output.splitlines()
                if line.strip() and len(line.strip()) > 5
            ]

            # Keep at most 2 rewrites, and skip any that are identical to the original
            for rw in rewrites[:2]:
                if rw.lower() != original_query.lower() and rw not in queries:
                    queries.append(rw)

            logger.info(
                "Query rewriter produced %d total queries: %s",
                len(queries),
                [q[:80] for q in queries],
            )
        except Exception:
            groq_token_window.settle(res_id, actual_tokens=0)
            raise

    except Exception as exc:
        logger.warning(
            "Query rewriter failed (graceful fallback to original query): %s", str(exc)
        )

    return queries

