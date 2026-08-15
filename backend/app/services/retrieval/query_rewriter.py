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
    if not getattr(settings, "ENABLE_SELECTIVE_QUERY_REWRITING", True):
        return True, "Selective query rewriting disabled in config."

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

    # Evaluate selective query rewriting heuristic filter
    qualified, reason = should_rewrite(original_query)
    if not qualified:
        logger.info("Query rewriter skipped for query '%s': %s", original_query[:50], reason)
        return queries

    try:
        from app.services.chat.groq_provider import GroqProvider

        provider = GroqProvider()
        client = provider._get_client()

        rewrite_tokens = (
            (len(_REWRITE_SYSTEM_PROMPT) + len(original_query) + settings.TOKEN_ESTIMATION_RATIO - 1)
            // settings.TOKEN_ESTIMATION_RATIO
            + 150
        )
        allowed, retry_after = groq_token_window.reserve(
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

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": original_query},
            ],
            model=settings.LLM_MODEL,
            temperature=0.7,
            max_tokens=150,
            timeout=5,
        )

        if not response.choices or not response.choices[0].message.content:
            logger.warning("Query rewriter received empty LLM response. Using original query only.")
            return queries

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

    except Exception as exc:
        logger.warning(
            "Query rewriter failed (graceful fallback to original query): %s", str(exc)
        )

    return queries
