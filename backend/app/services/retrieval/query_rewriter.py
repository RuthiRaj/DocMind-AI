"""
Query Rewriter Module.

Generates alternate phrasings of a user query using the Groq LLM to improve
semantic retrieval coverage across varied document vocabulary.
"""

import logging
from typing import List

from app.core.config import settings
from app.core.rate_limit import groq_token_window

logger = logging.getLogger(__name__)

# Compact system prompt for query expansion — optimized for low token usage
_REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's question into exactly 2 alternative phrasings. "
    "Use different vocabulary and sentence structure but preserve the exact same intent. "
    "Output ONLY the 2 rewrites, one per line, no numbering, no extra text."
)


def rewrite_query(original_query: str) -> List[str]:
    """
    Generates 2 alternate phrasings of the user's query using the Groq LLM.

    Falls back gracefully to returning only the original query if the LLM call
    fails, times out, or produces unusable output.

    Args:
        original_query (str): The user's original search query.

    Returns:
        List[str]: List of queries — always starts with the original, followed
                   by 0-2 rewrites depending on success.
    """
    queries = [original_query]

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
