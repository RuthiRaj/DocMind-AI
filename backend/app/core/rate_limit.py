import time
import logging
import threading
from typing import Dict, List, Tuple
from collections import deque
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.config import settings

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Sliding window in-memory rate limiter. Thread-safe and auto-cleaning.
    """
    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    def check_rate_limit(self, client_key: str, limit: int, window: int) -> Tuple[bool, int]:
        """
        Validates client request count within a sliding window.
        Returns:
            Tuple[bool, int]: (is_allowed, retry_after_seconds)
        """
        now = time.time()
        cutoff = now - window

        with self.lock:
            # Clean expired timestamps for this key
            timestamps = self.history.get(client_key, [])
            timestamps = [t for t in timestamps if t > cutoff]

            # Global lazy sweep if cache grows excessively large (e.g., > 2000 client IPs)
            if len(self.history) > 2000:
                self._clean_all_expired(now, cutoff)

            if len(timestamps) < limit:
                timestamps.append(now)
                self.history[client_key] = timestamps
                return True, 0
            else:
                self.history[client_key] = timestamps
                # Retry-After is window duration minus time elapsed since the oldest active timestamp
                oldest = timestamps[0]
                retry_after = max(1, int(math.ceil(oldest + window - now)))
                return False, retry_after

    def _clean_all_expired(self, now: float, cutoff: float) -> None:
        """
        Sweeps the complete storage map, pruning keys with zero active records.
        Must be invoked holding self.lock.
        """
        keys_to_delete = []
        for key, timestamps in self.history.items():
            active = [t for t in timestamps if t > cutoff]
            if not active:
                keys_to_delete.append(key)
            else:
                self.history[key] = active
        for key in keys_to_delete:
            del self.history[key]
        logger.info("Global rate limiter memory sweep completed. Removed %d inactive keys.", len(keys_to_delete))


# Global RateLimiter singleton instance
rate_limiter = RateLimiter()


import math
import uuid

class GroqTokenWindow:
    """
    Thread-safe in-process rolling token reservation window for Groq calls
    with post-call settlement and provider telemetry synchronization.
    """

    def __init__(self):
        self.reservations: Dict[str, Tuple[float, int]] = {}
        self.provider_remaining: int | None = None
        self.provider_limit: int | None = None
        self.provider_reset_ts: float | None = None
        self.lock = threading.Lock()

    def update_provider_telemetry(self, limit: int | None, remaining: int | None, reset_seconds: float | None) -> None:
        """Synchronizes window state with live response headers from Groq."""
        now = time.time()
        with self.lock:
            if limit is not None and limit > 0:
                self.provider_limit = limit
            if remaining is not None:
                self.provider_remaining = remaining
            if reset_seconds is not None and reset_seconds >= 0:
                self.provider_reset_ts = now + reset_seconds
                # If provider reset is near (e.g. within 2 seconds), prune expired local reservations
                if reset_seconds <= 2.0:
                    cutoff = now - 2.0
                    expired = [k for k, (ts, _) in self.reservations.items() if ts <= cutoff]
                    for k in expired:
                        del self.reservations[k]

    def reserve(self, tokens: int, limit: int, window: int = 60, session_id: str = "") -> Tuple[bool, int, str]:
        """
        Attempt to reserve estimated tokens within a sliding window.
        Returns:
            Tuple[bool, int, str]: (is_allowed, retry_after_seconds, reservation_id)
        """
        now = time.time()
        effective_limit = max(limit, self.provider_limit or limit)
        thread_id = threading.get_ident()

        with self.lock:
            # Check if provider reset timestamp has passed
            if self.provider_reset_ts and now >= self.provider_reset_ts:
                # Provider window has reset — clear older reservations
                self.reservations.clear()
                self.provider_reset_ts = None

            cutoff = now - window
            expired_keys = [k for k, (ts, _) in self.reservations.items() if ts <= cutoff]
            for k in expired_keys:
                del self.reservations[k]

            used_tokens = sum(t[1] for t in self.reservations.values())
            remaining_before = max(0, effective_limit - used_tokens)

            if used_tokens + tokens > effective_limit:
                # If provider reset timestamp is known, compute exact wait time
                if self.provider_reset_ts and self.provider_reset_ts > now:
                    retry_after = max(1, int(math.ceil(self.provider_reset_ts - now)))
                else:
                    # Calculate the exact timestamp when enough reservations will expire
                    sorted_reservations = sorted(self.reservations.values(), key=lambda x: x[0])
                    accumulated_freed = 0
                    needed_ts = now
                    for ts, tok in sorted_reservations:
                        accumulated_freed += tok
                        if (used_tokens - accumulated_freed) + tokens <= effective_limit:
                            needed_ts = ts
                            break

                    retry_after = max(1, min(window, int(math.ceil(needed_ts + window - now))))

                logger.warning(
                    "[TOKEN_WINDOW_RESERVE] thread=%s session=%s requested=%d used=%d remaining=%d limit=%d decision=REJECTED retry_after=%ds",
                    thread_id,
                    session_id or "N/A",
                    tokens,
                    used_tokens,
                    remaining_before,
                    effective_limit,
                    retry_after,
                )
                return False, retry_after, ""

            res_id = str(uuid.uuid4())
            self.reservations[res_id] = (now, tokens)
            used_after = used_tokens + tokens
            logger.info(
                "[TOKEN_WINDOW_RESERVE] thread=%s session=%s requested=%d used_before=%d used_after=%d remaining=%d limit=%d decision=GRANTED res_id=%s",
                thread_id,
                session_id or "N/A",
                tokens,
                used_tokens,
                used_after,
                effective_limit - used_after,
                effective_limit,
                res_id[:8],
            )
            return True, 0, res_id

    def settle(self, reservation_id: str, actual_tokens: int | None = None, session_id: str = "") -> None:
        """
        Updates an active reservation with actual token usage, or releases it on failure.
        """
        if not reservation_id:
            return

        thread_id = threading.get_ident()
        with self.lock:
            if reservation_id in self.reservations:
                ts, old_tokens = self.reservations[reservation_id]
                if actual_tokens is None or actual_tokens <= 0:
                    del self.reservations[reservation_id]
                    used_after = sum(t[1] for t in self.reservations.values())
                    logger.info(
                        "[TOKEN_WINDOW_SETTLE] thread=%s session=%s res_id=%s released_reserved=%d actual=0 used_after=%d",
                        thread_id,
                        session_id or "N/A",
                        reservation_id[:8],
                        old_tokens,
                        used_after,
                    )
                else:
                    self.reservations[reservation_id] = (ts, int(actual_tokens))
                    used_after = sum(t[1] for t in self.reservations.values())
                    diff = int(actual_tokens) - old_tokens
                    logger.info(
                        "[TOKEN_WINDOW_SETTLE] thread=%s session=%s res_id=%s reserved=%d actual=%d diff=%+d used_after=%d",
                        thread_id,
                        session_id or "N/A",
                        reservation_id[:8],
                        old_tokens,
                        int(actual_tokens),
                        diff,
                        used_after,
                    )

    def current_usage(self, window: int = 60) -> int:
        """Returns the current settled + reserved token sum in the sliding window."""
        now = time.time()
        with self.lock:
            if self.provider_reset_ts and now >= self.provider_reset_ts:
                self.reservations.clear()
                self.provider_reset_ts = None
                return 0
            cutoff = now - window
            return sum(tok for ts, tok in self.reservations.values() if ts > cutoff)

    def get_window_load(self, limit: int, window: int = 60) -> Tuple[int, int, float]:
        """
        Returns (used_tokens, effective_limit, usage_ratio) within the sliding window.
        """
        now = time.time()
        effective_limit = max(limit, self.provider_limit or limit)
        with self.lock:
            if self.provider_reset_ts and now >= self.provider_reset_ts:
                self.reservations.clear()
                self.provider_reset_ts = None
            cutoff = now - window
            used_tokens = sum(tok for ts, tok in self.reservations.values() if ts > cutoff)
            ratio = (used_tokens / effective_limit) if effective_limit > 0 else 0.0
            return used_tokens, effective_limit, ratio



groq_token_window = GroqTokenWindow()



class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    HTTP Middleware to intercept and rate-limit incoming API requests.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        method = request.method

        # Normalize path for checking (strip trailing slashes)
        norm_path = path.rstrip("/")

        # Define route patterns to protect
        # Expensive endpoints: embed, index, retrieve, chat
        expensive_patterns = ["/embed", "/index", "/retrieve", "/chat"]
        # General endpoints to protect: upload, process, chunk
        general_patterns = ["/upload", "/process", "/chunk"]

        is_expensive = False
        is_general = False

        if method == "POST":
            # Match expensive endpoints
            for pat in expensive_patterns:
                if norm_path.startswith(pat) or norm_path.endswith(pat):
                    is_expensive = True
                    break

            if not is_expensive:
                # Match general endpoints
                for pat in general_patterns:
                    if norm_path.startswith(pat) or norm_path.endswith(pat):
                        is_general = True
                        break

        # Only apply rate limiting if request matches one of our protected POST endpoints
        if is_expensive or is_general:
            # Extract client identity (IP Address)
            # Support X-Forwarded-For proxy headers safely
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
            else:
                client_ip = request.client.host if request.client else "127.0.0.1"

            if is_expensive:
                limit = settings.EXPENSIVE_REQUEST_LIMIT
                window = settings.EXPENSIVE_REQUEST_WINDOW_SECONDS
                key = f"expensive:{client_ip}"
            else:
                limit = settings.GENERAL_REQUEST_LIMIT
                window = settings.GENERAL_REQUEST_WINDOW_SECONDS
                key = f"general:{client_ip}"

            allowed, retry_after = rate_limiter.check_rate_limit(key, limit, window)
            if not allowed:
                logger.warning(
                    "Rate limit exceeded for client %s on endpoint '%s' %s. Retry after %d seconds.",
                    client_ip,
                    method,
                    path,
                    retry_after
                )
                # Form error payload using project error standards
                error_content = {
                    "success": False,
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests. Please try again later.",
                    "hint": f"Retry after {retry_after} seconds."
                }
                response = JSONResponse(
                    content=error_content,
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS
                )
                # Attach standard Retry-After header
                response.headers["Retry-After"] = str(retry_after)
                return response

        return await call_next(request)
