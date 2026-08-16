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
    """Thread-safe in-process rolling token reservation window for Groq calls with post-call settlement."""

    def __init__(self):
        self.reservations: Dict[str, Tuple[float, int]] = {}
        self.lock = threading.Lock()

    def reserve(self, tokens: int, limit: int, window: int = 60) -> Tuple[bool, int, str]:
        """
        Attempt to reserve estimated tokens within a sliding window.
        Returns:
            Tuple[bool, int, str]: (is_allowed, retry_after_seconds, reservation_id)
        """
        now = time.time()
        cutoff = now - window

        with self.lock:
            # Evict expired reservations
            expired_keys = [k for k, (ts, _) in self.reservations.items() if ts <= cutoff]
            for k in expired_keys:
                del self.reservations[k]

            used_tokens = sum(t[1] for t in self.reservations.values())
            if used_tokens + tokens > limit:
                if tokens > limit:
                    # Single request exceeds the entire window capacity
                    return False, window, ""

                # Calculate the exact timestamp when enough reservations will expire
                # such that (used_tokens - accumulated_freed) + tokens <= limit
                sorted_reservations = sorted(self.reservations.values(), key=lambda x: x[0])
                accumulated_freed = 0
                needed_ts = now
                for ts, tok in sorted_reservations:
                    accumulated_freed += tok
                    if (used_tokens - accumulated_freed) + tokens <= limit:
                        needed_ts = ts
                        break

                retry_after = max(1, int(math.ceil(needed_ts + window - now)))
                return False, retry_after, ""

            res_id = str(uuid.uuid4())
            self.reservations[res_id] = (now, tokens)
            return True, 0, res_id

    def settle(self, reservation_id: str, actual_tokens: int | None = None) -> None:
        """
        Updates an active reservation with actual token usage, or releases it on failure.
        """
        if not reservation_id:
            return

        with self.lock:
            if reservation_id in self.reservations:
                if actual_tokens is None or actual_tokens <= 0:
                    del self.reservations[reservation_id]
                else:
                    ts, _ = self.reservations[reservation_id]
                    self.reservations[reservation_id] = (ts, int(actual_tokens))

    def current_usage(self, window: int = 60) -> int:
        """Returns the current settled + reserved token sum in the sliding window."""
        now = time.time()
        cutoff = now - window
        with self.lock:
            return sum(tok for ts, tok in self.reservations.values() if ts > cutoff)



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
