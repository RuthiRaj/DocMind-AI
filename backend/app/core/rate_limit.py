import time
import logging
import threading
from typing import Dict, List, Tuple
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
                retry_after = int(oldest + window - now)
                return False, max(1, retry_after)

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
