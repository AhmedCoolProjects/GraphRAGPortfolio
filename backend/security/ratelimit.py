"""
In-memory token-bucket rate limiter, keyed by client.

Note: on serverless (Vercel) this is per-instance and resets on cold start — it
is a demonstration of the pattern, not a distributed limiter. For production
scale you'd back it with Redis; the interface here stays the same.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict


@dataclass
class _Bucket:
    tokens: float
    last: float


class RateLimiter:
    def __init__(self, rate_per_min: int = 20, burst: int = 8):
        self.rate = rate_per_min / 60.0      # tokens per second
        self.burst = float(burst)
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        """Consume one token. Returns True if allowed, False if rate-limited."""
        now = time.monotonic()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                self._buckets[key] = _Bucket(tokens=self.burst - 1, last=now)
                return True
            # refill
            b.tokens = min(self.burst, b.tokens + (now - b.last) * self.rate)
            b.last = now
            if b.tokens >= 1:
                b.tokens -= 1
                return True
            return False


# Shared limiter for the chat endpoints.
chat_limiter = RateLimiter(rate_per_min=20, burst=8)
