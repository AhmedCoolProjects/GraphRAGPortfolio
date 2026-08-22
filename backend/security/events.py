"""
In-memory security event log + aggregate stats, for the public dashboard.

A bounded ring buffer (no PII stored — only categories, risk, and a short
redacted snippet). Survives within a process; resets on restart.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Deque, Dict, List

_MAX_EVENTS = 200
_events: Deque[dict] = deque(maxlen=_MAX_EVENTS)
_counts: Counter = Counter()
_lock = threading.Lock()


def record(kind: str, risk: str = "none", detail: str = "",
           signatures: List[dict] | None = None) -> None:
    """kind: injection_blocked | rate_limited | pii_redacted | leak_blocked |
    clean. Detail must already be redacted/safe."""
    with _lock:
        _counts[kind] += 1
        _counts["total"] += 1
        _events.appendleft({
            "ts": time.time(),
            "kind": kind,
            "risk": risk,
            "detail": detail[:120],
            "signatures": signatures or [],
        })


def stats() -> Dict[str, int]:
    with _lock:
        return dict(_counts)


def recent(limit: int = 50) -> List[dict]:
    with _lock:
        return list(_events)[:limit]
