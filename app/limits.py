"""Free-tier protection: strangers will use this, and the LLM quotas are small.

- At most 2 agent runs at once (extra runs wait up to 15 s for a slot, then get "busy").
- At most 6 live runs per IP per 10 minutes (cached presets don't count).
"""

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable

from starlette.requests import Request

MAX_CONCURRENT_RUNS = 2
RUNS_PER_IP = 6
RUN_WINDOW_SECONDS = 10 * 60


class RateLimiter:
    """Sliding window: allow at most max_events per key in the last window_seconds."""

    def __init__(self, max_events: int, window_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.max_events = max_events
        self.window_seconds = window_seconds
        self.clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = self.clock()
        self._prune(now)
        events = self._events[key]
        if len(events) >= self.max_events:
            return False
        events.append(now)
        return True

    def _prune(self, now: float) -> None:
        """Drop events outside the window, and forget IPs with none left (keeps memory bounded)."""
        cutoff = now - self.window_seconds
        for key in list(self._events):
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if not events:
                del self._events[key]

    def __len__(self) -> int:
        return len(self._events)


def client_ip(request: Request) -> str:
    """The visitor's IP: the right-most X-Forwarded-For entry, which Render's
    proxy appends itself (entries to its left can be faked by the client).
    Falls back to the socket address when there is no proxy (local runs).
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and forwarded.split(",")[-1].strip():
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


SLOT_WAIT_SECONDS = 15  # how long a run may wait for a free slot before we say "busy"
run_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RUNS)
run_limiter = RateLimiter(RUNS_PER_IP, RUN_WINDOW_SECONDS)
