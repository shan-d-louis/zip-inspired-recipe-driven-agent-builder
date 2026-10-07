"""Free-tier protection: strangers will use this, and the LLM quotas are small.

- At most 2 agent runs at once (extra runs wait their turn).
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
        events = self._events[key]
        while events and events[0] <= now - self.window_seconds:
            events.popleft()
        if len(events) >= self.max_events:
            return False
        events.append(now)
        return True


def client_ip(request: Request) -> str:
    """The visitor's IP. Behind Render's proxy it's the first X-Forwarded-For entry.

    A client can fake that header, so this limit is a speed bump, not security.
    The global concurrency cap still bounds the damage.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


run_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RUNS)
run_limiter = RateLimiter(RUNS_PER_IP, RUN_WINDOW_SECONDS)
