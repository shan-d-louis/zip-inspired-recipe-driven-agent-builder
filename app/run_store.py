"""In-memory store of finished runs: runId -> RunRecord.

Entries expire after 30 minutes, and the oldest are evicted beyond 200.
Run IDs are random and unguessable, so one visitor can't read another's run.
"""

import secrets
import time
from collections import OrderedDict
from collections.abc import Callable

from app.models import RunRecord


class RunStore:
    def __init__(self, ttl_seconds: float = 30 * 60, max_entries: int = 200,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.clock = clock
        self._runs: OrderedDict[str, tuple[float, RunRecord]] = OrderedDict()  # oldest first

    def put(self, record: RunRecord) -> RunRecord:
        """Store a record under a new random runId and return it with the id set."""
        self._drop_expired()
        record = record.model_copy(update={"run_id": secrets.token_urlsafe(16)})
        self._runs[record.run_id] = (self.clock(), record)
        while len(self._runs) > self.max_entries:
            self._runs.popitem(last=False)
        return record

    def get(self, run_id: str) -> RunRecord | None:
        self._drop_expired()
        entry = self._runs.get(run_id)
        return entry[1] if entry else None

    def __len__(self) -> int:
        return len(self._runs)

    def _drop_expired(self) -> None:
        cutoff = self.clock() - self.ttl_seconds
        while self._runs and next(iter(self._runs.values()))[0] < cutoff:
            self._runs.popitem(last=False)
