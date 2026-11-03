"""Request admission receipts for the pinned TP4/DP4 Prefill runtime."""
import asyncio
from collections import OrderedDict
import logging
import os
import time

logger = logging.getLogger(__name__)


def trace(stage, rid, room=None, **fields):
    if os.environ.get("MAAS_BOOTSTRAP_TRACE") == "1":
        logger.info("PREFILL_DELIVERY stage=%s rid=%s room=%s fields=%s",
                    stage, rid, room, fields)


class AcknowledgedDispatch:
    def __init__(self, timeout=None, attempts=None):
        self.timeout = float(timeout if timeout is not None else os.environ.get("MAAS_ADMISSION_TIMEOUT", "2"))
        self.attempts = int(attempts if attempts is not None else os.environ.get("MAAS_ADMISSION_ATTEMPTS", "30"))
        # Keep retransmissions well inside the scheduler's tombstone lifetime.
        if not (self.timeout > 0 and self.attempts > 0 and self.timeout * self.attempts <= 120):
            raise ValueError("Admission retry budget must be positive and at most 120 seconds")
        self.pending = {}
        self.retries = 0

    def acknowledge(self, key):
        event = self.pending.get(key)
        if event is not None:
            event.set()
            trace("scheduler_ack", *key)

    async def send(self, key, send_once):
        if key in self.pending:
            raise RuntimeError(f"Duplicate pending admission: {key}")
        event = self.pending[key] = asyncio.Event()
        try:
            for attempt in range(1, self.attempts + 1):
                try:
                    # Bound socket backpressure as well as the receipt wait.
                    async with asyncio.timeout(self.timeout):
                        await send_once()
                        trace("sent", *key, attempt=attempt)
                        await event.wait()
                    return
                except TimeoutError:
                    if event.is_set():
                        return
                    if attempt == self.attempts:
                        raise TimeoutError(f"Prefill scheduler admission timed out: {key}") from None
                    self.retries += 1
                    logger.warning("PREFILL_ADMISSION_RETRY key=%s attempt=%d", key, attempt + 1)
        finally:
            self.pending.pop(key, None)


class AdmissionRegistry:
    """Deduplicate on the receiving scheduler, including cancelled attempts.

    Entries outlive the bounded retransmission window. Capacity exhaustion fails
    closed instead of evicting a live deduplication entry. A process restart is
    not an exactly-once boundary; clients must create a fresh attempt ID.
    """
    def __init__(self, ttl=600.0, capacity=65536, clock=time.monotonic):
        self.ttl, self.capacity, self.clock = ttl, capacity, clock
        self.accepted = OrderedDict()
        self.aborted = OrderedDict()

    def _prune(self):
        cutoff = self.clock() - self.ttl
        for table in (self.accepted, self.aborted):
            while table and next(iter(table.values())) < cutoff:
                table.popitem(last=False)

    def abort(self, rid):
        self._prune()
        if rid and rid not in self.aborted:
            if len(self.aborted) >= self.capacity:
                raise RuntimeError("Prefill cancellation registry full")
            self.aborted[rid] = self.clock()

    def admit(self, key, callback):
        self._prune()
        if key in self.accepted or key[0] in self.aborted:
            trace("duplicate_or_cancelled", *key)
            return False
        if len(self.accepted) >= self.capacity:
            raise RuntimeError("Prefill admission registry full")
        callback()
        self.accepted[key] = self.clock()
        trace("scheduler_accepted", *key)
        return True
