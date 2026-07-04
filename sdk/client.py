from __future__ import annotations
import asyncio
import json
import queue
import threading
import time
import uuid
from typing import Any, Dict, List, Optional
import httpx
import structlog

logger = structlog.get_logger(__name__)


class ArgusClient:
    """
    Python SDK for Argus. Thread-safe, async-batch posting.
    Fails silently when Argus is unreachable — never blocks inference.
    """

    def __init__(
        self,
        endpoint: str,
        model_id: str,
        flush_interval_ms: int = 100,
        max_batch_size: int = 50,
        timeout_seconds: int = 5,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model_id = model_id
        self._flush_interval = flush_interval_ms / 1000
        self._max_batch = max_batch_size
        self._timeout = timeout_seconds
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> "ArgusClient":
        self._running = True
        self._thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="argus-flush",
        )
        self._thread.start()
        return self

    def stop(self, flush_timeout: float = 5.0):
        self._running = False
        self._flush_remaining()

    def log(
        self,
        features: Dict[str, Any],
        prediction: Any = None,
        label: Any = None,
        metadata: Dict[str, Any] = {},
    ):
        """
        Log a prediction. Non-blocking. Thread-safe.
        Call this immediately after model.predict().
        """
        record = {
            "model_id": self.model_id,
            "request_id": str(uuid.uuid4()),
            "features": features,
            "prediction": prediction,
            "label": label,
            "metadata": metadata,
        }
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass  # silently drop when queue is full

    def _flush_loop(self):
        while self._running:
            time.sleep(self._flush_interval)
            self._flush_remaining()

    def _flush_remaining(self):
        records = []
        try:
            while len(records) < self._max_batch:
                records.append(self._queue.get_nowait())
        except queue.Empty:
            pass

        if records:
            self._send_batch(records)

    def _send_batch(self, records: List[dict]):
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self.endpoint}/ingest/batch",
                    json={"records": records},
                )
                resp.raise_for_status()
        except Exception as exc:
            logger.debug("argus_sdk.send_failed", error=str(exc), count=len(records))


# Module-level singleton
_client: Optional[ArgusClient] = None


def init(
    endpoint: str,
    model_id: str,
    flush_interval_ms: int = 100,
) -> ArgusClient:
    global _client
    _client = ArgusClient(
        endpoint=endpoint,
        model_id=model_id,
        flush_interval_ms=flush_interval_ms,
    ).start()
    return _client


def log(
    features: Dict[str, Any],
    prediction: Any = None,
    label: Any = None,
    metadata: Dict[str, Any] = {},
):
    if _client is None:
        raise RuntimeError("Call argus.init() before argus.log()")
    _client.log(features=features, prediction=prediction, label=label, metadata=metadata)
