from __future__ import annotations
import json
import time
from datetime import datetime
from typing import Optional
import redis.asyncio as aioredis
import structlog

from argus_core.models import IngestRequest, IngestResponse
from argus_core.metrics import INGEST_TOTAL, INGEST_BATCH_SIZE
from config.settings import get_config

logger = structlog.get_logger(__name__)


class IngestService:
    def __init__(self, redis_client: aioredis.Redis, registry):
        self._redis = redis_client
        self._registry = registry
        self._cfg = get_config().redis
    async def ingest(self, request: IngestRequest) -> IngestResponse:
        if not self._registry.model_exists(request.model_id):
            INGEST_TOTAL.labels(
                model_id=request.model_id, status="rejected"
            ).inc()
            raise ValueError(
                f"Model '{request.model_id}' not registered. "
                "Call POST /models to register first."
            )

        msg = {
            "model_id": request.model_id,
            "request_id": request.request_id,
            "features": json.dumps(request.features),
            "prediction": json.dumps(request.prediction),
            "label": json.dumps(request.label),
            "logged_at": (
                request.timestamp or datetime.utcnow()
            ).isoformat(),
            "metadata": json.dumps(request.metadata),
        }

        await self._redis.xadd(
            self._cfg.stream_key,
            msg,
            maxlen=100_000,
            approximate=True,
        )

        INGEST_TOTAL.labels(
            model_id=request.model_id, status="accepted"
        ).inc()
        logger.debug(
            "ingest.accepted",
            model_id=request.model_id,
            request_id=request.request_id,
        )
        return IngestResponse(request_id=request.request_id, queued=True)

    async def ingest_batch(
        self, requests: list
    ) -> dict:
        accepted = 0
        rejected = 0
        for req in requests:
            try:
                await self.ingest(req)
                accepted += 1
            except Exception:
                rejected += 1
        INGEST_BATCH_SIZE.observe(len(requests))
        return {"accepted": accepted, "rejected": rejected, "total": len(requests)}
