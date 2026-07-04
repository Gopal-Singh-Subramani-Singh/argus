from __future__ import annotations
import asyncio
import json
from datetime import datetime
from typing import Optional
import redis.asyncio as aioredis
import structlog

from argus_core.metrics import STREAM_LAG
from config.settings import get_config

logger = structlog.get_logger(__name__)


class StreamConsumer:
    """
    Redis Streams consumer that reads ingested feature logs
    and writes them to TimescaleDB.
    """

    def __init__(self, redis_client: aioredis.Redis, db_pool):
        self._redis = redis_client
        self._db = db_pool
        self._cfg = get_config().redis
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        try:
            await self._ensure_consumer_group()
        except Exception as e:
            logger.warning("consumer.group_create_failed", error=str(e))
        self._running = True
        self._task = asyncio.create_task(
            self._consume_loop(), name="stream_consumer"
        )
        logger.info(
            "consumer.started",
            stream=self._cfg.stream_key,
            group=self._cfg.consumer_group,
        )

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _ensure_consumer_group(self):
        try:
            await self._redis.xgroup_create(
                self._cfg.stream_key,
                self._cfg.consumer_group,
                id="0",
                mkstream=True,
            )
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def _consume_loop(self):
        while self._running:
            try:
                messages = await self._redis.xreadgroup(
                    self._cfg.consumer_group,
                    self._cfg.consumer_name,
                    {self._cfg.stream_key: ">"},
                    count=self._cfg.batch_size,
                    block=self._cfg.block_ms,
                )
                if not messages:
                    continue

                for stream_name, stream_messages in messages:
                    for msg_id, fields in stream_messages:
                        await self._process(msg_id, fields)

                pending = await self._redis.xpending(
                    self._cfg.stream_key, self._cfg.consumer_group
                )
                STREAM_LAG.set(pending.get("pending", 0))

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("consumer.error", error=str(exc))
                await asyncio.sleep(1)

    async def _process(self, msg_id: str, fields: dict):
        try:
            model_id = fields.get("model_id", "")
            request_id = fields.get("request_id", "")
            features = json.loads(fields.get("features", "{}"))
            prediction = json.loads(fields.get("prediction", "null"))
            label = json.loads(fields.get("label", "null"))
            logged_at_raw = fields.get("logged_at", "")
            metadata = json.loads(fields.get("metadata", "{}"))

            try:
                logged_at = datetime.fromisoformat(logged_at_raw)
            except Exception:
                logged_at = datetime.utcnow()

            if self._db:
                async with self._db.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO feature_logs
                        (logged_at, model_id, request_id,
                         features, prediction, label, metadata)
                        VALUES ($1, $2, $3::uuid, $4::jsonb,
                                $5::jsonb, $6::jsonb, $7::jsonb)
                        ON CONFLICT DO NOTHING
                        """,
                        logged_at, model_id, request_id,
                        json.dumps(features),
                        json.dumps(prediction),
                        json.dumps(label),
                        json.dumps(metadata),
                    )

            await self._redis.xack(
                self._cfg.stream_key,
                self._cfg.consumer_group,
                msg_id,
            )

        except Exception as exc:
            logger.error(
                "consumer.process_error",
                msg_id=str(msg_id),
                error=str(exc),
            )
