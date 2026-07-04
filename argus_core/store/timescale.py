from __future__ import annotations
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import asyncpg
import structlog

logger = structlog.get_logger(__name__)


class TimescaleClient:
    """
    Async TimescaleDB client wrapping asyncpg.
    Handles all time-series queries for drift scores and feature logs.
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def insert_drift_scores(
        self,
        model_id: str,
        scores: List[dict],
        scored_at: Optional[datetime] = None,
    ) -> None:
        if not scores:
            return
        ts = scored_at or datetime.utcnow()
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO drift_scores
                (scored_at, model_id, feature_name, method,
                 score, p_value, severity, sample_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                [
                    (
                        ts, model_id,
                        s.get("feature_name"), s.get("method"),
                        s.get("score"), s.get("p_value"),
                        s.get("severity"), s.get("sample_count"),
                    )
                    for s in scores
                ],
            )

    async def get_latest_drift_scores(
        self, model_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT feature_name, method, score, p_value,
                       severity, sample_count, scored_at
                FROM drift_scores
                WHERE model_id = $1
                ORDER BY scored_at DESC
                LIMIT $2
                """,
                model_id, limit,
            )
        return [dict(r) for r in rows]

    async def get_drift_history(
        self,
        model_id: str,
        feature_name: str,
        method: str,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        since = datetime.utcnow() - timedelta(days=days)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT scored_at, score, p_value, severity, sample_count
                FROM drift_scores
                WHERE model_id = $1
                  AND feature_name = $2
                  AND method = $3
                  AND scored_at >= $4
                ORDER BY scored_at ASC
                """,
                model_id, feature_name, method, since,
            )
        return [dict(r) for r in rows]

    async def insert_alert(
        self,
        model_id: str,
        rule_name: str,
        feature_name: Optional[str],
        method: str,
        score: float,
        threshold: float,
        severity: str,
        webhook_fired: bool = False,
        retrain_fired: bool = False,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO alert_history
                (model_id, rule_name, feature_name, method,
                 score, threshold, severity, webhook_fired, retrain_fired)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                model_id, rule_name, feature_name, method,
                score, threshold, severity, webhook_fired, retrain_fired,
            )

    async def get_alert_history(
        self, model_id: str, days: int = 7
    ) -> List[Dict[str, Any]]:
        since = datetime.utcnow() - timedelta(days=days)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT alerted_at, rule_name, feature_name,
                       method, score, threshold, severity,
                       webhook_fired, retrain_fired
                FROM alert_history
                WHERE model_id = $1 AND alerted_at >= $2
                ORDER BY alerted_at DESC
                """,
                model_id, since,
            )
        return [dict(r) for r in rows]

    async def get_feature_log_count(
        self, model_id: str, hours: int = 24
    ) -> int:
        since = datetime.utcnow() - timedelta(hours=hours)
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT COUNT(*) FROM feature_logs
                WHERE model_id = $1 AND logged_at >= $2
                """,
                model_id, since,
            )
        return int(result or 0)

    async def health_check(self) -> bool:
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False
