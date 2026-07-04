from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import duckdb
import structlog

from config.settings import get_config

logger = structlog.get_logger(__name__)


class DuckDBAnalytics:
    """
    DuckDB-powered 30-day drift history analytics.
    Operates on in-memory or file-backed DuckDB.
    Supports fast aggregations without touching TimescaleDB.
    """

    def __init__(self):
        cfg = get_config().duckdb
        self._db_path = cfg.db_path
        self._history_days = cfg.history_days
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self) -> None:
        self._conn = duckdb.connect(self._db_path)
        self._create_tables()
        logger.info("duckdb.connected", path=self._db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_history (
                scored_at    TIMESTAMP NOT NULL,
                model_id     VARCHAR NOT NULL,
                feature_name VARCHAR NOT NULL,
                method       VARCHAR NOT NULL,
                score        DOUBLE NOT NULL,
                p_value      DOUBLE,
                severity     VARCHAR,
                sample_count INTEGER
            )
            """
        )

    def ingest_scores(self, scores: List[Dict[str, Any]]) -> None:
        """Bulk-insert drift scores into DuckDB for analytics."""
        if not self._conn or not scores:
            return
        rows = [
            (
                s.get("scored_at", datetime.utcnow()),
                s.get("model_id", ""),
                s.get("feature_name", ""),
                s.get("method", ""),
                float(s.get("score", 0.0)),
                s.get("p_value"),
                s.get("severity"),
                s.get("sample_count"),
            )
            for s in scores
        ]
        self._conn.executemany(
            """
            INSERT INTO drift_history
            (scored_at, model_id, feature_name, method,
             score, p_value, severity, sample_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def query_history(
        self,
        model_id: str,
        feature_name: str,
        method: str,
        days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return time-series drift scores for a given feature/method."""
        if not self._conn:
            return []
        days = days or self._history_days
        since = datetime.utcnow() - timedelta(days=days)
        result = self._conn.execute(
            """
            SELECT scored_at, score, p_value, severity, sample_count
            FROM drift_history
            WHERE model_id = ?
              AND feature_name = ?
              AND method = ?
              AND scored_at >= ?
            ORDER BY scored_at ASC
            """,
            [model_id, feature_name, method, since],
        ).fetchall()
        cols = ["scored_at", "score", "p_value", "severity", "sample_count"]
        return [dict(zip(cols, row)) for row in result]

    def query_aggregated(
        self,
        model_id: str,
        days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return aggregated (avg, max, min) drift scores per feature/method."""
        if not self._conn:
            return []
        days = days or self._history_days
        since = datetime.utcnow() - timedelta(days=days)
        result = self._conn.execute(
            """
            SELECT
                feature_name,
                method,
                AVG(score)   AS avg_score,
                MAX(score)   AS max_score,
                MIN(score)   AS min_score,
                COUNT(*)     AS n_observations,
                MAX(scored_at) AS last_scored
            FROM drift_history
            WHERE model_id = ? AND scored_at >= ?
            GROUP BY feature_name, method
            ORDER BY max_score DESC
            """,
            [model_id, since],
        ).fetchall()
        cols = [
            "feature_name", "method",
            "avg_score", "max_score", "min_score",
            "n_observations", "last_scored",
        ]
        return [dict(zip(cols, row)) for row in result]

    def count_severity_events(
        self, model_id: str, days: Optional[int] = None
    ) -> Dict[str, int]:
        """Count severity events in the time window."""
        if not self._conn:
            return {}
        days = days or self._history_days
        since = datetime.utcnow() - timedelta(days=days)
        result = self._conn.execute(
            """
            SELECT severity, COUNT(*) as cnt
            FROM drift_history
            WHERE model_id = ? AND scored_at >= ?
            GROUP BY severity
            """,
            [model_id, since],
        ).fetchall()
        return {row[0]: row[1] for row in result if row[0]}

    def purge_old_data(self, days: Optional[int] = None) -> int:
        """Remove records older than `days` days. Returns deleted count."""
        if not self._conn:
            return 0
        days = days or self._history_days
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = self._conn.execute(
            "DELETE FROM drift_history WHERE scored_at < ? RETURNING *",
            [cutoff],
        ).fetchall()
        deleted = len(result)
        if deleted:
            logger.info("duckdb.purged", rows=deleted)
        return deleted
