from __future__ import annotations
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from argus_core.drift.ks_test import run_ks_test, ks_severity
from argus_core.drift.psi import compute_psi, compute_psi_categorical, psi_severity
from argus_core.drift.chi_squared import run_chi_squared, chi_squared_severity
from argus_core.drift.js_divergence import compute_js_divergence, js_severity
from argus_core.drift.shap_drift import (
    estimate_shap_without_model, shap_severity, compute_shap_drift
)
from argus_core.models import DriftScore, DriftReport, DriftSeverity
from argus_core.metrics import (
    DRIFT_SCORE, DRIFT_SEVERITY, DRIFT_RUNS_TOTAL, DRIFT_DURATION,
    PRODUCTION_SAMPLES, SEVERITY_MAP
)
from config.settings import get_config

logger = structlog.get_logger(__name__)

SEVERITY_ORDER = {"ok": 0, "info": 1, "warning": 2, "critical": 3}


class DriftEngine:
    def __init__(self, db_pool, model_registry):
        self._db = db_pool
        self._registry = model_registry
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._cfg = get_config().drift

    async def start(self):
        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_job(
            self._run_all_models,
            "interval",
            seconds=self._cfg.schedule_interval_seconds,
            id="drift_scheduler",
            next_run_time=datetime.utcnow(),
        )
        self._scheduler.start()
        logger.info(
            "drift_engine.started",
            interval_seconds=self._cfg.schedule_interval_seconds,
        )

    async def stop(self):
        if self._scheduler:
            self._scheduler.shutdown(wait=False)

    async def _run_all_models(self):
        models = await self._registry.list_models()
        logger.info("drift_engine.run", model_count=len(models))
        for model in models:
            try:
                await self._run_model(model["model_id"])
            except Exception as exc:
                logger.error(
                    "drift_engine.model_error",
                    model_id=model["model_id"],
                    error=str(exc),
                )

    async def run_model_now(self, model_id: str) -> DriftReport:
        return await self._run_model(model_id)

    async def _run_model(self, model_id: str) -> DriftReport:
        t0 = time.monotonic()
        cfg = self._cfg

        end_ts = datetime.utcnow()
        start_ts = end_ts - timedelta(hours=cfg.production_window_hours)
        ref_end = end_ts - timedelta(hours=cfg.production_window_hours)
        ref_start = ref_end - timedelta(days=cfg.reference_window_days)

        prod_rows = await self._fetch_features(model_id, start_ts, end_ts)
        ref_rows = await self._fetch_reference(model_id)

        PRODUCTION_SAMPLES.labels(model_id=model_id).set(len(prod_rows))

        if len(prod_rows) < cfg.min_samples:
            logger.info(
                "drift_engine.insufficient_samples",
                model_id=model_id,
                n=len(prod_rows),
                required=cfg.min_samples,
            )
            DRIFT_RUNS_TOTAL.labels(model_id=model_id, status="skipped").inc()
            return DriftReport(
                model_id=model_id,
                window_start=start_ts,
                window_end=end_ts,
                total_samples=len(prod_rows),
                scores=[],
                overall_severity=DriftSeverity.OK,
            )

        schema = await self._registry.get_feature_schema(model_id)
        scores: List[DriftScore] = []

        for feature in schema:
            fname = feature["name"]
            ftype = feature["type"]

            prod_vals = self._extract_feature(prod_rows, fname)
            ref_vals = self._extract_feature(ref_rows.get(fname, []), fname)

            if not prod_vals or not ref_vals:
                continue

            feature_scores = await self._compute_feature_drift(
                fname, ftype, ref_vals, prod_vals
            )
            scores.extend(feature_scores)

        if "shap_drift" in cfg.methods and len(schema) > 1:
            shap_score = await self._compute_shap_drift(
                model_id, schema, prod_rows, ref_rows
            )
            if shap_score:
                scores.append(shap_score)

        await self._persist_scores(model_id, scores)
        self._update_prometheus(model_id, scores)

        overall = self._overall_severity(scores)
        elapsed = time.monotonic() - t0
        DRIFT_DURATION.labels(model_id=model_id).observe(elapsed)
        DRIFT_RUNS_TOTAL.labels(model_id=model_id, status="ok").inc()

        logger.info(
            "drift_engine.completed",
            model_id=model_id,
            features=len(schema),
            scores=len(scores),
            severity=overall,
            duration_s=round(elapsed, 2),
        )

        return DriftReport(
            model_id=model_id,
            window_start=start_ts,
            window_end=end_ts,
            total_samples=len(prod_rows),
            scores=scores,
            overall_severity=DriftSeverity(overall),
        )

    async def _compute_feature_drift(
        self, fname: str, ftype: str, ref_vals: list, prod_vals: list
    ) -> List[DriftScore]:
        cfg = self._cfg
        results = []

        if ftype == "numeric":
            ref_arr = np.array([v for v in ref_vals if v is not None], dtype=float)
            prod_arr = np.array([v for v in prod_vals if v is not None], dtype=float)

            if "ks_test" in cfg.methods:
                stat, pval = run_ks_test(ref_arr, prod_arr, fname)
                sev = ks_severity(pval)
                results.append(DriftScore(
                    feature_name=fname, method="ks_test",
                    score=stat, p_value=pval,
                    severity=DriftSeverity(sev),
                    sample_count=len(prod_arr),
                ))

            if "psi" in cfg.methods:
                psi = compute_psi(ref_arr, prod_arr, feature_name=fname)
                sev = psi_severity(psi)
                results.append(DriftScore(
                    feature_name=fname, method="psi",
                    score=psi, severity=DriftSeverity(sev),
                    sample_count=len(prod_arr),
                ))

            if "js_divergence" in cfg.methods:
                jsd = compute_js_divergence(ref_arr, prod_arr, feature_name=fname)
                sev = js_severity(jsd)
                results.append(DriftScore(
                    feature_name=fname, method="js_divergence",
                    score=jsd, severity=DriftSeverity(sev),
                    sample_count=len(prod_arr),
                ))

        elif ftype == "categorical":
            ref_list = [str(v) for v in ref_vals if v is not None]
            prod_list = [str(v) for v in prod_vals if v is not None]

            if "chi_squared" in cfg.methods:
                stat, pval = run_chi_squared(ref_list, prod_list, fname)
                sev = chi_squared_severity(pval)
                results.append(DriftScore(
                    feature_name=fname, method="chi_squared",
                    score=stat, p_value=pval,
                    severity=DriftSeverity(sev),
                    sample_count=len(prod_list),
                ))

            if "psi" in cfg.methods:
                psi = compute_psi_categorical(ref_list, prod_list, fname)
                sev = psi_severity(psi)
                results.append(DriftScore(
                    feature_name=fname, method="psi",
                    score=psi, severity=DriftSeverity(sev),
                    sample_count=len(prod_list),
                ))

        return results

    async def _compute_shap_drift(
        self, model_id, schema, prod_rows, ref_rows
    ) -> Optional[DriftScore]:
        try:
            feature_names = [f["name"] for f in schema if f["type"] == "numeric"]
            if len(feature_names) < 2:
                return None

            def extract_matrix(rows, names):
                result = []
                for row in rows:
                    vals = []
                    feats = row if isinstance(row, dict) else {}
                    for n in names:
                        v = feats.get(n)
                        vals.append(float(v) if v is not None else 0.0)
                    result.append(vals)
                return np.array(result) if result else np.zeros((1, len(names)))

            prod_matrix = extract_matrix(prod_rows, feature_names)
            ref_matrix_data = [
                {k: v[i] if i < len(v) else 0.0
                 for k, v in ref_rows.items()
                 if k in feature_names}
                for i in range(min(len(prod_rows), 500))
            ]
            ref_matrix = extract_matrix(ref_matrix_data, feature_names)

            rank_corr, _ = estimate_shap_without_model(
                ref_matrix, prod_matrix, feature_names
            )
            sev = shap_severity(rank_corr)

            return DriftScore(
                feature_name="_model",
                method="shap_drift",
                score=rank_corr,
                severity=DriftSeverity(sev),
                sample_count=len(prod_rows),
            )
        except Exception as exc:
            logger.warning("shap_drift.failed", error=str(exc))
            return None

    async def _fetch_features(
        self, model_id: str, start: datetime, end: datetime
    ) -> List[dict]:
        if self._db is None:
            return []
        async with self._db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT features FROM feature_logs
                WHERE model_id = $1
                  AND logged_at >= $2
                  AND logged_at < $3
                ORDER BY logged_at DESC
                LIMIT 10000
                """,
                model_id, start, end,
            )
        result = []
        for r in rows:
            val = r["features"]
            if isinstance(val, str):
                import json
                val = json.loads(val)
            result.append(dict(val))
        return result

    async def _fetch_reference(self, model_id: str) -> Dict[str, list]:
        if self._db is None:
            # Fall back to in-memory registry reference
            result = {}
            for key, val in self._registry._reference.items():
                if key.startswith(f"{model_id}::"):
                    feature_name = key.split("::", 1)[1]
                    result[feature_name] = val.get("samples", [])
            return result
        async with self._db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT feature_name, stats
                FROM reference_distributions
                WHERE model_id = $1
                """,
                model_id,
            )
        result = {}
        for row in rows:
            stats_val = row["stats"]
            if isinstance(stats_val, str):
                import json
                stats_val = json.loads(stats_val)
            stats_data = dict(stats_val)
            result[row["feature_name"]] = stats_data.get("samples", [])
        return result

    def _extract_feature(self, rows, feature_name: str) -> list:
        if isinstance(rows, list):
            if rows and isinstance(rows[0], dict):
                return [r.get(feature_name) for r in rows if r.get(feature_name) is not None]
            return rows
        return []

    async def _persist_scores(self, model_id: str, scores: List[DriftScore]):
        if self._db is None or not scores:
            return
        now = datetime.utcnow()
        async with self._db.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO drift_scores
                (scored_at, model_id, feature_name, method, score, p_value,
                 severity, sample_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                [
                    (now, model_id, s.feature_name, s.method, s.score,
                     s.p_value, s.severity.value, s.sample_count)
                    for s in scores
                ],
            )

    def _update_prometheus(self, model_id: str, scores: List[DriftScore]):
        worst_per_feature: Dict[str, int] = {}
        for s in scores:
            DRIFT_SCORE.labels(
                model_id=model_id,
                feature_name=s.feature_name,
                method=s.method,
            ).set(s.score)
            sev_int = SEVERITY_MAP.get(s.severity.value, 0)
            current = worst_per_feature.get(s.feature_name, 0)
            worst_per_feature[s.feature_name] = max(current, sev_int)
        for fname, sev_int in worst_per_feature.items():
            DRIFT_SEVERITY.labels(
                model_id=model_id, feature_name=fname
            ).set(sev_int)

    def _overall_severity(self, scores: List[DriftScore]) -> str:
        if not scores:
            return "ok"
        return max(
            (s.severity.value for s in scores),
            key=lambda v: SEVERITY_ORDER.get(v, 0),
        )
