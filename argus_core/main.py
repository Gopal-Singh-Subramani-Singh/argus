from __future__ import annotations
import time
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta

import asyncpg
import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from config.settings import get_config
from argus_core.models import (
    ModelRegistrationRequest, ModelRegistrationResponse,
    IngestRequest, BatchIngestRequest, IngestResponse,
    ReferenceSetRequest, HealthResponse, DriftReport,
    ModelStatusResponse, DriftSeverity,
)
from argus_core.registry.model_registry import ModelRegistry
from argus_core.ingest import IngestService
from argus_core.consumer import StreamConsumer
from argus_core.drift.engine import DriftEngine
from argus_core.alerts.engine import AlertEngine
from argus_core.metrics import update_uptime

logger = structlog.get_logger(__name__)


@dataclass
class AppState:
    db_pool: Optional[asyncpg.Pool] = None
    redis: Optional[aioredis.Redis] = None
    registry: Optional[ModelRegistry] = None
    ingest: Optional[IngestService] = None
    consumer: Optional[StreamConsumer] = None
    drift_engine: Optional[DriftEngine] = None
    alert_engine: Optional[AlertEngine] = None
    start_time: float = field(default_factory=time.time)
    db_ok: bool = False
    redis_ok: bool = False


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url = cfg.redis.full_url()
    app_state.redis = aioredis.from_url(redis_url, decode_responses=True)
    try:
        await app_state.redis.ping()
        app_state.redis_ok = True
        logger.info("redis.connected")
    except Exception as e:
        logger.error("redis.failed", error=str(e))

    # ── TimescaleDB ──────────────────────────────────────────────────────────
    try:
        app_state.db_pool = await asyncpg.create_pool(
            dsn=cfg.timescaledb.asyncpg_dsn(),
            min_size=cfg.timescaledb.min_pool,
            max_size=cfg.timescaledb.max_pool,
            command_timeout=30,
        )
        app_state.db_ok = True
        logger.info("timescaledb.connected")
    except Exception as e:
        logger.warning("timescaledb.unavailable", error=str(e))
        app_state.db_pool = None

    # ── Services ─────────────────────────────────────────────────────────────
    app_state.registry    = ModelRegistry(app_state.db_pool)
    app_state.ingest      = IngestService(app_state.redis, app_state.registry)
    app_state.consumer    = StreamConsumer(app_state.redis, app_state.db_pool)
    app_state.drift_engine = DriftEngine(app_state.db_pool, app_state.registry)
    app_state.alert_engine = AlertEngine()

    # ── Reload persisted models from DB on startup ───────────────────────────
    if app_state.db_pool:
        await _reload_registry_from_db(app_state.registry, app_state.db_pool)

    await app_state.consumer.start()
    await app_state.drift_engine.start()

    logger.info("argus.started", api_key_enabled=bool(cfg.security.api_key))
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    await app_state.drift_engine.stop()
    await app_state.consumer.stop()
    if app_state.db_pool:
        await app_state.db_pool.close()
    if app_state.redis:
        await app_state.redis.aclose()
    logger.info("argus.shutdown")


async def _reload_registry_from_db(registry: ModelRegistry, pool: asyncpg.Pool):
    """Reload all registered models and their schemas from TimescaleDB on startup."""
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT model_id, name, version, feature_schema FROM models")
        for row in rows:
            model_id = row["model_id"]
            schema   = row["feature_schema"]
            if isinstance(schema, str):
                schema = json.loads(schema)
            features = list(schema) if schema else []
            registry._models[model_id] = {
                "model_id":      model_id,
                "name":          row["name"],
                "version":       row["version"],
                "features":      features,
                "registered_at": datetime.utcnow().isoformat(),
            }
            registry._schemas[model_id] = features
        logger.info("registry.reloaded", models=len(rows))
    except Exception as e:
        logger.warning("registry.reload_failed", error=str(e))


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Argus — ML Observability Platform",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Key Authentication Middleware ─────────────────────────────────────────

@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    api_key = None

    # Skip auth if no API key configured (dev mode)
    if not api_key:
        return await call_next(request)

    # Skip auth for public endpoints
    if request.url.path in cfg.security.public_paths:
        return await call_next(request)

    # Check header
    provided = request.headers.get("X-API-Key")
    if not provided or provided != api_key:
        logger.warning(
            "auth.rejected",
            path=request.url.path,
            ip=request.client.host if request.client else "unknown",
        )
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key. Include header: X-API-Key: <key>"},
        )

    return await call_next(request)


# ── Request logging middleware ────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.monotonic()
    response = await call_next(request)
    ms = round((time.monotonic() - t0) * 1000, 1)
    logger.info(
        "http",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        ms=ms,
    )
    return response


# ── Model Registry ────────────────────────────────────────────────────────────

@app.post("/models", response_model=ModelRegistrationResponse)
async def register_model(req: ModelRegistrationRequest):
    entry = await app_state.registry.register(
        model_id=req.model_id,
        name=req.name,
        version=req.version,
        features=[f.model_dump() for f in req.features],
        metadata=req.metadata,
    )
    return ModelRegistrationResponse(
        model_id=entry["model_id"],
        name=entry["name"],
        version=entry["version"],
        registered_at=datetime.fromisoformat(entry["registered_at"]),
    )


@app.post("/models/{model_id}/reference")
async def set_reference(model_id: str, req: ReferenceSetRequest):
    if not app_state.registry.model_exists(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    schema = await app_state.registry.get_feature_schema(model_id)
    for feature_def in schema:
        fname   = feature_def["name"]
        ftype   = feature_def["type"]
        samples = [row.get(fname) for row in req.features_data if row.get(fname) is not None]
        if samples:
            await app_state.registry.set_reference(model_id, fname, ftype, samples)
    return {"model_id": model_id, "status": "reference_set", "features": len(schema)}


@app.post("/models/{model_id}/reference/refresh")
async def refresh_reference(model_id: str):
    """Re-compute the reference distribution from the last 30 days of production data."""
    if not app_state.registry.model_exists(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    if not app_state.db_pool:
        raise HTTPException(status_code=503, detail="TimescaleDB not connected")

    cfg    = get_config()
    since  = datetime.utcnow() - timedelta(days=cfg.drift.reference_window_days)
    schema = await app_state.registry.get_feature_schema(model_id)

    refreshed = 0
    async with app_state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT features FROM feature_logs
            WHERE model_id = $1 AND logged_at >= $2
            ORDER BY logged_at DESC LIMIT 5000
            """,
            model_id, since,
        )

    if len(rows) < 50:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(rows)} samples in last {cfg.drift.reference_window_days} days. Need at least 50.",
        )

    parsed = []
    for r in rows:
        val = r["features"]
        if isinstance(val, str):
            val = json.loads(val)
        parsed.append(dict(val))

    for feature_def in schema:
        fname   = feature_def["name"]
        ftype   = feature_def["type"]
        samples = [row.get(fname) for row in parsed if row.get(fname) is not None]
        if samples:
            await app_state.registry.set_reference(model_id, fname, ftype, samples)
            refreshed += 1

    logger.info("reference.refreshed", model_id=model_id, features=refreshed, samples=len(rows))
    return {
        "model_id":         model_id,
        "status":           "reference_refreshed",
        "features_updated": refreshed,
        "samples_used":     len(rows),
    }


@app.get("/models")
async def list_models():
    return await app_state.registry.list_models()


@app.get("/models/{model_id}")
async def get_model(model_id: str):
    model = await app_state.registry.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    if not app_state.registry.model_exists(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    await app_state.registry.delete_model(model_id)
    return {"model_id": model_id, "status": "deleted"}


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    try:
        return await app_state.ingest.ingest(req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/ingest/batch")
async def ingest_batch(req: BatchIngestRequest):
    return await app_state.ingest.ingest_batch(req.records)


# ── Drift ─────────────────────────────────────────────────────────────────────

@app.post("/drift/{model_id}/run", response_model=DriftReport)
async def run_drift(model_id: str):
    if not app_state.registry.model_exists(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    report = await app_state.drift_engine.run_model_now(model_id)
    alerts = await app_state.alert_engine.evaluate(model_id, report.scores)
    logger.info("drift.run", model_id=model_id, scores=len(report.scores), alerts=len(alerts))
    return report


@app.get("/drift/{model_id}/latest")
async def latest_drift(model_id: str):
    if not app_state.db_pool:
        return {"error": "TimescaleDB not connected"}
    async with app_state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT feature_name, method, score, p_value,
                   severity, sample_count, scored_at
            FROM drift_scores
            WHERE model_id=$1
            ORDER BY scored_at DESC
            LIMIT 100
            """,
            model_id,
        )
    return [dict(r) for r in rows]


@app.get("/drift/{model_id}/history")
async def drift_history(
    model_id: str,
    feature: str,
    method: str = "psi",
    days: int = 30,
):
    """30-day time-series of drift scores for a specific feature and method."""
    if not app_state.db_pool:
        return {"error": "TimescaleDB not connected"}
    since = datetime.utcnow() - timedelta(days=days)
    async with app_state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT scored_at, score, p_value, severity, sample_count
            FROM drift_scores
            WHERE model_id=$1 AND feature_name=$2 AND method=$3
              AND scored_at >= $4
            ORDER BY scored_at ASC
            """,
            model_id, feature, method, since,
        )
    return {
        "model_id":     model_id,
        "feature_name": feature,
        "method":       method,
        "days":         days,
        "history":      [dict(r) for r in rows],
    }


@app.get("/alerts/{model_id}/history")
async def alert_history(model_id: str, days: int = 7):
    """Recent alert history for a model."""
    if not app_state.db_pool:
        return {"error": "TimescaleDB not connected"}
    since = datetime.utcnow() - timedelta(days=days)
    async with app_state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT alerted_at, rule_name, feature_name, method,
                   score, threshold, severity, webhook_fired, retrain_fired
            FROM alert_history
            WHERE model_id=$1 AND alerted_at >= $2
            ORDER BY alerted_at DESC
            LIMIT 200
            """,
            model_id, since,
        )
    return [dict(r) for r in rows]


# ── Observability ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    uptime = update_uptime()

    # Live-check Redis
    redis_ok = False
    if app_state.redis:
        try:
            await app_state.redis.ping()
            redis_ok = True
        except Exception:
            pass

    # Live-check TimescaleDB
    db_ok = False
    if app_state.db_pool:
        try:
            async with app_state.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_ok = True
        except Exception:
            pass

    return HealthResponse(
        status="ok" if (redis_ok and db_ok) else "degraded",
        timescaledb="ok" if db_ok else "unavailable",
        redis="ok" if redis_ok else "unavailable",
        uptime_seconds=round(uptime, 1),
    )


@app.get("/metrics")
async def metrics():
    update_uptime()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    cfg = get_config()
    return {
        "service":         "Argus",
        "version":         "0.1.0",
        "docs":            "/docs",
        "metrics":         "/metrics",
        "auth_required":   bool(cfg.security.api_key),
    }


# ── Internal webhook receivers ────────────────────────────────────────────────

@app.post("/internal/alert-received")
async def alert_received(request: Request):
    body = await request.json()
    logger.warning("ALERT RECEIVED", **{k: str(v) for k, v in body.items()})
    return {"status": "logged"}


@app.post("/internal/retrain-triggered")
async def retrain_triggered(request: Request):
    body = await request.json()
    logger.warning("RETRAINING TRIGGERED", **{k: str(v) for k, v in body.items()})
    # In production: replace this with your actual retraining pipeline call
    # e.g. trigger GitHub Actions, Airflow DAG, SageMaker Pipeline, etc.
    return {"status": "logged", "action": "wire_up_your_retraining_pipeline_here"}
