from __future__ import annotations
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_root_endpoint():
    from argus_core.main import app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "Argus"


@pytest.mark.asyncio
async def test_health_endpoint():
    from argus_core.main import app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "timescaledb" in data
    assert "uptime_seconds" in data


@pytest.mark.asyncio
async def test_metrics_endpoint():
    from argus_core.main import app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert b"argus_" in resp.content


@pytest.mark.asyncio
async def test_register_model():
    from argus_core.main import app, app_state
    from argus_core.registry.model_registry import ModelRegistry
    from unittest.mock import AsyncMock

    registry = ModelRegistry(db_pool=None)
    app_state.registry = registry

    payload = {
        "model_id": "test_model_001",
        "name": "Test Model",
        "version": "1.0",
        "features": [
            {"name": "age", "type": "numeric"},
            {"name": "income", "type": "numeric"},
            {"name": "category", "type": "categorical"},
        ],
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/models", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "test_model_001"


@pytest.mark.asyncio
async def test_ingest_unregistered_model_rejected():
    from argus_core.main import app, app_state
    from argus_core.registry.model_registry import ModelRegistry
    from unittest.mock import AsyncMock, MagicMock
    from argus_core.ingest import IngestService

    registry = ModelRegistry(db_pool=None)
    app_state.registry = registry

    # Use a mock redis that records calls but doesn't need a real connection
    mock_redis = MagicMock()
    mock_redis.xadd = AsyncMock(return_value="1-0")
    app_state.ingest = IngestService(mock_redis, registry)

    payload = {
        "model_id": "nonexistent_model",
        "features": {"age": 30},
        "prediction": 1,
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/ingest", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ingest_registered_model_accepted():
    from argus_core.main import app, app_state
    from unittest.mock import AsyncMock, MagicMock
    from argus_core.models import IngestResponse

    app_state.registry = MagicMock()
    app_state.registry.model_exists.return_value = True
    app_state.ingest = MagicMock()
    app_state.ingest.ingest = AsyncMock(
        return_value=IngestResponse(request_id="abc-123", queued=True)
    )

    payload = {
        "model_id": "fraud_v3",
        "features": {"age": 35, "income": 55000.0, "category": "B"},
        "prediction": 0,
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/ingest", json=payload)
    assert resp.status_code == 200
    assert resp.json()["queued"] is True
