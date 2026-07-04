from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from argus_core.ingest import IngestService
from argus_core.models import IngestRequest, IngestResponse


@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.xadd = AsyncMock(return_value="1-0")
    return redis


@pytest.fixture
def mock_registry_registered():
    registry = MagicMock()
    registry.model_exists.return_value = True
    return registry


@pytest.fixture
def mock_registry_unregistered():
    registry = MagicMock()
    registry.model_exists.return_value = False
    return registry


@pytest.mark.asyncio
async def test_ingest_accepted_for_registered_model(mock_redis, mock_registry_registered):
    service = IngestService(mock_redis, mock_registry_registered)
    req = IngestRequest(
        model_id="my_model",
        features={"age": 30, "income": 50000.0},
        prediction=1,
    )
    result = await service.ingest(req)
    assert result.queued is True
    assert result.status == "accepted"
    mock_redis.xadd.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_rejected_for_unregistered_model(mock_redis, mock_registry_unregistered):
    service = IngestService(mock_redis, mock_registry_unregistered)
    req = IngestRequest(
        model_id="unknown_model",
        features={"age": 30},
        prediction=0,
    )
    with pytest.raises(ValueError, match="not registered"):
        await service.ingest(req)


@pytest.mark.asyncio
async def test_ingest_batch_counts(mock_redis, mock_registry_registered):
    service = IngestService(mock_redis, mock_registry_registered)
    requests = [
        IngestRequest(model_id="my_model", features={"age": i}, prediction=0)
        for i in range(5)
    ]
    result = await service.ingest_batch(requests)
    assert result["accepted"] == 5
    assert result["rejected"] == 0
    assert result["total"] == 5


@pytest.mark.asyncio
async def test_ingest_batch_partial_rejection(mock_redis):
    registry = MagicMock()
    registry.model_exists.side_effect = lambda m: m == "good_model"
    service = IngestService(mock_redis, registry)
    requests = [
        IngestRequest(model_id="good_model", features={"x": 1}, prediction=0),
        IngestRequest(model_id="bad_model", features={"x": 1}, prediction=0),
        IngestRequest(model_id="good_model", features={"x": 2}, prediction=1),
    ]
    result = await service.ingest_batch(requests)
    assert result["accepted"] == 2
    assert result["rejected"] == 1
    assert result["total"] == 3
