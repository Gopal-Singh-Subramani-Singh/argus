from __future__ import annotations
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from argus_core.registry.model_registry import ModelRegistry
from argus_core.drift.engine import DriftEngine
from argus_core.alerts.engine import AlertEngine


@pytest.fixture
def numeric_reference():
    rng = np.random.default_rng(42)
    return rng.normal(loc=50.0, scale=10.0, size=500).tolist()


@pytest.fixture
def numeric_production_nodrift(numeric_reference):
    rng = np.random.default_rng(99)
    return rng.normal(loc=50.0, scale=10.0, size=200).tolist()


@pytest.fixture
def numeric_production_drift():
    rng = np.random.default_rng(7)
    return rng.normal(loc=75.0, scale=15.0, size=200).tolist()


@pytest.fixture
def categorical_reference():
    cats = ["A", "B", "C", "D"]
    rng = np.random.default_rng(42)
    probs = [0.4, 0.3, 0.2, 0.1]
    return rng.choice(cats, size=500, p=probs).tolist()


@pytest.fixture
def categorical_production_nodrift():
    cats = ["A", "B", "C", "D"]
    rng = np.random.default_rng(99)
    probs = [0.4, 0.3, 0.2, 0.1]
    return rng.choice(cats, size=200, p=probs).tolist()


@pytest.fixture
def categorical_production_drift():
    cats = ["A", "B", "C", "D"]
    rng = np.random.default_rng(7)
    probs = [0.1, 0.1, 0.4, 0.4]  # flipped
    return rng.choice(cats, size=200, p=probs).tolist()


@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=ModelRegistry)
    registry.model_exists.return_value = True
    registry.get_feature_schema = AsyncMock(return_value=[
        {"name": "age", "type": "numeric"},
        {"name": "income", "type": "numeric"},
        {"name": "category", "type": "categorical"},
    ])
    registry.list_models = AsyncMock(return_value=[
        {"model_id": "test_model", "name": "Test", "version": "1"}
    ])
    registry.get_model = AsyncMock(return_value={
        "model_id": "test_model", "name": "Test", "version": "1"
    })
    return registry
