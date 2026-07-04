from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, patch
from argus_core.alerts.engine import AlertEngine
from argus_core.models import DriftScore, DriftSeverity


def make_scores(method, score):
    return [DriftScore(
        feature_name="age",
        method=method,
        score=score,
        severity=DriftSeverity.WARNING,
        sample_count=100,
    )]


@pytest.mark.asyncio
async def test_alert_fires_on_threshold_breach():
    engine = AlertEngine()
    engine._rules = []
    from argus_core.models import AlertRule
    engine._rules = [
        AlertRule(
            name="test_psi",
            method="psi",
            threshold=0.1,
            operator="gt",
            severity=DriftSeverity.WARNING,
            description="test",
        )
    ]
    engine._webhooks = []

    scores = make_scores("psi", 0.3)
    alerts = await engine.evaluate("model_x", scores)
    assert len(alerts) == 1
    assert alerts[0].rule_name == "test_psi"


@pytest.mark.asyncio
async def test_no_alert_below_threshold():
    engine = AlertEngine()
    from argus_core.models import AlertRule
    engine._rules = [
        AlertRule(
            name="test_psi",
            method="psi",
            threshold=0.25,
            operator="gt",
            severity=DriftSeverity.CRITICAL,
            description="test",
        )
    ]
    engine._webhooks = []
    scores = make_scores("psi", 0.1)
    alerts = await engine.evaluate("model_x", scores)
    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_lt_operator():
    engine = AlertEngine()
    from argus_core.models import AlertRule
    engine._rules = [
        AlertRule(
            name="ks_critical",
            method="ks_test",
            threshold=0.05,
            operator="lt",
            severity=DriftSeverity.CRITICAL,
            description="test",
        )
    ]
    engine._webhooks = []
    scores = [DriftScore(
        feature_name="age", method="ks_test",
        score=0.03, p_value=0.03,
        severity=DriftSeverity.CRITICAL, sample_count=100,
    )]
    alerts = await engine.evaluate("m", scores)
    assert len(alerts) == 1


@pytest.mark.asyncio
async def test_multiple_features_multiple_alerts():
    engine = AlertEngine()
    from argus_core.models import AlertRule
    engine._rules = [
        AlertRule(
            name="psi_warn", method="psi", threshold=0.1,
            operator="gt", severity=DriftSeverity.WARNING, description=""
        )
    ]
    engine._webhooks = []
    scores = [
        DriftScore(feature_name="f1", method="psi", score=0.3,
                   severity=DriftSeverity.WARNING, sample_count=100),
        DriftScore(feature_name="f2", method="psi", score=0.2,
                   severity=DriftSeverity.WARNING, sample_count=100),
    ]
    alerts = await engine.evaluate("m", scores)
    assert len(alerts) == 2
