from __future__ import annotations
import numpy as np
import pytest
from argus_core.drift.js_divergence import compute_js_divergence, js_severity


def test_identical_distributions_near_zero(numeric_reference, numeric_production_nodrift):
    jsd = compute_js_divergence(
        np.array(numeric_reference),
        np.array(numeric_production_nodrift),
    )
    # Same distribution family — JSD should be low (< 0.15 for finite samples)
    assert jsd < 0.15


def test_drifted_distributions_high_jsd(numeric_reference, numeric_production_drift):
    jsd = compute_js_divergence(
        np.array(numeric_reference),
        np.array(numeric_production_drift),
    )
    assert jsd > 0.1


def test_jsd_bounded_zero_to_one():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 1000)
    b = rng.normal(10, 1, 1000)
    jsd = compute_js_divergence(a, b)
    assert 0.0 <= jsd <= 1.0


def test_severity():
    assert js_severity(0.4) == "critical"
    assert js_severity(0.2) == "warning"
    assert js_severity(0.06) == "info"
    assert js_severity(0.02) == "ok"


def test_insufficient_returns_zero():
    jsd = compute_js_divergence(np.array([1.0]), np.array([2.0]))
    assert jsd == 0.0
