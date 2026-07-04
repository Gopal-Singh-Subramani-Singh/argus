from __future__ import annotations
import numpy as np
import pytest
from argus_core.drift.shap_drift import (
    compute_shap_drift, shap_severity, estimate_shap_without_model
)


def make_shap_values(n_samples=200, n_features=5, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features))


def test_stable_importance_high_correlation():
    # Deterministic test: explicitly set importance order the same in both
    # Feature importances (mean abs): [10, 8, 6, 4, 2] for both ref and prod
    # This guarantees rank correlation = 1.0
    n = 100
    ref = np.zeros((n, 5))
    prod = np.zeros((n, 5))
    for i, scale in enumerate([10.0, 8.0, 6.0, 4.0, 2.0]):
        ref[:, i] = np.linspace(-scale, scale, n)
        prod[:, i] = np.linspace(-scale * 0.9, scale * 0.9, n)  # slight shift, same order
    corr, _ = compute_shap_drift(ref, prod)
    assert corr > 0.5


def test_shuffled_importance_low_correlation():
    rng = np.random.default_rng(42)
    ref = rng.standard_normal((200, 5))
    ref[:, 0] *= 10  # feature 0 is most important in reference
    prod = rng.standard_normal((200, 5))
    prod[:, 4] *= 10  # feature 4 is most important in production
    corr, per_feature = compute_shap_drift(ref, prod)
    assert corr < 0.9


def test_severity_thresholds():
    assert shap_severity(0.4) == "critical"
    assert shap_severity(0.6) == "warning"
    assert shap_severity(0.75) == "info"
    assert shap_severity(0.9) == "ok"


def test_returns_per_feature_deltas():
    ref = make_shap_values(seed=1)
    prod = make_shap_values(seed=99)
    features = ["age", "income", "cat", "score", "flag"]
    corr, deltas = compute_shap_drift(ref, prod, feature_names=features)
    assert set(deltas.keys()) == set(features)


def test_estimate_without_model():
    rng = np.random.default_rng(42)
    ref = rng.standard_normal((500, 4))
    prod = rng.standard_normal((200, 4)) * 5  # scale shift
    corr, per_feature = estimate_shap_without_model(ref, prod)
    assert 0.0 <= corr <= 1.0
    assert len(per_feature) == 4
