from __future__ import annotations
import numpy as np
import pytest
from argus_core.drift.psi import (
    compute_psi, compute_psi_categorical, psi_severity
)


def test_identical_data_psi_near_zero(numeric_reference, numeric_production_nodrift):
    psi = compute_psi(
        np.array(numeric_reference),
        np.array(numeric_production_nodrift),
    )
    assert psi < 0.1


def test_drifted_data_high_psi(numeric_reference, numeric_production_drift):
    psi = compute_psi(
        np.array(numeric_reference),
        np.array(numeric_production_drift),
    )
    assert psi > 0.25


def test_categorical_no_drift(categorical_reference, categorical_production_nodrift):
    psi = compute_psi_categorical(categorical_reference, categorical_production_nodrift)
    assert psi < 0.1


def test_categorical_drift(categorical_reference, categorical_production_drift):
    psi = compute_psi_categorical(categorical_reference, categorical_production_drift)
    assert psi > 0.1


def test_psi_severity_thresholds():
    assert psi_severity(0.3) == "critical"
    assert psi_severity(0.15) == "warning"
    assert psi_severity(0.07) == "info"
    assert psi_severity(0.02) == "ok"


def test_psi_nonnegative():
    rng = np.random.default_rng(0)
    ref = rng.exponential(scale=2.0, size=1000)
    prod = rng.exponential(scale=5.0, size=200)
    psi = compute_psi(ref, prod)
    assert psi >= 0.0
