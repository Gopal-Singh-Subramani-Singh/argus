from __future__ import annotations
import numpy as np
import pytest
from argus_core.drift.ks_test import run_ks_test, ks_severity


def test_identical_distributions_high_pvalue(numeric_reference, numeric_production_nodrift):
    stat, pval = run_ks_test(
        np.array(numeric_reference),
        np.array(numeric_production_nodrift),
    )
    assert stat < 0.2
    assert pval > 0.05
    assert ks_severity(pval) in ("ok", "info")


def test_different_distributions_low_pvalue(numeric_reference, numeric_production_drift):
    stat, pval = run_ks_test(
        np.array(numeric_reference),
        np.array(numeric_production_drift),
    )
    assert stat > 0.2
    assert pval < 0.05
    assert ks_severity(pval) == "critical"


def test_insufficient_samples_returns_defaults():
    stat, pval = run_ks_test(np.array([1.0]), np.array([2.0]))
    assert stat == 0.0
    assert pval == 1.0


def test_severity_thresholds():
    assert ks_severity(0.01) == "critical"
    assert ks_severity(0.07) == "warning"
    assert ks_severity(0.15) == "info"
    assert ks_severity(0.5) == "ok"


def test_handles_nan_values():
    ref = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    prod = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    stat, pval = run_ks_test(ref, prod)
    assert not np.isnan(stat)
    assert not np.isnan(pval)
