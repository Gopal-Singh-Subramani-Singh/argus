from __future__ import annotations
import pytest
from argus_core.drift.chi_squared import run_chi_squared, chi_squared_severity


def test_stable_categoricals_high_pvalue(categorical_reference, categorical_production_nodrift):
    stat, pval = run_chi_squared(categorical_reference, categorical_production_nodrift)
    assert pval > 0.05


def test_drifted_categoricals_low_pvalue(categorical_reference, categorical_production_drift):
    stat, pval = run_chi_squared(categorical_reference, categorical_production_drift)
    assert pval < 0.05
    assert chi_squared_severity(pval) == "critical"


def test_single_category_returns_defaults():
    stat, pval = run_chi_squared(["A"] * 100, ["A"] * 50)
    assert pval == 1.0


def test_new_category_in_production(categorical_reference):
    prod_with_new = categorical_reference[:100] + ["Z"] * 50
    stat, pval = run_chi_squared(categorical_reference, prod_with_new)
    assert stat >= 0.0


def test_severity():
    assert chi_squared_severity(0.01) == "critical"
    assert chi_squared_severity(0.07) == "warning"
    assert chi_squared_severity(0.5) == "ok"
