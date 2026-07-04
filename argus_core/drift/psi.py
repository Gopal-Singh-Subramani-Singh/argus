from __future__ import annotations
import numpy as np
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)

EPSILON = 1e-7  # avoid log(0)


def compute_psi(
    reference: np.ndarray,
    production: np.ndarray,
    bins: int = 10,
    feature_name: str = "unknown",
) -> float:
    """
    Population Stability Index (PSI).

    PSI = sum((actual% - expected%) * ln(actual% / expected%))

    Thresholds:
    - PSI < 0.1:  no significant change (OK)
    - PSI 0.1–0.25: moderate change (WARNING)
    - PSI > 0.25: significant change (CRITICAL)

    Works for both numeric (auto-binned) and pre-binned categorical data.
    """
    ref = np.array(reference, dtype=float)
    prod = np.array(production, dtype=float)
    ref = ref[~np.isnan(ref)]
    prod = prod[~np.isnan(prod)]

    if len(ref) < 2 or len(prod) < 2:
        logger.warning(
            "psi.insufficient_samples",
            feature=feature_name,
            ref_n=len(ref),
            prod_n=len(prod),
        )
        return 0.0

    bin_edges = np.percentile(ref, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    prod_counts, _ = np.histogram(prod, bins=bin_edges)

    ref_pct = ref_counts / (len(ref) + EPSILON)
    prod_pct = prod_counts / (len(prod) + EPSILON)

    ref_pct = np.where(ref_pct == 0, EPSILON, ref_pct)
    prod_pct = np.where(prod_pct == 0, EPSILON, prod_pct)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))

    logger.debug(
        "psi.result",
        feature=feature_name,
        psi=round(float(psi), 4),
        bins=len(bin_edges) - 1,
        ref_n=len(ref),
        prod_n=len(prod),
    )
    return float(psi)


def compute_psi_categorical(
    reference: list,
    production: list,
    feature_name: str = "unknown",
) -> float:
    """PSI for categorical features."""
    all_cats = list(set(reference) | set(production))
    ref_counts = {c: reference.count(c) for c in all_cats}
    prod_counts = {c: production.count(c) for c in all_cats}

    ref_n = len(reference) or 1
    prod_n = len(production) or 1

    psi = 0.0
    for cat in all_cats:
        ref_pct = max(ref_counts.get(cat, 0) / ref_n, EPSILON)
        prod_pct = max(prod_counts.get(cat, 0) / prod_n, EPSILON)
        psi += (prod_pct - ref_pct) * np.log(prod_pct / ref_pct)

    return float(psi)


def psi_severity(psi: float) -> str:
    if psi > 0.25:
        return "critical"
    elif psi > 0.1:
        return "warning"
    elif psi > 0.05:
        return "info"
    return "ok"
