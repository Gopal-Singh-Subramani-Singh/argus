from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Tuple, List
import structlog

logger = structlog.get_logger(__name__)


def run_chi_squared(
    reference: List,
    production: List,
    feature_name: str = "unknown",
) -> Tuple[float, float]:
    """
    Chi-squared test for categorical feature drift.

    Tests whether the frequency distribution of categories in production
    is consistent with the reference distribution.

    Returns (statistic, p_value).
    LOW p-value (<0.05) = drift detected.
    """
    all_cats = sorted(list(set(reference) | set(production)))
    if len(all_cats) < 2:
        return 0.0, 1.0

    ref_counts = np.array([reference.count(c) for c in all_cats], dtype=float)
    prod_counts = np.array([production.count(c) for c in all_cats], dtype=float)

    if ref_counts.sum() == 0 or prod_counts.sum() == 0:
        return 0.0, 1.0

    # Only use categories that appear in reference (expected > 0)
    # Treat new production-only categories as part of the "other" bucket
    ref_mask = ref_counts > 0
    if ref_mask.sum() < 2:
        return 0.0, 1.0

    ref_freq = ref_counts[ref_mask] / ref_counts[ref_mask].sum()
    prod_observed = prod_counts[ref_mask]
    # Scale expected to match the sum of observed (categories in reference only)
    expected = ref_freq * prod_observed.sum()

    if expected.sum() == 0:
        return 0.0, 1.0

    statistic, p_value = stats.chisquare(
        f_obs=prod_observed,
        f_exp=expected,
    )

    logger.debug(
        "chi_squared.result",
        feature=feature_name,
        statistic=round(float(statistic), 4),
        p_value=round(float(p_value), 4),
        categories=len(all_cats),
    )
    return float(statistic), float(p_value)


def chi_squared_severity(p_value: float) -> str:
    if p_value < 0.05:
        return "critical"
    elif p_value < 0.1:
        return "warning"
    return "ok"
