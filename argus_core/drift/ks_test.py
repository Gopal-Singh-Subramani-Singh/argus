from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Tuple
import structlog

logger = structlog.get_logger(__name__)


def run_ks_test(
    reference: np.ndarray,
    production: np.ndarray,
    feature_name: str = "unknown",
) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test.

    Compares the empirical CDFs of reference and production distributions.
    Returns (statistic, p_value).
    - statistic: max vertical distance between CDFs (0=identical, 1=completely different)
    - p_value: probability of seeing this statistic under H0 (same distribution)
    - LOW p-value (<0.05) means the distributions ARE different → drift detected
    """
    ref = np.array(reference, dtype=float)
    prod = np.array(production, dtype=float)

    ref = ref[~np.isnan(ref)]
    prod = prod[~np.isnan(prod)]

    if len(ref) < 2 or len(prod) < 2:
        logger.warning(
            "ks_test.insufficient_samples",
            feature=feature_name,
            ref_n=len(ref),
            prod_n=len(prod),
        )
        return 0.0, 1.0

    statistic, p_value = stats.ks_2samp(ref, prod)

    logger.debug(
        "ks_test.result",
        feature=feature_name,
        statistic=round(statistic, 4),
        p_value=round(p_value, 4),
        ref_n=len(ref),
        prod_n=len(prod),
    )
    return float(statistic), float(p_value)


def ks_severity(p_value: float) -> str:
    if p_value < 0.05:
        return "critical"
    elif p_value < 0.1:
        return "warning"
    elif p_value < 0.2:
        return "info"
    return "ok"
