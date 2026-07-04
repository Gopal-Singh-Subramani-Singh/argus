from __future__ import annotations
import numpy as np
from scipy.spatial.distance import jensenshannon
from typing import Tuple
import structlog

logger = structlog.get_logger(__name__)

EPSILON = 1e-10


def compute_js_divergence(
    reference: np.ndarray,
    production: np.ndarray,
    bins: int = 20,
    feature_name: str = "unknown",
) -> float:
    """
    Jensen-Shannon Divergence between reference and production distributions.

    JSD is the symmetric version of KL divergence, bounded [0, 1].
    JSD = 0: identical distributions
    JSD = 1: completely different distributions

    More sensitive to subtle tail changes than the KS test.
    """
    ref = np.array(reference, dtype=float)
    prod = np.array(production, dtype=float)
    ref = ref[~np.isnan(ref)]
    prod = prod[~np.isnan(prod)]

    if len(ref) < 2 or len(prod) < 2:
        return 0.0

    combined = np.concatenate([ref, prod])
    bin_min = np.percentile(combined, 1)
    bin_max = np.percentile(combined, 99)
    if bin_min == bin_max:
        return 0.0

    bin_edges = np.linspace(bin_min, bin_max, bins + 1)

    ref_hist, _ = np.histogram(ref, bins=bin_edges, density=False)
    prod_hist, _ = np.histogram(prod, bins=bin_edges, density=False)

    ref_prob = (ref_hist + EPSILON) / (ref_hist.sum() + EPSILON * len(ref_hist))
    prod_prob = (prod_hist + EPSILON) / (prod_hist.sum() + EPSILON * len(prod_hist))

    jsd = float(jensenshannon(ref_prob, prod_prob, base=2))
    jsd = max(0.0, min(1.0, jsd))

    logger.debug(
        "js_divergence.result",
        feature=feature_name,
        jsd=round(jsd, 4),
        ref_n=len(ref),
        prod_n=len(prod),
    )
    return jsd


def js_severity(jsd: float) -> str:
    if jsd > 0.3:
        return "critical"
    elif jsd > 0.1:
        return "warning"
    elif jsd > 0.05:
        return "info"
    return "ok"
