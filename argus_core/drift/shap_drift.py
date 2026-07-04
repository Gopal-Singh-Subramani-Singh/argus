from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr
import structlog

logger = structlog.get_logger(__name__)


def compute_shap_drift(
    reference_shap_values: np.ndarray,
    production_shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    SHAP feature importance drift via Spearman rank correlation.

    Computes mean absolute SHAP value per feature for reference and production.
    Returns (rank_correlation, per_feature_importance_delta).

    rank_correlation close to 1.0 = feature importance order is stable
    rank_correlation < 0.7 = feature importance order has shifted (CRITICAL)

    This is more stable than comparing raw SHAP values because:
    - SHAP values are unstable across different sample sets
    - Rank order of importance is much more robust
    - A model can have different raw SHAP magnitudes but same rank = same behaviour
    """
    ref = np.array(reference_shap_values)
    prod = np.array(production_shap_values)

    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if prod.ndim == 1:
        prod = prod.reshape(-1, 1)

    if ref.shape[1] != prod.shape[1]:
        logger.error(
            "shap_drift.feature_mismatch",
            ref_features=ref.shape[1],
            prod_features=prod.shape[1],
        )
        return 1.0, {}

    n_features = ref.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    ref_importance = np.mean(np.abs(ref), axis=0)
    prod_importance = np.mean(np.abs(prod), axis=0)

    if n_features == 1:
        delta = abs(ref_importance[0] - prod_importance[0]) / (ref_importance[0] + 1e-8)
        rank_corr = max(0.0, 1.0 - delta)
        return float(rank_corr), {feature_names[0]: float(delta)}

    correlation, p_value = spearmanr(ref_importance, prod_importance)
    rank_corr = float(correlation) if not np.isnan(correlation) else 1.0

    per_feature = {}
    for i, fname in enumerate(feature_names):
        ref_val = float(ref_importance[i])
        prod_val = float(prod_importance[i])
        delta = abs(ref_val - prod_val) / (ref_val + 1e-8)
        per_feature[fname] = round(delta, 4)

    logger.debug(
        "shap_drift.result",
        rank_correlation=round(rank_corr, 4),
        n_features=n_features,
        p_value=round(float(p_value), 4) if not np.isnan(p_value) else None,
    )
    return rank_corr, per_feature


def shap_severity(rank_correlation: float) -> str:
    if rank_correlation < 0.5:
        return "critical"
    elif rank_correlation < 0.7:
        return "warning"
    elif rank_correlation < 0.85:
        return "info"
    return "ok"


def estimate_shap_without_model(
    reference_data: np.ndarray,
    production_data: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Estimate SHAP drift without a model using feature variance as proxy.
    Used when a model artifact is not available.
    Computes coefficient of variation per feature as importance proxy.
    """
    ref = np.array(reference_data, dtype=float)
    prod = np.array(production_data, dtype=float)

    n_features = ref.shape[1] if ref.ndim > 1 else 1
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
        prod = prod.reshape(-1, 1)

    ref_cv = np.std(ref, axis=0) / (np.mean(np.abs(ref), axis=0) + 1e-8)
    prod_cv = np.std(prod, axis=0) / (np.mean(np.abs(prod), axis=0) + 1e-8)

    return compute_shap_drift(
        ref_cv.reshape(1, -1), prod_cv.reshape(1, -1), feature_names
    )
