"""
LogiData Adaptive Capacity & Risk Framework (ACRF)
===================================================
Author: Adebowale Jolaoso
Version: 1.0
Date: September 2026

A proprietary analytical framework for small and mid-sized enterprise
supply chain analytics. Implements three integrated components:

1. Data-Sparsity-Adjusted Safety Stock
2. Iterative Forecast-Error Correction
3. Composite Capacity-Risk Score (CCRS)

This module provides the core calculation engine. For usage examples,
see acrf_analysis_results.py.

Copyright (c) 2026 LogiData Analytics Solutions LLC. All rights reserved.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple


# ==============================================================================
# Component 1: Data-Sparsity-Adjusted Safety Stock
# ==============================================================================

def confidence_discount(n_observations: int,
                         enterprise_benchmark: int = 730) -> float:
    """
    Calculate the confidence-discount factor c(n).

    Approaches 1.0 as the number of available observations approaches
    enterprise-level data depth (~730 days = 2 years of daily data).
    Uses logarithmic scaling: the discount is steepest for very sparse
    data and flattens as data accumulates.

    Parameters
    ----------
    n_observations : int
        Number of historical demand observations available for a given
        SKU or client.
    enterprise_benchmark : int, default 730
        Number of observations at which c(n) = 1.0 (full confidence).
        Default is 730 (2 years of daily data).

    Returns
    -------
    float
        Confidence-discount factor in range [0.5, 1.0].
    """
    if n_observations <= 0:
        return 0.5
    c = min(1.0, np.log(1 + n_observations) / np.log(1 + enterprise_benchmark))
    return max(0.5, c)


def safety_stock_adjusted(avg_demand: float,
                           std_demand: float,
                           avg_lead_time: float,
                           std_lead_time: float,
                           service_level_z: float = 1.65,
                           n_observations: Optional[int] = None,
                           enterprise_benchmark: int = 730) -> Dict:
    """
    ACRF Data-Sparsity-Adjusted Safety Stock.

    Classical formula:
        SS = z * sqrt(L * sigma_D^2 + D^2 * sigma_L^2)

    ACRF-adjusted formula:
        SS(adj) = (z * c(n)) * sqrt(L * sigma_D^2 + D^2 * sigma_L^2)

    where c(n) is the confidence-discount factor that accounts for
    the number of historical observations available.

    Parameters
    ----------
    avg_demand : float
        Average daily demand (D).
    std_demand : float
        Standard deviation of daily demand (sigma_D).
    avg_lead_time : float
        Average supplier lead time in days (L).
    std_lead_time : float
        Standard deviation of lead time (sigma_L).
    service_level_z : float, default 1.65
        Z-score for target service level (1.65 = ~95%).
    n_observations : int, optional
        Number of historical observations. If None, no discount applied.
    enterprise_benchmark : int, default 730
        Observation count at which c(n) = 1.0.

    Returns
    -------
    dict
        ss_classical, ss_adjusted, confidence_discount, reduction_pct,
        capital_saved_units.
    """
    variance_component = np.sqrt(
        avg_lead_time * std_demand**2 + avg_demand**2 * std_lead_time**2
    )
    ss_classical = service_level_z * variance_component

    if n_observations is not None:
        c_n = confidence_discount(n_observations, enterprise_benchmark)
    else:
        c_n = 1.0

    ss_adjusted = (service_level_z * c_n) * variance_component

    return {
        'ss_classical': round(ss_classical, 1),
        'ss_adjusted': round(ss_adjusted, 1),
        'confidence_discount': round(c_n, 4),
        'reduction_pct': round((1 - c_n) * 100, 1),
        'capital_saved_units': round(ss_classical - ss_adjusted, 1)
    }


# ==============================================================================
# Component 2: Iterative Forecast-Error Correction
# ==============================================================================

def rolling_mape(actual: np.ndarray,
                  forecast: np.ndarray,
                  window: int = 30) -> List[Dict]:
    """
    Calculate rolling MAPE and derive a forecast-error correction factor.

    After each forecast period, the resulting MAPE is used to adjust
    the confidence-discount factor c(n), creating a feedback loop
    between forecast performance and inventory policy.

    Parameters
    ----------
    actual : np.ndarray
        Array of actual demand values.
    forecast : np.ndarray
        Array of forecast demand values.
    window : int, default 30
        Rolling window size for MAPE calculation.

    Returns
    -------
    list of dict
        Each dict contains: period, rolling_mape, mape_adjustment_factor.
    """
    results = []
    for i in range(window, len(actual)):
        a_window = actual[i - window:i]
        f_window = forecast[i - window:i]

        mask = a_window > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs(
                (a_window[mask] - f_window[mask]) / a_window[mask]
            )) * 100
        else:
            mape = 100.0

        # As MAPE decreases (forecast improves), the adjustment factor
        # approaches 1.0, allowing c(n) to increase toward full confidence.
        mape_factor = max(0.8, 1 - mape / 200)

        results.append({
            'period': i,
            'rolling_mape': round(mape, 2),
            'mape_adjustment_factor': round(mape_factor, 4)
        })

    return results


def adjusted_confidence_with_mape(c_n: float, mape_factor: float) -> float:
    """
    Apply MAPE-based correction to the confidence discount.

        c_adjusted = c_n * mape_factor

    This creates the feedback loop: as forecast accuracy improves
    over successive cycles, the safety-stock recommendation becomes
    more precise.
    """
    return round(min(1.0, c_n * mape_factor), 4)


# ==============================================================================
# Component 3: Composite Capacity-Risk Score (CCRS)
# ==============================================================================

def supplier_concentration_hhi(supplier_shares: pd.Series) -> Tuple[float, float]:
    """
    Calculate the Herfindahl-Hirschman Index for supplier concentration.

    Parameters
    ----------
    supplier_shares : pd.Series
        Normalized value counts of supplier assignments.

    Returns
    -------
    tuple of (disruption_score, hhi)
        disruption_score: 0-100 where higher = less concentrated (less risk).
        hhi: Raw Herfindahl-Hirschman Index (0 to 1).
    """
    hhi = (supplier_shares ** 2).sum()
    score = max(0, min(100, (1 - hhi) * 150))
    return round(score, 1), round(hhi, 4)


def composite_capacity_risk_score(utilization_efficiency: float,
                                   forecast_reliability: float,
                                   disruption_exposure: float,
                                   w_util: float = 0.40,
                                   w_forecast: float = 0.35,
                                   w_disruption: float = 0.25) -> Dict:
    """
    Calculate the Composite Capacity-Risk Score (CCRS).

        CCRS = w_util * Utilization + w_forecast * Forecast + w_disr * Disruption

    All inputs are normalized scores on a 0-100 scale.

    Parameters
    ----------
    utilization_efficiency : float
        How effectively existing capacity is deployed (0-100).
    forecast_reliability : float
        Inverse of rolling MAPE, normalized to 0-100.
    disruption_exposure : float
        Inverse of supplier concentration risk (0-100).
    w_util, w_forecast, w_disruption : float
        Component weights (must sum to 1.0).

    Returns
    -------
    dict
        ccrs_score, component breakdowns, risk_category.
    """
    assert abs(w_util + w_forecast + w_disruption - 1.0) < 0.001, \
        "Weights must sum to 1.0"

    ccrs = (w_util * utilization_efficiency +
            w_forecast * forecast_reliability +
            w_disruption * disruption_exposure)

    if ccrs >= 70:
        category = 'Low Risk'
    elif ccrs >= 50:
        category = 'Moderate Risk'
    else:
        category = 'High Risk'

    return {
        'ccrs_score': round(ccrs, 1),
        'utilization_component': round(w_util * utilization_efficiency, 1),
        'forecast_component': round(w_forecast * forecast_reliability, 1),
        'disruption_component': round(w_disruption * disruption_exposure, 1),
        'risk_category': category
    }


if __name__ == '__main__':
    # Quick self-test with the illustrative example from the methodology doc
    result = safety_stock_adjusted(
        avg_demand=40, std_demand=12,
        avg_lead_time=7, std_lead_time=1.5,
        service_level_z=1.65, n_observations=426  # ~14 months of weekdays
    )
    print("Self-test (Section 4 illustrative example):")
    print(f"  Classical SS: {result['ss_classical']} units")
    print(f"  Adjusted SS: {result['ss_adjusted']} units")
    print(f"  c(n): {result['confidence_discount']}")
    print(f"  Reduction: {result['reduction_pct']}%")
