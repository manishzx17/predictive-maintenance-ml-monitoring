"""
Input and prediction drift monitoring layer.
Preserves stable, validated logic from Phases 11 and 12.
"""

from typing import Dict, List, Any, Union
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


def calculate_psi(
    ref_series: Union[pd.Series, np.ndarray],
    curr_series: Union[pd.Series, np.ndarray],
    n_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Calculate Population Stability Index (PSI) for a continuous feature using
    quantile-based binning derived strictly from the reference distribution.

    Preserves the exact formulation from Phase 11.

    Parameters
    ----------
    ref_series : array-like
        Baseline reference series.
    curr_series : array-like
        Incoming/monitored current series.
    n_bins : int
        Number of quantile bins (default 10).
    eps : float
        Small probability clip value to prevent division by zero or log(0) (default 1e-4).

    Returns
    -------
    float
        Population Stability Index.
    """
    ref_arr = np.asarray(ref_series)
    curr_arr = np.asarray(curr_series)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.unique(np.quantile(ref_arr, quantiles))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts = pd.cut(pd.Series(ref_arr), bins=bin_edges).value_counts(sort=False)
    curr_counts = pd.cut(pd.Series(curr_arr), bins=bin_edges).value_counts(sort=False)

    ref_pct = np.clip(ref_counts / len(ref_arr), eps, None)
    curr_pct = np.clip(curr_counts / len(curr_arr), eps, None)

    psi_value = float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))
    return psi_value


def run_ks_test(
    ref_series: Union[pd.Series, np.ndarray],
    curr_series: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Run two-sample Kolmogorov-Smirnov test between reference and current distributions.

    Parameters
    ----------
    ref_series : array-like
        Reference sample.
    curr_series : array-like
        Current monitored sample.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'statistic' (D) and 'p_value'.
    """
    res = ks_2samp(ref_series, curr_series)
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
    }


def calculate_input_drift(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    feature_cols: List[str],
    n_bins: int = 10,
    eps: float = 1e-4,
) -> pd.DataFrame:
    """
    Calculate KS statistics and PSI across all specified continuous input features.

    PSI Severity thresholds (Phase 11):
    - PSI < 0.10: 'OK'
    - 0.10 <= PSI <= 0.25: 'WARNING'
    - PSI > 0.25: 'ALERT'

    Parameters
    ----------
    ref_df : pd.DataFrame
        Reference batch dataframe.
    curr_df : pd.DataFrame
        Current incoming batch dataframe.
    feature_cols : List[str]
        List of continuous feature column names.
    n_bins : int
        Number of quantile bins for PSI (default 10).
    eps : float
        Clip epsilon for PSI (default 1e-4).

    Returns
    -------
    pd.DataFrame
        Table with Feature, KS Statistic, p-value, PSI, and Severity.
    """
    records = []
    for col in feature_cols:
        ks_res = run_ks_test(ref_df[col], curr_df[col])
        psi_val = calculate_psi(ref_df[col], curr_df[col], n_bins=n_bins, eps=eps)

        if psi_val < 0.10:
            severity = "OK"
        elif psi_val <= 0.25:
            severity = "WARNING"
        else:
            severity = "ALERT"

        records.append({
            "Feature": col,
            "KS_Statistic": ks_res["statistic"],
            "KS_P_Value": ks_res["p_value"],
            "PSI": psi_val,
            "Severity": severity,
        })

    return pd.DataFrame(records)


def calculate_prediction_drift(
    p_ref: np.ndarray,
    p_curr: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate prediction probability drift between reference and incoming batches.
    Computes two-sample KS test and 1-Wasserstein distance (Earth Mover's Distance).

    Preserves Phase 12 methodology.

    Parameters
    ----------
    p_ref : np.ndarray
        Model-predicted probabilities on reference batch.
    p_curr : np.ndarray
        Model-predicted probabilities on current monitored batch.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'ks_statistic', 'ks_p_value', and 'wasserstein_distance'.
    """
    ks_res = ks_2samp(p_ref, p_curr)
    w_dist = float(wasserstein_distance(p_ref, p_curr))

    return {
        "ks_statistic": float(ks_res.statistic),
        "ks_p_value": float(ks_res.pvalue),
        "wasserstein_distance": w_dist,
    }
