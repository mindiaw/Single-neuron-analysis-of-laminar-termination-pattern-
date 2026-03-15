#!/usr/bin/env python3
"""Shared utility functions for the laminar analysis pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

log = logging.getLogger(__name__)
DEFAULT_CLR_PSEUDOCOUNT = 1e-6
ABS_TRANSFORM_CHOICES = ("none", "log1p", "sqrt", "power", "asinh")
DEFAULT_ABS_TRANSFORM_POWER = 0.75
DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR = 1.0


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path.name)


def _pearson_ward_embedding(X: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Row-center and L2-normalize rows so Euclidean distance follows
    Pearson/correlation geometry (up to a constant factor).
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    eps = 1e-12
    valid = norms[:, 0] > eps

    X_emb = np.zeros_like(Xc)
    X_emb[valid] = Xc[valid] / norms[valid]
    n_zero_var = int((~valid).sum())
    return X_emb, n_zero_var


def _normalize_metric_name(metric: str) -> str:
    """Normalize metric aliases to canonical names."""
    metric_l = str(metric).lower()
    if metric_l == "pearson":
        return "correlation"
    if metric_l in {"aitchison", "clr", "clr_euclidean", "euclidean_clr"}:
        return "aitchison"
    return metric_l


def is_aitchison_metric(metric: str) -> bool:
    """Return True when metric refers to Aitchison/CLR geometry."""
    return _normalize_metric_name(metric) == "aitchison"


def clr_transform_rows(
    X: np.ndarray,
    pseudocount: float = DEFAULT_CLR_PSEUDOCOUNT,
    context: str = "clr_transform_rows",
) -> np.ndarray:
    """
    Row-wise centered log-ratio (CLR) transform for compositional features.

    A small pseudocount is added elementwise before log transform so zeros are
    handled robustly. Negative values are invalid for CLR and raise ValueError.
    """
    if pseudocount <= 0:
        raise ValueError(f"{context}: pseudocount must be > 0 (got {pseudocount}).")

    X_arr = np.asarray(X, dtype=float).copy()
    if X_arr.size == 0:
        return X_arr

    finite = np.isfinite(X_arr)
    if not finite.all():
        n_bad = int((~finite).sum())
        log.warning("%s: %d non-finite value(s) encountered before CLR transform.", context, n_bad)

    min_val = float(np.nanmin(X_arr))
    if min_val < 0:
        n_neg = int((X_arr < 0).sum())
        raise ValueError(
            f"{context}: CLR requires nonnegative values; found {n_neg} negative value(s), "
            f"min={min_val:.6g}."
        )

    X_pos = X_arr + float(pseudocount)
    with np.errstate(divide="raise", invalid="raise"):
        logX = np.log(X_pos)
    clr = logX - logX.mean(axis=1, keepdims=True)
    return clr


def prepare_features_for_clustering(
    X: np.ndarray,
    method: str,
    metric: str,
    context: str,
    clr_pseudocount: float = DEFAULT_CLR_PSEUDOCOUNT,
) -> tuple[np.ndarray, str]:
    """
    Prepare feature matrix and effective metric for hierarchical clustering.

    Returns:
      X_prepared, metric_for_pdist

    Notes:
      - For method='ward', linkage is run directly on X_prepared and
        metric_for_pdist is informational only.
      - metric='aitchison' applies CLR transform and uses Euclidean geometry
        in CLR space.
    """
    X_arr = np.asarray(X, dtype=float)
    method_l = str(method).lower()
    metric_l = _normalize_metric_name(metric)

    if method_l == "ward":
        if metric_l == "euclidean":
            return X_arr, "euclidean"
        if metric_l == "correlation":
            X_emb, n_zero_var = _pearson_ward_embedding(X_arr)
            if n_zero_var > 0:
                log.warning(
                    "%s: %d row(s) had near-zero within-row variance during "
                    "Pearson-Ward embedding; keeping them as zero vectors.",
                    context, n_zero_var,
                )
            return X_emb, "euclidean"
        if metric_l == "aitchison":
            X_clr = clr_transform_rows(
                X_arr, pseudocount=clr_pseudocount, context=f"{context}[aitchison]"
            )
            return X_clr, "euclidean"
        raise ValueError(
            "For method='ward', metric must be one of: "
            "'euclidean', 'correlation', 'pearson', or 'aitchison' (aliases: clr, clr_euclidean)."
        )

    if metric_l == "aitchison":
        X_clr = clr_transform_rows(
            X_arr, pseudocount=clr_pseudocount, context=f"{context}[aitchison]"
        )
        return X_clr, "euclidean"
    return X_arr, metric_l


def hcluster(
    X: np.ndarray,
    n_clusters: int,
    method: str = "ward",
    metric: str = "euclidean",
) -> np.ndarray:
    """Return 1-based cluster labels via hierarchical clustering."""
    X_arr = np.asarray(X, dtype=float)
    method_l = str(method).lower()
    X_prepared, metric_eff = prepare_features_for_clustering(
        X_arr, method=method_l, metric=metric, context="hcluster"
    )
    if method_l == "ward":
        Z = linkage(X_prepared, method="ward")
    else:
        Z = linkage(pdist(X_prepared, metric=metric_eff), method=method_l)
    return fcluster(Z, t=n_clusters, criterion="maxclust")


def normalize_density_features(
    df: pd.DataFrame, feature_cols: list[str],
) -> pd.DataFrame:
    """Normalize density features to sum to 1 per row, dropping zero-sum rows."""
    df = df.copy()
    df[feature_cols] = df[feature_cols].fillna(0.0)
    row_sums = df[feature_cols].sum(axis=1)
    n_excluded = int((row_sums == 0).sum())
    df = df[row_sums > 0].copy()
    row_sums = row_sums[row_sums > 0]
    for c in feature_cols:
        df[c] = df[c] / row_sums
    log.info(
        "Density normalisation: %d rows retained, %d excluded (zero total density).",
        len(df), n_excluded,
    )
    return df


def apply_axon_length_threshold(
    df: pd.DataFrame,
    mode: str,
    min_target_axon_length_um: float = 1000.0,
    total_col: str = "aTotal",
) -> pd.DataFrame:
    """
    Apply an aTotal threshold for axon-length-based modes.

    The threshold is applied only for:
      - mode='axon'      (fraction of axon length by layer)
      - mode='axon_abs'  (absolute axon length by layer)
    """
    if mode not in {"axon", "axon_abs"}:
        return df
    if min_target_axon_length_um is None or min_target_axon_length_um <= 0:
        return df
    if total_col not in df.columns:
        raise ValueError(
            f"Mode '{mode}' requires column '{total_col}' for thresholding, "
            f"but it is missing. Available columns: {list(df.columns)}"
        )

    out = df.copy()
    totals = pd.to_numeric(out[total_col], errors="coerce").fillna(0.0)
    keep = totals >= float(min_target_axon_length_um)
    n_excluded = int((~keep).sum())
    out = out.loc[keep].copy()
    log.info(
        "Axon-length threshold (%s >= %.1f um): %d rows retained, %d excluded.",
        total_col, float(min_target_axon_length_um), len(out), n_excluded,
    )
    return out


def apply_terminal_count_threshold_for_axon_abs(
    df: pd.DataFrame,
    mode: str,
    min_target_terminals_for_axon_abs: int = 0,
    total_col: str = "tTotal",
) -> pd.DataFrame:
    """
    Apply a tTotal threshold only for mode='axon_abs'.

    This is intended as an additional quality filter for absolute axon-length
    analyses, e.g. requiring both:
      - aTotal >= 1000 um (handled by apply_axon_length_threshold), and
      - tTotal >= 1 (handled here).
    """
    if mode != "axon_abs":
        return df
    if min_target_terminals_for_axon_abs is None or int(min_target_terminals_for_axon_abs) <= 0:
        return df
    if total_col not in df.columns:
        raise ValueError(
            f"Mode '{mode}' requires column '{total_col}' for terminal thresholding, "
            f"but it is missing. Available columns: {list(df.columns)}"
        )

    out = df.copy()
    totals = pd.to_numeric(out[total_col], errors="coerce").fillna(0)
    keep = totals >= int(min_target_terminals_for_axon_abs)
    n_excluded = int((~keep).sum())
    out = out.loc[keep].copy()
    log.info(
        "Terminal-count threshold for axon_abs (%s >= %d): %d rows retained, %d excluded.",
        total_col, int(min_target_terminals_for_axon_abs), len(out), n_excluded,
    )
    return out


def transform_nonnegative(
    X: np.ndarray,
    transform: str = "log1p",
    context: str = "transform_nonnegative",
    power: float = DEFAULT_ABS_TRANSFORM_POWER,
    asinh_cofactor: float = DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR,
) -> np.ndarray:
    """
    Apply a nonnegative value transform to a feature matrix.

    Supported transforms:
      - none   : identity
      - log1p  : log(1 + x)
      - sqrt   : sqrt(x)
      - power  : x ** power  (power > 0)
      - asinh  : asinh(x / asinh_cofactor)  (asinh_cofactor > 0)

    Negative values are clipped to 0 with a warning.
    """
    X_arr = np.asarray(X, dtype=float).copy()
    if X_arr.size == 0:
        return X_arr

    transform_l = str(transform).lower()
    if transform_l not in ABS_TRANSFORM_CHOICES:
        raise ValueError(
            f"{context}: unknown transform {transform!r}; expected one of {ABS_TRANSFORM_CHOICES}."
        )

    finite = np.isfinite(X_arr)
    if not finite.all():
        n_bad = int((~finite).sum())
        log.warning("%s: %d non-finite value(s) encountered before transform.", context, n_bad)

    min_val = float(np.nanmin(X_arr))
    if min_val < 0:
        n_neg = int((X_arr < 0).sum())
        log.warning(
            "%s: %d negative value(s) encountered (min=%.6g); clipping to 0 before transform '%s'.",
            context, n_neg, min_val, transform_l,
        )
        X_arr = np.clip(X_arr, 0.0, None)

    if transform_l == "none":
        return X_arr
    if transform_l == "log1p":
        return np.log1p(X_arr)
    if transform_l == "sqrt":
        return np.sqrt(X_arr)
    if transform_l == "power":
        if power <= 0:
            raise ValueError(f"{context}: power transform requires power > 0 (got {power}).")
        return np.power(X_arr, float(power))
    if transform_l == "asinh":
        if asinh_cofactor <= 0:
            raise ValueError(
                f"{context}: asinh transform requires asinh_cofactor > 0 (got {asinh_cofactor})."
            )
        return np.arcsinh(X_arr / float(asinh_cofactor))

    raise AssertionError("unreachable")


def log1p_nonnegative(X: np.ndarray, context: str = "log1p_nonnegative") -> np.ndarray:
    """
    Apply log1p to a feature matrix, clipping negatives to 0 if present.

    Intended for absolute nonnegative features (counts/lengths). Returns a
    transformed copy and leaves the input untouched.
    """
    return transform_nonnegative(X, transform="log1p", context=context)
