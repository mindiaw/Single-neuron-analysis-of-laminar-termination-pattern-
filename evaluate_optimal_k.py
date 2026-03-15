#!/usr/bin/env python3
"""
Evaluate the optimal number of clusters (k) for laminar projection profiles.

Computes three metrics across k=2..k_max, separately for FF and FB neurons:
  - Silhouette score
  - Within-cluster variance
  - Gap statistic (Tibshirani, Walther & Hastie 2001)

Outputs (in --out_dir):
  - optimal_k_data.csv   — per-direction, per-k metric values
  - optimal_k_curves.png — 3-panel plot with optimal k marked

Usage:
  python evaluate_optimal_k.py \
    --in_csv output/ALL_CONNECTIONS__mapped_and_terminals.csv \
    --out_dir output/optimal_k \
    --mode terminals \
    --k_max 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

from constants import DIRECTION_PALETTE, get_feature_cols
from utils import (
    save_figure as _save,
    normalize_density_features,
    apply_axon_length_threshold,
    apply_terminal_count_threshold_for_axon_abs,
    transform_nonnegative,
    prepare_features_for_clustering,
    is_aitchison_metric,
    ABS_TRANSFORM_CHOICES,
    DEFAULT_ABS_TRANSFORM_POWER,
    DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def _within_cluster_dispersion(X: np.ndarray, labels: np.ndarray) -> float:
    """Total within-cluster sum of squared distances from cluster centroids."""
    wk = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        center = pts.mean(axis=0)
        wk += float(np.sum((pts - center) ** 2))
    return wk


def evaluate_optimal_k(
    df: pd.DataFrame,
    feature_cols: list[str],
    k_range: range,
    out_dir: Path,
    mode: str,
    method: str = "ward",
    metric: str = "euclidean",
    abs_transform: str = "none",
    abs_transform_power: float = DEFAULT_ABS_TRANSFORM_POWER,
    abs_transform_asinh_cofactor: float = DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR,
    n_gap_ref: int = 20,
) -> pd.DataFrame:
    """
    Evaluate silhouette score, within-cluster variance, and gap statistic
    across a range of k, separately for FF and FB neurons.

    Outputs:
        optimal_k_data.csv
        optimal_k_curves.png
    """
    rng = np.random.default_rng(seed=42)
    rows = []

    for direction, group in df.groupby("direction"):
        X_raw = group[feature_cols].dropna().to_numpy(dtype=float)
        if len(X_raw) < 4:
            log.warning("Group '%s' too small for optimal-k analysis (%d rows).", direction, len(X_raw))
            continue

        if mode in {"terminals_abs", "axon_abs"} and str(abs_transform).lower() != "none":
            X_raw = transform_nonnegative(
                X_raw,
                transform=abs_transform,
                context=f"evaluate_optimal_k[{direction}]",
                power=abs_transform_power,
                asinh_cofactor=abs_transform_asinh_cofactor,
            )

        method_l = str(method).lower()
        X, metric_eff = prepare_features_for_clustering(
            X_raw, method=method_l, metric=metric, context=f"evaluate_optimal_k[{direction}]"
        )

        if method_l == "ward":
            Z = linkage(X, method="ward")
        else:
            d = pdist(X, metric=metric_eff)
            Z = linkage(d, method=method_l)

        # Pre-generate reference datasets and their linkage matrices for gap
        n, d = X.shape
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        refs = []
        for _ in range(n_gap_ref):
            X_ref = rng.uniform(X_min, X_max, size=(n, d))
            if method_l == "ward":
                Z_ref = linkage(X_ref, method="ward")
            else:
                d_ref = pdist(X_ref, metric=metric_eff)
                Z_ref = linkage(d_ref, method=method_l)
            refs.append((X_ref, Z_ref))

        for k in k_range:
            if k >= len(X):
                continue
            labels = fcluster(Z, t=k, criterion="maxclust")
            n_unique = len(np.unique(labels))
            sil = float(silhouette_score(X, labels)) if n_unique > 1 else np.nan

            # Size-weighted within-cluster variance
            unique_labels = np.unique(labels)
            cluster_sizes = np.array([np.sum(labels == c) for c in unique_labels])
            cluster_vars = np.array([X[labels == c].var(axis=0).mean() for c in unique_labels])
            wcv = float(np.average(cluster_vars, weights=cluster_sizes))

            # Gap statistic
            log_wk = np.log(max(_within_cluster_dispersion(X, labels), 1e-10))
            log_wk_refs = []
            for X_ref, Z_ref in refs:
                labels_ref = fcluster(Z_ref, t=k, criterion="maxclust")
                wk_ref = _within_cluster_dispersion(X_ref, labels_ref)
                log_wk_refs.append(np.log(max(wk_ref, 1e-10)))
            log_wk_refs = np.array(log_wk_refs)
            gap = float(log_wk_refs.mean() - log_wk)
            gap_se = float(log_wk_refs.std() * np.sqrt(1.0 + 1.0 / n_gap_ref))

            rows.append({"direction": direction, "k": k,
                          "silhouette": sil, "within_cluster_variance": wcv,
                          "gap": gap, "gap_se": gap_se})

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "optimal_k_data.csv", index=False)

    # ── Plot: 3 panels ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for direction in result["direction"].unique():
        sub = result[result["direction"] == direction].sort_values("k")
        color = DIRECTION_PALETTE.get(direction)
        axes[0].plot(sub["k"], sub["silhouette"], marker="o", label=direction, color=color)
        axes[1].plot(sub["k"], sub["within_cluster_variance"], marker="o", label=direction, color=color)
        axes[2].errorbar(sub["k"], sub["gap"], yerr=sub["gap_se"],
                         marker="o", label=direction, color=color, capsize=3)

        # Mark optimal k on gap panel (first k where Gap(k) >= Gap(k+1) - se(k+1))
        ks = sub["k"].values
        gaps = sub["gap"].values
        ses = sub["gap_se"].values
        for i in range(len(ks) - 1):
            if gaps[i] >= gaps[i + 1] - ses[i + 1]:
                axes[2].axvline(ks[i], color=color, linestyle="--", alpha=0.5)
                axes[2].text(ks[i] + 0.3, gaps[i], f"k*={ks[i]}",
                             fontsize=8, color=color, fontweight="bold")
                break

    axes[0].set(xlabel="k (number of clusters)", ylabel="Silhouette score",
                title="Silhouette score vs k")
    axes[1].set(xlabel="k (number of clusters)", ylabel="Within-cluster variance",
                title="Within-cluster variance vs k")
    axes[2].set(xlabel="k (number of clusters)", ylabel="Gap statistic",
                title="Gap statistic vs k\n(Tibshirani et al. 2001)")
    for ax in axes:
        ax.legend(title="Direction")
        ax.grid(True, alpha=0.4)
    fig.tight_layout()
    _save(fig, out_dir / "optimal_k_curves.png")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate optimal cluster number (k) for laminar profiles."
    )
    ap.add_argument("--in_csv", required=True,
                    help="Path to ALL_CONNECTIONS__mapped_and_terminals.csv")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for optimal_k_data.csv and plots.")
    ap.add_argument("--mode", choices=["terminals", "axon", "density", "terminals_abs", "axon_abs"], default="terminals",
                    help="Feature set: fT*, fA*, dTL*, tL*, or aL* (default: terminals).")
    ap.add_argument("--min_target_axon_length_um", type=float, default=1000.0,
                    help="Minimum aTotal (um) for axon-based modes (default: 1000).")
    ap.add_argument(
        "--min_target_terminals_for_axon_abs",
        type=int,
        default=0,
        help=(
            "Additional minimum tTotal for mode='axon_abs' only (default: 0 = disabled). "
            "Example: set to 1 to require at least one terminal."
        ),
    )
    ap.add_argument("--k_max", type=int, default=20,
                    help="Maximum k to evaluate (default: 20).")
    ap.add_argument("--method", default="ward",
                    help="Hierarchical linkage method (default: ward).")
    ap.add_argument("--metric", default="euclidean",
                    help=(
                        "Distance metric for clustering geometry used in optimal-k evaluation. "
                        "For method='ward', supports euclidean or correlation/pearson "
                        "(via row-centering + L2 normalization), "
                        "or aitchison/clr (via CLR transform + Euclidean). "
                        "For non-ward methods, passed to pdist."
                    ))
    ap.add_argument(
        "--no_abs_log1p", action="store_true",
        help=(
            "Deprecated: disable log1p transform for absolute-feature modes. "
            "Use --abs_transform none."
        ),
    )
    ap.add_argument(
        "--abs_transform",
        choices=list(ABS_TRANSFORM_CHOICES),
        default="log1p",
        help=(
            "Transform for absolute-feature modes before optimal-k evaluation "
            "(default: log1p). Ignored for non-absolute modes."
        ),
    )
    ap.add_argument(
        "--abs_transform_power",
        type=float,
        default=DEFAULT_ABS_TRANSFORM_POWER,
        help="Exponent for --abs_transform power (default: 0.75).",
    )
    ap.add_argument(
        "--abs_transform_asinh_cofactor",
        type=float,
        default=DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR,
        help="Cofactor for --abs_transform asinh (default: 1.0).",
    )
    ap.add_argument("--log_level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging verbosity (default: INFO).")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    feature_cols = get_feature_cols(args.mode)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing feature columns for mode='{args.mode}': {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    if args.mode == "density":
        df = normalize_density_features(df, feature_cols)
    df = apply_axon_length_threshold(
        df, args.mode, min_target_axon_length_um=args.min_target_axon_length_um
    )
    df = apply_terminal_count_threshold_for_axon_abs(
        df,
        args.mode,
        min_target_terminals_for_axon_abs=args.min_target_terminals_for_axon_abs,
    )

    log.info("Loaded %d neurons, mode=%s", len(df), args.mode)
    log.info("Evaluating k=2..%d", args.k_max)
    abs_transform_for_abs = str(args.abs_transform).lower()
    if args.no_abs_log1p:
        if abs_transform_for_abs != "log1p":
            log.warning(
                "--no_abs_log1p is deprecated and ignored because --abs_transform=%s was provided.",
                abs_transform_for_abs,
            )
        else:
            abs_transform_for_abs = "none"
            log.info("Deprecated --no_abs_log1p detected: using --abs_transform none.")
    abs_transform_effective = abs_transform_for_abs if args.mode in {"terminals_abs", "axon_abs"} else "none"
    if abs_transform_effective != "none":
        log.info(
            "Absolute-feature mode detected (%s): applying %s transform for k-evaluation.",
            args.mode, abs_transform_effective
        )
    if is_aitchison_metric(args.metric):
        log.info("Using Aitchison geometry (CLR transform + Euclidean distance).")
        if args.mode in {"terminals_abs", "axon_abs"}:
            log.warning(
                "Aitchison metric selected with absolute mode '%s'. This is valid mathematically, "
                "but primarily intended for compositional profiles (terminals/axon/density).",
                args.mode,
            )

    evaluate_optimal_k(
        df, feature_cols, range(2, args.k_max + 1), out_dir,
        mode=args.mode,
        method=args.method, metric=args.metric,
        abs_transform=abs_transform_effective,
        abs_transform_power=args.abs_transform_power,
        abs_transform_asinh_cofactor=args.abs_transform_asinh_cofactor,
    )

    log.info("Done. Outputs in: %s", out_dir)


if __name__ == "__main__":
    main()
