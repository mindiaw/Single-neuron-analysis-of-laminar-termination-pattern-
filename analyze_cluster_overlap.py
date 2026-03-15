#!/usr/bin/env python3
"""
Analyses of laminar termination motif overlap between FF and FB projections.

Three complementary approaches:

  1. Centroid distance heatmap  — pairwise distance between FF and FB centroids
                                  under the configured clustering geometry.
                                  Low distance = shared motif.

  2. Combined clustering        — all neurons clustered jointly (ignoring direction).
                                  Direction purity per combined cluster quantifies how many
                                  motifs are shared vs direction-specific.

  3. Cross-assignment           — each FB neuron assigned to its nearest FF centroid
                                  (and vice versa). The assignment matrix shows which
                                  direction-specific clusters have the closest cross-direction
                                  analogue under the configured clustering geometry,
                                  and the distance distribution shows how well
                                  neurons actually fit foreign prototypes.

Inputs:
  --in_csv        ALL_CONNECTIONS__mapped_and_terminals.csv  (raw, no cluster column)
  --clustered_dir Directory with centroid CSVs and clustered CSVs from
                  cluster_termination_patterns.py

Outputs (in --out_dir):
  centroid_distances__<mode>.csv
  centroid_matching__<mode>.csv
  centroid_distance_heatmap.png
  combined_clustering__<mode>.csv
  combined_clustering.png
  cross_assignment__<mode>.csv
  cross_assignment.png

Usage:
  python analyze_cluster_overlap.py \\
    --in_csv output/ALL_CONNECTIONS__mapped_and_terminals.csv \\
    --clustered_dir output/clustering \\
    --out_dir output/overlap \\
    --mode terminals \\
    --k_ff 5 --k_fb 9 \\
    --k_combined 14
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
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from constants import (
    LAYER_LABELS,
    DIRECTION_PALETTE,
    get_feature_cols,
    get_mode_value_label,
    mode_is_absolute,
)
from utils import (
    save_figure as _save,
    hcluster as _hcluster,
    normalize_density_features,
    apply_axon_length_threshold,
    apply_terminal_count_threshold_for_axon_abs,
    prepare_features_for_clustering,
)

log = logging.getLogger(__name__)


def _feat_cols(mode: str) -> list[str]:
    return get_feature_cols(mode)


def _prepare_for_distance_geometry(
    X: np.ndarray,
    cluster_method: str,
    cluster_metric: str,
    context: str,
) -> tuple[np.ndarray, str]:
    """
    Prepare vectors and effective metric so distance calculations match the
    clustering geometry used elsewhere in the pipeline.
    """
    method_l = str(cluster_method).lower()
    X_prepared, metric_eff = prepare_features_for_clustering(
        X, method=method_l, metric=cluster_metric, context=context
    )
    return X_prepared, metric_eff


def _distance_matrix(X_a: np.ndarray, X_b: np.ndarray, metric: str, context: str) -> np.ndarray:
    """
    Compute pairwise distances and guard against non-finite outputs.
    """
    D = cdist(X_a, X_b, metric=metric)
    finite_mask = np.isfinite(D)
    if not finite_mask.all():
        n_bad = int((~finite_mask).sum())
        log.warning("%s: %d non-finite pairwise distance value(s) replaced with +inf.", context, n_bad)
        D = np.where(finite_mask, D, np.inf)
        if np.isinf(D).all(axis=1).any():
            raise ValueError(f"{context}: at least one row has no finite distances.")
    return D


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1 — Centroid distance heatmap
# ─────────────────────────────────────────────────────────────────────────────

def _optimal_centroid_matching(D: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return optimal one-to-one matching indices and mean matched distance.

    Uses the Hungarian algorithm on the FF x FB distance matrix. For unequal
    numbers of clusters, it matches min(n_ff, n_fb) pairs.
    """
    if D.ndim != 2 or D.shape[0] == 0 or D.shape[1] == 0:
        raise ValueError("Distance matrix must be non-empty and 2D.")
    row_ind, col_ind = linear_sum_assignment(D)
    mean_cost = float(D[row_ind, col_ind].mean())
    return row_ind, col_ind, mean_cost


def analysis_centroid_distances(
    centroids_ff: pd.DataFrame,
    centroids_fb: pd.DataFrame,
    feat_cols: list[str],
    mode: str,
    out_dir: Path,
    cluster_method: str = "ward",
    cluster_metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Compute pairwise distances between FF and FB centroids using the same
    feature geometry as clustering (method + metric).

    Small distance = the two clusters have near-identical laminar profiles
    → shared motif.  Large distance = direction-specific motif.

    Outputs:
        centroid_distances__<mode>.csv
        centroid_matching__<mode>.csv
        centroid_distance_heatmap.png
    """
    ff_sorted = centroids_ff.sort_values("cluster")
    fb_sorted = centroids_fb.sort_values("cluster")

    X_ff_raw = ff_sorted[feat_cols].to_numpy(dtype=float)
    X_fb_raw = fb_sorted[feat_cols].to_numpy(dtype=float)
    X_ff, metric_eff = _prepare_for_distance_geometry(
        X_ff_raw, cluster_method=cluster_method, cluster_metric=cluster_metric,
        context="analysis_centroid_distances[FF]",
    )
    X_fb, metric_eff_fb = _prepare_for_distance_geometry(
        X_fb_raw, cluster_method=cluster_method, cluster_metric=cluster_metric,
        context="analysis_centroid_distances[FB]",
    )
    if metric_eff_fb != metric_eff:
        raise ValueError(
            f"Internal metric mismatch after preparation: FF={metric_eff}, FB={metric_eff_fb}"
        )

    ff_labels = [f"FF-C{c}" for c in ff_sorted["cluster"]]
    fb_labels = [f"FB-C{c}" for c in fb_sorted["cluster"]]

    D_geom = _distance_matrix(
        X_ff, X_fb, metric=metric_eff, context="analysis_centroid_distances[geometry]"
    )
    # Raw-reference distances (kept for interpretability and backward compatibility)
    D_euc_raw = _distance_matrix(
        X_ff_raw, X_fb_raw, metric="euclidean", context="analysis_centroid_distances[raw_euclidean]"
    )
    D_cos_raw = _distance_matrix(
        X_ff_raw, X_fb_raw, metric="cosine", context="analysis_centroid_distances[raw_cosine]"
    )

    # Save geometry-consistent distances
    dist_df = pd.DataFrame(D_geom, index=ff_labels, columns=fb_labels)
    dist_df.to_csv(out_dir / f"centroid_distances__{mode}.csv")
    pd.DataFrame(D_euc_raw, index=ff_labels, columns=fb_labels).to_csv(
        out_dir / f"centroid_distances_raw_euclidean__{mode}.csv"
    )

    # Save optimal one-to-one FF↔FB centroid matches
    row_ind, col_ind, observed_match_cost = _optimal_centroid_matching(D_geom)
    matched_rows = []
    for i, j in zip(row_ind, col_ind):
        ff_cluster = int(ff_sorted.iloc[i]["cluster"])
        fb_cluster = int(fb_sorted.iloc[j]["cluster"])
        matched_rows.append({
            "ff_cluster": ff_cluster,
            "fb_cluster": fb_cluster,
            "distance_metric": metric_eff,
            "distance": float(D_geom[i, j]),
            "euclidean_distance": float(D_euc_raw[i, j]),
            "cosine_distance": float(D_cos_raw[i, j]),
        })
    matching_df = pd.DataFrame(matched_rows).sort_values(["ff_cluster", "fb_cluster"])
    matching_df.to_csv(out_dir / f"centroid_matching__{mode}.csv", index=False)
    log.info(
        "Global FF↔FB centroid matching (%s): %d pairs, mean matched distance = %.3f",
        metric_eff, len(matching_df), observed_match_cost,
    )
    for row in matching_df.itertuples(index=False):
        log.info(
            "Matched centroid pair: FF-C%d ↔ FB-C%d  (%s=%.3f, raw-euclidean=%.3f, raw-cosine=%.3f)",
            row.ff_cluster, row.fb_cluster, metric_eff,
            row.distance, row.euclidean_distance, row.cosine_distance,
        )

    # Identify best match per FF cluster
    for i, ff_lbl in enumerate(ff_labels):
        best_j = int(np.argmin(D_geom[i]))
        log.info(
            "Centroid match: %s → %s  (%s=%.3f, raw-euclidean=%.3f, raw-cosine=%.3f)",
            ff_lbl, fb_labels[best_j], metric_eff,
            D_geom[i, best_j], D_euc_raw[i, best_j], D_cos_raw[i, best_j],
        )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    vmax_geom = float(np.nanmax(D_geom))
    if not np.isfinite(vmax_geom) or vmax_geom <= 0:
        vmax_geom = 1.0
    vmax_cos = float(np.nanmax(D_cos_raw))
    if not np.isfinite(vmax_cos) or vmax_cos <= 0:
        vmax_cos = 1.0

    for ax, D, label, vmax in [
        (axes[0], D_geom, f"{metric_eff} distance (clustering geometry)", vmax_geom),
        (axes[1], D_cos_raw, "Cosine distance (raw reference)", vmax_cos),
    ]:
        df_plot = pd.DataFrame(D, index=ff_labels, columns=fb_labels)
        sns.heatmap(
            df_plot, annot=True, fmt=".2f",
            cmap="YlOrRd_r",          # reversed: low distance = dark = similar
            vmin=0, vmax=vmax, ax=ax,
            cbar_kws={"label": label, "shrink": 0.85},
        )
        ax.set_xlabel("FB cluster")
        ax.set_ylabel("FF cluster")
        ax.set_title(f"{label}\n({get_mode_value_label(mode).lower()})")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    match_str = f"  |  Mean matched dist = {observed_match_cost:.3f}"
    fig.suptitle(
        f"FF vs FB cluster centroid distances ({metric_eff}){match_str}\n"
        "Dark cells = similar laminar profiles (potential shared motifs)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    _save(fig, out_dir / "centroid_distance_heatmap.png")
    return dist_df


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 2 — Combined clustering
# ─────────────────────────────────────────────────────────────────────────────

def analysis_combined_clustering(
    df: pd.DataFrame,
    feat_cols: list[str],
    k_combined: int,
    mode: str,
    out_dir: Path,
    cluster_method: str = "ward",
    cluster_metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Pool all neurons and cluster jointly, then examine each combined cluster's
    FF:FB composition (direction purity).

    - Purity = 1.0 → cluster is entirely one direction (direction-specific motif)
    - Purity = 0.5 → cluster is 50:50 FF/FB (shared motif)

    Outputs:
        combined_clustering__<mode>.csv
        combined_clustering.png
    """
    # Exclude non-FF/FB neurons for purity analysis
    df_ff_fb = df[df["direction"].isin(["FF", "FB"])].copy()
    valid = df_ff_fb[feat_cols + ["direction", "connection_name", "neuron_id"]].dropna().copy()
    X     = valid[feat_cols].to_numpy(dtype=float)

    log.info("Combined clustering: %d neurons (FF+FB only), k=%d", len(valid), k_combined)
    labels = _hcluster(
        X, k_combined, method=cluster_method, metric=cluster_metric
    )
    valid["combined_cluster"] = labels

    # --- Direction purity per cluster
    rows = []
    for c in sorted(valid["combined_cluster"].unique()):
        sub  = valid[valid["combined_cluster"] == c]
        n    = len(sub)
        n_ff = int((sub["direction"] == "FF").sum())
        n_fb = int((sub["direction"] == "FB").sum())
        purity   = max(n_ff, n_fb) / n
        dominant = "FF" if n_ff > n_fb else ("FB" if n_fb > n_ff else "Mixed")
        rows.append({
            "cluster": c, "n": n, "n_FF": n_ff, "n_FB": n_fb,
            "FF_pct": n_ff / n, "FB_pct": n_fb / n,
            "purity": purity, "dominant": dominant,
        })
    purity_df = pd.DataFrame(rows)
    purity_df.to_csv(out_dir / f"combined_clustering__{mode}.csv", index=False)

    # --- Centroids of combined clusters
    centroids = valid.groupby("combined_cluster")[feat_cols].mean().sort_index()
    centroids.columns = LAYER_LABELS

    # --- Plot: 3 panels
    clusters_sorted = purity_df["cluster"].tolist()
    xlabels = [f"C{c}\n(n={r.n})" for c, r in zip(purity_df["cluster"], purity_df.itertuples())]

    fig = plt.figure(figsize=(16, 5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[2, 1, 2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Panel 1: stacked bar — FF / FB composition
    ff_pcts = purity_df["FF_pct"].to_numpy()
    fb_pcts = purity_df["FB_pct"].to_numpy()
    ax1.bar(xlabels, ff_pcts, color=DIRECTION_PALETTE["FF"], label="FF", edgecolor="white", lw=0.4)
    ax1.bar(xlabels, fb_pcts, bottom=ff_pcts, color=DIRECTION_PALETTE["FB"], label="FB",
            edgecolor="white", lw=0.4)
    ax1.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
                label="50% (shared)")
    ax1.set(
        title="FF / FB composition per combined cluster",
        xlabel="Combined cluster", ylabel="Proportion",
        ylim=(0, 1),
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: direction purity bar
    colors_bar = [DIRECTION_PALETTE.get(r.dominant, "#aaaaaa") for r in purity_df.itertuples()]
    ax2.barh(xlabels, purity_df["purity"].to_numpy(), color=colors_bar,
             edgecolor="white", linewidth=0.4)
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set(
        title="Direction purity",
        xlabel="Purity  (1 = pure, 0.5 = shared)",
        xlim=(0, 1),
    )
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="x")

    # Panel 3: centroid heatmap
    centroids_plot = centroids.copy()
    centroids_plot.index = xlabels
    vmax_centroid = float(np.nanmax(centroids_plot.to_numpy())) if mode_is_absolute(mode) else 1.0
    if not np.isfinite(vmax_centroid) or vmax_centroid <= 0:
        vmax_centroid = 1.0
    sns.heatmap(
        centroids_plot, annot=True, fmt=".2f", cmap="YlOrRd",
        vmin=0, vmax=vmax_centroid, ax=ax3,
        cbar_kws={"label": f"Mean {get_mode_value_label(mode).lower()}",
                  "shrink": 0.85},
    )
    ax3.set_title("Combined cluster centroids")
    ax3.set_xlabel("Target layer")
    ax3.set_ylabel("Combined cluster")
    ax3.tick_params(axis="x", rotation=0)
    ax3.tick_params(axis="y", rotation=0)

    fig.suptitle(
        f"Combined clustering (k={k_combined}, {mode})\n"
        "Clusters with purity < 0.75 contain both directions (candidate shared motifs)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    _save(fig, out_dir / "combined_clustering.png")

    n_shared = int((purity_df["purity"] < 0.75).sum())
    log.info(
        "Combined clustering: %d/%d clusters have purity < 0.75 (potential shared motifs).",
        n_shared, k_combined,
    )
    return purity_df


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3 — Cross-assignment
# ─────────────────────────────────────────────────────────────────────────────

def analysis_cross_assignment(
    df_clustered: pd.DataFrame,
    centroids_ff: pd.DataFrame,
    centroids_fb: pd.DataFrame,
    feat_cols: list[str],
    mode: str,
    out_dir: Path,
    cluster_method: str = "ward",
    cluster_metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Assign each FB neuron to its nearest FF centroid (and vice versa).

    The assignment matrix (original cluster × assigned cross-direction cluster)
    shows which direction-specific clusters are most similar to each other.
    The distance distributions show how well neurons fit foreign prototypes:
    small distance → the motif is nearly direction-agnostic.

    Outputs:
        cross_assignment__<mode>.csv
        cross_assignment.png
    """
    ff_neurons = df_clustered[df_clustered["direction"] == "FF"].copy()
    fb_neurons = df_clustered[df_clustered["direction"] == "FB"].copy()

    ff_neurons = ff_neurons.dropna(subset=feat_cols)
    fb_neurons = fb_neurons.dropna(subset=feat_cols)
    ff_neurons["cluster"] = ff_neurons["cluster"].astype(int)
    fb_neurons["cluster"] = fb_neurons["cluster"].astype(int)

    C_ff = centroids_ff.sort_values("cluster")
    C_fb = centroids_fb.sort_values("cluster")
    ff_clust = C_ff["cluster"].tolist()
    fb_clust = C_fb["cluster"].tolist()
    X_C_ff_raw = C_ff[feat_cols].to_numpy(dtype=float)
    X_C_fb_raw = C_fb[feat_cols].to_numpy(dtype=float)
    X_C_ff, metric_eff = _prepare_for_distance_geometry(
        X_C_ff_raw, cluster_method=cluster_method, cluster_metric=cluster_metric,
        context="analysis_cross_assignment[centroids_ff]",
    )
    X_C_fb, metric_eff_fb = _prepare_for_distance_geometry(
        X_C_fb_raw, cluster_method=cluster_method, cluster_metric=cluster_metric,
        context="analysis_cross_assignment[centroids_fb]",
    )
    if metric_eff_fb != metric_eff:
        raise ValueError(
            f"Internal metric mismatch after preparation: FF={metric_eff}, FB={metric_eff_fb}"
        )

    # FB → nearest FF
    X_fb_raw = fb_neurons[feat_cols].to_numpy(dtype=float)
    X_fb, metric_eff_fb_neurons = _prepare_for_distance_geometry(
        X_fb_raw, cluster_method=cluster_method, cluster_metric=cluster_metric,
        context="analysis_cross_assignment[neurons_fb]",
    )
    if metric_eff_fb_neurons != metric_eff:
        raise ValueError(
            f"Internal metric mismatch after preparation: centroids={metric_eff}, FB-neurons={metric_eff_fb_neurons}"
        )
    D_fb_to_ff = _distance_matrix(
        X_fb, X_C_ff, metric=metric_eff, context="analysis_cross_assignment[fb_to_ff]"
    )
    nearest_ff_idx = D_fb_to_ff.argmin(axis=1)
    fb_neurons["nearest_FF_cluster"] = [ff_clust[i] for i in nearest_ff_idx]
    fb_neurons["dist_to_nearest_FF"] = D_fb_to_ff.min(axis=1)

    # FF → nearest FB
    X_ff_raw = ff_neurons[feat_cols].to_numpy(dtype=float)
    X_ff, metric_eff_ff_neurons = _prepare_for_distance_geometry(
        X_ff_raw, cluster_method=cluster_method, cluster_metric=cluster_metric,
        context="analysis_cross_assignment[neurons_ff]",
    )
    if metric_eff_ff_neurons != metric_eff:
        raise ValueError(
            f"Internal metric mismatch after preparation: centroids={metric_eff}, FF-neurons={metric_eff_ff_neurons}"
        )
    D_ff_to_fb = _distance_matrix(
        X_ff, X_C_fb, metric=metric_eff, context="analysis_cross_assignment[ff_to_fb]"
    )
    nearest_fb_idx = D_ff_to_fb.argmin(axis=1)
    ff_neurons["nearest_FB_cluster"] = [fb_clust[i] for i in nearest_fb_idx]
    ff_neurons["dist_to_nearest_FB"] = D_ff_to_fb.min(axis=1)

    # Unify column names for a clean (non-sparse) output
    fb_out = fb_neurons[["neuron_id", "direction", "cluster", "nearest_FF_cluster", "dist_to_nearest_FF"]].copy()
    fb_out = fb_out.rename(columns={"nearest_FF_cluster": "nearest_cross_cluster",
                                     "dist_to_nearest_FF": "dist_to_nearest_cross"})
    ff_out = ff_neurons[["neuron_id", "direction", "cluster", "nearest_FB_cluster", "dist_to_nearest_FB"]].copy()
    ff_out = ff_out.rename(columns={"nearest_FB_cluster": "nearest_cross_cluster",
                                     "dist_to_nearest_FB": "dist_to_nearest_cross"})
    result = pd.concat([fb_out, ff_out], ignore_index=True)
    result["distance_metric"] = metric_eff
    result.to_csv(out_dir / f"cross_assignment__{mode}.csv", index=False)

    # --- Assignment matrices (counts)
    mat_fb_to_ff = (
        fb_neurons.groupby(["cluster", "nearest_FF_cluster"]).size()
        .unstack(fill_value=0)
        .reindex(index=sorted(fb_clust), columns=sorted(ff_clust), fill_value=0)
    )
    mat_ff_to_fb = (
        ff_neurons.groupby(["cluster", "nearest_FB_cluster"]).size()
        .unstack(fill_value=0)
        .reindex(index=sorted(ff_clust), columns=sorted(fb_clust), fill_value=0)
    )
    # Normalise rows to proportions
    mat_fb_to_ff_pct = mat_fb_to_ff.div(mat_fb_to_ff.sum(axis=1), axis=0).fillna(0)
    mat_ff_to_fb_pct = mat_ff_to_fb.div(mat_ff_to_fb.sum(axis=1), axis=0).fillna(0)

    mat_fb_to_ff_pct.index   = [f"FB-C{i}" for i in mat_fb_to_ff_pct.index]
    mat_fb_to_ff_pct.columns = [f"FF-C{i}" for i in mat_fb_to_ff_pct.columns]
    mat_ff_to_fb_pct.index   = [f"FF-C{i}" for i in mat_ff_to_fb_pct.index]
    mat_ff_to_fb_pct.columns = [f"FB-C{i}" for i in mat_ff_to_fb_pct.columns]

    # --- Plot: 4 panels (2 assignment matrices + 2 distance distributions)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Top-left: FB original cluster → nearest FF cluster
    sns.heatmap(
        mat_fb_to_ff_pct, annot=True, fmt=".2f", cmap="Blues",
        vmin=0, vmax=1, ax=axes[0][0],
        cbar_kws={"label": "Proportion of FB cluster", "shrink": 0.85},
    )
    axes[0][0].set_title("FB clusters → nearest FF centroid\n(rows = original FB cluster)")
    axes[0][0].set_xlabel("Nearest FF cluster")
    axes[0][0].set_ylabel("FB cluster (original)")
    axes[0][0].tick_params(axis="x", rotation=0)
    axes[0][0].tick_params(axis="y", rotation=0)

    # Top-right: FF original cluster → nearest FB cluster
    sns.heatmap(
        mat_ff_to_fb_pct, annot=True, fmt=".2f", cmap="Oranges",
        vmin=0, vmax=1, ax=axes[0][1],
        cbar_kws={"label": "Proportion of FF cluster", "shrink": 0.85},
    )
    axes[0][1].set_title("FF clusters → nearest FB centroid\n(rows = original FF cluster)")
    axes[0][1].set_xlabel("Nearest FB cluster")
    axes[0][1].set_ylabel("FF cluster (original)")
    axes[0][1].tick_params(axis="x", rotation=0)
    axes[0][1].tick_params(axis="y", rotation=0)

    # Bottom-left: distance distribution — FB neurons to nearest FF centroid
    fb_plot = fb_neurons[["cluster", "dist_to_nearest_FF"]].copy()
    fb_plot["cluster"] = fb_plot["cluster"].apply(lambda x: f"FB-C{x}")
    clusters_fb = sorted(fb_plot["cluster"].unique())
    sns.violinplot(
        data=fb_plot, x="cluster", y="dist_to_nearest_FF",
        order=clusters_fb, color=DIRECTION_PALETTE["FB"],
        inner="quartile", linewidth=0.8, ax=axes[1][0],
    )
    axes[1][0].set(
        title="Distance of FB neurons to nearest FF centroid",
        xlabel="FB cluster (original)", ylabel=f"{metric_eff} distance to nearest FF centroid",
    )
    axes[1][0].grid(True, alpha=0.3, axis="y")

    # Bottom-right: distance distribution — FF neurons to nearest FB centroid
    ff_plot = ff_neurons[["cluster", "dist_to_nearest_FB"]].copy()
    ff_plot["cluster"] = ff_plot["cluster"].apply(lambda x: f"FF-C{x}")
    clusters_ff = sorted(ff_plot["cluster"].unique())
    sns.violinplot(
        data=ff_plot, x="cluster", y="dist_to_nearest_FB",
        order=clusters_ff, color=DIRECTION_PALETTE["FF"],
        inner="quartile", linewidth=0.8, ax=axes[1][1],
    )
    axes[1][1].set(
        title="Distance of FF neurons to nearest FB centroid",
        xlabel="FF cluster (original)", ylabel=f"{metric_eff} distance to nearest FB centroid",
    )
    axes[1][1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Cross-direction cluster assignment ({mode}, metric={metric_eff})\n"
        "Diagonal assignment matrices + small distances are consistent with shared motifs",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    _save(fig, out_dir / "cross_assignment.png")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Analyse laminar termination motif overlap between FF and FB projections."
    )
    ap.add_argument("--in_csv", required=True,
                    help="Path to ALL_CONNECTIONS__mapped_and_terminals.csv")
    ap.add_argument("--clustered_dir", required=True,
                    help="Directory with centroid CSVs and clustered CSVs from "
                         "cluster_termination_patterns.py.")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for figures and CSVs.")
    ap.add_argument("--mode", choices=["terminals", "axon", "density", "terminals_abs", "axon_abs"], default="terminals",
                    help="Feature set to analyse (default: terminals).")
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
    ap.add_argument("--scope", choices=["direction", "connection_direction"], default="direction",
                    help="Clustering scope used in step 3 (default: direction).")
    ap.add_argument("--k_ff", type=int, default=None,
                    help="Expected number of FF clusters from step 3 (scope=direction).")
    ap.add_argument("--k_fb", type=int, default=None,
                    help="Expected number of FB clusters from step 3 (scope=direction).")
    ap.add_argument("--k_separate", type=int, default=None,
                    help="Deprecated alias: sets both --k_ff and --k_fb to the same value.")
    ap.add_argument("--k_combined", type=int, default=10,
                    help="k for the combined (pooled) clustering (default: 10). "
                         "Use k_ff + k_fb as a natural starting point.")
    ap.add_argument("--cluster_method", default="ward",
                    help="Hierarchical linkage method for overlap analyses that recluster.")
    ap.add_argument("--cluster_metric", default="euclidean",
                    help=(
                        "Distance metric for overlap analyses that recluster. "
                        "For method='ward', supports euclidean, correlation/pearson, or aitchison/clr."
                    ))
    ap.add_argument("--n_permutations", type=int, default=1000,
                    help="Deprecated (unused): retained for backward compatibility.")
    ap.add_argument("--log_level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    in_csv        = Path(args.in_csv)
    clustered_dir = Path(args.clustered_dir)
    out_dir       = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Backward-compatible handling for deprecated --k_separate.
    if args.k_separate is not None:
        if args.k_ff is None:
            args.k_ff = args.k_separate
        if args.k_fb is None:
            args.k_fb = args.k_separate
        if args.k_ff != args.k_separate or args.k_fb != args.k_separate:
            log.warning(
                "--k_separate is deprecated and ignored because explicit --k_ff/--k_fb were provided."
            )
    if args.n_permutations is not None:
        log.info("Permutation testing is disabled; --n_permutations is ignored.")

    feat_cols = _feat_cols(args.mode)
    log.info("Clustering geometry for overlap analyses: method=%s, metric=%s",
             args.cluster_method, args.cluster_metric)

    # --- Load raw data
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")
    df = pd.read_csv(in_csv)
    log.info("Loaded raw data: %d neurons", len(df))

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for mode='{args.mode}': {missing}")

    if args.mode == "density":
        df = normalize_density_features(df, feat_cols)
    df = apply_axon_length_threshold(
        df, args.mode, min_target_axon_length_um=args.min_target_axon_length_um
    )
    df = apply_terminal_count_threshold_for_axon_abs(
        df,
        args.mode,
        min_target_terminals_for_axon_abs=args.min_target_terminals_for_axon_abs,
    )

    # --- Load centroid CSVs
    def _load_centroids(direction: str) -> pd.DataFrame:
        if args.scope == "direction":
            path = clustered_dir / f"centroids__{args.mode}__{direction}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Centroid file not found: {path}")
            cent = pd.read_csv(path)
            if "cluster" not in cent.columns and "Unnamed: 0" in cent.columns:
                cent = cent.rename(columns={"Unnamed: 0": "cluster"})
            if "cluster" not in cent.columns:
                raise ValueError(f"Centroid file missing 'cluster' column: {path}")
            cent["cluster"] = pd.to_numeric(cent["cluster"], errors="raise").astype(int)
            log.info("Loaded %s (%d clusters)", path.name, len(cent))
            return cent
        else:
            # connection_direction scope: aggregate per-connection centroids
            import glob as globmod
            pattern = str(clustered_dir / f"centroids__{args.mode}__*__{direction}.csv")
            paths = sorted(globmod.glob(pattern))
            if not paths:
                raise FileNotFoundError(
                    f"No centroid files found matching: {pattern}"
                )
            parts = []
            for p in paths:
                cent = pd.read_csv(p)
                if "cluster" not in cent.columns and "Unnamed: 0" in cent.columns:
                    cent = cent.rename(columns={"Unnamed: 0": "cluster"})
                if "cluster" not in cent.columns:
                    raise ValueError(f"Centroid file missing 'cluster' column: {p}")
                cent["cluster"] = pd.to_numeric(cent["cluster"], errors="raise").astype(int)
                parts.append(cent)
            combined = pd.concat(parts, ignore_index=True)
            if combined["cluster"].duplicated().any():
                dup_n = int(combined["cluster"].duplicated().sum())
                raise ValueError(
                    "Duplicate centroid cluster ids detected while aggregating "
                    f"connection_direction centroids ({dup_n} duplicates). "
                    "Re-run step 4 with a version that assigns globally unique cluster ids."
                )
            log.info("Aggregated %d centroid files for %s (%d centroids total)",
                     len(paths), direction, len(combined))
            return combined

    centroids_ff = _load_centroids("FF")
    centroids_fb = _load_centroids("FB")

    # Validate centroid feature columns
    for cent, direction in [(centroids_ff, "FF"), (centroids_fb, "FB")]:
        missing_c = [c for c in feat_cols if c not in cent.columns]
        if missing_c:
            raise ValueError(
                f"Centroid file ({direction}) missing feature columns: {missing_c}"
            )

    if args.scope == "direction":
        if args.k_ff is not None and len(centroids_ff) != args.k_ff:
            raise ValueError(
                f"FF centroid count mismatch: expected k_ff={args.k_ff}, found {len(centroids_ff)}."
            )
        if args.k_fb is not None and len(centroids_fb) != args.k_fb:
            raise ValueError(
                f"FB centroid count mismatch: expected k_fb={args.k_fb}, found {len(centroids_fb)}."
            )
        log.info("Validated centroid counts: FF=%d, FB=%d", len(centroids_ff), len(centroids_fb))
    elif args.k_ff is not None or args.k_fb is not None:
        log.info(
            "scope=connection_direction: skipping strict k_ff/k_fb validation "
            "because total centroids depend on the number of connection groups."
        )

    # --- Load clustered CSV (for cross-assignment, needs cluster column)
    clustered_path = clustered_dir / f"ALL_CONNECTIONS__clustered__{args.mode}__{args.scope}.csv"
    if not clustered_path.exists():
        raise FileNotFoundError(f"Clustered CSV not found: {clustered_path}")
    df_clustered = pd.read_csv(clustered_path)
    df_clustered = apply_axon_length_threshold(
        df_clustered, args.mode, min_target_axon_length_um=args.min_target_axon_length_um
    )
    df_clustered = apply_terminal_count_threshold_for_axon_abs(
        df_clustered,
        args.mode,
        min_target_terminals_for_axon_abs=args.min_target_terminals_for_axon_abs,
    )
    log.info("Loaded clustered data: %d neurons", len(df_clustered))

    # --- Run analyses
    log.info("── Analysis 1: Centroid distance heatmap")
    analysis_centroid_distances(
        centroids_ff,
        centroids_fb,
        feat_cols,
        args.mode,
        out_dir,
        cluster_method=args.cluster_method,
        cluster_metric=args.cluster_metric,
    )

    log.info("── Analysis 2: Combined clustering (k=%d)", args.k_combined)
    analysis_combined_clustering(df, feat_cols, args.k_combined, args.mode, out_dir,
                                 cluster_method=args.cluster_method,
                                 cluster_metric=args.cluster_metric)

    log.info("── Analysis 3: Cross-assignment")
    analysis_cross_assignment(
        df_clustered,
        centroids_ff,
        centroids_fb,
        feat_cols,
        args.mode,
        out_dir,
        cluster_method=args.cluster_method,
        cluster_metric=args.cluster_metric,
    )

    log.info("All overlap analyses complete. Outputs in: %s", out_dir)


if __name__ == "__main__":
    main()
