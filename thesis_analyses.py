#!/usr/bin/env python3
"""
Analysis functions for termination motif characterization.

Includes only:
  - source layer -> target layer profiles/statistics
  - bootstrap cluster stability
  - source layer enrichment per cluster
  - connection contribution/enrichment per cluster
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu

from constants import LAYER_ORDER, LAYER_LABELS, get_mode_value_label, mode_is_absolute
from utils import save_figure as _save, hcluster as _hcluster

log = logging.getLogger(__name__)


def _bh_fdr(pvals: list[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    pvals_sorted = np.array(pvals, dtype=float)[order]
    adjusted = np.minimum(1.0, pvals_sorted * n / np.arange(1, n + 1))
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    result = np.empty(n)
    result[order] = adjusted
    return result


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _attach_or_compute_clusters(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int,
    clustered_csv: Path | None,
    cluster_method: str,
    cluster_metric: str,
    n_clusters_by_direction: dict[str, int] | None,
    context: str,
) -> pd.DataFrame:
    """
    Return a copy of df with an integer 'cluster' column.

    Priority:
      1) Use pre-computed clusters from clustered_csv if available.
      2) Otherwise cluster per direction using feature_cols.
    """
    work = df.copy()

    if clustered_csv is not None and clustered_csv.exists():
        cdf = pd.read_csv(clustered_csv)
        if "cluster" in cdf.columns and "neuron_id" in cdf.columns:
            if "connection_name" in cdf.columns and "connection_name" in work.columns:
                merge_keys = ["neuron_id", "connection_name"]
            else:
                merge_keys = ["neuron_id"]
            if "cluster" in work.columns:
                work = work.drop(columns=["cluster"])
            work = work.merge(cdf[merge_keys + ["cluster"]], on=merge_keys, how="left")
            log.info("%s: using pre-computed clusters from %s (merge keys: %s)",
                     context, clustered_csv.name, merge_keys)
        else:
            log.warning(
                "%s: clustered CSV missing 'cluster' or 'neuron_id'; falling back to own clustering.",
                context,
            )
            clustered_csv = None

    if clustered_csv is None or "cluster" not in work.columns:
        clusters_all = []
        for direction in sorted(work["direction"].unique()):
            mask = work["direction"] == direction
            sub = work.loc[mask, feature_cols].dropna()
            k_dir = (
                n_clusters_by_direction.get(direction, n_clusters)
                if n_clusters_by_direction is not None
                else n_clusters
            )
            if k_dir is None or k_dir < 2:
                log.warning("%s: direction '%s' has invalid k=%s; skipping.", context, direction, k_dir)
                continue
            if len(sub) < k_dir:
                log.warning("%s: direction '%s' has n=%d < k=%d; skipping.",
                            context, direction, len(sub), k_dir)
                continue
            labels = _hcluster(sub.values, n_clusters=k_dir, method=cluster_method, metric=cluster_metric)
            clusters_all.append(pd.Series(labels, index=sub.index, name="cluster"))
        if not clusters_all:
            log.warning("%s: no valid direction groups for clustering.", context)
            return pd.DataFrame()
        work["cluster"] = pd.concat(clusters_all).astype(int)

    work = work.dropna(subset=["cluster"]).copy()
    work["cluster"] = work["cluster"].astype(int)
    return work


def analysis_source_to_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    mode: str,
) -> pd.DataFrame:
    """
    Show mean target-layer profiles grouped by source (soma) layer.
    Kruskal-Wallis test for each target layer across source layers.
    """
    src_order = ["1", "2/3", "4", "5", "6"]
    sub = df[df["source_layer"].isin(src_order)].copy()
    if len(sub) < 5:
        log.warning("Fewer than 5 neurons have a mapped source layer; skipping source->target analysis.")
        return pd.DataFrame()

    profile = sub.groupby("source_layer")[feature_cols].mean().reindex(src_order)

    rows = []
    for col, label in zip(feature_cols, LAYER_LABELS):
        groups = [(sl, sub[sub["source_layer"] == sl][col].dropna().to_numpy()) for sl in src_order]
        groups = [(sl, g) for sl, g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
        arrays = [g for _, g in groups]
        try:
            stat, p = kruskal(*arrays)
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append({"target_layer": label, "H_statistic": float(stat), "p_value": float(p)})

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        stats_df["p_adjusted"] = _bh_fdr(stats_df["p_value"].tolist())
        stats_df["significant"] = stats_df["p_adjusted"] < 0.05

    posthoc_rows = []
    if not stats_df.empty:
        sig_layers = stats_df[stats_df["significant"]]["target_layer"].tolist()
        sig_cols = [col for col, lbl in zip(feature_cols, LAYER_LABELS) if lbl in sig_layers]
        for col, label in zip(sig_cols, sig_layers):
            groups = [(sl, sub[sub["source_layer"] == sl][col].dropna().to_numpy()) for sl in src_order]
            groups = [(sl, g) for sl, g in groups if len(g) >= 3]
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    sl_a, g_a = groups[i]
                    sl_b, g_b = groups[j]
                    try:
                        u_stat, p_pw = mannwhitneyu(g_a, g_b, alternative="two-sided")
                    except ValueError:
                        u_stat, p_pw = np.nan, np.nan
                    posthoc_rows.append({
                        "target_layer": label,
                        "source_layer_A": f"L{sl_a}",
                        "source_layer_B": f"L{sl_b}",
                        "U_statistic": float(u_stat),
                        "p_value": float(p_pw),
                    })
    posthoc_df = pd.DataFrame(posthoc_rows)
    if not posthoc_df.empty:
        posthoc_df["p_adjusted"] = _bh_fdr(posthoc_df["p_value"].tolist())
        posthoc_df["significant"] = posthoc_df["p_adjusted"] < 0.05
        posthoc_df.to_csv(out_dir / "source_target_posthoc.csv", index=False)
        log.info("Saved: source_target_posthoc.csv (%d pairwise tests, %d significant)",
                 len(posthoc_df), int(posthoc_df["significant"].sum()))

    stats_df.to_csv(out_dir / "source_target_stats.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    hm_data = profile.copy()
    hm_data.columns = LAYER_LABELS
    hm_data.index = [f"L{s}" for s in src_order]
    vmax_hm = float(np.nanmax(hm_data.to_numpy())) if mode_is_absolute(mode) else 1.0
    if not np.isfinite(vmax_hm) or vmax_hm <= 0:
        vmax_hm = 1.0
    sns.heatmap(hm_data, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=vmax_hm, ax=axes[0])
    axes[0].set(
        title=f"Mean {get_mode_value_label(mode).lower()}: source -> target layer",
        xlabel="Target layer",
        ylabel="Source layer",
    )

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(src_order)))
    for i, sl in enumerate(src_order):
        if sl in profile.index and not profile.loc[sl].isna().all():
            axes[1].plot(
                LAYER_LABELS,
                profile.loc[sl, feature_cols].to_numpy(),
                marker="o",
                label=f"L{sl}",
                color=colors[i],
            )
    axes[1].set(
        xlabel="Target layer",
        ylabel=f"Mean {get_mode_value_label(mode).lower()}",
        title=f"Layer {get_mode_value_label(mode).lower()} profiles by source layer",
    )
    axes[1].legend(title="Source layer")
    axes[1].grid(True, alpha=0.4)
    fig.tight_layout()
    _save(fig, out_dir / "source_target_profiles.png")
    return stats_df


def analysis_bootstrap_stability(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int,
    n_bootstrap: int,
    out_dir: Path,
    method: str = "ward",
    metric: str = "euclidean",
    n_clusters_by_direction: dict[str, int] | None = None,
) -> None:
    """
    Compute co-clustering stability matrices via bootstrap resampling per direction.
    """
    rng = np.random.default_rng(seed=42)

    for direction, group in df.groupby("direction"):
        k_dir = (
            n_clusters_by_direction.get(direction, n_clusters)
            if n_clusters_by_direction is not None
            else n_clusters
        )
        if k_dir is None or k_dir < 2:
            log.warning("Group '%s': invalid n_clusters=%s — skipping stability analysis.", direction, k_dir)
            continue

        valid = group[feature_cols].dropna()
        n = len(valid)
        if n < k_dir + 1:
            log.warning("Group '%s': only %d valid neurons for k=%d — skipping stability analysis.",
                        direction, n, k_dir)
            continue

        X = valid.to_numpy(dtype=float)
        co_occur = np.zeros((n, n), dtype=np.float32)
        count_pairs = np.zeros((n, n), dtype=np.float32)

        log.info("Bootstrapping '%s': n=%d, k=%d, iterations=%d", direction, n, k_dir, n_bootstrap)

        for _ in range(n_bootstrap):
            boot_idx = rng.integers(0, n, size=n)
            unique_idx = np.unique(boot_idx)
            if len(unique_idx) < k_dir:
                continue

            X_boot = X[boot_idx]
            labels_boot_full = _hcluster(X_boot, n_clusters=k_dir, method=method, metric=metric)

            label_arr = np.full(n, -1, dtype=np.int32)
            for orig_pos, label in zip(boot_idx, labels_boot_full):
                if label_arr[orig_pos] == -1:
                    label_arr[orig_pos] = label

            in_boot = (label_arr >= 0)
            pair_mask = np.outer(in_boot, in_boot)
            count_pairs += pair_mask.astype(np.float32)
            same_cluster = (label_arr[:, None] == label_arr[None, :]) & pair_mask
            co_occur += same_cluster.astype(np.float32)

        with np.errstate(divide="ignore", invalid="ignore"):
            stability = np.where(count_pairs > 0, co_occur / count_pairs, 0.0)

        ids = valid.index.astype(str)
        pd.DataFrame(stability, index=ids, columns=ids).to_csv(out_dir / f"cluster_stability__{direction}.csv")

        full_labels = _hcluster(X, n_clusters=k_dir, method=method, metric=metric)
        order = np.argsort(full_labels)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(stability[np.ix_(order, order)], vmin=0, vmax=1, cmap="YlOrRd",
                       aspect="auto", interpolation="none")
        plt.colorbar(im, ax=ax, label="Co-clustering proportion")
        ax.set(
            title=(f"Bootstrap cluster stability — {direction}\n"
                   f"n={n}, {n_bootstrap} iterations, k={k_dir}"),
            xlabel="Neuron (sorted by cluster)",
            ylabel="Neuron (sorted by cluster)",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        _save(fig, out_dir / f"cluster_stability__{direction}.png")


def analysis_source_enrichment(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int,
    out_dir: Path,
    clustered_csv: Path | None = None,
    cluster_method: str = "ward",
    cluster_metric: str = "euclidean",
    n_clusters_by_direction: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Source-layer enrichment per (direction, cluster) via Fisher's exact test + BH-FDR.
    """
    from scipy.stats import fisher_exact

    src_order = ["1", "2/3", "4", "5", "6"]
    src_labels = ["L1", "L2/3", "L4", "L5", "L6"]

    if "source_layer" not in df.columns:
        log.warning("analysis_source_enrichment: 'source_layer' missing — skipping.")
        return pd.DataFrame()

    df = _attach_or_compute_clusters(
        df=df,
        feature_cols=feature_cols,
        n_clusters=n_clusters,
        clustered_csv=clustered_csv,
        cluster_method=cluster_method,
        cluster_metric=cluster_metric,
        n_clusters_by_direction=n_clusters_by_direction,
        context="analysis_source_enrichment",
    )
    if df.empty:
        log.warning(
            "analysis_source_enrichment: no valid rows with cluster labels. "
            "Saving empty source_enrichment.csv and skipping plot."
        )
        df.to_csv(out_dir / "source_enrichment.csv", index=False)
        return df

    rows: list[dict] = []
    directions = sorted(df["direction"].unique())
    for direction in directions:
        dsub = df[(df["direction"] == direction) & (df["source_layer"].isin(src_order))].copy()
        n_total = len(dsub)
        if n_total == 0:
            continue

        expected = dsub["source_layer"].value_counts()
        expected_frac = {sl: expected.get(sl, 0) / n_total for sl in src_order}

        for clust in sorted(dsub["cluster"].unique()):
            csub = dsub[dsub["cluster"] == clust]
            n_clust = len(csub)
            clust_counts = csub["source_layer"].value_counts()

            for sl, sl_label in zip(src_order, src_labels):
                obs_in = clust_counts.get(sl, 0)
                obs_out = n_clust - obs_in
                exp_frac = expected_frac[sl]
                n_sl = expected.get(sl, 0)
                not_in = n_sl - obs_in
                rest = n_total - n_clust - not_in

                table = [[obs_in, obs_out], [not_in, rest]]
                _, pval = fisher_exact(table, alternative="two-sided")

                obs_frac = obs_in / n_clust if n_clust > 0 else 0.0
                pseudo = 1 / (2 * n_total)
                log2e = np.log2((obs_frac + pseudo) / (exp_frac + pseudo))

                rows.append({
                    "direction": direction,
                    "cluster": clust,
                    "source_layer": sl_label,
                    "n_cluster": n_clust,
                    "observed_n": obs_in,
                    "observed_frac": round(obs_frac, 4),
                    "expected_frac": round(exp_frac, 4),
                    "log2_enrichment": round(log2e, 4),
                    "pvalue": pval,
                })

    result = pd.DataFrame(rows)
    if result.empty:
        log.warning(
            "analysis_source_enrichment: no valid (direction, cluster, source_layer) rows to test. "
            "Saving empty source_enrichment.csv and skipping plot."
        )
        result.to_csv(out_dir / "source_enrichment.csv", index=False)
        return result

    result["qvalue"] = _bh_fdr(result["pvalue"].tolist())
    result["sig"] = result["qvalue"].apply(_sig_label)
    result.to_csv(out_dir / "source_enrichment.csv", index=False)
    log.info("Saved: source_enrichment.csv (%d rows)", len(result))

    n_rows_max = int(result.groupby("direction")["cluster"].nunique().max())
    fig, axes = plt.subplots(1, len(directions), figsize=(5 * len(directions), 0.6 * n_rows_max + 1.5))
    if len(directions) == 1:
        axes = [axes]

    vmax = max(abs(result["log2_enrichment"].min()), abs(result["log2_enrichment"].max()), 1.0)

    for ax, direction in zip(axes, directions):
        dsub = result[result["direction"] == direction]
        if dsub.empty:
            ax.set_visible(False)
            continue

        pivot_val = dsub.pivot(index="cluster", columns="source_layer", values="log2_enrichment")
        pivot_sig = dsub.pivot(index="cluster", columns="source_layer", values="sig")
        pivot_val = pivot_val.reindex(columns=src_labels).fillna(0)
        pivot_sig = pivot_sig.reindex(columns=src_labels).fillna("ns")
        ns_mask = pivot_sig == "ns"
        pivot_display = pivot_val.copy()
        pivot_display[ns_mask] = 0.0

        sizes = dsub.drop_duplicates("cluster").set_index("cluster")["n_cluster"]
        pivot_display.index = [f"C{c} (n={sizes.get(c, '?')})" for c in pivot_display.index]

        sns.heatmap(
            pivot_display,
            annot=False,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            center=0,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "log2(observed / expected)", "shrink": 0.8},
        )

        for i in range(ns_mask.shape[0]):
            for j in range(ns_mask.shape[1]):
                if ns_mask.iloc[i, j]:
                    ax.add_patch(plt.Rectangle(
                        (j, i), 1, 1, fill=True, facecolor="#e0e0e0", alpha=0.85,
                        edgecolor="white", linewidth=0.5
                    ))

        for i, r in enumerate(pivot_val.index):
            for j, c in enumerate(pivot_val.columns):
                val = pivot_val.loc[r, c]
                sig = pivot_sig.loc[r, c]
                if sig == "ns":
                    txt = f"({val:.2f})"
                    color = "#888888"
                    fsize = 8
                else:
                    txt = f"{val:.2f}\n{sig}"
                    color = "black" if abs(val) < vmax * 0.7 else "white"
                    fsize = 9
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                        fontsize=fsize, color=color, fontweight="bold" if sig != "ns" else "normal")

        ax.set_title(f"Source layer enrichment — {direction}", fontsize=11)
        ax.set_xlabel("Source layer")
        ax.set_ylabel("Cluster")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    _save(fig, out_dir / "source_enrichment.png")
    return result


def analysis_connection_enrichment(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int,
    out_dir: Path,
    clustered_csv: Path | None = None,
    cluster_method: str = "ward",
    cluster_metric: str = "euclidean",
    n_clusters_by_direction: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Connection-level enrichment per (direction, cluster) via Fisher's exact test + BH-FDR.
    """
    from scipy.stats import fisher_exact

    if "connection_name" not in df.columns:
        log.warning("analysis_connection_enrichment: 'connection_name' missing — skipping.")
        return pd.DataFrame()

    df = _attach_or_compute_clusters(
        df=df,
        feature_cols=feature_cols,
        n_clusters=n_clusters,
        clustered_csv=clustered_csv,
        cluster_method=cluster_method,
        cluster_metric=cluster_metric,
        n_clusters_by_direction=n_clusters_by_direction,
        context="analysis_connection_enrichment",
    )
    if df.empty:
        log.warning(
            "analysis_connection_enrichment: no valid rows with cluster labels. "
            "Saving empty connection_enrichment.csv and skipping plot."
        )
        df.to_csv(out_dir / "connection_enrichment.csv", index=False)
        return df

    rows: list[dict] = []
    directions = sorted(df["direction"].unique())
    for direction in directions:
        dsub = df[(df["direction"] == direction) & (df["connection_name"].notna())].copy()
        n_total = len(dsub)
        if n_total == 0:
            continue

        expected = dsub["connection_name"].value_counts()
        conn_order = sorted(expected.index.tolist())
        expected_frac = {conn: expected.get(conn, 0) / n_total for conn in conn_order}

        for clust in sorted(dsub["cluster"].unique()):
            csub = dsub[dsub["cluster"] == clust]
            n_clust = len(csub)
            clust_counts = csub["connection_name"].value_counts()

            for conn in conn_order:
                obs_in = int(clust_counts.get(conn, 0))
                obs_out = int(n_clust - obs_in)
                exp_frac = float(expected_frac[conn])
                n_conn = int(expected.get(conn, 0))
                not_in = int(n_conn - obs_in)
                rest = int(n_total - n_clust - not_in)

                table = [[obs_in, obs_out], [not_in, rest]]
                _, pval = fisher_exact(table, alternative="two-sided")

                obs_frac = obs_in / n_clust if n_clust > 0 else 0.0
                pseudo = 1 / (2 * n_total)
                log2e = np.log2((obs_frac + pseudo) / (exp_frac + pseudo))

                rows.append({
                    "direction": direction,
                    "cluster": clust,
                    "connection_name": conn,
                    "n_cluster": n_clust,
                    "observed_n": obs_in,
                    "observed_frac": round(obs_frac, 4),
                    "expected_frac": round(exp_frac, 4),
                    "log2_enrichment": round(log2e, 4),
                    "pvalue": pval,
                })

    result = pd.DataFrame(rows)
    if result.empty:
        log.warning(
            "analysis_connection_enrichment: no valid (direction, cluster, connection) rows to test. "
            "Saving empty connection_enrichment.csv and skipping plot."
        )
        result.to_csv(out_dir / "connection_enrichment.csv", index=False)
        return result

    result["qvalue"] = _bh_fdr(result["pvalue"].tolist())
    result["sig"] = result["qvalue"].apply(_sig_label)
    result.to_csv(out_dir / "connection_enrichment.csv", index=False)
    log.info("Saved: connection_enrichment.csv (%d rows)", len(result))

    n_rows_max = int(result.groupby("direction")["cluster"].nunique().max())
    n_cols_max = int(result.groupby("direction")["connection_name"].nunique().max())
    fig_w = max(6.0, 0.8 * n_cols_max) * max(1, len(directions))
    fig_h = max(3.0, 0.6 * n_rows_max + 1.5)
    fig, axes = plt.subplots(1, len(directions), figsize=(fig_w, fig_h))
    if len(directions) == 1:
        axes = [axes]

    vmax = max(abs(result["log2_enrichment"].min()), abs(result["log2_enrichment"].max()), 1.0)

    for ax, direction in zip(axes, directions):
        dsub = result[result["direction"] == direction]
        if dsub.empty:
            ax.set_visible(False)
            continue

        conn_order = (
            dsub.groupby("connection_name")["expected_frac"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        pivot_val = dsub.pivot(index="cluster", columns="connection_name", values="log2_enrichment")
        pivot_sig = dsub.pivot(index="cluster", columns="connection_name", values="sig")
        pivot_val = pivot_val.reindex(columns=conn_order).fillna(0)
        pivot_sig = pivot_sig.reindex(columns=conn_order).fillna("ns")
        ns_mask = pivot_sig == "ns"
        pivot_display = pivot_val.copy()
        pivot_display[ns_mask] = 0.0

        sizes = dsub.drop_duplicates("cluster").set_index("cluster")["n_cluster"]
        pivot_display.index = [f"C{c} (n={sizes.get(c, '?')})" for c in pivot_display.index]

        sns.heatmap(
            pivot_display,
            annot=False,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            center=0,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "log2(observed / expected)", "shrink": 0.8},
        )

        for i in range(ns_mask.shape[0]):
            for j in range(ns_mask.shape[1]):
                if ns_mask.iloc[i, j]:
                    ax.add_patch(plt.Rectangle(
                        (j, i), 1, 1, fill=True, facecolor="#e0e0e0", alpha=0.85,
                        edgecolor="white", linewidth=0.5
                    ))

        for i, r in enumerate(pivot_val.index):
            for j, c in enumerate(pivot_val.columns):
                val = pivot_val.loc[r, c]
                sig = pivot_sig.loc[r, c]
                if sig == "ns":
                    txt = f"({val:.2f})"
                    color = "#888888"
                    fsize = 7
                else:
                    txt = f"{val:.2f}\n{sig}"
                    color = "black" if abs(val) < vmax * 0.7 else "white"
                    fsize = 8
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                        fontsize=fsize, color=color, fontweight="bold" if sig != "ns" else "normal")

        ax.set_title(f"Connection enrichment — {direction}", fontsize=11)
        ax.set_xlabel("Connection")
        ax.set_ylabel("Cluster")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    _save(fig, out_dir / "connection_enrichment.png")
    return result
