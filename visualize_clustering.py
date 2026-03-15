#!/usr/bin/env python3
"""
Visualizations for hierarchical clustering of neuron termination patterns.

Operates on outputs from cluster_termination_patterns.py:
  - ALL_CONNECTIONS__clustered__<mode>__<scope>.csv
  - centroids__<mode>__<group_tag>.csv

Outputs (in --out_dir):
  cluster_centroids.png       — Mean per-layer values per cluster (heatmaps)
  cluster_profiles__<mode>.png — Within-cluster per-layer value distributions
  cluster_composition.png     — Connection → cluster stacked bar charts
  cluster_source_layers.png   — Source soma layer distribution per cluster
  single_neuron_heatmap__<mode>__<group>.png — Single-neuron laminar heatmaps

Usage:
  python visualize_clustering.py \\
    --clustered_dir output/clustering \\
    --out_dir output/clustering
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns

from constants import (
    LAYER_LABELS,
    DIRECTION_PALETTE,
    LAYER_PALETTE,
    get_feature_cols,
    get_terminal_count_cols,
    get_axon_length_cols,
    get_mode_value_label,
    mode_is_absolute,
)
from utils import (
    save_figure as _save,
    transform_nonnegative,
    ABS_TRANSFORM_CHOICES,
    DEFAULT_ABS_TRANSFORM_POWER,
    DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Centroid heatmaps (all requested modes, FF + FB)
# ─────────────────────────────────────────────────────────────────────────────

def plot_centroids(
    centroids: dict[str, pd.DataFrame],
    cluster_sizes: dict[str, dict[int, int]],
    out_dir: Path,
    abs_display_transform: str = "none",
    abs_display_transform_power: float = DEFAULT_ABS_TRANSFORM_POWER,
    abs_display_transform_asinh_cofactor: float = DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR,
    centroid_annotate_abs: bool = False,
    centroid_abs_overlays: dict[str, pd.DataFrame] | None = None,
    centroid_fraction_diverging: bool = True,
) -> None:
    """
    Grid of centroid heatmaps with one row per available mode and two columns
    (FF, FB).

    Cell values are mean per-layer values:
      - fractions for terminals / axon
      - normalized densities for density
      - counts / lengths for absolute modes

    Optional overlay:
      - for terminals/axon, annotate each cell as:
          fraction\n(mean absolute layer value)
        using mean tL* / aL* from clustered rows.
    """
    # Determine which modes are present from the centroid keys
    available_modes = sorted({k.split("__")[0] for k in centroids})
    directions = ["FF", "FB"]
    n_modes = len(available_modes)
    panels = [(m, d) for m in available_modes for d in directions]
    fig, axes = plt.subplots(n_modes, 2, figsize=(12, 4.5 * n_modes), squeeze=False)

    for (mode, direction), ax in zip(panels, axes.flat):
        key = f"{mode}__{direction}"
        df  = centroids.get(key)
        if df is None or df.empty:
            ax.set_visible(False)
            continue

        feat_cols = get_feature_cols(mode)
        present   = [c for c in feat_cols if c in df.columns]

        data = (df.set_index("cluster")[present].copy()
                if "cluster" in df.columns else df[present].copy())
        cluster_ids_raw = data.index.tolist()
        cluster_ids: list[int | str] = []
        for cid in cluster_ids_raw:
            try:
                cluster_ids.append(int(cid))
            except (TypeError, ValueError):
                cluster_ids.append(cid)  # fallback for non-integer labels
        data.columns = LAYER_LABELS[:len(present)]
        sizes = cluster_sizes.get(key, {})
        data.index = [f"C{i} (n={sizes.get(int(i), '?')})" if isinstance(i, (int, np.integer)) else f"C{i} (n=?)"
                      for i in cluster_ids]

        abs_display_transform_l = str(abs_display_transform).lower()
        use_abs_display_transform = mode_is_absolute(mode) and (abs_display_transform_l != "none")
        if use_abs_display_transform:
            data_plot = pd.DataFrame(
                transform_nonnegative(
                    data.to_numpy(dtype=float),
                    transform=abs_display_transform_l,
                    context=f"plot_centroids[{mode}__{direction}]",
                    power=abs_display_transform_power,
                    asinh_cofactor=abs_display_transform_asinh_cofactor,
                ),
                index=data.index,
                columns=data.columns,
            )
        else:
            data_plot = data

        vmax = float(np.nanmax(data_plot.to_numpy())) if mode_is_absolute(mode) else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        value_label = get_mode_value_label(mode).lower()
        if use_abs_display_transform:
            value_label = f"{abs_display_transform_l}({value_label})"
        annot_data = True
        fmt = ".2f"
        title_extra = ""
        if (
            centroid_annotate_abs
            and mode in {"terminals", "axon"}
            and centroid_abs_overlays is not None
            and key in centroid_abs_overlays
        ):
            abs_df = centroid_abs_overlays[key].copy()
            abs_df = abs_df.reindex(index=cluster_ids, columns=LAYER_LABELS[:len(present)])
            if abs_df.notna().any().any():
                annot_mat = np.empty(data_plot.shape, dtype=object)
                for i in range(data_plot.shape[0]):
                    for j in range(data_plot.shape[1]):
                        frac_v = float(data.iloc[i, j])
                        abs_v = abs_df.iloc[i, j]
                        if pd.isna(abs_v):
                            annot_mat[i, j] = f"{frac_v:.2f}"
                        else:
                            annot_mat[i, j] = f"{frac_v:.2f}\n({float(abs_v):.0f})"
                annot_data = annot_mat
                fmt = ""
                title_extra = " [fraction + mean abs]"

        use_centroid_fraction_diverging = centroid_fraction_diverging and mode in {"terminals", "axon"}
        heatmap_kwargs = {
            "annot": annot_data,
            "fmt": fmt,
            "ax": ax,
            "cbar_kws": {"label": f"Mean {value_label}", "shrink": 0.85},
        }
        if use_centroid_fraction_diverging:
            heatmap_kwargs["cmap"] = "bwr"
            heatmap_kwargs["norm"] = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
        else:
            heatmap_kwargs["cmap"] = "YlOrRd"
            heatmap_kwargs["vmin"] = 0
            heatmap_kwargs["vmax"] = vmax
        sns.heatmap(data_plot, **heatmap_kwargs)
        title_prefix = {"terminals": "Terminals", "axon": "Axon", "density": "Density"}
        title_suffix = (
            f" (display: {abs_display_transform_l})" if use_abs_display_transform else ""
        )
        ax.set_title(f"{title_prefix.get(mode, mode)} — {direction}{title_suffix}{title_extra}", fontsize=11)
        ax.set_xlabel("Target layer")
        ax.set_ylabel("Cluster")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Cluster centroids: mean per-layer values", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "cluster_centroids.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Within-cluster per-layer value distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_profiles(df: pd.DataFrame, mode: str, out_dir: Path) -> None:
    """
    For FF and FB separately, a grouped boxplot showing the spread of each
    layer value within each cluster.

    x-axis = cluster, hue = layer, y-axis = feature value.
    This reveals whether clusters are tight (distinctive) or diffuse.
    """
    feat_cols = get_feature_cols(mode)
    missing   = [c for c in feat_cols if c not in df.columns]
    if missing:
        log.warning("plot_cluster_profiles: missing columns %s — skipping.", missing)
        return

    long = (
        df[["cluster", "direction"] + feat_cols]
        .melt(id_vars=["cluster", "direction"], var_name="layer_col", value_name="fraction")
        .assign(layer=lambda d: d["layer_col"].map(dict(zip(feat_cols, LAYER_LABELS))))
        .dropna(subset=["fraction"])
    )
    long["cluster"] = long["cluster"].astype(int)

    directions      = sorted(long["direction"].unique())
    fig, axes       = plt.subplots(len(directions), 1,
                                   figsize=(max(9, 2 * long["cluster"].nunique()), 5 * len(directions)),
                                   squeeze=False)

    for ax, direction in zip(axes[:, 0], directions):
        sub             = long[long["direction"] == direction]
        clusters_sorted = sorted(sub["cluster"].unique())
        sizes           = (df[df["direction"] == direction]
                           .groupby(df["cluster"].astype(int)).size().to_dict())

        sns.boxplot(
            data=sub, x="cluster", y="fraction", hue="layer",
            order=clusters_sorted, hue_order=LAYER_LABELS,
            palette=LAYER_PALETTE, ax=ax,
            linewidth=0.8, fliersize=2,
        )
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([f"C{c}\n(n={sizes.get(c, '?')})" for c in clusters_sorted])
        ax.set(
            title=f"{direction}: layer {get_mode_value_label(mode).lower()} distribution per cluster",
            xlabel="Cluster", ylabel=get_mode_value_label(mode),
        )
        if not mode_is_absolute(mode):
            ax.set_ylim(-0.05, 1.05)
        ax.legend(title="Layer", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    mode_suptitles = {
        "terminals": "terminal fraction",
        "axon": "axon fraction",
        "density": "normalised terminal density",
        "terminals_abs": "terminal count",
        "axon_abs": "axon length",
    }
    fig.suptitle(f"Within-cluster {mode_suptitles.get(mode, mode)} distributions", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / f"cluster_profiles__{mode}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Connection composition per cluster (stacked bars)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_composition(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Stacked bar charts: for each cluster, what fraction of neurons come from
    each connection? FF and FB shown side by side.

    Reveals whether clusters are connection-specific or connection-agnostic.
    """
    if "connection_name" not in df.columns:
        log.warning("plot_cluster_composition: 'connection_name' column missing — skipping.")
        return

    directions = sorted(df["direction"].unique())
    fig, axes  = plt.subplots(1, len(directions), figsize=(7 * len(directions), 5))
    if len(directions) == 1:
        axes = [axes]

    for ax, direction in zip(axes, directions):
        grp = df[df["direction"] == direction].copy()
        if grp.empty:
            ax.set_visible(False)
            continue

        grp["cluster"] = grp["cluster"].astype(int)
        ct = (
            grp.groupby(["cluster", "connection_name"]).size()
            .unstack(fill_value=0)
        )
        ct_pct      = ct.div(ct.sum(axis=1), axis=0)
        connections = ct_pct.columns.tolist()
        colors      = sns.color_palette("tab10", len(connections))

        bottom = np.zeros(len(ct_pct))
        clusters_sorted = sorted(ct_pct.index.tolist())
        ct_pct  = ct_pct.loc[clusters_sorted]
        sizes   = ct.loc[clusters_sorted].sum(axis=1).to_dict()
        xlabels = [f"C{c}\n(n={sizes.get(c, '?')})" for c in clusters_sorted]

        for conn, color in zip(connections, colors):
            vals = ct_pct[conn].to_numpy()
            ax.bar(
                xlabels, vals,
                bottom=bottom, color=color, label=conn,
                edgecolor="white", linewidth=0.4,
            )
            bottom += vals

        ax.set(
            title=f"Connection composition per cluster — {direction}",
            xlabel="Cluster", ylabel="Proportion of neurons",
            ylim=(0, 1),
        )
        ax.legend(title="Connection", fontsize=7,
                  bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, out_dir / "cluster_composition.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Source soma layer distribution per cluster
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_source_layers(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Heatmap of source (soma) layer × cluster for FF and FB.
    Cell values = fraction of neurons in that cluster from each source layer.

    Answers: do neurons originating from specific cortical depths cluster together?
    """
    if "source_layer" not in df.columns:
        log.warning("plot_cluster_source_layers: 'source_layer' column missing — skipping.")
        return

    src_order  = ["1", "2/3", "4", "5", "6"]
    src_labels = ["L1", "L2/3", "L4", "L5", "L6"]
    directions = sorted(df["direction"].unique())

    fig, axes = plt.subplots(1, len(directions), figsize=(6 * len(directions), 5))
    if len(directions) == 1:
        axes = [axes]

    for ax, direction in zip(axes, directions):
        grp = df[(df["direction"] == direction) &
                 (df["source_layer"].isin(src_order))].copy()
        if grp.empty:
            ax.set_visible(False)
            continue

        grp["cluster"] = grp["cluster"].astype(int)
        ct = (
            grp.groupby(["cluster", "source_layer"]).size()
            .unstack(fill_value=0)
            .reindex(columns=src_order, fill_value=0)
        )
        sizes          = ct.sum(axis=1).to_dict()
        ct_pct         = ct.div(ct.sum(axis=1), axis=0).fillna(0)
        ct_pct.columns = src_labels
        ct_pct.index   = [f"C{i} (n={sizes.get(i, '?')})" for i in ct_pct.index]

        sns.heatmap(
            ct_pct, annot=True, fmt=".2f", cmap="Blues",
            vmin=0, vmax=1, ax=ax,
            cbar_kws={"label": "Fraction of cluster neurons", "shrink": 0.85},
        )
        ax.set_title(f"Source layer distribution per cluster — {direction}", fontsize=11)
        ax.set_xlabel("Source (soma) layer")
        ax.set_ylabel("Cluster")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    _save(fig, out_dir / "cluster_source_layers.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Single-neuron laminar heatmap
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_magnitude_spec(df: pd.DataFrame, mode: str) -> tuple[str, str] | None:
    """Return (column, label) for a per-neuron magnitude strip if available."""
    if mode in {"terminals", "terminals_abs"} and "tTotal" in df.columns:
        return "tTotal", "Total terminals"
    if mode in {"axon", "axon_abs"} and "aTotal" in df.columns:
        return "aTotal", "Total axon length (um)"
    if mode == "density":
        if "tTotal" in df.columns:
            return "tTotal", "Total terminals"
        if "aTotal" in df.columns:
            return "aTotal", "Total axon length (um)"
    return None


def _safe_filename_component(text: str) -> str:
    """Return a filesystem-safe slug for figure filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")


def plot_single_neuron_heatmap(
    df: pd.DataFrame,
    mode: str,
    out_dir: Path,
    abs_display_transform: str = "none",
    abs_display_transform_power: float = DEFAULT_ABS_TRANSFORM_POWER,
    abs_display_transform_asinh_cofactor: float = DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR,
    shape_plus_magnitude: bool = False,
    by_connection: bool = False,
    order_within_cluster_by_ttotal: bool = False,
    single_neuron_fraction_diverging: bool = True,
) -> None:
    """
    Figure per direction (FF / FB):

    Top panel  — transposed heatmap: rows = target layers (L1 → L6),
                 columns = individual neurons grouped by cluster.
                 If shape_plus_magnitude=True, this panel shows per-neuron
                 normalized layer fractions (shape-only view).

    Middle panel (optional) — per-neuron magnitude strip aligned to the same
                              neuron columns.

    Bottom panel — stacked bar chart showing source (soma) layer composition
                   per cluster, aligned to the same x-axis so cluster
                   boundaries match exactly.

    Separate figures for FF and FB.
    If by_connection=True, split further by connection_name.
    """
    feat_cols = get_feature_cols(mode)
    missing   = [c for c in feat_cols if c not in df.columns]
    if missing:
        log.warning("plot_single_neuron_heatmap: missing columns %s — skipping.", missing)
        return

    src_order  = ["1", "2/3", "4", "5", "6"]
    src_labels = ["L1", "L2/3", "L4", "L5", "L6"]
    # Ordered cool-to-warm palette to stay visually compatible with the
    # diverging single-neuron heatmap while remaining categorical.
    src_palette = ["#2166ac", "#67a9cf", "#d9d9d9", "#e7a1b0", "#b2182b"]

    groups: list[tuple[str, str, pd.DataFrame]] = []
    if by_connection:
        if "connection_name" not in df.columns:
            log.warning(
                "plot_single_neuron_heatmap: 'connection_name' column missing — "
                "cannot split by connection."
            )
            return
        for (direction, connection_name), sub in (
            df.groupby(["direction", "connection_name"], sort=True)
        ):
            if sub.empty:
                continue
            file_tag = (
                f"{direction}__{_safe_filename_component(connection_name)}"
            )
            title_tag = f"{direction} | {connection_name}"
            groups.append((title_tag, file_tag, sub.copy()))
    else:
        directions = sorted(df["direction"].unique())
        for direction in directions:
            sub = df[df["direction"] == direction].copy()
            if sub.empty:
                continue
            groups.append((str(direction), str(direction), sub))

    for title_tag, file_tag, sub in groups:

        sub["cluster"] = sub["cluster"].astype(int)
        sub["_dominant"] = sub[feat_cols].values.argmax(axis=1)
        sub["_dom_val"]  = sub[feat_cols].values.max(axis=1)
        if order_within_cluster_by_ttotal:
            if "tTotal" in sub.columns:
                sub["_ttotal_sort"] = pd.to_numeric(
                    sub["tTotal"], errors="coerce"
                ).fillna(-np.inf)
                sub = sub.sort_values(
                    ["cluster", "_ttotal_sort", "_dominant", "_dom_val"],
                    ascending=[True, False, True, False],
                ).reset_index(drop=True)
            else:
                log.warning(
                    "plot_single_neuron_heatmap: 'tTotal' missing — falling back "
                    "to default within-cluster ordering."
                )
                sub = sub.sort_values(
                    ["cluster", "_dominant", "_dom_val"],
                    ascending=[True, True, False],
                ).reset_index(drop=True)
        else:
            sub = sub.sort_values(
                ["cluster", "_dominant", "_dom_val"],
                ascending=[True, True, False],
            ).reset_index(drop=True)

        layer_raw = sub[feat_cols].to_numpy(dtype=float)  # (n_neurons, n_layers)
        abs_display_transform_l = str(abs_display_transform).lower()
        use_abs_display_transform = (
            (not shape_plus_magnitude)
            and mode_is_absolute(mode)
            and (abs_display_transform_l != "none")
        )
        if shape_plus_magnitude:
            # Shape-only display: normalize each neuron's layer profile to sum to 1.
            row_sums = np.nansum(layer_raw, axis=1, keepdims=True)
            layer_disp = np.zeros_like(layer_raw)
            np.divide(layer_raw, row_sums, out=layer_disp, where=row_sums > 0)
        else:
            layer_disp = layer_raw
            if use_abs_display_transform:
                layer_disp = transform_nonnegative(
                    layer_disp,
                    transform=abs_display_transform_l,
                    context=f"plot_single_neuron_heatmap[{mode}__{file_tag}]",
                    power=abs_display_transform_power,
                    asinh_cofactor=abs_display_transform_asinh_cofactor,
                )

        mat = layer_disp.T  # (n_layers, n_neurons)
        clusters = sub["cluster"].values

        # Compute cluster boundary positions and labels
        boundaries = []
        labels     = []
        prev = clusters[0]
        start = 0
        for i, c in enumerate(clusters):
            if c != prev:
                boundaries.append(i)
                labels.append((start, i, prev))
                start = i
                prev  = c
        labels.append((start, len(clusters), prev))

        n_neurons = len(sub)
        fig_w = max(6, n_neurons * 0.018 + 2.5)

        mag_spec = _resolve_magnitude_spec(sub, mode) if shape_plus_magnitude else None
        has_mag_strip = mag_spec is not None
        if has_mag_strip:
            mag_col, mag_label = mag_spec
            mag_raw = pd.to_numeric(sub[mag_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            lo = float(np.nanquantile(mag_raw, 0.01))
            hi = float(np.nanquantile(mag_raw, 0.99))
            if not np.isfinite(lo):
                lo = 0.0
            if not np.isfinite(hi):
                hi = float(np.nanmax(mag_raw)) if len(mag_raw) else 1.0
            if hi <= lo:
                hi = lo + 1e-9
            mag_disp = np.clip((mag_raw - lo) / (hi - lo), 0.0, 1.0)
        else:
            mag_raw = None
            mag_label = ""
            mag_disp = None

        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(fig_w, 6.2 if has_mag_strip else 5.5))
        gs  = GridSpec(3 if has_mag_strip else 2, 2, figure=fig,
                       width_ratios=[1, 0.03],
                       height_ratios=[3, 0.6, 1.2] if has_mag_strip else [3, 1.2],
                       hspace=0.08, wspace=0.03)

        ax_heat = fig.add_subplot(gs[0, 0])
        ax_cbar = fig.add_subplot(gs[0, 1])
        if has_mag_strip:
            ax_mag = fig.add_subplot(gs[1, 0], sharex=ax_heat)
            ax_src = fig.add_subplot(gs[2, 0], sharex=ax_heat)
            ax_empty_1 = fig.add_subplot(gs[1, 1])
            ax_empty_2 = fig.add_subplot(gs[2, 1])
            ax_empty_1.set_visible(False)
            ax_empty_2.set_visible(False)
        else:
            ax_mag = None
            ax_src = fig.add_subplot(gs[1, 0], sharex=ax_heat)
            ax_empty = fig.add_subplot(gs[1, 1])
            ax_empty.set_visible(False)

        # ── Top panel: single-neuron heatmap ──
        if shape_plus_magnitude:
            vmax_heat = 1.0
        else:
            vmax_heat = float(np.nanmax(mat)) if mode_is_absolute(mode) else 1.0
        if not np.isfinite(vmax_heat) or vmax_heat <= 0:
            vmax_heat = 1.0
        use_fraction_diverging = (
            single_neuron_fraction_diverging
            and (shape_plus_magnitude or not mode_is_absolute(mode))
        )
        heatmap_kwargs = {
            "aspect": "auto",
            "interpolation": "nearest",
        }
        if use_fraction_diverging:
            heatmap_kwargs["cmap"] = "bwr"
            heatmap_kwargs["norm"] = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
        else:
            heatmap_kwargs["cmap"] = "YlOrRd"
            heatmap_kwargs["vmin"] = 0
            heatmap_kwargs["vmax"] = vmax_heat
        im = ax_heat.imshow(mat, **heatmap_kwargs)

        for b in boundaries:
            ax_heat.axvline(b - 0.5, color="black", linewidth=1.2)

        ax_heat.set_yticks(range(len(LAYER_LABELS)))
        ax_heat.set_yticklabels(LAYER_LABELS, fontsize=9)
        ax_heat.set_ylabel("Target layer")
        ax_heat.tick_params(axis="x", labelbottom=False)

        mode_labels = {
            "terminals": "Terminal fraction",
            "axon": "Axon fraction",
            "density": "Norm. terminal density",
            "terminals_abs": "Terminal count",
            "axon_abs": "Axon length (um)",
        }
        if shape_plus_magnitude:
            mode_label = "Per-neuron layer fraction"
        else:
            mode_label = mode_labels.get(mode, mode)
        if use_abs_display_transform and not shape_plus_magnitude:
            mode_label = f"{abs_display_transform_l}({mode_label})"
        fig.colorbar(im, cax=ax_cbar, label=mode_label)

        if shape_plus_magnitude:
            title_suffix = " [shape-only display]"
        elif use_abs_display_transform:
            title_suffix = f" [display {abs_display_transform_l}]"
        else:
            title_suffix = ""
        if order_within_cluster_by_ttotal:
            title_suffix = f"{title_suffix} [within-cluster: tTotal desc]"
        ax_heat.set_title(
            f"{title_tag}: single-neuron {mode_label.lower()} profiles  "
            f"({n_neurons} neurons){title_suffix}",
            fontsize=11,
        )

        # ── Middle panel: magnitude strip ──
        if has_mag_strip and ax_mag is not None and mag_disp is not None:
            ax_mag.imshow(
                mag_disp[np.newaxis, :], aspect="auto", cmap="Greens", vmin=0, vmax=1,
                interpolation="nearest",
            )
            for b in boundaries:
                ax_mag.axvline(b - 0.5, color="black", linewidth=1.2)
            ax_mag.set_yticks([0])
            ax_mag.set_yticklabels([mag_label], fontsize=7)
            ax_mag.tick_params(axis="x", labelbottom=False)
            ax_mag.set_ylabel("Strength", fontsize=8)
            ax_mag.set_title("Per-neuron magnitude (display clipped to 1-99%)", fontsize=8.5, pad=2)
            med = float(np.nanmedian(mag_raw)) if mag_raw is not None and len(mag_raw) else np.nan
            ax_mag.text(
                1.002, 1.02, f"med={med:.1f}",
                transform=ax_mag.transAxes, va="top", ha="left",
                fontsize=6.8, color="#444",
            )

        # ── Bottom panel: source layer stacked bars ──
        # Stack from top (L1) to bottom (L6) by building bars from y=1 downward
        has_source = "source_layer" in sub.columns
        if has_source:
            for s_start, s_end, clust_id in labels:
                clust_neurons = sub.iloc[s_start:s_end]
                src_counts = clust_neurons["source_layer"].value_counts()
                total = src_counts.sum()
                if total == 0:
                    continue

                bar_x     = (s_start + s_end - 1) / 2
                bar_width = s_end - s_start

                top = 1.0
                for src_lay, color in zip(src_order, src_palette):
                    frac = src_counts.get(src_lay, 0) / total
                    ax_src.bar(
                        bar_x, frac, width=bar_width,
                        bottom=top - frac, color=color, edgecolor="white",
                        linewidth=0.3,
                    )
                    top -= frac

            for b in boundaries:
                ax_src.axvline(b - 0.5, color="black", linewidth=1.2)

            ax_src.set_ylim(0, 1)
            ax_src.set_ylabel("Source\nlayer frac.", fontsize=8)
            ax_src.set_xlim(-0.5, n_neurons - 0.5)

            # Legend for source layers
            from matplotlib.patches import Patch
            legend_handles = [Patch(facecolor=c, label=l)
                              for l, c in zip(src_labels, src_palette)]
            ax_src.legend(
                handles=legend_handles, title="Source layer", fontsize=7,
                title_fontsize=8, bbox_to_anchor=(1.04, 1), loc="upper left",
                ncol=1,
            )
        else:
            ax_src.set_visible(False)
            log.warning("plot_single_neuron_heatmap: 'source_layer' column "
                        "missing — skipping source layer panel.")

        # X-axis: cluster labels on the lowest visible panel
        xticks    = []
        xtick_lbl = []
        for s, e, c in labels:
            mid = (s + e - 1) / 2
            xticks.append(mid)
            n = e - s
            xtick_lbl.append(f"C{c}\n(n={n})")

        x_panel = ax_src if ax_src.get_visible() else (ax_mag if ax_mag is not None else ax_heat)
        x_panel.set_xticks(xticks)
        x_panel.set_xticklabels(xtick_lbl, fontsize=8)
        x_panel.set_xlabel("Cluster")

        fig.tight_layout()
        prefix = "single_neuron_heatmap_by_connection" if by_connection else "single_neuron_heatmap"
        _save(fig, out_dir / f"{prefix}__{mode}__{file_tag}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visualizations for cluster_termination_patterns.py outputs."
    )
    ap.add_argument(
        "--clustered_dir", required=True,
        help="Directory containing clustering outputs from cluster_termination_patterns.py."
    )
    ap.add_argument(
        "--out_dir", default=None,
        help="Output directory for figures (default: same as --clustered_dir)."
    )
    ap.add_argument(
        "--mode", choices=["terminals", "axon", "density", "terminals_abs", "axon_abs", "both"], default="both",
        help="Which mode(s) to visualize (default: both)."
    )
    ap.add_argument(
        "--scope", choices=["direction", "connection_direction"], default="direction",
        help="Clustering scope used in step 3 (default: direction)."
    )
    ap.add_argument(
        "--abs_log1p_display", action="store_true",
        help=(
            "Deprecated: apply log1p to absolute-mode heatmap display values. "
            "Use --abs_display_transform log1p."
        ),
    )
    ap.add_argument(
        "--abs_display_transform",
        choices=list(ABS_TRANSFORM_CHOICES),
        default="none",
        help=(
            "Display transform for absolute modes (terminals_abs, axon_abs) in centroid "
            "and single-neuron heatmaps (default: none)."
        ),
    )
    ap.add_argument(
        "--abs_display_transform_power",
        type=float,
        default=DEFAULT_ABS_TRANSFORM_POWER,
        help="Exponent for --abs_display_transform power (default: 0.75).",
    )
    ap.add_argument(
        "--abs_display_transform_asinh_cofactor",
        type=float,
        default=DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR,
        help="Cofactor for --abs_display_transform asinh (default: 1.0).",
    )
    ap.add_argument(
        "--shape_plus_magnitude", action="store_true",
        help=(
            "For single-neuron heatmaps, show shape-only per-neuron layer fractions "
            "plus a separate per-neuron magnitude strip (tTotal/aTotal)."
        ),
    )
    ap.add_argument(
        "--single_neuron_by_connection", action="store_true",
        help=(
            "Also export single-neuron heatmaps split by connection_name "
            "(one figure per direction x connection)."
        ),
    )
    ap.add_argument(
        "--order_within_cluster_by_ttotal", action="store_true",
        help=(
            "Order neurons within each cluster by total terminals (tTotal, descending) "
            "in single-neuron heatmaps."
        ),
    )
    ap.add_argument(
        "--single_neuron_fraction_diverging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a diverging blue-white-red colormap for single-neuron heatmaps "
            "in non-absolute modes (default: enabled)."
        ),
    )
    ap.add_argument(
        "--centroid_annotate_abs", action="store_true",
        help=(
            "For mode terminals/axon centroid heatmaps, annotate each cell as "
            "fraction with mean absolute layer value in parentheses "
            "(tL* for terminals, aL* for axon)."
        ),
    )
    ap.add_argument(
        "--centroid_fraction_diverging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a diverging blue-white-red colormap for centroid heatmaps in "
            "fraction modes (terminals/axon) (default: enabled)."
        ),
    )
    ap.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)."
    )
    args = ap.parse_args()

    abs_display_transform = str(args.abs_display_transform).lower()
    if args.abs_log1p_display:
        if abs_display_transform != "none":
            log.warning(
                "--abs_log1p_display is deprecated and ignored because --abs_display_transform=%s was provided.",
                abs_display_transform,
            )
        else:
            abs_display_transform = "log1p"

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    clustered_dir = Path(args.clustered_dir)
    out_dir       = Path(args.out_dir) if args.out_dir else clustered_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = ["terminals", "axon"] if args.mode == "both" else [args.mode]

    scope = args.scope

    # --- Load clustered CSVs
    dfs: dict[str, pd.DataFrame] = {}
    for mode in modes:
        path = clustered_dir / f"ALL_CONNECTIONS__clustered__{mode}__{scope}.csv"
        if path.exists():
            dfs[mode] = pd.read_csv(path)
            log.info("Loaded %s (%d rows)", path.name, len(dfs[mode]))
        else:
            log.warning("Clustered CSV not found: %s", path)

    if not dfs:
        raise FileNotFoundError(
            f"No clustered CSVs found in {clustered_dir}. "
            "Run cluster_termination_patterns.py first."
        )

    # --- Load centroid CSVs
    # With scope=direction, centroids are centroids__<mode>__FF.csv / __FB.csv.
    # With scope=connection_direction, centroids are per-connection:
    #   centroids__<mode>__<connection>__FF.csv — aggregate them per direction.
    centroids: dict[str, pd.DataFrame] = {}
    for mode in modes:
        for direction in ("FF", "FB"):
            if scope == "direction":
                path = clustered_dir / f"centroids__{mode}__{direction}.csv"
                if path.exists():
                    cent = pd.read_csv(path)
                    if "cluster" not in cent.columns and "Unnamed: 0" in cent.columns:
                        cent = cent.rename(columns={"Unnamed: 0": "cluster"})
                    if "cluster" not in cent.columns:
                        raise ValueError(f"Centroid file missing 'cluster' column: {path}")
                    cent["cluster"] = pd.to_numeric(cent["cluster"], errors="raise").astype(int)
                    centroids[f"{mode}__{direction}"] = cent
                    log.info("Loaded %s (%d clusters)", path.name,
                             len(centroids[f"{mode}__{direction}"]))
                else:
                    log.warning("Centroid file not found: %s", path)
            else:
                # Discover per-connection centroid files ending with __<direction>.csv
                import glob as globmod
                pattern = str(clustered_dir / f"centroids__{mode}__*__{direction}.csv")
                paths = sorted(globmod.glob(pattern))
                if paths:
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
                            "Re-run step 4 with globally unique cluster ids."
                        )
                    centroids[f"{mode}__{direction}"] = combined
                    log.info("Aggregated %d centroid files for %s__%s (%d centroids total)",
                             len(paths), mode, direction, len(combined))
                else:
                    log.warning("No centroid files found matching: %s", pattern)

    # --- Compute per-cluster neuron counts for each mode × direction
    cluster_sizes: dict[str, dict[int, int]] = {}
    for mode, df in dfs.items():
        for direction, grp in df.groupby("direction"):
            key = f"{mode}__{direction}"
            cluster_sizes[key] = (
                grp["cluster"].astype(int).value_counts().to_dict()
            )

    # Optional absolute-value overlays for centroid annotations
    centroid_abs_overlays: dict[str, pd.DataFrame] = {}
    if args.centroid_annotate_abs:
        overlay_cols_by_mode = {
            "terminals": get_terminal_count_cols(),
            "axon": get_axon_length_cols(),
        }
        for mode, abs_cols in overlay_cols_by_mode.items():
            if mode not in dfs:
                continue
            mode_df = dfs[mode]
            missing_abs = [c for c in abs_cols if c not in mode_df.columns]
            if missing_abs:
                log.warning(
                    "centroid_annotate_abs: mode '%s' missing absolute columns %s; skipping overlay.",
                    mode, missing_abs
                )
                continue
            for direction, grp in mode_df.groupby("direction"):
                key = f"{mode}__{direction}"
                if "cluster" not in grp.columns:
                    continue
                overlay = (
                    grp.groupby(grp["cluster"].astype(int))[abs_cols]
                    .mean()
                    .sort_index()
                )
                overlay.columns = LAYER_LABELS[:len(abs_cols)]
                centroid_abs_overlays[key] = overlay

    # --- Figure 1: Centroid heatmaps (both modes in one figure)
    if centroids:
        log.info("── Figure 1: Cluster centroid heatmaps")
        plot_centroids(
            centroids,
            cluster_sizes,
            out_dir,
            abs_display_transform=abs_display_transform,
            abs_display_transform_power=args.abs_display_transform_power,
            abs_display_transform_asinh_cofactor=args.abs_display_transform_asinh_cofactor,
            centroid_annotate_abs=args.centroid_annotate_abs,
            centroid_abs_overlays=centroid_abs_overlays,
            centroid_fraction_diverging=args.centroid_fraction_diverging,
        )

    # --- Figures 2–4: per-mode (use terminals for composition/source plots)
    for mode, df in dfs.items():
        log.info("── Figure 2 (%s): Within-cluster layer-value profiles", mode)
        plot_cluster_profiles(df, mode, out_dir)

    # Composition and source-layer plots are most meaningful for terminals
    primary_df = dfs.get("terminals", next(iter(dfs.values())))
    primary_mode = "terminals" if "terminals" in dfs else next(iter(dfs))

    log.info("── Figure 3: Connection composition per cluster (%s)", primary_mode)
    plot_cluster_composition(primary_df, out_dir)

    log.info("── Figure 4: Source soma layer distribution per cluster (%s)", primary_mode)
    plot_cluster_source_layers(primary_df, out_dir)

    # --- Figure 5: Single-neuron laminar heatmaps
    for mode, df in dfs.items():
        log.info("── Figure 5 (%s): Single-neuron laminar heatmap", mode)
        plot_single_neuron_heatmap(
            df,
            mode,
            out_dir,
            abs_display_transform=abs_display_transform,
            abs_display_transform_power=args.abs_display_transform_power,
            abs_display_transform_asinh_cofactor=args.abs_display_transform_asinh_cofactor,
            shape_plus_magnitude=args.shape_plus_magnitude,
            order_within_cluster_by_ttotal=args.order_within_cluster_by_ttotal,
            single_neuron_fraction_diverging=args.single_neuron_fraction_diverging,
        )
        if args.single_neuron_by_connection:
            log.info("── Figure 5b (%s): Single-neuron heatmap by connection", mode)
            plot_single_neuron_heatmap(
                df,
                mode,
                out_dir,
                abs_display_transform=abs_display_transform,
                abs_display_transform_power=args.abs_display_transform_power,
                abs_display_transform_asinh_cofactor=args.abs_display_transform_asinh_cofactor,
                shape_plus_magnitude=args.shape_plus_magnitude,
                by_connection=True,
                order_within_cluster_by_ttotal=args.order_within_cluster_by_ttotal,
                single_neuron_fraction_diverging=args.single_neuron_fraction_diverging,
            )

    log.info("All clustering visualizations complete. Outputs in: %s", out_dir)


if __name__ == "__main__":
    main()
