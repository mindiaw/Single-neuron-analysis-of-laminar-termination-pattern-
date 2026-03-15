#!/usr/bin/env python3
"""
Standalone raw-data visualizations for laminar profiles.

This script is intentionally separate from run_thesis_pipeline.py so the thesis
pipeline stays motif-focused while still allowing optional exploratory/raw views.

Inputs:
  - ALL_CONNECTIONS__mapped_and_terminals.csv

Outputs (in --out_dir):
  - raw_counts_by_direction.png
  - raw_counts_by_direction.csv
  - raw_counts_by_source_layer_direction.png
  - raw_counts_by_source_layer_direction.csv
  - raw_top_connections_by_direction.png
  - raw_top_connections_by_direction.csv
  - raw_termination_pattern_summary__<mode>.png
  - raw_termination_direction_means__<mode>.csv
  - raw_dominant_layer_counts__<mode>.csv
  - raw_dominant_layer_fractions__<mode>.csv
  - raw_layer_distributions__<mode>.png
  - raw_single_neuron_heatmap__<mode>__<direction>.png
  - raw_connection_means__<mode>__<direction>.png
  - raw_connection_means__<mode>__<direction>.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns

from constants import LAYER_LABELS, DIRECTION_PALETTE, get_feature_cols, get_mode_value_label, mode_is_absolute
from utils import (
    save_figure as _save,
    normalize_density_features,
    apply_axon_length_threshold,
    apply_terminal_count_threshold_for_axon_abs,
    transform_nonnegative,
    ABS_TRANSFORM_CHOICES,
    DEFAULT_ABS_TRANSFORM_POWER,
    DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR,
)

log = logging.getLogger(__name__)


def _display_matrix(
    X: np.ndarray,
    mode: str,
    abs_display_transform: str,
    abs_display_transform_power: float,
    abs_display_transform_asinh_cofactor: float,
    context: str,
) -> np.ndarray:
    X_out = np.asarray(X, dtype=float)
    transform = str(abs_display_transform).lower()
    if mode_is_absolute(mode) and transform != "none":
        return transform_nonnegative(
            X_out,
            transform=transform,
            context=context,
            power=abs_display_transform_power,
            asinh_cofactor=abs_display_transform_asinh_cofactor,
        )
    return X_out


def _direction_color_map(directions: list[str]) -> dict[str, str]:
    """Color map for available directions with stable fallback colors."""
    fallback = sns.color_palette("tab10", len(directions))
    colors: dict[str, str] = {}
    for i, d in enumerate(directions):
        colors[d] = DIRECTION_PALETTE.get(d, fallback[i])
    return colors


def plot_raw_count_summaries(df: pd.DataFrame, out_dir: Path, max_connections: int) -> None:
    """Descriptive neuron-count figures/tables across direction/source/connection."""
    if "direction" not in df.columns:
        log.warning("plot_raw_count_summaries: 'direction' missing — skipping count summaries.")
        return

    has_neuron_id = "neuron_id" in df.columns
    if has_neuron_id:
        by_direction = df[["direction", "neuron_id"]].dropna().drop_duplicates()
    else:
        log.warning(
            "plot_raw_count_summaries: 'neuron_id' missing — falling back to row-level counts."
        )
        by_direction = df[["direction"]].dropna().copy()

    # 1) Counts by direction (unique neurons when neuron_id is available)
    direction_counts = by_direction["direction"].value_counts().sort_values(ascending=False)
    if not direction_counts.empty:
        col_name = "n_unique_neurons" if has_neuron_id else "n_rows"
        direction_counts.rename(col_name).to_csv(out_dir / "raw_counts_by_direction.csv", header=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        directions = direction_counts.index.tolist()
        colors = [_direction_color_map(directions)[d] for d in directions]
        ax.bar(directions, direction_counts.values, color=colors, edgecolor="white", linewidth=0.6)
        for i, n in enumerate(direction_counts.values):
            ax.text(i, n, f"{int(n)}", ha="center", va="bottom", fontsize=9)
        ax.set(
            xlabel="Direction",
            ylabel="Unique neuron count" if has_neuron_id else "Row count",
            title="Unique neuron counts by direction" if has_neuron_id else "Row counts by direction",
        )
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        _save(fig, out_dir / "raw_counts_by_direction.png")

    # 2) Counts by source layer x direction (unique neurons when neuron_id is available)
    if "source_layer" in df.columns:
        src_order = ["1", "2/3", "4", "5", "6"]
        src_labels = ["L1", "L2/3", "L4", "L5", "L6"]
        sub = df[df["source_layer"].isin(src_order) & df["direction"].notna()].copy()
        if has_neuron_id:
            sub = sub[["direction", "source_layer", "neuron_id"]].dropna().drop_duplicates()
        else:
            sub = sub[["direction", "source_layer"]].dropna()
        if not sub.empty:
            ct = (
                sub.groupby(["direction", "source_layer"]).size()
                .unstack(fill_value=0)
                .reindex(columns=src_order, fill_value=0)
            )
            ct.columns = src_labels
            ct = ct.sort_index()
            ct.to_csv(out_dir / "raw_counts_by_source_layer_direction.csv")

            fig_h = max(3.2, 1.3 + 0.5 * len(ct))
            fig, ax = plt.subplots(figsize=(7, fig_h))
            sns.heatmap(
                ct,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar_kws={"label": "Unique neuron count" if has_neuron_id else "Row count", "shrink": 0.85},
                ax=ax,
            )
            ax.set(
                xlabel="Source (soma) layer",
                ylabel="Direction",
                title=(
                    "Unique neuron counts by source layer and direction"
                    if has_neuron_id
                    else "Row counts by source layer and direction"
                ),
            )
            ax.tick_params(axis="x", rotation=0)
            ax.tick_params(axis="y", rotation=0)
            fig.tight_layout()
            _save(fig, out_dir / "raw_counts_by_source_layer_direction.png")

    # 3) Top connection counts by direction (unique neurons per connection-direction when possible)
    if "connection_name" in df.columns:
        sub = df[df["connection_name"].notna() & df["direction"].notna()].copy()
        if has_neuron_id:
            sub = sub[["connection_name", "direction", "neuron_id"]].dropna().drop_duplicates()
        else:
            sub = sub[["connection_name", "direction"]].dropna()
        if not sub.empty:
            top_connections = (
                sub["connection_name"].value_counts()
                .head(max_connections)
                .index.tolist()
            )
            conn_ct = (
                sub[sub["connection_name"].isin(top_connections)]
                .groupby(["connection_name", "direction"])
                .size()
                .unstack(fill_value=0)
            )
            if not conn_ct.empty:
                conn_ct["__total__"] = conn_ct.sum(axis=1)
                conn_ct = conn_ct.sort_values("__total__", ascending=False).drop(columns="__total__")
                conn_ct.to_csv(out_dir / "raw_top_connections_by_direction.csv")

                directions = conn_ct.columns.tolist()
                colors = [_direction_color_map(directions)[d] for d in directions]

                fig_h = max(4.0, 1.8 + 0.32 * len(conn_ct))
                fig, ax = plt.subplots(figsize=(9.0, fig_h))
                conn_ct.plot(
                    kind="barh",
                    stacked=True,
                    color=colors,
                    ax=ax,
                    edgecolor="white",
                    linewidth=0.3,
                )
                ax.invert_yaxis()
                ax.set(
                    xlabel="Unique neuron count" if has_neuron_id else "Row count",
                    ylabel="Connection",
                    title=(
                        f"Top {len(conn_ct)} connections by unique neuron count (stacked by direction)"
                        if has_neuron_id
                        else f"Top {len(conn_ct)} connections by row count (stacked by direction)"
                    ),
                )
                ax.legend(title="Direction", fontsize=8)
                ax.grid(True, alpha=0.3, axis="x")
                fig.tight_layout()
                _save(fig, out_dir / "raw_top_connections_by_direction.png")


def plot_raw_termination_pattern_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    mode: str,
    out_dir: Path,
) -> None:
    """Descriptive termination-pattern summary (means + dominant layer mix)."""
    if "direction" not in df.columns:
        log.warning("plot_raw_termination_pattern_summary: 'direction' missing — skipping.")
        return

    valid = df[["direction"] + feature_cols].dropna().copy()
    if valid.empty:
        log.warning("plot_raw_termination_pattern_summary: no valid rows after dropna.")
        return

    # Per-direction means/SEM
    means = valid.groupby("direction")[feature_cols].mean()
    sems = valid.groupby("direction")[feature_cols].sem().fillna(0.0)
    counts = valid.groupby("direction").size().rename("n_neurons")
    means_out = means.copy()
    means_out.insert(0, "n_neurons", counts.reindex(means_out.index).astype(int))
    means_out.to_csv(out_dir / f"raw_termination_direction_means__{mode}.csv")

    # Dominant layer distribution
    X = valid[feature_cols].to_numpy(dtype=float)
    dom_idx = X.argmax(axis=1)
    valid["dominant_layer"] = [LAYER_LABELS[i] for i in dom_idx]
    dom_counts = (
        valid.groupby(["direction", "dominant_layer"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LAYER_LABELS, fill_value=0)
    )
    dom_fracs = dom_counts.div(dom_counts.sum(axis=1), axis=0).fillna(0.0)
    dom_counts.to_csv(out_dir / f"raw_dominant_layer_counts__{mode}.csv")
    dom_fracs.to_csv(out_dir / f"raw_dominant_layer_fractions__{mode}.csv")

    # Figure: direction means + dominant-layer composition
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1.0]})

    directions = means.index.tolist()
    cmap = _direction_color_map(directions)
    x = np.arange(len(LAYER_LABELS))
    for direction in directions:
        y = means.loc[direction, feature_cols].to_numpy(dtype=float)
        e = sems.loc[direction, feature_cols].to_numpy(dtype=float)
        color = cmap[direction]
        axes[0].plot(x, y, marker="o", linewidth=2.0, color=color, label=f"{direction} (n={int(counts[direction])})")
        axes[0].fill_between(x, y - e, y + e, color=color, alpha=0.15, linewidth=0.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(LAYER_LABELS)
    if not mode_is_absolute(mode):
        axes[0].set_ylim(-0.05, 1.05)
    axes[0].set(
        xlabel="Target layer",
        ylabel=f"Mean {get_mode_value_label(mode)}",
        title="Mean laminar profile by direction (± SEM)",
    )
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].legend(title="Direction", fontsize=8)

    dom_fracs.plot(
        kind="bar",
        stacked=True,
        ax=axes[1],
        colormap="viridis",
        edgecolor="white",
        linewidth=0.3,
    )
    axes[1].set(
        xlabel="Direction",
        ylabel="Fraction of neurons",
        ylim=(0, 1),
        title="Dominant termination layer composition",
    )
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Dominant layer", fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Raw termination pattern summary ({mode})", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / f"raw_termination_pattern_summary__{mode}.png")


def plot_raw_layer_distributions(df: pd.DataFrame, feature_cols: list[str], mode: str, out_dir: Path) -> None:
    """Plot layer-wise value distributions split by direction."""
    if "direction" not in df.columns:
        log.warning("plot_raw_layer_distributions: 'direction' missing — skipping.")
        return

    long = (
        df[["direction"] + feature_cols]
        .melt(id_vars=["direction"], var_name="layer_col", value_name="value")
        .dropna(subset=["value", "direction"])
    )
    if long.empty:
        log.warning("plot_raw_layer_distributions: no valid rows after dropping NaN values.")
        return

    long["layer"] = long["layer_col"].map(dict(zip(feature_cols, LAYER_LABELS)))
    directions = sorted(long["direction"].unique())
    palette = {d: DIRECTION_PALETTE.get(d, "#888888") for d in directions}

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=long,
        x="layer",
        y="value",
        hue="direction",
        order=LAYER_LABELS,
        hue_order=directions,
        palette=palette,
        fliersize=1.8,
        linewidth=0.8,
        ax=ax,
    )
    if not mode_is_absolute(mode):
        ax.set_ylim(-0.05, 1.05)
    ax.set(
        xlabel="Target layer",
        ylabel=get_mode_value_label(mode),
        title=f"Raw layer distributions ({mode})",
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="Direction", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / f"raw_layer_distributions__{mode}.png")


def plot_raw_single_neuron_heatmaps(
    df: pd.DataFrame,
    feature_cols: list[str],
    mode: str,
    out_dir: Path,
    abs_display_transform: str,
    abs_display_transform_power: float,
    abs_display_transform_asinh_cofactor: float,
    single_neuron_fraction_diverging: bool = True,
) -> None:
    """Plot per-neuron laminar profiles for each direction without clustering."""
    if "direction" not in df.columns:
        log.warning("plot_raw_single_neuron_heatmaps: 'direction' missing — skipping.")
        return

    for direction in sorted(df["direction"].dropna().unique()):
        sub = df[df["direction"] == direction].copy()
        if sub.empty:
            continue

        valid = sub[feature_cols].dropna()
        if valid.empty:
            log.warning("Raw single-neuron heatmap (%s): no valid rows after dropna.", direction)
            continue

        X = valid.to_numpy(dtype=float)
        dominant = X.argmax(axis=1)
        dominant_value = X.max(axis=1)
        order = np.lexsort((-dominant_value, dominant))
        X = X[order]

        X_plot = _display_matrix(
            X,
            mode=mode,
            abs_display_transform=abs_display_transform,
            abs_display_transform_power=abs_display_transform_power,
            abs_display_transform_asinh_cofactor=abs_display_transform_asinh_cofactor,
            context=f"plot_raw_single_neuron_heatmaps[{mode}__{direction}]",
        )
        mat = X_plot.T

        fig_w = max(6.0, 2.5 + 0.02 * X.shape[0])
        fig, ax = plt.subplots(figsize=(fig_w, 3.8))
        vmax = float(np.nanmax(mat)) if mode_is_absolute(mode) else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        use_fraction_diverging = single_neuron_fraction_diverging and not mode_is_absolute(mode)
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
            heatmap_kwargs["vmax"] = vmax
        im = ax.imshow(mat, **heatmap_kwargs)
        ax.set_yticks(range(len(LAYER_LABELS)))
        ax.set_yticklabels(LAYER_LABELS, fontsize=9)
        ax.set_xlabel("Neuron (sorted by dominant layer)")
        ax.set_ylabel("Target layer")
        ax.set_xticks([])

        value_label = get_mode_value_label(mode)
        transform = str(abs_display_transform).lower()
        if mode_is_absolute(mode) and transform != "none":
            value_label = f"{transform}({value_label})"
        ax.set_title(
            f"Raw per-neuron laminar profiles — {direction} (n={X.shape[0]})",
            fontsize=11,
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label=value_label)

        fig.tight_layout()
        _save(fig, out_dir / f"raw_single_neuron_heatmap__{mode}__{direction}.png")


def plot_raw_connection_means(
    df: pd.DataFrame,
    feature_cols: list[str],
    mode: str,
    out_dir: Path,
    abs_display_transform: str,
    abs_display_transform_power: float,
    abs_display_transform_asinh_cofactor: float,
    max_connections: int,
) -> None:
    """Plot connection-level mean laminar profiles (top connections by n)."""
    if "connection_name" not in df.columns or "direction" not in df.columns:
        log.warning(
            "plot_raw_connection_means: requires 'connection_name' and 'direction' columns — skipping."
        )
        return

    for direction in sorted(df["direction"].dropna().unique()):
        sub = df[(df["direction"] == direction) & (df["connection_name"].notna())].copy()
        if sub.empty:
            continue

        counts = sub["connection_name"].value_counts()
        top_connections = counts.head(max_connections).index.tolist()
        agg = (
            sub[sub["connection_name"].isin(top_connections)]
            .groupby("connection_name")[feature_cols]
            .mean()
            .reindex(top_connections)
        )
        if agg.empty:
            continue

        agg_out = agg.copy()
        agg_out.index.name = "connection_name"
        agg_out.to_csv(out_dir / f"raw_connection_means__{mode}__{direction}.csv")

        plot_vals = _display_matrix(
            agg.to_numpy(dtype=float),
            mode=mode,
            abs_display_transform=abs_display_transform,
            abs_display_transform_power=abs_display_transform_power,
            abs_display_transform_asinh_cofactor=abs_display_transform_asinh_cofactor,
            context=f"plot_raw_connection_means[{mode}__{direction}]",
        )
        plot_df = pd.DataFrame(plot_vals, index=agg.index, columns=LAYER_LABELS)

        vmax = float(np.nanmax(plot_df.to_numpy())) if mode_is_absolute(mode) else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        fig_h = max(3.5, 1.3 + 0.25 * len(plot_df))
        fig, ax = plt.subplots(figsize=(8.0, fig_h))
        value_label = get_mode_value_label(mode)
        transform = str(abs_display_transform).lower()
        if mode_is_absolute(mode) and transform != "none":
            value_label = f"{transform}({value_label})"
        sns.heatmap(
            plot_df,
            annot=False,
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": f"Mean {value_label}", "shrink": 0.85},
            ax=ax,
        )
        ax.set(
            title=f"Raw connection-mean profiles — {direction} (top {len(plot_df)} by n)",
            xlabel="Target layer",
            ylabel="Connection",
        )
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        _save(fig, out_dir / f"raw_connection_means__{mode}__{direction}.png")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Standalone raw-data visualizations for laminar profiles."
    )
    ap.add_argument("--in_csv", required=True, help="Path to ALL_CONNECTIONS__mapped_and_terminals.csv")
    ap.add_argument("--out_dir", required=True, help="Directory for raw-visualization outputs.")
    ap.add_argument(
        "--mode",
        choices=["terminals", "axon", "density", "terminals_abs", "axon_abs"],
        default="terminals",
        help="Feature mode for plotting.",
    )
    ap.add_argument(
        "--min_target_axon_length_um",
        type=float,
        default=1000.0,
        help="Minimum aTotal (um) for axon-based modes (default: 1000).",
    )
    ap.add_argument(
        "--min_target_terminals_for_axon_abs",
        type=int,
        default=0,
        help=(
            "Additional minimum tTotal for mode='axon_abs' only (default: 0 = disabled). "
            "Example: set to 1 to require at least one terminal."
        ),
    )
    ap.add_argument(
        "--log1p_display",
        action="store_true",
        help="Deprecated: for absolute modes, display values as log1p(value) in heatmaps.",
    )
    ap.add_argument(
        "--abs_display_transform",
        choices=list(ABS_TRANSFORM_CHOICES),
        default="none",
        help="Display transform for absolute modes in raw heatmaps (default: none).",
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
        "--single_neuron_fraction_diverging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a diverging blue-white-red colormap for raw single-neuron "
            "heatmaps in non-absolute modes (default: enabled)."
        ),
    )
    ap.add_argument(
        "--max_connections",
        type=int,
        default=25,
        help=(
            "Maximum number of top connections to show in connection-based summaries "
            "(raw top-connection counts + connection-mean heatmaps; default: 25)."
        ),
    )
    ap.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = ap.parse_args()

    abs_display_transform = str(args.abs_display_transform).lower()
    if args.log1p_display:
        if abs_display_transform != "none":
            log.warning(
                "--log1p_display is deprecated and ignored because --abs_display_transform=%s was provided.",
                abs_display_transform,
            )
        else:
            abs_display_transform = "log1p"

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

    log.info("Loaded %d rows for raw visualization (mode=%s).", len(df), args.mode)
    if mode_is_absolute(args.mode) and abs_display_transform != "none":
        log.info("Absolute mode with display transform enabled: %s.", abs_display_transform)

    plot_raw_count_summaries(df, out_dir, max_connections=max(1, int(args.max_connections)))
    plot_raw_termination_pattern_summary(df, feature_cols, args.mode, out_dir)
    plot_raw_layer_distributions(df, feature_cols, args.mode, out_dir)
    plot_raw_single_neuron_heatmaps(
        df,
        feature_cols,
        args.mode,
        out_dir,
        abs_display_transform,
        args.abs_display_transform_power,
        args.abs_display_transform_asinh_cofactor,
        args.single_neuron_fraction_diverging,
    )
    plot_raw_connection_means(
        df,
        feature_cols,
        args.mode,
        out_dir,
        abs_display_transform,
        args.abs_display_transform_power,
        args.abs_display_transform_asinh_cofactor,
        max_connections=max(1, int(args.max_connections)),
    )
    log.info("Raw visualizations complete. Outputs in: %s", out_dir)


if __name__ == "__main__":
    main()
