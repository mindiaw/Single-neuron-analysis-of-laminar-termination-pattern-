#!/usr/bin/env python3
"""
Thesis-focused orchestrator for termination motif analyses.

Pipeline steps:
  1. mapping_to_ccf        — map somas and build per-neuron laminar features
  2. evaluate_optimal_k    — evaluate k per direction
  3. cluster_termination   — hierarchical clustering
  4. visualize_clustering  — motif-centric clustering figures
  5. analyze_motif_core    — source/connection contribution + stability analyses
  6. analyze_overlap       — shared/similar FF-FB motif analyses

Notes:
  - Raw exploratory visualization stage is excluded from core steps.
  - Optional raw descriptive plots can be run via --run_raw_viz.
  - Modes/flags are preserved from the full pipeline where relevant.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from utils import (
    ABS_TRANSFORM_CHOICES,
    DEFAULT_ABS_TRANSFORM_POWER,
    DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR,
)

log = logging.getLogger("run_thesis_pipeline")

STEPS = {
    1: ("mapping_to_ccf", "Map somas to CCF and summarize terminals"),
    2: ("evaluate_optimal_k", "Evaluate optimal cluster number (k)"),
    3: ("cluster_termination", "Hierarchical clustering"),
    4: ("visualize_clustering", "Cluster visualizations"),
    5: ("analyze_motif_core", "Source-layer motif analyses"),
    6: ("analyze_overlap", "FF/FB shared-motif overlap analysis"),
}


MAPPED_CSV_NAME = "ALL_CONNECTIONS__mapped_and_terminals.csv"
QC_CSV_NAME = "ALL_CONNECTIONS__qc_source_mismatches.csv"
MAPPED_QC_FILTERED_CSV_NAME = "ALL_CONNECTIONS__mapped_and_terminals__qc_filtered.csv"


def _run(cmd: list[str], step_name: str) -> None:
    log.info("Running: %s", " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    dt = time.time() - t0
    if result.returncode != 0:
        log.error("FAILED [%s] (exit code %d, %.1fs)", step_name, result.returncode, dt)
        sys.exit(result.returncode)
    log.info("Done [%s] (%.1fs)", step_name, dt)


def _effective_cluster_counts(args: argparse.Namespace) -> tuple[int, int]:
    k_ff = args.n_clusters_ff if args.n_clusters_ff is not None else args.n_clusters
    k_fb = args.n_clusters_fb if args.n_clusters_fb is not None else args.n_clusters
    return int(k_ff), int(k_fb)


def _resolve_steps(only: list[int] | None, skip: list[int] | None) -> list[int]:
    """Resolve thesis pipeline steps from CLI controls."""
    all_steps = sorted(STEPS.keys())
    if only:
        return sorted(set(only) & set(all_steps))
    if skip:
        return [s for s in all_steps if s not in skip]
    return all_steps


def _csv_data_rows(path: Path) -> int:
    """Count data rows in a CSV (excluding header)."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return max(0, sum(1 for _ in f) - 1)


def _drop_qc_mismatches(mapped_df: pd.DataFrame, qc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove QC-mismatch rows from the mapped table.

    Preferred join key is (neuron_id, connection_name) to avoid cross-connection
    collisions for reused neuron identifiers.
    """
    if qc_df.empty:
        return mapped_df.copy()

    has_pair_key = all(c in mapped_df.columns and c in qc_df.columns for c in ("neuron_id", "connection_name"))
    if has_pair_key:
        key_cols = ["neuron_id", "connection_name"]
    elif "neuron_id" in mapped_df.columns and "neuron_id" in qc_df.columns:
        key_cols = ["neuron_id"]
        log.warning(
            "QC filtering fallback: using neuron_id-only key because connection_name was unavailable. "
            "This may over-filter when neuron IDs repeat across connections."
        )
    else:
        raise ValueError(
            "Cannot filter QC mismatches: required key columns are missing. "
            f"Mapped columns: {list(mapped_df.columns)}; QC columns: {list(qc_df.columns)}"
        )

    mapped_keys = mapped_df[key_cols].astype(str).agg("||".join, axis=1)
    qc_keys = set(qc_df[key_cols].astype(str).agg("||".join, axis=1).tolist())
    keep_mask = ~mapped_keys.isin(qc_keys)
    return mapped_df.loc[keep_mask].copy()


def _analysis_input_csv(args: argparse.Namespace) -> Path:
    """
    Return the mapped CSV used for downstream analyses.

    If --exclude_qc_mismatches is enabled, generate and return a filtered copy.
    """
    mapped_csv = args.out_dir / MAPPED_CSV_NAME
    if not mapped_csv.exists():
        log.error("Input not found: %s — run step 1 first.", mapped_csv)
        sys.exit(1)

    if not args.exclude_qc_mismatches:
        return mapped_csv

    qc_csv = args.out_dir / QC_CSV_NAME
    if not qc_csv.exists():
        log.error(
            "QC mismatch file not found: %s. Run step 1 first to generate QC metadata.",
            qc_csv,
        )
        sys.exit(1)

    mapped_df = pd.read_csv(mapped_csv)
    qc_df = pd.read_csv(qc_csv)
    filtered_df = _drop_qc_mismatches(mapped_df, qc_df)
    out_csv = args.out_dir / MAPPED_QC_FILTERED_CSV_NAME
    filtered_df.to_csv(out_csv, index=False)
    log.info(
        "QC filtering enabled: removed %d mismatch rows (%d -> %d). Using %s",
        len(mapped_df) - len(filtered_df), len(mapped_df), len(filtered_df), out_csv.name
    )
    return out_csv


def _ensure_clustered_matches_input(clustered_csv: Path, in_csv: Path, context: str) -> bool:
    """
    Validate row-count compatibility between clustered output and its input table.
    """
    if not clustered_csv.exists():
        return False
    n_clustered = _csv_data_rows(clustered_csv)
    n_input = _csv_data_rows(in_csv)
    if n_clustered != n_input:
        log.warning(
            "%s: clustered CSV row count (%d) != input row count (%d): %s vs %s",
            context, n_clustered, n_input, clustered_csv.name, in_csv.name
        )
        return False
    return True


def step1_mapping(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, "mapping_to_ccf.py",
        "--data_dir", str(args.data_dir),
        "--out_dir", str(args.out_dir),
        "--min_target_terminals", str(args.min_target_terminals),
    ]
    if args.config is not None:
        cmd += ["--config", str(args.config)]
    if args.no_axon_length:
        cmd.append("--no_axon_length")
    _run(cmd, "mapping_to_ccf")


def step2_optimal_k(args: argparse.Namespace) -> None:
    in_csv = _analysis_input_csv(args)
    out_dir = args.out_dir / "optimal_k"
    cmd = [
        sys.executable, "evaluate_optimal_k.py",
        "--in_csv", str(in_csv),
        "--out_dir", str(out_dir),
        "--mode", args.mode,
        "--k_max", str(args.k_max),
        "--method", args.cluster_method,
        "--metric", args.cluster_metric,
        "--min_target_axon_length_um", str(args.min_target_axon_length_um),
        "--min_target_terminals_for_axon_abs", str(args.min_target_terminals_for_axon_abs),
        "--abs_transform", args.abs_transform,
        "--abs_transform_power", str(args.abs_transform_power),
        "--abs_transform_asinh_cofactor", str(args.abs_transform_asinh_cofactor),
    ]
    if args.no_abs_log1p:
        cmd.append("--no_abs_log1p")
    _run(cmd, "evaluate_optimal_k")

    log.info("Review optimal-k outputs:")
    log.info("  %s", out_dir / "optimal_k_data.csv")
    log.info("  %s", out_dir / "optimal_k_curves.png")


def step3_clustering(args: argparse.Namespace) -> None:
    in_csv = _analysis_input_csv(args)
    k_ff, k_fb = _effective_cluster_counts(args)
    out_dir = args.out_dir / "clustering"
    cmd = [
        sys.executable, "cluster_termination_patterns.py",
        "--in_csv", str(in_csv),
        "--out_dir", str(out_dir),
        "--mode", args.mode,
        "--scope", args.scope,
        "--n_clusters", str(args.n_clusters),
        "--n_clusters_ff", str(k_ff),
        "--n_clusters_fb", str(k_fb),
        "--method", args.cluster_method,
        "--metric", args.cluster_metric,
        "--min_target_axon_length_um", str(args.min_target_axon_length_um),
        "--min_target_terminals_for_axon_abs", str(args.min_target_terminals_for_axon_abs),
        "--abs_transform", args.abs_transform,
        "--abs_transform_power", str(args.abs_transform_power),
        "--abs_transform_asinh_cofactor", str(args.abs_transform_asinh_cofactor),
    ]
    if args.no_abs_log1p:
        cmd.append("--no_abs_log1p")
    _run(cmd, "cluster_termination_patterns")


def step4_visualize_clustering(args: argparse.Namespace) -> None:
    clustered_dir = args.out_dir / "clustering"
    if not clustered_dir.exists():
        log.error("Clustering directory not found: %s — run step 3 first.", clustered_dir)
        sys.exit(1)
    cmd = [
        sys.executable, "visualize_clustering.py",
        "--clustered_dir", str(clustered_dir),
        "--out_dir", str(clustered_dir),
        "--mode", args.mode,
        "--scope", args.scope,
        "--abs_display_transform", args.abs_display_transform,
        "--abs_display_transform_power", str(args.abs_display_transform_power),
        "--abs_display_transform_asinh_cofactor", str(args.abs_display_transform_asinh_cofactor),
    ]
    if args.abs_log1p_display:
        cmd.append("--abs_log1p_display")
    if args.shape_plus_magnitude:
        cmd.append("--shape_plus_magnitude")
    if args.single_neuron_by_connection:
        cmd.append("--single_neuron_by_connection")
    if args.order_within_cluster_by_ttotal:
        cmd.append("--order_within_cluster_by_ttotal")
    if args.centroid_annotate_abs:
        cmd.append("--centroid_annotate_abs")
    if not args.single_neuron_fraction_diverging:
        cmd.append("--no-single_neuron_fraction_diverging")
    if not args.centroid_fraction_diverging:
        cmd.append("--no-centroid_fraction_diverging")
    _run(cmd, "visualize_clustering")


def step5_motif_core(args: argparse.Namespace) -> None:
    in_csv = _analysis_input_csv(args)
    clustered_csv = args.out_dir / "clustering" / f"ALL_CONNECTIONS__clustered__{args.mode}__{args.scope}.csv"
    k_ff, k_fb = _effective_cluster_counts(args)
    out_dir = args.out_dir / "source_layers"
    cmd = [
        sys.executable, "analyze_motif_core.py",
        "--in_csv", str(in_csv),
        "--out_dir", str(out_dir),
        "--mode", args.mode,
        "--n_clusters", str(args.n_clusters),
        "--n_clusters_ff", str(k_ff),
        "--n_clusters_fb", str(k_fb),
        "--n_bootstrap", str(args.n_bootstrap),
        "--cluster_method", args.cluster_method,
        "--cluster_metric", args.cluster_metric,
        "--min_target_axon_length_um", str(args.min_target_axon_length_um),
        "--min_target_terminals_for_axon_abs", str(args.min_target_terminals_for_axon_abs),
    ]
    if _ensure_clustered_matches_input(clustered_csv, in_csv, context="step5_motif_core"):
        cmd += ["--clustered_csv", str(clustered_csv)]
    _run(cmd, "analyze_motif_core")


def step6_overlap(args: argparse.Namespace) -> None:
    in_csv = _analysis_input_csv(args)
    clustered_dir = args.out_dir / "clustering"
    if not clustered_dir.exists():
        log.error("Clustering directory not found: %s — run step 3 first.", clustered_dir)
        sys.exit(1)
    clustered_csv = clustered_dir / f"ALL_CONNECTIONS__clustered__{args.mode}__{args.scope}.csv"
    if args.exclude_qc_mismatches and not _ensure_clustered_matches_input(
        clustered_csv, in_csv, context="step6_overlap"
    ):
        log.error(
            "Step 6 requires clustering outputs aligned with the filtered analysis input. "
            "Re-run step 3 (and optionally step 4) with --exclude_qc_mismatches."
        )
        sys.exit(1)
    k_ff, k_fb = _effective_cluster_counts(args)
    out_dir = args.out_dir / "overlap"
    cmd = [
        sys.executable, "analyze_cluster_overlap.py",
        "--in_csv", str(in_csv),
        "--clustered_dir", str(clustered_dir),
        "--out_dir", str(out_dir),
        "--mode", args.mode,
        "--scope", args.scope,
        "--k_ff", str(k_ff),
        "--k_fb", str(k_fb),
        "--k_combined", str(k_ff + k_fb),
        "--cluster_method", args.cluster_method,
        "--cluster_metric", args.cluster_metric,
        "--n_permutations", str(args.n_permutations),
        "--min_target_axon_length_um", str(args.min_target_axon_length_um),
        "--min_target_terminals_for_axon_abs", str(args.min_target_terminals_for_axon_abs),
    ]
    _run(cmd, "analyze_cluster_overlap")


def step_raw_visualization(args: argparse.Namespace) -> None:
    """Optional standalone raw-data visualizations."""
    in_csv = _analysis_input_csv(args)
    raw_out_dir = args.raw_viz_out_dir
    cmd = [
        sys.executable, "visualize_raw_data.py",
        "--in_csv", str(in_csv),
        "--out_dir", str(raw_out_dir),
        "--mode", args.raw_viz_mode,
        "--min_target_axon_length_um", str(args.min_target_axon_length_um),
        "--min_target_terminals_for_axon_abs", str(args.min_target_terminals_for_axon_abs),
        "--abs_display_transform", args.abs_display_transform,
        "--abs_display_transform_power", str(args.abs_display_transform_power),
        "--abs_display_transform_asinh_cofactor", str(args.abs_display_transform_asinh_cofactor),
        "--max_connections", str(args.raw_viz_max_connections),
        "--log_level", args.log_level,
    ]
    if args.raw_viz_log1p_display:
        cmd.append("--log1p_display")
    if not args.single_neuron_fraction_diverging:
        cmd.append("--no-single_neuron_fraction_diverging")
    _run(cmd, "visualize_raw_data")


STEP_FUNCS = {
    1: step1_mapping,
    2: step2_optimal_k,
    3: step3_clustering,
    4: step4_visualize_clustering,
    5: step5_motif_core,
    6: step6_overlap,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the thesis-focused motif analysis pipeline."
    )

    ap.add_argument("--data_dir", type=str, default="data",
                    help="Directory containing soma and terminal CSVs.")
    ap.add_argument("--out_dir", type=str, default="output/thesis",
                    help="Root output directory for thesis artifacts.")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to connections config JSON (optional).")
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    help="Run only these thesis steps.")
    ap.add_argument("--skip", type=int, nargs="+", default=None,
                    help="Skip these thesis steps.")

    ap.add_argument("--mode", choices=["terminals", "axon", "density", "terminals_abs", "axon_abs"],
                    default="terminals",
                    help="Feature mode for clustering/analysis.")
    ap.add_argument("--scope", choices=["direction", "connection_direction"],
                    default="direction",
                    help="Clustering scope.")
    ap.add_argument("--n_clusters", type=int, default=5,
                    help="Default number of clusters (fallback).")
    ap.add_argument("--n_clusters_ff", type=int, default=None,
                    help="Number of clusters for FF (fallback: --n_clusters).")
    ap.add_argument("--n_clusters_fb", type=int, default=None,
                    help="Number of clusters for FB (fallback: --n_clusters).")
    ap.add_argument("--cluster_method", type=str, default="ward",
                    help="Linkage method for clustering-based steps.")
    ap.add_argument("--cluster_metric", type=str, default="euclidean",
                    help="Distance metric for clustering-based steps (supports euclidean, correlation/pearson, aitchison/clr).")

    ap.add_argument("--k_max", type=int, default=20,
                    help="Max k in optimal-k evaluation.")
    ap.add_argument("--n_bootstrap", type=int, default=100,
                    help="Bootstrap iterations for stability analysis.")
    ap.add_argument("--n_permutations", type=int, default=1000,
                    help="Deprecated (unused): retained for backward compatibility.")
    ap.add_argument("--min_target_terminals", type=int, default=3,
                    help="Minimum total terminals in target area to keep a neuron.")
    ap.add_argument("--min_target_axon_length_um", type=float, default=1000.0,
                    help="Minimum total axon length (um) for axon-based modes.")
    ap.add_argument(
        "--min_target_terminals_for_axon_abs", type=int, default=0,
        help=(
            "Additional minimum tTotal applied only for mode='axon_abs' in downstream steps "
            "(default: 0 = disabled). Example: 1."
        ),
    )
    ap.add_argument("--no_axon_length", action="store_true",
                    help="Skip axon-length summarization in mapping step.")
    ap.add_argument(
        "--exclude_qc_mismatches", action="store_true",
        help=(
            "For downstream analysis steps (2-6 and optional raw viz), remove rows listed in "
            "ALL_CONNECTIONS__qc_source_mismatches.csv and use a filtered input table."
        ),
    )

    ap.add_argument("--no_abs_log1p", action="store_true",
                    help="Deprecated: disable absolute transform for absolute-feature modes; use --abs_transform none.")
    ap.add_argument(
        "--abs_transform",
        choices=list(ABS_TRANSFORM_CHOICES),
        default="log1p",
        help=(
            "Transform for absolute-feature modes in steps 2-3 "
            "(none/log1p/sqrt/power/asinh; default: log1p)."
        ),
    )
    ap.add_argument(
        "--abs_transform_power",
        type=float,
        default=DEFAULT_ABS_TRANSFORM_POWER,
        help=f"Exponent for --abs_transform power (default: {DEFAULT_ABS_TRANSFORM_POWER}).",
    )
    ap.add_argument(
        "--abs_transform_asinh_cofactor",
        type=float,
        default=DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR,
        help=f"Cofactor for --abs_transform asinh (default: {DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR}).",
    )
    ap.add_argument("--abs_log1p_display", action="store_true",
                    help="Deprecated: apply log1p to absolute-mode display values in step 4; use --abs_display_transform log1p.")
    ap.add_argument(
        "--abs_display_transform",
        choices=list(ABS_TRANSFORM_CHOICES),
        default="none",
        help=(
            "Display transform for absolute-feature visualizations in step 4 and optional raw viz "
            "(none/log1p/sqrt/power/asinh; default: none)."
        ),
    )
    ap.add_argument(
        "--abs_display_transform_power",
        type=float,
        default=DEFAULT_ABS_TRANSFORM_POWER,
        help=f"Exponent for --abs_display_transform power (default: {DEFAULT_ABS_TRANSFORM_POWER}).",
    )
    ap.add_argument(
        "--abs_display_transform_asinh_cofactor",
        type=float,
        default=DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR,
        help=f"Cofactor for --abs_display_transform asinh (default: {DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR}).",
    )
    ap.add_argument("--shape_plus_magnitude", action="store_true",
                    help="Enable shape+strength single-neuron heatmaps in step 4.")
    ap.add_argument("--single_neuron_by_connection", action="store_true",
                    help="Export single-neuron heatmaps split by connection in step 4.")
    ap.add_argument("--order_within_cluster_by_ttotal", action="store_true",
                    help="Order neurons within clusters by tTotal in step 4.")
    ap.add_argument("--centroid_annotate_abs", action="store_true",
                    help="Annotate step-4 centroid heatmap cells with fraction and mean absolute layer value.")
    ap.add_argument(
        "--single_neuron_fraction_diverging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a blue-white-red 0/0.5/1 colormap for single-neuron fraction "
            "heatmaps in step 4 and optional raw viz (default: enabled)."
        ),
    )
    ap.add_argument(
        "--centroid_fraction_diverging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a blue-white-red 0/0.5/1 colormap for centroid fraction "
            "heatmaps in step 4 (default: enabled)."
        ),
    )
    ap.add_argument("--run_raw_viz", action="store_true",
                    help="Run optional standalone raw-data descriptive visualizations after selected steps.")
    ap.add_argument("--raw_viz_mode", choices=["terminals", "axon", "density", "terminals_abs", "axon_abs"],
                    default=None,
                    help="Mode for optional raw visualizations (default: same as --mode).")
    ap.add_argument("--raw_viz_out_dir", type=str, default=None,
                    help="Output directory for optional raw visualizations (default: <out_dir>/raw_viz).")
    ap.add_argument("--raw_viz_max_connections", type=int, default=25,
                    help="Top connections to show in optional raw visualization summaries (default: 25).")
    ap.add_argument("--raw_viz_log1p_display", action="store_true",
                    help="Deprecated: apply log1p display for absolute modes in optional raw visualizations.")

    ap.add_argument("--log_level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging verbosity.")

    args = ap.parse_args()
    args.data_dir = Path(args.data_dir)
    args.out_dir = Path(args.out_dir)
    args.abs_transform = str(args.abs_transform).lower()
    args.abs_display_transform = str(args.abs_display_transform).lower()
    args.raw_viz_mode = args.raw_viz_mode if args.raw_viz_mode is not None else args.mode
    args.raw_viz_out_dir = Path(args.raw_viz_out_dir) if args.raw_viz_out_dir else (args.out_dir / "raw_viz")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    steps_to_run = _resolve_steps(only=args.only, skip=args.skip)

    if not steps_to_run:
        log.error("No steps selected to run.")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 64)
    log.info("Thesis Motif Pipeline")
    log.info("Steps: %s", ", ".join(f"{s} ({STEPS[s][1]})" for s in steps_to_run))
    log.info("Output root: %s", args.out_dir)
    log.info("Exclude QC mismatches (steps 2-6/raw viz): %s", args.exclude_qc_mismatches)
    log.info(
        "Axon-abs terminal threshold: tTotal >= %d (active only when mode=axon_abs)",
        int(args.min_target_terminals_for_axon_abs),
    )
    log.info(
        "Absolute transform (steps 2-3): %s (power=%.4g, asinh_cofactor=%.4g)",
        args.abs_transform,
        float(args.abs_transform_power),
        float(args.abs_transform_asinh_cofactor),
    )
    log.info(
        "Absolute display transform (step 4/raw viz): %s (power=%.4g, asinh_cofactor=%.4g)",
        args.abs_display_transform,
        float(args.abs_display_transform_power),
        float(args.abs_display_transform_asinh_cofactor),
    )
    log.info("=" * 64)

    t0 = time.time()
    for s in steps_to_run:
        log.info("-" * 64)
        log.info("Step %d: %s", s, STEPS[s][1])
        log.info("-" * 64)
        STEP_FUNCS[s](args)

    if args.run_raw_viz:
        log.info("-" * 64)
        log.info("Optional: Raw descriptive visualizations")
        log.info("-" * 64)
        step_raw_visualization(args)

    dt = time.time() - t0
    log.info("=" * 64)
    log.info("Thesis motif pipeline complete (%.1fs total)", dt)
    log.info("=" * 64)


if __name__ == "__main__":
    main()
