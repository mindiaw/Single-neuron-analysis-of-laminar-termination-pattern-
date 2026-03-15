#!/usr/bin/env python3
"""
Core motif-focused analyses for the thesis pipeline.

This script intentionally runs only the analyses tied to the thesis questions:
  1) Source layer -> target layer profiles/statistics
  2) Bootstrap cluster stability
  3) Source layer enrichment per termination cluster
  4) Connection contribution/enrichment per termination cluster

Inputs:
  - ALL_CONNECTIONS__mapped_and_terminals.csv
  - Optional clustered CSV from cluster_termination_patterns.py

Outputs (in --out_dir):
  - source_target_stats.csv
  - source_target_posthoc.csv
  - source_target_profiles.png
  - cluster_stability__FF.csv / __FB.csv
  - cluster_stability__FF.png / __FB.png
  - source_enrichment.csv
  - source_enrichment.png
  - connection_enrichment.csv
  - connection_enrichment.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from thesis_analyses import (
    analysis_source_to_target,
    analysis_bootstrap_stability,
    analysis_source_enrichment,
    analysis_connection_enrichment,
)
from constants import get_feature_cols
from utils import (
    normalize_density_features,
    apply_axon_length_threshold,
    apply_terminal_count_threshold_for_axon_abs,
)

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run motif-focused source-layer and overlap-support analyses."
    )
    ap.add_argument("--in_csv", required=True,
                    help="Path to ALL_CONNECTIONS__mapped_and_terminals.csv")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for motif-focused analysis artifacts.")
    ap.add_argument("--mode", choices=["terminals", "axon", "density", "terminals_abs", "axon_abs"],
                    default="terminals",
                    help="Feature mode used for clustering/analysis.")
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
    ap.add_argument("--n_clusters", type=int, default=12,
                    help="Default clusters per direction for reclustered analyses.")
    ap.add_argument("--n_clusters_ff", type=int, default=None,
                    help="Override clusters for FF (fallback: --n_clusters).")
    ap.add_argument("--n_clusters_fb", type=int, default=None,
                    help="Override clusters for FB (fallback: --n_clusters).")
    ap.add_argument("--n_bootstrap", type=int, default=100,
                    help="Bootstrap iterations for stability analysis.")
    ap.add_argument("--cluster_method", default="ward",
                    help="Hierarchical linkage method for reclustered analyses.")
    ap.add_argument("--cluster_metric", default="euclidean",
                    help=(
                        "Distance metric for reclustered analyses. "
                        "For method='ward', supports euclidean, correlation/pearson, or aitchison/clr."
                    ))
    ap.add_argument("--clustered_csv", default=None,
                    help="Optional path to pre-clustered CSV with 'cluster' column.")
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

    k_by_direction = {
        "FF": args.n_clusters_ff if args.n_clusters_ff is not None else args.n_clusters,
        "FB": args.n_clusters_fb if args.n_clusters_fb is not None else args.n_clusters,
    }
    clustered_path = Path(args.clustered_csv) if args.clustered_csv else None

    log.info("Loaded %d neurons, mode=%s", len(df), args.mode)
    log.info("Output directory: %s", out_dir)
    log.info("Cluster counts: default=%d, FF=%d, FB=%d",
             args.n_clusters, k_by_direction["FF"], k_by_direction["FB"])
    log.info("Clustering geometry: method=%s, metric=%s",
             args.cluster_method, args.cluster_metric)

    log.info("── Analysis A: Source layer -> target layer profiles")
    analysis_source_to_target(df, feature_cols, out_dir, mode=args.mode)

    log.info("── Analysis B: Bootstrap cluster stability")
    analysis_bootstrap_stability(
        df,
        feature_cols,
        args.n_clusters,
        args.n_bootstrap,
        out_dir,
        method=args.cluster_method,
        metric=args.cluster_metric,
        n_clusters_by_direction=k_by_direction,
    )

    log.info("── Analysis C: Source layer enrichment")
    analysis_source_enrichment(
        df,
        feature_cols,
        args.n_clusters,
        out_dir,
        clustered_csv=clustered_path,
        cluster_method=args.cluster_method,
        cluster_metric=args.cluster_metric,
        n_clusters_by_direction=k_by_direction,
    )

    log.info("── Analysis D: Connection enrichment")
    analysis_connection_enrichment(
        df,
        feature_cols,
        args.n_clusters,
        out_dir,
        clustered_csv=clustered_path,
        cluster_method=args.cluster_method,
        cluster_metric=args.cluster_metric,
        n_clusters_by_direction=k_by_direction,
    )

    log.info("Motif-focused analyses complete. Outputs in: %s", out_dir)


if __name__ == "__main__":
    main()
