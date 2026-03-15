#!/usr/bin/env python3
"""
Hierarchical clustering of neuron termination patterns by cortical layer.

Takes the ALL_CONNECTIONS__mapped_and_terminals.csv output from mapping_to_ccf.py
and performs hierarchical clustering on per-neuron layer fraction features
(fraction of terminals or axon length in each cortical layer).

Outputs (in --out_dir):
  - ALL_CONNECTIONS__clustered__<mode>__<scope>.csv
      Full dataset with a 'cluster' column added.
      For scope='connection_direction', local labels are preserved in
      'cluster_local' and group tags in 'cluster_group'.
  - centroids__<mode>__<group_tag>.csv
      Mean feature values per cluster for each group.

Usage:
  python cluster_termination_patterns.py \\
    --in_csv output/ALL_CONNECTIONS__mapped_and_terminals.csv \\
    --out_dir output/clustering \\
    --mode terminals \\
    --scope direction \\
    --n_clusters 12
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from constants import get_feature_cols
from utils import (
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


def cluster_block(
    df_block: pd.DataFrame,
    feature_cols,
    method,
    metric,
    n_clusters,
    abs_transform: str = "none",
    abs_transform_power: float = DEFAULT_ABS_TRANSFORM_POWER,
    abs_transform_asinh_cofactor: float = DEFAULT_ABS_TRANSFORM_ASINH_COFACTOR,
):
    """
    Perform hierarchical clustering on a group of neurons.

    Rows with any non-finite feature values are dropped before clustering.
    If no valid rows remain, the original block is returned unchanged with
    'cluster' set to NaN and the linkage matrix is None.

    Args:
        df_block: DataFrame containing at least the columns in feature_cols.
        feature_cols: List of column names to use as clustering features.
        method: Scipy linkage method (ward, average, complete, single, etc.).
        metric: Distance metric for pdist (euclidean, cosine, correlation, etc.).
                With method='ward', supports:
                  - euclidean  (standard Ward)
                  - correlation / pearson (via Pearson-Ward embedding)
                  - aitchison / clr (via CLR transform + Euclidean Ward)
        n_clusters: Number of clusters to extract via maxclust criterion.
        abs_transform: Optional transform for absolute-feature clustering
                       (none/log1p/sqrt/power/asinh).

    Returns:
        (clustered_df, Z):
            clustered_df — DataFrame with 'cluster' column added. Contains only
                           the valid (non-NaN) rows unless all rows were invalid.
            Z            — Scipy linkage matrix, or None if clustering was skipped.
    """
    X = df_block[feature_cols].to_numpy(dtype=float)
    keep = np.isfinite(X).all(axis=1)
    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        log.warning(
            "cluster_block: dropping %d row(s) with missing/infinite feature values.", n_dropped
        )

    df_valid = df_block.loc[keep].copy()
    X_valid = df_valid[feature_cols].to_numpy(dtype=float)

    if len(df_valid) == 0:
        log.warning(
            "cluster_block: no valid rows remain after dropping NaN features. "
            "Returning block with NaN cluster."
        )
        result = df_block.copy()
        result["cluster"] = np.nan
        return result, None

    if str(abs_transform).lower() != "none":
        X_valid = transform_nonnegative(
            X_valid,
            transform=abs_transform,
            context="cluster_block",
            power=abs_transform_power,
            asinh_cofactor=abs_transform_asinh_cofactor,
        )

    method_l = str(method).lower()
    X_prepared, metric_eff = prepare_features_for_clustering(
        X_valid, method=method_l, metric=metric, context="cluster_block"
    )

    if method_l == "ward":
        Z = linkage(X_prepared, method="ward")
    else:
        d = pdist(X_prepared, metric=metric_eff)
        Z = linkage(d, method=method_l)

    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    df_valid["cluster"] = labels
    return df_valid, Z


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Hierarchical clustering of neuron terminal/axon fraction patterns "
            "by cortical layer."
        )
    )
    ap.add_argument(
        "--in_csv", required=True,
        help="Path to ALL_CONNECTIONS__mapped_and_terminals.csv"
    )
    ap.add_argument("--out_dir", required=True, help="Directory for output files.")
    ap.add_argument(
        "--mode", choices=["terminals", "axon", "density", "terminals_abs", "axon_abs"], default="terminals",
        help="Feature set: terminal fractions (fT*), axon fractions (fA*), terminal density (dTL*), terminal counts (tL*), or axon lengths (aL*)."
    )
    ap.add_argument(
        "--min_target_axon_length_um", type=float, default=1000.0,
        help="Minimum aTotal (um) for axon-based modes (default: 1000)."
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
        "--scope", choices=["direction", "connection_direction"], default="direction",
        help=(
            "Grouping for clustering: 'direction' clusters within FF vs FB; "
            "'connection_direction' clusters within each connection_name x direction."
        )
    )
    ap.add_argument(
        "--n_clusters", type=int, default=5,
        help="Default number of clusters per group (default: 5)."
    )
    ap.add_argument(
        "--n_clusters_ff", type=int, default=None,
        help="Override number of clusters for FF groups (uses --n_clusters if not set)."
    )
    ap.add_argument(
        "--n_clusters_fb", type=int, default=None,
        help="Override number of clusters for FB groups (uses --n_clusters if not set)."
    )
    ap.add_argument(
        "--method", default="ward",
        help="Hierarchical linkage method: ward, average, complete, single, weighted, centroid, median."
    )
    ap.add_argument(
        "--metric", default="euclidean",
        help=(
            "Distance metric. For non-ward methods, passed to scipy.spatial.distance.pdist "
            "(e.g., euclidean, cosine, correlation). For method='ward', supports: "
            "euclidean (standard Ward) or correlation/pearson "
            "(via row-centering + L2 normalization before Ward), "
            "or aitchison/clr (via CLR transform + Euclidean Ward)."
        )
    )
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
            "Transform for absolute-feature modes before clustering "
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
    ap.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)."
    )
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

    # Validate required feature columns are present before doing any work
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input CSV is missing feature columns for mode='{args.mode}': {missing_cols}. "
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
            "Absolute-feature mode detected (%s): applying %s transform before clustering.",
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

    if args.scope == "direction":
        group_cols = ["direction"]
    else:
        group_cols = ["connection_name", "direction"]

    # Build per-direction k lookup
    k_default = args.n_clusters
    k_for_direction = {
        "FF": args.n_clusters_ff if args.n_clusters_ff is not None else k_default,
        "FB": args.n_clusters_fb if args.n_clusters_fb is not None else k_default,
    }

    clustered_parts = []
    next_global_cluster = 1

    for keys, block in df.groupby(group_cols, dropna=False):
        tag = "__".join(str(k) for k in keys) if isinstance(keys, tuple) else str(keys)
        log.info("Clustering group '%s': %d neurons", tag, len(block))

        # Determine k: use direction-specific value if available
        direction = keys[-1] if isinstance(keys, tuple) else keys
        k = k_for_direction.get(direction, k_default)

        clustered_block, Z = cluster_block(
            block.copy(), feature_cols=feature_cols,
            method=args.method, metric=args.metric,
            n_clusters=k,
            abs_transform=abs_transform_effective,
            abs_transform_power=args.abs_transform_power,
            abs_transform_asinh_cofactor=args.abs_transform_asinh_cofactor,
        )

        # In connection_direction scope, local cluster labels restart in each group.
        # Re-map to globally unique ids so downstream analyses do not mix clusters
        # from different connections that happen to share the same local label.
        if args.scope == "connection_direction" and "cluster" in clustered_block.columns:
            valid_cluster_mask = clustered_block["cluster"].notna()
            if valid_cluster_mask.any():
                clustered_block["cluster_local"] = clustered_block["cluster"]
                clustered_block["cluster_group"] = tag

                local_ids = sorted(
                    clustered_block.loc[valid_cluster_mask, "cluster"].astype(int).unique().tolist()
                )
                local_to_global = {
                    local: global_id
                    for global_id, local in enumerate(local_ids, start=next_global_cluster)
                }
                clustered_block.loc[valid_cluster_mask, "cluster"] = (
                    clustered_block.loc[valid_cluster_mask, "cluster"].astype(int).map(local_to_global)
                )
                log.info(
                    "Group '%s': remapped %d local cluster ids to global range [%d, %d].",
                    tag, len(local_ids), min(local_to_global.values()), max(local_to_global.values())
                )
                next_global_cluster += len(local_ids)

        clustered_parts.append(clustered_block)

        # Skip centroid output if clustering was skipped (empty valid block)
        if Z is None:
            log.warning(
                "Group '%s': skipping centroid output (no valid neurons after filtering).", tag
            )
            continue

        centroids = (
            clustered_block
            .groupby("cluster")[feature_cols]
            .mean()
            .sort_index()
        )
        if not centroids.empty:
            centroids.index = centroids.index.astype(int)
            centroids.index.name = "cluster"
        log.info("Group '%s': %d clusters.", tag, len(centroids))
        centroids.to_csv(out_dir / f"centroids__{args.mode}__{tag}.csv")

    out = pd.concat(clustered_parts, ignore_index=True)
    out_path = out_dir / f"ALL_CONNECTIONS__clustered__{args.mode}__{args.scope}.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved %s (%d rows)", out_path.name, len(out))


if __name__ == "__main__":
    main()
