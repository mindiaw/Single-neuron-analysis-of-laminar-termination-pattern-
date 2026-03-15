#!/usr/bin/env python3
"""
Tests for the termination motif pipeline.

Run with:
    python -m pytest test_pipeline.py -v
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock


def _ensure_mock(module_path: str, attrs: dict | None = None):
    """Insert a mock module into sys.modules if not already importable."""
    if module_path in sys.modules:
        return
    parts = module_path.split(".")
    for i in range(len(parts)):
        partial = ".".join(parts[: i + 1])
        if partial not in sys.modules:
            mod = types.ModuleType(partial)
            sys.modules[partial] = mod
        if i > 0:
            parent = ".".join(parts[:i])
            setattr(sys.modules[parent], parts[i], sys.modules[partial])
    if attrs:
        for k, v in attrs.items():
            setattr(sys.modules[module_path], k, v)


# mapping_to_ccf imports AllenSDK at module import time
_ensure_mock("allensdk.core.reference_space_cache", {"ReferenceSpaceCache": MagicMock()})

import numpy as np
import pandas as pd
import pytest

from constants import get_feature_cols
from mapping_to_ccf import (
    _strip_excel_wrapping,
    normalize_neuron_id,
    parse_area_layer_from_acronym,
    layer_bucket,
    area_matches,
)
from cluster_termination_patterns import cluster_block
from utils import (
    hcluster,
    normalize_density_features,
    apply_axon_length_threshold,
    apply_terminal_count_threshold_for_axon_abs,
    transform_nonnegative,
)
from thesis_analyses import analysis_source_enrichment, analysis_connection_enrichment
from analyze_cluster_overlap import (
    _optimal_centroid_matching,
    _prepare_for_distance_geometry,
    analysis_centroid_distances,
)
from run_thesis_pipeline import _effective_cluster_counts, _resolve_steps, _drop_qc_mismatches
import run_thesis_pipeline as thesis_pipeline


class TestFeatureColumns:
    def test_terminals_mode_columns(self):
        assert get_feature_cols("terminals") == ["fT1", "fT23", "fT4", "fT5", "fT6"]

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            get_feature_cols("bad_mode")


class TestMappingParsing:
    def test_excel_value_cleanup(self):
        assert _strip_excel_wrapping('="3"') == "3"
        assert _strip_excel_wrapping('"hello"') == "hello"

    def test_neuron_id_normalization(self):
        assert normalize_neuron_id('="221057_071.swc"') == "221057_071"

    def test_area_layer_parsing_and_bucket(self):
        area, layer = parse_area_layer_from_acronym("SSp-bfd6a")
        assert area == "SSp-bfd"
        assert layer == "6a"
        assert layer_bucket(layer) == "6"

    def test_area_matching_is_strict(self):
        assert area_matches("SSp-bfd", "SSp")
        assert not area_matches("VISal", "VISa")


class TestClusteringCore:
    def _toy_df(self, n=40, seed=42):
        rng = np.random.default_rng(seed)
        feat_cols = ["fT1", "fT23", "fT4", "fT5", "fT6"]
        data = rng.dirichlet(np.ones(5), size=n)
        return pd.DataFrame(data, columns=feat_cols), feat_cols

    def test_cluster_block_ward(self):
        df, feat_cols = self._toy_df()
        out, Z = cluster_block(df, feat_cols, "ward", "euclidean", n_clusters=4)
        assert Z is not None
        assert "cluster" in out.columns
        assert len(out) == len(df)
        assert len(out["cluster"].unique()) == 4

    def test_cluster_block_non_ward_metric(self):
        df, feat_cols = self._toy_df()
        out, Z = cluster_block(df, feat_cols, "average", "cosine", n_clusters=3)
        assert Z is not None
        assert len(out["cluster"].unique()) == 3

    def test_cluster_block_handles_nan_rows(self):
        df, feat_cols = self._toy_df(n=20)
        df.loc[[0, 5], "fT1"] = np.nan
        out, Z = cluster_block(df, feat_cols, "ward", "euclidean", n_clusters=2)
        assert Z is not None
        assert len(out) == 18

    def test_cluster_block_all_nan(self):
        feat_cols = ["fT1", "fT23", "fT4", "fT5", "fT6"]
        df = pd.DataFrame({c: [np.nan] * 5 for c in feat_cols})
        out, Z = cluster_block(df, feat_cols, "ward", "euclidean", n_clusters=2)
        assert Z is None
        assert out["cluster"].isna().all()

    def test_hcluster_supports_ward_correlation(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(24, 5))
        labels = hcluster(X, n_clusters=4, method="ward", metric="correlation")
        assert len(labels) == len(X)
        assert set(labels).issubset({1, 2, 3, 4})

    def test_hcluster_supports_ward_aitchison(self):
        rng = np.random.default_rng(11)
        X = rng.dirichlet(np.ones(5), size=30)
        labels = hcluster(X, n_clusters=4, method="ward", metric="aitchison")
        assert len(labels) == len(X)
        assert set(labels).issubset({1, 2, 3, 4})

    def test_hcluster_supports_nonward_aitchison(self):
        rng = np.random.default_rng(12)
        X = rng.dirichlet(np.ones(5), size=28)
        labels = hcluster(X, n_clusters=3, method="average", metric="aitchison")
        assert len(labels) == len(X)
        assert set(labels).issubset({1, 2, 3})

    def test_hcluster_aitchison_negative_values_raise(self):
        X = np.array([[0.2, 0.3, 0.5], [0.1, -0.2, 1.1], [0.4, 0.3, 0.3]])
        with pytest.raises(ValueError):
            hcluster(X, n_clusters=2, method="ward", metric="aitchison")

    def test_hcluster_invalid_ward_metric_raises(self):
        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.25, 0.75]])
        with pytest.raises(ValueError):
            hcluster(X, n_clusters=2, method="ward", metric="cityblock")


class TestFilteringHelpers:
    def test_density_normalization_excludes_zero_rows(self):
        df = pd.DataFrame({
            "d1": [1.0, 0.0, np.nan],
            "d2": [2.0, 0.0, 3.0],
            "d3": [3.0, 0.0, 7.0],
        })
        out = normalize_density_features(df, ["d1", "d2", "d3"])
        assert len(out) == 2
        assert np.allclose(out[["d1", "d2", "d3"]].sum(axis=1).to_numpy(), 1.0)

    def test_axon_threshold_applies_only_to_axon_modes(self):
        df = pd.DataFrame({"aTotal": [500.0, 1000.0, 1500.0], "x": [1, 2, 3]})
        kept_axon = apply_axon_length_threshold(df, mode="axon_abs", min_target_axon_length_um=1000.0)
        kept_term = apply_axon_length_threshold(df, mode="terminals", min_target_axon_length_um=1000.0)
        assert len(kept_axon) == 2
        assert len(kept_term) == len(df)

    def test_terminal_threshold_for_axon_abs_only(self):
        df = pd.DataFrame({"tTotal": [0, 1, 2], "x": [1, 2, 3]})
        kept_axon_abs = apply_terminal_count_threshold_for_axon_abs(
            df, mode="axon_abs", min_target_terminals_for_axon_abs=1
        )
        kept_other = apply_terminal_count_threshold_for_axon_abs(
            df, mode="terminals_abs", min_target_terminals_for_axon_abs=1
        )
        assert len(kept_axon_abs) == 2
        assert len(kept_other) == len(df)

    def test_terminal_threshold_for_axon_abs_requires_ttotal(self):
        df = pd.DataFrame({"y": [1, 2, 3]})
        with pytest.raises(ValueError):
            apply_terminal_count_threshold_for_axon_abs(
                df, mode="axon_abs", min_target_terminals_for_axon_abs=1
            )

    def test_transform_nonnegative_variants(self):
        x = np.array([[0.0, 1.0, 4.0]])
        assert np.allclose(transform_nonnegative(x, transform="none"), x)
        assert np.allclose(transform_nonnegative(x, transform="log1p"), np.log1p(x))
        assert np.allclose(transform_nonnegative(x, transform="sqrt"), np.sqrt(x))
        assert np.allclose(transform_nonnegative(x, transform="power", power=0.5), np.sqrt(x))
        assert np.allclose(
            transform_nonnegative(x, transform="asinh", asinh_cofactor=2.0),
            np.arcsinh(x / 2.0),
        )

    def test_transform_nonnegative_validates_parameters(self):
        x = np.array([[0.0, 1.0, 4.0]])
        with pytest.raises(ValueError):
            transform_nonnegative(x, transform="power", power=0.0)
        with pytest.raises(ValueError):
            transform_nonnegative(x, transform="asinh", asinh_cofactor=0.0)

    def test_transform_nonnegative_clips_negatives(self):
        x = np.array([[-1.0, 2.0]])
        out = transform_nonnegative(x, transform="none")
        assert np.allclose(out, np.array([[0.0, 2.0]]))


class TestSourceEnrichment:
    def test_source_enrichment_empty_rows_returns_empty(self, tmp_path):
        feat_cols = ["fT1", "fT23", "fT4", "fT5", "fT6"]
        df = pd.DataFrame({
            "neuron_id": [f"N{i}" for i in range(8)],
            "direction": ["FF"] * 4 + ["FB"] * 4,
            "connection_name": ["C"] * 8,
            "source_layer": ["NA"] * 8,
            "fT1": np.linspace(0.1, 0.2, 8),
            "fT23": np.linspace(0.2, 0.3, 8),
            "fT4": np.linspace(0.3, 0.1, 8),
            "fT5": np.linspace(0.2, 0.2, 8),
            "fT6": np.linspace(0.2, 0.2, 8),
        })
        out = analysis_source_enrichment(
            df,
            feature_cols=feat_cols,
            n_clusters=2,
            out_dir=tmp_path,
            cluster_method="ward",
            cluster_metric="euclidean",
        )
        assert isinstance(out, pd.DataFrame)
        assert out.empty
        assert (tmp_path / "source_enrichment.csv").exists()

    def test_source_enrichment_clustered_merge_uses_connection(self, tmp_path):
        feat_cols = ["fT1", "fT23", "fT4", "fT5", "fT6"]
        df = pd.DataFrame({
            "neuron_id": ["N1", "N1", "N2", "N3"],
            "connection_name": ["C1", "C2", "C1", "C2"],
            "direction": ["FF", "FF", "FB", "FB"],
            "source_layer": ["2/3", "5", "2/3", "5"],
            "fT1": [0.2, 0.1, 0.3, 0.2],
            "fT23": [0.3, 0.2, 0.2, 0.2],
            "fT4": [0.1, 0.1, 0.1, 0.1],
            "fT5": [0.2, 0.4, 0.2, 0.4],
            "fT6": [0.2, 0.2, 0.2, 0.1],
        })
        clustered = pd.DataFrame({
            "neuron_id": ["N1", "N1", "N2", "N3"],
            "connection_name": ["C1", "C2", "C1", "C2"],
            "cluster": [1, 2, 1, 2],
        })
        clustered_csv = tmp_path / "clustered.csv"
        clustered.to_csv(clustered_csv, index=False)

        out = analysis_source_enrichment(
            df,
            feature_cols=feat_cols,
            n_clusters=2,
            out_dir=tmp_path,
            clustered_csv=clustered_csv,
            cluster_method="ward",
            cluster_metric="euclidean",
        )
        assert isinstance(out, pd.DataFrame)
        assert not out.empty
        assert set(out["cluster"].unique()) == {1, 2}


class TestConnectionEnrichment:
    def test_connection_enrichment_missing_connection_column_returns_empty(self, tmp_path):
        feat_cols = ["fT1", "fT23", "fT4", "fT5", "fT6"]
        df = pd.DataFrame({
            "neuron_id": ["N1", "N2", "N3", "N4"],
            "direction": ["FF", "FF", "FB", "FB"],
            "fT1": [0.2, 0.1, 0.3, 0.2],
            "fT23": [0.3, 0.2, 0.2, 0.2],
            "fT4": [0.1, 0.1, 0.1, 0.1],
            "fT5": [0.2, 0.4, 0.2, 0.4],
            "fT6": [0.2, 0.2, 0.2, 0.1],
        })
        out = analysis_connection_enrichment(
            df,
            feature_cols=feat_cols,
            n_clusters=2,
            out_dir=tmp_path,
            cluster_method="ward",
            cluster_metric="euclidean",
        )
        assert isinstance(out, pd.DataFrame)
        assert out.empty

    def test_connection_enrichment_clustered_merge(self, tmp_path):
        feat_cols = ["fT1", "fT23", "fT4", "fT5", "fT6"]
        df = pd.DataFrame({
            "neuron_id": ["N1", "N2", "N3", "N4", "N5", "N6"],
            "connection_name": ["C1", "C2", "C1", "C2", "C1", "C2"],
            "direction": ["FF", "FF", "FF", "FB", "FB", "FB"],
            "source_layer": ["2/3", "5", "2/3", "5", "2/3", "5"],
            "fT1": [0.2, 0.1, 0.3, 0.2, 0.15, 0.25],
            "fT23": [0.3, 0.2, 0.2, 0.2, 0.25, 0.15],
            "fT4": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "fT5": [0.2, 0.4, 0.2, 0.4, 0.3, 0.2],
            "fT6": [0.2, 0.2, 0.2, 0.1, 0.2, 0.3],
        })
        clustered = pd.DataFrame({
            "neuron_id": ["N1", "N2", "N3", "N4", "N5", "N6"],
            "connection_name": ["C1", "C2", "C1", "C2", "C1", "C2"],
            "cluster": [1, 2, 1, 2, 1, 2],
        })
        clustered_csv = tmp_path / "clustered.csv"
        clustered.to_csv(clustered_csv, index=False)

        out = analysis_connection_enrichment(
            df,
            feature_cols=feat_cols,
            n_clusters=2,
            out_dir=tmp_path,
            clustered_csv=clustered_csv,
            cluster_method="ward",
            cluster_metric="euclidean",
        )
        assert isinstance(out, pd.DataFrame)
        assert not out.empty
        assert set(out["cluster"].unique()) == {1, 2}
        assert (tmp_path / "connection_enrichment.csv").exists()


class TestThesisPipelineHelpers:
    def test_effective_cluster_counts_fallback(self):
        args = SimpleNamespace(n_clusters=7, n_clusters_ff=None, n_clusters_fb=None)
        assert _effective_cluster_counts(args) == (7, 7)

    def test_effective_cluster_counts_override(self):
        args = SimpleNamespace(n_clusters=7, n_clusters_ff=5, n_clusters_fb=9)
        assert _effective_cluster_counts(args) == (5, 9)

    def test_resolve_steps_default_all(self):
        assert _resolve_steps(only=None, skip=None) == [1, 2, 3, 4, 5, 6]

    def test_resolve_steps_only_takes_precedence(self):
        assert _resolve_steps(only=[2, 6, 99], skip=[2]) == [2, 6]

    def test_drop_qc_mismatches_uses_connection_aware_key(self):
        mapped = pd.DataFrame({
            "neuron_id": ["N1", "N1", "N2", "N3"],
            "connection_name": ["C1", "C2", "C1", "C2"],
            "value": [10, 20, 30, 40],
        })
        qc = pd.DataFrame({
            "neuron_id": ["N1", "N3"],
            "connection_name": ["C1", "C2"],
        })
        out = _drop_qc_mismatches(mapped, qc)
        assert len(out) == 2
        kept = set(map(tuple, out[["neuron_id", "connection_name"]].to_numpy()))
        assert kept == {("N1", "C2"), ("N2", "C1")}

    def test_step2_passes_abs_transform_options(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr(thesis_pipeline, "_analysis_input_csv", lambda _args: tmp_path / "in.csv")
        monkeypatch.setattr(thesis_pipeline, "_run", lambda cmd, step: captured.setdefault("cmd", cmd))
        args = SimpleNamespace(
            out_dir=tmp_path,
            mode="axon_abs",
            k_max=12,
            cluster_method="ward",
            cluster_metric="euclidean",
            min_target_axon_length_um=1000.0,
            min_target_terminals_for_axon_abs=1,
            abs_transform="sqrt",
            abs_transform_power=0.6,
            abs_transform_asinh_cofactor=1.5,
            no_abs_log1p=True,
        )
        thesis_pipeline.step2_optimal_k(args)
        cmd = captured["cmd"]
        assert "--abs_transform" in cmd and "sqrt" in cmd
        assert "--abs_transform_power" in cmd and "0.6" in cmd
        assert "--abs_transform_asinh_cofactor" in cmd and "1.5" in cmd
        assert "--no_abs_log1p" in cmd

    def test_step4_passes_abs_display_transform_options(self, tmp_path, monkeypatch):
        captured = {}
        (tmp_path / "clustering").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(thesis_pipeline, "_run", lambda cmd, step: captured.setdefault("cmd", cmd))
        args = SimpleNamespace(
            out_dir=tmp_path,
            mode="axon_abs",
            scope="direction",
            abs_display_transform="asinh",
            abs_display_transform_power=0.75,
            abs_display_transform_asinh_cofactor=3.0,
            abs_log1p_display=True,
            shape_plus_magnitude=False,
            single_neuron_by_connection=False,
            order_within_cluster_by_ttotal=False,
            centroid_annotate_abs=False,
            single_neuron_fraction_diverging=True,
            centroid_fraction_diverging=True,
        )
        thesis_pipeline.step4_visualize_clustering(args)
        cmd = captured["cmd"]
        assert "--abs_display_transform" in cmd and "asinh" in cmd
        assert "--abs_display_transform_power" in cmd and "0.75" in cmd
        assert "--abs_display_transform_asinh_cofactor" in cmd and "3.0" in cmd
        assert "--abs_log1p_display" in cmd
        assert "--no-single_neuron_fraction_diverging" not in cmd
        assert "--no-centroid_fraction_diverging" not in cmd

    def test_step4_passes_palette_opt_outs(self, tmp_path, monkeypatch):
        captured = {}
        (tmp_path / "clustering").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(thesis_pipeline, "_run", lambda cmd, step: captured.setdefault("cmd", cmd))
        args = SimpleNamespace(
            out_dir=tmp_path,
            mode="terminals",
            scope="direction",
            abs_display_transform="none",
            abs_display_transform_power=0.75,
            abs_display_transform_asinh_cofactor=1.0,
            abs_log1p_display=False,
            shape_plus_magnitude=False,
            single_neuron_by_connection=False,
            order_within_cluster_by_ttotal=False,
            centroid_annotate_abs=False,
            single_neuron_fraction_diverging=False,
            centroid_fraction_diverging=False,
        )
        thesis_pipeline.step4_visualize_clustering(args)
        cmd = captured["cmd"]
        assert "--no-single_neuron_fraction_diverging" in cmd
        assert "--no-centroid_fraction_diverging" in cmd

    def test_raw_viz_passes_abs_display_transform_options(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr(thesis_pipeline, "_analysis_input_csv", lambda _args: tmp_path / "in.csv")
        monkeypatch.setattr(thesis_pipeline, "_run", lambda cmd, step: captured.setdefault("cmd", cmd))
        args = SimpleNamespace(
            out_dir=tmp_path,
            raw_viz_out_dir=tmp_path / "raw_viz",
            raw_viz_mode="axon_abs",
            min_target_axon_length_um=1000.0,
            min_target_terminals_for_axon_abs=1,
            abs_display_transform="power",
            abs_display_transform_power=0.8,
            abs_display_transform_asinh_cofactor=1.0,
            raw_viz_max_connections=10,
            log_level="INFO",
            raw_viz_log1p_display=True,
            single_neuron_fraction_diverging=True,
        )
        thesis_pipeline.step_raw_visualization(args)
        cmd = captured["cmd"]
        assert "--abs_display_transform" in cmd and "power" in cmd
        assert "--abs_display_transform_power" in cmd and "0.8" in cmd
        assert "--abs_display_transform_asinh_cofactor" in cmd and "1.0" in cmd
        assert "--log1p_display" in cmd
        assert "--no-single_neuron_fraction_diverging" not in cmd

    def test_raw_viz_passes_single_neuron_palette_opt_out(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr(thesis_pipeline, "_analysis_input_csv", lambda _args: tmp_path / "in.csv")
        monkeypatch.setattr(thesis_pipeline, "_run", lambda cmd, step: captured.setdefault("cmd", cmd))
        args = SimpleNamespace(
            out_dir=tmp_path,
            raw_viz_out_dir=tmp_path / "raw_viz",
            raw_viz_mode="terminals",
            min_target_axon_length_um=1000.0,
            min_target_terminals_for_axon_abs=0,
            abs_display_transform="none",
            abs_display_transform_power=0.75,
            abs_display_transform_asinh_cofactor=1.0,
            raw_viz_max_connections=10,
            log_level="INFO",
            raw_viz_log1p_display=False,
            single_neuron_fraction_diverging=False,
        )
        thesis_pipeline.step_raw_visualization(args)
        cmd = captured["cmd"]
        assert "--no-single_neuron_fraction_diverging" in cmd


class TestOverlapMatching:
    def test_optimal_centroid_matching_diagonal(self):
        D = np.array([
            [0.1, 0.8, 0.9],
            [0.7, 0.2, 0.8],
            [0.6, 0.9, 0.3],
        ])
        row_ind, col_ind, mean_cost = _optimal_centroid_matching(D)
        assert row_ind.tolist() == [0, 1, 2]
        assert col_ind.tolist() == [0, 1, 2]
        assert np.isclose(mean_cost, (0.1 + 0.2 + 0.3) / 3.0)

    def test_prepare_for_distance_geometry_ward_correlation(self):
        X = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ])
        X_prepared, metric_eff = _prepare_for_distance_geometry(
            X,
            cluster_method="ward",
            cluster_metric="correlation",
            context="test_prepare_for_distance_geometry",
        )
        assert metric_eff == "euclidean"
        assert np.allclose(X_prepared.mean(axis=1), 0.0, atol=1e-10)
        norms = np.linalg.norm(X_prepared, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_analysis_centroid_distances_honors_metric(self, tmp_path):
        feat_cols = ["fT1", "fT23", "fT4", "fT5", "fT6"]
        centroids_ff = pd.DataFrame({
            "cluster": [1, 2],
            "fT1": [1.0, 0.0],
            "fT23": [0.0, 1.0],
            "fT4": [0.0, 0.0],
            "fT5": [0.0, 0.0],
            "fT6": [0.0, 0.0],
        })
        centroids_fb = pd.DataFrame({
            "cluster": [1, 2],
            "fT1": [0.0, 1.0],
            "fT23": [1.0, 0.0],
            "fT4": [0.0, 0.0],
            "fT5": [0.0, 0.0],
            "fT6": [0.0, 0.0],
        })

        dist_df = analysis_centroid_distances(
            centroids_ff=centroids_ff,
            centroids_fb=centroids_fb,
            feat_cols=feat_cols,
            mode="terminals",
            out_dir=tmp_path,
            cluster_method="average",
            cluster_metric="correlation",
        )
        # Correlation distance for these one-hot-like vectors is 1.25
        assert np.isclose(float(dist_df.iloc[0, 0]), 1.25, atol=1e-6)
        assert (tmp_path / "centroid_distances__terminals.csv").exists()
        assert (tmp_path / "centroid_matching__terminals.csv").exists()
