#!/usr/bin/env python3
"""
Map soma coordinates (Allen CCFv3) -> source area/layer using AllenSDK,
then summarize target-layer terminal points per neuron, optionally also axon length per target layer.

Outputs (in --out_dir):
  - <connection_name>__mapped_and_terminals.csv
  - <connection_name>__qc_source_mismatches.csv
  - ALL_CONNECTIONS__mapped_and_terminals.csv
  - ALL_CONNECTIONS__qc_source_mismatches.csv

Requirements:
  pip install allensdk pandas numpy

Assumptions:
- Soma CSV has columns: NeuronId, x, y, z (coords in microns, Allen reference space orientation)
- terminal_points_byNeuron CSV is wide:
    first col = structure acronym (e.g., SSs2/3)
    remaining cols = neuron IDs
    values = counts (possibly Excel-ish strings like ="3")
- axon_length_byNeuron CSV has the same wide format but float values (e.g., ="2952.38")

Connection configurations can be supplied via --config (a JSON file) instead of the
built-in hardcoded list. See connections_config.json for the expected format.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from allensdk.core.reference_space_cache import ReferenceSpaceCache

from constants import CANONICAL_LAYERS


# -----------------------------
# Constants
# -----------------------------

CCF_DEFAULT_RESOLUTION_UM: int = 10
SOMA_REQUIRED_COLS: List[str] = ["NeuronId", "x", "y", "z"]

log = logging.getLogger(__name__)


# -----------------------------
# Utilities: cleaning/parsing
# -----------------------------

_LAYER_RE = re.compile(r"^(?P<area>.+?)(?P<layer>1|2/3|4|5|6a|6b)$")


def _strip_excel_wrapping(v):
    """Convert values like '=\"8066.4\"' or '=\"0\"' or '\"foo\"' to raw."""
    if pd.isna(v):
        return v
    s = str(v).strip()
    if s.startswith('="') and s.endswith('"') and len(s) >= 4:
        return s[2:-1]
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        return s[1:-1]
    return s


def normalize_neuron_id(raw: str) -> str:
    """Unify neuron ids across files:  =\"221057_071.swc\" -> 221057_071"""
    s = _strip_excel_wrapping(raw)
    s = str(s).replace(".swc", "").strip()
    return s


def parse_area_layer_from_acronym(acronym: str) -> Tuple[str | None, str | None]:
    """VISp2/3 -> (VISp, 2/3); SSp-bfd6a -> (SSp-bfd, 6a); ACAd -> (ACAd, None)"""
    if acronym is None or (isinstance(acronym, float) and np.isnan(acronym)):
        return None, None
    a = str(acronym).strip()
    m = _LAYER_RE.match(a)
    if m:
        return m.group("area"), m.group("layer")
    return a, None


def layer_bucket(layer: str | None) -> str | None:
    """Collapse 6a/6b -> 6; keep 1,2/3,4,5."""
    if layer is None:
        return None
    if layer in ("6a", "6b"):
        return "6"
    return layer


def area_matches(actual: str | None, expected: str) -> bool:
    """Check if an actual CCF area matches an expected area.

    Matches on:
      - Exact match (case-insensitive): TEa == TEa, VISp == VISp
      - Hyphen-delimited subdivision:   SSp-bfd starts with SSp-

    Does NOT use general prefix matching, which would cause false positives
    like VISal matching VISa or VISpm matching VISp.  For accurate parent→child
    matching (e.g. ACA → ACAd/ACAv), use get_area_descendants() with the Allen
    structure tree instead.
    """
    if actual is None:
        return False
    a_low = actual.lower()
    e_low = expected.lower()
    if a_low == e_low:
        return True
    if a_low.startswith(e_low + "-"):
        return True
    return False


def get_area_descendants(area_name: str,
                         structure_tree,
                         id_to_acronym: Dict[int, str]) -> set[str]:
    """Return lowercase acronyms for *area_name* and all its CCF descendants.

    Uses the Allen structure tree for accurate parent→child resolution,
    avoiding false positives like VISal matching VISa.

    Falls back to exact + hyphen matching when the area is not found in the
    tree (e.g. custom or abbreviated names).
    """
    # Build case-insensitive acronym → structure-id lookup
    acronym_to_id: Dict[str, int] = {
        acr.lower(): sid for sid, acr in id_to_acronym.items()
    }
    key = area_name.lower()

    if key not in acronym_to_id:
        log.warning(
            "Area '%s' not found in structure tree; using exact + hyphen match only.",
            area_name,
        )
        return {key}

    parent_id = acronym_to_id[key]
    try:
        desc_id_sets = structure_tree.descendant_ids([parent_id])
        desc_ids = set(desc_id_sets[0]) if desc_id_sets else set()
    except Exception:
        log.warning(
            "Failed to query descendants for '%s'; using exact + hyphen match only.",
            area_name,
        )
        return {key}

    desc_ids.add(parent_id)
    return {id_to_acronym[sid].lower() for sid in desc_ids if sid in id_to_acronym}


# -----------------------------
# Input validation
# -----------------------------

def validate_soma_csv(path: Path) -> pd.DataFrame:
    """
    Load and validate a soma coordinates CSV.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required columns are missing.

    Returns:
        The loaded DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Soma CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in SOMA_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Soma CSV '{path.name}' is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    return df


def validate_wide_csv(path: Path) -> None:
    """
    Validate that a wide-format neuron CSV exists and has at least 2 columns.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If fewer than 2 columns (no neuron data columns).
    """
    if not path.exists():
        raise FileNotFoundError(f"Wide-format CSV not found: {path}")
    header = pd.read_csv(path, nrows=0)
    if len(header.columns) < 2:
        raise ValueError(
            f"Wide CSV '{path.name}' must have at least 2 columns "
            f"(structure acronym + neuron columns). Found: {list(header.columns)}"
        )


# -----------------------------
# AllenSDK: download/load CCF
# -----------------------------

def load_ccf(cache_dir: Path,
             resolution_um: int = CCF_DEFAULT_RESOLUTION_UM,
             reference_space_key: str = os.path.join("annotation", "ccf_2017"),
             structure_graph_id: int = 1):
    """
    Downloads (on first run) or loads from cache the Allen CCFv3 annotation
    volume and structure tree.

    Args:
        cache_dir: Directory for AllenSDK cache files.
        resolution_um: Voxel resolution in microns (10, 25, or 50).
        reference_space_key: AllenSDK reference space identifier.
        structure_graph_id: Allen structure graph ID (1 = adult mouse).

    Returns:
        annotation: 3D numpy array of structure IDs.
        id_to_acronym: Dict mapping structure_id (int) -> acronym (str).
        tree: AllenSDK StructureTree (used by get_area_descendants).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("Loading CCF annotation volume (resolution=%dµm) from %s", resolution_um, cache_dir)
    rspc = ReferenceSpaceCache(
        resolution_um,
        reference_space_key,
        manifest=cache_dir / "manifest.json"
    )
    annotation, _ = rspc.get_annotation_volume()
    tree = rspc.get_structure_tree(structure_graph_id=structure_graph_id)
    id_to_acronym = tree.value_map(lambda s: s["id"], lambda s: s["acronym"])
    log.info("CCF loaded: annotation shape=%s, %d structures", annotation.shape, len(id_to_acronym))
    return annotation, id_to_acronym, tree


# -----------------------------
# Soma mapping
# -----------------------------

def map_somas_to_ccf(soma_df: pd.DataFrame,
                     annotation: np.ndarray,
                     id_to_acronym: Dict[int, str],
                     resolution_um: int,
                     x_col="x", y_col="y", z_col="z") -> pd.DataFrame:
    """
    Map soma coordinates (microns) to CCF structure IDs and area/layer labels.

    Coordinates are converted to voxel indices by dividing by resolution_um.
    Voxels that fall outside the annotation volume bounds are clipped to the
    nearest edge, and a warning is emitted listing the affected neuron IDs.

    Args:
        soma_df: DataFrame with at least [neuron_id, x_col, y_col, z_col].
        annotation: 3D CCF annotation array.
        id_to_acronym: Mapping from structure_id (int) -> acronym (str).
        resolution_um: Voxel size in microns.
        x_col, y_col, z_col: Column names for the coordinate axes.

    Returns:
        Copy of soma_df with added columns: structure_id, structure_acronym,
        source_area, source_layer_raw, source_layer.
    """
    out = soma_df.copy()
    coords = out[[x_col, y_col, z_col]].to_numpy(dtype=float)
    ijk = np.rint(coords / float(resolution_um)).astype(int)

    # Detect out-of-bounds before clipping so we can warn
    lo = np.array([0, 0, 0])
    hi = np.array(annotation.shape) - 1
    oob_mask = np.any((ijk < lo) | (ijk > hi), axis=1)
    n_oob = int(oob_mask.sum())
    if n_oob > 0:
        oob_ids = (
            out.loc[oob_mask, "neuron_id"].tolist()
            if "neuron_id" in out.columns
            else list(np.where(oob_mask)[0])
        )
        log.warning(
            "%d soma(s) have coordinates outside the CCF annotation volume bounds "
            "and will be clipped to the nearest edge voxel. "
            "Results for these neurons may be inaccurate. Affected neuron_ids: %s",
            n_oob, oob_ids
        )

    ijk = np.clip(ijk, lo, hi)

    out["oob_clipped"] = oob_mask

    out["structure_id"] = annotation[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    out["structure_acronym"] = out["structure_id"].map(id_to_acronym)

    area_layer = out["structure_acronym"].apply(parse_area_layer_from_acronym)
    out["source_area"] = area_layer.apply(lambda t: t[0])
    out["source_layer_raw"] = area_layer.apply(lambda t: t[1])
    out["source_layer"] = out["source_layer_raw"].apply(layer_bucket)

    return out


# -----------------------------
# Wide matrix summarizers
# -----------------------------

def _summarize_target_layers_from_wide_matrix(
    wide_csv: Path,
    target_area: str,
    value_kind: str,  # "int" or "float"
    prefix: str,      # "t" for terminals, "a" for axon length
    target_area_variants: set[str] | None = None,
) -> pd.DataFrame:
    """
    Summarize per-neuron values by cortical layer from a wide-format CSV.

    Wide CSV format:
      - First column: structure acronym (e.g., SSs2/3)
      - Remaining columns: neuron IDs
      - Values: terminal counts or axon lengths (possibly Excel-wrapped strings)

    Rows matching target_area + a valid layer suffix [1, 2/3, 4, 5, 6a, 6b] are
    selected. Values that cannot be parsed as numbers are set to 0 and a warning
    is emitted with the count of failures per column.

    Args:
        wide_csv: Path to the wide-format CSV.
        target_area: Brain area acronym to filter rows (e.g., "VISal").
        value_kind: "int" for terminal counts, "float" for axon lengths.
        prefix: Column prefix for output ("t" -> tL1, tL23 ...; "a" -> aL1, aL23 ...).
        target_area_variants: Set of lowercase acronyms (from get_area_descendants)
            that should count as the target area.  When provided, row matching
            uses this set instead of area_matches().

    Returns:
        DataFrame with columns: neuron_id, <prefix>L1, <prefix>L23, <prefix>L4,
        <prefix>L5, <prefix>L6, <prefix>Total.
    """
    wide = pd.read_csv(wide_csv)

    structure_col = wide.columns[0]
    wide[structure_col] = wide[structure_col].apply(_strip_excel_wrapping)

    neuron_cols = list(wide.columns[1:])
    norm_cols = [normalize_neuron_id(c) for c in neuron_cols]
    wide = wide.rename(columns={old: new for old, new in zip(neuron_cols, norm_cols)})

    def _row_matches_target(acronym: str) -> bool:
        area, layer = parse_area_layer_from_acronym(acronym)
        if layer is None or area is None:
            return False
        if target_area_variants is not None:
            return area.lower() in target_area_variants
        return area_matches(area, target_area)

    mask = wide[structure_col].astype(str).apply(_row_matches_target)

    n_matched = int(mask.sum())
    n_total = len(wide)
    log.debug("%s: matched %d/%d rows for target_area='%s'", wide_csv.name, n_matched, n_total, target_area)

    if n_matched == 0:
        area_prefixes = sorted({parse_area_layer_from_acronym(s)[0] for s in wide[structure_col].astype(str)})
        log.warning(
            "%s: no rows matched target_area='%s'. "
            "Available area prefixes in this file: %s",
            wide_csv.name, target_area, area_prefixes
        )

    w = wide.loc[mask].copy()

    # Layer labels
    parsed = w[structure_col].apply(parse_area_layer_from_acronym)
    w["layer_raw"] = parsed.apply(lambda t: t[1])
    w["layer"] = w["layer_raw"].apply(layer_bucket)

    neuron_ids = [c for c in w.columns if c not in (structure_col, "layer_raw", "layer")]

    # Convert values and warn on failed coercions
    for c in neuron_ids:
        w[c] = w[c].apply(_strip_excel_wrapping)
        numeric = pd.to_numeric(w[c], errors="coerce")
        n_failed = int(numeric.isna().sum())
        if n_failed > 0:
            log.warning(
                "%s: %d value(s) in column '%s' could not be parsed as %s and were set to 0.",
                wide_csv.name, n_failed, c, value_kind
            )
        if value_kind == "int":
            w[c] = numeric.fillna(0).astype(int)
        else:
            w[c] = numeric.fillna(0.0).astype(float)

    # Sum per canonical layer
    out = pd.DataFrame({"neuron_id": neuron_ids})

    def colname(L: str) -> str:
        return f"{prefix}L{L.replace('/','')}"

    for L in CANONICAL_LAYERS:
        rows_L = (w["layer"] == L)
        if rows_L.any():
            out[colname(L)] = w.loc[rows_L, neuron_ids].sum(axis=0).to_numpy()
        else:
            out[colname(L)] = 0 if value_kind == "int" else 0.0

    out[f"{prefix}Total"] = out[[colname(L) for L in CANONICAL_LAYERS]].sum(axis=1)
    return out


def summarize_target_layer_terminals(
    tp_csv: Path, target_area: str,
    target_area_variants: set[str] | None = None,
) -> pd.DataFrame:
    """Summarize terminal point counts by target cortical layer from a wide-format CSV."""
    return _summarize_target_layers_from_wide_matrix(
        tp_csv, target_area, "int", "t",
        target_area_variants=target_area_variants,
    )


def summarize_target_layer_axon_length(
    ax_csv: Path, target_area: str,
    target_area_variants: set[str] | None = None,
) -> pd.DataFrame:
    """Summarize axon length (µm) by target cortical layer from a wide-format CSV."""
    return _summarize_target_layers_from_wide_matrix(
        ax_csv, target_area, "float", "a",
        target_area_variants=target_area_variants,
    )


def autodetect_axon_length_path(terminal_csv: Path) -> Optional[Path]:
    """
    Auto-detect the axon length CSV alongside a terminal points CSV.

    If terminal CSV is named terminal_points_byNeuron_X.csv, looks for
    axon_length_byNeuron_X.csv in the same directory.

    Returns the detected path, or None if not found.
    """
    name = terminal_csv.name
    if name.startswith("terminal_points_byNeuron_"):
        candidate = terminal_csv.with_name(
            name.replace("terminal_points_byNeuron_", "axon_length_byNeuron_")
        )
        if candidate.exists():
            return candidate
    return None


# -----------------------------
# End-to-end per connection
# -----------------------------

def process_connection(cfg: dict,
                       annotation: np.ndarray,
                       id_to_acronym: Dict[int, str],
                       resolution_um: int,
                       out_dir: Path,
                       min_target_terminals: int = 3,
                       include_axon_length: bool = True,
                       structure_tree=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline for one source→target connection.

    Validates input files, loads soma coordinates, maps them to CCF,
    merges terminal/axon summaries, computes fractional distributions and
    terminal density, runs QC, and saves per-connection output CSVs.

    Args:
        cfg: Connection config dict with keys: name, direction, source_area,
             target_area, soma_csv, terminal_csv, [axon_length_csv (optional)].
        annotation: 3D CCF annotation array.
        id_to_acronym: Dict mapping structure_id (int) -> acronym (str).
        resolution_um: CCF voxel resolution in microns.
        out_dir: Directory for output CSV files.
        min_target_terminals: Neurons with fewer total target-area terminals
            are excluded from output.
        include_axon_length: If True, attempt to load and merge axon length data.
        structure_tree: AllenSDK StructureTree for accurate area matching.
            When provided, get_area_descendants() is used instead of
            prefix-based area_matches().

    Returns:
        merged_df: Per-neuron soma mapping + terminal/axon fractions.
        qc_mismatch_df: Subset of neurons whose mapped source_area differs
            from the expected source_area in cfg.
    """
    soma_csv = Path(cfg["soma_csv"])
    tp_csv = Path(cfg["terminal_csv"])

    log.info("Processing connection: %s", cfg["name"])

    # Pre-compute area descendant sets for accurate matching
    target_variants = None
    source_variants = None
    if structure_tree is not None:
        target_variants = get_area_descendants(
            cfg["target_area"], structure_tree, id_to_acronym
        )
        source_variants = get_area_descendants(
            cfg["source_area"], structure_tree, id_to_acronym
        )

    # --- Validate and load soma CSV
    somas = validate_soma_csv(soma_csv)
    validate_wide_csv(tp_csv)

    somas["neuron_id"] = somas["NeuronId"].apply(normalize_neuron_id)

    for col in ("x", "y", "z"):
        somas[col] = somas[col].apply(_strip_excel_wrapping)
        somas[col] = pd.to_numeric(somas[col], errors="coerce")

    n_before = len(somas)
    somas = somas.dropna(subset=["x", "y", "z"]).copy()
    n_dropped = n_before - len(somas)
    if n_dropped > 0:
        log.warning(
            "%s: dropped %d soma(s) with non-numeric coordinates.",
            cfg["name"], n_dropped
        )

    # Check for and remove duplicate neuron IDs
    dupes = somas["neuron_id"].duplicated(keep="first")
    if dupes.any():
        log.warning(
            "%s: %d duplicate neuron_id(s) found in soma CSV and removed (keeping first): %s",
            cfg["name"], int(dupes.sum()),
            sorted(somas.loc[dupes, "neuron_id"].unique().tolist()),
        )
        somas = somas.loc[~dupes].copy()

    # --- Map to CCF
    somas_mapped = map_somas_to_ccf(
        somas[["neuron_id", "x", "y", "z"]].copy(),
        annotation=annotation,
        id_to_acronym=id_to_acronym,
        resolution_um=resolution_um,
        x_col="x", y_col="y", z_col="z"
    )

    # --- Terminals
    t = summarize_target_layer_terminals(
        tp_csv, target_area=cfg["target_area"],
        target_area_variants=target_variants,
    )

    merged = somas_mapped.merge(t, on="neuron_id", how="left")
    tcols = [c for c in merged.columns if c.startswith("tL") or c == "tTotal"]
    merged[tcols] = merged[tcols].fillna(0).astype(int)

    # Filter neurons with too few terminals in target area
    if min_target_terminals is not None and min_target_terminals > 0:
        n_before_filter = len(merged)
        merged = merged.loc[merged["tTotal"] >= int(min_target_terminals)].copy()
        n_filtered = n_before_filter - len(merged)
        if n_filtered > 0:
            log.info(
                "%s: removed %d neuron(s) with fewer than %d terminal(s) in target area.",
                cfg["name"], n_filtered, min_target_terminals
            )

    # --- Optional: axon length
    ax_path = None
    if include_axon_length:
        ax_path = cfg.get("axon_length_csv", None)
        if ax_path:
            ax_path = Path(ax_path)
            if not ax_path.exists():
                log.warning(
                    "%s: specified axon_length_csv does not exist: %s. Skipping axon length.",
                    cfg["name"], ax_path
                )
                ax_path = None
        else:
            ax_path = autodetect_axon_length_path(tp_csv)
            if ax_path:
                log.info("%s: auto-detected axon length file: %s", cfg["name"], ax_path.name)

    if ax_path is not None:
        validate_wide_csv(ax_path)
        a = summarize_target_layer_axon_length(
            ax_path, target_area=cfg["target_area"],
            target_area_variants=target_variants,
        )
        merged = merged.merge(a, on="neuron_id", how="left")
        acols = [c for c in merged.columns if c.startswith("aL") or c == "aTotal"]
        merged[acols] = merged[acols].fillna(0.0).astype(float)
        merged["axon_length_file"] = str(ax_path)
    else:
        merged["axon_length_file"] = ""

    # --- Fractions by terminal counts
    for c in [c for c in merged.columns if c.startswith("tL")]:
        merged[c.replace("tL", "fT")] = np.where(
            merged["tTotal"] > 0,
            merged[c] / merged["tTotal"],
            np.nan
        )

    # --- Fractions by axon length (if present)
    if "aTotal" in merged.columns:
        for c in [c for c in merged.columns if c.startswith("aL")]:
            merged[c.replace("aL", "fA")] = np.where(
                merged["aTotal"] > 0,
                merged[c] / merged["aTotal"],
                np.nan
            )

        # Terminal density per layer: terminals per micron
        for L in ["1", "23", "4", "5", "6"]:
            tcol = f"tL{L}"
            acol = f"aL{L}"
            if tcol in merged.columns and acol in merged.columns:
                merged[f"dTL{L}"] = np.where(merged[acol] > 0, merged[tcol] / merged[acol], np.nan)

    # Metadata columns
    merged["connection_name"] = cfg["name"]
    merged["direction"] = cfg["direction"]
    merged["expected_source_area"] = cfg["source_area"]
    merged["expected_target_area"] = cfg["target_area"]

    # QC: flag neurons whose soma mapped to an unexpected source area.
    # When the structure tree is available, use the descendant set for
    # accurate parent→child matching; otherwise fall back to area_matches().
    if source_variants is not None:
        qc = merged.loc[
            merged["source_area"].notna()
            & ~merged["source_area"].apply(lambda x: x.lower() in source_variants)
        ].copy()
    else:
        qc = merged.loc[
            merged["source_area"].notna()
            & ~merged["source_area"].apply(lambda x: area_matches(x, cfg["source_area"]))
        ].copy()

    if len(qc) > 0:
        log.warning(
            "%s: %d neuron(s) mapped to unexpected source area. Expected: '%s'. "
            "Actual areas found: %s. See QC file for details.",
            cfg["name"], len(qc), cfg["source_area"],
            qc["source_area"].value_counts().to_dict()
        )

    log.info(
        "%s: done. %d neurons in output, %d QC mismatches.",
        cfg["name"], len(merged), len(qc)
    )

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_out = out_dir / f"{cfg['name']}__mapped_and_terminals.csv"
    qc_out = out_dir / f"{cfg['name']}__qc_source_mismatches.csv"
    merged.to_csv(merged_out, index=False)
    qc.to_csv(qc_out, index=False)

    return merged, qc


# -----------------------------
# Connection config loading
# -----------------------------

def load_connections_config(config_path: Path, data_dir: Path) -> List[dict]:
    """
    Load connection configurations from a JSON file.

    Each entry must have: name, direction, source_area, target_area,
    soma_csv, terminal_csv. Paths in soma_csv and terminal_csv are resolved
    relative to data_dir if not absolute.

    Args:
        config_path: Path to the JSON config file.
        data_dir: Base directory for resolving relative CSV paths.

    Returns:
        List of connection config dicts with resolved absolute paths.

    Raises:
        ValueError: If any connection entry is missing required keys.
    """
    with open(config_path) as f:
        connections = json.load(f)

    required_keys = {"name", "direction", "source_area", "target_area", "soma_csv", "terminal_csv"}
    for i, cfg in enumerate(connections):
        missing = required_keys - set(cfg.keys())
        if missing:
            raise ValueError(
                f"Connection #{i} ('{cfg.get('name', '?')}') in {config_path.name} "
                f"is missing required keys: {missing}"
            )
        # Resolve relative paths against data_dir
        for key in ("soma_csv", "terminal_csv", "axon_length_csv"):
            if key in cfg and cfg[key] and not Path(cfg[key]).is_absolute():
                cfg[key] = str(data_dir / cfg[key])

    return connections


# -----------------------------
# Main
# -----------------------------

_DEFAULT_CONNECTIONS_CONFIG = "connections_config.json"


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Map neuron soma coordinates to Allen CCFv3 and summarize "
            "target-layer terminal/axon distributions."
        )
    )
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Directory containing soma and terminal/axon CSVs.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Directory for output CSVs.")
    ap.add_argument("--config", type=str, default=None,
                    help=(
                        f"Path to connections JSON config file. "
                        f"Defaults to connections_config.json in --data_dir. "
                        f"Falls back to the built-in connection list if not found."
                    ))
    ap.add_argument("--ccf_cache_dir", type=str, default="allen_ccf_cache",
                    help="Directory for Allen CCF cache files (default: allen_ccf_cache).")
    ap.add_argument("--resolution_um", type=int, default=CCF_DEFAULT_RESOLUTION_UM,
                    help=f"CCF voxel resolution in microns (default: {CCF_DEFAULT_RESOLUTION_UM}).")
    ap.add_argument("--min_target_terminals", type=int, default=3,
                    help="Exclude neurons with fewer total terminals in target area (default: 3).")
    ap.add_argument("--no_axon_length", action="store_true",
                    help="Disable axon-length merging even if axon_length_byNeuron_*.csv exists.")
    ap.add_argument("--log_level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging verbosity (default: INFO).")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.ccf_cache_dir)

    # --- Load connections config
    config_path = Path(args.config) if args.config else data_dir / _DEFAULT_CONNECTIONS_CONFIG

    if not config_path.exists():
        raise FileNotFoundError(
            f"Connections config not found: {config_path}\n"
            f"Create connections_config.json in --data_dir or pass --config <path>."
        )
    log.info("Loading connections from config: %s", config_path)
    connections = load_connections_config(config_path, data_dir)

    annotation, id_to_acronym, tree = load_ccf(
        cache_dir=cache_dir,
        resolution_um=args.resolution_um,
        reference_space_key=os.path.join("annotation", "ccf_2017"),
        structure_graph_id=1
    )

    merged_all, qc_all = [], []

    for cfg in connections:
        merged, qc = process_connection(
            cfg=cfg,
            annotation=annotation,
            id_to_acronym=id_to_acronym,
            resolution_um=args.resolution_um,
            out_dir=out_dir,
            min_target_terminals=args.min_target_terminals,
            include_axon_length=(not args.no_axon_length),
            structure_tree=tree,
        )
        merged_all.append(merged)
        qc_all.append(qc)

    merged_all_df = pd.concat(merged_all, ignore_index=True) if merged_all else pd.DataFrame()
    merged_all_df.to_csv(out_dir / "ALL_CONNECTIONS__mapped_and_terminals.csv", index=False)
    log.info(
        "Saved ALL_CONNECTIONS__mapped_and_terminals.csv (%d rows total)", len(merged_all_df)
    )

    qc_all_df = pd.concat(qc_all, ignore_index=True) if qc_all else pd.DataFrame()
    qc_all_df.to_csv(out_dir / "ALL_CONNECTIONS__qc_source_mismatches.csv", index=False)
    log.info(
        "Saved ALL_CONNECTIONS__qc_source_mismatches.csv (%d rows total)", len(qc_all_df)
    )


if __name__ == "__main__":
    main()
