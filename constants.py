"""Shared constants for the laminar analysis pipeline."""

import seaborn as sns

# ── Cortical layer definitions ──
# Canonical layers after bucketing 6a/6b -> 6
CANONICAL_LAYERS: list[str] = ["1", "2/3", "4", "5", "6"]

# Layer codes used in column names (no slash)
LAYER_ORDER: list[str] = ["1", "23", "4", "5", "6"]

# Human-readable layer labels for plots
LAYER_LABELS: list[str] = ["L1", "L2/3", "L4", "L5", "L6"]

# ── Direction definitions ──
DIRECTION_PALETTE: dict[str, str] = {
    "FF":  "#e07b54",
    "FB":  "#5b7eb5",
    "LAT": "#7ab648",
    "UNK": "#999999",
}

# ── Layer color palette ──
LAYER_PALETTE: dict[str, tuple] = dict(
    zip(LAYER_LABELS, sns.color_palette("husl", len(LAYER_LABELS)))
)


# ── Feature column helpers ──
def get_terminal_fraction_cols() -> list[str]:
    """Return terminal fraction column names: fT1, fT23, fT4, fT5, fT6."""
    return [f"fT{L}" for L in LAYER_ORDER]


def get_axon_fraction_cols() -> list[str]:
    """Return axon-length fraction column names: fA1, fA23, fA4, fA5, fA6."""
    return [f"fA{L}" for L in LAYER_ORDER]


def get_density_cols() -> list[str]:
    """Return terminal density column names: dTL1, dTL23, dTL4, dTL5, dTL6."""
    return [f"dTL{L}" for L in LAYER_ORDER]


def get_terminal_count_cols() -> list[str]:
    """Return terminal count column names: tL1, tL23, tL4, tL5, tL6."""
    return [f"tL{L}" for L in LAYER_ORDER]


def get_axon_length_cols() -> list[str]:
    """Return axon-length column names: aL1, aL23, aL4, aL5, aL6."""
    return [f"aL{L}" for L in LAYER_ORDER]


def mode_is_absolute(mode: str) -> bool:
    """Return True for absolute-feature modes (counts / lengths)."""
    return mode in {"terminals_abs", "axon_abs"}


def get_mode_value_label(mode: str) -> str:
    """Human-readable value label for plots."""
    labels = {
        "terminals": "Fraction",
        "axon": "Fraction",
        "density": "Normalized density",
        "terminals_abs": "Terminal count",
        "axon_abs": "Axon length (um)",
    }
    if mode not in labels:
        raise ValueError(f"Unknown mode {mode!r}")
    return labels[mode]


def get_feature_cols(mode: str) -> list[str]:
    """Return feature columns for the given mode."""
    if mode == "terminals":
        return get_terminal_fraction_cols()
    elif mode == "axon":
        return get_axon_fraction_cols()
    elif mode == "density":
        return get_density_cols()
    elif mode == "terminals_abs":
        return get_terminal_count_cols()
    elif mode == "axon_abs":
        return get_axon_length_cols()
    else:
        raise ValueError(
            f"Unknown mode {mode!r}; expected one of "
            "'terminals', 'axon', 'density', 'terminals_abs', or 'axon_abs'"
        )
