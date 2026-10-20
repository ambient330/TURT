"""Color tables and display-range helpers for velocity-style radar fields."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_hex

# Py-ART registers its colormaps (including "Carbone42") with matplotlib as
# a side effect of importing pyart.graph, which pulls them in from the
# "cmweather" package. Import cmweather directly too, in case pyart itself
# failed to import but cmweather is still available on its own.
try:
    import pyart  # noqa: F401
except ImportError:
    pyart = None
try:
    import cmweather  # noqa: F401  (registers "Carbone42" etc.)
except ImportError:
    pass

VEL_BOUNDS = [-82, -75, -68, -61, -54, -47, -40, -33, -26, -19, -13, -10, -5,
              -1, 1, 5, 10, 13, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82]

_CARBONE42_FALLBACK_COLORS = [
    "#00A800", "#00C800", "#00E800", "#33FF33", "#77FF77",
    "#AACFAA", "#C9B9B9", "#CC8888", "#DD5555", "#EE3333",
    "#FF0000", "#FF4400", "#FF8800",
]

ND_COLOR = "#000000"

# Matched as whole underscore-separated tokens (not raw substrings), so "VE"
# or "VR" match cleanly without false-positiving on e.g. "average_reflectivity".
_VELOCITY_NAME_TOKENS = ("velocity", "vel", "ve", "vr", "doppler", "vt")

# The color scale always spans at least +/-MIN_COLOR_RANGE_MS, even if the
# data's own Nyquist is smaller (avoids everything looking saturated).
MIN_COLOR_RANGE_MS = 60.0


def is_velocity_like_field(name):
    if not name:
        return False
    tokens = str(name).lower().replace("-", "_").split("_")
    for tok in tokens:
        if tok in _VELOCITY_NAME_TOKENS or tok.startswith("vel") or tok.startswith("doppler"):
            return True
    return False


def display_range(nyquist):
    """+/- color-scale bound: the data's own Nyquist, floored at MIN_COLOR_RANGE_MS."""
    nyquist = abs(nyquist) if nyquist is not None and np.isfinite(nyquist) else MIN_COLOR_RANGE_MS
    return max(nyquist, MIN_COLOR_RANGE_MS)


def resolve_display_bounds(nyquist, vmin_override=None, vmax_override=None):
    """(vmin, vmax) actually used for the color scale. An explicit, valid
    override wins; otherwise +/-Nyquist (floored)."""
    if (
        vmin_override is not None and vmax_override is not None
        and np.isfinite(vmin_override) and np.isfinite(vmax_override)
        and vmax_override > vmin_override
    ):
        return float(vmin_override), float(vmax_override)
    disp = display_range(nyquist)
    return -disp, disp


def get_carbone42_colormap():
    """The continuous Py-ART Carbone42 colormap, or a smooth built-in
    fallback gradient if Py-ART/cmweather isn't available."""
    for name in ("Carbone42", "pyart_Carbone42"):
        try:
            return plt.get_cmap(name)
        except (ValueError, KeyError):
            continue
    return LinearSegmentedColormap.from_list("carbone42_fallback", _CARBONE42_FALLBACK_COLORS)


def get_pyart_balance_colormap():
    for name in ("balance", "pyart_balance", "cmo.balance"):
        try:
            return plt.get_cmap(name)
        except (ValueError, KeyError):
            continue
    return LinearSegmentedColormap.from_list(
        "balance_fallback",
        ["#1B3B6F", "#3E7CB1", "#8FB8DE", "#D2E3F3", "#F7F7F7",
         "#F6D6C9", "#E8996E", "#C85D3B", "#7F1D1D"],
    )


def get_red_green_colormap():
    return LinearSegmentedColormap.from_list(
        "red_green_diverging",
        ["#67001F", "#B2182B", "#D6604D", "#F4A582", "#F7F7F7",
         "#D9F0D3", "#7FBF7B", "#3B9F4C", "#00441B"],
    )


CONTOUR_CMAP_CHOICES = {
    "Carbone42": get_carbone42_colormap,
    "pyart_balance": get_pyart_balance_colormap,
    "Red-Green": get_red_green_colormap,
}


def make_velocity_cmap_norm(nyquist=33.0, vmin_override=None, vmax_override=None):
    """(cmap, norm): continuous Carbone42 over the resolved display bounds,
    ND (black) for masked/NaN data, out-of-range values clamped to the end colors."""
    vmin, vmax = resolve_display_bounds(nyquist, vmin_override, vmax_override)
    cmap = get_carbone42_colormap().copy()
    cmap.set_bad(ND_COLOR)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    return cmap, norm


def color_for_value(v, nyquist=33.0, vmin_override=None, vmax_override=None):
    """Hex color for a single value from the continuous gradient (ND -> black)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ND_COLOR
    vmin, vmax = resolve_display_bounds(nyquist, vmin_override, vmax_override)
    cmap = get_carbone42_colormap()
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    return to_hex(cmap(norm(v)))
