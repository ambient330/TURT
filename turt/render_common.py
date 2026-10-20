"""Shared rendering helpers: options, per-cell styling, legend strip, contour overlay."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .colors import (
    CONTOUR_CMAP_CHOICES, ND_COLOR, VEL_BOUNDS, color_for_value,
    get_carbone42_colormap, is_velocity_like_field, make_velocity_cmap_norm,
)


@dataclass
class RenderOptions:
    nyquist: float = 33.0
    elevation_deg: float = 0.5
    refl_grid: Optional[np.ndarray] = None
    refl_field_name: Optional[str] = None
    refl_levels: List[float] = field(default_factory=list)
    vmin_override: Optional[float] = None
    vmax_override: Optional[float] = None
    contour_cmap_choice: str = "Carbone42"
    true_oriented: bool = False


def cell_color_and_text(v, nyquist, vmin_override=None, vmax_override=None):
    """(fill color, label text, label text color) for one possibly-masked value."""
    masked = np.ma.is_masked(v)
    vf = None if masked else float(v)
    if not masked and vf is not None and np.isnan(vf):
        masked = True
    color = color_for_value(vf, nyquist, vmin_override, vmax_override) if not masked else ND_COLOR
    txt = f"{vf:.0f}" if not masked else "ND"
    txt_color = "white" if (not masked and abs(vf) > 20) or masked else "black"
    return color, txt, txt_color


def add_legend_strip(fig, nyquist, vmin_override=None, vmax_override=None):
    """Bottom legend strip: smooth Carbone42 gradient + ND end cap."""
    cmap_cont, norm_cont = make_velocity_cmap_norm(nyquist, vmin_override, vmax_override)
    vmin, vmax = norm_cont.vmin, norm_cont.vmax

    legend_ax = fig.add_axes([0.15, 0.03, 0.68, 0.025])
    gradient = np.linspace(vmin, vmax, 256).reshape(1, -1)
    legend_ax.imshow(gradient, aspect="auto", cmap=cmap_cont, norm=norm_cont,
                     extent=[vmin, vmax, 0, 1])
    tick_vals = [b for b in VEL_BOUNDS if vmin <= b <= vmax]
    if not tick_vals:
        # Custom range that doesn't overlap the velocity tick set.
        tick_vals = list(np.linspace(vmin, vmax, 6))
    legend_ax.set_xticks(tick_vals)
    legend_ax.set_xticklabels([f"{b:g}" for b in tick_vals], fontsize=6)
    legend_ax.set_yticks([])

    nd_ax = fig.add_axes([0.05, 0.03, 0.06, 0.025])
    nd_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=ND_COLOR))
    nd_ax.set_xlim(0, 1)
    nd_ax.set_ylim(0, 1)
    nd_ax.set_xticks([0.5])
    nd_ax.set_xticklabels(["ND"], fontsize=6)
    nd_ax.set_yticks([])


def draw_reflectivity_contours(ax, x_grid, y_grid, refl_grid, refl_levels,
                               refl_field_name=None, primary_field_name=None,
                               contour_cmap_choice="Carbone42"):
    """Contour overlay of a second field on the same cell-center grid as the
    fill, so lines align cell-for-cell. Drawn in alternating black/white
    (with opposite-color outlines) when the person chose "Black & White" or
    when both fields are velocity-like; otherwise shaded by the chosen
    colormap. Higher levels get thicker lines. Best-effort: never blocks the
    primary render."""
    if refl_grid is None or not refl_levels:
        return
    try:
        z = np.ma.filled(refl_grid.astype(float), np.nan)
        if not np.isfinite(z).any():
            return
        levels = sorted(set(float(v) for v in refl_levels))
        linewidths = [float(np.clip(0.6 + 0.2 * i, 0.6, 1.8)) for i in range(len(levels))]

        force_bw = (
            contour_cmap_choice == "Black & White"
            or (is_velocity_like_field(primary_field_name) and is_velocity_like_field(refl_field_name))
        )

        if force_bw:
            for i, lvl in enumerate(levels):
                col = "black" if i % 2 == 0 else "white"
                outline = "white" if col == "black" else "black"
                cs = ax.contour(x_grid, y_grid, z, levels=[lvl], colors=col,
                                linewidths=[linewidths[i]], alpha=0.95)
                stroke = [patheffects.Stroke(linewidth=linewidths[i] + 1.0, foreground=outline),
                          patheffects.Normal()]
                # matplotlib >=3.8: single artist; older: list of collections.
                if hasattr(cs, "set_path_effects"):
                    cs.set_path_effects(stroke)
                else:
                    for coll in cs.collections:
                        coll.set_path_effects(stroke)
                ax.clabel(cs, inline=True, fontsize=8, fmt="%g")
        else:
            cmap = CONTOUR_CMAP_CHOICES.get(contour_cmap_choice, get_carbone42_colormap)()
            if is_velocity_like_field(refl_field_name):
                vmax = max(abs(v) for v in levels) or 1.0
                norm = Normalize(vmin=-vmax, vmax=vmax)
            else:
                vmin, vmax = min(levels), max(levels)
                if vmin == vmax:
                    vmin, vmax = vmin - 1.0, vmax + 1.0
                norm = Normalize(vmin=vmin, vmax=vmax)
            cs = ax.contour(x_grid, y_grid, z, levels=levels, cmap=cmap, norm=norm,
                            linewidths=linewidths, alpha=0.9)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%g")
    except Exception:
        import traceback
        print(traceback.format_exc())
