"""FlatTableRenderer: perpendicular range-vs-azimuth table (no curvature/skew)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .colors import resolve_display_bounds
from .render_common import (
    RenderOptions, add_legend_strip, cell_color_and_text, draw_reflectivity_contours,
)


class FlatTableRenderer:
    """Azimuth along X, range gates along Y (row 0 = nearest gate at the
    BOTTOM), plain square cells. Easy to read straight across, at the cost
    of not being true-to-shape; an arrow marks which way is toward the radar.
    """

    tag = "table"

    def render(self, fname, grid, opts: RenderOptions, rankine_fit=None, rankine_requested=False):
        """Returns the matplotlib Figure (raises on failure)."""
        nyquist = opts.nyquist
        disp_vals = grid["disp_vals"]
        rng_centers_km = grid["rng_centers_km"]
        az_centers_deg = grid["az_centers_deg"]
        range_agg, az_agg = grid["range_agg"], grid["az_agg"]
        native_range_km = grid["native_range_km"]

        nrows, ncols = disp_vals.shape  # rows = range, cols = azimuth
        if nrows == 0 or ncols == 0:
            raise ValueError(f"Empty grid selected (shape {disp_vals.shape}).")

        cell_w = min(0.6, 30.0 / ncols)
        cell_h = min(0.5, 24.0 / nrows)
        fig_w = max(8.0, min(36.0, ncols * cell_w + 3.0))
        fig_h = max(7.0, min(30.0, nrows * cell_h + 2.4))
        fig = Figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111)

        agg_factor = max(range_agg, az_agg)
        font_size = float(np.clip(min(cell_w, cell_h) * 16.0, 2.5, 10.0))
        show_text = not grid.get("ungridded")

        for r in range(nrows):        # 0 = nearest range gate (bottom)
            for c in range(ncols):     # 0 = first azimuth (left)
                color, txt, txt_color = cell_color_and_text(
                    disp_vals[r, c], nyquist, opts.vmin_override, opts.vmax_override)
                ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=color,
                                           edgecolor="black", linewidth=0.3, alpha=0.95))
                if show_text:
                    ax.text(c + 0.5, r + 0.5, txt, ha="center", va="center",
                            fontsize=font_size, color=txt_color)

        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)
        ax.set_aspect("auto")

        xg, yg = np.meshgrid(np.arange(ncols) + 0.5, np.arange(nrows) + 0.5)
        draw_reflectivity_contours(ax, xg, yg, opts.refl_grid, opts.refl_levels,
                                   opts.refl_field_name, fname, opts.contour_cmap_choice)

        tick_step = max(1, round(ncols / 18))
        xtick_idx = np.arange(0, ncols, tick_step)
        ax.set_xticks(xtick_idx + 0.5)
        ax.set_xticklabels([f"{az_centers_deg[i] % 360:.1f}" for i in xtick_idx],
                           rotation=90, fontsize=6)

        tick_step_y = max(1, round(nrows / 14))
        ytick_idx = np.arange(0, nrows, tick_step_y)
        ax.set_yticks(ytick_idx + 0.5)
        ax.set_yticklabels([f"{rng_centers_km[i]:.2f}" for i in ytick_idx], fontsize=6)

        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Range gate, distance from radar (km)")

        res_bit = (
            f", range x{range_agg}, azimuth x{az_agg} averaged" if agg_factor > 1 else
            ", ungridded (native gates, no averaging)" if grid.get("ungridded") else
            f", native res ({native_range_km * 1000:.0f} m gates)"
        )
        vmin_disp, vmax_disp = resolve_display_bounds(nyquist, opts.vmin_override, opts.vmax_override)
        if opts.vmin_override is not None and opts.vmax_override is not None:
            color_bit = f", color range {vmin_disp:g} to {vmax_disp:g}"
        elif vmax_disp > abs(nyquist) + 0.5:
            color_bit = f", color range +/-{vmax_disp:.0f} m/s"
        else:
            color_bit = ""
        refl_bit = (
            f" | {opts.refl_field_name} contours @ {', '.join(f'{v:g}' for v in opts.refl_levels)} dBZ"
            if opts.refl_grid is not None and opts.refl_levels else ""
        )
        rankine_title_bit = ""
        if rankine_fit is not None:
            rankine_title_bit = (
                f" | Rankine Vmax={rankine_fit['v_max_ms']:.1f} m/s, "
                f"Rmax={rankine_fit['r_max_km'] * 1000:.0f} m"
            )
        elif rankine_requested:
            ax.text(0.02, 0.02, "Rankine fit: no result (see status bar)",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="black"))
        ax.set_title(
            f"{fname} inside box  ({nrows} range gates x {ncols} azimuths, flat table, "
            f"Nyquist={nyquist:.1f} m/s{color_bit}{res_bit}{refl_bit}{rankine_title_bit})"
        )

        # Arrow marking which way is toward the radar (row 0 is nearest).
        ax.annotate("", xy=(-0.09, 0.02), xytext=(-0.09, 0.30),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
                    annotation_clip=False)
        ax.text(-0.09, 0.34, "Toward\nRadar", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7)

        add_legend_strip(fig, nyquist, opts.vmin_override, opts.vmax_override)
        fig.tight_layout(rect=[0.03, 0.07, 1, 1])
        return fig
