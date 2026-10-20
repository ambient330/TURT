"""FanRenderer: true-shaped polar "fan" rendering of the box-selected sector."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .colors import resolve_display_bounds
from .polar_grid import beam_height_km
from .render_common import (
    RenderOptions, add_legend_strip, cell_color_and_text, draw_reflectivity_contours,
)


class FanRenderer:
    """Draws each cell's real quadrilateral corners (true range/azimuth
    geometry) on a plain Cartesian axes tightly cropped to the data -- a
    compact fan like a hand-drawn TVS table, rather than a full polar plot
    that wastes the canvas on the whole 360-degree circle.

    By default the sector is rotated so its central azimuth points up (a
    local down-range/cross-range frame); opts.true_oriented keeps true
    compass north-up instead. Every cell is a genuine native gate (or an
    average of a few); every cell is labeled with its value or "ND".
    """

    tag = "fan"

    def render(self, fname, grid, opts: RenderOptions, rankine_fit=None,
               circulation_result=None, rankine_requested=False, circ_requested=False):
        """Returns the matplotlib Figure (raises on failure)."""
        nyquist = opts.nyquist
        elevation_deg = opts.elevation_deg
        true_oriented = opts.true_oriented

        disp_vals = grid["disp_vals"]
        rng_edges_km = grid["rng_edges_km"]
        az_edges_deg = grid["az_edges_deg"]
        rng_centers_km = grid["rng_centers_km"]
        az_centers_deg = grid["az_centers_deg"]
        range_agg, az_agg = grid["range_agg"], grid["az_agg"]
        native_range_km = grid["native_range_km"]

        nrows, ncols = disp_vals.shape  # rows = range, cols = azimuth
        if nrows == 0 or ncols == 0:
            raise ValueError(f"Empty grid selected (shape {disp_vals.shape}).")

        # Rotate into a local frame centered on the box's own central azimuth
        # (keeps a narrow sector near-rectangular); skip if true north-up.
        mean_az_deg = 0.0 if true_oriented else float(np.median(az_centers_deg))
        rel_az_edges_rad = np.deg2rad(az_edges_deg - mean_az_deg)
        rel_az_centers_rad = np.deg2rad(az_centers_deg - mean_az_deg)

        # --- every cell's true quadrilateral corners ----------------------
        # local x = cross-range, local y = down-range
        cell_polys = []
        all_x, all_y = [], []
        min_w, min_h = np.inf, np.inf
        cx_grid = np.zeros((nrows, ncols))
        cy_grid = np.zeros((nrows, ncols))
        for r in range(nrows):
            r0, r1 = rng_edges_km[r], rng_edges_km[r + 1]
            rc = rng_centers_km[r]
            for c in range(ncols):
                t0, t1 = rel_az_edges_rad[c], rel_az_edges_rad[c + 1]
                tc = rel_az_centers_rad[c]
                xs = [r0 * np.sin(t0), r0 * np.sin(t1), r1 * np.sin(t1), r1 * np.sin(t0)]
                ys = [r0 * np.cos(t0), r0 * np.cos(t1), r1 * np.cos(t1), r1 * np.cos(t0)]
                cx, cy = rc * np.sin(tc), rc * np.cos(tc)
                cell_polys.append((r, c, list(zip(xs, ys)), cx, cy))
                cx_grid[r, c], cy_grid[r, c] = cx, cy
                all_x.extend(xs)
                all_y.extend(ys)
                min_w = min(min_w, max(xs) - min(xs))
                min_h = min(min_h, max(ys) - min(ys))

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        span_x = max(x_max - x_min, 1e-6)
        span_y = max(y_max - y_min, 1e-6)

        # Size the figure to the real data extent (equal km per inch).
        max_plot_in = 20.0
        scale = max_plot_in / max(span_x, span_y)
        fig_w = float(np.clip(span_x * scale + 2.8, 7.0, 24.0))
        fig_h = float(np.clip(span_y * scale + 2.4, 7.0, 24.0))

        fig = Figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")

        # Font size from the actual physical cell size, not the cell count.
        cell_in = min(min_w, min_h) * scale
        font_size = float(np.clip(cell_in * 90.0, 2.5, 10.0))
        show_text = not grid.get("ungridded")

        for r, c, corners, cx, cy in cell_polys:
            color, txt, txt_color = cell_color_and_text(
                disp_vals[r, c], nyquist, opts.vmin_override, opts.vmax_override)
            ax.add_patch(plt.Polygon(corners, closed=True, facecolor=color,
                                     edgecolor="black", linewidth=0.3, alpha=0.95))
            if show_text:
                ax.text(cx, cy, txt, ha="center", va="center", fontsize=font_size, color=txt_color)

        draw_reflectivity_contours(ax, cx_grid, cy_grid, opts.refl_grid, opts.refl_levels,
                                   opts.refl_field_name, fname, opts.contour_cmap_choice)

        pad_x, pad_y = 0.03 * span_x, 0.03 * span_y
        right_pad = pad_x * 6.0  # room for the height-axis labels
        ax.set_xlim(x_min - pad_x, x_max + right_pad)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.axis("off")

        # --- azimuth labels along the FAR (widest) edge, capped in count and
        # pushed outward along each column's own radial direction ----------
        max_az_labels = 18
        az_tick_step = max(1, round(ncols / max_az_labels))
        az_label_fontsize = float(np.clip((span_x / (ncols / az_tick_step)) * scale * 45.0, 4.0, 9.0))
        label_offset = pad_y * 1.8
        for c in range(0, ncols, az_tick_step):
            theta_c = rel_az_centers_rad[c]
            far_x = rng_edges_km[-1] * np.sin(theta_c)
            far_y = rng_edges_km[-1] * np.cos(theta_c)
            lx = far_x + label_offset * np.sin(theta_c)
            ly = far_y + label_offset * np.cos(theta_c)
            ax.text(lx, ly, f"{az_centers_deg[c] % 360:.1f}°",
                    ha="center", va="bottom", fontsize=az_label_fontsize, rotation=90)

        # --- range labels (left) and beam-height labels (right), anchored to
        # each row's actual edge point and offset tangentially outward ------
        theta_left = rel_az_edges_rad[0]
        theta_right = rel_az_edges_rad[-1]
        bisector_theta = rel_az_centers_rad[ncols // 2]
        edge_offset = pad_x * 1.6
        left_out = (-np.cos(theta_left), np.sin(theta_left))
        right_out = (np.cos(theta_right), -np.sin(theta_right))

        rng_tick_step = max(1, round(nrows / 12))
        for r in range(0, nrows, rng_tick_step):
            rc = rng_centers_km[r]
            lx = rc * np.sin(theta_left) + edge_offset * left_out[0]
            ly = rc * np.cos(theta_left) + edge_offset * left_out[1]
            ax.text(lx, ly, f"{rc:.2f} km", ha="right", va="center", fontsize=7)
            height_km = beam_height_km(rc, elevation_deg)
            rx = rc * np.sin(theta_right) + edge_offset * right_out[0]
            ry = rc * np.cos(theta_right) + edge_offset * right_out[1]
            ax.text(rx, ry, f"{height_km:.2f} km", ha="left", va="center", fontsize=7)
        far_rc = rng_edges_km[-1]
        hx = far_rc * np.sin(theta_right) + edge_offset * right_out[0]
        hy = far_rc * np.cos(theta_right) + edge_offset * right_out[1]
        ax.text(hx, hy, "Height\n(km)", ha="left", va="bottom", fontsize=7)

        # --- "toward radar" arrow along the sector's bisector -------------
        perp = (-np.cos(bisector_theta), np.sin(bisector_theta))
        side_margin = pad_x * 3.2
        far_pt = (rng_edges_km[-1] * np.sin(bisector_theta) + perp[0] * side_margin,
                  rng_edges_km[-1] * np.cos(bisector_theta) + perp[1] * side_margin)
        near_pt = (rng_edges_km[0] * np.sin(bisector_theta) + perp[0] * side_margin,
                   rng_edges_km[0] * np.cos(bisector_theta) + perp[1] * side_margin)
        ax.annotate("", xy=near_pt, xytext=far_pt, xycoords="data", textcoords="data",
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5), annotation_clip=False)
        mid_pt = ((far_pt[0] + near_pt[0]) / 2.0, (far_pt[1] + near_pt[1]) / 2.0)
        ax.text(mid_pt[0] + perp[0] * pad_x * 1.2, mid_pt[1] + perp[1] * pad_x * 1.2,
                "Toward\nRadar", ha="center", va="center", fontsize=7, rotation=0)

        # --- title pieces --------------------------------------------------
        agg_factor = max(range_agg, az_agg)
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
        orient_bit = "true north-up" if true_oriented else f"centered {mean_az_deg % 360:.1f}°"
        refl_bit = (
            f" | {opts.refl_field_name} contours @ {', '.join(f'{v:g}' for v in opts.refl_levels)} dBZ"
            if opts.refl_grid is not None and opts.refl_levels else ""
        )
        rankine_title_bit = ""
        circ_title_bit = ""

        # --- Rankine overlay: center marker + core-radius circle, rotated
        # into this plot's local frame (distances preserved) ----------------
        if rankine_fit is not None:
            mean_az_rad = np.deg2rad(mean_az_deg)
            xc_local = (rankine_fit["xc_km"] * np.cos(mean_az_rad)
                        - rankine_fit["yc_km"] * np.sin(mean_az_rad))
            yc_local = (rankine_fit["yc_km"] * np.cos(mean_az_rad)
                        + rankine_fit["xc_km"] * np.sin(mean_az_rad))
            r_max = rankine_fit["r_max_km"]
            ax.plot(xc_local, yc_local, marker="x", color="black",
                    markersize=11, markeredgewidth=2.2, zorder=25)
            ax.add_patch(plt.Circle((xc_local, yc_local), r_max, fill=False,
                                    edgecolor="black", linewidth=1.6, linestyle="--", zorder=24))
            rankine_title_bit = (f" | Rankine Vmax={rankine_fit['v_max_ms']:.1f} m/s, "
                                 f"Rmax={r_max * 1000:.0f} m")
        elif rankine_requested:
            ax.text(0.02, 0.02, "Rankine fit: no result (see status bar)",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="black"))

        # --- DJWR circulation center + labeled dotted circles (Fig. 3
        # style: black x with concentric circles rho1, rho2, rho3, ...) ------
        if circulation_result is not None and circulation_result.get("circles"):
            r0c = circulation_result["r0_km"]
            beta0c = circulation_result["beta0_deg"]
            beta0_rel_rad = np.deg2rad(beta0c - mean_az_deg)
            xc_c = r0c * np.sin(beta0_rel_rad)
            yc_c = r0c * np.cos(beta0_rel_rad)

            ax.plot(xc_c, yc_c, marker="x", color="black",
                    markersize=12, markeredgewidth=2.4, zorder=27)
            for i, circ in enumerate(circulation_result["circles"]):
                rho = circ["rho_km"]
                ax.add_patch(plt.Circle((xc_c, yc_c), rho, fill=False,
                                        edgecolor="red", linewidth=1.2, linestyle=":", zorder=26))
                label_theta = np.deg2rad(20.0 + i * 18.0)
                ax.text(xc_c + rho * np.sin(label_theta), yc_c + rho * np.cos(label_theta),
                        f"\u03c1{i + 1}", color="red", fontsize=8, ha="center", va="center", zorder=28,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.75, edgecolor="none"))
            circ_title_bit = f" | CC r0={r0c:.2f} km, \u03b20={beta0c % 360:.1f}\u00b0"
        elif circ_requested:
            ax.text(0.98, 0.02, "Circulation center: no result (see status bar)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85, edgecolor="black"))

        fig.suptitle(
            f"{fname} inside box  ({nrows} range gates x {ncols} azimuths, "
            f"{orient_bit}, Nyquist={nyquist:.1f} m/s{color_bit}{res_bit}"
            f"{refl_bit}{rankine_title_bit}{circ_title_bit})",
            fontsize=10,
        )

        add_legend_strip(fig, nyquist, opts.vmin_override, opts.vmax_override)
        fig.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.14)
        return fig
