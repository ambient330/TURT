"""ResultsWindow: one popup with stacked, typeset tables for the Rankine fit,
DJWR circulation analysis and WDSS-II Patch parameters."""

import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class ResultsWindow:
    def __init__(self, parent):
        self.parent = parent

    # ---------------------------------------------------------- helpers
    @staticmethod
    def _sci_mathtext(value, sig=2):
        """Mathtext scientific notation, e.g. 248000 -> '$2.48\\times10^{5}$'."""
        if value == 0 or not np.isfinite(value):
            return "$0$"
        exp = int(np.floor(np.log10(abs(value))))
        mant = value / (10.0 ** exp)
        sign = "-" if mant < 0 else ""
        return fr"${sign}{abs(mant):.{sig}f}\times10^{{{exp}}}$"

    @staticmethod
    def _style_report_table(tbl, fontsize=11, row_scale=1.9):
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)
        tbl.scale(1, row_scale)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if r == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#e8e8e8")
            else:
                cell.set_facecolor("#ffffff" if r % 2 else "#f7f7f7")

    def _open_results_window(self, title, fig):
        win = tk.Toplevel(self.parent)
        win.title(title)
        fig_w_in, fig_h_in = fig.get_size_inches()
        win.geometry(f"{int(np.clip(fig_w_in * 100, 480, 1300))}x{int(np.clip(fig_h_in * 100, 220, 900))}")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, win).update()
        canvas.draw()
        return win

    # --------------------------------------------------------- sections
    def _rankine_section(self, rankine_fit):
        rot_sense = "cyclonic" if rankine_fit["v_max_ms"] >= 0 else "anticyclonic"
        vort_str = self._sci_mathtext(rankine_fit["core_vorticity_s"])
        center_note = "manually picked, held fixed" if rankine_fit.get("center_fixed") else "fit"
        rows = [
            [r"Center $(x_c,\,y_c)$",
             f"({rankine_fit['xc_km']:.2f}, {rankine_fit['yc_km']:.2f}) km  ({center_note})"],
            [r"$V_{max}$ (max tangential wind)",
             f"{rankine_fit['v_max_ms']:.1f} m s$^{{-1}}$  ({rot_sense})"],
            [r"$R_{max}$ (core radius)",
             f"{rankine_fit['r_max_km'] * 1000:.0f} m  (diam {2 * rankine_fit['r_max_km'] * 1000:.0f} m)"],
            [r"Background wind $(u_0,\,v_0)$",
             f"({rankine_fit['u0_ms']:.1f}, {rankine_fit['v0_ms']:.1f}) m s$^{{-1}}$"],
            [r"Core vorticity $\zeta=2V_{max}/R_{max}$", f"{vort_str} s$^{{-1}}$"],
            [r"Fit residual RMSE",
             f"{rankine_fit['rmse_ms']:.2f} m s$^{{-1}}$  ($n$={rankine_fit['n_points']})"],
        ]
        if rankine_fit.get("quality_ratio") is not None:
            rows.append([r"RMSE / $\Delta V$ (quality ratio)", f"{rankine_fit['quality_ratio']:.3f}"])
        return {"title": "Rankine Combined Vortex Fit", "col_labels": ["Parameter", "Value"],
                "rows": rows, "cell_loc": "left", "col_widths": {0: 0.52, 1: 0.48}}

    def _circulation_section(self, circulation_result, rankine_fit=None):
        r0c = circulation_result["r0_km"]
        beta0c = circulation_result["beta0_deg"] % 360.0
        center_source = circulation_result.get("center_source")
        center_bit = {"manual": "manually picked", "refined": "refined"}.get(center_source, "cell center")

        show_rmax_col = rankine_fit is not None and rankine_fit.get("r_max_km", 0) > 0
        col_labels = [r"$\rho$ (km)", r"$\Gamma_\alpha$ (m$^2$ s$^{-1}$)",
                      r"$V_{tang}$ (m s$^{-1}$)", r"$V_{infl}$ (m s$^{-1}$)", r"$\gamma$ ($^{\circ}$)"]
        if show_rmax_col:
            col_labels.append(r"$\rho / R_{max}$")

        rows = []
        for i, circ in enumerate(circulation_result["circles"]):
            row = [fr"$\rho_{{{i + 1}}}$ = {circ['rho_km']:.2f}",
                   self._sci_mathtext(circ["gamma_alpha"]),
                   f"{circ['v_tang']:.1f}", f"{circ['v_infl']:.1f}",
                   f"{circ['inflow_angle_deg']:.0f}"]
            if show_rmax_col:
                row.append(fr"${circ['rho_km'] / rankine_fit['r_max_km']:.2f}\times$")
            rows.append(row)

        subtitle = (r"Circulation center (%s): $r_0=%.2f$ km, $\beta_0=%.1f^{\circ}$"
                    % (center_bit, r0c, beta0c))
        return {"title": "DJWR Doppler Circulation Analysis", "subtitle": subtitle,
                "col_labels": col_labels, "rows": rows}

    def _patch_section(self, patch_result, rankine_fit=None):
        p = patch_result
        rav_note = " (approx: Diam/2, no fitted center)" if p.get("rav_is_fallback") else ""
        storm_bit = (
            f"storm motion applied: {p['storm_speed_ms']:.1f} m s$^{{-1}}$ from "
            f"{p['storm_dir_deg']:.0f}$^{{\\circ}}$"
            if p.get("storm_speed_ms", 0) != 0 else
            "no storm-motion correction applied (ground-relative field assumed)"
        )
        subtitle = (
            r"$V_{in}$=%.1f m s$^{-1}$ @ ($r$=%.2fkm, $\theta$=%.0f$^{\circ}$);  "
            r"$V_{out}$=%.1f m s$^{-1}$ @ ($r$=%.2fkm, $\theta$=%.0f$^{\circ}$)   [%s]"
            % (p["v_in_gr"], p["r_in_km"], p["theta_in_deg"],
               p["v_out_gr"], p["r_out_km"], p["theta_out_deg"], storm_bit)
        )
        zeta_str = self._sci_mathtext(p["zeta_bulk_s"])
        delta_str = self._sci_mathtext(p["delta_bulk_s"])
        circ_str = self._sci_mathtext(p["circulation_m2s"])
        gamma_str = self._sci_mathtext(p["gamma_bulk_m2s"])

        rows = [
            [r"$V_{diff}=V_{out}-V_{in}$", f"{p['v_diff']:.1f} m s$^{{-1}}$"],
            [r"$V_{asym}=|V_{out}|-|V_{in}|$", f"{p['v_asym']:.1f} m s$^{{-1}}$"],
            [r"$V_{av}=(|V_{out}|+|V_{in}|)/2$", f"{p['v_av']:.1f} m s$^{{-1}}$"],
            [r"$D_{iam}$ (peak-to-peak, from radar $d_{in},d_{out}$)", f"{p['diam_km'] * 1000:.0f} m"],
        ]
        if p.get("r_in_center_km") is not None and p.get("r_out_center_km") is not None:
            rows.append([r"$R_{in}$ (inbound peak, rel. to center)", f"{p['r_in_center_km'] * 1000:.0f} m"])
            rows.append([r"$R_{out}$ (outbound peak, rel. to center)", f"{p['r_out_center_km'] * 1000:.0f} m"])
        rav_value = f"{p['r_av_km'] * 1000:.0f} m"
        if (not p.get("rav_is_fallback")) and rankine_fit is not None and rankine_fit.get("r_max_km", 0) > 0:
            rav_value += fr"   (${p['r_av_km'] / rankine_fit['r_max_km']:.2f}\times R_{{max}}^{{Rankine}}$)"
        rows.append([r"$R_{av}=(R_{out}+R_{in})/2$  ($\approx R_{mv}$)" + rav_note, rav_value])
        rows.extend([
            [r"$D$ of max vorticity (single couplet -> = $D_{iam}$)", f"{p['d_max_vort_km'] * 1000:.0f} m"],
            [r"Bulk vertical vorticity $\zeta_{bulk}$", f"{zeta_str} s$^{{-1}}$"],
            [r"Bulk horizontal divergence $\delta_{bulk}$", f"{delta_str} s$^{{-1}}$"],
            [r"Bulk angular momentum $\Gamma_{bulk}=V_{av}R_{av}$", f"{gamma_str} m$^2$ s$^{{-1}}$"],
            [r"Circulation $C=(\pi/2)V_{diff}/D_{iam}$", f"{circ_str} s$^{{-1}}$"],
            [r"Centripetal acceleration $F_a=V_{av}^2/R_{av}$", f"{p['centripetal_accel_ms2']:.2f} m s$^{{-2}}$"],
        ])
        return {"title": "Patch (WDSS-II) Velocity-Couplet Parameters", "subtitle": subtitle,
                "col_labels": ["Parameter", "Value"], "rows": rows, "cell_loc": "left",
                "col_widths": {0: 0.56, 1: 0.44}, "fontsize": 10}

    # ------------------------------------------------------------- show
    def show(self, rankine_fit=None, circulation_result=None, patch_result=None):
        sections = []
        if rankine_fit is not None:
            sections.append(self._rankine_section(rankine_fit))
        if circulation_result is not None and circulation_result.get("circles"):
            sections.append(self._circulation_section(circulation_result, rankine_fit))
        if patch_result is not None:
            sections.append(self._patch_section(patch_result, rankine_fit))
        if not sections:
            return None

        row_units = [len(s["rows"]) + (2.4 if s.get("subtitle") else 1.6) for s in sections]
        fig_h = 0.52 * sum(row_units) + 0.6
        fig_w = max(7.5, max(1.55 * len(s["col_labels"]) for s in sections))

        fig = Figure(figsize=(fig_w, max(3.2, fig_h)))
        gs = fig.add_gridspec(len(sections), 1, height_ratios=row_units, hspace=0.55)
        fig.suptitle("Vortex Diagnostics Summary", fontsize=15, fontweight="bold", y=0.995)

        for i, sec in enumerate(sections):
            ax = fig.add_subplot(gs[i, 0])
            ax.axis("off")
            ax.set_title(sec["title"], fontsize=12.5, fontweight="bold", loc="left", pad=6)

            table_bbox = [0.0, 0.0, 1.0, 1.0]
            if sec.get("subtitle"):
                ax.text(0.0, 0.97, sec["subtitle"], transform=ax.transAxes,
                        fontsize=9, ha="left", va="top")
                table_bbox = [0.0, 0.0, 1.0, 0.80]

            tbl = ax.table(cellText=sec["rows"], colLabels=sec["col_labels"],
                           cellLoc=sec.get("cell_loc", "center"), colLoc="center", bbox=table_bbox)
            self._style_report_table(tbl, fontsize=sec.get("fontsize", 10.5), row_scale=1.5)
            col_widths = sec.get("col_widths")
            if col_widths:
                for (r, c), cell in tbl.get_celld().items():
                    if c in col_widths:
                        cell.set_width(col_widths[c])

        fig.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.02)
        return self._open_results_window("Vortex Diagnostics Summary", fig)
