"""ArcCrossSectionWindow: interactive azimuthal cross-section diagnostics."""

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class ArcCrossSectionWindow:
    """A single plot axis with a Product dropdown (raw Doppler vs Rankine
    model, tangential velocity, shear/curvature/total vorticity, angular
    momentum, cumulative circulation, cyclostrophic pressure deficit).

    Most products draw one colored line per arc radius over azimuth. "Raw
    Doppler vs Rankine Model" instead shows ONE arc at a time (second
    dropdown), in azimuthal distance (m), with red "Doppler" / blue "Model"
    lines and an RMSE title -- inversion-free, so it's a clean goodness-of-fit
    check (needs a Rankine fit). "Tangential Velocity" overlays the Rankine
    model's axisymmetric Vt(rho) as a dashed horizontal line per radius.
    """

    def __init__(self, parent, arcs, xc_km, yc_km, air_density,
                 rankine_fit=None, field_name="", extrema_result=None):
        if not arcs:
            raise ValueError("No arcs to display.")
        self.arcs = arcs
        self.xc_km, self.yc_km = xc_km, yc_km
        self.air_density = air_density
        self.rankine_fit = rankine_fit
        self.field_name = field_name
        self.extrema_result = extrema_result

        have_rankine_vd = rankine_fit is not None and any(a.get("vd_rankine") is not None for a in arcs)

        products = []  # (key, dropdown label, axis/title text)
        if have_rankine_vd:
            products.append(("vd_compare", "Raw Doppler vs Rankine Model", ""))
        products += [
            ("vt", "Tangential Velocity (Vt) vs Rankine model",
             r"Tangential Velocity  $V_t$  (m s$^{-1}$)"),
            ("shear_vorticity", "Shear Vorticity",
             r"Shear Vorticity  $\zeta_{shear}=\frac{V_t(\rho+\delta)-V_t(\rho-\delta)}{2\delta}$  (s$^{-1}$)"),
            ("curvature_vorticity", "Curvature Vorticity",
             r"Curvature Vorticity  $\zeta_{curv}=V_t/\rho$  (s$^{-1}$)"),
            ("total_vorticity", "Vertical Vorticity (total)",
             r"Vertical Vorticity  $\zeta=\zeta_{shear}+\zeta_{curv}$  (s$^{-1}$)"),
            ("angular_momentum", "Angular Momentum",
             r"Angular Momentum  $M=V_t\,\rho$  (m$^2$ s$^{-1}$)"),
            ("circulation_cumulative", "Cumulative Circulation",
             r"Cumulative Circulation  $\Gamma(\theta)=\oint V_t\,ds$  (m$^2$ s$^{-1}$)"),
            ("pressure_deficit", "Cyclostrophic Pressure Deficit",
             r"Cyclostrophic Pressure Deficit  $\Delta p=\frac{1}{2}\rho_{air}V_t^2$  (Pa)"),
        ]
        self.combo_labels = [p[1] for p in products]
        self.key_by_label = {p[1]: p[0] for p in products}
        self.title_by_label = {p[1]: p[2] for p in products}
        radius_labels = [f"{a['rho_km']:.2f} km" for a in arcs]
        self.arc_by_radius_label = dict(zip(radius_labels, arcs))

        self.win = tk.Toplevel(parent)
        self.win.title("Azimuthal Cross-Section Diagnostics")
        self.win.geometry("1020x760")

        ctrl = ttk.Frame(self.win)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(ctrl, text="Product:").pack(side=tk.LEFT, padx=(0, 6))
        self.product_var = tk.StringVar(value=self.combo_labels[0])
        self.combo = ttk.Combobox(ctrl, textvariable=self.product_var, values=self.combo_labels,
                                  state="readonly", width=38)
        self.combo.pack(side=tk.LEFT)

        # Shown only for the single-arc "Raw Doppler vs Rankine Model" product.
        self.radius_label_widget = ttk.Label(ctrl, text="Arc radius:")
        self.radius_var = tk.StringVar(value=radius_labels[0])
        self.radius_combo = ttk.Combobox(ctrl, textvariable=self.radius_var, values=radius_labels,
                                         state="readonly", width=10)

        note = ("" if rankine_fit is not None else
                "(No Rankine fit available -- enable it for a model overlay/comparison.)")
        ttk.Label(ctrl, text=note, foreground="#666").pack(side=tk.LEFT, padx=(10, 0))

        self.fig = Figure(figsize=(9.6, 6.4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self.win).update()

        cmap = plt.get_cmap("viridis")
        n_arcs = max(len(arcs), 1)
        self.colors = [cmap(0.12 + 0.76 * i / max(n_arcs - 1, 1)) for i in range(n_arcs)]
        self.beta0_deg = float(np.degrees(np.arctan2(xc_km, yc_km))) % 360.0
        self.r0_km = float(np.hypot(xc_km, yc_km))

        self.combo.bind("<<ComboboxSelected>>", self.redraw)
        self.radius_combo.bind("<<ComboboxSelected>>", self.redraw)
        self.redraw()

    # ---------------------------------------------------------------
    def redraw(self, *_args):
        label = self.product_var.get()
        key = self.key_by_label.get(label, self.key_by_label[self.combo_labels[0]])
        self.ax.clear()
        if key == "vd_compare":
            self._draw_doppler_vs_model()
        else:
            self._draw_arc_lines(key, self.title_by_label.get(label, ""))
        self.canvas.draw_idle()

    def _draw_doppler_vs_model(self):
        ax, fig = self.ax, self.fig
        self.radius_label_widget.pack(side=tk.LEFT, padx=(14, 4))
        self.radius_combo.pack(side=tk.LEFT)

        arc = self.arc_by_radius_label.get(self.radius_var.get(), self.arcs[0])
        rho_m = arc["rho_km"] * 1000.0
        s_m = rho_m * np.deg2rad(arc["theta_deg"])  # azimuthal distance along the arc (m)

        doppler_speed = np.abs(arc["vd_raw"])
        ax.plot(s_m, doppler_speed, color="red", linewidth=1.0, marker="^", markersize=4,
                markerfacecolor="red", label="Doppler")

        rmse_val = None
        if arc.get("vd_rankine") is not None:
            model_speed = np.abs(arc["vd_rankine"])
            ax.plot(s_m, model_speed, color="blue", linewidth=1.0, marker="^", markersize=4,
                    markerfacecolor="blue", label="Model (Rankine)")
            rmse_val = float(np.sqrt(np.nanmean((doppler_speed - model_speed) ** 2)))

        # Mark the outbound (+) / inbound (-) peak bins from the arc-extrema
        # search if this arc's radius was one of the searched ones.
        if self.extrema_result is not None:
            match = next((pr for pr in self.extrema_result["per_radius"]
                          if abs(pr["rho_km"] - arc["rho_km"]) < 1e-6), None)
            if match is not None:
                for xy, vd_val, mlabel in ((match["pos_xy_km"], match["pos_vd"], "+"),
                                           (match["neg_xy_km"], match["neg_vd"], "\u2212")):
                    dx, dy = xy[0] - self.xc_km, xy[1] - self.yc_km
                    s_pt = rho_m * (float(np.arctan2(dx, dy)) % (2.0 * np.pi))
                    ax.plot(s_pt, abs(vd_val), marker="x", color="black",
                            markersize=10, markeredgewidth=2.2, zorder=10)
                    ax.annotate(mlabel, (s_pt, abs(vd_val)), textcoords="offset points",
                                xytext=(0, 8), fontsize=10, fontweight="bold",
                                color="black", ha="center", zorder=10)

        ax.set_xlabel("Azimuthal Distance (m)", fontsize=10)
        ax.set_ylabel("Wind Speed (m/s)", fontsize=10)
        ax.set_xlim(0, s_m[-1] if len(s_m) else None)
        ax.grid(axis="y", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)

        rf = self.rankine_fit
        rmax_m = rf["r_max_km"] * 1000.0 if rf is not None else float("nan")
        vmax_ms = rf["v_max_ms"] if rf is not None else float("nan")
        bits = [f"range={arc['rho_km']:.2f} km", f"Rmax={rmax_m:.0f} m", f"Vmax={vmax_ms:.1f} m/s"]
        if rmse_val is not None:
            bits.append(f"RMSE={rmse_val:.1f} m/s")
        prefix = f"{self.field_name}: " if self.field_name else ""
        ax.set_title(f"{prefix}Azimuthal Cross Section of Doppler and Model Wind Speeds\n"
                     f"({', '.join(bits)})", fontsize=11)
        fig.suptitle("")
        fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.10)

    def _draw_arc_lines(self, key, title):
        ax, fig, rf = self.ax, self.fig, self.rankine_fit
        self.radius_label_widget.pack_forget()
        self.radius_combo.pack_forget()

        for arc, color in zip(self.arcs, self.colors):
            ax.plot(arc["theta_deg"], arc[key], color=color, linewidth=1.8,
                    label=fr"$\rho$={arc['rho_km']:.2f} km")
            if key == "vt" and rf is not None and rf.get("r_max_km", 0) > 0:
                rmax, vmax, rho = rf["r_max_km"], rf["v_max_ms"], arc["rho_km"]
                vt_rankine = vmax * rho / rmax if rho <= rmax else vmax * rmax / rho
                ax.axhline(vt_rankine, color=color, linewidth=1.3, linestyle="--", alpha=0.75)

        ax.axhline(0.0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel("Azimuth around vortex center (deg from N)", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        if key == "vt" and rf is not None:
            ax.text(0.01, 0.02, "dashed = Rankine-fit model $V_t(\\rho)$ (axisymmetric)",
                    transform=ax.transAxes, fontsize=7.5, color="#555")
        fig.suptitle(
            fr"center $(x_c,y_c)$=({self.xc_km:.2f}, {self.yc_km:.2f}) km  "
            fr"($r_0$={self.r0_km:.2f} km, $\beta_0$={self.beta0_deg:.1f}$^{{\circ}}$),  "
            fr"$\rho_{{air}}$={self.air_density:.2f} kg m$^{{-3}}$",
            fontsize=10.5,
        )
        fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.10)
