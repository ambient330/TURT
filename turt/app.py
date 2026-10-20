"""TurtApp: the TURT (Tornadic-vortex Utility Radar Toolkit) Tk application shell,
wiring the independent pieces together."""

import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")  # must precede any pyplot import in the package modules below

import numpy as np  # noqa: E402

from .analysis import ManualInputs, VortexAnalysis  # noqa: E402
from .arc_window import ArcCrossSectionWindow  # noqa: E402
from .control_panel import ControlPanel  # noqa: E402
from .fan_export import FanRadarExporter  # noqa: E402
from .fan_renderer import FanRenderer  # noqa: E402
from .figure_window import FigurePresenter  # noqa: E402
from .flat_renderer import FlatTableRenderer  # noqa: E402
from .polar_grid import block_reduce_mean, prepare_polar_grid  # noqa: E402
from .ppi_view import PPIView  # noqa: E402
from .radar_session import RadarSession  # noqa: E402
from .render_common import RenderOptions  # noqa: E402
from .results_window import ResultsWindow  # noqa: E402
from .selection import build_fan_selection  # noqa: E402

RADAR_FILETYPES = [
    ("All radar files", "*.*"),
    ("NEXRAD Level 2 (Archive2)", "*.gz *.Z *.ar2v *.V06 *.V03"),
    ("CfRadial / NetCDF", "*.nc *.nc4"),
    ("Sigmet/IRIS", "*.sigmet *.RAW*"),
    ("MDV", "*.mdv"),
    ("UF", "*.uf"),
    ("All files", "*"),
]

APP_TITLE = "TURT -- Tornadic-vortex Utility Radar Toolkit"


class TurtApp(tk.Tk):
    def __init__(self, radar_path=None):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1050x800")

        self.session = RadarSession()
        self._last_fan = None   # (FanSelection, grid) from the most recent zoom, for export

        self.panel = ControlPanel(self, {
            "import_file": self.open_file,
            "refresh": self.refresh_plot,
            "clear_box": self.clear_boundary,
            "zoom": self.close_and_zoom,
            "diagnostics": self.show_diagnostics,
            "pick_center": lambda: self.view.picker.start_center_pick(),
            "clear_center": lambda: self.view.picker.clear_all(),
            "pick_pos_peak": lambda: self.view.picker.start_pick_pos_peak(),
            "pick_neg_peak": lambda: self.view.picker.start_pick_neg_peak(),
            "export_fan": self.export_fan,
        })
        self.view = PPIView(self, self.notify, on_labels_changed=self.panel.set_manual_labels)
        self.presenter = FigurePresenter(self)

        if radar_path:
            self.load_radar(radar_path)
        else:
            self.after(200, self.prompt_import_at_startup)

    # ----------------------------------------------------------- status
    def notify(self, message, clear_after_ms=None):
        self.panel.status_var.set(message)
        if clear_after_ms:
            self.after(clear_after_ms, lambda: self.panel.status_var.set(""))

    # ---------------------------------------------------------- loading
    def open_file(self):
        path = filedialog.askopenfilename(title="Import radar file", filetypes=RADAR_FILETYPES)
        if path:
            self.load_radar(path)

    def prompt_import_at_startup(self):
        use_file = messagebox.askyesno(
            "Import radar file?",
            "Would you like to import a radar file now?\n\n"
            "Choose 'Yes' to browse for a file (NEXRAD Level 2 (Archive2)/"
            "NetCDF, Sigmet, MDV, UF, etc. -- anything Py-ART can read).\n"
            "Choose 'No' to start with a synthetic demo TVS instead.",
        )
        if use_file:
            path = filedialog.askopenfilename(title="Import radar file", filetypes=RADAR_FILETYPES)
            self.load_radar(path if path else None)
        else:
            self.load_radar(None)

    def load_radar(self, path):
        self.view.picker.clear_all()
        self._last_fan = None
        outcome = self.session.load(path)
        self.notify(outcome.status)
        if outcome.error:
            messagebox.showerror("Load error", outcome.error)
        if outcome.warning:
            messagebox.showwarning(*outcome.warning)
        if outcome.loaded_path:
            self.title(f"{APP_TITLE} -- {outcome.loaded_path}")

        fields = self.session.fields
        self.panel.set_fields(fields, "velocity" if "velocity" in fields else (fields[0] if fields else ""),
                              self.session.guess_reflectivity_field())
        self.panel.set_nsweeps(self.session.nsweeps)

        self.clear_boundary()
        self.refresh_plot()

        if outcome.loaded_path:
            text = self.session.degenerate_warning(self.session.clamp_sweep(self.panel.sweep_var.get()))
            if text:
                messagebox.showwarning("Data looks too small / thin", text)

    # ------------------------------------------------------------- plot
    def refresh_plot(self):
        if self.session.radar is None:
            return
        fname = self.panel.field_var.get()
        if fname not in self.session.radar.fields:
            return
        vmin_o, vmax_o = self.panel.parse_color_range_override()
        self.view.show_sweep(self.session, fname, self.panel.sweep_var.get(), vmin_o, vmax_o)

    def show_diagnostics(self):
        messagebox.showinfo("Sweep diagnostics", self.session.last_diag)

    def clear_boundary(self):
        self.view.box.clear()

    # ------------------------------------------------------- zoom / fan
    def _build_contour_grid(self, sel, grid):
        """Aggregate the contour field onto EXACTLY the primary field's
        range/azimuth grid so contours line up cell-for-cell with the fill."""
        refl_name = self.panel.refl_field_var.get()
        if not refl_name or refl_name == "None" or refl_name not in self.session.radar.fields:
            return None, refl_name, []
        levels = self.panel.parse_refl_levels()
        try:
            refl_full = self.session.field_slice(refl_name, sel.s_start, sel.s_end)
            refl_t = np.ma.transpose(sel.crop(refl_full))
            refl_grid, _, _ = block_reduce_mean(refl_t, grid["range_agg"], grid["az_agg"])
            return refl_grid, refl_name, levels
        except Exception:
            import traceback
            print(traceback.format_exc())
            self.notify(f"Could not aggregate reflectivity field '{refl_name}' -- showing velocity only.")
            return None, refl_name, levels

    def close_and_zoom(self):
        if self.session.radar is None:
            return
        if not self.view.box.has_box:
            self.notify("Drag out a box on the PPI first.", 2000)
            return

        fname = self.panel.field_var.get()
        sel, err = build_fan_selection(self.session, self.panel.sweep_var.get(), fname,
                                       self.view.box.points)
        if sel is None:
            self.notify(err, 2000)
            return

        try:
            grid = prepare_polar_grid(
                sel.sub, sel.az_sub, sel.rng_sub,
                user_multiplier=self.panel.supersample_factor(),
                force_native=self.panel.ungridded,
            )
        except Exception:
            import traceback
            print(traceback.format_exc())
            self.notify("Could not build a range/azimuth grid from this selection.", 3000)
            return

        refl_grid, refl_name, refl_levels = self._build_contour_grid(sel, grid)
        vmin_o, vmax_o = self.panel.parse_color_range_override()

        picker = self.view.picker
        manual = ManualInputs(picker.center_km, picker.pos_peak_km, picker.neg_peak_km)
        aopts = self.panel.analysis_options()
        results = VortexAnalysis().run(sel, aopts, manual)

        self.view.draw_arc_extrema_markers(results.extrema_result)
        if results.arcs:
            try:
                ArcCrossSectionWindow(
                    self, results.arcs, results.arc_center_km[0], results.arc_center_km[1],
                    results.arc_air_density, rankine_fit=results.rankine_fit,
                    field_name=fname, extrema_result=results.extrema_result)
            except Exception:
                import traceback
                print(traceback.format_exc())

        ropts = RenderOptions(
            nyquist=sel.nyquist, elevation_deg=sel.elev_deg,
            refl_grid=refl_grid, refl_field_name=refl_name, refl_levels=refl_levels,
            vmin_override=vmin_o, vmax_override=vmax_o,
            contour_cmap_choice=self.panel.contour_cmap_var.get(),
            true_oriented=self.panel.true_oriented,
        )
        try:
            if self.panel.use_polar_fan:
                renderer = FanRenderer()
                fig = renderer.render(
                    fname, grid, ropts, rankine_fit=results.rankine_fit,
                    circulation_result=results.circulation_result,
                    rankine_requested=aopts.fit_rankine, circ_requested=aopts.circ_center)
            else:
                renderer = FlatTableRenderer()
                fig = renderer.render(fname, grid, ropts, rankine_fit=results.rankine_fit,
                                      rankine_requested=aopts.fit_rankine)
            save_msg = self.presenter.present(fig, fname, renderer.tag)
        except Exception:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            messagebox.showerror(
                "Could not render/save the grid",
                f"An error occurred while building or saving the TVS grid image:\n\n{tb}")
            return

        # One consolidated popup for Rankine / DJWR / Patch numbers.
        if results.has_summary_tables():
            try:
                ResultsWindow(self).show(results.rankine_fit, results.circulation_result,
                                         results.patch_result)
            except Exception:
                import traceback
                print(traceback.format_exc())

        self._last_fan = (sel, grid)
        if self.panel.export_on_zoom_var.get():
            self.export_fan()

        agg = max(grid["range_agg"], grid["az_agg"])
        res_note = (
            f"range averaged {grid['range_agg']}x, azimuth averaged {grid['az_agg']}x" if agg > 1 else
            f"native resolution ({grid['native_range_km'] * 1000:.0f} m gates)"
        )
        msg = (f"Zoomed to a {grid['disp_vals'].shape[0]}x{grid['disp_vals'].shape[1]} "
               f"range x azimuth grid inside box ({res_note}).")
        if results.messages:
            self.notify(f"{save_msg}  {msg}  Notes: " + " | ".join(results.messages), 15000)
        else:
            self.notify(f"{save_msg}  {msg}", 6000)

    # ----------------------------------------------------- radar export
    def export_fan(self):
        """Write the most recently zoomed fan sector out as a CfRadial radar file."""
        if self._last_fan is None:
            messagebox.showinfo("Nothing to export",
                                "Draw a box and click 'Zoom to TVS Grid' first -- the export "
                                "writes that fan sector as radar data.")
            return
        if not FanRadarExporter.is_available():
            messagebox.showwarning("Py-ART not found",
                                   "Exporting radar data needs Py-ART.\n\nInstall with: pip install arm_pyart")
            return

        sel, grid = self._last_fan
        exporter = FanRadarExporter(self.session.radar)
        path = filedialog.asksaveasfilename(
            title="Export fan as radar file",
            initialfile=exporter.default_filename(sel, grid),
            defaultextension=".nc",
            filetypes=[("CfRadial NetCDF", "*.nc"), ("All files", "*")],
        )
        if not path:
            return
        names = None if self.panel.export_all_fields_var.get() else [sel.field_name]
        try:
            out = exporter.export(sel, grid, path, field_names=names)
        except Exception:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            messagebox.showerror("Export failed", f"Could not write the radar file:\n\n{tb}")
            return
        msg = (f"Exported fan sector ({out.nrays} rays x {out.ngates} gates, "
               f"{len(out.fields)} field(s)) to {path}")
        if self.panel.export_dummy_var.get():
            dpath = exporter.dummy_path(path)
            fill = "displayed" if self.panel.dummy_fill_displayed_var.get() else "native"
            try:
                dout = exporter.export_dummy(sel, grid, dpath, field_names=names, fill=fill)
            except Exception:
                import traceback
                tb = traceback.format_exc()
                print(tb)
                messagebox.showerror("Dummy-radar export failed",
                                     f"The fan file was written, but the full-sweep dummy radar "
                                     f"could not be:\n\n{tb}")
            else:
                msg += f"  |  dummy radar ({dout.nrays} rays x {dout.ngates} gates, fill={fill}) to {dpath}"
        self.notify(msg, 10000)


def main(path=None):
    TurtApp(path).mainloop()
