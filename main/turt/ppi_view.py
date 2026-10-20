"""PPIView: the main PPI canvas. Owns the figure/mesh and routes mouse events
to the BoxSelector or ManualPicker."""

import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .box_selector import BoxSelector
from .colors import make_velocity_cmap_norm
from .manual_picks import ManualPicker


class PPIView:
    def __init__(self, parent, notify, on_labels_changed=None):
        self.fig = Figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Distance East of Radar (km)")
        self.ax.set_ylabel("Distance North of Radar (km)")
        self.ax.set_aspect("equal")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, parent).update()

        self.box = BoxSelector(self.ax, self.canvas, notify)
        self.picker = ManualPicker(self.ax, self.canvas, notify, on_labels_changed)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self.box.on_motion)
        self.canvas.mpl_connect("button_release_event", self.box.on_release)

        self._mesh = None
        self._colorbar = None
        self._arc_extrema_artists = []

    # ------------------------------------------------------------ events
    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.picker.handle_click(event.xdata, event.ydata):
            return
        self.box.on_press(event)

    # -------------------------------------------------------------- plot
    def show_sweep(self, session, fname, sweep, vmin_override=None, vmax_override=None):
        """Draw `fname` for `sweep`. Swaps only the mesh + colorbar (not the
        whole axes) so the drawn box/picks survive sweep/field changes.
        Returns the sweep-diagnostics string."""
        sweep = session.clamp_sweep(sweep)
        s_start, s_end = session.sweep_slice(sweep)
        data = session.field_slice(fname, s_start, s_end)
        gx, gy = session.gate_xy_km(s_start, s_end)
        diag = session.sweep_diagnostics(sweep, fname, s_start, s_end, gx, gy, data)
        nyq = session.get_nyquist(s_start)

        for attr in ("_mesh", "_colorbar"):
            obj = getattr(self, attr)
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

        cmap, norm = make_velocity_cmap_norm(nyq, vmin_override, vmax_override)
        mesh = self.ax.pcolormesh(gx, gy, data, cmap=cmap, norm=norm, shading="auto")
        self.ax.set_xlabel("Distance East of Radar (km)")
        self.ax.set_ylabel("Distance North of Radar (km)")
        self.ax.set_title(f"{fname} -- sweep {sweep}")
        self.ax.set_aspect("equal")
        self._colorbar = self.fig.colorbar(mesh, ax=self.ax, label=fname, fraction=0.046, pad=0.04)
        self._mesh = mesh
        self.canvas.draw_idle()
        return diag

    # ----------------------------------------------- arc-extrema markers
    def clear_arc_extrema_markers(self):
        for art in self._arc_extrema_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._arc_extrema_artists = []

    def draw_arc_extrema_markers(self, extrema_result):
        """Mark the two averaged outbound (+) / inbound (-) peak points that
        defined the refined vortex-center estimate. None just clears."""
        self.clear_arc_extrema_markers()
        if extrema_result is None:
            return
        for xy, color, lbl in ((extrema_result["pos_xy_km"], "red", "+"),
                               (extrema_result["neg_xy_km"], "blue", "\u2212")):
            (artist,) = self.ax.plot(xy[0], xy[1], marker="x", color=color,
                                     markersize=11, markeredgewidth=2.5, zorder=22)
            self._arc_extrema_artists.append(artist)
            self._arc_extrema_artists.append(self.ax.annotate(
                lbl, xy, textcoords="offset points", xytext=(6, 6),
                fontsize=10, fontweight="bold", color=color, zorder=22))
        self.canvas.draw_idle()
