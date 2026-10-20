"""AnalysisSettingsDialog: the numeric parameters that used to clutter the toolbar."""

import tkinter as tk
from tkinter import ttk


class AnalysisSettingsDialog:
    """A small non-modal window bound to the ControlPanel's Tk variables, so
    edits take effect immediately and persist when the window is closed."""

    def __init__(self, parent, panel):
        self.win = tk.Toplevel(parent)
        self.win.title("TURT -- Analysis Settings")
        self.win.resizable(False, False)
        body = ttk.Frame(self.win, padding=10)
        body.pack(fill=tk.BOTH, expand=True)

        def entry_row(frame, row, label, var, width=14, hint=""):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Entry(frame, textvariable=var, width=width).grid(row=row, column=1, sticky="w", pady=2)
            if hint:
                ttk.Label(frame, text=hint, foreground="#666").grid(row=row, column=2, sticky="w", padx=8)

        circ = ttk.LabelFrame(body, text="DJWR circulation circles", padding=8)
        circ.pack(fill=tk.X, pady=(0, 8))
        entry_row(circ, 0, "Circle radii (comma-sep):", panel.circ_radii_var, hint="km, or xRmax if checked")
        ttk.Checkbutton(circ, text="Read radii as \u00d7Rmax multipliers (auto-runs the Rankine fit)",
                        variable=panel.circ_radii_from_rankine_var
                        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=2)

        patch = ttk.LabelFrame(body, text="Patch (WDSS-II) storm motion", padding=8)
        patch.pack(fill=tk.X, pady=(0, 8))
        entry_row(patch, 0, "Storm speed C (m/s):", panel.storm_speed_var, hint="0 = field already ground-relative")
        entry_row(patch, 1, "Storm direction \u03b2 (deg):", panel.storm_dir_var)

        arc = ttk.LabelFrame(body, text="Azimuthal cross-section", padding=8)
        arc.pack(fill=tk.X, pady=(0, 8))
        entry_row(arc, 0, "Arc radii (km, comma-sep):", panel.arc_radii_var)
        entry_row(arc, 1, "Air density (kg/m\u00b3):", panel.arc_density_var)
        entry_row(arc, 2, "Radial \u0394 for centered diff (km):", panel.arc_delta_var, hint="blank = auto")

        ttk.Button(body, text="Close", command=self.win.destroy).pack(side=tk.RIGHT)

    def lift(self):
        self.win.deiconify()
        self.win.lift()

    def alive(self):
        try:
            return bool(self.win.winfo_exists())
        except tk.TclError:
            return False
