"""ControlPanel: a compact, dropdown-driven toolbar (no analysis logic).

Row 1  file / field / sweep / color range + Clear Box, Zoom, Sweep Info
Row 2  View, Grid and contour-overlay dropdowns
Row 3  Analyses / Vortex Center / Export menus + Settings, and manual-pick status
"""

import tkinter as tk
from tkinter import ttk

import numpy as np

from .analysis import AnalysisOptions
from .manual_picks import ManualPicker
from .settings_dialog import AnalysisSettingsDialog


def _parse_float_list(text, default):
    try:
        vals = sorted(float(s) for s in text.split(",") if s.strip())
        return vals or list(default)
    except Exception:
        return list(default)


def _parse_float(text, default):
    try:
        return float(text)
    except Exception:
        return default


class ControlPanel:
    """callbacks: import_file, refresh, clear_box, zoom, diagnostics,
    pick_center, clear_center, pick_pos_peak, pick_neg_peak, export_fan."""

    # label -> (use polar fan, true north-up)
    VIEW_CHOICES = {
        "Polar fan (radar-centered)": (True, False),
        "Polar fan (true north-up)": (True, True),
        "Flat table": (False, False),
    }
    # label -> (resolution multiplier, ungridded)
    GRID_CHOICES = {
        "Auto (data-adaptive)": (1.0, False),
        "Coarser (0.5x)": (0.5, False),
        "Finer (2x)": (2.0, False),
        "Finest (4x)": (4.0, False),
        "Ungridded (native, no averaging)": (1.0, True),
    }
    CONTOUR_CMAPS = ["Carbone42", "pyart_balance", "Red-Green", "Black & White"]
    NO_MANUAL_TEXT = "Vortex center: automatic"

    def __init__(self, parent, callbacks):
        self.parent = parent
        self.cb = callbacks
        self._settings = None
        self._make_variables()
        self._build_main_row()
        self._build_display_row()
        self._build_menu_row()
        self.status_var = tk.StringVar(
            value="Click and drag on the PPI to draw a box around the TVS, then release to finalize.")
        ttk.Label(parent, textvariable=self.status_var, foreground="#333").pack(
            side=tk.TOP, fill=tk.X, padx=8)

    # --------------------------------------------------------- variables
    def _make_variables(self):
        self.field_var = tk.StringVar(value="velocity")
        self.color_range_var = tk.StringVar(value="")   # blank = automatic
        self.sweep_var = tk.IntVar(value=0)

        self.view_var = tk.StringVar(value="Polar fan (radar-centered)")
        self.grid_var = tk.StringVar(value="Auto (data-adaptive)")
        self.refl_field_var = tk.StringVar(value="None")
        self.refl_levels_var = tk.StringVar(value="20,30,40,50")
        self.contour_cmap_var = tk.StringVar(value="Carbone42")

        # Analyses menu
        self.fit_rankine_var = tk.BooleanVar(value=False)
        self.circ_center_var = tk.BooleanVar(value=False)
        self.patch_var = tk.BooleanVar(value=False)
        self.arc_xsec_var = tk.BooleanVar(value=False)

        # Settings dialog
        self.circ_radii_var = tk.StringVar(value="1,2,3")
        self.circ_radii_from_rankine_var = tk.BooleanVar(value=False)
        self.storm_speed_var = tk.StringVar(value="0")
        self.storm_dir_var = tk.StringVar(value="0")
        self.arc_radii_var = tk.StringVar(value="1,2,3")
        self.arc_density_var = tk.StringVar(value="1.0")
        self.arc_delta_var = tk.StringVar(value="")

        # Export menu
        self.export_all_fields_var = tk.BooleanVar(value=True)
        self.export_dummy_var = tk.BooleanVar(value=True)
        self.dummy_fill_displayed_var = tk.BooleanVar(value=False)
        self.export_on_zoom_var = tk.BooleanVar(value=False)

        self.manual_label_var = tk.StringVar(value=self.NO_MANUAL_TEXT)

    # --------------------------------------------------------------- rows
    def _row(self, pady=(0, 4)):
        bar = ttk.Frame(self.parent)
        bar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=pady)
        return bar

    def _build_main_row(self):
        bar = self._row(pady=4)
        ttk.Button(bar, text="Import Radar File...", command=self.cb["import_file"]).pack(side=tk.LEFT, padx=3)

        ttk.Label(bar, text="Field:").pack(side=tk.LEFT, padx=(12, 2))
        self.field_combo = ttk.Combobox(bar, textvariable=self.field_var, width=16, state="readonly")
        self.field_combo.pack(side=tk.LEFT)
        self.field_combo.bind("<<ComboboxSelected>>", lambda e: self.cb["refresh"]())

        ttk.Label(bar, text="Sweep:").pack(side=tk.LEFT, padx=(12, 2))
        self.sweep_spin = ttk.Spinbox(bar, from_=0, to=0, width=4, textvariable=self.sweep_var,
                                      command=self.cb["refresh"])
        self.sweep_spin.pack(side=tk.LEFT)

        # Blank = automatic (+/-Nyquist floored at 60 m/s); e.g. "0,70" for reflectivity.
        ttk.Label(bar, text="Color range (min,max):").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(bar, textvariable=self.color_range_var, width=10).pack(side=tk.LEFT)
        self.color_range_var.trace_add("write", lambda *a: self.cb["refresh"]())

        ttk.Button(bar, text="Clear Box", command=self.cb["clear_box"]).pack(side=tk.LEFT, padx=(16, 3))
        ttk.Button(bar, text="Zoom to TVS Grid", command=self.cb["zoom"]).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Sweep Info", command=self.cb["diagnostics"]).pack(side=tk.LEFT, padx=3)

    def _build_display_row(self):
        bar = self._row()
        ttk.Label(bar, text="View:").pack(side=tk.LEFT, padx=(3, 2))
        ttk.Combobox(bar, textvariable=self.view_var, width=26, state="readonly",
                     values=list(self.VIEW_CHOICES)).pack(side=tk.LEFT)

        ttk.Label(bar, text="Grid:").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Combobox(bar, textvariable=self.grid_var, width=30, state="readonly",
                     values=list(self.GRID_CHOICES)).pack(side=tk.LEFT)

        ttk.Label(bar, text="Contours:").pack(side=tk.LEFT, padx=(12, 2))
        self.refl_field_combo = ttk.Combobox(bar, textvariable=self.refl_field_var, width=14,
                                             state="readonly", values=["None"])
        self.refl_field_combo.pack(side=tk.LEFT)
        ttk.Label(bar, text="at").pack(side=tk.LEFT, padx=3)
        ttk.Entry(bar, textvariable=self.refl_levels_var, width=12).pack(side=tk.LEFT)
        ttk.Combobox(bar, textvariable=self.contour_cmap_var, width=13, state="readonly",
                     values=self.CONTOUR_CMAPS).pack(side=tk.LEFT, padx=(4, 0))

    def _build_menu_row(self):
        bar = self._row()

        def menubutton(text):
            mb = ttk.Menubutton(bar, text=text)
            menu = tk.Menu(mb, tearoff=False)
            mb["menu"] = menu
            mb.pack(side=tk.LEFT, padx=3)
            return menu

        m = menubutton("Analyses \u25be")
        m.add_checkbutton(label="Rankine vortex fit  (box tightly around the couplet)",
                          variable=self.fit_rankine_var)
        m.add_checkbutton(label="DJWR circulation center + circles", variable=self.circ_center_var)
        m.add_checkbutton(label="Patch (WDSS-II) velocity-couplet parameters", variable=self.patch_var)
        m.add_checkbutton(label="Azimuthal cross-section (vorticity, ang. momentum, circulation, \u0394p)",
                          variable=self.arc_xsec_var)
        m.add_separator()
        m.add_command(label="Settings...", command=self.open_settings)

        m = menubutton("Vortex Center \u25be")
        m.add_command(label="Pick vortex center on PPI", command=self.cb["pick_center"])
        m.add_command(label="Pick +peak (outbound) on PPI", command=self.cb["pick_pos_peak"])
        m.add_command(label="Pick \u2212peak (inbound) on PPI", command=self.cb["pick_neg_peak"])
        m.add_separator()
        m.add_command(label="Clear manual center / peaks (back to automatic)", command=self.cb["clear_center"])

        m = menubutton("Export \u25be")
        m.add_command(label="Export fan as radar file...", command=self.cb["export_fan"])
        m.add_separator()
        m.add_checkbutton(label="All fields  (off = displayed field only)", variable=self.export_all_fields_var)
        m.add_checkbutton(label="Also write full-sweep dummy radar (*_fullsweep.nc)", variable=self.export_dummy_var)
        m.add_checkbutton(label="Dummy radar: fill with fan-averaged values  (off = native gates)",
                          variable=self.dummy_fill_displayed_var)
        m.add_separator()
        m.add_checkbutton(label="Export automatically on every zoom", variable=self.export_on_zoom_var)

        ttk.Button(bar, text="Settings...", command=self.open_settings).pack(side=tk.LEFT, padx=(8, 3))
        ttk.Label(bar, textvariable=self.manual_label_var, foreground="#333").pack(side=tk.LEFT, padx=(12, 0))

    # ---------------------------------------------------------- behavior
    def open_settings(self):
        if self._settings is not None and self._settings.alive():
            self._settings.lift()
        else:
            self._settings = AnalysisSettingsDialog(self.parent, self)

    # ---------------------------------------------------- state accessors
    @property
    def use_polar_fan(self):
        return self.VIEW_CHOICES.get(self.view_var.get(), (True, False))[0]

    @property
    def true_oriented(self):
        return self.VIEW_CHOICES.get(self.view_var.get(), (True, False))[1]

    @property
    def ungridded(self):
        return self.GRID_CHOICES.get(self.grid_var.get(), (1.0, False))[1]

    def supersample_factor(self):
        return self.GRID_CHOICES.get(self.grid_var.get(), (1.0, False))[0]

    def set_fields(self, fields, default_field, refl_guess):
        self.field_combo["values"] = fields
        if fields:
            self.field_var.set(default_field)
        self.refl_field_combo["values"] = ["None"] + list(fields)
        self.refl_field_var.set(refl_guess if refl_guess else "None")

    def set_nsweeps(self, nsweeps):
        self.sweep_spin.configure(to=max(0, nsweeps - 1))
        self.sweep_var.set(0)

    def set_manual_labels(self, center_text, peaks_text):
        """Collapse the picker's two texts into one status label."""
        parts = [t for t in (center_text, peaks_text)
                 if t not in (ManualPicker.NO_CENTER_TEXT, ManualPicker.NO_PEAKS_TEXT)]
        self.manual_label_var.set("   ".join(parts) if parts else self.NO_MANUAL_TEXT)

    def parse_color_range_override(self):
        """(vmin, vmax) from the "Color range" field, or (None, None) if blank/invalid."""
        txt = self.color_range_var.get().strip()
        if not txt:
            return None, None
        try:
            parts = [p.strip() for p in txt.split(",")]
            if len(parts) != 2:
                return None, None
            vmin, vmax = float(parts[0]), float(parts[1])
            if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
                return None, None
            return vmin, vmax
        except Exception:
            return None, None

    def parse_refl_levels(self):
        return _parse_float_list(self.refl_levels_var.get(), [20.0, 30.0, 40.0, 50.0])

    def analysis_options(self):
        delta_txt = self.arc_delta_var.get().strip()
        return AnalysisOptions(
            fit_rankine=self.fit_rankine_var.get(),
            circ_center=self.circ_center_var.get(),
            circ_radii=_parse_float_list(self.circ_radii_var.get(), [1.0, 2.0, 3.0]),
            circ_radii_from_rankine=self.circ_radii_from_rankine_var.get(),
            patch=self.patch_var.get(),
            storm_speed_ms=_parse_float(self.storm_speed_var.get(), 0.0),
            storm_dir_deg=_parse_float(self.storm_dir_var.get(), 0.0),
            arc_xsec=self.arc_xsec_var.get(),
            arc_radii=_parse_float_list(self.arc_radii_var.get(), [1.0, 2.0, 3.0]),
            air_density=_parse_float(self.arc_density_var.get(), 1.0),
            arc_delta_km=_parse_float(delta_txt, None) if delta_txt else None,
        )
