"""RadarSession: owns the loaded radar volume and all sweep/gate lookups."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import pyart
except ImportError:
    pyart = None

from .synthetic import build_synthetic_tvs_radar


@dataclass
class LoadOutcome:
    """What happened during RadarSession.load(); the UI decides how to show it."""
    status: str
    loaded_path: Optional[str] = None
    error: Optional[str] = None                      # message for an error dialog
    warning: Optional[Tuple[str, str]] = None        # (title, text) for a warning dialog


class RadarSession:
    def __init__(self):
        self.radar = None
        self.path = None
        self.last_diag = "No data plotted yet."

    # ----------------------------------------------------------- loading
    def load(self, path) -> LoadOutcome:
        """Load `path` with Py-ART, or fall back to the synthetic demo volume."""
        self.path = None
        if path and pyart is not None:
            try:
                self.radar = pyart.io.read(path)
                self.path = path
                return LoadOutcome(f"Loaded {path}", loaded_path=path)
            except Exception as exc:
                self.radar = build_synthetic_tvs_radar()
                return LoadOutcome(
                    "Load failed -- showing synthetic demo TVS.",
                    error=f"Could not read this file with Py-ART:\n{exc}",
                )

        self.radar = build_synthetic_tvs_radar()
        warning = None
        if path and pyart is None:
            warning = (
                "Py-ART not found",
                "arm_pyart is not installed, so real radar files can't be "
                "read. Showing a synthetic demo TVS instead.\n\n"
                "Install with: pip install arm_pyart",
            )
        return LoadOutcome("Showing synthetic demo TVS volume.", warning=warning)

    # ---------------------------------------------------------- metadata
    @property
    def fields(self):
        return sorted(self.radar.fields.keys()) if self.radar is not None else []

    @property
    def nsweeps(self):
        return getattr(self.radar, "nsweeps", 1)

    def guess_reflectivity_field(self):
        return next(
            (f for f in self.fields if any(k in f.lower() for k in ("reflectivity", "dbz", "ref"))),
            None,
        )

    def clamp_sweep(self, sweep):
        return max(0, min(self.nsweeps - 1, int(sweep)))

    # ------------------------------------------------------ sweep geometry
    def sweep_slice(self, sweep):
        """(first_ray, last_ray) for a sweep. Prefers Py-ART's own get_slice()."""
        if hasattr(self.radar, "get_slice"):
            try:
                sl = self.radar.get_slice(sweep)
                return sl.start, sl.stop - 1
            except Exception:
                pass
        s_start = int(self.radar.sweep_start_ray_index["data"][sweep])
        s_end = int(self.radar.sweep_end_ray_index["data"][sweep])
        return s_start, s_end

    def gate_xy_km(self, s_start, s_end):
        """Cartesian (x, y) gate coordinates in km for rays [s_start, s_end].
        Uses Py-ART's georeferenced gate_x/gate_y when available, else a flat trig fallback."""
        try:
            gx = np.asarray(self.radar.gate_x[s_start:s_end + 1, :]) / 1000.0
            gy = np.asarray(self.radar.gate_y[s_start:s_end + 1, :]) / 1000.0
            if gx.size and np.isfinite(gx).any():
                return gx, gy
        except Exception:
            pass

        az = self.radar.azimuth["data"][s_start:s_end + 1]
        rng_km = self.radar.range["data"] / 1000.0
        elev = None
        try:
            elev = self.radar.elevation["data"][s_start:s_end + 1]
        except Exception:
            pass
        az_rad = np.deg2rad(az)
        if elev is not None:
            horiz = rng_km[None, :] * np.cos(np.deg2rad(elev))[:, None]
        else:
            horiz = rng_km[None, :]
        return horiz * np.sin(az_rad)[:, None], horiz * np.cos(az_rad)[:, None]

    def field_slice(self, fname, s_start, s_end):
        data = self.radar.fields[fname]["data"][s_start:s_end + 1, :]
        return np.ma.masked_invalid(data)

    def azimuth_slice(self, s_start, s_end):
        return self.radar.azimuth["data"][s_start:s_end + 1]

    def range_km(self):
        return self.radar.range["data"] / 1000.0

    def get_nyquist(self, s_start, default=33.0):
        """Best-effort Nyquist velocity lookup for the given ray index."""
        nyq = default
        ip = getattr(self.radar, "instrument_parameters", None) or {}
        nv = ip.get("nyquist_velocity")
        if nv is not None:
            try:
                nyq = float(np.ma.filled(nv["data"], np.nan)[s_start])
                if not np.isfinite(nyq):
                    nyq = default
            except Exception:
                nyq = default
        return nyq

    def get_elevation(self, s_start, default=0.5):
        """Best-effort antenna elevation (deg) for the given ray index."""
        elev = default
        try:
            ev = self.radar.elevation["data"]
            v = float(np.ma.filled(np.asarray(ev, dtype=float), np.nan)[s_start])
            if np.isfinite(v):
                elev = v
        except Exception:
            pass
        return elev

    # -------------------------------------------------------- diagnostics
    def sweep_diagnostics(self, sweep, fname, s_start, s_end, gx, gy, data):
        nrays = s_end - s_start + 1
        ngates = self.radar.range["data"].shape[0]
        finite = np.isfinite(gx) & np.isfinite(gy)
        if not np.ma.is_masked(data):
            valid = np.isfinite(np.asarray(data, dtype=float))
        else:
            valid = ~np.ma.getmaskarray(data)
        n_valid = int(np.count_nonzero(valid))
        rng_span = (
            f"{self.radar.range['data'].min()/1000.0:.1f}-"
            f"{self.radar.range['data'].max()/1000.0:.1f} km"
        )
        az = self.azimuth_slice(s_start, s_end)
        az_span = f"{az.min():.1f}-{az.max():.1f} deg" if nrays else "n/a"
        extent = (
            f"x:{np.nanmin(gx):.2f}..{np.nanmax(gx):.2f}  "
            f"y:{np.nanmin(gy):.2f}..{np.nanmax(gy):.2f} km"
            if finite.any() else "x/y: no finite gate coordinates!"
        )
        msg = (
            f"Sweep {sweep}: {nrays} rays x {ngates} gates | az {az_span} | "
            f"range {rng_span} | {fname} valid gates: {n_valid}/{data.size} | "
            f"extent {extent}"
        )
        print(msg)
        self.last_diag = msg
        return msg

    def degenerate_warning(self, sweep):
        """Heuristic: a suspiciously tiny plotted extent is almost always a
        split-cut / sweep-selection problem. Returns warning text or None."""
        try:
            s_start, s_end = self.sweep_slice(sweep)
            gx, gy = self.gate_xy_km(s_start, s_end)
            if not np.isfinite(gx).any():
                return None
            span_x = np.nanmax(gx) - np.nanmin(gx)
            span_y = np.nanmax(gy) - np.nanmin(gy)
            nrays = s_end - s_start + 1
            if max(span_x, span_y) < 1.0 or nrays <= 2:
                return (
                    "This sweep's plotted extent is only "
                    f"~{max(span_x, span_y):.3f} km across ({nrays} ray(s)).\n\n"
                    "That usually means one of:\n"
                    "  - This is a split-cut volume and sweep 0 is a short/"
                    "calibration or surveillance-only cut -- try the other "
                    "sweeps in the 'Sweep' box.\n"
                    "  - The selected field is empty/masked for this sweep -- "
                    "try a different Field.\n"
                    "  - Py-ART misread this file's range/azimuth geometry.\n\n"
                    "Click 'Sweep Info' after changing sweep/field to see the "
                    "ray/gate counts and value ranges."
                )
        except Exception:
            pass
        return None
