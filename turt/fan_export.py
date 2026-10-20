"""
FanRadarExporter: write the box-selected fan sector out as a real radar
file (CfRadial NetCDF via Py-ART), so it can be re-opened in Py-ART, GURT,
the other tools in this toolkit, etc.

The exported radar is a single-sweep PPI whose rays are the fan's azimuth
columns and whose gates are the fan's range rows -- i.e. exactly the grid
shown in the fan plot. If the fan was averaged (range_agg / az_agg > 1),
every exported field is block-averaged the same way, so the file matches
what was plotted; with "Ungridded" (or native-resolution data) the export
is a plain native crop.

A second mode, build_dummy_radar(), instead keeps the ORIGINAL sweep geometry
(every ray/azimuth and every range gate of the source sweep, e.g. all 360
azimuths) and fills only the fan's cells with data; everything outside the
fan is masked (missing). Handy for tools that expect a complete sweep.

Not carried over: per-ray platform/georeference arrays (rotation, tilt,
heading, ...), calibration tables and anything that doesn't map cleanly
onto a cropped/averaged sweep. Radar location, sweep metadata, timing,
Nyquist velocity and other per-ray instrument parameters are preserved.
"""

import copy
import datetime

import numpy as np

from .polar_grid import block_reduce_mean, block_reduce_mean_1d

try:
    import pyart
except ImportError:
    pyart = None


def _with_data(meta_dict, data):
    """Copy a Py-ART metadata dict and swap in new data."""
    out = {k: copy.deepcopy(v) for k, v in meta_dict.items() if k != "data"}
    out["data"] = data
    return out


class FanRadarExporter:
    def __init__(self, radar):
        self.radar = radar

    @staticmethod
    def is_available():
        return pyart is not None

    @staticmethod
    def default_filename(selection, grid):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"turt_fan_sweep{selection.sweep}_{stamp}.nc"

    # ------------------------------------------------------------------
    def build_radar(self, selection, grid, field_names=None):
        """Build a new single-sweep pyart.core.Radar from the fan sector.

        field_names: iterable of field names to include (default: all).
        """
        if pyart is None:
            raise RuntimeError("Py-ART is required to export radar data (pip install arm_pyart).")

        radar = self.radar
        range_agg, az_agg = grid["range_agg"], grid["az_agg"]
        n_range, n_az = grid["disp_vals"].shape
        n_range_used, n_az_used = n_range * range_agg, n_az * az_agg

        ray_abs = selection.absolute_ray_indices[:n_az_used]
        g0 = selection.gate_lo
        g1 = g0 + n_range_used

        # ---- fields: crop to the fan window, then block-average like the fan
        names = list(field_names) if field_names is not None else list(radar.fields.keys())
        fields = {}
        for name in names:
            if name not in radar.fields:
                continue
            raw = np.ma.masked_invalid(radar.fields[name]["data"][ray_abs][:, g0:g1])
            reduced, _, _ = block_reduce_mean(np.ma.transpose(raw), range_agg, az_agg)
            arr = np.ma.transpose(reduced)  # (n_az, n_range)
            meta = {k: copy.deepcopy(v) for k, v in radar.fields[name].items() if k != "data"}
            meta.setdefault("_FillValue", -9999.0)
            fields[name] = dict(meta, data=np.ma.masked_invalid(arr.astype("float32")))

        # ---- coordinates (reuse the fan grid's own centers)
        range_m = np.asarray(grid["rng_centers_km"], dtype="float32") * 1000.0
        azimuth = (np.asarray(grid["az_centers_deg"], dtype="float32")) % 360.0

        def ray_reduce(arr):
            arr = np.asarray(np.ma.filled(arr, np.nan), dtype=float)[ray_abs]
            return block_reduce_mean_1d(arr, az_agg, n_az_used)

        elevation = ray_reduce(radar.elevation["data"]).astype("float32")
        times = ray_reduce(radar.time["data"])

        range_meta = _with_data(radar.range, range_m)
        range_meta["meters_to_center_of_first_gate"] = float(range_m[0])
        if range_m.size > 1:
            range_meta["meters_between_gates"] = float(np.median(np.diff(range_m)))

        # ---- per-ray instrument parameters are cropped/averaged; others copied
        inst = None
        ip = getattr(radar, "instrument_parameters", None)
        if ip:
            inst = {}
            for key, val in ip.items():
                try:
                    d = np.ma.asarray(val["data"])
                    if d.ndim == 1 and d.shape[0] == radar.nrays:
                        inst[key] = _with_data(val, ray_reduce(d).astype(d.dtype
                                                                       if d.dtype.kind == "f" else "float32"))
                    else:
                        inst[key] = copy.deepcopy(val)
                except Exception:
                    continue

        sweep = selection.sweep
        n = n_az
        metadata = dict(radar.metadata) if getattr(radar, "metadata", None) else {}
        stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
        metadata["history"] = (
            str(metadata.get("history", "")).rstrip()
            + f"\n{stamp}: fan sector export (source sweep {sweep}, "
              f"{n_range} gates x {n_az} rays, range_agg={range_agg}, az_agg={az_agg})"
        ).strip()

        sweep_mode = _with_data(radar.sweep_mode, np.asarray(radar.sweep_mode["data"])[sweep:sweep + 1])
        fixed_angle = _with_data(radar.fixed_angle, np.asarray(radar.fixed_angle["data"])[sweep:sweep + 1])

        return pyart.core.Radar(
            time=_with_data(radar.time, times),
            _range=range_meta,
            fields=fields,
            metadata=metadata,
            scan_type=radar.scan_type,
            latitude=copy.deepcopy(radar.latitude),
            longitude=copy.deepcopy(radar.longitude),
            altitude=copy.deepcopy(radar.altitude),
            sweep_number=_with_data(radar.sweep_number, np.array([0], dtype="int32")),
            sweep_mode=sweep_mode,
            fixed_angle=fixed_angle,
            sweep_start_ray_index=_with_data(radar.sweep_start_ray_index, np.array([0], dtype="int32")),
            sweep_end_ray_index=_with_data(radar.sweep_end_ray_index, np.array([n - 1], dtype="int32")),
            azimuth=_with_data(radar.azimuth, azimuth),
            elevation=_with_data(radar.elevation, elevation),
            instrument_parameters=inst,
        )

    def export(self, selection, grid, path, field_names=None, file_format="NETCDF4"):
        """Build the fan radar and write it as CfRadial. Returns the radar object."""
        out = self.build_radar(selection, grid, field_names)
        pyart.io.write_cfradial(path, out, format=file_format)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def dummy_path(path):
        """'foo.nc' -> 'foo_fullsweep.nc'"""
        import os
        root, ext = os.path.splitext(path)
        return f"{root}_fullsweep{ext or '.nc'}"

    def build_dummy_radar(self, selection, grid, field_names=None, fill="native"):
        """Full-sweep "dummy" radar: all rays and gates of the source sweep,
        with only the fan's cells holding data (rest masked).

        fill="native"    -> the fan window's original native gate values.
        fill="displayed" -> the fan plot's (possibly averaged) values, each
                            block value repeated over the native cells it
                            covers (leftover trimmed rows/cols stay masked).
        """
        if pyart is None:
            raise RuntimeError("Py-ART is required to export radar data (pip install arm_pyart).")
        if fill not in ("native", "displayed"):
            raise ValueError("fill must be 'native' or 'displayed'")

        radar = self.radar
        sweep = selection.sweep
        out = radar.extract_sweeps([sweep])       # full geometry/metadata, single sweep
        n_rays, n_gates = out.nrays, out.ngates

        rays = np.asarray(selection.ray_idx_order)            # sweep-relative ray indices
        g0, g1 = selection.gate_lo, selection.gate_hi + 1
        range_agg, az_agg = grid["range_agg"], grid["az_agg"]
        n_range, n_az = grid["disp_vals"].shape
        n_range_used, n_az_used = n_range * range_agg, n_az * az_agg

        names = list(field_names) if field_names is not None else list(radar.fields.keys())
        for name in list(out.fields.keys()):
            if name not in names:
                del out.fields[name]

        for name in names:
            if name not in radar.fields:
                continue
            full = np.ma.masked_all((n_rays, n_gates), dtype="float32")
            src = np.ma.masked_invalid(
                radar.fields[name]["data"][selection.s_start:selection.s_end + 1])
            if fill == "native":
                rr = rays
                cc = np.arange(g0, g1)
                full[np.ix_(rr, cc)] = src[rr][:, g0:g1]
            else:
                raw = src[rays[:n_az_used]][:, g0:g0 + n_range_used]
                red, _, _ = block_reduce_mean(np.ma.transpose(raw), range_agg, az_agg)
                blocks = np.ma.transpose(red)                       # (n_az, n_range)
                expanded = np.ma.repeat(np.ma.repeat(blocks, az_agg, axis=0), range_agg, axis=1)
                rr = rays[:n_az_used]
                cc = np.arange(g0, g0 + n_range_used)
                full[np.ix_(rr, cc)] = expanded
            meta = out.fields.get(name, {})
            meta = {k: v for k, v in meta.items() if k != "data"} or {
                k: copy.deepcopy(v) for k, v in radar.fields[name].items() if k != "data"}
            meta.setdefault("_FillValue", -9999.0)
            out.fields[name] = dict(meta, data=full)

        stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
        out.metadata["history"] = (
            str(out.metadata.get("history", "")).rstrip()
            + f"\n{stamp}: full-sweep dummy radar (source sweep {sweep}); only the fan window "
              f"({len(rays)} rays x {g1 - g0} gates, fill={fill}) holds data"
        ).strip()
        return out

    def export_dummy(self, selection, grid, path, field_names=None, fill="native",
                     file_format="NETCDF4"):
        out = self.build_dummy_radar(selection, grid, field_names, fill)
        pyart.io.write_cfradial(path, out, format=file_format)
        return out
