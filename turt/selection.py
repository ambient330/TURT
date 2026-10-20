"""FanSelection: turns a drag-box (Cartesian km) into a cropped ray/gate window."""

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath

from .polar_grid import circular_index_span


@dataclass
class FanSelection:
    """The sector of one sweep/field bounded by the user's drag-box.

    ray_idx_order are ray indices WITHIN the sweep, in azimuthal order
    (stitched across the 0/360 seam if the box wrapped); add s_start for
    absolute volume ray indices. The gate window is [gate_lo, gate_hi].
    """
    sweep: int
    s_start: int
    s_end: int
    field_name: str
    ray_idx_order: np.ndarray
    ray_lo: int
    ray_wrapped: bool
    gate_lo: int
    gate_hi: int
    # cropped (n_ray_sel, n_gate_sel) arrays
    sub: np.ma.MaskedArray
    gx_sub: np.ndarray
    gy_sub: np.ndarray
    az_sub: np.ndarray            # degrees
    rng_sub: np.ndarray           # km
    # full-sweep context (circles/arcs can extend beyond the box)
    data: np.ma.MaskedArray
    az_data: np.ndarray
    rng_data: np.ndarray
    nyquist: float
    elev_deg: float

    @property
    def absolute_ray_indices(self):
        return self.s_start + self.ray_idx_order

    def crop(self, full_sweep_field):
        """Crop another full-sweep (ray, gate) array to this selection's window."""
        return full_sweep_field[self.ray_idx_order][:, self.gate_lo:self.gate_hi + 1]


def build_fan_selection(session, sweep, field_name, box_pts):
    """Returns (FanSelection, None) on success or (None, error_message)."""
    radar = session.radar
    if radar is None or field_name not in radar.fields:
        return None, "Field not available."
    if len(box_pts) < 3:
        return None, "Drag out a box on the PPI first."

    sweep = session.clamp_sweep(sweep)
    s_start, s_end = session.sweep_slice(sweep)
    az_data = session.azimuth_slice(s_start, s_end)
    rng_data = session.range_km()
    gx, gy = session.gate_xy_km(s_start, s_end)

    poly_path = MplPath(list(box_pts) + [box_pts[0]])
    pts_flat = np.column_stack([gx.ravel(), gy.ravel()])
    inside = poly_path.contains_points(pts_flat).reshape(gx.shape)
    hit = np.argwhere(inside)
    if hit.size == 0:
        return None, "No gates inside box."

    nrays_total = gx.shape[0]
    ray_lo, ray_hi, ray_wrapped = circular_index_span(hit[:, 0], nrays_total)
    gate_lo, gate_hi = int(hit[:, 1].min()), int(hit[:, 1].max())

    if ray_wrapped:
        # Box straddles the 0/360 seam: stitch tail + head of the ray axis.
        ray_idx_order = np.concatenate([np.arange(ray_lo, nrays_total), np.arange(0, ray_hi + 1)])
    else:
        ray_idx_order = np.arange(ray_lo, ray_hi + 1)

    data = session.field_slice(field_name, s_start, s_end)
    ref_ray = s_start if ray_wrapped else s_start + ray_lo

    sel = FanSelection(
        sweep=sweep, s_start=s_start, s_end=s_end, field_name=field_name,
        ray_idx_order=ray_idx_order, ray_lo=ray_lo, ray_wrapped=ray_wrapped,
        gate_lo=gate_lo, gate_hi=gate_hi,
        sub=data[ray_idx_order][:, gate_lo:gate_hi + 1],
        gx_sub=gx[ray_idx_order][:, gate_lo:gate_hi + 1],
        gy_sub=gy[ray_idx_order][:, gate_lo:gate_hi + 1],
        az_sub=np.asarray(az_data)[ray_idx_order],
        rng_sub=rng_data[gate_lo:gate_hi + 1],
        data=data, az_data=np.asarray(az_data), rng_data=rng_data,
        nyquist=session.get_nyquist(ref_ray),
        elev_deg=session.get_elevation(ref_ray),
    )
    return sel, None
