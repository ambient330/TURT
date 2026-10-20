"""Range/azimuth grid helpers: aggregation, interpolation, beam geometry."""

import numpy as np

# Reference "coarse" native gate spacing (km) used as the pivot for
# auto-resolution: finer data is averaged over a few gates/rays, data at or
# coarser than this is displayed at native resolution.
_REF_NATIVE_SPACING_KM = 0.25  # typical legacy WSR-88D gate spacing
_MIN_AGGREGATION = 2
_MAX_AGGREGATION = 6

# 4/3-Earth-radius model (standard atmospheric refraction) for beam height.
_EFFECTIVE_EARTH_RADIUS_KM = (4.0 / 3.0) * 6371.0


def auto_aggregation_count(native_spacing_km):
    """How many native samples to average along an axis given its native
    spacing (km): finer than ~250 m -> 2..6 samples; otherwise 1 (native)."""
    native_spacing_km = (
        native_spacing_km if np.isfinite(native_spacing_km) and native_spacing_km > 0
        else _REF_NATIVE_SPACING_KM
    )
    if native_spacing_km < _REF_NATIVE_SPACING_KM:
        agg = int(round(np.clip(_REF_NATIVE_SPACING_KM / native_spacing_km,
                                 _MIN_AGGREGATION, _MAX_AGGREGATION)))
    else:
        agg = 1
    return max(1, agg)


def bin_edges_from_centers(centers):
    """len(centers)+1 bin edges implied by midpoints between consecutive
    centers, extrapolated at the two ends."""
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        d = 0.5
        return np.array([centers[0] - d, centers[0] + d])
    mid = (centers[:-1] + centers[1:]) / 2.0
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def beam_height_km(range_km, elevation_deg):
    """Height above the radar (km) for slant range `range_km` at antenna
    elevation `elevation_deg`, standard 4/3-Earth-radius equation."""
    range_km = np.asarray(range_km, dtype=float)
    elev_rad = np.deg2rad(elevation_deg)
    return (
        range_km * np.sin(elev_rad)
        + (range_km ** 2) * (np.cos(elev_rad) ** 2) / (2.0 * _EFFECTIVE_EARTH_RADIUS_KM)
    )


def bilinear_interp_full_sweep(data_field, az_data_deg, rng_data_km, r_km, beta_deg):
    """Bilinear interpolation of a masked (ray, gate) field to an arbitrary
    (range, azimuth) point. az_data_deg must be monotonically increasing and
    UNWRAPPED. Returns NaN outside the data extent or if any of the 4
    bracketing corners is masked/missing."""
    ngates = rng_data_km.shape[0]
    nrays = az_data_deg.shape[0]
    if ngates < 2 or nrays < 2:
        return np.nan

    dr = float(rng_data_km[1] - rng_data_km[0])
    if dr == 0:
        return np.nan
    gate_f = (r_km - rng_data_km[0]) / dr
    if gate_f < 0 or gate_f > ngates - 1:
        return np.nan
    i0 = int(np.floor(gate_f))
    i0 = min(max(i0, 0), ngates - 2)
    tg = gate_f - i0

    if beta_deg < az_data_deg[0] or beta_deg > az_data_deg[-1]:
        return np.nan
    k0 = int(np.searchsorted(az_data_deg, beta_deg, side="right") - 1)
    k0 = min(max(k0, 0), nrays - 2)
    daz = az_data_deg[k0 + 1] - az_data_deg[k0]
    tk = (beta_deg - az_data_deg[k0]) / daz if daz != 0 else 0.0

    corners = [data_field[k0, i0], data_field[k0, i0 + 1],
               data_field[k0 + 1, i0], data_field[k0 + 1, i0 + 1]]
    for c in corners:
        if np.ma.is_masked(c):
            return np.nan
        cf = float(np.ma.filled(c, np.nan))
        if not np.isfinite(cf):
            return np.nan

    c00, c10, c01, c11 = (float(c) for c in corners)
    top = c00 * (1 - tg) + c10 * tg
    bot = c01 * (1 - tg) + c11 * tg
    return top * (1 - tk) + bot * tk


def block_reduce_mean(arr2d, row_agg, col_agg, is_bool=False):
    """Average (or, if is_bool, OR-reduce) arr2d in row_agg x col_agg blocks,
    trimming leftovers. Returns (reduced, rows_used, cols_used)."""
    arr2d = np.ma.asarray(arr2d)
    nrows, ncols = arr2d.shape
    row_agg = max(1, min(row_agg, nrows))
    col_agg = max(1, min(col_agg, ncols))
    nrows_trim = (nrows // row_agg) * row_agg
    ncols_trim = (ncols // col_agg) * col_agg
    arr = arr2d[:nrows_trim, :ncols_trim]
    reshaped = arr.reshape(nrows_trim // row_agg, row_agg, ncols_trim // col_agg, col_agg)
    if is_bool:
        out = np.asarray(reshaped).any(axis=(1, 3))
    else:
        out = np.ma.mean(reshaped, axis=(1, 3))
    return out, nrows_trim, ncols_trim


def block_reduce_mean_1d(arr1d, agg, n_used):
    """Average arr1d (trimmed to n_used entries) in groups of `agg`."""
    arr1d = np.asarray(arr1d, dtype=float)[:n_used]
    agg = max(1, agg)
    return arr1d.reshape(n_used // agg, agg).mean(axis=1)


def circular_index_span(idx, n):
    """Shortest circular arc [start, end] of a circularly ordered index array
    (ray 0 neighbors ray n-1) containing all of `idx`. Returns
    (start, end, wrapped); wrapped means the arc crosses the n-1 -> 0 seam,
    so callers must concatenate [start..n-1] and [0..end]. A plain min/max
    would otherwise collapse a box drawn near due north into ~the whole sweep."""
    idx = np.unique(idx)
    if idx.size == n:
        return 0, n - 1, False
    if idx.size == 1:
        return int(idx[0]), int(idx[0]), False

    diffs = np.diff(idx)
    wrap_gap = (idx[0] + n) - idx[-1]
    all_gaps = np.concatenate([diffs, [wrap_gap]])
    max_gap_pos = int(np.argmax(all_gaps))

    if max_gap_pos == len(diffs):
        return int(idx[0]), int(idx[-1]), False

    start = int(idx[max_gap_pos + 1])
    end = int(idx[max_gap_pos])
    return start, end, True


def prepare_polar_grid(sub, az_sub, rng_sub, user_multiplier=1.0, force_native=False):
    """Prepare a box-selected sector for display in its native range/azimuth
    shape (rows = range gates, columns = azimuth samples).

    `sub` is the (ray, gate) window already cropped to the range/azimuth
    bounding window of the drag-box; that window IS the selection (cells
    are not re-tested against the Cartesian polygon, which would leave blank
    triangles in the corners of the fan).

    Resolution is chosen from the data's own native spacing (see
    auto_aggregation_count) and scaled by `user_multiplier` (>1 finer,
    <1 coarser). `force_native` forces 1x1 aggregation (no averaging).

    Returns a dict: disp_vals (n_range, n_az), rng_centers_km, az_centers_deg
    (unwrapped), rng_edges_km, az_edges_deg, range_agg, az_agg,
    native_range_km, native_az_km, ungridded.
    """
    rng_sub = np.asarray(rng_sub, dtype=float)
    az_sub_deg = np.asarray(az_sub, dtype=float)

    range_res_km = (
        float(np.median(np.abs(np.diff(np.sort(rng_sub))))) if rng_sub.size > 1 else 0.25
    )
    az_res_deg = (
        float(np.median(np.abs(np.diff(np.sort(az_sub_deg % 360.0))))) if az_sub_deg.size > 1 else 1.0
    )
    range_res_km = range_res_km if np.isfinite(range_res_km) and range_res_km > 0 else 0.25
    az_res_deg = az_res_deg if np.isfinite(az_res_deg) and az_res_deg > 0 else 1.0
    mid_range_km = float(np.median(rng_sub)) if rng_sub.size else 1.0
    native_az_km = mid_range_km * np.deg2rad(az_res_deg)
    native_az_km = native_az_km if np.isfinite(native_az_km) and native_az_km > 0 else range_res_km
    native_range_km = range_res_km

    if force_native:
        range_agg = 1
        az_agg = 1
    else:
        range_agg_auto = auto_aggregation_count(native_range_km)
        az_agg_auto = auto_aggregation_count(native_az_km)
        user_multiplier = user_multiplier if np.isfinite(user_multiplier) and user_multiplier > 0 else 1.0
        range_agg = max(1, int(round(range_agg_auto / user_multiplier)))
        az_agg = max(1, int(round(az_agg_auto / user_multiplier)))

    # Never average over more samples than the selection actually contains.
    range_agg = max(1, min(range_agg, rng_sub.size))
    az_agg = max(1, min(az_agg, az_sub_deg.size))

    # Unwrap azimuth so a box straddling 0/360 stays continuous.
    az_unwrapped_deg = np.rad2deg(np.unwrap(np.deg2rad(az_sub_deg)))

    # (ray, gate) -> (gate, ray) = (range, az)
    vals_t = np.ma.transpose(sub)
    disp_vals, n_range_used, n_az_used = block_reduce_mean(vals_t, range_agg, az_agg)

    rng_centers_km = block_reduce_mean_1d(rng_sub, range_agg, n_range_used)
    az_centers_deg = block_reduce_mean_1d(az_unwrapped_deg, az_agg, n_az_used)

    return {
        "disp_vals": disp_vals,
        "rng_centers_km": rng_centers_km,
        "az_centers_deg": az_centers_deg,
        "rng_edges_km": bin_edges_from_centers(rng_centers_km),
        "az_edges_deg": bin_edges_from_centers(az_centers_deg),
        "range_agg": range_agg,
        "az_agg": az_agg,
        "native_range_km": native_range_km,
        "native_az_km": native_az_km,
        "ungridded": bool(force_native),
    }
