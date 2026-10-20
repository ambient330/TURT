"""
WDSS-II "Patch" single-Doppler velocity-couplet diagnostics (Stumpf et al.
1998 and descendants): from the box's inbound/outbound velocity extrema,
derive ground-relative peak velocities, Vdiff/Vasym/Vav, peak-to-peak
diameter, average radius, bulk vorticity/divergence, angular momentum,
circulation and centripetal acceleration.
"""

import numpy as np


def _law_of_cosines_distance_km(r1_km, theta1_deg, r2_km, theta2_deg, alpha_deg):
    """Distance (km) between two radar-coordinate points (r, theta) at a
    COMMON elevation alpha: sqrt(r1^2 + r2^2 - 2 r1 r2 (cos^2(a) cos(dtheta) + sin^2(a)))."""
    alpha_rad = np.deg2rad(alpha_deg)
    dtheta_rad = np.deg2rad(theta1_deg - theta2_deg)
    val = (
        r1_km ** 2 + r2_km ** 2
        - 2.0 * r1_km * r2_km * (
            (np.cos(alpha_rad) ** 2) * np.cos(dtheta_rad) + np.sin(alpha_rad) ** 2
        )
    )
    return float(np.sqrt(max(val, 0.0)))


def compute_patch_parameters(vel_sub, gx_sub, gy_sub, rng_sub_km, az_sub_deg,
                              alpha_deg, storm_speed_ms=0.0, storm_dir_deg=0.0,
                              center_xy_km=None):
    """Compute the Patch parameters from the box sector's inbound/outbound
    velocity extrema (vel_sub shape (n_azimuth, n_range)).

    Ground-relative velocity: Vgr = Vcorr + C*sin(beta - theta); leave
    storm_speed_ms=0 when the field is already ground-relative (normal for
    raw WSR-88D data).

    center_xy_km: optional vortex center (north-up km) for Average Radius /
    Bulk Angular Momentum / Centripetal Acceleration. Diam uses each peak's
    distance FROM THE RADAR; Rin/Rout are each peak's distance FROM THE
    VORTEX CENTER, so Rav is the Patch equivalent of Rmv. Without a center
    Rav falls back to Diam/2.

    Returns a dict, or None if there is no clear inbound/outbound pair.
    """
    vals = np.ma.filled(vel_sub.astype(float), np.nan)
    finite = np.isfinite(vals) & np.isfinite(gx_sub) & np.isfinite(gy_sub)
    if finite.sum() < 4:
        return None
    vals_f = np.where(finite, vals, np.nan)
    if not (np.isfinite(np.nanmax(vals_f)) and np.isfinite(np.nanmin(vals_f))):
        return None

    idx_out = np.unravel_index(int(np.nanargmax(vals_f)), vals_f.shape)
    idx_in = np.unravel_index(int(np.nanargmin(vals_f)), vals_f.shape)
    v_out_raw = float(vals_f[idx_out])
    v_in_raw = float(vals_f[idx_in])
    if v_out_raw <= 0.0 or v_in_raw >= 0.0:
        return None  # not a genuine inbound/outbound couplet

    x_out, y_out = float(gx_sub[idx_out]), float(gy_sub[idx_out])
    x_in, y_in = float(gx_sub[idx_in]), float(gy_sub[idx_in])
    theta_out = float(az_sub_deg[idx_out[0]]) % 360.0
    theta_in = float(az_sub_deg[idx_in[0]]) % 360.0
    r_out = float(rng_sub_km[idx_out[1]])
    r_in = float(rng_sub_km[idx_in[1]])

    beta_rad = np.deg2rad(storm_dir_deg)
    v_out_gr = v_out_raw + storm_speed_ms * np.sin(beta_rad - np.deg2rad(theta_out))
    v_in_gr = v_in_raw + storm_speed_ms * np.sin(beta_rad - np.deg2rad(theta_in))

    v_diff = v_out_gr - v_in_gr
    v_asym = abs(v_out_gr) - abs(v_in_gr)
    v_av = (abs(v_out_gr) + abs(v_in_gr)) / 2.0

    diam_km = _law_of_cosines_distance_km(r_in, theta_in, r_out, theta_out, alpha_deg)

    if center_xy_km is not None:
        xc, yc = center_xy_km
        r0 = float(np.hypot(xc, yc))
        beta0 = float(np.degrees(np.arctan2(xc, yc))) % 360.0
        r_in_center = _law_of_cosines_distance_km(r0, beta0, r_in, theta_in, alpha_deg)
        r_out_center = _law_of_cosines_distance_km(r0, beta0, r_out, theta_out, alpha_deg)
        r_av_km = (r_in_center + r_out_center) / 2.0
        rav_is_fallback = False
    else:
        r_in_center = r_out_center = None
        r_av_km = diam_km / 2.0
        rav_is_fallback = True

    # Bulk shear decomposition: alpha_beam = azimuth to the couplet midpoint;
    # theta_c = compass bearing of the inbound->outbound chord.
    x_mid, y_mid = (x_in + x_out) / 2.0, (y_in + y_out) / 2.0
    alpha_beam_deg = float(np.degrees(np.arctan2(x_mid, y_mid))) % 360.0
    theta_c_deg = float(np.degrees(np.arctan2(x_out - x_in, y_out - y_in))) % 360.0
    ang_diff_rad = np.deg2rad(alpha_beam_deg - theta_c_deg)

    diam_m = diam_km * 1000.0
    if diam_m > 0:
        zeta_bulk = 2.0 * (v_diff / diam_m) * np.sin(ang_diff_rad)
        delta_bulk = 2.0 * (v_diff / diam_m) * np.cos(ang_diff_rad)
        circ_patch = (np.pi / 2.0) * v_diff / diam_m
    else:
        zeta_bulk = delta_bulk = circ_patch = float("nan")

    r_av_m = r_av_km * 1000.0
    gamma_bulk = v_av * r_av_m
    f_accel = (v_av ** 2) / r_av_m if r_av_m > 0 else float("nan")

    return {
        "v_in_raw": v_in_raw, "v_out_raw": v_out_raw,
        "v_in_gr": v_in_gr, "v_out_gr": v_out_gr,
        "theta_in_deg": theta_in, "theta_out_deg": theta_out,
        "r_in_km": r_in, "r_out_km": r_out,
        "v_diff": v_diff, "v_asym": v_asym, "v_av": v_av,
        "diam_km": diam_km, "r_av_km": r_av_km, "rav_is_fallback": rav_is_fallback,
        "r_in_center_km": r_in_center, "r_out_center_km": r_out_center,
        "d_max_vort_km": diam_km,
        "alpha_beam_deg": alpha_beam_deg, "theta_c_deg": theta_c_deg,
        "zeta_bulk_s": zeta_bulk, "delta_bulk_s": delta_bulk,
        "gamma_bulk_m2s": gamma_bulk, "circulation_m2s": circ_patch,
        "centripetal_accel_ms2": f_accel,
        "storm_speed_ms": storm_speed_ms, "storm_dir_deg": storm_dir_deg,
    }
