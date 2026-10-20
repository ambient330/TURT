"""
Azimuthal cross-section diagnostics: single-Doppler tangential-wind
inversion around concentric arcs centered on a vortex center, decomposed
into shear and curvature vorticity (centered difference across rho +/- delta
at each azimuthal sample), plus angular momentum, cumulative circulation and
cyclostrophic pressure deficit.

    zeta = zeta_curv + zeta_shear
    zeta_curv  = Vt / rho
    zeta_shear = [Vt(rho+delta) - Vt(rho-delta)] / (2 delta)
"""

import numpy as np

from .polar_grid import bilinear_interp_full_sweep
from .rankine import rankine_predicted_vr


def _tangential_wind_estimate(data_field, az_data_deg, rng_data_km,
                               xc_km, yc_km, x_km, y_km, proj_threshold=0.2):
    """Estimate tangential wind Vt at (x_km, y_km) around a vortex at
    (xc_km, yc_km) by inverting the radial-velocity measurement assuming
    locally axisymmetric, purely tangential flow: VD = Vt * proj.

    Returns (Vt, proj, vd_raw). Vt is NaN at/near the center, off the data,
    masked, or where |proj| < proj_threshold (beam nearly perpendicular to
    the flow; inversion unstable). vd_raw is the plain interpolated Doppler
    velocity, independent of the inversion.
    """
    dx, dy = x_km - xc_km, y_km - yc_km
    rho = float(np.hypot(dx, dy))
    if rho < 1e-6:
        return np.nan, np.nan, np.nan
    tx, ty = -dy / rho, dx / rho

    r_radar = float(np.hypot(x_km, y_km))
    if r_radar < 1e-6:
        return np.nan, np.nan, np.nan
    proj = tx * (x_km / r_radar) + ty * (y_km / r_radar)

    theta_radar = float(np.degrees(np.arctan2(x_km, y_km))) % 360.0
    vd = bilinear_interp_full_sweep(data_field, az_data_deg, rng_data_km, r_radar, theta_radar)
    if not np.isfinite(vd) or abs(proj) < proj_threshold:
        return np.nan, proj, vd
    return vd / proj, proj, vd


def detect_vortex_center_from_arc_extrema(data_field, az_data_deg, rng_data_km,
                                           xc_guess_km, yc_guess_km, radii_km,
                                           n_points=360):
    """Refine a vortex-center guess: at each radius, find the most positive
    (outbound) and most negative (inbound) raw Doppler bins on a FULL circle
    around the guess; the midpoint of the across-radii-averaged peak
    locations is the refined center (classic two-point TVS technique).

    Returns dict(xc_km, yc_km, pos_xy_km, neg_xy_km, diameter_km,
    radii_used_km, per_radius) or None if no valid peak pair was found.
    """
    radii_km = [float(r) for r in radii_km if r is not None and r > 0]
    if not radii_km:
        return None

    theta_rad = np.linspace(0.0, 2.0 * np.pi, int(n_points), endpoint=False)
    sin_t, cos_t = np.sin(theta_rad), np.cos(theta_rad)

    pos_points, neg_points, per_radius = [], [], []
    for rho in radii_km:
        xs = xc_guess_km + rho * sin_t
        ys = yc_guess_km + rho * cos_t
        vd = np.full(n_points, np.nan)
        for k in range(n_points):
            r_radar = float(np.hypot(xs[k], ys[k]))
            theta_radar = float(np.degrees(np.arctan2(xs[k], ys[k]))) % 360.0
            vd[k] = bilinear_interp_full_sweep(data_field, az_data_deg, rng_data_km, r_radar, theta_radar)

        if not np.isfinite(vd).any():
            continue
        i_pos = int(np.nanargmax(vd))
        i_neg = int(np.nanargmin(vd))
        if not (np.isfinite(vd[i_pos]) and np.isfinite(vd[i_neg])):
            continue

        pos_xy = (float(xs[i_pos]), float(ys[i_pos]))
        neg_xy = (float(xs[i_neg]), float(ys[i_neg]))
        pos_points.append(pos_xy)
        neg_points.append(neg_xy)
        per_radius.append({
            "rho_km": rho,
            "pos_xy_km": pos_xy, "pos_vd": float(vd[i_pos]),
            "neg_xy_km": neg_xy, "neg_vd": float(vd[i_neg]),
        })

    if not pos_points or not neg_points:
        return None

    pos_avg = (float(np.mean([p[0] for p in pos_points])), float(np.mean([p[1] for p in pos_points])))
    neg_avg = (float(np.mean([p[0] for p in neg_points])), float(np.mean([p[1] for p in neg_points])))

    return {
        "xc_km": 0.5 * (pos_avg[0] + neg_avg[0]),
        "yc_km": 0.5 * (pos_avg[1] + neg_avg[1]),
        "pos_xy_km": pos_avg,
        "neg_xy_km": neg_avg,
        "diameter_km": float(np.hypot(pos_avg[0] - neg_avg[0], pos_avg[1] - neg_avg[1])),
        "radii_used_km": radii_km,
        "per_radius": per_radius,
    }


def compute_arc_cross_sections(data_field, az_data_deg, rng_data_km,
                                xc_km, yc_km, radii_km, air_density=1.0,
                                n_points=72, delta_km=None, proj_threshold=0.2,
                                rankine_params=None):
    """Azimuthal cross-section diagnostics around concentric arcs of radius
    rho centered at (xc_km, yc_km), sampled over the full 360 deg at
    n_points azimuths.

    delta_km: half-spacing of the flanking mini-arcs for the shear-vorticity
    centered difference (default: a small fraction of the smallest radius).
    rankine_params: optional 6-tuple for rankine_predicted_vr; if given, the
    model's Doppler velocity is evaluated at the same arc points as the raw
    Doppler (an inversion-free goodness-of-fit comparison).

    Cyclostrophic pressure deficit: dp = 0.5 * air_density * Vt^2.

    Returns a list of per-radius dicts (empty if no radius had enough data).
    """
    radii_km = [float(r) for r in radii_km if r is not None and r > 0]
    if not radii_km:
        return []
    if delta_km is None or delta_km <= 0:
        delta_km = max(0.05, 0.1 * min(radii_km))

    theta_deg = np.linspace(0.0, 360.0, int(n_points), endpoint=False)
    theta_rad = np.deg2rad(theta_deg)
    n = theta_deg.size
    dtheta_rad = float(theta_rad[1] - theta_rad[0]) if n > 1 else 0.0

    arcs = []
    for rho in radii_km:
        if rho <= delta_km:
            continue

        vt = np.full(n, np.nan)
        vt_hi = np.full(n, np.nan)
        vt_lo = np.full(n, np.nan)
        vd_raw = np.full(n, np.nan)
        x_arc = np.full(n, np.nan)
        y_arc = np.full(n, np.nan)
        for k, th in enumerate(theta_rad):
            sin_th, cos_th = np.sin(th), np.cos(th)
            x0, y0 = xc_km + rho * sin_th, yc_km + rho * cos_th
            xh, yh = xc_km + (rho + delta_km) * sin_th, yc_km + (rho + delta_km) * cos_th
            xl, yl = xc_km + (rho - delta_km) * sin_th, yc_km + (rho - delta_km) * cos_th
            x_arc[k], y_arc[k] = x0, y0
            vt[k], _, vd_raw[k] = _tangential_wind_estimate(
                data_field, az_data_deg, rng_data_km, xc_km, yc_km, x0, y0, proj_threshold)
            vt_hi[k], _, _ = _tangential_wind_estimate(
                data_field, az_data_deg, rng_data_km, xc_km, yc_km, xh, yh, proj_threshold)
            vt_lo[k], _, _ = _tangential_wind_estimate(
                data_field, az_data_deg, rng_data_km, xc_km, yc_km, xl, yl, proj_threshold)

        n_valid = int(np.isfinite(vt).sum())
        if n_valid < max(6, n // 6):
            continue  # too little of the ring is observable

        # Circular linear-interpolation gap fill (near the unstable "N"/"F"
        # tangent points) so the integrals aren't biased low by zeros.
        idx = np.arange(n)
        for arr in (vt, vt_hi, vt_lo, vd_raw):
            good = np.isfinite(arr)
            if 0 < good.sum() < n:
                arr[:] = np.interp(idx, idx[good], arr[good], period=n)

        rho_m = rho * 1000.0
        delta_m = delta_km * 1000.0

        shear_vorticity = (vt_hi - vt_lo) / (2.0 * delta_m)
        curvature_vorticity = vt / rho_m

        vd_rankine = None
        if rankine_params is not None:
            try:
                vd_rankine = rankine_predicted_vr(rankine_params, x_arc, y_arc)
            except Exception:
                vd_rankine = None

        if n > 1:
            avg_contrib = 0.5 * (vt[:-1] + vt[1:]) * rho_m * dtheta_rad
            circulation_cumulative = np.concatenate([[0.0], np.cumsum(avg_contrib)])
        else:
            circulation_cumulative = np.zeros(n)

        arcs.append({
            "rho_km": rho,
            "theta_deg": theta_deg,
            "vt": vt,
            "vd_raw": vd_raw,
            "vd_rankine": vd_rankine,
            "shear_vorticity": shear_vorticity,
            "curvature_vorticity": curvature_vorticity,
            "total_vorticity": shear_vorticity + curvature_vorticity,
            "angular_momentum": vt * rho_m,
            "circulation_cumulative": circulation_cumulative,
            "pressure_deficit": 0.5 * air_density * vt ** 2,
            "delta_km": delta_km,
            "n_valid": n_valid,
            "n_points": n,
        })

    return arcs
