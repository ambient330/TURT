"""
DJWR Doppler-circulation center finding and circle-based Doppler
circulation / areal-contraction-rate calculations.

  DJWR  = Davies-Jones, Wood, and Rasmussen 2020, J. Atmos. Oceanic Technol.,
          37, 1117-1133.
  The center-finding appendix and the circle-based kinematics follow Wood,
  Davies-Jones, Snyder, and Hopper (2025), which builds on DJWR.
"""

import numpy as np

from .polar_grid import bilinear_interp_full_sweep

_DJWR_RING_HALF_POINTS = 60  # -> 2*60 = 120 points per circle


def _shoelace_L(f_vals, g_vals):
    """Cyclic line-integral sum L(f,g) = sum_m [f_m g_(m+1) - f_(m+1) g_m]
    (DJWR Eq. 13). Circulation = 0.5*L(VD, r)."""
    f = np.asarray(f_vals, dtype=float)
    g = np.asarray(g_vals, dtype=float)
    return float(np.nansum(f * np.roll(g, -1) - np.roll(f, -1) * g))


def _cell_doppler_circulation_grid(vel_sub, rng_sub_km, range_spacing_km):
    """Vectorized Doppler *cell* circulation, DJWR Eq. 15.
    vel_sub: masked (n_az, n_range). Returns (dgamma (n_az-1, n_range-1),
    cell-center ranges km), or (None, None) if too small."""
    V = np.ma.filled(vel_sub.astype(float), np.nan)
    if V.shape[0] < 2 or V.shape[1] < 2:
        return None, None
    diff_ray = V[1:, :] - V[:-1, :]
    dgamma = 0.5 * range_spacing_km * (diff_ray[:, :-1] + diff_ray[:, 1:])
    rng_sub_km = np.asarray(rng_sub_km, dtype=float)
    return dgamma, 0.5 * (rng_sub_km[:-1] + rng_sub_km[1:])


def _locate_circulation_center(dgamma, rng_centers_km, az_centers_deg, range_spacing_km):
    """Wood et al. (2025) appendix A: find the cell with the largest |circulation| and
    refine within it via a cross-term-free 2nd-order Taylor fit through the
    4 axis-aligned neighbors. Returns dict(r0_km, beta0_deg, cell_value,
    refined) or None."""
    if dgamma is None or not np.isfinite(dgamma).any():
        return None

    abs_vals = np.where(np.isfinite(dgamma), np.abs(dgamma), -np.inf)
    k0, i0 = np.unravel_index(np.argmax(abs_vals), abs_vals.shape)
    nrows, ncols = dgamma.shape

    def unrefined():
        return {
            "r0_km": float(rng_centers_km[i0]),
            "beta0_deg": float(az_centers_deg[k0]),
            "cell_value": float(dgamma[k0, i0]),
            "refined": False,
        }

    if i0 <= 0 or i0 >= ncols - 1 or k0 <= 0 or k0 >= nrows - 1:
        return unrefined()

    gamma00 = dgamma[k0, i0]
    gamma_xp, gamma_xm = dgamma[k0, i0 + 1], dgamma[k0, i0 - 1]
    gamma_yp, gamma_ym = dgamma[k0 + 1, i0], dgamma[k0 - 1, i0]
    if not all(np.isfinite(v) for v in (gamma00, gamma_xp, gamma_xm, gamma_yp, gamma_ym)):
        return unrefined()

    denom_x = 2.0 * gamma00 - gamma_xp - gamma_xm
    denom_y = 2.0 * gamma00 - gamma_yp - gamma_ym
    frac_x = 0.5 * (gamma_xp - gamma_xm) / denom_x if denom_x != 0 else 0.0
    frac_y = 0.5 * (gamma_yp - gamma_ym) / denom_y if denom_y != 0 else 0.0
    frac_x = float(np.clip(frac_x, -0.5, 0.5))
    frac_y = float(np.clip(frac_y, -0.5, 0.5))

    daz_native = float(az_centers_deg[k0 + 1] - az_centers_deg[k0 - 1]) / 2.0
    return {
        "r0_km": float(rng_centers_km[i0]) + frac_x * range_spacing_km,
        "beta0_deg": float(az_centers_deg[k0]) + frac_y * daz_native,
        "cell_value": float(gamma00),
        "refined": True,
    }


def _circle_ring_points(r0_km, beta0_deg, alpha_deg, rho_km, n_half=_DJWR_RING_HALF_POINTS):
    """Ordered closed ring of (range, azimuth) points on a circle of radius
    rho_km around (r0_km, alpha_deg, beta0_deg), equally spaced in slant
    range (DJWR Eqs. 27 and 30). Returns (r_pts_km, beta_pts_deg), 2*n_half long."""
    cos_a = np.cos(np.deg2rad(alpha_deg))
    m = np.arange(0, n_half + 1, dtype=float)
    rm = r0_km - rho_km + (2.0 * rho_km * m) / n_half

    numer = rho_km ** 2 - (rm - r0_km) ** 2
    denom = 4.0 * rm * r0_km * (cos_a ** 2)
    denom_safe = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    ratio = np.clip(numer / denom_safe, 0.0, 1.0)
    delta = np.nan_to_num(2.0 * np.rad2deg(np.arcsin(np.sqrt(ratio))), nan=0.0)

    beta_plus = beta0_deg + delta
    beta_minus = beta0_deg - delta

    if n_half > 1:
        r_back = rm[n_half - 1:0:-1]
        b_back = beta_minus[n_half - 1:0:-1]
    else:
        r_back = rm[:0]
        b_back = beta_minus[:0]

    return np.concatenate([rm, r_back]), np.concatenate([beta_plus, b_back])


def _doppler_circle_kinematics(data_field, az_data_deg, rng_data_km,
                                r0_km, beta0_deg, alpha_deg, rho_km,
                                n_half=_DJWR_RING_HALF_POINTS):
    """Observed Doppler circulation and areal contraction rate around one
    circle (DJWR Eqs. 14/24) plus derived actual (doubled)
    circulation, mean tangential/inflow velocities, total wind speed and
    inflow angle (Wood et al. 2025, Eqs. 10-14). None if too much of the ring is missing."""
    r_pts, beta_pts = _circle_ring_points(r0_km, beta0_deg, alpha_deg, rho_km, n_half)
    vd_pts = np.array([
        bilinear_interp_full_sweep(data_field, az_data_deg, rng_data_km, r, b)
        for r, b in zip(r_pts, beta_pts)
    ])
    n_total = len(vd_pts)
    n_valid = int(np.isfinite(vd_pts).sum())
    if n_valid < n_total * 0.8:
        return None

    if n_valid < n_total:
        # Lightweight stand-in for DJWR's variational gap-filling.
        idx = np.arange(n_total)
        good = np.isfinite(vd_pts)
        vd_pts = np.interp(idx, idx[good], vd_pts[good], period=n_total)

    cos_a = np.cos(np.deg2rad(alpha_deg))
    gamma_ob = 0.5 * _shoelace_L(vd_pts, r_pts)
    neg_area_rate_ob = 0.5 * cos_a * _shoelace_L(r_pts * vd_pts, beta_pts)

    v_tang = gamma_ob / (np.pi * rho_km) if rho_km > 0 else float("nan")
    v_infl = -neg_area_rate_ob / (np.pi * rho_km) if rho_km > 0 else float("nan")

    return {
        "rho_km": rho_km,
        "gamma_ob": float(gamma_ob),
        "gamma_alpha": float(2.0 * gamma_ob),
        "neg_area_rate_ob": float(neg_area_rate_ob),
        "v_tang": float(v_tang),
        "v_infl": float(v_infl),
        "q_spd": float(np.hypot(v_infl, v_tang)),
        "inflow_angle_deg": float(np.degrees(np.arctan2(gamma_ob, -neg_area_rate_ob))),
        "n_valid": n_valid,
        "n_total": n_total,
    }


def compute_djwr_circulation_analysis(sub, rng_sub_km, az_sub_deg,
                                       full_data_field, full_az_deg,
                                       full_rng_km, alpha_deg, radii_km,
                                       range_spacing_km, manual_center_rb=None):
    """Locate the Doppler circulation center (r0, alpha, beta0) in the
    box-selected sector, then compute circle-based circulation, areal
    contraction rate and derived kinematics for each radius. Circles are
    sampled from the FULL sweep (they can extend beyond the box).

    manual_center_rb: optional (r0_km, beta0_deg) that bypasses the
    automatic center search. Returns None if no center could be located.
    """
    if manual_center_rb is not None:
        r0_km, beta0_deg = float(manual_center_rb[0]), float(manual_center_rb[1])
        center_source = "manual"
    else:
        dgamma, rng_centers = _cell_doppler_circulation_grid(sub, rng_sub_km, range_spacing_km)
        if dgamma is None:
            return None
        az_sub_deg = np.asarray(az_sub_deg, dtype=float)
        az_centers = 0.5 * (az_sub_deg[:-1] + az_sub_deg[1:])
        center = _locate_circulation_center(dgamma, rng_centers, az_centers, range_spacing_km)
        if center is None:
            return None
        r0_km = center["r0_km"]
        beta0_deg = center["beta0_deg"]
        center_source = "refined" if center.get("refined", False) else "cell center"

    circles = []
    for rho in radii_km:
        if rho is None or rho <= 0:
            continue
        result = _doppler_circle_kinematics(
            full_data_field, full_az_deg, full_rng_km,
            r0_km, beta0_deg, alpha_deg, float(rho),
        )
        if result is not None:
            circles.append(result)

    return {
        "r0_km": r0_km,
        "beta0_deg": beta0_deg,
        "alpha_deg": alpha_deg,
        "center_refined": center_source == "refined",
        "center_source": center_source,
        "circles": circles,
    }
