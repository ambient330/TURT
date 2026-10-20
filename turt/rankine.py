"""Rankine combined-vortex (+ uniform background wind) fit to Doppler velocity."""

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None

SCIPY_AVAILABLE = least_squares is not None


def rankine_predicted_vr(params, x_km, y_km):
    """Forward model: predicted Doppler (radial) velocity at gates
    (x_km, y_km -- radar-relative, north-up, km) for a Rankine combined
    vortex plus uniform background wind.

    params = (xc, yc, Rmax_km, Vmax_ms, u0_ms, v0_ms). Positive Vmax is
    cyclonic. V_t = Vmax*(r/Rmax) inside the core, Vmax*(Rmax/r) outside.
    The measured velocity is the component of the total wind along the
    radar-to-gate direction (beam-elevation effects ignored, the standard
    low-level TVS approximation).
    """
    xc, yc, r_max, v_max, u0, v0 = params
    dx = x_km - xc
    dy = y_km - yc
    r = np.hypot(dx, dy)
    r_safe = np.where(r < 1e-6, 1e-6, r)
    r_max_safe = max(float(r_max), 1e-6)
    v_t = np.where(r <= r_max_safe, v_max * (r / r_max_safe), v_max * (r_max_safe / r_safe))
    tx = -dy / r_safe
    ty = dx / r_safe
    wind_x = v_t * tx + u0
    wind_y = v_t * ty + v0
    rng = np.hypot(x_km, y_km)
    rng_safe = np.where(rng < 1e-6, 1e-6, rng)
    return wind_x * (x_km / rng_safe) + wind_y * (y_km / rng_safe)


def fit_rankine_vortex(vel_sub, gx_sub, gy_sub, fixed_center=None):
    """Fit a Rankine vortex + background wind to the box's raw radial
    velocities via nonlinear least squares.

    fixed_center: optional (xc_km, yc_km) held fixed (only Rmax, Vmax and
    the background wind are optimized). Otherwise the center is fit too,
    seeded from the classic TDA-style inbound/outbound extrema midpoint.

    Returns a dict of fitted parameters / fit-quality metrics, or None if
    scipy is missing, there is too little data, or the fit fails.
    """
    if least_squares is None:
        return None

    mask = np.ma.getmaskarray(vel_sub) if np.ma.is_masked(vel_sub) else np.zeros(vel_sub.shape, dtype=bool)
    vals = np.ma.filled(vel_sub.astype(float), np.nan)
    valid = (~mask) & np.isfinite(vals) & np.isfinite(gx_sub) & np.isfinite(gy_sub)
    if valid.sum() < 12:
        return None

    x = np.asarray(gx_sub, dtype=float)[valid]
    y = np.asarray(gy_sub, dtype=float)[valid]
    vr = vals[valid]
    box_span = max(float(x.max() - x.min()), float(y.max() - y.min()), 0.1)

    # Initial guess from the inbound/outbound extrema (far more reliable than
    # the box centroid, which can drift into a degenerate "background wind
    # explains everything" local minimum).
    i_max = int(np.nanargmax(vr))
    i_min = int(np.nanargmin(vr))
    x_max_pt, y_max_pt = x[i_max], y[i_max]
    x_min_pt, y_min_pt = x[i_min], y[i_min]
    extrema_dist = float(np.hypot(x_max_pt - x_min_pt, y_max_pt - y_min_pt))
    v_max0 = float((vr[i_max] - vr[i_min]) / 2.0)

    if fixed_center is not None:
        xc_fixed, yc_fixed = float(fixed_center[0]), float(fixed_center[1])
        r_max0 = max(
            float(np.hypot(x_max_pt - xc_fixed, y_max_pt - yc_fixed)),
            float(np.hypot(x_min_pt - xc_fixed, y_min_pt - yc_fixed)),
            0.05,
        )
    else:
        xc0 = (x_max_pt + x_min_pt) / 2.0
        yc0 = (y_max_pt + y_min_pt) / 2.0
        r_max0 = max(extrema_dist / 2.0, 0.05)

    r_max0 = float(np.clip(r_max0, 0.01, box_span))

    if fixed_center is not None:
        def residuals(params, x_=x, y_=y, vr_=vr):
            r_max, v_max, u0, v0 = params
            return rankine_predicted_vr((xc_fixed, yc_fixed, r_max, v_max, u0, v0), x_, y_) - vr_

        try:
            # Stage 1: vortex alone (background wind pinned at 0).
            result1 = least_squares(
                residuals, [r_max0, v_max0, 0.0, 0.0],
                bounds=([0.01, -120.0, 0.0, 0.0], [box_span, 120.0, 1e-6, 1e-6]), method="trf",
            )
            # Stage 2: unlock the background wind and refine.
            result = least_squares(
                residuals, list(result1.x[:2]) + [0.0, 0.0],
                bounds=([0.01, -120.0, -60.0, -60.0], [box_span, 120.0, 60.0, 60.0]), method="trf",
            )
        except Exception:
            return None
        if not result.success:
            return None
        r_max, v_max, u0, v0 = result.x
        xc, yc = xc_fixed, yc_fixed
    else:
        lower = [float(x.min()), float(y.min()), 0.01, -120.0, -60.0, -60.0]
        upper = [float(x.max()), float(y.max()), box_span, 120.0, 60.0, 60.0]
        r_max0 = float(np.clip(r_max0, lower[2], upper[2]))

        def residuals(params, x_=x, y_=y, vr_=vr):
            return rankine_predicted_vr(params, x_, y_) - vr_

        try:
            p0_stage1 = [xc0, yc0, r_max0, v_max0, 0.0, 0.0]
            result1 = least_squares(
                residuals, p0_stage1,
                bounds=(list(lower[:4]) + [0.0, 0.0], list(upper[:4]) + [1e-6, 1e-6]), method="trf",
            )
            result = least_squares(residuals, list(result1.x[:4]) + [0.0, 0.0],
                                   bounds=(lower, upper), method="trf")
        except Exception:
            return None
        if not result.success:
            return None
        xc, yc, r_max, v_max, u0, v0 = result.x

    rmse = float(np.sqrt(np.mean(result.fun ** 2)))
    vr_range = float(np.nanmax(vr) - np.nanmin(vr))
    return {
        "xc_km": float(xc),
        "yc_km": float(yc),
        "r_max_km": float(r_max),
        "v_max_ms": float(v_max),
        "u0_ms": float(u0),
        "v0_ms": float(v0),
        "rmse_ms": rmse,
        "n_points": int(valid.sum()),
        # Characteristic vorticity of a Rankine core: zeta = 2*Vmax/Rmax.
        "core_vorticity_s": float(2.0 * v_max / max(r_max, 1e-6) / 1000.0),
        "quality_ratio": float(rmse / vr_range) if vr_range > 0 else None,
        "center_fixed": fixed_center is not None,
    }
