"""Built-in synthetic TVS demo volume (used when no file is supplied)."""

import numpy as np

try:
    import pyart
except ImportError:
    pyart = None


def build_synthetic_tvs_radar():
    """Small synthetic single-sweep PPI with a tight velocity couplet and a
    hook-like reflectivity lobe, so the tool runs standalone for a demo."""
    nrays = 360
    ngates = 200
    gate_spacing = 250.0  # meters
    az = np.arange(nrays, dtype=float)
    rng = (np.arange(ngates) + 1) * gate_spacing

    rng_km = rng / 1000.0
    az_rad = np.deg2rad(az)[:, None]
    gx = rng_km[None, :] * np.sin(az_rad)
    gy = rng_km[None, :] * np.cos(az_rad)

    # TVS couplet centered near (x=1.0 km, y=1.0 km), radius ~1 km
    cx, cy = 1.0, 1.0
    dx = gx - cx
    dy = gy - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    core_radius = 0.9
    vmax = 30.0
    tangential = vmax * np.exp(-(r / core_radius) ** 2) * 2.5
    vel = tangential * np.cos(theta - np.pi / 2.0)

    vel = vel + np.random.normal(0, 1.2, size=vel.shape)
    vel = np.ma.masked_invalid(vel)

    storm_r = np.sqrt((gx - 2.0) ** 2 + (gy - 0.5) ** 2)
    dbz = 45.0 * np.exp(-(storm_r / 4.0) ** 2)
    hook_r = np.sqrt((gx - (cx - 1.3)) ** 2 + (gy - (cy - 0.8)) ** 2)
    dbz += 20.0 * np.exp(-(hook_r / 1.2) ** 2)
    dbz += np.random.normal(0, 2.0, size=dbz.shape)
    dbz = np.ma.masked_invalid(dbz)

    if pyart is not None:
        radar = pyart.testing.make_empty_ppi_radar(ngates, nrays, 1)
        radar.range["data"] = rng.astype("float32")
        radar.azimuth["data"] = az.astype("float32")
        radar.elevation["data"] = np.full(nrays, 0.5, dtype="float32")
        radar.fixed_angle["data"] = np.array([0.5], dtype="float32")
        radar.sweep_start_ray_index["data"] = np.array([0])
        radar.sweep_end_ray_index["data"] = np.array([nrays - 1])
        radar.instrument_parameters = {
            "nyquist_velocity": {"data": np.full(nrays, 33.0, dtype="float32")}
        }
        radar.add_field(
            "velocity",
            {"data": vel.astype("float32"), "units": "m/s",
             "long_name": "Radial velocity", "_FillValue": -9999.0},
            replace_existing=True,
        )
        radar.add_field(
            "reflectivity",
            {"data": dbz.astype("float32"), "units": "dBZ",
             "long_name": "Reflectivity", "_FillValue": -9999.0},
            replace_existing=True,
        )
        return radar

    # Minimal stand-in object matching the attributes the GUI reads
    class FakeRadar:
        pass

    r = FakeRadar()
    r.range = {"data": rng.astype("float32")}
    r.azimuth = {"data": az.astype("float32")}
    r.nsweeps = 1
    r.nrays = nrays
    r.sweep_start_ray_index = {"data": np.array([0])}
    r.sweep_end_ray_index = {"data": np.array([nrays - 1])}
    r.instrument_parameters = {
        "nyquist_velocity": {"data": np.full(nrays, 33.0, dtype="float32")}
    }
    r.fields = {
        "velocity": {"data": vel.astype("float32"), "units": "m/s", "_FillValue": -9999.0},
        "reflectivity": {"data": dbz.astype("float32"), "units": "dBZ", "_FillValue": -9999.0},
    }
    return r
