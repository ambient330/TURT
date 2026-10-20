"""VortexAnalysis: runs the optional Rankine / DJWR / Patch / arc analyses
on a FanSelection. No GUI code -- usable from scripts and notebooks."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .arc_sections import compute_arc_cross_sections, detect_vortex_center_from_arc_extrema
from .colors import is_velocity_like_field
from .djwr import compute_djwr_circulation_analysis
from .patch import compute_patch_parameters
from .rankine import SCIPY_AVAILABLE, fit_rankine_vortex


@dataclass
class AnalysisOptions:
    fit_rankine: bool = False
    circ_center: bool = False
    circ_radii: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    circ_radii_from_rankine: bool = False
    patch: bool = False
    storm_speed_ms: float = 0.0
    storm_dir_deg: float = 0.0
    arc_xsec: bool = False
    arc_radii: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    air_density: float = 1.0
    arc_delta_km: Optional[float] = None


@dataclass
class ManualInputs:
    """Snapshot of the user's manual picks (north-up km), any may be None."""
    center_km: Optional[Tuple[float, float]] = None
    pos_peak_km: Optional[Tuple[float, float]] = None
    neg_peak_km: Optional[Tuple[float, float]] = None


@dataclass
class AnalysisResults:
    rankine_fit: Optional[dict] = None
    circulation_result: Optional[dict] = None
    patch_result: Optional[dict] = None
    arcs: list = field(default_factory=list)
    arc_center_km: Optional[Tuple[float, float]] = None
    arc_air_density: float = 1.0
    extrema_result: Optional[dict] = None
    messages: List[str] = field(default_factory=list)

    def has_summary_tables(self):
        return (
            self.rankine_fit is not None
            or (self.circulation_result is not None and self.circulation_result.get("circles"))
            or self.patch_result is not None
        )


def resolve_vortex_center_km(manual_center_km=None, rankine_fit=None, circulation_result=None):
    """Best-available vortex center (north-up km): manual > Rankine fit >
    DJWR circulation center. None if none are available."""
    if manual_center_km is not None:
        return manual_center_km
    if rankine_fit is not None:
        return (rankine_fit["xc_km"], rankine_fit["yc_km"])
    if circulation_result is not None and circulation_result.get("circles"):
        r0c = circulation_result["r0_km"]
        beta0_rad = np.deg2rad(circulation_result["beta0_deg"])
        return (r0c * np.sin(beta0_rad), r0c * np.cos(beta0_rad))
    return None


class VortexAnalysis:
    """Stateless runner; call run(selection, options, manual) -> AnalysisResults."""

    def run(self, sel, opts: AnalysisOptions, manual: ManualInputs = None) -> AnalysisResults:
        manual = manual or ManualInputs()
        res = AnalysisResults()

        res.rankine_fit = self._rankine(sel, opts, manual, res)
        res.circulation_result = self._circulation(sel, opts, manual, res)
        if opts.patch:
            res.patch_result = self._patch(sel, opts, manual, res)
        if opts.arc_xsec:
            self._arc_sections(sel, opts, manual, res)
        return res

    # ---------------------------------------------------------- Rankine
    def _rankine(self, sel, opts, manual, res):
        need = opts.fit_rankine or (opts.circ_center and opts.circ_radii_from_rankine)
        if not need:
            return None
        if not SCIPY_AVAILABLE:
            res.messages.append("Rankine fit needs scipy -- install with: pip install scipy")
            return None
        if not is_velocity_like_field(sel.field_name):
            res.messages.append(
                f"Rankine fit skipped -- '{sel.field_name}' doesn't look like a velocity field.")
            return None
        try:
            fit = fit_rankine_vortex(sel.sub, sel.gx_sub, sel.gy_sub, fixed_center=manual.center_km)
        except Exception:
            import traceback
            print(traceback.format_exc())
            return None
        if fit is None:
            res.messages.append(
                "Rankine fit did not converge (or too little valid data) -- "
                "try a tighter box around the couplet.")
        elif manual.center_km is not None:
            res.messages.append(
                "Rankine fit used the manually picked center "
                "(only Rmax/Vmax/background wind were optimized).")
        return fit

    # ------------------------------------------------------------- DJWR
    def _circulation(self, sel, opts, manual, res):
        if not opts.circ_center:
            return None
        if not is_velocity_like_field(sel.field_name):
            res.messages.append(
                f"Circulation-center analysis skipped -- '{sel.field_name}' doesn't "
                "look like a velocity field.")
            return None

        radii = list(opts.circ_radii)
        if opts.circ_radii_from_rankine:
            if res.rankine_fit is not None and res.rankine_fit["r_max_km"] > 0:
                rmax_km = res.rankine_fit["r_max_km"]
                radii = [m * rmax_km for m in radii]
                res.messages.append(
                    f"Circle radii set from Rankine fit: Rmax={rmax_km * 1000:.0f} m -> "
                    f"rho = {', '.join(f'{r:.2f}' for r in radii)} km.")
            else:
                res.messages.append(
                    "Radii-from-Rankine requested but the Rankine fit did not converge -- "
                    "using the entered values as literal km instead.")

        rng_data = sel.rng_data
        range_spacing_km = float(np.median(np.abs(np.diff(rng_data)))) if rng_data.size > 1 else 0.25
        az_sub_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(sel.az_sub)))
        full_az_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(sel.az_data)))

        manual_center_rb = None
        if manual.center_km is not None:
            mx, my = manual.center_km
            manual_center_rb = (float(np.hypot(mx, my)), float(np.degrees(np.arctan2(mx, my))) % 360.0)
            res.messages.append(
                "Circulation circles centered on the manually picked "
                f"point (r0={manual_center_rb[0]:.2f} km, \u03b20={manual_center_rb[1]:.1f}\u00b0).")

        try:
            out = compute_djwr_circulation_analysis(
                sel.sub, sel.rng_sub, az_sub_unwrapped, sel.data, full_az_unwrapped,
                rng_data, sel.elev_deg, radii, range_spacing_km,
                manual_center_rb=manual_center_rb,
            )
        except Exception:
            import traceback
            print(traceback.format_exc())
            out = None
        if out is None:
            res.messages.append(
                "Circulation-center analysis: could not locate a center (too much "
                "missing data in the box) -- try a tighter/cleaner box around the couplet.")
        elif not out.get("circles"):
            res.messages.append(
                "Circulation center located, but none of the requested circle radii had "
                "enough valid data around their perimeter -- try smaller radii or a cleaner sector.")
        return out

    # ------------------------------------------------------------ Patch
    def _patch(self, sel, opts, manual, res):
        center = resolve_vortex_center_km(manual.center_km, res.rankine_fit, res.circulation_result)
        try:
            az_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(sel.az_sub)))
            out = compute_patch_parameters(
                sel.sub, sel.gx_sub, sel.gy_sub, sel.rng_sub, az_unwrapped,
                sel.elev_deg, storm_speed_ms=opts.storm_speed_ms,
                storm_dir_deg=opts.storm_dir_deg, center_xy_km=center,
            )
        except Exception:
            import traceback
            print(traceback.format_exc())
            out = None
        if out is None:
            res.messages.append(
                "Patch analysis: no clear inbound/outbound velocity couplet found in the "
                "box (or too little valid data).")
        return out

    # -------------------------------------------------- arc cross-sections
    def _arc_sections(self, sel, opts, manual, res):
        center = resolve_vortex_center_km(manual.center_km, res.rankine_fit, res.circulation_result)
        if center is None:
            res.messages.append(
                "Azimuthal cross-section needs a vortex center -- pick one manually, or "
                "enable the Rankine fit or circulation-center checkboxes.")
            return

        arc_radii = list(opts.arc_radii)
        full_az_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(sel.az_data)))

        # Manually picked +/- peaks override the automatic arc-extrema search.
        extrema = None
        if manual.pos_peak_km is not None and manual.neg_peak_km is not None:
            pos_xy, neg_xy = manual.pos_peak_km, manual.neg_peak_km
            diameter_km = float(np.hypot(pos_xy[0] - neg_xy[0], pos_xy[1] - neg_xy[1]))
            extrema = {
                "xc_km": 0.5 * (pos_xy[0] + neg_xy[0]),
                "yc_km": 0.5 * (pos_xy[1] + neg_xy[1]),
                "pos_xy_km": pos_xy, "neg_xy_km": neg_xy,
                "diameter_km": diameter_km,
                "radii_used_km": [], "per_radius": [], "manual": True,
            }
            center = (extrema["xc_km"], extrema["yc_km"])
            res.messages.append(
                "Azimuthal cross-section: vortex center set from the manually picked "
                f"+/- peaks (diameter={diameter_km * 1000:.0f} m).")
        else:
            search_radii = arc_radii[:min(3, len(arc_radii))]
            try:
                extrema = detect_vortex_center_from_arc_extrema(
                    sel.data, full_az_unwrapped, sel.rng_data, center[0], center[1], search_radii)
            except Exception:
                import traceback
                print(traceback.format_exc())
                extrema = None
            if extrema is not None:
                center = (extrema["xc_km"], extrema["yc_km"])
                res.messages.append(
                    "Azimuthal cross-section: vortex center refined from the detected "
                    f"outbound/inbound peaks (diameter={extrema['diameter_km'] * 1000:.0f} m, "
                    f"searched \u03c1={', '.join(f'{r:.2f}' for r in search_radii)} km).")
            else:
                res.messages.append(
                    "Azimuthal cross-section: arc-extrema center refinement found no clear "
                    "peak pair -- using the previous center.")
        res.extrema_result = extrema

        rankine_params = None
        if res.rankine_fit is not None:
            f = res.rankine_fit
            rankine_params = (f["xc_km"], f["yc_km"], f["r_max_km"], f["v_max_ms"], f["u0_ms"], f["v0_ms"])

        try:
            arcs = compute_arc_cross_sections(
                sel.data, full_az_unwrapped, sel.rng_data, center[0], center[1], arc_radii,
                air_density=opts.air_density, delta_km=opts.arc_delta_km,
                rankine_params=rankine_params,
            )
        except Exception:
            import traceback
            print(traceback.format_exc())
            arcs = []

        if not arcs:
            res.messages.append(
                "Azimuthal cross-section: no requested radius had enough valid data "
                "around its ring -- try different radii.")
            return
        res.arcs = arcs
        res.arc_center_km = center
        res.arc_air_density = opts.air_density
