"""
TURT -- Tornadic-vortex Utility Radar Toolkit.

Drag a box around a velocity couplet on a PPI, zoom it into a gate-level
range/azimuth "fan", fit and diagnose the vortex, and export the fan as
real radar data. Split into independent modules:

Pure science / data modules (no GUI imports; usable from scripts and notebooks):
    colors.py        velocity colormap + display-range helpers
    synthetic.py     built-in synthetic TVS demo volume
    polar_grid.py    range/azimuth aggregation, interpolation, geometry helpers
    rankine.py       Rankine combined-vortex fit
    djwr.py          DJWR Doppler circulation analysis
    patch.py         WDSS-II "Patch" velocity-couplet parameters
    arc_sections.py  azimuthal cross-section diagnostics
    radar_session.py RadarSession: loads/holds a radar volume, sweep/gate helpers
    selection.py     FanSelection: box -> cropped ray/gate window
    analysis.py      VortexAnalysis: runs Rankine/DJWR/Patch/arc analyses on a selection
    fan_export.py    FanRadarExporter: fan -> radar file, and full-sweep dummy radar

Rendering (matplotlib only, no Tk):
    render_common.py shared cell styling, legend strip, contour overlay
    fan_renderer.py  FanRenderer   (true-shaped polar fan)
    flat_renderer.py FlatTableRenderer

Tk GUI pieces:
    box_selector.py  BoxSelector   (drag-box on the PPI)
    manual_picks.py  ManualPicker  (manual vortex center / +/- peak picks)
    ppi_view.py      PPIView       (main PPI canvas, composes the two above)
    control_panel.py ControlPanel  (compact toolbar: dropdowns + menus)
    settings_dialog.py AnalysisSettingsDialog (radii, storm motion, density, ...)
    results_window.py ResultsWindow
    arc_window.py    ArcCrossSectionWindow
    figure_window.py FigurePresenter (save PNG + popout window)
    app.py           TurtApp       (wires everything together)

Startup:
    banner.py        terminal splash text
    launch.py        launch(): splash -> imports -> app
"""

__title__ = "TURT"
__fullname__ = "Tornadic-vortex Utility Radar Toolkit"
__version__ = "1.0.0"
