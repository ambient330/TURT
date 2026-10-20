"""ManualPicker: manual vortex-center and outbound/inbound peak picking on the PPI."""

import numpy as np


class ManualPicker:
    """Single-click picking modes layered on the PPI axes.

    - center: a fixed vortex center (green star) that overrides automatic
      centering for the Rankine fit (center held fixed), DJWR circles, Patch
      and arc cross-sections.
    - +peak / -peak: the outbound / inbound velocity bins; once both are set
      their midpoint automatically becomes the manual center.

    Coordinates are true north-up Cartesian km, radar-relative.
    """

    def __init__(self, ax, canvas, notify, on_labels_changed=None):
        """notify(message, clear_after_ms=None); on_labels_changed(center_text, peaks_text)."""
        self.ax = ax
        self.canvas = canvas
        self.notify = notify
        self.on_labels_changed = on_labels_changed

        self.center_km = None
        self.pos_peak_km = None
        self.neg_peak_km = None
        self._center_artist = None
        self._pos_artists = []
        self._neg_artists = []

        self._picking_center = False
        self._picking_pos = False
        self._picking_neg = False

    NO_CENTER_TEXT = "No manual center set (using automatic centering)."
    NO_PEAKS_TEXT = "No manual peaks set (using automatic detection)."

    # ------------------------------------------------------------ arming
    @property
    def armed(self):
        return self._picking_center or self._picking_pos or self._picking_neg

    def start_center_pick(self):
        self._picking_center, self._picking_pos, self._picking_neg = True, False, False
        self.notify("Click on the PPI where the vortex center should be.")

    def start_pick_pos_peak(self):
        self._picking_pos, self._picking_neg, self._picking_center = True, False, False
        self.notify("Click on the PPI at the OUTBOUND (+) velocity peak.")

    def start_pick_neg_peak(self):
        self._picking_neg, self._picking_pos, self._picking_center = True, False, False
        self.notify("Click on the PPI at the INBOUND (\u2212) velocity peak.")

    def handle_click(self, x_km, y_km):
        """Consume a click if a pick mode is armed. Returns True if consumed."""
        if self._picking_pos:
            self._set_peak("pos", x_km, y_km)
            self._picking_pos = False
            self.notify("Outbound (+) peak set. Pick the inbound (\u2212) peak next, or "
                        "re-pick this one anytime.", 4000)
            return True
        if self._picking_neg:
            self._set_peak("neg", x_km, y_km)
            self._picking_neg = False
            self.notify("Inbound (\u2212) peak set. With both peaks picked, their midpoint "
                        "is now the manual vortex center.", 4000)
            return True
        if self._picking_center:
            self.set_center(x_km, y_km)
            self._picking_center = False
            self.notify("Manual vortex center set -- it will be used for the Rankine fit "
                        "and/or DJWR circulation circles on the next 'Zoom to TVS Grid'. "
                        "Click 'Clear Center' to go back to automatic.", 4000)
            return True
        return False

    # ------------------------------------------------------------ center
    def set_center(self, x_km, y_km):
        self.center_km = (float(x_km), float(y_km))
        self._remove(self._center_artist)
        (self._center_artist,) = self.ax.plot(
            x_km, y_km, marker="*", color="lime", markersize=18,
            markeredgecolor="black", markeredgewidth=1.2, zorder=20)
        self._labels_changed()
        self.canvas.draw_idle()

    def clear_all(self):
        """Clear the manual center AND the +/- peaks (they're linked)."""
        self.center_km = None
        self._picking_center = False
        self._remove(self._center_artist)
        self._center_artist = None
        self._clear_peaks()
        self._labels_changed()
        self.canvas.draw_idle()

    # ------------------------------------------------------------- peaks
    def _set_peak(self, which, x_km, y_km):
        x_km, y_km = float(x_km), float(y_km)
        color = "red" if which == "pos" else "blue"
        label = "+" if which == "pos" else "\u2212"
        old = self._pos_artists if which == "pos" else self._neg_artists
        if which == "pos":
            self.pos_peak_km = (x_km, y_km)
        else:
            self.neg_peak_km = (x_km, y_km)
        for art in old:
            self._remove(art)

        (marker,) = self.ax.plot(x_km, y_km, marker="x", color=color, markersize=12,
                                 markeredgewidth=2.5, zorder=21)
        txt = self.ax.annotate(label, (x_km, y_km), textcoords="offset points", xytext=(6, 6),
                               fontsize=11, fontweight="bold", color=color, zorder=21)
        if which == "pos":
            self._pos_artists = [marker, txt]
        else:
            self._neg_artists = [marker, txt]

        if self.pos_peak_km is not None and self.neg_peak_km is not None:
            self.set_center(0.5 * (self.pos_peak_km[0] + self.neg_peak_km[0]),
                            0.5 * (self.pos_peak_km[1] + self.neg_peak_km[1]))
        else:
            self._labels_changed()
        self.canvas.draw_idle()

    def _clear_peaks(self):
        self.pos_peak_km = self.neg_peak_km = None
        self._picking_pos = self._picking_neg = False
        for art in self._pos_artists + self._neg_artists:
            self._remove(art)
        self._pos_artists, self._neg_artists = [], []

    # ------------------------------------------------------------ labels
    @staticmethod
    def _remove(artist):
        if artist is None:
            return
        try:
            artist.remove()
        except Exception:
            pass

    def center_text(self):
        if self.center_km is None:
            return self.NO_CENTER_TEXT
        x, y = self.center_km
        r_km = float(np.hypot(x, y))
        beta_deg = float(np.degrees(np.arctan2(x, y))) % 360.0
        return f"Manual center: x={x:.2f} km, y={y:.2f} km (r={r_km:.2f} km, \u03b2={beta_deg:.1f}\u00b0)"

    def peaks_text(self):
        parts = []
        if self.pos_peak_km is not None:
            parts.append(f"+peak=({self.pos_peak_km[0]:.2f}, {self.pos_peak_km[1]:.2f}) km")
        if self.neg_peak_km is not None:
            parts.append(f"\u2212peak=({self.neg_peak_km[0]:.2f}, {self.neg_peak_km[1]:.2f}) km")
        return "  ".join(parts) if parts else self.NO_PEAKS_TEXT

    def _labels_changed(self):
        if self.on_labels_changed:
            self.on_labels_changed(self.center_text(), self.peaks_text())
