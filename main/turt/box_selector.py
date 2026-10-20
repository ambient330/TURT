"""BoxSelector: click-drag rectangle on the PPI axes."""

from matplotlib.patches import Rectangle as MplRectanglePatch


class BoxSelector:
    MIN_BOX_KM = 0.05  # ignore accidental micro-drags / plain clicks

    def __init__(self, ax, canvas, notify):
        """notify(message, clear_after_ms=None) reports status to the UI."""
        self.ax = ax
        self.canvas = canvas
        self.notify = notify
        self.points = []           # 4 rectangle corners (km) once finalized
        self._artists = []
        self._dragging = False
        self._start = None
        self._rect = None

    @property
    def has_box(self):
        return len(self.points) >= 3

    @property
    def dragging(self):
        return self._dragging

    def on_press(self, event):
        # Starting a new drag always replaces any previous box.
        self.clear()
        self._dragging = True
        self._start = (event.xdata, event.ydata)
        rect = MplRectanglePatch((event.xdata, event.ydata), 0, 0, fill=False,
                                 edgecolor="yellow", linewidth=1.6, linestyle="-", zorder=15)
        self.ax.add_patch(rect)
        self._rect = rect
        self._artists = [rect]
        self.canvas.draw_idle()

    def on_motion(self, event):
        if not self._dragging or self._rect is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x0, y0 = self._start
        self._update_rect(x0, y0, event.xdata, event.ydata)
        self.canvas.draw_idle()

    def on_release(self, event):
        if not self._dragging:
            return
        self._dragging = False
        if self._start is None:
            return
        x0, y0 = self._start

        # Use the release point if on the axes; otherwise the rectangle's last
        # extent, so dragging off the plot still finalizes sensibly.
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            x1, y1 = event.xdata, event.ydata
        elif self._rect is not None:
            rx, ry = self._rect.get_xy()
            x1, y1 = rx + self._rect.get_width(), ry + self._rect.get_height()
        else:
            self.clear()
            return

        if abs(x1 - x0) < self.MIN_BOX_KM or abs(y1 - y0) < self.MIN_BOX_KM:
            self.clear()
            self.notify("Box too small -- click and drag to draw a box around the TVS.", 2500)
            return

        self._update_rect(x0, y0, x1, y1)
        xlo, xhi = sorted((x0, x1))
        ylo, yhi = sorted((y0, y1))
        self.points = [(xlo, ylo), (xhi, ylo), (xhi, yhi), (xlo, yhi)]
        self.canvas.draw_idle()
        self.notify("Box set -- click 'Zoom to TVS Grid' (or draw a new box to replace it).", 3000)

    def _update_rect(self, x0, y0, x1, y1):
        xlo, xhi = sorted((x0, x1))
        ylo, yhi = sorted((y0, y1))
        self._rect.set_xy((xlo, ylo))
        self._rect.set_width(xhi - xlo)
        self._rect.set_height(yhi - ylo)

    def clear(self):
        for art in self._artists:
            try:
                art.remove()
            except Exception:
                pass
        self._artists = []
        self.points = []
        self._dragging = False
        self._start = None
        self._rect = None
        self.canvas.draw_idle()
