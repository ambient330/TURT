"""FigurePresenter: save a figure as a 500-DPI PNG and pop it out live in Tk."""

import datetime
import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class FigurePresenter:
    def __init__(self, parent):
        self.parent = parent

    def present(self, fig, fname, tag):
        """Ask where to save, save at 500 DPI, then show the same figure in its
        own popout window. Returns a status-bar message."""
        fig_w, fig_h = fig.get_size_inches()
        default_name = f"turt_grid_{fname}_{tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = filedialog.asksaveasfilename(
            title="Save TVS grid image", initialfile=default_name,
            defaultextension=".png", filetypes=[("PNG image", "*.png")],
        )
        if not save_path:
            # Cancelled -- fall back to a default location so the render isn't thrown away.
            save_path = os.path.join(os.path.expanduser("~"), default_name)

        fig.savefig(save_path, dpi=500, bbox_inches="tight")

        win = tk.Toplevel(self.parent)
        win.title(f"TVS Grid -- {fname} ({tag})  [saved: {os.path.basename(save_path)}]")
        win.geometry(f"{int(np.clip(fig_w * 95, 500, 1500))}x{int(np.clip(fig_h * 95, 400, 1200))}")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, win).update()
        canvas.draw()

        return f"Saved TVS grid image ({fig_w:.1f}x{fig_h:.1f} in @ 500 DPI) to {save_path}"
