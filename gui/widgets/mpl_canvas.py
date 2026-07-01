"""Embed a matplotlib figure (or a saved PNG) inside a customtkinter frame."""
from __future__ import annotations

from pathlib import Path

import customtkinter as ctk

from gui import theme


class MplCanvas(ctk.CTkFrame):
    def __init__(self, master, **kw):
        kw.setdefault("fg_color", theme.PANEL)
        kw.setdefault("corner_radius", theme.CORNER)
        super().__init__(master, **kw)
        self._canvas = None
        self._placeholder = ctk.CTkLabel(
            self, text="No plot yet", font=theme.FONT_BODY,
            text_color=theme.TEXT_MUTED,
        )
        self._placeholder.pack(expand=True)

    def _clear(self):
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._placeholder is not None:
            self._placeholder.pack_forget()

    def show_figure(self, fig):
        """Display a matplotlib Figure."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self._clear()
        fig.patch.set_facecolor(theme.PANEL)
        self._canvas = FigureCanvasTkAgg(fig, master=self)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    def show_image(self, path: str | Path):
        """Load a saved PNG into a fresh figure and display it."""
        import matplotlib.image as mpimg
        from matplotlib.figure import Figure

        p = Path(path)
        if not p.exists():
            self.show_message(f"missing: {p.name}")
            return
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(mpimg.imread(str(p)))
        ax.axis("off")
        fig.tight_layout(pad=0)
        self.show_figure(fig)

    def show_message(self, msg: str):
        self._clear()
        self._placeholder = ctk.CTkLabel(
            self, text=msg, font=theme.FONT_BODY, text_color=theme.TEXT_MUTED,
        )
        self._placeholder.pack(expand=True)
