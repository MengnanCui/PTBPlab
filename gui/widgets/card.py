"""A rounded titled container — the basic building block of every view."""
from __future__ import annotations

import customtkinter as ctk

from gui import theme


class Card(ctk.CTkFrame):
    """A padded, rounded panel with an optional title. Put content in
    `self.body`."""

    def __init__(self, master, title: str | None = None, **kw):
        kw.setdefault("fg_color", theme.PANEL)
        kw.setdefault("corner_radius", theme.CORNER)
        kw.setdefault("border_width", 1)
        kw.setdefault("border_color", theme.BORDER)
        super().__init__(master, **kw)

        if title:
            self.title_label = ctk.CTkLabel(
                self, text=title, font=theme.FONT_H2, text_color=theme.TEXT_BRIGHT,
                anchor="w",
            )
            self.title_label.pack(fill="x", padx=14, pady=(12, 6))

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
