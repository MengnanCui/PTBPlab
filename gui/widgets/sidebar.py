"""VSCode-style activity bar: a vertical column of glyph buttons."""
from __future__ import annotations

from typing import Callable, Dict

import customtkinter as ctk

from gui import theme


class Sidebar(ctk.CTkFrame):
    def __init__(self, master, on_select: Callable[[str], None]):
        super().__init__(master, fg_color=theme.SIDEBAR, corner_radius=0, width=64)
        self.grid_propagate(False)
        self._on_select = on_select
        self._buttons: Dict[str, ctk.CTkButton] = {}
        self._active: str | None = None

        for key, glyph, label in theme.NAV_ITEMS:
            btn = ctk.CTkButton(
                self, text=glyph, width=48, height=48,
                corner_radius=theme.CORNER_SM,
                fg_color="transparent", hover_color=theme.PANEL_ALT,
                text_color=theme.TEXT_MUTED, font=(theme.FONT_FAMILY, 22),
                command=lambda k=key: self._on_select(k),
            )
            btn.pack(pady=(10, 0), padx=8)
            self._add_tooltip(btn, label)
            self._buttons[key] = btn

    def _add_tooltip(self, widget, text: str):
        # Lightweight hover label (Tk has no native tooltip).
        tip = {"win": None}

        def show(_e):
            if tip["win"] is not None:
                return
            x = widget.winfo_rootx() + 56
            y = widget.winfo_rooty() + 10
            win = ctk.CTkToplevel(widget)
            win.overrideredirect(True)
            win.geometry(f"+{x}+{y}")
            win.attributes("-topmost", True)
            ctk.CTkLabel(
                win, text=text, font=theme.FONT_SMALL, fg_color=theme.PANEL_ALT,
                text_color=theme.TEXT, corner_radius=theme.CORNER_SM, padx=8, pady=3,
            ).pack()
            tip["win"] = win

        def hide(_e):
            if tip["win"] is not None:
                tip["win"].destroy()
                tip["win"] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    def set_active(self, key: str):
        for k, btn in self._buttons.items():
            if k == key:
                btn.configure(fg_color=theme.ACCENT_ACTIVE, text_color=theme.TEXT_BRIGHT)
            else:
                btn.configure(fg_color="transparent", text_color=theme.TEXT_MUTED)
        self._active = key
