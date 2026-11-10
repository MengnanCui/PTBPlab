"""Compact form builder: labelled rows bound to tkinter variables.

Keeps the views terse — each `add_*` call lays out a "Label ........ [widget]"
row and returns the variable so callers can read/write values. `values()`
dumps the whole form as a dict.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import customtkinter as ctk

from gui import theme


class FormBuilder:
    def __init__(self, master):
        self.master = master
        self._vars: Dict[str, object] = {}
        master.grid_columnconfigure(1, weight=1)
        self._row = 0

    def _label(self, text: str):
        lbl = ctk.CTkLabel(
            self.master, text=text, font=theme.FONT_BODY,
            text_color=theme.TEXT, anchor="w",
        )
        lbl.grid(row=self._row, column=0, sticky="w", padx=(4, 12), pady=6)

    def add_entry(self, key: str, label: str, default: str = "",
                  placeholder: str = "") -> ctk.StringVar:
        self._label(label)
        var = ctk.StringVar(value=default)
        ent = ctk.CTkEntry(
            self.master, textvariable=var, placeholder_text=placeholder,
            corner_radius=theme.CORNER_SM, border_color=theme.BORDER,
            fg_color=theme.BG, font=theme.FONT_BODY,
        )
        ent.grid(row=self._row, column=1, sticky="ew", pady=6)
        self._vars[key] = var
        self._row += 1
        return var

    def add_combo(self, key: str, label: str, options: List[str],
                  default: Optional[str] = None,
                  command: Optional[Callable[[str], None]] = None) -> ctk.StringVar:
        self._label(label)
        var = ctk.StringVar(value=default or (options[0] if options else ""))
        combo = ctk.CTkOptionMenu(
            self.master, variable=var, values=options, command=command,
            corner_radius=theme.CORNER_SM, fg_color=theme.BG,
            button_color=theme.ACCENT, button_hover_color=theme.ACCENT_HOVER,
            font=theme.FONT_BODY,
        )
        combo.grid(row=self._row, column=1, sticky="ew", pady=6)
        self._vars[key] = var
        self._row += 1
        return var

    def add_switch(self, key: str, label: str, default: bool = False) -> ctk.BooleanVar:
        self._label(label)
        var = ctk.BooleanVar(value=default)
        sw = ctk.CTkSwitch(
            self.master, text="", variable=var,
            progress_color=theme.ACCENT,
        )
        sw.grid(row=self._row, column=1, sticky="w", pady=6)
        self._vars[key] = var
        self._row += 1
        return var

    def add_static(self, label: str, value: str):
        """A non-editable info row."""
        self._label(label)
        lbl = ctk.CTkLabel(
            self.master, text=value, font=theme.FONT_BODY,
            text_color=theme.TEXT_MUTED, anchor="w",
        )
        lbl.grid(row=self._row, column=1, sticky="w", pady=6)
        self._row += 1
        return lbl

    def values(self) -> Dict[str, object]:
        return {k: v.get() for k, v in self._vars.items()}

    def var(self, key: str):
        return self._vars[key]
