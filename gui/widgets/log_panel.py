"""A read-only, auto-scrolling text panel for streaming command output."""
from __future__ import annotations

from pathlib import Path

import customtkinter as ctk

from gui import theme


class LogPanel(ctk.CTkFrame):
    def __init__(self, master, **kw):
        kw.setdefault("fg_color", theme.BG)
        kw.setdefault("corner_radius", theme.CORNER)
        super().__init__(master, **kw)
        self.textbox = ctk.CTkTextbox(
            self, font=theme.FONT_CODE, fg_color=theme.BG,
            text_color=theme.TEXT, wrap="none", corner_radius=theme.CORNER,
        )
        self.textbox.pack(fill="both", expand=True, padx=2, pady=2)
        self.textbox.configure(state="disabled")

    def append(self, line: str):
        """Append a line. Safe to call from any thread via `.after`."""
        def _do():
            self.textbox.configure(state="normal")
            self.textbox.insert("end", line + "\n")
            self.textbox.see("end")
            self.textbox.configure(state="disabled")
        try:
            self.after(0, _do)
        except Exception:
            _do()

    def clear(self):
        self.textbox.configure(state="normal")
        self.textbox.delete("1.0", "end")
        self.textbox.configure(state="disabled")

    def load_file(self, path: str | Path):
        self.clear()
        p = Path(path)
        if not p.exists():
            self.append(f"(no such file: {p})")
            return
        self.textbox.configure(state="normal")
        self.textbox.insert("end", p.read_text(errors="replace"))
        self.textbox.see("end")
        self.textbox.configure(state="disabled")
