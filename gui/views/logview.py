"""Logs: browse the consolidated gui_run_*.log files and tail their content."""
from __future__ import annotations

from pathlib import Path

import customtkinter as ctk

from gui import theme
from gui.widgets.card import Card
from gui.widgets.log_panel import LogPanel


class LogView(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color=theme.BG)
        self.app = app

        outer = ctk.CTkFrame(self, fg_color=theme.BG)
        outer.pack(fill="both", expand=True, padx=14, pady=14)
        outer.grid_columnconfigure(1, weight=1)
        outer.grid_rowconfigure(0, weight=1)

        # left: file list
        left = Card(outer, title="Log files")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        top = ctk.CTkFrame(left.body, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkButton(top, text="Refresh", width=90, corner_radius=theme.CORNER_SM,
                      fg_color=theme.PANEL_ALT, hover_color=theme.ACCENT_ACTIVE,
                      command=self.refresh).pack(side="left")
        ctk.CTkButton(top, text="Latest", width=80, corner_radius=theme.CORNER_SM,
                      fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                      command=self._open_latest).pack(side="left", padx=(8, 0))
        self.list_body = ctk.CTkScrollableFrame(left.body, fg_color="transparent",
                                                width=240)
        self.list_body.pack(fill="both", expand=True, pady=(8, 0))

        # right: content
        right = Card(outer, title="Content")
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        self.panel = LogPanel(right.body)
        self.panel.pack(fill="both", expand=True)

    @property
    def _log_dir(self) -> Path:
        return self.app.project_dir / "gui_logs"

    def on_show(self):
        self.refresh()

    def _logs(self):
        d = self._log_dir
        if not d.is_dir():
            return []
        return sorted(d.glob("gui_run_*.log"), reverse=True)

    def refresh(self):
        for w in self.list_body.winfo_children():
            w.destroy()
        logs = self._logs()
        if not logs:
            ctk.CTkLabel(self.list_body, text="(no logs yet)", font=theme.FONT_SMALL,
                         text_color=theme.TEXT_MUTED, anchor="w").pack(fill="x")
            return
        for p in logs:
            ctk.CTkButton(
                self.list_body, text=p.name, anchor="w", height=30,
                corner_radius=theme.CORNER_SM, fg_color="transparent",
                hover_color=theme.PANEL_ALT, text_color=theme.TEXT,
                font=theme.FONT_SMALL, command=lambda x=p: self.panel.load_file(x),
            ).pack(fill="x", pady=1)

    def _open_latest(self):
        logs = self._logs()
        if logs:
            self.panel.load_file(logs[0])
