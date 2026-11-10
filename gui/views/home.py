"""Home: project directory + quick navigation + recent runs."""
from __future__ import annotations

from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from gui import theme
from gui.widgets.card import Card


class HomeView(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color=theme.BG)
        self.app = app

        wrap = ctk.CTkScrollableFrame(self, fg_color=theme.BG)
        wrap.pack(fill="both", expand=True, padx=18, pady=18)

        ctk.CTkLabel(
            wrap, text="PTBP DFTB Parameterization Studio",
            font=theme.FONT_H1, text_color=theme.TEXT_BRIGHT, anchor="w",
        ).pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            wrap, text="Generate Slater-Koster files, build EOS training sets, "
                       "and drive Bayesian / PSO parameter optimisation.",
            font=theme.FONT_BODY, text_color=theme.TEXT_MUTED, anchor="w",
        ).pack(fill="x", pady=(0, 16))

        # Project directory card.
        proj = Card(wrap, title="Project directory")
        proj.pack(fill="x", pady=(0, 14))
        row = ctk.CTkFrame(proj.body, fg_color="transparent")
        row.pack(fill="x")
        self.dir_var = ctk.StringVar(value=str(self.app.project_dir))
        ctk.CTkEntry(row, textvariable=self.dir_var, font=theme.FONT_BODY,
                     fg_color=theme.BG, border_color=theme.BORDER).pack(
            side="left", fill="x", expand=True)
        ctk.CTkButton(row, text="Browse", width=90, corner_radius=theme.CORNER_SM,
                      fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                      command=self._browse).pack(side="left", padx=(10, 0))

        # Quick-nav card.
        nav = Card(wrap, title="Quick start")
        nav.pack(fill="x", pady=(0, 14))
        grid = ctk.CTkFrame(nav.body, fg_color="transparent")
        grid.pack(fill="x")
        actions = [
            ("\U0001F9EC  Structures & EOS builder", "structures"),
            ("⚙  Generate SKF files", "generate"),
            ("▶  Optimize parameters", "optimize"),
            ("\U0001FA7A  Calculator / dependencies", "calculator"),
        ]
        for i, (label, key) in enumerate(actions):
            ctk.CTkButton(
                grid, text=label, anchor="w", height=44,
                corner_radius=theme.CORNER, fg_color=theme.PANEL_ALT,
                hover_color=theme.ACCENT_ACTIVE, text_color=theme.TEXT,
                font=theme.FONT_BODY, command=lambda k=key: self.app.show_view(k),
            ).grid(row=i // 2, column=i % 2, sticky="ew", padx=6, pady=6)
        grid.grid_columnconfigure((0, 1), weight=1)

        # Recent runs card.
        self.recent = Card(wrap, title="Recent run folders")
        self.recent.pack(fill="x")
        self._recent_body = self.recent.body

    def on_show(self):
        self.dir_var.set(str(self.app.project_dir))
        for w in self._recent_body.winfo_children():
            w.destroy()
        runs = sorted(self.app.project_dir.glob("run_*"), reverse=True)[:8]
        if not runs:
            ctk.CTkLabel(self._recent_body, text="No run_* folders yet.",
                         font=theme.FONT_BODY, text_color=theme.TEXT_MUTED,
                         anchor="w").pack(fill="x")
            return
        for r in runs:
            if not r.is_dir():
                continue
            ctk.CTkButton(
                self._recent_body, text=f"\U0001F4C8  {r.name}", anchor="w",
                height=34, corner_radius=theme.CORNER_SM, fg_color="transparent",
                hover_color=theme.PANEL_ALT, text_color=theme.TEXT,
                font=theme.FONT_BODY,
                command=lambda p=r: self._open_run(p),
            ).pack(fill="x", pady=2)

    def _open_run(self, path: Path):
        self.app.shared["monitor_run_dir"] = str(path)
        self.app.show_view("monitor")

    def _browse(self):
        d = filedialog.askdirectory(initialdir=str(self.app.project_dir))
        if d:
            self.app.set_project_dir(d)
            self.dir_var.set(d)
            self.on_show()
