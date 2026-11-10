"""PtbpGuiApp — the main window: activity bar + swappable content views.

Views are imported lazily (so matplotlib/ASE only load when a view is first
opened) and cached. Cross-view handoff (e.g. Optimize → Monitor after a run
starts) goes through `self.shared`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import customtkinter as ctk

from gui import theme

# view key → (module suffix, class name). Lazily imported in _make_view.
VIEW_REGISTRY = {
    "home":       ("home", "HomeView"),
    "structures": ("structures", "StructuresView"),
    "generate":   ("generate", "GenerateView"),
    "optimize":   ("optimize", "OptimizeView"),
    "monitor":    ("monitor", "MonitorView"),
    "calculator": ("calculator", "CalculatorView"),
    "logview":    ("logview", "LogView"),
}


class PtbpGuiApp(ctk.CTk):
    def __init__(self, project_dir: Optional[str | Path] = None):
        super().__init__()
        theme.apply()

        # Clean logging + mute customtkinter's benign redraw-race tracebacks.
        from gui.core.logging_setup import (configure_logging,
                                            install_tk_exception_handler)
        self.log = configure_logging()
        install_tk_exception_handler(self, self.log)

        self.title("PTBP — DFTB Parameterization Studio")
        self.geometry("1160x760")
        self.minsize(940, 620)
        self.configure(fg_color=theme.BG)

        # Shared, cross-view state.
        self.project_dir: Path = Path(project_dir or Path.cwd()).resolve()
        self.shared: Dict[str, object] = {"monitor_run_dir": None}

        self._views: Dict[str, ctk.CTkFrame] = {}
        self._current: Optional[str] = None

        self._build_layout()
        self.show_view("home")

    # ---- layout ----
    def _build_layout(self):
        from gui.widgets.sidebar import Sidebar

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.sidebar = Sidebar(self, on_select=self.show_view)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="ns")

        # Top header bar.
        self.header = ctk.CTkFrame(self, fg_color=theme.PANEL, corner_radius=0,
                                   height=44, border_width=0)
        self.header.grid(row=0, column=1, sticky="ew")
        self.header.grid_propagate(False)
        self.header_title = ctk.CTkLabel(
            self.header, text="", font=theme.FONT_H2, text_color=theme.TEXT_BRIGHT,
        )
        self.header_title.pack(side="left", padx=18)
        self.header_project = ctk.CTkLabel(
            self.header, text=f"\U0001F4C1 {self.project_dir}",
            font=theme.FONT_SMALL, text_color=theme.TEXT_MUTED,
        )
        self.header_project.pack(side="right", padx=18)

        # Content container (views are packed into here).
        self.content = ctk.CTkFrame(self, fg_color=theme.BG, corner_radius=0)
        self.content.grid(row=1, column=1, sticky="nsew")

    def _make_view(self, key: str) -> ctk.CTkFrame:
        import importlib

        mod_suffix, cls_name = VIEW_REGISTRY[key]
        mod = importlib.import_module(f"gui.views.{mod_suffix}")
        cls = getattr(mod, cls_name)
        return cls(self.content, self)

    # ---- navigation ----
    def show_view(self, key: str):
        if key not in VIEW_REGISTRY:
            return
        if self._current == key:
            return
        if self._current and self._current in self._views:
            self._views[self._current].pack_forget()

        if key not in self._views:
            self._views[key] = self._make_view(key)
        self._views[key].pack(fill="both", expand=True)

        self._current = key
        self.sidebar.set_active(key)
        label = dict((k, lbl) for k, _g, lbl in theme.NAV_ITEMS).get(key, key)
        self.header_title.configure(text=label)

        # Let a view refresh when (re)shown.
        refresh = getattr(self._views[key], "on_show", None)
        if callable(refresh):
            refresh()

    def set_project_dir(self, path: str | Path):
        self.project_dir = Path(path).resolve()
        self.header_project.configure(text=f"\U0001F4C1 {self.project_dir}")


def main(project_dir: Optional[str] = None) -> int:
    app = PtbpGuiApp(project_dir=project_dir)
    app.mainloop()
    return 0
