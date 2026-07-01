"""Monitor: inspect a run folder — best params, convergence curve, artifacts.

Works on any finished (or in-progress) run folder, re-plotting the
best-so-far curve from `result.pkl` when present and otherwise embedding the
saved `convergence.png`.
"""
from __future__ import annotations

import json
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from gui import theme
from gui.widgets.card import Card
from gui.widgets.mpl_canvas import MplCanvas


class MonitorView(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color=theme.BG)
        self.app = app

        outer = ctk.CTkFrame(self, fg_color=theme.BG)
        outer.pack(fill="both", expand=True, padx=14, pady=14)
        outer.grid_columnconfigure(0, weight=2, uniform="c")
        outer.grid_columnconfigure(1, weight=3, uniform="c")
        outer.grid_rowconfigure(1, weight=1)

        # top bar: run dir selector
        bar = ctk.CTkFrame(outer, fg_color="transparent")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ctk.CTkLabel(bar, text="Run folder", font=theme.FONT_BODY,
                     text_color=theme.TEXT).pack(side="left", padx=(4, 8))
        self.run_var = ctk.StringVar()
        ctk.CTkEntry(bar, textvariable=self.run_var, font=theme.FONT_BODY,
                     fg_color=theme.BG, border_color=theme.BORDER).pack(
            side="left", fill="x", expand=True)
        ctk.CTkButton(bar, text="…", width=36, corner_radius=theme.CORNER_SM,
                      fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                      command=self._browse).pack(side="left", padx=(8, 0))
        ctk.CTkButton(bar, text="Refresh", width=90, corner_radius=theme.CORNER_SM,
                      fg_color=theme.PANEL_ALT, hover_color=theme.ACCENT_ACTIVE,
                      command=self.refresh).pack(side="left", padx=(8, 0))

        # left: summary + artifacts
        summ = Card(outer, title="Best result")
        summ.grid(row=1, column=0, sticky="nsew", padx=(0, 7))
        self.summary = ctk.CTkTextbox(summ.body, font=theme.FONT_CODE,
                                      fg_color=theme.BG, text_color=theme.TEXT,
                                      wrap="word")
        self.summary.pack(fill="both", expand=True)
        self.summary.configure(state="disabled")

        art = Card(outer, title="Artifacts")
        art.grid(row=2, column=0, sticky="nsew", padx=(0, 7), pady=(14, 0))
        self.art_body = art.body

        # right: convergence plot
        plot = Card(outer, title="Convergence")
        plot.grid(row=1, column=1, rowspan=2, sticky="nsew", padx=(7, 0))
        self.canvas = MplCanvas(plot.body)
        self.canvas.pack(fill="both", expand=True)

    def on_show(self):
        rd = self.app.shared.get("monitor_run_dir")
        if rd and not self.run_var.get():
            self.run_var.set(str(rd))
        self.refresh()

    def _browse(self):
        d = filedialog.askdirectory(initialdir=str(self.app.project_dir))
        if d:
            self.run_var.set(d)
            self.refresh()

    def refresh(self):
        run_dir = Path(self.run_var.get()) if self.run_var.get().strip() else None
        if not run_dir or not run_dir.is_dir():
            self._set_summary("Select a run folder.")
            self.canvas.show_message("No run selected")
            self._list_artifacts(None)
            return
        self._load_summary(run_dir)
        self._plot(run_dir)
        self._list_artifacts(run_dir)

    def _set_summary(self, text):
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("end", text)
        self.summary.configure(state="disabled")

    def _load_summary(self, run_dir: Path):
        lines = [f"Run: {run_dir.name}"]
        sj = run_dir / "summary.json"
        res = self._load_result(run_dir)
        if sj.exists():
            try:
                d = json.loads(sj.read_text())
                lines += [
                    f"optimizer : {d.get('optimizer')}",
                    f"mode      : {d.get('mode')}",
                    f"best x    : {d.get('x_best')}",
                    f"best loss : {d.get('fun_best')}",
                    f"n evals   : {d.get('n_evaluations')}",
                    f"runtime s : {d.get('runtime_s')}",
                ]
            except Exception as e:
                lines.append(f"(summary.json unreadable: {e})")
        elif res is not None:
            lines += [
                f"optimizer : {getattr(res, 'optimizer', '?')}",
                f"best x    : {getattr(res, 'x', '?')}",
                f"best loss : {getattr(res, 'fun', '?')}",
                f"n evals   : {len(getattr(res, 'func_vals', []) or [])}",
            ]
        else:
            lines.append("(no summary.json or result.pkl yet)")
        self._set_summary("\n".join(lines))

    def _load_result(self, run_dir: Path):
        rp = run_dir / "result.pkl"
        if not rp.exists():
            return None
        try:
            import pickle
            with open(rp, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None

    def _plot(self, run_dir: Path):
        res = self._load_result(run_dir)
        vals = list(getattr(res, "func_vals", []) or []) if res is not None else []
        if vals:
            from matplotlib.figure import Figure
            best = []
            cur = float("inf")
            for v in vals:
                cur = min(cur, float(v))
                best.append(cur)
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_facecolor(theme.PANEL)
            xs = range(1, len(vals) + 1)
            ax.plot(xs, vals, "o-", color=theme.WARN, alpha=0.5, label="per-eval μ")
            ax.plot(xs, best, "-", color=theme.OK, label="best so far")
            ax.set_xlabel("evaluation", color=theme.TEXT)
            ax.set_ylabel("μ (loss)", color=theme.TEXT)
            ax.tick_params(colors=theme.TEXT_MUTED)
            try:
                ax.set_yscale("log")
            except Exception:
                pass
            ax.legend()
            fig.tight_layout()
            self.canvas.show_figure(fig)
            return
        png = run_dir / "convergence.png"
        if png.exists():
            self.canvas.show_image(png)
        else:
            self.canvas.show_message("No result.pkl / convergence.png yet")

    def _list_artifacts(self, run_dir):
        for w in self.art_body.winfo_children():
            w.destroy()
        if run_dir is None:
            return
        from gui.core.runner import ARTIFACT_NAMES
        any_found = False
        for name in ARTIFACT_NAMES:
            p = run_dir / name
            if p.exists():
                any_found = True
                ctk.CTkLabel(self.art_body, text=f"• {name}", font=theme.FONT_SMALL,
                             text_color=theme.TEXT, anchor="w").pack(fill="x")
        if not any_found:
            ctk.CTkLabel(self.art_body, text="(no artifacts yet)",
                         font=theme.FONT_SMALL, text_color=theme.TEXT_MUTED,
                         anchor="w").pack(fill="x")
