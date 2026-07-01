"""Optimize parameters (`ptbp optimize`).

Left: the full configuration form bound to a SetupState. Right: live YAML +
command preview and a Run button that writes the YAML, launches the job via
JobRunner, and hands the run folder to the Monitor view.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from gui import theme
from gui.core import gui_state as gs
from gui.widgets.card import Card
from gui.widgets.form import FormBuilder
from gui.widgets.log_panel import LogPanel


class OptimizeView(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color=theme.BG)
        self.app = app
        self.state = gs.new_optimize_state()
        self._runner = None

        outer = ctk.CTkFrame(self, fg_color=theme.BG)
        outer.pack(fill="both", expand=True, padx=14, pady=14)
        outer.grid_columnconfigure(0, weight=3, uniform="c")
        outer.grid_columnconfigure(1, weight=4, uniform="c")
        outer.grid_rowconfigure(0, weight=1)

        # ---- left: form (scrollable) ----
        form_card = Card(outer, title="Configuration")
        form_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        scroll = ctk.CTkScrollableFrame(form_card.body, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # dataset row (entry + browse) above the form grid
        drow = ctk.CTkFrame(scroll, fg_color="transparent")
        drow.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(drow, text="Dataset", font=theme.FONT_BODY,
                     text_color=theme.TEXT).pack(side="left", padx=(4, 8))
        self.dataset = ctk.StringVar()
        ctk.CTkEntry(drow, textvariable=self.dataset, font=theme.FONT_BODY,
                     fg_color=theme.BG, border_color=theme.BORDER).pack(
            side="left", fill="x", expand=True)
        ctk.CTkButton(drow, text="…", width=36, corner_radius=theme.CORNER_SM,
                      fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                      command=self._browse).pack(side="left", padx=(8, 0))

        self.detect_label = ctk.CTkLabel(
            scroll, text="", font=theme.FONT_SMALL, text_color=theme.TEXT_MUTED,
            anchor="w", wraplength=360, justify="left")
        self.detect_label.pack(fill="x", padx=4, pady=(0, 6))

        grid = ctk.CTkFrame(scroll, fg_color="transparent")
        grid.pack(fill="x")
        fb = FormBuilder(grid)
        self.target = fb.add_combo("target", "Target (loss)",
                                   list(gs.TARGET_TO_MODE.keys()), "Auto-detect")
        self.optimizer = fb.add_combo("optimizer", "Optimizer",
                                      list(gs.OPTIMIZER_TO_VALUE.keys()),
                                      "Bayesian — GP (bo)")
        self.n_calls = fb.add_entry("n_calls", "n_calls (blank=default)", "")
        self.n_particles = fb.add_entry("n_particles", "n_particles (pso/parallel)", "4")
        self.superposition = fb.add_combo("superposition", "Superposition",
                                          gs.SUPERPOSITIONS, "density")

        # parameter switches
        self._label_row(scroll, "Parameters to optimise")
        prow = ctk.CTkFrame(scroll, fg_color="transparent")
        prow.pack(fill="x", padx=4)
        self.param_vars = {}
        for name in gs.OPTIMIZABLE_PARAMS:
            v = ctk.BooleanVar(value=True)
            ctk.CTkSwitch(prow, text=name, variable=v, progress_color=theme.ACCENT,
                          font=theme.FONT_SMALL).pack(side="left", padx=(0, 14))
            self.param_vars[name] = v

        grid2 = ctk.CTkFrame(scroll, fg_color="transparent")
        grid2.pack(fill="x", pady=(8, 0))
        fb2 = FormBuilder(grid2)
        self.e0s_strategy = fb2.add_combo("e0s", "E0s strategy", gs.E0S_STRATEGIES, "auto")
        self.e0s_json = fb2.add_entry("e0s_json", "E0s JSON (manual)",
                                      "", placeholder='{"29": -45244.9}')
        self.xc = fb2.add_entry("xc", "XC functional", gs.DEFAULT_XC)
        self.kpt = fb2.add_entry("kpt", "k-point density", "5.0")
        self.eos_points = fb2.add_entry("eos_points", "EOS points", "11")
        self.multi = fb2.add_entry("multi", "multi_element (space-sep)", "")
        self.seed = fb2.add_entry("seed", "seed (blank=123)", "")
        self.output = fb2.add_entry("output", "Output dir (blank=auto)", "")

        # ---- right: preview + run ----
        right = Card(outer, title="Review & run")
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        btns = ctk.CTkFrame(right.body, fg_color="transparent")
        btns.pack(fill="x")
        ctk.CTkButton(btns, text="Preview", corner_radius=theme.CORNER_SM,
                      fg_color=theme.PANEL_ALT, hover_color=theme.ACCENT_ACTIVE,
                      command=self._preview).pack(side="left")
        self.run_btn = ctk.CTkButton(btns, text="▶ Run", corner_radius=theme.CORNER_SM,
                                     fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                                     command=self._run)
        self.run_btn.pack(side="left", padx=(10, 0))
        self.cancel_btn = ctk.CTkButton(btns, text="Cancel", corner_radius=theme.CORNER_SM,
                                        fg_color=theme.PANEL_ALT, hover_color=theme.ERR,
                                        command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=(10, 0))

        ctk.CTkLabel(right.body, text="ptbp_run.yaml", font=theme.FONT_SMALL,
                     text_color=theme.TEXT_MUTED, anchor="w").pack(fill="x", pady=(10, 2))
        self.yaml_box = ctk.CTkTextbox(right.body, height=150, font=theme.FONT_CODE,
                                       fg_color=theme.BG, text_color=theme.TEXT, wrap="none")
        self.yaml_box.pack(fill="x")
        self.cmd_label = ctk.CTkLabel(right.body, text="", font=theme.FONT_CODE,
                                      text_color=theme.OK, anchor="w",
                                      wraplength=520, justify="left")
        self.cmd_label.pack(fill="x", pady=(6, 6))
        self.log = LogPanel(right.body)
        self.log.pack(fill="both", expand=True)

        self.dataset.trace_add("write", lambda *_: self._on_dataset_change())

    def _label_row(self, master, text):
        ctk.CTkLabel(master, text=text, font=theme.FONT_BODY, text_color=theme.TEXT,
                     anchor="w").pack(fill="x", padx=4, pady=(8, 2))

    # ---- dataset detection ----
    def _browse(self):
        f = filedialog.askopenfilename(
            initialdir=str(self.app.project_dir),
            filetypes=[("Datasets", "*.xyz *.traj *.extxyz *.json"), ("All", "*.*")])
        if f:
            self.dataset.set(f)

    def _on_dataset_change(self):
        path = self.dataset.get().strip()
        if not path or not Path(path).exists():
            self.detect_label.configure(text="")
            return
        try:
            from cli.auto_mode import detect_mode
            mode, why = detect_mode(path)
            self.detect_label.configure(
                text=f"auto-detected target: {gs.MODE_TO_TARGET.get(mode, mode)} — {why}")
        except Exception as e:
            self.detect_label.configure(text=f"detect failed: {e}")

    # ---- state sync ----
    def _sync_state(self):
        s = self.state
        s.dataset_path = Path(self.dataset.get()).resolve() if self.dataset.get().strip() else None
        s.mode = gs.TARGET_TO_MODE[self.target.get()]
        s.optimizer = gs.OPTIMIZER_TO_VALUE[self.optimizer.get()]
        s.n_calls = int(self.n_calls.get()) if self.n_calls.get().strip() else None
        s.n_particles = int(self.n_particles.get()) if self.n_particles.get().strip() else 4
        s.superposition = self.superposition.get()
        s.parameters = [n for n, v in self.param_vars.items() if v.get()]
        s.e0s_strategy = self.e0s_strategy.get()
        s.e0s_manual_json = self.e0s_json.get().strip()
        s.xc = self.xc.get().strip() or gs.DEFAULT_XC
        s.kpt_density = float(self.kpt.get()) if self.kpt.get().strip() else 5.0
        s.eos_points = int(self.eos_points.get()) if self.eos_points.get().strip() else 11
        s.multi_element = self.multi.get().split()
        s.seed = int(self.seed.get()) if self.seed.get().strip() else None

    def _preview(self):
        try:
            self._sync_state()
        except ValueError as e:
            self.cmd_label.configure(text=f"# invalid numeric input: {e}",
                                     text_color=theme.ERR)
            return False
        yaml_text, cmd = gs.optimize_preview(self.state)
        self.yaml_box.delete("1.0", "end")
        self.yaml_box.insert("end", yaml_text)
        self.cmd_label.configure(text=cmd, text_color=theme.OK)
        problems = gs.validate_optimize(self.state)
        if problems:
            self.log.clear()
            self.log.append("Cannot run yet:")
            for p in problems:
                self.log.append(f"  - {p}")
            return False
        return True

    def _run(self):
        if not self._preview():
            return
        from cli.tui.state import assemble_optimize_argv
        from gui.core.runner import JobRunner, make_log_path

        # Resolve an explicit run folder so Monitor can find the artifacts.
        if self.output.get().strip():
            run_dir = Path(self.output.get()).resolve()
        else:
            ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
            run_dir = (self.app.project_dir / f"run_{ts}").resolve()

        # Persist the YAML the command references, then build argv + --output.
        yaml_path = self.app.project_dir / "ptbp_run.yaml"
        gs.write_optimize_yaml(self.state, yaml_path)
        self.state.output_yaml = yaml_path
        argv = assemble_optimize_argv(self.state) + ["--output", str(run_dir)]

        log_path = make_log_path(self.app.project_dir / "gui_logs")
        self.app.shared["monitor_run_dir"] = str(run_dir)

        self.log.clear()
        self.log.append(f"$ {' '.join(argv)}\n")
        self.run_btn.configure(state="disabled", text="running…")
        self.cancel_btn.configure(state="normal")
        self._runner = JobRunner(argv, log_path, cwd=self.app.project_dir)

        def on_exit(code):
            def done():
                self.run_btn.configure(state="normal", text="▶ Run")
                self.cancel_btn.configure(state="disabled")
            self.after(0, done)
            self.log.append(f"\n[exit {code}] log: {log_path}")

        self._runner.start(on_line=self.log.append, on_exit=on_exit)

    def _cancel(self):
        if self._runner is not None:
            self._runner.cancel()
            self.log.append("\n[cancelled by user]")
