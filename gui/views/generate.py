"""Generate SKF files (`ptbp gen`). Preview the command; optionally run it."""
from __future__ import annotations

from pathlib import Path

import customtkinter as ctk

from gui import theme
from gui.core import gui_state as gs
from gui.widgets.card import Card
from gui.widgets.form import FormBuilder
from gui.widgets.log_panel import LogPanel


class GenerateView(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color=theme.BG)
        self.app = app
        self.state = gs.new_gen_state()
        self._runner = None

        outer = ctk.CTkFrame(self, fg_color=theme.BG)
        outer.pack(fill="both", expand=True, padx=14, pady=14)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_columnconfigure(1, weight=1)
        outer.grid_rowconfigure(0, weight=1)

        # --- form ---
        form_card = Card(outer, title="Parameters")
        form_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        grid = ctk.CTkFrame(form_card.body, fg_color="transparent")
        grid.pack(fill="x")
        fb = FormBuilder(grid)
        self.symbols = fb.add_entry("symbols", "Element symbols", "H O",
                                    placeholder="e.g. H O Cu")
        self.params = fb.add_combo("params", "Parameter set", gs.PARAM_SETS, "PTBP",
                                   command=lambda _v: self._toggle_personal())
        self.mode = fb.add_combo("mode", "SKF part", gs.SKF_MODES, "full")
        self.xc = fb.add_entry("xc", "XC functional", gs.DEFAULT_XC)
        self.out = fb.add_entry("out", "Output dir", "")
        self.r0 = fb.add_entry("r0", "Personal r0 (r0_w r0_d p …)", "")
        self.rep = fb.add_entry("rep", "Personal rep (sigma kxc …)", "")

        btns = ctk.CTkFrame(form_card.body, fg_color="transparent")
        btns.pack(fill="x", pady=(12, 0))
        ctk.CTkButton(btns, text="Preview", corner_radius=theme.CORNER_SM,
                      fg_color=theme.PANEL_ALT, hover_color=theme.ACCENT_ACTIVE,
                      command=self._preview).pack(side="left")
        self.run_btn = ctk.CTkButton(btns, text="▶ Run", corner_radius=theme.CORNER_SM,
                                     fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                                     command=self._run)
        self.run_btn.pack(side="left", padx=(10, 0))
        self.warn = ctk.CTkLabel(form_card.body, text="", font=theme.FONT_SMALL,
                                 text_color=theme.WARN, anchor="w", wraplength=420,
                                 justify="left")
        self.warn.pack(fill="x", pady=(8, 0))

        # --- preview / output ---
        right = Card(outer, title="Command & output")
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        self.cmd_box = ctk.CTkTextbox(right.body, height=64, font=theme.FONT_CODE,
                                      fg_color=theme.BG, text_color=theme.OK,
                                      wrap="word")
        self.cmd_box.pack(fill="x")
        self.log = LogPanel(right.body)
        self.log.pack(fill="both", expand=True, pady=(10, 0))

        self._toggle_personal()

    def on_show(self):
        r = self._readiness()
        if not r.skf_ready:
            self.warn.configure(
                text="hotcent / pylibxc not detected — Preview works, but Run will "
                     "only compute where they're installed (see Calculator).")
        else:
            self.warn.configure(text="")

    def _readiness(self):
        from gui.core import env_doctor
        return env_doctor.check()

    def _toggle_personal(self):
        # Personal r0/rep only matter for --params Personal; hint the user but
        # keep the fields always-visible (validation happens on preview/run).
        hint = "required for Personal" if self.params.get() == "Personal" else "Personal only"
        self.r0.set(self.r0.get())  # no-op; kept for symmetry
        self._personal_hint = hint

    def _sync_state(self):
        self.state.gen_symbols = self.symbols.get().split()
        self.state.gen_params = self.params.get()
        self.state.gen_skf_mode = self.mode.get()
        self.state.gen_xc = self.xc.get().strip() or gs.DEFAULT_XC
        self.state.gen_out = Path(self.out.get()).resolve() if self.out.get().strip() else None
        self.state.gen_r0 = [float(x) for x in self.r0.get().split()] if self.r0.get().strip() else []
        self.state.gen_rep = [float(x) for x in self.rep.get().split()] if self.rep.get().strip() else []

    def _preview(self) -> str | None:
        try:
            self._sync_state()
        except ValueError as e:
            self._set_cmd(f"# invalid numeric input: {e}")
            return None
        problems = gs.validate_gen(self.state)
        cmd = gs.gen_preview(self.state)
        self._set_cmd(cmd)
        if problems:
            self.log.clear()
            self.log.append("Cannot run yet:")
            for p in problems:
                self.log.append(f"  - {p}")
            return None
        return cmd

    def _set_cmd(self, text):
        self.cmd_box.delete("1.0", "end")
        self.cmd_box.insert("end", text)

    def _run(self):
        cmd = self._preview()
        if cmd is None:
            return
        from cli.tui.state import assemble_gen_argv
        from gui.core.runner import JobRunner, make_log_path

        argv = assemble_gen_argv(self.state)
        log_path = make_log_path(self.app.project_dir / "gui_logs")
        self.log.clear()
        self.log.append(f"$ {cmd}\n")
        self.run_btn.configure(state="disabled", text="running…")
        self._runner = JobRunner(argv, log_path, cwd=self.app.project_dir)

        def on_exit(code):
            self.after(0, lambda: self.run_btn.configure(state="normal", text="▶ Run"))
            self.log.append(f"\n[exit {code}] log: {log_path}")

        self._runner.start(on_line=self.log.append, on_exit=on_exit)
