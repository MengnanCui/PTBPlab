"""Structures: load a database or single structure, inspect, preview in 2D,
and build EOS training sets / perturbed structures (ASE-only, runs anywhere).
"""
from __future__ import annotations

from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from gui import theme
from gui.widgets.card import Card
from gui.widgets.mpl_canvas import MplCanvas


class StructuresView(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color=theme.BG)
        self.app = app
        self.atoms_list = []          # loaded structures
        self.index = 0

        outer = ctk.CTkFrame(self, fg_color=theme.BG)
        outer.pack(fill="both", expand=True, padx=14, pady=14)
        outer.grid_columnconfigure(0, weight=1, uniform="col")
        outer.grid_columnconfigure(1, weight=1, uniform="col")
        outer.grid_rowconfigure(1, weight=1)

        # --- Load & inspect (top-left) ---
        load = Card(outer, title="Load & inspect")
        load.grid(row=0, column=0, sticky="nsew", padx=(0, 7), pady=(0, 14))
        row = ctk.CTkFrame(load.body, fg_color="transparent")
        row.pack(fill="x")
        self.path_var = ctk.StringVar()
        ctk.CTkEntry(row, textvariable=self.path_var, font=theme.FONT_BODY,
                     placeholder_text="dataset or structure (xyz/traj/extxyz/json)",
                     fg_color=theme.BG, border_color=theme.BORDER).pack(
            side="left", fill="x", expand=True)
        ctk.CTkButton(row, text="Open", width=70, corner_radius=theme.CORNER_SM,
                      fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                      command=self._browse).pack(side="left", padx=(8, 0))
        ctk.CTkButton(row, text="Inspect", width=80, corner_radius=theme.CORNER_SM,
                      fg_color=theme.PANEL_ALT, hover_color=theme.ACCENT_ACTIVE,
                      command=self._inspect).pack(side="left", padx=(8, 0))
        self.report = ctk.CTkTextbox(load.body, height=150, font=theme.FONT_CODE,
                                     fg_color=theme.BG, text_color=theme.TEXT,
                                     wrap="word")
        self.report.pack(fill="both", expand=True, pady=(10, 0))
        self.report.configure(state="disabled")

        # --- 2D preview (top-right) ---
        prev = Card(outer, title="Structure preview (2D)")
        prev.grid(row=0, column=1, sticky="nsew", padx=(7, 0), pady=(0, 14))
        nav = ctk.CTkFrame(prev.body, fg_color="transparent")
        nav.pack(fill="x")
        ctk.CTkButton(nav, text="◀", width=40, corner_radius=theme.CORNER_SM,
                      fg_color=theme.PANEL_ALT, hover_color=theme.ACCENT_ACTIVE,
                      command=lambda: self._step(-1)).pack(side="left")
        self.idx_label = ctk.CTkLabel(nav, text="—", font=theme.FONT_BODY,
                                      text_color=theme.TEXT)
        self.idx_label.pack(side="left", expand=True)
        ctk.CTkButton(nav, text="▶", width=40, corner_radius=theme.CORNER_SM,
                      fg_color=theme.PANEL_ALT, hover_color=theme.ACCENT_ACTIVE,
                      command=lambda: self._step(1)).pack(side="left")
        self.canvas = MplCanvas(prev.body)
        self.canvas.pack(fill="both", expand=True, pady=(8, 0))

        # --- EOS builder (bottom-left) ---
        eos = Card(outer, title="EOS training-set builder")
        eos.grid(row=1, column=0, sticky="nsew", padx=(0, 7))
        self._build_eos_controls(eos.body)

        # --- Perturb (bottom-right) ---
        pert = Card(outer, title="Perturb positions")
        pert.grid(row=1, column=1, sticky="nsew", padx=(7, 0))
        self._build_perturb_controls(pert.body)

    # ---- EOS controls ----
    def _build_eos_controls(self, body):
        from gui.widgets.form import FormBuilder

        grid = ctk.CTkFrame(body, fg_color="transparent")
        grid.pack(fill="x")
        fb = FormBuilder(grid)
        self.vmin = fb.add_entry("vmin", "Min volume %", "90")
        self.vmax = fb.add_entry("vmax", "Max volume %", "110")
        self.npts = fb.add_entry("npts", "Points", "11")
        self.eos_out = fb.add_entry("out", "Output file", "eos_dataset.xyz")

        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(btns, text="Generate + Save EOS series",
                      corner_radius=theme.CORNER_SM, fg_color=theme.ACCENT,
                      hover_color=theme.ACCENT_HOVER,
                      command=self._make_eos).pack(side="left")
        self.eos_status = ctk.CTkLabel(body, text="", font=theme.FONT_SMALL,
                                       text_color=theme.TEXT_MUTED, anchor="w",
                                       wraplength=380, justify="left")
        self.eos_status.pack(fill="x", pady=(8, 0))

    # ---- Perturb controls ----
    def _build_perturb_controls(self, body):
        from gui.widgets.form import FormBuilder

        grid = ctk.CTkFrame(body, fg_color="transparent")
        grid.pack(fill="x")
        fb = FormBuilder(grid)
        self.sigma = fb.add_entry("sigma", "Sigma (Å)", "0.05")
        self.pseed = fb.add_entry("seed", "Seed", "1")
        self.ncopies = fb.add_entry("ncopies", "Copies", "5")
        self.pert_out = fb.add_entry("out", "Output file", "perturbed.xyz")

        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(btns, text="Generate + Save perturbations",
                      corner_radius=theme.CORNER_SM, fg_color=theme.ACCENT,
                      hover_color=theme.ACCENT_HOVER,
                      command=self._make_perturb).pack(side="left")
        self.pert_status = ctk.CTkLabel(body, text="", font=theme.FONT_SMALL,
                                        text_color=theme.TEXT_MUTED, anchor="w",
                                        wraplength=380, justify="left")
        self.pert_status.pack(fill="x", pady=(8, 0))

    # ---- actions ----
    def _browse(self):
        f = filedialog.askopenfilename(
            initialdir=str(self.app.project_dir),
            filetypes=[("Structures", "*.xyz *.traj *.extxyz *.json"),
                       ("All files", "*.*")])
        if f:
            self.path_var.set(f)
            self._load(f)

    def _load(self, path):
        import ase.io
        try:
            self.atoms_list = ase.io.read(path, ":")
        except Exception as e:
            self._set_report(f"Read failed: {type(e).__name__}: {e}")
            self.atoms_list = []
            return
        self.index = 0
        self._render_current()

    def _inspect(self):
        path = self.path_var.get().strip()
        if not path:
            return
        if not self.atoms_list:
            self._load(path)
        from cli.dataset_inspect import inspect
        from gui.core.structure_tools import summarize
        rep = inspect(path)
        text = rep.render()
        if self.atoms_list:
            s = summarize(self.atoms_list)
            text += f"\n\n{s['n_structures']} loaded, {s['n_periodic']} periodic."
        self._set_report(text)

    def _set_report(self, text):
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("end", text)
        self.report.configure(state="disabled")

    def _step(self, d):
        if not self.atoms_list:
            return
        self.index = (self.index + d) % len(self.atoms_list)
        self._render_current()

    def _render_current(self):
        if not self.atoms_list:
            self.idx_label.configure(text="—")
            self.canvas.show_message("No structure loaded")
            return
        atoms = self.atoms_list[self.index]
        name = atoms.info.get("structure_name", atoms.get_chemical_formula())
        self.idx_label.configure(
            text=f"{self.index + 1}/{len(self.atoms_list)}  ·  {name}")
        try:
            from ase.visualize.plot import plot_atoms
            from matplotlib.figure import Figure
            fig = Figure(figsize=(4, 3.4), dpi=100)
            ax = fig.add_subplot(111)
            plot_atoms(atoms, ax, rotation="10x,-10y,0z")
            ax.set_axis_off()
            fig.tight_layout(pad=0.2)
            self.canvas.show_figure(fig)
        except Exception as e:
            self.canvas.show_message(f"preview error: {e}")

    def _base_atoms(self):
        if not self.atoms_list:
            self.eos_status.configure(text="Load a structure first.",
                                      text_color=theme.WARN)
            self.pert_status.configure(text="Load a structure first.",
                                       text_color=theme.WARN)
            return None
        return self.atoms_list[self.index]

    def _resolve_out(self, name):
        p = Path(name)
        return p if p.is_absolute() else (self.app.project_dir / p)

    def _make_eos(self):
        base = self._base_atoms()
        if base is None:
            return
        from gui.core import structure_tools as st
        try:
            series = st.make_eos_series(
                base, float(self.vmin.get()), float(self.vmax.get()),
                int(self.npts.get()))
            out = st.combine_and_write([series], self._resolve_out(self.eos_out.get()))
        except Exception as e:
            self.eos_status.configure(text=f"Error: {e}", text_color=theme.ERR)
            return
        self.eos_status.configure(
            text=f"✓ Wrote {len(series)} structures → {out}",
            text_color=theme.OK)

    def _make_perturb(self):
        base = self._base_atoms()
        if base is None:
            return
        from gui.core import structure_tools as st
        try:
            n = int(self.ncopies.get())
            seed0 = int(self.pseed.get())
            sigma = float(self.sigma.get())
            copies = [st.perturb_positions(base, sigma, seed=seed0 + i)
                      for i in range(n)]
            out = st.combine_and_write([copies], self._resolve_out(self.pert_out.get()))
        except Exception as e:
            self.pert_status.configure(text=f"Error: {e}", text_color=theme.ERR)
            return
        self.pert_status.configure(
            text=f"✓ Wrote {n} perturbed copies → {out}", text_color=theme.OK)
