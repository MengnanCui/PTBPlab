"""Calculator settings: dependency doctor + DFT-code compatibility knobs."""
from __future__ import annotations

import os

import customtkinter as ctk

from gui import theme
from gui.core import gui_state as gs
from gui.widgets.card import Card


class CalculatorView(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color=theme.BG)
        self.app = app

        wrap = ctk.CTkScrollableFrame(self, fg_color=theme.BG)
        wrap.pack(fill="both", expand=True, padx=16, pady=16)

        # DFTB+ / env card
        env = Card(wrap, title="DFTB+ binary")
        env.pack(fill="x", pady=(0, 14))
        row = ctk.CTkFrame(env.body, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkLabel(row, text="ASE_DFTB_COMMAND", font=theme.FONT_BODY,
                     text_color=theme.TEXT).pack(side="left", padx=(0, 10))
        self.cmd_var = ctk.StringVar(value=os.environ.get("ASE_DFTB_COMMAND", ""))
        ctk.CTkEntry(row, textvariable=self.cmd_var, font=theme.FONT_BODY,
                     placeholder_text="/path/to/dftb+", fg_color=theme.BG,
                     border_color=theme.BORDER).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row, text="Apply", width=80, corner_radius=theme.CORNER_SM,
                      fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                      command=self._apply_cmd).pack(side="left", padx=(8, 0))
        ctk.CTkLabel(env.body,
                     text="Set for this GUI session (child ptbp processes inherit it).",
                     font=theme.FONT_SMALL, text_color=theme.TEXT_MUTED,
                     anchor="w").pack(fill="x", pady=(6, 0))

        # dependency doctor
        doc = Card(wrap, title="Dependency doctor")
        doc.pack(fill="x", pady=(0, 14))
        top = ctk.CTkFrame(doc.body, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkButton(top, text="Re-check", width=100, corner_radius=theme.CORNER_SM,
                      fg_color=theme.PANEL_ALT, hover_color=theme.ACCENT_ACTIVE,
                      command=self.refresh).pack(side="left")
        self.ready_label = ctk.CTkLabel(top, text="", font=theme.FONT_SMALL,
                                        text_color=theme.TEXT_MUTED, anchor="w")
        self.ready_label.pack(side="left", padx=12)
        self.doc_body = ctk.CTkFrame(doc.body, fg_color="transparent")
        self.doc_body.pack(fill="x", pady=(8, 0))

        # defaults
        dfl = Card(wrap, title="Calculation defaults")
        dfl.pack(fill="x")
        info = (f"XC functional     : {gs.DEFAULT_XC}\n"
                f"k-point density   : 5.0\n"
                f"Superposition     : density (PTBP/Prior) / potential (QNplusRep)\n"
                f"SCC / Fermi / MP   : configured in utils/calculator.py")
        ctk.CTkLabel(dfl.body, text=info, font=theme.FONT_CODE,
                     text_color=theme.TEXT, anchor="w", justify="left").pack(fill="x")

    def on_show(self):
        self.refresh()

    def _apply_cmd(self):
        val = self.cmd_var.get().strip()
        if val:
            os.environ["ASE_DFTB_COMMAND"] = val
        else:
            os.environ.pop("ASE_DFTB_COMMAND", None)
        self.refresh()

    def refresh(self):
        from gui.core import env_doctor

        for w in self.doc_body.winfo_children():
            w.destroy()
        r = env_doctor.check()
        for p in r.probes:
            row = ctk.CTkFrame(self.doc_body, fg_color="transparent")
            row.pack(fill="x", pady=1)
            glyph = "✓" if p.ok else "✗"
            color = theme.OK if p.ok else theme.ERR
            ctk.CTkLabel(row, text=glyph, font=theme.FONT_BODY, text_color=color,
                         width=20).pack(side="left")
            ctk.CTkLabel(row, text=p.name, font=theme.FONT_BODY, text_color=theme.TEXT,
                         width=140, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=f"{p.detail}   ({p.needed_for})",
                         font=theme.FONT_SMALL, text_color=theme.TEXT_MUTED,
                         anchor="w").pack(side="left", fill="x", expand=True)
        self.ready_label.configure(
            text=f"structures: {'ok' if r.structures_ready else '—'}  ·  "
                 f"SKF: {'ok' if r.skf_ready else '—'}  ·  "
                 f"optimize(bo): {'ok' if r.optimize_bo_ready else '—'}  ·  "
                 f"optimize(pso): {'ok' if r.optimize_pso_ready else '—'}")
