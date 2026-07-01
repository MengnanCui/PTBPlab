"""PTBP desktop GUI (customtkinter).

A modern, VSCode-styled front end over the PTBP toolkit. The heavy compute
paths (real optimisation / SKF generation) shell out to the `ptbp` CLI as a
subprocess, so they only *compute* where the conda-only deps (DFTB+, hotcent,
pylibxc) are installed. Everything else — structure editing / EOS-set
generation (ASE only), dataset inspection, command/YAML assembly, dependency
checks — runs anywhere.

Design: a thin Tk layer (`gui.app`, `gui.views`, `gui.widgets`) over a pure
Python, Tk-free core (`gui.core`) that holds all the logic and is unit-tested
headlessly.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
