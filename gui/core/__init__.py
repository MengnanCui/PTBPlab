"""Tk-free core logic for the PTBP GUI.

Every module here is importable and testable without a display and without
customtkinter. The Tk layer (gui.app / gui.views / gui.widgets) is a thin
skin over these.
"""
from __future__ import annotations
