"""Headless smoke test for the Tk layer.

Skips cleanly where customtkinter / Tk / a display are unavailable (e.g. this
CI container's Python is built without _tkinter). Where a display exists, it
constructs the app and cycles through every view to catch layout/import errors.
"""
import os

import pytest

pytest.importorskip("customtkinter")

if not os.environ.get("DISPLAY"):
    pytest.skip("no display available for Tk smoke test", allow_module_level=True)


def test_app_builds_and_cycles_views():
    from gui.app import PtbpGuiApp, VIEW_REGISTRY

    app = PtbpGuiApp()
    try:
        for key in VIEW_REGISTRY:
            app.show_view(key)
            app.update_idletasks()
    finally:
        app.destroy()
