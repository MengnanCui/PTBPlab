"""Top-level Textual app for `ptbp setup`.

The App holds a single SetupState instance shared across screens. Each
screen registers a Next/Back binding; pushing the next screen advances
the wizard, popping returns to the previous one.

We keep the App skeleton minimal — the actual screens live under
`cli/tui/screens/` and are imported lazily so a missing optional
dependency on one screen doesn't break the whole `ptbp setup` startup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual.app import App
from textual.binding import Binding

from cli.tui.state import SetupState


class PtbpSetupApp(App):
    """One-shot setup wizard. Exits when the user reaches the final
    screen and clicks Save (or Cancel anywhere via Ctrl-C)."""

    CSS = """
    Screen {
        align: center middle;
    }
    .wizard-step {
        padding: 1 2;
        border: solid $accent;
        width: 80;
        height: auto;
        max-height: 30;
    }
    .step-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    .step-buttons {
        align-horizontal: right;
        height: auto;
    }
    Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding('ctrl+c', 'quit', 'Cancel', show=True),
    ]

    def __init__(self, initial_flow: Optional[str] = None,
                 output_yaml: Optional[Path] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.state = SetupState()
        if output_yaml is not None:
            self.state.output_yaml = output_yaml
        self._initial_flow = initial_flow

    def on_mount(self) -> None:
        """Push the first screen depending on whether the user chose a
        flow on the CLI (`ptbp setup optimize` / `ptbp setup gen`) or
        wants the welcome menu."""
        if self._initial_flow == 'optimize':
            from cli.tui.screens.optimize_step1_dataset import OptimizeStep1Dataset
            self.state.flow = 'optimize'
            self.push_screen(OptimizeStep1Dataset())
        elif self._initial_flow == 'gen':
            from cli.tui.screens.gen_step1_symbols import GenStep1Symbols
            self.state.flow = 'gen'
            self.push_screen(GenStep1Symbols())
        else:
            from cli.tui.screens.welcome import WelcomeScreen
            self.push_screen(WelcomeScreen())
