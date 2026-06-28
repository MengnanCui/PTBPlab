"""Welcome screen — top-level optimize / gen choice."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Static


class WelcomeScreen(Screen):
    """Lets the user pick between the two wizard flows."""

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("PTBP Setup Wizard", classes="step-title"),
            Static(
                "What do you want to set up?\n\n"
                " • [bold]optimize[/]   — parameter optimisation against a DFT dataset\n"
                "                  (auto-fits E0s, picks mode, runs BO/PSO/parallel_BO)\n\n"
                " • [bold]gen[/]        — generate Slater-Koster files from published or\n"
                "                  custom parameters (no DFTB+ required)\n",
                markup=True,
            ),
            Horizontal(
                Button("optimize", id="btn-optimize", variant="primary"),
                Button("gen", id="btn-gen"),
                Button("Quit", id="btn-quit", variant="default"),
                classes="step-buttons",
            ),
            classes="wizard-step",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-optimize":
            self.app.state.flow = 'optimize'
            from cli.tui.screens.optimize_step1_dataset import OptimizeStep1Dataset
            self.app.push_screen(OptimizeStep1Dataset())
        elif event.button.id == "btn-gen":
            self.app.state.flow = 'gen'
            from cli.tui.screens.gen_step1_symbols import GenStep1Symbols
            self.app.push_screen(GenStep1Symbols())
        elif event.button.id == "btn-quit":
            self.app.exit()
