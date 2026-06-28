"""Gen wizard step 1 — element symbols."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, Static

from ase.data import atomic_numbers


class GenStep1Symbols(Screen):
    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Step 1 of 4  •  Element symbols", classes="step-title"),
            Static("Space-separated chemical symbols (e.g. `H O Cu`)."),
            Input(placeholder="H O Cu", id="syms"),
            Static("", id="syms-feedback"),
            Horizontal(
                Button("Back", id="btn-back"),
                Button("Next", id="btn-next", variant="primary"),
                classes="step-buttons",
            ),
            classes="wizard-step",
        )

    def on_mount(self) -> None:
        if self.app.state.gen_symbols:
            self.query_one('#syms', Input).value = ' '.join(self.app.state.gen_symbols)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            raw = self.query_one('#syms', Input).value.strip()
            if not raw:
                self.query_one('#syms-feedback', Static).update(
                    "[red]At least one symbol required.[/]")
                return
            syms = raw.split()
            unknown = [s for s in syms if s not in atomic_numbers]
            if unknown:
                self.query_one('#syms-feedback', Static).update(
                    f"[red]Unknown symbol(s): {', '.join(unknown)}[/]")
                return
            self.app.state.gen_symbols = syms
            from cli.tui.screens.gen_step2_params import GenStep2Params
            self.app.push_screen(GenStep2Params())
