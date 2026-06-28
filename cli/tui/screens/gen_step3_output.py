"""Gen wizard step 3 — output dir + xc + (Personal) custom values."""
from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, Static


class GenStep3Output(Screen):
    def compose(self) -> ComposeResult:
        children = [
            Static("Step 3 of 4  •  Output + xc", classes="step-title"),
            Static("Output directory for the .skf files (default: cwd):"),
            Input(placeholder="./skfs", id="out"),
            Static("Exchange-correlation functional:"),
            Input(value="GGA_X_PBE+GGA_C_PBE", id="xc"),
        ]
        # Personal-only fields
        if self.app.state.gen_params == 'Personal':
            n = len(self.app.state.gen_symbols)
            children += [
                Static(f"\nPersonal r0 values: {3*n} floats "
                       "(r0_w r0_d p per element):"),
                Input(placeholder="2.0 5.0 2.0  3.0 5.0 2.0", id="r0"),
                Static(f"Personal rep values: {2*n} floats "
                       "(sigma_rep kxc per element):"),
                Input(placeholder="0.6 0.0  0.6 0.0", id="rep"),
            ]
        children.append(Horizontal(
            Button("Back", id="btn-back"),
            Button("Next", id="btn-next", variant="primary"),
            classes="step-buttons",
        ))
        yield Vertical(*children, classes="wizard-step")

    def on_mount(self) -> None:
        s = self.app.state
        if s.gen_out:
            self.query_one('#out', Input).value = str(s.gen_out)
        self.query_one('#xc', Input).value = s.gen_xc
        if s.gen_params == 'Personal':
            if s.gen_r0:
                self.query_one('#r0', Input).value = ' '.join(map(str, s.gen_r0))
            if s.gen_rep:
                self.query_one('#rep', Input).value = ' '.join(map(str, s.gen_rep))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            s = self.app.state
            out = self.query_one('#out', Input).value.strip()
            s.gen_out = Path(out).expanduser() if out else None
            s.gen_xc = self.query_one('#xc', Input).value.strip() or 'GGA_X_PBE+GGA_C_PBE'
            if s.gen_params == 'Personal':
                try:
                    s.gen_r0 = list(map(float,
                                        self.query_one('#r0', Input).value.split()))
                except ValueError:
                    s.gen_r0 = []
                try:
                    s.gen_rep = list(map(float,
                                         self.query_one('#rep', Input).value.split()))
                except ValueError:
                    s.gen_rep = []

            from cli.tui.screens.gen_step4_review import GenStep4Review
            self.app.push_screen(GenStep4Review())
