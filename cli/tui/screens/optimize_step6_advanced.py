"""Optimize step 6 — advanced knobs (xc, kpts, eos_points, etc.)."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, RadioSet, RadioButton, Static


class OptimizeStep6Advanced(Screen):
    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Step 6 of 7  •  Advanced", classes="step-title"),
            Static("Most users can leave these defaults."),

            Static("xc functional:"),
            Input(value="GGA_X_PBE+GGA_C_PBE", id="xc"),

            Static("kpt_density:"),
            Input(value="5.0", id="kpts"),

            Static("eos_points (ignored unless mode = energygeometry):"),
            Input(value="11", id="eos-points"),

            Static("parameters to optimise (space-separated):"),
            Input(value="r0_w r0_d sigma_rep", id="params"),

            Static("multi_element (space-separated; leave empty for global):"),
            Input(placeholder="Cu", id="multi"),

            Static("superposition:"),
            RadioSet(
                RadioButton("density", id="sp-density"),
                RadioButton("potential", id="sp-potential"),
                id="sp-set",
            ),
            Horizontal(
                Button("Back", id="btn-back"),
                Button("Next", id="btn-next", variant="primary"),
                classes="step-buttons",
            ),
            classes="wizard-step",
        )

    def on_mount(self) -> None:
        s = self.app.state
        self.query_one('#xc', Input).value = s.xc
        self.query_one('#kpts', Input).value = str(s.kpt_density)
        self.query_one('#eos-points', Input).value = str(s.eos_points)
        self.query_one('#params', Input).value = ' '.join(s.parameters)
        self.query_one('#multi', Input).value = ' '.join(s.multi_element)
        rs = self.query_one('#sp-set', RadioSet)
        rs._selected = 0 if s.superposition == 'density' else 1

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            s = self.app.state
            s.xc = self.query_one('#xc', Input).value.strip() or 'GGA_X_PBE+GGA_C_PBE'
            try:
                s.kpt_density = float(self.query_one('#kpts', Input).value.strip() or 5.0)
            except ValueError:
                s.kpt_density = 5.0
            try:
                s.eos_points = int(self.query_one('#eos-points', Input).value.strip() or 11)
            except ValueError:
                s.eos_points = 11
            params_raw = self.query_one('#params', Input).value.strip()
            s.parameters = params_raw.split() if params_raw else ['r0_w', 'r0_d', 'sigma_rep']
            multi_raw = self.query_one('#multi', Input).value.strip()
            s.multi_element = multi_raw.split() if multi_raw else []
            rs = self.query_one('#sp-set', RadioSet)
            s.superposition = 'density' if (rs.pressed_index <= 0) else 'potential'

            from cli.tui.screens.optimize_step7_review import OptimizeStep7Review
            self.app.push_screen(OptimizeStep7Review())
