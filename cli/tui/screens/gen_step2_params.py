"""Gen wizard step 2 — parameter set choice."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, RadioSet, RadioButton, Static


PARAMS_OPTIONS = [
    ('PTBP',      "Published; robust for solids (this paper's recommended set)"),
    ('QNplusRep', "QN bandstructure + PTBP repulsion"),
    ('Prior',     "Empirical: r0_w = 2·r_cov, r0_d = 4·r_cov"),
    ('Personal',  "Bring your own r0 / repulsion values"),
]


class GenStep2Params(Screen):
    def compose(self) -> ComposeResult:
        radios = [RadioButton(f"{k}  —  {desc}", id=f"p-{k}")
                  for k, desc in PARAMS_OPTIONS]
        yield Vertical(
            Static("Step 2 of 4  •  Parameter set", classes="step-title"),
            RadioSet(*radios, id="p-set"),
            Horizontal(
                Button("Back", id="btn-back"),
                Button("Next", id="btn-next", variant="primary"),
                classes="step-buttons",
            ),
            classes="wizard-step",
        )

    def on_mount(self) -> None:
        rs = self.query_one('#p-set', RadioSet)
        for i, (k, _) in enumerate(PARAMS_OPTIONS):
            if k == self.app.state.gen_params:
                rs._selected = i
                break

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            rs = self.query_one('#p-set', RadioSet)
            idx = rs.pressed_index if rs.pressed_index >= 0 else 0
            self.app.state.gen_params = PARAMS_OPTIONS[idx][0]
            from cli.tui.screens.gen_step3_output import GenStep3Output
            self.app.push_screen(GenStep3Output())
