"""Optimize step 4 — pick the outer-loop optimiser + budget."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, RadioSet, RadioButton, Static


OPTIMIZER_DESCRIPTIONS = {
    'bo':          "skopt gp_minimize — sequential, sample-efficient",
    'parallel_bo': "skopt Optimizer (batched) — fork-pool parallel evaluations",
    'pso':         "pyswarms GlobalBestPSO — population-based, multi-modal",
}


class OptimizeStep4Optimizer(Screen):
    def compose(self) -> ComposeResult:
        radios = [
            RadioButton(f"{k}  —  {v}", id=f"opt-{k}")
            for k, v in OPTIMIZER_DESCRIPTIONS.items()
        ]
        yield Vertical(
            Static("Step 4 of 7  •  Optimizer", classes="step-title"),
            Static("Which outer-loop optimiser?"),
            RadioSet(*radios, id="opt-set"),
            Static("\nTotal evaluations (n_calls):"),
            Input(placeholder="11", id="n-calls"),
            Static("Particles / batch width (PSO + parallel_bo):"),
            Input(placeholder="4", id="n-particles"),
            Horizontal(
                Button("Back", id="btn-back"),
                Button("Next", id="btn-next", variant="primary"),
                classes="step-buttons",
            ),
            classes="wizard-step",
        )

    def on_mount(self) -> None:
        rs = self.query_one('#opt-set', RadioSet)
        for i, k in enumerate(OPTIMIZER_DESCRIPTIONS.keys()):
            if k == (self.app.state.optimizer or 'bo'):
                rs._selected = i
                break
        if self.app.state.n_calls is not None:
            self.query_one('#n-calls', Input).value = str(self.app.state.n_calls)
        self.query_one('#n-particles', Input).value = str(self.app.state.n_particles)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            rs = self.query_one('#opt-set', RadioSet)
            idx = rs.pressed_index if rs.pressed_index >= 0 else 0
            self.app.state.optimizer = list(OPTIMIZER_DESCRIPTIONS.keys())[idx]
            try:
                nc = self.query_one('#n-calls', Input).value.strip()
                self.app.state.n_calls = int(nc) if nc else None
            except ValueError:
                self.app.state.n_calls = None
            try:
                np_ = self.query_one('#n-particles', Input).value.strip()
                self.app.state.n_particles = int(np_) if np_ else 4
            except ValueError:
                self.app.state.n_particles = 4
            from cli.tui.screens.optimize_step5_e0s import OptimizeStep5E0s
            self.app.push_screen(OptimizeStep5E0s())
