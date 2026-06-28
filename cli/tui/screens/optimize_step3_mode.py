"""Optimize step 3 — choose the loss mode."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, RadioSet, RadioButton, Static


MODE_DESCRIPTIONS = {
    'auto':            "let the wizard decide based on the dataset",
    'energygeometry':  "EOS scan: μ_E + μ_V + μ_B from Birch-Murnaghan fit",
    'dataset':         "flat trajectory: (1−c)μ_E + cμ_F over per-structure energies",
    'reaction':        "multi-formula dataset; same loss as dataset + per-formula CSV",
    'bandstructure':   "band structure mismatch (HOMO/LUMO)",
}


class OptimizeStep3Mode(Screen):
    def compose(self) -> ComposeResult:
        radio_buttons = []
        detected = ''
        if self.app.state.dataset_report:
            detected = self.app.state.dataset_report.detected_mode
        for mode, desc in MODE_DESCRIPTIONS.items():
            label = f"{mode}  —  {desc}"
            if mode == 'auto' and detected:
                label = f"auto (= {detected})  —  {desc}"
            radio_buttons.append(RadioButton(label, id=f"mode-{mode}"))

        yield Vertical(
            Static("Step 3 of 7  •  Mode", classes="step-title"),
            Static("Which loss should the optimiser minimise?"),
            RadioSet(*radio_buttons, id="mode-set"),
            Horizontal(
                Button("Back", id="btn-back"),
                Button("Next", id="btn-next", variant="primary"),
                classes="step-buttons",
            ),
            classes="wizard-step",
        )

    def on_mount(self) -> None:
        # Pre-select either the user's prior choice or auto.
        current = self.app.state.mode or 'auto'
        rs = self.query_one('#mode-set', RadioSet)
        for i, mode in enumerate(MODE_DESCRIPTIONS.keys()):
            if mode == current:
                rs._selected = i
                break

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            rs = self.query_one('#mode-set', RadioSet)
            idx = rs.pressed_index if rs.pressed_index >= 0 else 0
            chosen = list(MODE_DESCRIPTIONS.keys())[idx]
            # 'auto' resolves to the detected mode at YAML emission time.
            if chosen == 'auto' and self.app.state.dataset_report:
                chosen = self.app.state.dataset_report.detected_mode
            self.app.state.mode = chosen
            from cli.tui.screens.optimize_step4_optimizer import OptimizeStep4Optimizer
            self.app.push_screen(OptimizeStep4Optimizer())
