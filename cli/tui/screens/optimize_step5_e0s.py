"""Optimize step 5 — choose the E0s strategy."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, RadioSet, RadioButton, Static


E0S_OPTIONS = [
    ('auto',
     "Auto — fit from dataset energies + cache to <ref>/E0s_dft.json"),
    ('cache',
     "Load from existing cache file"),
    ('manual',
     "Specify manually as a JSON dict"),
]


class OptimizeStep5E0s(Screen):
    def compose(self) -> ComposeResult:
        radios = [RadioButton(label, id=f"e0s-{key}") for key, label in E0S_OPTIONS]
        yield Vertical(
            Static("Step 5 of 7  •  E0s", classes="step-title"),
            Static(
                "Atomic reference energies. Used to compute formation\n"
                "energy from per-structure DFT energies."),
            RadioSet(*radios, id="e0s-set"),
            Static("\nCache path (only for 'Load from cache'):"),
            Input(placeholder="/path/to/E0s_dft.json", id="e0s-cache"),
            Static("Manual JSON (only for 'Specify manually'):"),
            Input(placeholder='{"6": -37.8, "1": -0.5}', id="e0s-json"),
            Horizontal(
                Button("Back", id="btn-back"),
                Button("Next", id="btn-next", variant="primary"),
                classes="step-buttons",
            ),
            classes="wizard-step",
        )

    def on_mount(self) -> None:
        rs = self.query_one('#e0s-set', RadioSet)
        for i, (key, _) in enumerate(E0S_OPTIONS):
            if key == (self.app.state.e0s_strategy or 'auto'):
                rs._selected = i
                break
        if self.app.state.e0s_cache_path:
            self.query_one('#e0s-cache', Input).value = str(self.app.state.e0s_cache_path)
        if self.app.state.e0s_manual_json:
            self.query_one('#e0s-json', Input).value = self.app.state.e0s_manual_json

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            from pathlib import Path
            rs = self.query_one('#e0s-set', RadioSet)
            idx = rs.pressed_index if rs.pressed_index >= 0 else 0
            self.app.state.e0s_strategy = E0S_OPTIONS[idx][0]
            cache_str = self.query_one('#e0s-cache', Input).value.strip()
            self.app.state.e0s_cache_path = Path(cache_str).expanduser() if cache_str else None
            self.app.state.e0s_manual_json = self.query_one('#e0s-json', Input).value.strip()

            from cli.tui.screens.optimize_step6_advanced import OptimizeStep6Advanced
            self.app.push_screen(OptimizeStep6Advanced())
