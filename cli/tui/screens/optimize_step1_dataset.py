"""Optimize wizard step 1 of 7 — dataset path."""
from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, Static

from cli.dataset_inspect import inspect


class OptimizeStep1Dataset(Screen):
    """Pick the DFT reference dataset path. Validates on Next."""

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Step 1 of 7  •  Dataset", classes="step-title"),
            Static(
                "Path to the DFT reference (xyz / extxyz / traj / json).\n"
                "Tip: tab-completion in the field is not native; paste an "
                "absolute path or use a relative path from your shell cwd.",
            ),
            Input(placeholder="/path/to/my_data.xyz", id="path-input"),
            Static("", id="path-feedback"),
            Horizontal(
                Button("Back", id="btn-back"),
                Button("Next", id="btn-next", variant="primary"),
                classes="step-buttons",
            ),
            classes="wizard-step",
        )

    def on_mount(self) -> None:
        # Pre-fill from state in case the user came back from a later step.
        if self.app.state.dataset_path:
            self.query_one('#path-input', Input).value = str(self.app.state.dataset_path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            raw = self.query_one('#path-input', Input).value.strip()
            if not raw:
                self.query_one('#path-feedback', Static).update(
                    "[red]Please enter a path.[/]")
                return
            p = Path(raw).expanduser().resolve()
            if not p.exists():
                self.query_one('#path-feedback', Static).update(
                    f"[red]File not found: {p}[/]")
                return

            report = inspect(p)
            if report.error:
                self.query_one('#path-feedback', Static).update(
                    f"[red]{report.error}[/]")
                return

            self.app.state.dataset_path = p
            self.app.state.dataset_report = report
            # Pre-seed mode + eos_points from auto-detection.
            if not self.app.state.mode or self.app.state.mode == 'auto':
                self.app.state.mode = report.detected_mode

            from cli.tui.screens.optimize_step2_validation import OptimizeStep2Validation
            self.app.push_screen(OptimizeStep2Validation())
