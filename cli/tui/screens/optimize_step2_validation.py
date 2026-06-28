"""Optimize step 2 — show the dataset report."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Markdown, Static


def _report_to_markdown(report) -> str:
    if report is None:
        return "No report available."
    if report.error:
        return f"### Dataset\n`{report.path}`\n\n**Read failed:** {report.error}"

    lines = [f"### Dataset\n`{report.path}`\n"]
    lines.append(f"- **{report.n_structures}** structures")
    nE, nF, n = report.energies_present, report.forces_present, report.n_structures
    lines.append(f"- Energies: **{nE}/{n}** present"
                 + ("" if nE == n else f"  *(missing: {n-nE})*"))
    lines.append(f"- Forces:   **{nF}/{n}** present"
                 + ("" if nF == n else f"  *(missing: {n-nF})*"))
    if report.formulas:
        top = ", ".join(f"`{k}` × {v}" for k, v in
                        sorted(report.formulas.items(), key=lambda kv: -kv[1])[:6])
        more = "" if report.n_formulas <= 6 else f" + {report.n_formulas - 6} more"
        lines.append(f"- Formulas ({report.n_formulas}): {top}{more}")
    lines.append(f"\n**Auto-detected mode**: `{report.detected_mode}` "
                 f"({report.detection_rationale})")
    if report.warnings:
        lines.append("\n**Warnings:**")
        for w in report.warnings:
            lines.append(f"- {w}")
    return "\n".join(lines)


class OptimizeStep2Validation(Screen):
    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Step 2 of 7  •  Validation", classes="step-title"),
            Markdown(id="report"),
            Horizontal(
                Button("Back", id="btn-back"),
                Button("Next", id="btn-next", variant="primary"),
                classes="step-buttons",
            ),
            classes="wizard-step",
        )

    def on_mount(self) -> None:
        self.query_one('#report', Markdown).update(
            _report_to_markdown(self.app.state.dataset_report))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            from cli.tui.screens.optimize_step3_mode import OptimizeStep3Mode
            self.app.push_screen(OptimizeStep3Mode())
