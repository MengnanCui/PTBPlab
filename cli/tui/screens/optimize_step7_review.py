"""Optimize step 7 — review YAML preview, save & exit."""
from __future__ import annotations

import yaml
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Markdown, Static

from cli.tui.state import (assemble_optimize_yaml,
                           assemble_optimize_command,
                           assemble_optimize_argv)


class OptimizeStep7Review(Screen):
    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Step 7 of 7  •  Review", classes="step-title"),
            Markdown(id="review"),
            Horizontal(
                Button("Back", id="btn-back"),
                Button("Save & Exit", id="btn-save"),
                Button("Save & Run", id="btn-run", variant="primary"),
                classes="step-buttons",
            ),
            classes="wizard-step",
        )

    def on_mount(self) -> None:
        s = self.app.state
        out_dict = assemble_optimize_yaml(s)
        yaml_text = yaml.safe_dump(out_dict, default_flow_style=False, sort_keys=True).rstrip()
        cmd = assemble_optimize_command(s)
        md = (
            "### About to save\n\n"
            f"**Output:** `{s.output_yaml}`\n\n"
            "```yaml\n" + yaml_text + "\n```\n\n"
            "**Equivalent command:**\n\n"
            f"```bash\n{cmd}\n```\n\n"
            "*Save & Exit*: write the YAML and quit (you run the command yourself).\n\n"
            "*Save & Run*: write the YAML and `exec` the command immediately —\n"
            "the wizard process is replaced by `ptbp optimize`, output streams\n"
            "to your terminal as if you'd typed it directly."
        )
        self.query_one('#review', Markdown).update(md)

    def _save_yaml(self) -> None:
        s = self.app.state
        out_dict = assemble_optimize_yaml(s)
        target = s.output_yaml
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'w') as fh:
            yaml.safe_dump(out_dict, fh, default_flow_style=False, sort_keys=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        s = self.app.state
        if event.button.id == "btn-save":
            self._save_yaml()
            self.app.exit(result={
                'yaml_path': str(s.output_yaml),
                'command':   assemble_optimize_command(s),
            })
            return
        if event.button.id == "btn-run":
            self._save_yaml()
            self.app.exit(result={
                'launch_now': True,
                'yaml_path':  str(s.output_yaml),
                'command':    assemble_optimize_command(s),
                'argv':       assemble_optimize_argv(s),
            })
