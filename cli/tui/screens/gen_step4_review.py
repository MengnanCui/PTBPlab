"""Gen wizard step 4 — review and exit. Prints the equivalent command."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Markdown, Static

from cli.tui.state import assemble_gen_command, assemble_gen_argv


class GenStep4Review(Screen):
    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Step 4 of 4  •  Review", classes="step-title"),
            Markdown(id="review"),
            Horizontal(
                Button("Back", id="btn-back"),
                Button("Print & Exit", id="btn-print"),
                Button("Run now", id="btn-run", variant="primary"),
                classes="step-buttons",
            ),
            classes="wizard-step",
        )

    def on_mount(self) -> None:
        cmd = assemble_gen_command(self.app.state)
        md = (
            "### Equivalent command\n\n"
            f"```bash\n{cmd}\n```\n\n"
            "**Print & Exit**: just print the command; you run it yourself.\n\n"
            "**Run now**: `exec` the command immediately — the wizard process\n"
            "is replaced by `ptbp gen`, SKF generation streams to your terminal.\n\n"
            "No DFTB+ required either way — gen only uses hotcent + physrep + pylibxc."
        )
        self.query_one('#review', Markdown).update(md)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        s = self.app.state
        cmd = assemble_gen_command(s)
        if event.button.id == "btn-print":
            self.app.exit(result={'command': cmd})
            return
        if event.button.id == "btn-run":
            self.app.exit(result={
                'launch_now': True,
                'command':    cmd,
                'argv':       assemble_gen_argv(s),
            })
