"""Subprocess runner + consolidated single-file logging for the GUI.

The GUI never imports the heavy optimiser in-process. Instead it shells out to
the `ptbp` CLI via `JobRunner`, streaming stdout live into a panel while
mirroring *everything* into one consolidated `gui_run_<ts>.log` (header +
resolved command + full output + a trailing artifact manifest). This satisfies
the "run + monitor" and "keep logs in a single file" requirements and keeps the
Tk main loop responsive (reading happens on a worker thread).
"""
from __future__ import annotations

import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence

# Files a finished PTBP run drops into its run folder; listed in the log footer.
ARTIFACT_NAMES = (
    "ptbp_run.yaml",
    "summary.json",
    "result.pkl",
    "log.out",
    "par.out",
    "convergence.png",
    "eval.png",
    "obj.png",
    "results.tgz",
)


def make_log_path(log_dir: str | Path) -> Path:
    """`<log_dir>/gui_run_<UTC-timestamp>.log`, dir created if needed."""
    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    return d / f"gui_run_{ts}.log"


def _header(argv: Sequence[str], cwd: Path) -> str:
    return (
        "=" * 70 + "\n"
        f"PTBP GUI run\n"
        f"  started : {datetime.utcnow().isoformat()}Z\n"
        f"  cwd     : {cwd}\n"
        f"  command : {' '.join(argv)}\n"
        + "=" * 70 + "\n\n"
    )


def _artifact_manifest(cwd: Path) -> str:
    lines = ["\n" + "=" * 70, "Artifacts:"]
    found = False
    for name in ARTIFACT_NAMES:
        p = cwd / name
        if p.exists():
            found = True
            lines.append(f"  - {p}")
    # Also point at nested run folders created by `ptbp optimize --output`.
    for run_dir in sorted(cwd.glob("run_*")):
        if run_dir.is_dir():
            found = True
            lines.append(f"  - {run_dir}/ (run folder)")
    if not found:
        lines.append("  (none found)")
    lines.append("=" * 70 + "\n")
    return "\n".join(lines)


class JobRunner:
    """Run one `ptbp` (or any) command, streaming to a callback + a log file.

    Use `start()` for the GUI (non-blocking, worker thread) or `run_sync()`
    for tests / headless self-checks (blocking, returns the exit code).
    """

    def __init__(
        self,
        argv: Sequence[str],
        log_path: str | Path,
        cwd: Optional[str | Path] = None,
        env: Optional[dict] = None,
    ) -> None:
        self.argv: List[str] = list(argv)
        self.log_path = Path(log_path)
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.env = env
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self.returncode: Optional[int] = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _run(
        self,
        on_line: Optional[Callable[[str], None]],
        on_exit: Optional[Callable[[int], None]],
    ) -> int:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as log:
            log.write(_header(self.argv, self.cwd))
            log.flush()
            try:
                self._proc = subprocess.Popen(
                    self.argv,
                    cwd=str(self.cwd),
                    env=self.env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except (OSError, ValueError) as e:
                msg = f"[runner] failed to start {self.argv[0]!r}: {e}\n"
                log.write(msg)
                if on_line:
                    on_line(msg.rstrip("\n"))
                self.returncode = 127
                if on_exit:
                    on_exit(127)
                return 127

            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                log.write(line)
                log.flush()
                if on_line:
                    on_line(line.rstrip("\n"))
            self._proc.wait()
            self.returncode = self._proc.returncode
            log.write(f"\n[runner] process exited with code {self.returncode}\n")
            log.write(_artifact_manifest(self.cwd))
            log.flush()

        if on_exit:
            on_exit(self.returncode)
        return self.returncode

    def start(
        self,
        on_line: Optional[Callable[[str], None]] = None,
        on_exit: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Launch on a background thread. Returns immediately."""
        if self.is_running:
            raise RuntimeError("job already running")
        self._thread = threading.Thread(
            target=self._run, args=(on_line, on_exit), daemon=True
        )
        self._thread.start()

    def run_sync(
        self,
        on_line: Optional[Callable[[str], None]] = None,
        on_exit: Optional[Callable[[int], None]] = None,
    ) -> int:
        """Run to completion on the calling thread; return the exit code."""
        return self._run(on_line, on_exit)

    def cancel(self) -> None:
        """Terminate the running process, if any."""
        if self.is_running and self._proc is not None:
            self._proc.terminate()

    def wait(self, timeout: Optional[float] = None) -> None:
        """Join the worker thread (for tests / orderly shutdown)."""
        if self._thread is not None:
            self._thread.join(timeout)
