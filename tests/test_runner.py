"""Tests for gui.core.runner (subprocess + consolidated single-file log)."""
import sys

from gui.core.runner import JobRunner, make_log_path


def test_make_log_path(tmp_path):
    p = make_log_path(tmp_path / "logs")
    assert p.parent.exists()
    assert p.name.startswith("gui_run_")
    assert p.suffix == ".log"


def test_run_sync_streams_and_writes_single_log(tmp_path):
    log = tmp_path / "run.log"
    lines = []
    runner = JobRunner(
        [sys.executable, "-c", "print('hello-from-child')"],
        log_path=log,
        cwd=tmp_path,
    )
    exits = []
    rc = runner.run_sync(on_line=lines.append, on_exit=exits.append)

    assert rc == 0
    assert exits == [0]
    assert any("hello-from-child" in ln for ln in lines)

    text = log.read_text()
    # one consolidated file: header + command + streamed output + footer manifest
    assert "PTBP GUI run" in text
    assert "hello-from-child" in text
    assert "process exited with code 0" in text
    assert "Artifacts:" in text


def test_run_sync_reports_missing_binary(tmp_path):
    log = tmp_path / "run.log"
    runner = JobRunner(["__definitely_not_a_real_binary__"], log_path=log, cwd=tmp_path)
    rc = runner.run_sync()
    assert rc == 127
    assert log.exists()
    assert "failed to start" in log.read_text()


def test_artifact_manifest_lists_present_files(tmp_path):
    (tmp_path / "convergence.png").write_text("x")
    (tmp_path / "summary.json").write_text("{}")
    log = tmp_path / "run.log"
    JobRunner([sys.executable, "-c", "print(1)"], log_path=log, cwd=tmp_path).run_sync()
    text = log.read_text()
    assert "convergence.png" in text
    assert "summary.json" in text
