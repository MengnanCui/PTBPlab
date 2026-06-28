"""End-to-end smoke test of `ptbp --skf_generator full` for a single
element. Doesn't require DFTB+ — only hotcent + physrep — so this is the
fastest end-to-end check we can run in CI (~30 s on macOS)."""
import os
import shutil
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_skf_generator_H_PTBP(tmp_path, monkeypatch):
    """Generates H-H.skf using the published PTBP hyperparameters; checks
    the file exists and has both a band block and a Spline block."""
    monkeypatch.chdir(tmp_path)

    cmd = [
        sys.executable, "-m", "parameterize"
        # parameterize.py runs cli/run.py via runpy; passing args via sys.argv
    ]
    # Use ptbp script if installed, else fall back to runpy
    ptbp = shutil.which("ptbp")
    if ptbp:
        cmd = [ptbp]
    else:
        # parameterize.py reads sys.argv; build it explicitly
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        cmd = [sys.executable, str(repo_root / "parameterize.py")]

    cmd += ["--skf_generator", "full",
            "--symbols", "H",
            "--known_parameters", "PTBP"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (
        f"ptbp exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )

    assert (tmp_path / "H-H.skf").exists(), "H-H.skf was not written"
    skf_content = (tmp_path / "H-H.skf").read_text()
    # Sanity: a healthy SKF has the band-structure block plus a Spline block.
    assert "Spline" in skf_content, "SKF missing the Spline (repulsive) block"
    # The band block has many integral lines — naive proxy: file > 5 KB.
    assert (tmp_path / "H-H.skf").stat().st_size > 5_000, (
        "H-H.skf is suspiciously small — band block may have been truncated "
        "(was the case-insensitive `os.rename(fname, fname.lower())` bug "
        "reintroduced?)")
