"""SKF generator tests. The slow ones are end-to-end runs of `ptbp` that don't
require DFTB+ — only hotcent + physrep — so they are the fastest end-to-end
checks we can run in CI (~30 s on macOS for H). The fast ones cover how a
failed band-structure (hotcent) step is reported."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

SKF_COLUMNS = ("Hdd0 Hdd1 Hdd2 Hpd0 Hpd1 Hpp0 Hpp1 Hsd0 Hsp0 Hss0 "
               "Sdd0 Sdd1 Sdd2 Spd0 Spd1 Spp0 Spp1 Ssd0 Ssp0 Sss0").split()

# Overlap integrals of the published PTBP set (Zenodo record 10677796,
# ParameterSets/PTBP/elementary.zip), keyed by (distance in bohr, column).
# Overlaps depend only on the confined basis, so they pin the confinement
# scheme (hotcent's perturbative default is off by ~1e-2 and fails for Si).
PTBP_OVERLAPS = {
    "Si-Si.skf": {
        (2.0, "Sss0"): 0.544267032606, (2.0, "Spp0"): -0.008739042436528,
        (2.0, "Ssd0"): 0.1165442596466, (2.0, "Sdd0"): -0.02951910297591,
        (4.0, "Sss0"): 0.1789202074613, (4.0, "Spp0"): -0.3610515083452,
        (4.0, "Ssd0"): 0.2572028318937, (4.0, "Sdd0"): 0.1581480139963,
    },
    "H-H.skf": {(2.0, "Sss0"): 0.3448904628897, (4.0, "Sss0"): 0.02358431372426},
}


def _ptbp_cmd():
    """The `ptbp` script if installed, else parameterize.py (it reads sys.argv)."""
    ptbp = shutil.which("ptbp")
    if ptbp:
        return [ptbp]
    return [sys.executable, str(REPO_ROOT / "parameterize.py")]


def _skf_integral(path, r, column):
    """Value of `column` at distance r (bohr) in a homonuclear simple-format SKF."""
    lines = path.read_text().splitlines()
    dr = float(lines[0].replace(",", " ").split()[0])
    values = []
    for token in lines[3 + round(r / dr) - 1].replace(",", " ").split():
        count, _, value = token.rpartition("*")  # SKF shorthand: 19*0.0
        values += [float(value)] * int(count or 1)
    return values[SKF_COLUMNS.index(column)]


@pytest.mark.slow
def test_skf_generator_H_PTBP(tmp_path, monkeypatch):
    """Generates H-H.skf using the published PTBP hyperparameters; checks
    the file exists and has both a band block and a Spline block."""
    monkeypatch.chdir(tmp_path)

    cmd = _ptbp_cmd() + ["--skf_generator", "full",
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


@pytest.mark.slow
def test_skf_generator_Si_PTBP(tmp_path, monkeypatch):
    """Regression: Si's empty 3d shell made hotcent fail inside the Pool, the
    error was swallowed and physrep then crashed with a TypeError. All H/Si
    pair files must be written and match the published PTBP overlaps."""
    monkeypatch.chdir(tmp_path)

    cmd = _ptbp_cmd() + ["gen", "H", "Si", "--params", "PTBP"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (
        f"ptbp exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )

    for name in ("H-H.skf", "H-Si.skf", "Si-H.skf", "Si-Si.skf"):
        skf = tmp_path / name
        assert skf.exists(), f"{name} was not written"
        assert "Spline" in skf.read_text(), f"{name} missing the Spline (repulsive) block"
        assert skf.stat().st_size > 5_000, f"{name} is suspiciously small"
    assert not list(tmp_path.glob("SKF_*")), "per-pair hotcent folders were left behind"

    for name, overlaps in PTBP_OVERLAPS.items():
        for (r, column), expected in overlaps.items():
            assert _skf_integral(tmp_path / name, r, column) == pytest.approx(expected, abs=1e-6), (
                f"{name} {column} at r={r} bohr differs from the published PTBP set")


@pytest.mark.parametrize("method, args, missing", [
    ("write_skf", ("PTBP",), "H-H.skf"),
    ("write_specific_skf", ([14], "PTBP"), "H-Si.skf"),
])
def test_physrep_missing_band_skf_raises(tmp_path, monkeypatch, method, args, missing):
    """physrep used to "raise" through logging.raiseExceptions (a bool), giving
    TypeError: 'bool' object is not callable. It must name the missing file."""
    from tools.physrep import physrep
    from utils.parameters_set import get_PTBP

    monkeypatch.chdir(tmp_path)
    ptbp = get_PTBP()
    pr = physrep([1, 14], npoints=700,
                 scale_cov=[ptbp["H"]["sigma_rep"], ptbp["Si"]["sigma_rep"]],
                 k_xc=[0.0, 0.0])

    with pytest.raises(FileNotFoundError, match=re.escape(f"{missing} not found")):
        getattr(pr, method)(*args)


def test_full_hotcent_reports_failed_pairs(tmp_path, monkeypatch):
    """A hotcent failure inside the Pool used to be swallowed by error_callback,
    so the missing SKFs only surfaced later in physrep. full_hotcent must raise,
    naming each failed pair and its exception."""
    from utils.hotpy import hotpy

    # Named like the method: the Pool pickles bound methods by name. The fork
    # start method (utils/_compat.py) lets workers inherit the monkeypatch.
    def generate_skf(self, symbol_pair, *args):
        if "Si" in symbol_pair:
            raise ValueError("boom")

    monkeypatch.setattr(hotpy, "generate_skf", generate_skf)
    monkeypatch.chdir(tmp_path)
    symbols = ["H", "C", "Si"]
    hp = hotpy({s: [3.0, 6.0] for s in symbols}, {s: 2.0 for s in symbols})

    with pytest.raises(RuntimeError) as excinfo:
        hp.full_hotcent("density")

    message = str(excinfo.value)
    assert "for 2 of 3 element pair(s)" in message
    assert "H-Si: ValueError: boom" in message
    assert "C-Si: ValueError: boom" in message
    assert "H-C" not in message
    assert isinstance(excinfo.value.__cause__, ValueError)
