"""The hotcent tangent (utils.hotcent_tangent, milestone M4c) with a stand-in for hotcent
that writes SKF files whose tables scale with the confinement radii: central differences,
the r0_d probe and its cache, the checks on the SKF grid and repulsive, the contraction."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from pydftb.skf import SKFile, _expand_fortran
from utils import hotcent_tangent as HT

ROOT = Path(__file__).resolve().parents[1]
CU_SKF_GZ = ROOT / "pydftb" / "tests" / "data" / "skf" / "ptbp_hcocu" / "Cu-Cu.skf.gz"
R0 = (2.749, 12.036)
HP = {"xc": "GGA_X_PBE+GGA_C_PBE", "superposition": "density", "opt_elem": [], "label": "ptbp"}


def _lines():
    with gzip.open(CU_SKF_GZ, "rt") as fh:
        return fh.read().splitlines()


def _write(outdir, factor, rows=None, poly=None):
    """Cu-Cu.skf with the table times ``factor`` (and optionally fewer rows / a new
    repulsive polynomial)."""
    lines = _lines()
    ngrid = int(_expand_fortran(lines[0])[1])
    start = 3                                                # homonuclear: header, onsite, mass
    for k in range(ngrid - 1):
        vals = _expand_fortran(lines[start + k])
        lines[start + k] = " ".join(repr(float(v) * factor) for v in vals)
    if rows is not None:
        head = lines[0].split()
        head[1] = str(rows)
        lines[0] = " ".join(head)
    if poly is not None:
        mass = _expand_fortran(lines[2])
        mass[1], mass[9] = poly, 4.0                         # c2 and the polynomial's cut-off
        lines[2] = " ".join(repr(float(v)) for v in mass)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "Cu-Cu.skf").write_text("\n".join(lines) + "\n")


@pytest.fixture
def fake(monkeypatch):
    """run_hotcent stand-in: table x (1 + a (r0_w - 2.749) + b (r0_d - 12.036))."""
    calls = []
    state = {"a": 0.1, "b": 0.0, "rows": None, "poly": None}

    def run(hp_settings, confinement, outdir, workers=1):
        (w, d), = confinement[0].values()
        calls.append((w, d))
        bad = w > R0[0] + 1e-12                              # changes on the plus side only
        _write(outdir, 1.0 + state["a"] * (w - R0[0]) + state["b"] * (d - R0[1]),
               rows=state["rows"] if bad else None, poly=state["poly"] if bad else None)
        return 0.0

    monkeypatch.setattr(HT, "run_hotcent", run)
    monkeypatch.setattr(HT, "_PROBES", {})
    return calls, state


def _conf(w, d):
    return {"Cu": [w, d]}, {"Cu": 2.0}


def test_central_difference_and_zero_r0_d(tmp_path, fake):
    calls, _ = fake
    _write(tmp_path / "base", 1.0)
    t = HT.compute(_conf, HP, str(tmp_path / "base"), ["Cu"], *R0, probe_key="k")
    base = SKFile.read(str(tmp_path / "base" / "Cu-Cu.skf"), homo=True).table
    assert np.allclose(t["r0_w"].tables[("Cu", "Cu")], 0.1 * base, rtol=1e-9, atol=1e-14)
    assert t["r0_w"].kinds() == ("tables",)
    assert t["r0_d"].zero and t["r0_d"].probed and t["r0_d"].kinds() == ()
    assert len(calls) == 3                                   # r0_w +- h and the r0_d probe
    assert not (tmp_path / "base" / "tangent").exists()      # removed after use
    calls.clear()
    t2 = HT.compute(_conf, HP, str(tmp_path / "base"), ["Cu"], *R0, probe_key="k")
    assert len(calls) == 2 and t2["r0_d"].zero               # the probe is cached per run setting


def test_r0_d_that_matters_gets_central_differences(tmp_path, fake):
    calls, state = fake
    state["b"] = 0.02
    _write(tmp_path / "base", 1.0)
    t = HT.compute(_conf, HP, str(tmp_path / "base"), ["Cu"], *R0, probe_key="k2")
    base = SKFile.read(str(tmp_path / "base" / "Cu-Cu.skf"), homo=True).table
    assert not t["r0_d"].zero and t["r0_d"].step == HT.DEFAULT_STEPS[1]
    assert np.allclose(t["r0_d"].tables[("Cu", "Cu")], 0.02 * base, rtol=1e-8, atol=1e-14)
    assert len(calls) == 4


@pytest.mark.parametrize("change, match", [("rows", "grid"), ("poly", "repulsive")])
def test_unsupported_changes_are_errors(tmp_path, fake, change, match):
    _, state = fake
    if change == "rows":
        state["rows"] = int(_expand_fortran(_lines()[0])[1]) - 5
    else:
        state["poly"] = 0.5
    _write(tmp_path / "base", 1.0)
    with pytest.raises(ValueError, match=match):
        HT.compute(_conf, HP, str(tmp_path / "base"), ["Cu"], *R0)


def test_contract():
    t = HT.Tangent("r0_w", 0.04, tables={("Cu", "Cu"): np.full((2, 3), 2.0)},
                   onsite={"Cu": np.array([0.0, 1.0])})
    g = {"tables": {("Cu", "Cu"): np.ones((2, 3)), ("Cu", "O"): np.ones(4)},
         "onsite": {"Cu": np.array([5.0, 3.0])}, "hubbard": {"Cu": None}}
    assert HT.contract(g, t) == 12.0 + 3.0
    assert HT.contract(g, HT.Tangent("r0_d", 0.0, zero=True)) == 0.0
