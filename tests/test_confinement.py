"""hotcent's confinement scheme (milestone M4e): self-consistent by default, so that r0_d
confines the density as in hotcent 1.x; ``--confinement perturbative`` keeps the
hotcent >= 2.0 default, with which r0_d changes nothing."""
import json
import sys
from unittest.mock import patch

import pytest

from utils import hotpy as hp_mod


@pytest.fixture(autouse=True)
def _restore():
    yield
    hp_mod.set_confinement("selfconsistent")


def test_setting():
    assert hp_mod.confinement() == "selfconsistent"
    hp_mod.set_confinement("perturbative")
    assert hp_mod.confinement() == "perturbative"
    with pytest.raises(ValueError, match="confinement must be one of"):
        hp_mod.set_confinement("frozen")


class _FakeAtom:
    calls = []

    def __init__(self, symbol, **kwargs):
        _FakeAtom.calls.append(kwargs)

    def run(self):
        pass


def test_atomic_dft_gets_the_scheme(monkeypatch):
    pytest.importorskip("hotcent")
    monkeypatch.setattr(hp_mod, "AtomicDFT", _FakeAtom)
    _FakeAtom.calls.clear()
    h = hp_mod.hotpy({"Cu": [2.749, 12.036]}, {"Cu": 2.0})
    h.hot_hamiltonian([2.749, 12.036], 2.0, "GGA_X_PBE+GGA_C_PBE", "Cu")
    hp_mod.set_confinement("perturbative")
    h.hot_hamiltonian([2.749, 12.036], 2.0, "GGA_X_PBE+GGA_C_PBE", "Cu")
    assert [c["perturbative_confinement"] for c in _FakeAtom.calls] == [False, True]
    assert _FakeAtom.calls[0]["confinement"].r0 == 12.036          # r0_d: the density
    assert {c.r0 for c in _FakeAtom.calls[0]["wf_confinement"].values()} == {2.749}


def test_cli_and_yaml(tmp_path, monkeypatch):
    from cli import config
    from cli.arg_parser import build_default_arg_parser
    from cli.main import dispatch
    assert config.YAML_KEY_TO_ARG["confinement"] == "confinement"
    dataset = tmp_path / "my_data.xyz"
    dataset.write_text("0\n\n")
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_run_path(path, run_name=None):
        captured["argv"] = list(sys.argv)

    for n, extra in enumerate(([], ["--confinement", "perturbative"])):
        with patch("cli.cmd_optimize.runpy.run_path", side_effect=fake_run_path):
            assert dispatch(["optimize", str(dataset), "--output", str(tmp_path / f"run{n}"),
                             "--mode", "dataset", *extra]) == 0
        argv = captured["argv"]
        assert ("--confinement" in argv) == bool(extra)          # the default adds nothing
        monkeypatch.setattr(sys, "argv", argv)
        parsed = build_default_arg_parser()
        assert parsed.confinement == (extra[1] if extra else "selfconsistent")


def test_plan_records_it():
    from utils.parallel import CpuBudget, simple_plan
    from utils.parallel.plan import Plan
    plan = simple_plan(CpuBudget(4, {"--ncores": 4}), 8, "pydftb")
    assert "hotcent" not in plan.to_json()
    plan.hotcent = {"confinement": "selfconsistent", "version": "2.1.0"}
    data = json.loads(json.dumps(plan.to_json()))
    assert data["hotcent"] == plan.hotcent and Plan.from_json(data).hotcent == plan.hotcent
    assert "[plan]   hotcent 2.1.0: selfconsistent confinement" in plan.render()


@pytest.mark.slow
def test_r0_d_changes_the_skf_only_when_self_consistent(tmp_path, monkeypatch):
    """H-H with hotcent itself: r0_d = 3 and 4 Bohr give different tables only with the
    self-consistent scheme."""
    pytest.importorskip("hotcent")
    import contextlib
    import io
    files = {}
    for scheme in ("selfconsistent", "perturbative"):
        hp_mod.set_confinement(scheme)
        for r0_d in (3.0, 4.0):
            out = tmp_path / f"{scheme}_{r0_d}"
            out.mkdir()
            monkeypatch.chdir(out)
            h = hp_mod.hotpy({"H": [1.5, r0_d]}, {"H": 2.0})
            with contextlib.redirect_stdout(io.StringIO()):
                h.full_hotcent_reference(superposition="density", opt_elem=[])
            files[scheme, r0_d] = (out / "H-H.skf").read_bytes()
    assert files["selfconsistent", 3.0] != files["selfconsistent", 4.0]
    assert files["perturbative", 3.0] == files["perturbative", 4.0]
    assert files["selfconsistent", 3.0] != files["perturbative", 3.0]
