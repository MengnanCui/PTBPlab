"""hotcent tables of a BO step (utils.hotcent_parallel): byte-identical to hotpy's
original full_hotcent for any number of processes."""
import hashlib
import sys

import pytest

hc = pytest.importorskip("hotcent")
pytest.importorskip("pylibxc")

from utils import hotcent_parallel as hcp            # noqa: E402
from utils.hotpy import hotpy                         # noqa: E402
from utils.parallel import CpuBudget, set_plan, simple_plan   # noqa: E402

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork: Linux"),
              pytest.mark.skipif(not hcp.sharding_supported(), reason="hotcent is not 2.1.0")]

XC = "GGA_X_PBE+GGA_C_PBE"


def _files(d, log_order=True):
    """sha256 of the SKF files and pot.out; with several jobs the original's pot.out has
    its lines in the order the concurrent jobs happened to write them: compare sorted."""
    out = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.skf"))}
    lines = (d / "pot.out").read_text().splitlines()
    out["pot.out"] = lines if log_order else sorted(lines)
    return out


def _run(tmp_path, monkeypatch, name, r0, opt_elem=(), ncores=None, reference=False,
         log_order=True):
    out = tmp_path / name
    out.mkdir()
    monkeypatch.chdir(out)
    hp = hotpy()
    hp.set_variables(r0, {s: 2.0 for s in r0}, xc=XC, superposition="density")
    set_plan(None if ncores is None else simple_plan(CpuBudget(ncores, {"--ncores": ncores}), 66))
    try:
        if reference:
            hp.full_hotcent_reference(superposition="density", opt_elem=list(opt_elem))
        else:
            hp.full_hotcent(superposition="density", opt_elem=list(opt_elem))
    finally:
        set_plan(None)
    return _files(out, log_order)


def test_cu_identical_for_any_number_of_processes(tmp_path, monkeypatch):
    r0 = {"Cu": [2.749, 12.036]}
    ref = _run(tmp_path, monkeypatch, "ref", r0, reference=True)
    assert set(ref) == {"Cu-Cu.skf", "pot.out"}
    for n in (None, 1, 4):
        assert _run(tmp_path, monkeypatch, f"n{n}", r0, ncores=n) == ref


def test_two_elements_and_opt_elem(tmp_path, monkeypatch):
    r0 = {"H": [2.2, 5.0], "C": [3.3, 7.5]}
    ref = _run(tmp_path, monkeypatch, "ref", r0, reference=True)
    assert set(ref) == {"H-H.skf", "C-H.skf", "H-C.skf", "C-C.skf", "pot.out"}
    assert _run(tmp_path, monkeypatch, "n3", r0, ncores=3) == ref
    r0 = {"H": [2.2, 5.0], "C": [3.3, 7.5], "O": [2.9, 6.5]}
    ref = _run(tmp_path, monkeypatch, "opt_ref", r0, opt_elem=["O"], reference=True,
               log_order=False)                                    # two concurrent jobs
    assert set(ref) == {"H-H.skf", "O-H.skf", "H-O.skf", "O-O.skf", "C-C.skf", "O-C.skf",
                        "C-O.skf", "pot.out"}
    assert _run(tmp_path, monkeypatch, "opt_n4", r0, opt_elem=["O"], ncores=4,
                log_order=False) == ref


def test_jobs_and_tables_follow_generate_skf():
    assert hcp.skf_jobs(["Cu"]) == [["Cu", "Cu"]]
    assert hcp.skf_jobs(["H", "C", "O"]) == [["H", "C"], ["H", "O"], ["C", "O"]]
    assert hcp.skf_jobs(["H", "C", "O"], ["O"]) == [["H", "O"], ["C", "O"]]
    assert hcp.job_tables(["Cu", "Cu"]) == [("Cu", "Cu")] * 3     # the old triple work
    assert hcp.job_tables(["H", "C"]) == [("H", "H"), ("C", "H"), ("C", "C")]


def test_a_failing_atom_fails_its_jobs_only(tmp_path, monkeypatch, capsys):
    real = hotpy.hot_hamiltonian

    def broken(self, r0_x, p_x, xc, symbol_x, label="ptbp"):
        if symbol_x == "C":
            raise RuntimeError("atom did not converge")
        return real(self, r0_x, p_x, xc, symbol_x, label)

    monkeypatch.setattr(hotpy, "hot_hamiltonian", broken)
    files = _run(tmp_path, monkeypatch, "fail", {"H": [2.2, 5.0], "C": [3.3, 7.5], "O": [2.9, 6.5]},
                 ncores=2)
    # jobs: (H, C) and (C, O) fail, (H, O) writes its three tables
    assert set(files) == {"H-H.skf", "O-H.skf", "H-O.skf", "O-O.skf", "pot.out"}
    assert capsys.readouterr().out.count("Exception: atom did not converge") == 2
