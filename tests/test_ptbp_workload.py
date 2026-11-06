"""pydftb's GPU self-check computes PTBP's Cu DFTB stage without the dataset file
(``pydftb.validation.ptbp_workload``, milestone M7e): the structures of dft/dft.xyz, the k-point
grids of ``utils.dftb_stage`` and the calculator of ``utils.engines``."""
from pathlib import Path

import ase.io
import numpy as np
import pytest

import utils.engines as eng
from pydftb.validation import ptbp_workload as W
from utils import data, dftb_stage

ROOT = Path(__file__).resolve().parents[1]
BASE = {"calc_type": "static", "poly_rep": "Yes", "SCC": "Yes", "slater_p": "./"}  # run_dataset


@pytest.fixture(autouse=True)
def _restore():
    saved = eng.settings()
    yield
    eng._SETTINGS.clear()
    eng._SETTINGS.update(saved)


def _stage():
    """dft/dft.xyz and the tasks of PTBP's stage (Boltzmann weights of run.py at 3000 K, k-point
    density 5.0, EOS groups of 11)."""
    dft = ase.io.read(ROOT / "dft" / "dft.xyz", ":")[:66]
    _, bolt = data.get_bolt_factor(dft[5::11], "boltzmann", 3000)
    return dft, dftb_stage.make_tasks(dft, bolt, 5.0, BASE)


def test_the_structures_and_grids_are_the_stages():
    dft, tasks = _stage()
    work = W.structures()
    assert len(work) == len(tasks) == 66
    for s, mole, task in zip(work, dft, tasks):
        assert s.phase == mole.info["structure_name"]
        assert s.atoms.get_chemical_symbols() == mole.get_chemical_symbols()
        assert np.abs(s.atoms.cell.array - mole.cell.array).max() < 1e-8
        assert np.abs(s.atoms.positions - mole.positions).max() < 1e-8
        assert s.kpts == tuple(task.dftb_args["kpts"])


def _arguments(monkeypatch, make) -> dict:
    """The arguments ``make()`` passes to pydftb's ASE calculator."""
    import pydftb.ase_calc
    seen = {}

    class Fake:
        def __init__(self, skdir, **kw):
            seen.update(kw, skdir=skdir)
    monkeypatch.setattr(pydftb.ase_calc, "PyDFTB", Fake)
    make()
    return seen


def test_the_calculator_is_the_stages(monkeypatch):
    eng.configure(engine="pydftb")
    dft, tasks = _stage()
    fcc = W.structures(phases=["fcc"])[-1]                 # fcc 10: a k-point grid
    ptbp = _arguments(monkeypatch, lambda: eng._pydftb_calculator(dft[-1], tasks[-1].dftb_args))
    assert ptbp["kpts"] == fcc.kpts
    assert _arguments(monkeypatch, lambda: W.calculator(fcc, skdir="./")) == ptbp


def test_the_calculator_on_a_gpu_is_the_stages(monkeypatch):
    pytest.importorskip("torch")
    eng.configure(engine="pydftb", pydftb_backend="torch", pydftb_device="cpu",
                  pydftb_dtype="float32")
    dft, tasks = _stage()
    dimer = W.structures(phases=["dimer"])[0]
    ptbp = _arguments(monkeypatch, lambda: eng._pydftb_calculator(dft[0], tasks[0].dftb_args))
    mine = _arguments(monkeypatch, lambda: W.calculator(dimer, skdir="./", backend="torch",
                                                        device="cpu", dtype="float32"))
    assert mine == ptbp and ptbp["kpts"] == (1, 1, 1)
