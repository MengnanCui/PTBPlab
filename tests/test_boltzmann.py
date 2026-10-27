"""Boltzmann factors of the energygeometry loss within each composition (milestone M5b-4).

The factors weigh the EOS groups by their DFT total energy per atom at 3000 K.  Energies per atom
of different compositions are not comparable (-1085 eV/atom for B2N2, -2224 for N3Na, -40511 for
I4N16), so before M5b-4 only the lowest composition kept a weight.  They are now computed within
each composition (its reduced formula): the factors that composition alone would get.  One
composition (every dataset of one element) keeps the factors of before, bit for bit; every user of
the factors (the DFTB stage's k-points, both loss paths, utils.loss_grad, the plan, the printout)
takes them from utils.data.get_bolt_factor; the checkpoint's meta file of a run with several
compositions records the new factors, so that a checkpoint of before is not resumed."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import ase.io
import numpy as np
import pytest
from ase.units import kB

import utils.data as data
import utils.loss as L
from tests.test_types import _box
from utils import recommender as R
from utils import typemodel as TM

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cu():
    return ase.io.read(ROOT / "dft" / "dft.xyz", ":")                # 6 phases of Cu


@pytest.fixture(scope="module")
def nitrides():
    return ase.io.read(ROOT / "dft" / "nitrides.xyz", ":")           # 18 compositions


def _factors_before(atoms, temperature=3000):
    """get_bolt_factor(atoms, "boltzmann", T) before M5b-4 (the reference): one Boltzmann
    distribution over every structure."""
    n = len(atoms)
    e = [a.get_total_energy() / a.get_global_number_of_atoms() for a in atoms]
    emin = np.array(e).min()
    w = [np.exp(-(e[i] - emin) / (kB * temperature)) for i in range(n)]
    norm = w / sum(w)
    return norm, np.array([[norm[k]] * 11 for k in range(n)]).flatten()


def test_composition_is_the_reduced_formula(nitrides):
    names = [_box(s, np.zeros((len(s), 3))) for s in
             (["B", "B", "N", "N"], ["Na", "N", "N", "N"], ["Cu"] * 4, ["K", "K"] + ["N"] * 6,
              ["C"] * 3 + ["N"] * 4)]
    assert [data.composition(a) for a in names] == ["BN", "N3Na", "Cu", "KN3", "C3N4"]
    comps = data.group_compositions(nitrides, 11)
    assert len(comps) == 18 == len(set(comps)) and comps[0] == "BN" and comps[16] == "IN4"
    assert data.group_compositions(nitrides[:22], 11) == ["BN", "C3N4"]


def test_one_composition_keeps_the_factors_of_before(cu):
    """Cu (dimer ... fcc, one composition): the factors and their types are those of before,
    bit for bit; 'identical' does not change."""
    centre = cu[5::11]
    assert {a.get_chemical_formula(empirical=True) for a in centre} == {"Cu"}
    new, new11 = data.get_bolt_factor(centre, "boltzmann", 3000)
    old, old11 = _factors_before(centre)
    assert new.tobytes() == old.tobytes() and new11.tobytes() == old11.tobytes()
    assert new.dtype == old.dtype and new11.dtype == old11.dtype and type(new) is type(old)
    assert [round(float(v), 4) for v in new] == [0.0, 0.0017, 0.008, 0.1266, 0.3998, 0.4639]
    ident, ident11 = data.get_bolt_factor(centre, "identical", 3000)
    assert ident.tolist() == [1] * 6 and ident11.tolist() == [1] * 66


def test_factors_within_each_composition(cu, nitrides):
    """dft/nitrides.xyz: before, only I4N16 had a weight (1, the others 0); now each of the 18
    compositions, one EOS group each, gets 1.  Cu's phases mixed with B2N2 and N3Na (in any
    order): Cu's factors are those of Cu alone, bit for bit, the nitrides get 1."""
    centre = nitrides[5::11]
    assert _factors_before(centre)[0].tolist() == [0.0] * 16 + [1.0, 0.0]
    new, new11 = data.get_bolt_factor(centre, "boltzmann", 3000)
    assert new.tolist() == [1.0] * 18 and new11.tolist() == [1.0] * 198
    cu_c, bn, nan3 = cu[5::11], nitrides[5], nitrides[27]
    mixed = [cu_c[0], bn, cu_c[1], cu_c[2], nan3, cu_c[3], cu_c[4], cu_c[5]]
    f, f11 = data.get_bolt_factor(mixed, "boltzmann", 3000)
    alone = data.get_bolt_factor(cu_c, "boltzmann", 3000)[0]
    assert f[[0, 2, 3, 5, 6, 7]].tobytes() == alone.tobytes() and f[[1, 4]].tolist() == [1.0, 1.0]
    assert f11.tobytes() == np.repeat(f, 11).tobytes() and f.dtype == f11.dtype == np.float64
    # the plan's weights (ptbp plan, utils.recommender) are the run's
    two = nitrides[0:11] + nitrides[22:33]
    assert R.bolt_for(two, "energygeometry", 11) == [1.0] * 22
    assert R.bolt_for(two, "energygeometry", 11, multi_element=["N"]) == [1] * 22


def _two_compositions(tmp_path):
    """B2N2 and N3Na of dft/nitrides.xyz (examples/04's 22 structures), one EOS group each."""
    images = ase.io.read(ROOT / "dft" / "nitrides.xyz", ":")
    (tmp_path / "data").mkdir(exist_ok=True)
    path = tmp_path / "data" / "bn_nan3.xyz"
    ase.io.write(path, images[0:11] + images[22:33])
    return path


def test_run_and_plan_use_the_factors_per_composition(tmp_path):
    """ptbp optimize (energygeometry, parameters per element; --postprocess-only stops after the
    plan) on B2N2 and N3Na: the run's factors are 1 and 1 (before: 0 and 1), printed per
    composition, and its plan computes B2N2 with its k-points (before: Gamma only, weight < 0.1);
    ptbp plan gives the same."""
    dataset = _two_compositions(tmp_path)
    out = subprocess.run(
        [sys.executable, str(ROOT / "parameterize.py"), "optimize", str(dataset), "--mode",
         "energygeometry", "--engine", "pydftb", "--ncores", "1", "--postprocess-only",
         "--output", str(tmp_path / "run")],
        cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert "result.pkl" in out.stderr, out.stderr[-2000:]
    assert "Boltzmann factor: [1.0, 1.0]" in out.stdout
    assert "within each of the 2 compositions" in out.stdout
    assert "  BN: [1.0]\n" in out.stdout and "  N3Na: [1.0]\n" in out.stdout
    plan = json.loads((tmp_path / "run" / "plan.json").read_text())
    assert plan["workload"]["structures"] == 22 and plan["workload"]["gamma_only"] == 0
    out = subprocess.run(
        [sys.executable, str(ROOT / "parameterize.py"), "plan", str(dataset), "--mode",
         "energygeometry", "--engine", "pydftb", "--ncores", "1", "--json"],
        cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout)["workload"]["gamma_only"] == 0


def _loss(dataset, mode="energygeometry", multi_element=None, types=None,
          checkpoint="checkpoint.pkl"):
    args = argparse.Namespace(optimization_option=mode, eos_points=11, checkpoint=checkpoint,
                              multi_element=multi_element)
    return L.Loss(args_input=args, dft_input=dataset, types=types)


def test_checkpoint_records_the_factors_per_composition(tmp_path, monkeypatch):
    """The meta file of an energygeometry run on two compositions records "boltzmann": "per
    composition" (both layouts), and such a run refuses a checkpoint without it (written before
    M5b-4: other losses); one composition, --multi_element and the dataset mode write the meta
    file of before and resume as before."""
    from scipy.optimize import OptimizeResult
    from skopt import dump
    monkeypatch.chdir(tmp_path)
    bn = _box(["B", "B", "N", "N"], [[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]])
    bn4 = _box(["B"] * 4 + ["N"] * 4, [[k, 0, 0] for k in range(8)])
    nan3 = _box(["Na", "N", "N", "N"], [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])
    two, one = [bn] * 11 + [nan3] * 11, [bn] * 11 + [bn4] * 11
    assert _loss(two).boltzmann_record() == L.BOLTZMANN_PER_COMPOSITION == "per composition"
    assert _loss(one).boltzmann_record() is None                       # BN twice
    assert _loss(two, multi_element=["N"]).boltzmann_record() is None  # identical weights
    assert _loss(two, mode="dataset").boltzmann_record() is None

    seen = {}

    def fake_gp_minimize(**kw):
        seen.clear()
        seen.update(kw)
        return SimpleNamespace(x_iters=kw["x0"], func_vals=kw["y0"], x=kw["x0"], fun=None)

    monkeypatch.setattr(L, "gp_minimize", fake_gp_minimize)
    meta_path = tmp_path / "checkpoint.pkl.meta.json"
    per_type = TM.TypeModel.from_dataset(two)
    for dataset, types, record in ((two, None, "per composition"), (two, per_type, "per composition"),
                                   (one, None, None)):
        for p in (meta_path, tmp_path / "checkpoint.pkl"):
            p.unlink(missing_ok=True)
        loss = _loss(dataset, types=types)
        loss.Bayesian_optimization(loss.cov_radius, lambda res: None, 11)
        meta = json.loads(meta_path.read_text())
        before = (types.meta("energygeometry") if types is not None else
                  {"ptbp_loss_version": L.SHARED_LOSS_VERSION,
                   "penalty_radius_bohr": loss.cov_radius, "mode": "energygeometry"})
        assert meta == (before if record is None else dict(before, boltzmann=record))
        # resuming: a checkpoint of the same factors, then one without the record
        x0 = [list(map(float, seen["x0"]))] if seen["y0"] is None else seen["x0"]
        dump(OptimizeResult(x_iters=x0, func_vals=np.array([0.9]), x=x0[0], fun=0.9,
                            specs={"args": {}}), str(tmp_path / "checkpoint.pkl"))
        _loss(dataset, types=types).Bayesian_optimization(loss.cov_radius, lambda res: None, 10)
        assert [float(v) for v in seen["y0"]] == [0.9]
        meta_path.write_text(json.dumps(before))
        if record is None:                                         # as before M5b-4
            _loss(dataset, types=types).Bayesian_optimization(loss.cov_radius, lambda r: None, 10)
            assert json.loads(meta_path.read_text()) == before
        else:
            with pytest.raises(RuntimeError, match="written before M5b-4"):
                _loss(dataset, types=types).Bayesian_optimization(loss.cov_radius,
                                                                  lambda r: None, 10)
    # a checkpoint with the record, resumed without these factors (here --multi_element)
    meta_path.write_text(json.dumps({"ptbp_loss_version": L.SHARED_LOSS_VERSION,
                                     "penalty_radius_bohr": L.penalty_radius(two, ["N"]),
                                     "mode": "energygeometry", "boltzmann": "per composition"}))
    loss = _loss(two, multi_element=["N"])
    loss.opt_elem = [7]
    loss.cov_radius = L.penalty_radius(two, ["N"])
    with pytest.raises(RuntimeError, match="records Boltzmann factors 'per composition'"):
        loss.Bayesian_optimization(loss.cov_radius, lambda res: None, 10)
