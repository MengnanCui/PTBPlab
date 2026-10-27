"""Two old bugs of the code of before M5b (one parameter set for every element: one element, or
--shared-parameters), fixed in milestone M5b-3.

1. The energygeometry loss on EOS groups of several elements: physrep got one sigma_rep for the
   group's elements (IndexError at the first step).  Every element of a group now gets the
   shared sigma, as the dataset/reaction loss gives it to every element of the dataset.
2. get_elements_from_dataset (utils.data, and its copy in Loss) on structures with different
   numbers of elements: NumPy >= 1.24 refuses the ragged array it built, so every
   ``ptbp optimize`` run on such a dataset stopped at the start (cli/run.py).  The result is
   unchanged, values and types, for every dataset that worked.

M5b-4: --shared-parameters with a --multi_element that lists every element of a dataset of several
elements is refused at the start (the code of one parameter set gave radii to the first listed
element only)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols

import utils.data as data
import utils.engines as eng
import utils.loss as L
from tests.test_types import SKF, _bnna, _box, _copy_skf, _ho_args
from tools.physrep import physrep
from utils import loss_grad as LG
from utils import typemodel as TM
from utils.parameters_set import get_PTBP

ROOT = Path(__file__).resolve().parents[1]
linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork backend")


@pytest.fixture(autouse=True)
def _restore_engine():
    saved = eng.settings()
    switches = eng.type_switching()
    yield
    eng._SETTINGS.clear()
    eng._SETTINGS.update(saved)
    eng.set_type_switching(switches)


def _tables_without_repulsive(names):
    """hotcent replaced by the stored PTBP tables of ``names`` cut before their repulsive spline
    (as hotcent writes them): every spline in a step's files is then one the loss wrote."""
    def full_hotcent(self, superposition, opt_elem=(), label="ptbp"):
        _copy_skf(Path("."), SKF / "ptbp_hcocu", names)
        for p in Path(".").glob("*.skf"):
            lines = p.read_text().splitlines(keepends=True)
            if "Spline \n" in lines:
                p.write_text("".join(lines[:lines.index("Spline \n")]))
    return full_hotcent


def _eos_group(formula, a0, n=11):
    """An EOS group: rock salt ``formula`` (primitive cell) at the lattice constants a0 x
    0.94-1.06, with a made-up DFT energy per atom of -100 + 40 (s - 1)^2 eV (minimum at a0) and
    no forces.  With the stored tables the DFTB minimum of CO lies inside the scan and reaches
    a0 = 3.95 A at sigma ~ 0.72; that of OH (a0 = 3.2 A) hardly depends on sigma."""
    out = []
    for s in np.linspace(0.94, 1.06, n):
        atoms = bulk(formula, "rocksalt", a=a0 * s)
        e = (-100.0 + 40.0 * (s - 1.0) ** 2) * len(atoms)
        atoms.calc = SinglePointCalculator(atoms, energy=e, forces=np.zeros((len(atoms), 3)))
        out.append(atoms)
    return out


# ---- 1. energygeometry with one parameter set on EOS groups of several elements -------------
@linux
def test_shared_parameters_on_eos_groups_of_several_elements(tmp_path, monkeypatch):
    """energygeometry with one parameter set for H, C and O (--shared-parameters, the code of
    before M5b) on two EOS groups of two elements, rock salt CO and OH (pydftb; hotcent replaced
    by the stored tables).  Before M5b-3 physrep got one sigma for a group's two elements:
    IndexError, the run stopped at its first step.  Now every element of a group gets the shared
    sigma: sigma* lies inside the bounds (the CO group sets it), every pair's table gets the
    repulsive at sigma* (C-H too, which no group holds), no plots of one element's phases; at
    sigma* and two other sigma the loss is utils.loss_grad.mu's (the gradient optimisers' copy)
    and that of parameters per element (M5b's path) with sigma_H = sigma_C = sigma_O but for
    the penalty, bit for bit."""
    eng.configure(engine="pydftb")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(L.hotpy, "full_hotcent", _tables_without_repulsive({"H", "C", "O"}))
    dft = _eos_group("CO", 3.95) + _eos_group("OH", 3.2)
    model = TM.TypeModel.from_dataset(dft, shared=True)
    assert model.legacy                                       # the code of before M5b
    _, bolt = data.get_bolt_factor(dft[5::11], "boltzmann", 3000)        # as cli/run.py
    args = _ho_args(tmp_path, "energygeometry")
    args.eos_points, args.E0s, args.kpt_density = 11, None, 4.0
    loss = L.Loss(args_input=args, dft_input=dft, bolt_f11=bolt, opt_elem=[], path=tmp_path,
                  types=model)
    assert not loss.typed
    found = {}
    search = L.Loss.Repulsion_optimization

    def record(self, conf, c0, c1):                           # sigma* and the loss at sigma*
        found["sigma"], found["mu"] = search(self, conf, c0, c1)
        return found["sigma"], found["mu"]

    monkeypatch.setattr(L.Loss, "Repulsion_optimization", record)
    x = [2.6, 5.0]
    mu = float(loss.Loss_EnergyGeometry(x))
    assert np.isfinite(mu) and loss.last_failed is False and loss.worst_success == mu
    sigma = float(found["sigma"])
    assert mu == float(found["mu"]) and 0.2 < sigma < 0.8
    d = tmp_path / "results" / "opt_2.600_5.000"

    # the tables: as physrep writes them with sigma* for every element
    ref = tmp_path / "ref"
    ref.mkdir()
    monkeypatch.chdir(ref)
    _tables_without_repulsive({"H", "C", "O"})(None, "density")
    physrep([1, 6, 8], rmin=0.5, rmax=20.0, npoints=700, scale_cov=[sigma] * 3,
            k_xc=[0.0] * 3).write_skf("ptbp")
    names = sorted(p.name for p in ref.glob("*.skf"))
    assert len(names) == 9
    for name in names:
        assert "Spline" in (d / name).read_text(), name
        assert (d / name).read_bytes() == (ref / name).read_bytes(), name
    assert not (d / "dmin.png").exists() and not (d / "dftb_ener.png").exists()
    par = [float(v) for v in (tmp_path / "par.out").read_text().split(",")]
    assert par[2] == pytest.approx(sigma, abs=1e-8) and par[-1] == pytest.approx(mu, rel=1e-10)

    # the loss of the step's files at sigma: Loss.parallel_repulsion (at sigma*, the search's),
    # utils.loss_grad.mu, and the terms of parameters per element with every sigma equal
    monkeypatch.chdir(d)
    step = LG.StepData.from_files(loss, *x)
    assert [[int(z) for z in g[0]] for g in step.groups] == [[6, 8], [1, 8]]
    per_element = L.Loss(args_input=args, dft_input=dft, bolt_f11=bolt, opt_elem=[],
                         path=tmp_path, types=TM.TypeModel.from_dataset(dft))
    assert per_element.typed
    typed_step = LG.StepData.from_files(per_element, x=x * 3)
    for s in (sigma, 0.3, 0.5):
        ref_mu = mu if s == sigma else float(loss.parallel_repulsion(np.array([s]), *x, False))
        assert float(LG.mu(step, s)).hex() == ref_mu.hex()
        parts = {}
        LG.mu(typed_step, [s] * 3, parts)
        assert ((0.5 * parts["mu_V"] + 0.5 * parts["mu_E"]) * loss.penalty(*x)).hex() \
            == ref_mu.hex()


# ---- 2. structures with different numbers of elements -----------------------------------------
def _molecule(formula):
    """Atoms of ``formula`` in a cubic box (8 A), 0.9 A apart along x (element lists only)."""
    n = len(Atoms(formula))
    return _box(Atoms(formula).get_chemical_symbols(),
                [[4.0 + 0.9 * k - 0.45 * (n - 1), 4.0, 4.0] for k in range(n)], a=8.0)


def _hcoh_dataset(seed):
    """H2O, OH, CH4 and H2 in boxes (8 A) with made-up reference energies and forces: three,
    two, two and one element(s)."""
    rng = np.random.default_rng(seed)
    base = [(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
            (["O", "H"], [[0, 0, 0], [0.98, 0, 0]]),
            (["C", "H", "H", "H", "H"], [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                                         [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]]),
            (["H", "H"], [[0, 0, 0], [0.74, 0, 0]])]
    out = []
    for sym, pos in base:
        atoms = _box(sym, np.array(pos, float) + 4.0, a=8.0)
        e = (-7.0 * sym.count("H") - 430.0 * sym.count("O") - 150.0 * sym.count("C")
             + rng.normal(0.5))
        atoms.calc = SinglePointCalculator(atoms, energy=e,
                                           forces=rng.normal(scale=0.5, size=(len(sym), 3)))
        out.append(atoms)
    return out


def _elements_before(dataset, opt_elem):
    """get_elements_from_dataset before M5b-3: the reference for the datasets it took."""
    tot = np.unique(np.array([np.unique(m.numbers).tolist() for m in dataset]).flatten())
    excl = [i for i in tot if i not in opt_elem]
    return (excl + opt_elem, excl,
            [chemical_symbols[i] for i in excl] + [chemical_symbols[n] for n in opt_elem],
            [chemical_symbols[i] for i in excl])


def test_elements_of_structures_with_different_numbers_of_elements():
    """H2O, OH, CH4, H2 and Cu: the code of before refused them (a ragged array).  Datasets it
    took give exactly what they gave, values and types (the dataset's elements as NumPy
    integers, opt_elem as given): they name files and enter caches keyed by repr()."""
    ragged = [_molecule(f) for f in ("OH2", "OH", "CH4", "H2", "Cu")]
    with pytest.raises(ValueError):
        _elements_before(ragged, [])
    numpy_int = type(_elements_before([_molecule("H2")], [])[0][0])     # np.int64 here
    loss = L.Loss(args_input=argparse.Namespace(optimization_option="energygeometry"),
                  cov_radius=1.0)
    same = [([_molecule("Cu"), _molecule("Cu2")], []),
            ([_molecule(f) for f in ("OH2", "OH", "CH4", "CH")], []),
            ([_molecule(f) for f in ("OH2", "OH", "CH4", "CH")], [8]),
            ([_molecule("NaN3"), _molecule("B2N2")], []),
            ([_molecule("NaN3"), _molecule("B2N2")], [29, 7]),
            ([], [8])]
    for get in (data.get_elements_from_dataset, loss.get_elements_from_dataset):
        nums, excl, syms, syms_excl = get(ragged, [])
        assert nums == excl == [1, 6, 8, 29] and syms == syms_excl == ["H", "C", "O", "Cu"]
        assert all(type(z) is numpy_int for z in nums)
        nums, excl, syms, syms_excl = get(ragged, [8])
        assert nums == [1, 6, 29, 8] and excl == [1, 6, 29]
        assert type(nums[-1]) is int and syms == ["H", "C", "Cu", "O"]
        for dataset, opt in same:
            new, old = get(dataset, opt), _elements_before(dataset, opt)
            assert repr(new) == repr(old)
            assert [type(v) for part in new for v in part] == [type(v) for part in old for v in part]


@linux
def test_shared_parameters_on_structures_with_different_numbers_of_elements(tmp_path, monkeypatch):
    """The dataset loss with one parameter set for H, C and O (the code of before M5b) on H2O,
    OH, CH4 and H2 (pydftb, hotcent replaced by the stored tables): before M5b-3 the first step
    stopped in Loss.set_parameters (the ragged array).  Now a loss on every structure, which
    utils.loss_grad.mu (also the code of before, in its dataset branch) reproduces bit for bit."""
    eng.configure(engine="pydftb")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(L.hotpy, "full_hotcent",
                        lambda self, superposition, opt_elem=(), label="ptbp":
                        _copy_skf(Path("."), SKF / "ptbp_hcocu", {"H", "C", "O"}))
    dft = _hcoh_dataset(11)
    assert TM.TypeModel.from_dataset(dft, shared=True).legacy
    args = _ho_args(tmp_path)
    args.E0s = {1: -13.6, 6: -150.0, 8: -430.0}
    loss = L.Loss(args_input=args, dft_input=dft, bolt_f11=[0.05] * len(dft), opt_elem=[],
                  path=tmp_path)
    found = {}
    search = L.Loss.Repulsion_optimization

    def record(self, conf, c0, c1):
        found["sigma"], found["mu"] = search(self, conf, c0, c1)
        return found["sigma"], found["mu"]

    monkeypatch.setattr(L.Loss, "Repulsion_optimization", record)
    x = [2.5, 4.5]
    mu = float(loss.Loss_EnergyGeometry(x))
    assert np.isfinite(mu) and loss.last_failed is False and mu == float(found["mu"])
    d = tmp_path / "results" / "opt_2.500_4.500"
    assert (d / "search_index.txt").read_text() == "0,1,2,3"
    monkeypatch.chdir(d)
    step = LG.StepData.from_files(loss, *x)
    assert [int(z) for z in step.numbers] == [1, 6, 8]
    assert float(LG.mu(step, float(found["sigma"]))).hex() == mu.hex()


def test_optimize_starts_on_structures_with_different_numbers_of_elements(tmp_path):
    """ptbp optimize on H2O, OH, CH4 and H2, parameters per element (the default) and
    --shared-parameters: before M5b-3 both stopped at the start (cli/run.py lists the dataset's
    elements for every run); now both get to the plan (--postprocess-only stops there)."""
    (tmp_path / "data").mkdir()
    dataset = tmp_path / "data" / "hco.xyz"
    ase.io.write(dataset, _hcoh_dataset(3))
    for name, extra in (("per_element", []), ("shared", ["--shared-parameters"])):
        out = subprocess.run(
            [sys.executable, str(ROOT / "parameterize.py"), "optimize", str(dataset),
             "--mode", "reaction", "--engine", "pydftb", "--ncores", "1", "--postprocess-only",
             "--output", str(tmp_path / name), *extra],
            cwd=tmp_path, capture_output=True, text=True, timeout=900)
        assert "result.pkl" in out.stderr and "inhomogeneous" not in out.stderr, out.stderr[-2000:]
        assert (tmp_path / name / "plan.json").exists()
        assert ("types (shared parameters)" if extra else "types (parameters per type)") in out.stdout


# ---- 3. --shared-parameters with a --multi_element of every element (M5b-4) -------------------
def test_shared_parameters_refuse_a_multi_element_of_every_element():
    """--shared-parameters with --multi_element listing every element of a dataset of several
    elements: nothing is left at PTBP's parameters, so the layout is that of before M5b, whose
    code reads --multi_element in its old sense and gives radii to the first listed element only
    (B of B, N, Na): refused, with the ways out.  Some of the elements, a dataset of one element,
    or either option alone work as before."""
    for listed in (["B", "N", "Na"], ["Na", "N", "B", "N"]):
        with pytest.raises(ValueError, match="cannot be combined") as err:
            TM.TypeModel.from_dataset(_bnna(), shared=True, multi_element=listed)
        assert "(B, N, Na)" in str(err.value) and "--shared-parameters alone" in str(err.value)
    some = TM.TypeModel.from_dataset(_bnna(), shared=True, multi_element=["B", "N"])
    assert not some.legacy and [s.labels for s in some.sets] == [("B", "N")]
    assert some.r0([3.0, 5.0])["Na"] == (get_PTBP()["Na"]["r0_w"], get_PTBP()["Na"]["r0_d"])
    assert TM.TypeModel.from_dataset(_bnna(), multi_element=["B", "N", "Na"]).n_sigma == 3
    assert TM.TypeModel.from_dataset(_bnna(), shared=True).legacy
    cu = [_box(["Cu", "Cu"], [[0, 0, 0], [2.5, 0, 0]])]
    assert TM.TypeModel.from_dataset(cu, shared=True, multi_element=["Cu"]).legacy


def test_optimize_refuses_shared_parameters_with_every_element_at_the_start(tmp_path):
    """ptbp optimize --shared-parameters --multi-element B N Na stops at the start with the
    message, before any file of the run (par.out, plan.json, results/); ptbp plan refuses it
    too."""
    images = ase.io.read(ROOT / "dft" / "nitrides.xyz", ":")
    (tmp_path / "data").mkdir()
    dataset = tmp_path / "data" / "bn_nan3.xyz"
    ase.io.write(dataset, images[0:11] + images[22:33])            # B2N2 and N3Na
    run = tmp_path / "run"
    opts = ["--mode", "energygeometry", "--engine", "pydftb", "--ncores", "1",
            "--shared-parameters", "--multi-element", "B", "N", "Na"]
    out = subprocess.run(
        [sys.executable, str(ROOT / "parameterize.py"), "optimize", str(dataset), *opts,
         "--output", str(run)], cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert out.returncode != 0
    assert "types: --shared-parameters cannot be combined" in out.stderr, out.stderr[-2000:]
    assert not any((run / name).exists() for name in ("par.out", "plan.json", "results"))
    out = subprocess.run(
        [sys.executable, str(ROOT / "parameterize.py"), "plan", str(dataset), *opts],
        cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert out.returncode == 2 and "ptbp plan: --shared-parameters cannot be combined" in out.stdout
