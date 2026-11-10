"""The A-series options of PTBP's loss and engines (pydftb/docs/design/A0-accuracy-programme.html,
section 8).

* Every default is the loss of before, bit for bit: the loss code of commit 66a6498 (before the
  options) and today's run side by side on the same step files (dataset, reaction,
  energygeometry, parameters per type): the same mu (``float.hex``), the same log lines, the same
  residual terms.
* Each option on toy steps: --energy_reduction mean/group, --force_weight, --energy_offset
  per-element (least squares, invariance under per-element E0 shifts, the terms and their sigma
  derivatives), the band term (W L_band inside the penalty; L_band with pydftb on a molecule),
  --mode bands (the optimisers' cost function, refused by the gradient optimisers),
  --sigma_finish (the finer sigma search on a step's files, the files written at the new
  sigma), the engines' DFT alignment (KSPACING grids, E(sigma -> 0), pydftb-only options), and
  the CLI (legacy flags, ptbp_run.yaml).

No hotcent, no DFTB+: the steps are synthetic files (energies and forces of a few Li/O
structures), the isolated atoms' DFTB energies come from the loss's cache.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.singlepoint import SinglePointCalculator

import utils.data as data
import utils.engines as eng
import utils.loss as L
from utils import loss_grad as LG

ROOT = Path(__file__).resolve().parents[1]
REF_COMMIT = "66a6498"            # the last commit before the A-series options
R0 = (3.1, 6.2)
E0_DFT = {3: -0.29, 8: -1.52}
E0_DFTB = {3: -1.71, 8: -4.42}
SKF_HCOCU = ROOT / "pydftb" / "tests" / "data" / "skf" / "ptbp_hcocu"
LI_SKF = ROOT / "utils" / "Li-Li.skf"


@pytest.fixture(autouse=True)
def _restore_engine():
    saved = eng.settings()
    yield
    eng._SETTINGS.clear()
    eng._SETTINGS.update(saved)
    eng.reset_alignment()


# ---- the loss code of before ----------------------------------------------------------
@pytest.fixture(scope="module")
def ref():
    """utils/loss.py and utils/loss_grad.py of REF_COMMIT, as modules of their own."""
    mods = {}
    for name in ("loss", "loss_grad"):
        try:
            out = subprocess.run(["git", "-C", str(ROOT), "show", f"{REF_COMMIT}:utils/{name}.py"],
                                 capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as err:
            pytest.skip(f"git is not available ({err})")
        if out.returncode:
            pytest.skip(f"git show {REF_COMMIT}:utils/{name}.py failed (a shallow clone?): "
                        f"{out.stderr.strip()[:200]}")
        mod = types.ModuleType(f"ptbp_ref_{name}_{REF_COMMIT}")
        mod.__file__ = str(ROOT / "utils" / f"{name}.py")
        sys.modules[mod.__name__] = mod                 # dataclasses look their module up
        exec(compile(out.stdout, f"<{REF_COMMIT}:utils/{name}.py>", "exec"), mod.__dict__)
        mods[name] = mod
    yield SimpleNamespace(L=mods["loss"], LG=mods["loss_grad"])
    for mod in mods.values():
        sys.modules.pop(mod.__name__, None)


# ---- synthetic steps --------------------------------------------------------------------
def _li_o_structures():
    """Eight small Li/O structures in four families, slightly displaced (deterministic)."""
    li2o = 0.5 * 4.62 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    raw = [(bulk("Li", "bcc", a=3.44, cubic=True), "li_bcc"),
           (bulk("Li", "bcc", a=3.30, cubic=True), "li_bcc"),
           (bulk("Li", "fcc", a=4.30), "li_fcc"),
           (Atoms("OLi2", scaled_positions=[(0, 0, 0), (.25, .25, .25), (.75, .75, .75)],
                  cell=li2o, pbc=True), "li2o"),
           (Atoms("OLi2", scaled_positions=[(0, 0, 0), (.25, .25, .25), (.75, .75, .75)],
                  cell=0.97 * li2o, pbc=True), "li2o"),
           (Atoms("O2", positions=[(4, 4, 4), (4, 4, 5.25)], cell=[8.0] * 3, pbc=True), "dimer"),
           (Atoms("LiO", positions=[(4, 4, 4), (4, 4, 5.7)], cell=[8.0] * 3, pbc=True), "dimer"),
           (Atoms("Li2", positions=[(4, 4, 4), (4, 4, 6.7)], cell=[8.0] * 3, pbc=True), "dimer")]
    out = []
    for k, (a, family) in enumerate(raw):
        a = a.copy()
        n = len(a)
        a.positions += 0.04 * np.sin(1.3 * np.arange(3 * n).reshape(n, 3) + k)
        a.info["family"] = family
        out.append(a)
    return out


def _li_eos_structures():
    """Two EOS groups of 11 (Li bcc, Li fcc)."""
    out = []
    for proto, a0 in ((bulk("Li", "bcc", a=3.44, cubic=True), 3.44), (bulk("Li", "fcc", a=4.33), 4.33)):
        for s in np.linspace(0.96, 1.04, 11):
            a = proto.copy()
            a.set_cell(proto.cell * s, scale_atoms=True)
            a.info["family"] = "li_bcc" if len(proto) == 2 else "li_fcc"
            out.append(a)
    return out


def _write_step(directory: Path, structures, eos_like: bool = False) -> list:
    """dft_final.xyz, dftb_band.xyz and search_index.txt of a BO step after its DFTB stage;
    returns the DFT structures (with their energies, the loss's dataset)."""
    directory.mkdir(parents=True, exist_ok=True)
    dft, band = [], []
    e_atom = {3: -1.90, 8: -4.95}
    for k, a in enumerate(structures):
        n = len(a)
        comp = sum(e_atom[int(z)] for z in a.numbers)
        idx = np.arange(3 * n).reshape(n, 3)
        if eos_like:
            s = (a.get_volume() / n / (20.35 if a.info["family"] == "li_bcc" else 20.30)) ** (1 / 3)
            e_d = comp + n * 1.5 * (s ** 3 - 1) ** 2 + (0.005 * n if a.info["family"] == "li_fcc" else 0)
            e_b = comp + n * (0.25 + 0.4 * (s ** 3 - 1) ** 2 + 0.02 * (s - 1))
            f_d = np.zeros((n, 3))
            f_b = np.zeros((n, 3))
        else:
            e_d = comp + 0.05 * n * np.sin(0.7 * k)
            e_b = comp - 2.6 * n + 0.04 * n * np.cos(1.1 * k)   # sigma* inside the range
            f_d = 0.3 * np.sin(0.9 * idx + k)
            f_b = f_d + 0.08 * np.cos(1.7 * idx + k)
        d = a.copy()
        d.calc = SinglePointCalculator(d, energy=float(e_d), forces=f_d)
        b = a.copy()
        b.calc = SinglePointCalculator(b, energy=float(e_b), forces=f_b)
        dft.append(d)
        band.append(b)
    ase.io.write(directory / "dft_final.xyz", dft)
    ase.io.write(directory / "dftb_band.xyz", band)
    (directory / "search_index.txt").write_text(",".join(str(i) for i in range(len(structures))))
    return ase.io.read(directory / "dft_final.xyz", ":")


def _args(tmp_path, mode, eos, tag="", **kw):
    return argparse.Namespace(eos_points=eos, kpt_density=5.0, multi_element=None,
                              optimization_option=mode, log_file=str(tmp_path / f"log{tag}.out"),
                              para_file="par.out", E0s=dict(E0_DFT), xc="GGA_X_PBE+GGA_C_PBE",
                              superposition="density", results_dir="results", **kw)


def _loss(module, tmp_path, mode, structures, eos=1, tag="", types_model=None, **kw):
    if mode == "energygeometry":
        _, bolt = data.get_bolt_factor(structures[eos // 2::eos], "boltzmann", 3000)
    else:
        bolt = [1.0] * len(structures)
    loss = module.Loss(args_input=_args(tmp_path, mode, eos, tag, **kw), dft_input=structures,
                       bolt_f11=list(bolt), opt_elem=[], types=types_model)
    zs = frozenset(int(z) for a in structures for z in a.numbers)
    loss._E0_dftb_cache = {zs: {z: E0_DFTB[z] for z in zs}}
    return loss


def _hex(x):
    return float(x).hex()


# ---- 1. the defaults are the loss of before, bit for bit ---------------------------------
@pytest.mark.parametrize("mode", ["dataset", "reaction"])
def test_defaults_are_the_dataset_loss_of_before(ref, tmp_path, monkeypatch, mode):
    structures = _li_o_structures()
    step_dir = tmp_path / "step"
    structures = _write_step(step_dir, structures)
    monkeypatch.chdir(step_dir)
    old = _loss(ref.L, tmp_path, mode, structures, tag="_old")
    new = _loss(L, tmp_path, mode, structures, tag="_new")
    for sigma in (0.12, 0.33, 0.47, 0.61):
        m_old = old.unparallel_repulsion(np.array([sigma]), *R0, False)
        m_new = new.unparallel_repulsion(np.array([sigma]), *R0, False)
        assert _hex(m_new) == _hex(m_old)
        s_old = ref.LG.StepData.from_files(old, *R0)
        s_new = LG.StepData.from_files(new, *R0)
        assert _hex(LG.mu(s_new, sigma)) == _hex(m_old)
        assert _hex(ref.LG.mu(s_old, sigma)) == _hex(m_old)
        t_old, t_new = ref.LG.terms(s_old, sigma), LG.terms(s_new, sigma)
        for name in ("w", "r", "drds", "a_e", "a_e0"):
            assert np.array_equal(getattr(t_new, name), getattr(t_old, name)), name
        assert t_new.kind == t_old.kind and t_new.force == t_old.force
    assert (tmp_path / "log_new.out").read_text() == (tmp_path / "log_old.out").read_text()


def test_defaults_are_the_energygeometry_loss_of_before(ref, tmp_path, monkeypatch):
    structures = _li_eos_structures()
    step_dir = tmp_path / "step"
    structures = _write_step(step_dir, structures, eos_like=True)
    monkeypatch.chdir(step_dir)
    old = _loss(ref.L, tmp_path, "energygeometry", structures, eos=11, tag="_old")
    new = _loss(L, tmp_path, "energygeometry", structures, eos=11, tag="_new")
    for sigma in (0.15, 0.33, 0.47):
        m_old = old.parallel_repulsion(np.array([sigma]), *R0, False)
        m_new = new.parallel_repulsion(np.array([sigma]), *R0, False)
        assert _hex(m_new) == _hex(m_old)
        s_new = LG.StepData.from_files(new, *R0)
        assert _hex(LG.mu(s_new, sigma)) == _hex(m_old)
        if importlib.util.find_spec("torch") is not None:     # the EOS sensitivity uses torch
            t_old = ref.LG.terms(ref.LG.StepData.from_files(old, *R0), sigma)
            t_new = LG.terms(s_new, sigma)
            for name in ("w", "r", "drds", "a_e"):
                assert np.array_equal(getattr(t_new, name), getattr(t_old, name)), name
    assert (tmp_path / "log_new.out").read_text() == (tmp_path / "log_old.out").read_text()


def test_defaults_are_the_typed_loss_of_before(ref, tmp_path, monkeypatch):
    """Parameters per type (one set per element, M5b): Repulsion_optimization_types's loss."""
    from utils.typemodel import TypeModel
    structures = _li_o_structures()
    step_dir = tmp_path / "step"
    structures = _write_step(step_dir, structures)
    monkeypatch.chdir(step_dir)
    model = TypeModel.from_dataset(structures)
    assert not model.legacy and model.n_sigma == 2
    x = [3.1, 6.2, 2.6, 5.0]
    old = _loss(ref.L, tmp_path, "dataset", structures, types_model=model)
    new = _loss(L, tmp_path, "dataset", structures, types_model=model)
    s_old = ref.LG.StepData.from_files(old, x=x)
    s_new = LG.StepData.from_files(new, x=x)
    for svec in ([0.3, 0.45], [0.12, 0.7], [0.6, 0.2]):
        assert _hex(LG.mu(s_new, svec)) == _hex(ref.LG.mu(s_old, svec))
        t_old, t_new = ref.LG.terms_typed(s_old, svec), LG.terms_typed(s_new, svec)
        for name in ("w", "r", "drds", "a_e", "a_e0"):
            assert np.array_equal(getattr(t_new, name), getattr(t_old, name)), name


def test_defaults_resolve_to_the_old_constants():
    args = argparse.Namespace()
    reduction, offset, c = L.dataset_options(args)
    assert (reduction, offset) == ("sum", "none")
    assert c == LG.DS_C == 0.1 and L.band_weight(args) == 0.0
    assert eng.alignment() == eng._ALIGN_DEFAULTS and not eng.aligned()


# ---- 2. the dataset options -------------------------------------------------------------
@pytest.fixture
def ds_step(tmp_path, monkeypatch):
    structures = _li_o_structures()
    step_dir = tmp_path / "step"
    structures = _write_step(step_dir, structures)
    monkeypatch.chdir(step_dir)
    return structures, step_dir


def _mu_parts(loss, sigma):
    parts = {}
    mu = LG.mu(LG.StepData.from_files(loss, *R0), sigma, parts)
    return mu, parts


def test_energy_reduction_and_force_weight(tmp_path, ds_step):
    structures, _ = ds_step
    sigma = 0.45
    base = _loss(L, tmp_path, "dataset", structures)
    _, p_sum = _mu_parts(base, sigma)
    diffs = np.asarray(p_sum["diffs"])
    assert p_sum["mu_E"] == pytest.approx(diffs.sum(), rel=1e-15)
    families = [a.info["family"] for a in structures]
    expected = {"mean": diffs.mean(),
                "group": np.mean([diffs[[f == g for f in families]].mean()
                                  for g in dict.fromkeys(families)])}
    for reduction, mu_e in expected.items():
        loss = _loss(L, tmp_path, "dataset", structures, tag=reduction, energy_reduction=reduction,
                     force_weight=0.3)
        mu, parts = _mu_parts(loss, sigma)
        assert parts["mu_E"] == pytest.approx(mu_e, rel=1e-12)
        assert np.allclose(parts["diffs"], diffs, rtol=0, atol=1e-15)       # no offsets: |r|
        assert mu == pytest.approx((0.7 * parts["mu_E"] + 0.3 * parts["mu_F"]) * parts["penalty"],
                                   rel=1e-13)
        # Loss computes the same loss as loss_grad.mu, bit for bit
        assert _hex(loss.unparallel_repulsion(np.array([sigma]), *R0, False)) == _hex(mu)
        assert loss.no_success_failure_value() == L.NO_SUCCESS_FAILURE    # a mean, not a sum
    assert base.no_success_failure_value() == L.NO_SUCCESS_FAILURE * len(structures)


def test_group_of_a_structure():
    a = Atoms("Li2O", positions=np.zeros((3, 3)))
    assert L.structure_group(a) == "Li2O"
    a.info["config_type"] = "bulk"
    assert L.structure_group(a) == "bulk"
    a.info["family"] = "li2o_eos"
    assert L.structure_group(a) == "li2o_eos"
    w = L.reduction_weights([a, a, Atoms("O2")], "group")
    assert np.allclose(w, [0.25, 0.25, 0.5]) and w.sum() == pytest.approx(1.0)


def test_per_element_offsets_are_least_squares(tmp_path, ds_step):
    structures, _ = ds_step
    sigma = 0.45
    base = _loss(L, tmp_path, "dataset", structures)
    _, p = _mu_parts(base, sigma)
    loss = _loss(L, tmp_path, "dataset", structures, tag="off", energy_offset="per-element")
    mu, parts = _mu_parts(loss, sigma)
    # the raw residuals with sign, from the step (as the loss computes them)
    step = LG.StepData.from_files(base, *R0)
    t = LG.terms(step, sigma)
    n = len(structures)
    r = t.r[:n]
    a, zs = L.offset_design(structures)
    assert zs == [3, 8] and np.allclose(a.sum(1), 1.0)
    c, *_ = np.linalg.lstsq(a, r, rcond=None)
    assert np.allclose(parts["diffs"], np.abs(r - a @ c), rtol=0, atol=1e-12)
    offs = L.element_offsets(structures, r, np.ones(n))
    assert np.allclose([offs[3], offs[8]], c, atol=1e-12)
    assert parts["mu_E"] == pytest.approx(np.abs(r - a @ c).sum(), rel=1e-12)
    assert parts["mu_E"] < p["mu_E"]
    assert _hex(loss.unparallel_repulsion(np.array([sigma]), *R0, False)) == _hex(mu)


def test_per_element_offsets_absorb_the_isolated_atoms(tmp_path, ds_step):
    """A shift of the DFTB isolated-atom energies changes the absolute errors, not the loss
    with per-element offsets."""
    structures, _ = ds_step
    sigma = 0.4
    results = {}
    for name, offset in (("abs", "none"), ("off", "per-element")):
        vals = []
        for shift in (0.0, 1.0):
            loss = _loss(L, tmp_path, "dataset", structures, tag=f"{name}{shift}",
                         energy_offset=offset)
            zs = frozenset({3, 8})
            loss._E0_dftb_cache = {zs: {3: E0_DFTB[3] + 0.3 * shift, 8: E0_DFTB[8] - 0.7 * shift}}
            vals.append(LG.mu(LG.StepData.from_files(loss, *R0), sigma))
        results[name] = vals
    assert abs(results["abs"][1] - results["abs"][0]) > 1e-2
    assert results["off"][1] == pytest.approx(results["off"][0], rel=1e-10)


@pytest.mark.parametrize("opts", [{"energy_offset": "per-element"},
                                  {"energy_reduction": "group", "energy_offset": "per-element",
                                   "force_weight": 0.25},
                                  {"energy_reduction": "mean"}])
def test_option_terms_give_the_loss_and_its_sigma_derivative(tmp_path, ds_step, opts):
    structures, _ = ds_step
    loss = _loss(L, tmp_path, "dataset", structures, tag="t", **opts)
    step = LG.StepData.from_files(loss, *R0)
    for sigma in (0.42, 0.55):
        t = LG.terms(step, sigma)
        m = LG.mu(step, sigma)
        assert abs(t.value() - m) <= 1e-12 * abs(m)
        h = 1e-5
        fd = (LG.terms(step, sigma + h).r - LG.terms(step, sigma - h).r) / (2 * h)
        scale = max(np.abs(fd).max(), 1e-6)
        assert np.abs(t.drds - fd).max() <= 1e-4 * scale


def test_option_terms_with_types(tmp_path, ds_step):
    from utils.typemodel import TypeModel
    structures, _ = ds_step
    model = TypeModel.from_dataset(structures)
    loss = _loss(L, tmp_path, "dataset", structures, types_model=model,
                 energy_offset="per-element", energy_reduction="group")
    step = LG.StepData.from_files(loss, x=[3.1, 6.2, 2.6, 5.0])
    svec = [0.45, 0.5]
    t = LG.terms_typed(step, svec)
    m = LG.mu(step, svec)
    assert abs(t.value() - m) <= 1e-12 * abs(m)
    h = 1e-5
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        fd = (LG.terms_typed(step, np.add(svec, e)).r - LG.terms_typed(step, np.subtract(svec, e)).r) / (2 * h)
        scale = max(np.abs(fd).max(), 1e-6)
        assert np.abs(t.drds[:, k] - fd).max() <= 1e-4 * scale


def test_bad_options_are_refused():
    with pytest.raises(ValueError):
        L.dataset_options(argparse.Namespace(energy_reduction="median"))
    with pytest.raises(ValueError):
        L.dataset_options(argparse.Namespace(energy_offset="per-structure"))
    with pytest.raises(ValueError):
        L.dataset_options(argparse.Namespace(force_weight=1.5))
    with pytest.raises(ValueError):
        L.band_weight(argparse.Namespace(band_weight=-1.0))


# ---- 3. the band term ---------------------------------------------------------------------
def test_band_term_enters_inside_the_penalty(tmp_path, ds_step):
    structures, _ = ds_step
    sigma = 0.45
    loss = _loss(L, tmp_path, "dataset", structures, band_weight=0.5)
    _, p0 = _mu_parts(loss, sigma)
    loss._band_term = 0.37
    mu, p = _mu_parts(loss, sigma)
    assert mu == pytest.approx((0.9 * p["mu_E"] + 0.1 * p["mu_F"] + 0.37) * p["penalty"], rel=1e-13)
    assert _hex(loss.unparallel_repulsion(np.array([sigma]), *R0, False)) == _hex(mu)


def test_band_term_in_energygeometry(tmp_path, monkeypatch):
    structures = _li_eos_structures()
    step_dir = tmp_path / "step"
    structures = _write_step(step_dir, structures, eos_like=True)
    monkeypatch.chdir(step_dir)
    loss = _loss(L, tmp_path, "energygeometry", structures, eos=11, band_weight=1.0)
    m0 = LG.mu(LG.StepData.from_files(loss, *R0), 0.3, parts := {})
    loss._band_term = 0.21
    m1 = LG.mu(LG.StepData.from_files(loss, *R0), 0.3)
    assert m1 - m0 == pytest.approx(0.21 * parts["penalty"], rel=1e-12)
    assert _hex(loss.parallel_repulsion(np.array([0.3]), *R0, False)) == _hex(m1)


def _h2o_skf(tmp_path) -> Path:
    skdir = tmp_path / "skf"
    skdir.mkdir()
    for pair in ("H-H", "H-O", "O-H", "O-O"):
        shutil.copy(SKF_HCOCU / f"{pair}.skf.gz", skdir / f"{pair}.skf.gz")
    return skdir


def _h2o_ref(skdir, scale=1.0, shift=0.5, name="h2o"):
    from pydftb.bandloss import dftb_bands
    from pydftb.bandref import BandRef
    h2o = molecule("H2O")
    h2o.center(vacuum=6.0)
    probe = BandRef(name=name, kind="molecule", kpts=[[0, 0, 0]], kweights=[1.0],
                    energies=np.zeros((1, 1, 6))).set_atoms(h2o)
    at = h2o.copy()
    calc = L.band_calculator(str(skdir), at, probe, 5.0)
    at.calc = calc
    at.get_potential_energy()
    b = dftb_bands(calc, at, probe)
    return BandRef(name=name, kind="molecule", kpts=[[0, 0, 0]], kweights=[1.0],
                   energies=b.energies * scale + shift,
                   occupations=np.array([[[1, 1, 1, 1, 0, 0]]], float),
                   meta={"group": "molecules"}).set_atoms(h2o)


def test_band_loss_of_its_own_bands_is_zero(tmp_path, monkeypatch):
    pytest.importorskip("pydftb")
    skdir = _h2o_skf(tmp_path)
    refs = tmp_path / "refs"
    refs.mkdir()
    _h2o_ref(skdir).to_json(str(refs / "h2o.json"))
    args = argparse.Namespace(band_refs=[str(refs / "*.json")], band_weight=2.0,
                              band_settings=None, kpt_density=5.0)
    loss = L.Loss(args_input=args, dft_input=[molecule("H2O")], bolt_f11=[1.0], opt_elem=[])
    monkeypatch.chdir(skdir)
    value = loss.band_loss_value(".")
    assert value == pytest.approx(0.0, abs=1e-8)
    import json
    rec = json.loads((skdir / "band_loss.json").read_text())
    assert rec["per_structure"][0]["shift"] == pytest.approx(-0.5, abs=1e-8)
    assert loss.band_term() == pytest.approx(0.0, abs=1e-7)
    # a stretched reference: a positive loss, W times it in the term
    _h2o_ref(skdir, scale=1.08).to_json(str(refs / "h2o.json"))
    loss._band_refs = None
    value = loss.band_loss_value(".")
    assert value > 0.05
    assert loss.band_term() == pytest.approx(2.0 * value, rel=1e-12)
    # settings from JSON
    loss.args_input.band_settings = '{"gap_weight": 0.0, "levels": [1]}'
    assert loss.band_settings().gap_weight == 0.0


def test_a_failed_band_reference_fails_the_point(tmp_path, monkeypatch):
    from pydftb.bandref import BandRef
    skdir = _h2o_skf(tmp_path)
    li = Atoms("Li2", positions=[(0, 0, 0), (0, 0, 2.7)])
    li.center(vacuum=5.0)
    bad = BandRef(name="li2", kind="molecule", kpts=[[0, 0, 0]], kweights=[1.0],
                  energies=np.zeros((1, 1, 2))).set_atoms(li)
    path = tmp_path / "li2.json"
    bad.to_json(str(path))
    args = argparse.Namespace(band_refs=[str(path)], band_weight=1.0, kpt_density=5.0)
    loss = L.Loss(args_input=args, dft_input=[li], bolt_f11=[1.0], opt_elem=[])
    monkeypatch.chdir(skdir)
    with pytest.raises(L.LossFailure):
        loss.band_loss_value(".")


def test_band_kpoints_rule():
    from pydftb.bandref import BandRef
    li = bulk("Li", "bcc", a=3.44, cubic=True)
    ref = BandRef(name="li", kind="path", kpts=[[0, 0, 0]], kweights=[1.0],
                  energies=np.zeros((1, 1, 2)))
    assert L.band_kpoints(li, ref, 5.0) == tuple(
        __import__("utils.calculator", fromlist=["x"]).monkhorstpack2kptdensity(li, 5.0)[1])
    ref.meta = {"kspacing": 0.15}
    assert L.band_kpoints(li, ref, 5.0) == {"size": [13, 13, 13], "gamma": True}
    ref.meta = {"kspacing": "gamma"}
    assert L.band_kpoints(li, ref, 5.0) is None
    ref.meta = {"kpts": "4x4x1"}
    assert L.band_kpoints(li, ref, 5.0) == {"size": [4, 4, 1], "gamma": True}
    ref.meta = {}
    eng.set_alignment(kspacing=0.25)
    assert L.band_kpoints(li, ref, 5.0) == {"size": [8, 8, 8], "gamma": True}
    assert L.band_kpoints(molecule("H2O"), ref, 5.0) is None


# ---- 4. --mode bands ----------------------------------------------------------------------
def test_mode_bands_routes_the_optimisers_to_the_band_loss(tmp_path):
    from utils.typemodel import TypeModel
    structures = _li_o_structures()
    loss = _loss(L, tmp_path, "bands", structures, band_refs=["x.json"])
    dims, lo, hi, cost = loss._bo_dimensions_and_cost(loss.cov_radius)
    assert cost.__func__ is L.Loss.Loss_Bands and [d.name for d in dims] == ["r0_w", "r0_d"]
    typed = _loss(L, tmp_path, "bands", structures, types_model=TypeModel.from_dataset(structures),
                  band_refs=["x.json"])
    dims, lo, hi, cost = typed._bo_dimensions_and_cost(typed.cov_radius)
    assert cost.__func__ is L.Loss.Loss_Bands and len(dims) == 4
    eg = _loss(L, tmp_path, "dataset", structures)
    assert eg._bo_dimensions_and_cost(eg.cov_radius)[3].__func__ is L.Loss.Loss_EnergyGeometry
    assert loss.no_success_failure_value() == L.NO_SUCCESS_FAILURE


def test_gradient_optimisers_refuse_the_band_loss(tmp_path):
    structures = _li_o_structures()
    for mode, kw in (("bands", {}), ("dataset", {"band_weight": 1.0})):
        loss = _loss(L, tmp_path, mode, structures, band_refs=["x.json"], optimizer="gradient", **kw)
        with pytest.raises(SystemExit):
            loss.optimize(n_calls=1)


# ---- 5. --sigma_finish ----------------------------------------------------------------------
def test_sigma_finish_refines_sigma_on_the_step_files(tmp_path, ds_step):
    structures, step_dir = ds_step
    loss = _loss(L, tmp_path, "dataset", structures)
    loss.get_atomic_energy = lambda numbers, ref_dataset=None: dict(E0_DFTB)   # no engine
    step = LG.StepData.from_files(loss, *R0, directory=str(step_dir))
    # the optimiser's point: the best of a coarse grid (PTBP's search writes SKF files)
    sigma_hat, mu_hat = min(((float(v), float(LG.mu(step, float(v))))
                             for v in np.linspace(0.1, 0.8, 8)), key=lambda t: t[1])
    (step_dir / "par.out").write_text(
        "#r0_w,r0_d,c_rep,kxc,cov_radius,mu\n"
        f"{R0[0]:15.8f}, {R0[1]:15.8f}, {sigma_hat:15.8f}, {0.0:20.8e}, "
        f"{loss.cov_radius:15.7f}, {mu_hat:25.11f}\n")
    out = loss.sigma_finish(list(R0), step_dir, mu_before=mu_hat, write=False)
    assert out["mu"] <= mu_hat * (1 + 1e-12)
    assert 0.1 <= out["sigma"][0] <= 0.8
    assert out["sigma_before"] == [pytest.approx(sigma_hat, abs=1e-8)]
    assert out["mu_before"] == mu_hat and out["band_term"] == 0.0
    assert out["mu"] < mu_hat                      # the coarse grid is not the minimum
    assert LG.mu(step, out["sigma"][0]) == out["mu"]
    assert L._par_out_point(str(step_dir / "par.out"), list(R0))[0] == [pytest.approx(sigma_hat, abs=1e-8)]


def test_sigma_finish_writes_the_files_at_the_new_sigma(tmp_path, monkeypatch):
    """A Li step with its SKF file: the repulsive spline and dftb_full.xyz at sigma_finish."""
    structures = [a for a in _li_o_structures() if set(a.numbers) == {3}]
    step_dir = tmp_path / "step"
    structures = _write_step(step_dir, structures)
    shutil.copy(LI_SKF, step_dir / "Li-Li.skf")
    monkeypatch.chdir(step_dir)
    loss = _loss(L, tmp_path, "dataset", structures)
    loss.get_atomic_energy = lambda numbers, ref_dataset=None: {3: E0_DFTB[3]}
    out = loss.sigma_finish(list(R0), step_dir, mu_before=1e9)
    assert out["improved"] and out.get("written")
    text = (step_dir / "Li-Li.skf").read_text()
    assert text.count("Spline") == 1
    full = ase.io.read(step_dir / "dftb_full.xyz", ":")
    pr = L._physrep([3], [out["sigma"][0]], [0.0])
    e_rep, _ = pr.evaluate_pbc(structures[0])
    assert full[0].info["rep_energy"] == pytest.approx(e_rep, rel=1e-12)


# ---- 6. the engines' DFT alignment ---------------------------------------------------------
def test_kspacing_grid_is_vasps_rule():
    li = bulk("Li", "bcc", a=3.44, cubic=True)
    assert eng.kspacing_grid(li, 0.15) == [13, 13, 13]          # 2 pi / 3.44 / 0.15 = 12.18
    slab = li.repeat((1, 1, 3))
    slab.pbc = [True, True, False]
    assert eng.kspacing_grid(slab, 0.25) == [8, 8, 1]
    try:
        from utils.accuracy import kpoints_for
    except Exception:                                           # noqa: BLE001
        return
    for cell in (li, bulk("Li", "hcp", a=3.1, c=5.0), bulk("Cu", "fcc", a=3.6)):
        assert eng.kspacing_grid(cell, 0.15) == kpoints_for(cell, 0.15)["size"]


def test_alignment_changes_the_calculators_only_when_set(tmp_path):
    shutil.copy(LI_SKF, tmp_path / "Li-Li.skf")
    li = bulk("Li", "bcc", a=3.44, cubic=True)
    args = {"calc_type": "static", "poly_rep": "Yes", "SCC": "Yes",
            "slater_p": str(tmp_path) + "/", "kpts": [4, 4, 4]}
    eng.configure(engine="pydftb")
    plain = eng.make_calculator(li, args)
    assert plain.kpts_spec == (4, 4, 4) and plain.time_reversal is True
    assert type(plain).__name__ == "PyDFTB"
    eng.set_alignment(kspacing=0.15)
    aligned = eng.make_calculator(li, args)
    assert aligned.kpts_spec == {"size": [13, 13, 13], "gamma": True}
    assert aligned.time_reversal is None
    fixed = eng.make_calculator(li, dict(args, kpts_fixed=True, kpts=[1, 1, 1]))
    assert fixed.kpts_spec == (1, 1, 1)
    eng.reset_alignment()
    again = eng.make_calculator(li, args)
    assert again.kpts_spec == (4, 4, 4) and again.time_reversal is True


def test_energy_kind_sigma0_is_total_minus_ts_over_two(tmp_path):
    shutil.copy(LI_SKF, tmp_path / "Li-Li.skf")
    li = bulk("Li", "bcc", a=3.44).repeat((3, 1, 1))     # three electrons: a metal
    li.rattle(0.05, seed=2)
    args = {"calc_type": "static", "poly_rep": "Yes", "SCC": "Yes",
            "slater_p": str(tmp_path) + "/", "kpts": [2, 6, 6]}
    eng.configure(engine="pydftb", smearing="gaussian", smearing_temperature=0.05 / 8.617333262e-5)
    a = li.copy()
    a.calc = eng.make_calculator(a, args)
    e_total, f_total = a.get_potential_energy(), a.get_forces()
    raw = a.calc.raw
    eng.set_alignment(energy_kind="sigma0")
    b = li.copy()
    b.calc = eng.make_calculator(b, args)
    e0 = b.get_potential_energy()
    from ase import units
    assert abs(raw.ts) > 1e-8                                   # a metal: TS != 0
    assert e0 == pytest.approx(raw.e_zero * units.Hartree, abs=1e-9)
    assert e0 == pytest.approx(e_total - 0.5 * raw.ts * units.Hartree, abs=1e-9)
    assert np.allclose(b.get_forces(), f_total, atol=1e-10)


def test_isolated_atoms_keep_the_gamma_point_with_kspacing(monkeypatch):
    seen = []

    def fake_calculator(atoms, dftb_args):
        seen.append(dict(dftb_args))
        return SinglePointCalculator(atoms, energy=-1.0)

    monkeypatch.setattr(L, "dftbplus_calculator", fake_calculator)
    loss = L.Loss(args_input=argparse.Namespace(), dft_input=[Atoms("Li")], bolt_f11=[1.0],
                  opt_elem=[], cov_radius=1.0)
    loss.get_atomic_energy([3])
    assert seen[-1] == {"kpts": [1, 1, 1], "poly_rep": "No", "SCC": "Yes", "slater_p": "./",
                        "calc_type": "static"}                  # as before
    eng.set_alignment(kspacing=0.15)
    loss._E0_dftb_cache = {}
    loss.get_atomic_energy([3])
    assert seen[-1]["kpts_fixed"] is True and seen[-1]["kpts"] == [1, 1, 1]


def test_alignment_options_are_checked():
    with pytest.raises(ValueError):
        eng.set_alignment(third_order="full")                    # no hubbard_derivs
    eng.reset_alignment()
    with pytest.raises(ValueError):
        eng.set_alignment(kspacing=-0.1)
    eng.reset_alignment()
    with pytest.raises(ValueError):
        eng.set_alignment(energy_kind="free")
    eng.reset_alignment()
    eng.set_alignment(third_order="full", hubbard_derivs={"Li": -0.04}, hubbard={"Li": 0.17})
    assert eng.pydftb_alignment_kwargs() == {"third_order": "full",
                                             "hubbard_derivs": {"Li": -0.04},
                                             "hubbard": {"Li": 0.17}}
    with pytest.raises(ValueError):
        eng.check_alignment_engine("dftbplus")
    eng.check_alignment_engine("pydftb")
    eng.reset_alignment()
    eng.set_alignment(kspacing=0.15)
    eng.check_alignment_engine("dftbplus")                      # both engines


# ---- 7. the CLI -----------------------------------------------------------------------------
def _optimize_args(argv):
    import cli.cmd_optimize as co
    parser = argparse.ArgumentParser()
    co.add_parser(parser.add_subparsers())
    return co, parser.parse_args(["optimize", "data.xyz", *argv])


def test_cli_defaults_add_no_legacy_flags():
    co, args = _optimize_args([])
    assert co._a_series_argv(args) == []


def test_cli_options_reach_the_legacy_parser(tmp_path, monkeypatch):
    co, args = _optimize_args(
        ["--energy-reduction", "group", "--force-weight", "0.2", "--energy-offset", "per-element",
         "--band-weight", "0.5", "--band-refs", "bands/*.json", "--fixed-r0", "3.1", "6.2",
         "--sigma-finish", "--kspacing", "0.15", "--energy-kind", "sigma0", "--third-order",
         "full", "--hubbard-derivs", '{"Li": -0.04}', "--mode", "bands"])
    flags = co._a_series_argv(args)
    from cli.arg_parser import build_default_arg_parser
    monkeypatch.setattr(sys, "argv", ["run.py", "--optimization_option", "bands", *flags])
    legacy = build_default_arg_parser()
    assert legacy.energy_reduction == "group" and legacy.force_weight == 0.2
    assert legacy.energy_offset == "per-element" and legacy.band_weight == 0.5
    assert legacy.band_refs == [str(Path("bands/*.json").resolve())]
    assert legacy.fixed_r0 == [3.1, 6.2] and legacy.sigma_finish is True
    assert legacy.kspacing == 0.15 and legacy.energy_kind == "sigma0"
    assert legacy.third_order == "full" and legacy.hubbard_derivs == '{"Li": -0.04}'
    assert L.dataset_options(legacy) == ("group", "per-element", 0.2)


def test_ptbp_run_yaml_has_no_a_series_defaults(tmp_path):
    import yaml
    from cli import config
    _, args = _optimize_args([])
    config.write_effective(args, tmp_path)
    written = yaml.safe_load((tmp_path / "ptbp_run.yaml").read_text())
    assert not set(written) & set(config.A_SERIES_DEFAULTS)
    _, args = _optimize_args(["--energy-reduction", "mean", "--band-weight", "0.3"])
    config.write_effective(args, tmp_path)
    written = yaml.safe_load((tmp_path / "ptbp_run.yaml").read_text())
    assert written["energy_reduction"] == "mean" and written["band_weight"] == 0.3
    assert "energy_offset" not in written
