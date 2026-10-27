"""The sigma_rep search of a BO step (M3b): equation-of-state fits in memory with the same
files, the step's files parsed once, the repulsion evaluated without a process pool.  M5b-4:
L-BFGS-B's sigma* is paired with the loss at sigma* (scipy can return the loss of another
evaluation), on the code of one parameter set and with parameters per type."""
import argparse
import gzip
import hashlib
import shutil
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

import utils.data as data
import utils.engines as eng
import utils.loss as L
from utils import dftb_stage
from utils import loss_grad as LG
from utils import typemodel as TM

ROOT = Path(__file__).resolve().parents[1]
CU_SKF_GZ = ROOT / "pydftb" / "tests" / "data" / "skf" / "ptbp_hcocu" / "Cu-Cu.skf.gz"


def _digests(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}


@pytest.fixture(scope="module")
def images():
    return ase.io.read(ROOT / "dft" / "dft.xyz", ":22")             # two EOS phases


def test_eos_fits_in_memory_write_the_same_files(tmp_path, monkeypatch, images):
    loss = L.Loss()
    (tmp_path / "files").mkdir()
    (tmp_path / "memory").mkdir()
    monkeypatch.chdir(tmp_path / "files")
    ref = loss.eos_atoms(images, 11, "dft")
    monkeypatch.chdir(tmp_path / "memory")
    files = {}
    out = loss.eos_atoms(images, 11, "dft", files=files)
    assert not any((tmp_path / "memory").iterdir())                 # recorded, not written
    loss.write_eos_files(files)
    assert repr(out) == repr(ref)
    assert _digests(tmp_path / "memory") == _digests(tmp_path / "files")


def test_a_failed_fit_gives_the_fallback_every_time(monkeypatch, images):
    class Broken:
        def __init__(self, *a, **k):
            pass

        def fit(self, warn=False):
            raise RuntimeError("no minimum")

    monkeypatch.setattr(L, "EOS", Broken)
    monkeypatch.setattr(L, "_EOS_FITS", type(L._EOS_FITS)())
    for _ in range(2):
        with pytest.raises(RuntimeError, match="no minimum"):
            L._eos_fit(images[:11])
        out = L.Loss().eos_atoms(images[:11], 11, "dftb", files={})
        assert out["eos_Cu_0"]["dftb_volume"] == 1000


def test_step_files_are_parsed_once_per_content(tmp_path, images):
    path = tmp_path / "step.xyz"
    ase.io.write(path, images[:3])
    first = L._read_step_file(path)
    assert L._read_step_file(path)[0] is first[0]                   # cached
    ase.io.write(path, images[3:5])
    again = L._read_step_file(path)
    assert len(again) == 2 and again[0] is not first[0]             # new content, new parse


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork backend")
def test_repulsion_without_pool_and_bitwise_like_the_reference(tmp_path, monkeypatch, images):
    """A two-group step: the loss of the vectorised evaluate_pbc equals the original loop's."""
    with gzip.open(CU_SKF_GZ, "rb") as fi, open(tmp_path / "Cu-Cu.skf", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    monkeypatch.chdir(tmp_path)
    saved = eng.settings()
    eng.configure(engine="pydftb")
    try:
        _, bolt = data.get_bolt_factor(images[5::11], "boltzmann", 3000)
        args = argparse.Namespace(eos_points=11, kpt_density=5.0, multi_element=None,
                                  optimization_option="energygeometry",
                                  log_file=str(tmp_path / "log.out"))
        loss = L.Loss(args_input=args, dft_input=images, bolt_f11=bolt, opt_elem=[])
        dftb_stage.write_outputs(*dftb_stage.run_dataset(images, bolt, 11, 5.0))
    finally:
        eng._SETTINGS.update(saved)

    def no_pool(*a, **k):
        raise AssertionError("parallel_repulsion must not start a process pool")

    monkeypatch.setattr(L.mp, "Pool", no_pool)
    mu = loss.parallel_repulsion(np.array([0.33]), 2.749, 12.036, True)
    skf = (tmp_path / "Cu-Cu.skf").read_bytes()
    monkeypatch.setenv("PTBP_PHYSREP_REFERENCE", "1")
    assert loss.parallel_repulsion(np.array([0.33]), 2.749, 12.036, True).hex() == mu.hex()
    assert (tmp_path / "Cu-Cu.skf").read_bytes() == skf
    lines = (tmp_path / "log.out").read_text().splitlines()
    assert len(lines) == 2 and lines[0] == lines[1]


# ---- L-BFGS-B's sigma* and the loss at sigma* (M5b-4) ------------------------------------------
X0 = 0.5926215820312503              # sigma* of the M5b-3 report (section 8)
linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork backend")


def _jumpy(s):
    """A loss that jumps by 73.1 when sigma moves by less than 1e-9, as an EOS fit that switches
    between two solutions: low at X0 and below X0 - 5e-10, high in between and above X0."""
    low = 1.8 + (s - 0.59) ** 2
    return low if (s <= X0 - 5e-10 or s == X0) else low + 73.1


def _jumpy_right(s):
    """The jump only above X0 (only the forward difference at X0 sees it): the line search's last
    trial point has a loss a little below X0's."""
    low = 1.8 + (s - 0.59) ** 2
    return low if s <= X0 else low + 73.1


def _grid_ends_at_x0(func, ranges, args=(), **kw):
    """scipy.optimize.brute stand-in: the grid search (and its polish) ends at X0."""
    x = np.array([X0])
    return x, func(x, *args)


def _spy_minimize(monkeypatch):
    """[(x, fun, result)] of every L-BFGS-B call, as scipy returns them."""
    raw, real = [], L.minimize

    def spy(fun, x0, *a, **kw):
        res = real(fun, x0, *a, **kw)
        raw.append(([float(v) for v in np.atleast_1d(res.x)], float(res.fun), res))
        return res

    monkeypatch.setattr(L, "minimize", spy)
    return raw


@pytest.mark.parametrize("mode", ["energygeometry", "dataset"])
@pytest.mark.parametrize("loss_of", [_jumpy, _jumpy_right])
def test_lbfgsb_sigma_comes_with_the_loss_at_sigma(monkeypatch, capsys, mode, loss_of):
    """One parameter set (Loss.Repulsion_optimization: energygeometry evaluates
    parallel_repulsion, dataset/reaction unparallel_repulsion) on a jumpy loss: scipy's L-BFGS-B
    fails its first line search, puts x back to X0 and returns it with the loss of its last trial
    point (74.9, or a little below X0's), as in the M5b-3 report (sigma* = 0.5926215820 with
    mu = 74.874, mu(sigma*) = 1.7954559).  The search returns the loss recorded at X0."""
    raw = _spy_minimize(monkeypatch)
    monkeypatch.setattr(L, "brute", _grid_ends_at_x0)

    def fake(self, sigma, r0_w, r0_d, write_rep=True):
        return np.float64(loss_of(float(np.atleast_1d(sigma)[0])))

    monkeypatch.setattr(L.Loss, "parallel_repulsion", fake)
    monkeypatch.setattr(L.Loss, "unparallel_repulsion", fake)
    loss = L.Loss(args_input=argparse.Namespace(optimization_option=mode), cov_radius=1.0)
    sigma, mu = loss.Repulsion_optimization([2.6, 5.0, 2.0], 0.1, 0.8)
    (x_raw, fun_raw, _), = raw
    assert x_raw == [X0] and fun_raw != loss_of(X0)           # scipy: X0 with another loss
    assert float(sigma) == X0 and float(mu) == loss_of(X0) < 2.0
    assert "of another evaluation" in capsys.readouterr().out


@pytest.mark.parametrize("fitted, loss_of", [(["N"], _jumpy), (["N"], _jumpy_right),
                                             (["B", "N"], _jumpy_right)])
def test_lbfgsb_sigma_comes_with_the_loss_at_sigma_per_type(tmp_path, monkeypatch, fitted,
                                                            loss_of):
    """Parameters per type (Loss.Repulsion_optimization_types on utils.loss_grad.mu): one fitted
    set (n = 1, the search of before M5b) and two (n = 2: the joint L-BFGS-B, kept only if its
    loss is below the current point's).  scipy returns X0 with the loss of another evaluation;
    for n = 2 that loss was a little below X0's and the search kept it as X0's.  Now the search
    returns X0 with its own loss."""
    from tests.test_types import _bnna, _no_files_step
    m = TM.TypeModel.from_dataset(_bnna(), multi_element=fitted)
    assert not m.legacy and m.n_sigma == len(fitted)

    def fake_mu(step, sigma, parts=None):
        s = [float(v) for v in np.atleast_1d(sigma)]
        v = loss_of(s[0]) + sum(3.0 * (t - 0.4) ** 2 for t in s[1:])
        if parts is not None:
            parts.update(mu_E=v, mu_F=0.0, penalty=4.0, diffs=np.zeros(2))
        return v

    raw = _spy_minimize(monkeypatch)
    monkeypatch.setattr(L, "brute", _grid_ends_at_x0)
    monkeypatch.setattr(LG, "mu", fake_mu)
    monkeypatch.setattr(LG.StepData, "from_files", classmethod(_no_files_step))
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(optimization_option="dataset", log_file=str(tmp_path / "log.out"))
    loss = L.Loss(args_input=args, dft_input=_bnna(), types=m)
    svec, mu = loss.Repulsion_optimization_types([3.0, 4.0] * m.n_sigma)
    x_raw, fun_raw, _ = raw[0]
    assert x_raw == [X0] * m.n_sigma and fun_raw != fake_mu(None, x_raw)   # scipy's pairing
    assert svec == [X0] * m.n_sigma and mu == fake_mu(None, svec)


def test_lbfgsb_result_without_a_mismatch_is_scipys(tmp_path, monkeypatch):
    """A smooth loss: L-BFGS-B's x and fun belong together, and both searches return scipy's
    result itself (the code of one parameter set: the same object; per type: the same float)."""
    from tests.test_types import _bnna, _no_files_step
    raw = _spy_minimize(monkeypatch)

    def smooth(s):
        return 1.8 + 3.0 * (s - 0.47) ** 2

    monkeypatch.setattr(L.Loss, "parallel_repulsion", lambda self, sigma, r0_w, r0_d, w=True:
                        np.float64(smooth(float(np.atleast_1d(sigma)[0]))))
    loss = L.Loss(args_input=argparse.Namespace(optimization_option="energygeometry"),
                  cov_radius=1.0)
    sigma, mu = loss.Repulsion_optimization([2.6, 5.0, 2.0], 0.1, 0.8)
    (x_raw, fun_raw, res), = raw
    assert mu is res.fun and float(sigma) == x_raw[0] and fun_raw == smooth(x_raw[0])

    raw.clear()
    m = TM.TypeModel.from_dataset(_bnna(), multi_element=["N"])

    def fake_mu(step, sigma, parts=None):
        v = smooth(float(np.atleast_1d(sigma)[0]))
        if parts is not None:
            parts.update(mu_E=v, mu_F=0.0, penalty=4.0, diffs=np.zeros(2))
        return v

    monkeypatch.setattr(LG, "mu", fake_mu)
    monkeypatch.setattr(LG.StepData, "from_files", classmethod(_no_files_step))
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(optimization_option="dataset", log_file=str(tmp_path / "log.out"))
    svec, mu = L.Loss(args_input=args, dft_input=_bnna(), types=m).Repulsion_optimization_types(
        [3.0, 4.0])
    (x_raw, fun_raw, res), = raw
    assert svec == x_raw and mu.hex() == float(res.fun).hex()


def _molecules_in_boxes(name, n=11):
    """An EOS group of the M5b-3 report's synthetic data (section 8): a CO or OH molecule (1.13 or
    0.98 A, tilted) in a cubic box of 3.4 or 3.0 A, box and molecule scaled by 0.94-1.06, made-up
    DFT energies of -100 + 40 (s - 1)^2 eV per atom.  The DFTB energy of such a box has its minimum
    outside the scan: the EOS fit extrapolates, and switches between two solutions when sigma_rep
    changes by ~1e-9 (losses ~1.9 and ~75)."""
    sym, pos0, a = {"co": (["C", "O"], [[0, 0, 0], [1.13, 0, 0]], 3.4),
                    "oh": (["O", "H"], [[0, 0, 0], [0.98, 0, 0]], 3.0)}[name]
    pos = np.array(pos0, float) @ np.array([[0.8, 0.5, 0.33], [0, 0.6, 0.0], [0, 0, 0.7]]).T
    pos = pos / np.linalg.norm(pos[1] - pos[0]) * np.linalg.norm(np.array(pos0[1]))
    pos += a / 2
    out = []
    for s in np.linspace(0.94, 1.06, n):
        atoms = Atoms(sym, positions=pos * s, cell=np.eye(3) * a * s, pbc=True)
        e = -100.0 * len(sym) + 40.0 * (s - 1.0) ** 2 * len(sym)
        atoms.calc = SinglePointCalculator(atoms, energy=e, forces=np.zeros((len(sym), 3)))
        out.append(atoms)
    return out


@linux
def test_sigma_search_on_a_jumpy_eos_fit(tmp_path, monkeypatch):
    """The data of the M5b-3 report (section 8) through the whole BO step: EOS groups of CO and
    OH in boxes (_molecules_in_boxes), weights 0.5 each (with the factors per composition of
    M5b-4, 1 and 1, this data does not trip L-BFGS-B), one parameter set, pydftb, the stored
    tables.  L-BFGS-B's line search fails and scipy returns sigma* with the loss of another
    evaluation; the step's loss (returned, in par.out) is now the loss at sigma*, which
    parallel_repulsion and utils.loss_grad.mu compute anew, bit for bit."""
    from tests.test_shared_parameters import _tables_without_repulsive
    from tests.test_types import _ho_args
    saved, switches = eng.settings(), eng.type_switching()
    eng.configure(engine="pydftb")
    try:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(L.hotpy, "full_hotcent", _tables_without_repulsive({"H", "C", "O"}))
        dft = _molecules_in_boxes("co") + _molecules_in_boxes("oh")
        model = TM.TypeModel.from_dataset(dft, shared=True)
        assert model.legacy
        args = _ho_args(tmp_path, "energygeometry")
        args.eos_points, args.E0s, args.kpt_density = 11, None, 5.0
        loss = L.Loss(args_input=args, dft_input=dft, bolt_f11=[0.5] * len(dft), opt_elem=[],
                      path=tmp_path, types=model)
        raw = _spy_minimize(monkeypatch)
        found = {}
        search = L.Loss.Repulsion_optimization

        def record(self, conf, c0, c1):
            found["sigma"], found["mu"] = search(self, conf, c0, c1)
            return found["sigma"], found["mu"]

        monkeypatch.setattr(L.Loss, "Repulsion_optimization", record)
        x = [2.6, 5.0]
        mu = float(loss.Loss_EnergyGeometry(x))
        sigma = float(found["sigma"])
        (x_raw, fun_raw, _), = raw
        assert x_raw == [sigma] and mu == float(found["mu"]) and loss.last_failed is False
        monkeypatch.chdir(tmp_path / "results" / "opt_2.600_5.000")
        step = LG.StepData.from_files(loss, *x)
        at_sigma = float(loss.parallel_repulsion(np.array([sigma]), *x, False))
        if fun_raw == at_sigma:
            pytest.skip("scipy paired sigma* with its own loss on this machine's numbers")
        assert mu.hex() == at_sigma.hex() == float(LG.mu(step, sigma)).hex()
        par = [float(v) for v in (tmp_path / "par.out").read_text().split(",")]
        assert par[2] == pytest.approx(sigma, abs=1e-8) and par[-1] == pytest.approx(mu, rel=1e-10)
    finally:
        eng._SETTINGS.clear()
        eng._SETTINGS.update(saved)
        eng.set_type_switching(switches)
