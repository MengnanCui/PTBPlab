"""The A-series fit of environment models (pydftb.adaptive_fit): band samples (a self-fit to
DFTB bands of a perturbed model recovers it, molecules and a periodic cell, band-only with no
energy samples), the offset modes (per-element offsets recovered), models without analytic
dF/dx (finite differences), the k-point forms of the A1 scans, and refusals."""
import numpy as np
import pytest
from ase.build import bulk, molecule

from pydftb import DFTB, ParamSet
from pydftb.adaptive import RBF, AdaptiveParams
from pydftb.adaptive_fit import (BandSample, Sample, _kpoints, evaluate_all, evaluate_band_sample,
                                 evaluate_bands, fit_adaptive)
from pydftb.bandref import BandRef
from pydftb.bands import eigenstates
from pydftb.constants import BOHR, HARTREE
from pydftb.validation.cases import SKF_SETS

SK = SKF_SETS["ptbp_hcocu"]


@pytest.fixture(scope="module")
def params():
    return ParamSet(SK, max_l={"H": 0, "O": 1, "Cu": 2})


def _waters():
    out = []
    for k, (d, ang) in enumerate([(0.92, 100), (1.03, 109), (1.10, 118), (0.97, 128)]):
        w = molecule("H2O")
        w.set_distance(0, 1, d, fix=0)
        w.set_angle(1, 0, 2, ang)
        w.positions[2] += [0.02 * k, -0.01, 0.015]
        out.append(w)
    return out


def _o_model(x=(0.0, 0.0)):
    return AdaptiveParams("O", RBF(()), onsite={"p": (x[0],)}, hopping={"p": (x[1],)})


def _molecule_ref(at, sk, model, name, shift=-1.3):
    """A 'molecule' BandRef from DFTB with ``model`` (plus a shift: the fit has a free one)."""
    lab = at.get_chemical_symbols()
    eng = DFTB(sk, scc_tol=1e-12, adaptive=[model])
    r = eng.calculate(lab, at.positions / BOHR, forces=False)
    st = eigenstates(eng, lab, at.positions / BOHR, None, None, r.q_in)
    return BandRef(name=name, kind="molecule", kpts=[[0, 0, 0]], kweights=[1.0],
                   energies=st.eigenvalues * HARTREE + shift,
                   occupations=st.occupations / 2.0).set_atoms(at)


def test_band_only_self_fit_of_molecules(params):
    """No energy samples (A2: the electronic part fitted to bands alone)."""
    sk = params.skset(["O", "H"])
    truth = _o_model((0.03, 0.08))
    bs = [BandSample(_molecule_ref(w, sk, truth, f"w{k}"), group="water")
          for k, w in enumerate(_waters())]
    start = _o_model()
    before = evaluate_bands([start], bs, sk, {"scc_tol": 1e-12})
    assert min(b["loss"] for b in before) > 0.05                    # eV: really off at the start
    fit = fit_adaptive([start], [], sk, {"scc_tol": 1e-12}, l2=0.0, processes=1,
                       max_evaluations=40, band_samples=bs, band_weight=1.0)
    assert fit.band_loss < 1e-6 and len(fit.band_results) == 4
    assert np.abs(fit.x - truth.vector()).max() < 1e-4
    assert np.isnan(fit.rmse) and len(fit.residuals) == 0


def _cu_cell():
    at = bulk("Cu", "fcc", a=3.7, cubic=True)
    at.rattle(0.08, seed=1)
    at.cell[0, 1] += 0.15
    return at[:3]


def test_band_self_fit_of_a_metal_on_gamma_centred_grids(params):
    sk = params.skset(["Cu"])
    at = _cu_cell()
    truth = AdaptiveParams("Cu", RBF(()), onsite={"d": (0.02,)}, hopping={"s": (0.06,)})
    opts = {"scc_tol": 1e-11, "temperature": 1000.0}
    grid = {"size": (2, 2, 2), "gamma": True}
    kpath = np.array([[0.0, 0.0, 0.0], [0.15, 0.1, 0.0], [0.3, 0.2, 0.1], [0.5, 0.0, 0.25]])
    lab = at.get_chemical_symbols()
    geo = (lab, at.positions / BOHR, at.cell.array / BOHR, np.ones(3, bool))
    eng = DFTB(sk, kpts=_kpoints(at, grid), adaptive=[truth], **opts)
    r = eng.calculate(*geo, forces=False)
    st = eigenstates(eng, *geo, r.q_in, kpts=(kpath, np.ones(len(kpath))))
    ref = BandRef(name="cu3", kind="path", kpts=kpath, kweights=np.ones(len(kpath)),
                  energies=st.eigenvalues * HARTREE + 0.7, fermi=r.fermi * HARTREE + 0.7,
                  occupations=st.occupations / 2.0, level=np.ones(len(kpath), int)).set_atoms(at)
    bs = [BandSample(ref, kpts=grid)]
    start = AdaptiveParams("Cu", RBF(()), onsite={"d": (0.0,)}, hopping={"s": (0.0,)})
    c0, r0, j0 = evaluate_band_sample([start], bs[0], sk, opts, jacobian=True)
    _, _, j1 = evaluate_band_sample([start], bs[0], sk, opts, jacobian=True, method="scc")
    assert c0["metal"] and np.abs(r0).max() > 1e-3
    assert np.abs(j0 - j1).max() < 0.05 * np.abs(j1).max()         # metal: tiny charge response
    fit = fit_adaptive([start], [], sk, opts, l2=0.0, processes=1, max_evaluations=30,
                       band_samples=bs, band_weight=1.0)
    assert fit.band_loss < 1e-5
    assert np.abs(fit.x - truth.vector()).max() < 1e-3


def test_kpoint_forms():
    at = _cu_cell()
    mp = _kpoints(at, (2, 2, 2))                       # Monkhorst-Pack (shifted for even n)
    gc = _kpoints(at, {"size": [2, 2, 2], "gamma": True})
    assert not np.any(np.all(np.abs(mp[0]) < 1e-12, axis=1))
    assert np.any(np.all(np.abs(gc[0]) < 1e-12, axis=1))   # Gamma-centred: contains Gamma
    assert abs(gc[1].sum() - 1.0) < 1e-12
    k = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    kw = _kpoints(at, (k, np.array([1.0, 3.0])))
    assert np.allclose(sorted(kw[1]), [0.25, 0.75])
    assert _kpoints(at, None) is None
    # a sample with a Gamma-centred dict runs
    sk = ParamSet(SK, max_l={"Cu": 2}).skset(["Cu"])
    m = AdaptiveParams("Cu", RBF(()), onsite={"d": (0.0,)})
    e, jac, _, _ = evaluate_all([m], [Sample(at, 0.0, kpts={"size": (2, 2, 2), "gamma": True})],
                                sk, {"temperature": 1000.0}, m.vector())
    assert np.isfinite(e).all() and jac.shape == (1, 1)


def _composition_samples(sk, truth, c_el):
    """Energies per atom of a known model, shifted by per-element offsets c_el (eV/atom)."""
    mols = []
    for name in ("H2O", "H2O2", "H2"):
        for k in range(2):
            a = molecule(name)
            a.rattle(0.03, seed=k + 5)
            mols.append(a)
    out = []
    for k, a in enumerate(mols):
        lab = a.get_chemical_symbols()
        r = DFTB(sk, scc_tol=1e-11, adaptive=[truth]).calculate(lab, a.positions / BOHR)
        shift = sum(c_el[e] for e in lab) / len(lab)
        out.append(Sample(a, r.free_energy * HARTREE / len(lab) - shift, name=str(k)))
    return out


def test_per_element_offsets_are_recovered(params):
    sk = params.skset(["O", "H"])
    truth = _o_model((0.03, 0.08))
    c_el = {"H": 0.2, "O": -0.5}
    data = _composition_samples(sk, truth, c_el)
    start = _o_model()
    fit = fit_adaptive([start], data, sk, {"scc_tol": 1e-11}, l2=0.0, processes=1,
                       max_evaluations=40, offsets="per_element")
    assert fit.rmse < 1e-6
    assert np.abs(fit.x - truth.vector()).max() < 1e-4
    assert fit.offsets["H"] == pytest.approx(0.2, abs=1e-5)
    assert fit.offsets["O"] == pytest.approx(-0.5, abs=1e-5)
    # one global offset cannot absorb per-element offsets across compositions
    glob = fit_adaptive([start], data, sk, {"scc_tol": 1e-11}, l2=0.0, processes=1,
                        max_evaluations=20)
    assert glob.rmse > 1e-3 and glob.offsets is None
    # absolute energies (no offset) of offset-free data
    data0 = _composition_samples(sk, truth, {"H": 0.0, "O": 0.0})
    none = fit_adaptive([start], data0, sk, {"scc_tol": 1e-11}, l2=0.0, processes=1,
                        max_evaluations=30, offsets="none")
    assert none.rmse < 1e-6 and none.offset == 0.0


class _NoGradient(AdaptiveParams):
    """An environment model whose engine gives no dF/dx (as some r0-mixing models)."""

    def gradient_from(self, result):
        return None


def test_models_without_gradients_use_finite_differences(params):
    sk = params.skset(["O", "H"])
    samples = [Sample(w, 0.0, name=str(k)) for k, w in enumerate(_waters())]
    m = _o_model((0.02, 0.05))
    ng = _NoGradient("O", RBF(()), onsite={"p": (0.02,)}, hopping={"p": (0.05,)})
    assert type(ng.with_vector(ng.vector())) is _NoGradient
    opts = {"scc_tol": 1e-12}
    e1, j1, _, _ = evaluate_all([m], samples, sk, opts, m.vector(), with_forces=False)
    e2, j2, _, _ = evaluate_all([ng], samples, sk, opts, ng.vector(), with_forces=False)
    assert np.array_equal(e1, e2)
    assert np.abs(j1 - j2).max() < 1e-6 * np.abs(j1).max()


def test_defaults_and_refusals(params):
    sk = params.skset(["O", "H"])
    m = _o_model()
    with pytest.raises(ValueError, match="nothing to fit"):
        fit_adaptive([m], [], sk)
    with pytest.raises(ValueError, match="offsets"):
        fit_adaptive([m], [Sample(_waters()[0], 0.0)], sk, offsets="atomic")
    w = _waters()[0]
    bs = BandSample(_molecule_ref(w, sk, m, "w"))
    with pytest.raises(ValueError, match="band_jacobian"):
        evaluate_band_sample([m], bs, sk, {}, jacobian=True, method="exact")
    # band samples with band_weight 0 change nothing
    data = [Sample(a, 0.1 * k, name=str(k)) for k, a in enumerate(_waters())]
    a = fit_adaptive([m], data, sk, {"scc_tol": 1e-11}, l2=1e-4, processes=1, max_evaluations=8)
    b = fit_adaptive([m], data, sk, {"scc_tol": 1e-11}, l2=1e-4, processes=1, max_evaluations=8,
                     band_samples=[bs], band_weight=0.0)
    assert np.array_equal(a.x, b.x) and a.rmse == b.rmse and b.band_loss is None
