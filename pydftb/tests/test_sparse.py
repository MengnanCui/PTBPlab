"""Sparse matrices for the linear-scaling purification (milestone M7c, DFTB+'s ``Solver = NTPoly
{ Sparse = Yes }``): the CSR helpers against dense arithmetic, the pattern of the neighbour
blocks, pydftb's SCC with NTPoly(sparse=True) against the dense NTPoly for every Hamiltonian it
takes (NTPoly's dense branch of the products and every product as CSR), the frozen DFTB+ 24.1
NTPoly cases through the sparse path (all of them: ``refgen --check --solver sparse`` and
``sparse-csr``), a truncated water box against the eigensolver, the ASE calculator and the
refusals.
"""
import dataclasses
import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp
from ase import Atoms
from ase.build import molecule
from ase.cluster import Octahedron

from pydftb import csr
from pydftb.ase_calc import PyDFTB
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.hamiltonian import fold
from pydftb.params import ParamSet
from pydftb.purification import NTPoly
from pydftb.typeswitch import TypeSwitch
from pydftb.validation import refgen
from pydftb.validation.cases import SKF_SETS

from .conftest import SKF_HCOCU

TIGHT = dict(method="trs2", tolerance=1e-10, truncation=1e-15)
CSR = dict(dense_fill=2.0)                     # every product as CSR
TORCH = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")


def _run(name, solver="eigh", **over):
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], **over)
    return refgen.run_pydftb(case, refgen._inputs(case), solver=solver)


def _deviations(a, b):
    return (abs(a.energy - b.energy), np.abs(a.charges - b.charges).max(),
            np.abs(a.forces - b.forces).max())


def _random(n, density, seed, complex_=False):
    rng = np.random.default_rng(seed)
    a = sp.random(n, n, density, format="csr", random_state=rng)
    if complex_:
        a = a + 1j * sp.random(n, n, density, format="csr", random_state=rng)
    return a.tocsr()


def _water_box(n, seed=0):
    """n waters at about 1 g/cm3 on a jittered cubic grid (as the M7c probes)."""
    rng = np.random.default_rng(seed)
    side = (n * 29.9) ** (1 / 3)
    m = int(np.ceil(n ** (1 / 3)))
    box = Atoms(cell=[side] * 3, pbc=True)
    k = 0
    for i in range(m):
        for j in range(m):
            for l in range(m):
                if k == n:
                    break
                w = molecule("H2O")
                w.rotate(rng.uniform(0, 360), rng.normal(size=3))
                w.translate((np.array([i, j, l]) + 0.5) * side / m + rng.uniform(-0.3, 0.3, 3))
                box += w
                k += 1
    return box


@pytest.fixture(scope="module")
def sk():
    return ParamSet(SKF_HCOCU).skset(["C", "O", "H"])


# ---- CSR arithmetic -------------------------------------------------------------------------
@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
def test_products(complex_):
    """The CSR product drops |c| < threshold (pydftb's dense rule); the thread panels give the
    serial product to the bit; NTPoly's dense branch is the dense GEMM to the bit."""
    a, b = _random(900, 0.03, 1, complex_), _random(900, 0.03, 2, complex_)
    dense = a.toarray() @ b.toarray()
    thr = 0.3 * np.abs(dense).max()
    with csr.options(threads=1, dense_fill=2.0):
        c1 = csr.matmul(a, b, thr)
    with csr.options(threads=3, dense_fill=2.0):
        c3 = csr.matmul(a, b, thr)
    assert a.nnz > csr.SMALL_PRODUCT                       # the panels are used
    assert c1.shape == c3.shape and (c1 != c3).nnz == 0
    ref = np.where(np.abs(dense) < thr, 0.0, dense)
    assert np.abs(c1.toarray() - ref).max() < 1e-12
    assert 0 < c1.nnz < np.count_nonzero(dense)            # something was dropped
    with csr.options(dense_fill=0.0):
        cd = csr.matmul(a, b, thr)
    assert np.array_equal(cd.toarray(), ref)
    assert csr.drop(c1, 2.0 * thr).nnz < c1.nnz and csr.drop(c1, 0.0) is c1


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
def test_helpers(complex_):
    """Trace, tr(A^H B), Gershgorin bounds, column sums, conjugate transpose, identity and
    equality on CSR matrices: the dense results."""
    a = _random(60, 0.2, 3, complex_)
    a = a + csr.ct(a)                                      # Hermitian
    b = _random(60, 0.2, 4, complex_)
    ad, bd = a.toarray(), b.toarray()
    assert abs(csr.trace(a) - np.trace(ad).real) < 1e-13
    assert abs(csr.dot(a, b) - np.vdot(ad, bd).real) < 1e-12
    d, r = np.diag(ad).real, np.abs(ad).sum(axis=0) - np.abs(np.diag(ad))
    assert np.allclose(csr.gershgorin(a), ((d - r).min(), (d + r).max()), atol=1e-13)
    assert np.allclose(csr.abs_column_sums(b), np.abs(bd).sum(axis=0), atol=1e-13)
    assert np.array_equal(csr.ct(b).toarray(), bd.conj().T)
    assert np.array_equal(csr.eye(5, b.dtype).toarray(), np.eye(5))
    assert csr.equal(b, b.copy()) and not csr.equal(b, 2.0 * b)


# ---- the pattern of the neighbour blocks -------------------------------------------------------
def test_pattern(sk):
    """H0 and S folded onto the pattern of a periodic water box at the Gamma point and at a
    k-point: the dense folds at the pattern's places, zero elsewhere; the transposed entries;
    sampling a CSR matrix; Mulliken row sums and the values at the pair elements."""
    box = _water_box(8)
    calc = DFTB(sk, solver=NTPoly(sparse=True), scc=False)
    st = calc._setup(box.get_chemical_symbols(), box.positions / BOHR, box.cell.array / BOHR,
                     np.ones(3, bool), False, {})
    pat, pb, n = st.sparse.pattern, st.pairs, st.basis.norb
    assert st.h0 is None and st.smat is None
    assert np.array_equal(pat.rows[pat.elem], pb.rows) and np.array_equal(pat.cols[pat.elem],
                                                                            pb.cols)
    assert np.array_equal(pat.rows[pat.elem_t], pb.cols)
    assert np.array_equal(pat.rows[pat.diag], np.arange(n))
    h_dense = fold(pb, pb.h, n, None, st.eps0)
    h = pat.matrix(st.sparse.h0[0]).toarray()
    assert np.abs(h - h_dense).max() < 1e-15
    k = np.array([0.25, 0.0, 0.5])
    s_k = fold(pb, pb.s, n, k, np.ones(n))
    s_pat = pat.fold(pb.s, np.ones(n), [k])
    assert np.abs(pat.matrix(s_pat[0]).toarray() - s_k).max() < 1e-15
    assert np.abs(pat.sample(sp.csr_matrix(h_dense)) - st.sparse.h0[0]).max() < 1e-15
    rho = np.random.default_rng(5).normal(size=(n, n))
    rho = rho + rho.T
    q = pat.row_sums(pat.sample(sp.csr_matrix(rho)) * st.sparse.s[0])
    assert np.abs(q - (rho * fold(pb, pb.s, n, None, np.ones(n))).sum(1)).max() < 1e-12
    vals = pat.at_pairs(pat.sample(sp.csr_matrix(rho))[None], [0.7])
    assert np.abs(vals - 0.7 * rho[pb.cols, pb.rows]).max() < 1e-15


# ---- pydftb's SCC on CSR matrices ---------------------------------------------------------------
@pytest.mark.parametrize("name", [
    "h2o", "h2o_nonscc", "ch3oh_custom_hubbards",       # molecules, non-SCC, custom Hubbards
    "water_box_gamma", "water_box_field",           # periodic Gamma point, field (stress)
    "water_slab_dipole",                            # k-points, dipole correction, stress
    "o2_triplet", "o2_triplet_shell",               # collinear spin
    "water_box_dftb3", "water_dimer_hdamp",         # DFTB3, H damping
    "water_box_d3bj_3body", "water_dimer_h5",       # dispersion, H5
    "h2o_gb_field_pc", "h2o_site",                  # solvation, point charges, site potentials
    "meoh_water_halogen", "h2o_read_charges"])      # halogen correction, a charge file
def test_sparse_is_the_dense_ntpoly(name):
    """NTPoly(sparse=True) gives the dense NTPoly's SCC (a tight TRS2): energies, charges,
    forces and stress to rounding in the same SCC iterations, with NTPoly's dense branch of
    the products and with every product as CSR."""
    why = refgen.missing_requirement(refgen.load(name))   # the D3 case needs dftd3
    if why:
        pytest.skip(why)
    a = _run(name, NTPoly(**TIGHT))
    for extra in ({}, CSR):
        b = _run(name, NTPoly(**TIGHT, sparse=True, **extra))
        de, dq, df = _deviations(a, b)
        assert de <= 1e-11 and dq <= 1e-11 and df <= 1e-11, (extra, de, dq, df)
        assert a.n_scc == b.n_scc and b.eigenvalues is None
        assert abs(a.fermi - b.fermi) <= 1e-9 if a.fermi_spin is None else \
            np.abs(np.subtract(a.fermi_spin, b.fermi_spin)).max() <= 1e-9
        if a.stress is not None:
            assert np.abs(a.stress - b.stress).max() <= 1e-13


def test_tight_sparse_is_the_eigensolver():
    """The tight TRS2 on CSR matrices gives the eigensolver's SCC (as the dense one, M7a)."""
    a, b = _run("water_box_gamma"), _run("water_box_gamma", NTPoly(**TIGHT, sparse=True, **CSR))
    de, dq, df = _deviations(a, b)
    assert de <= 2e-10 and dq <= 2e-10 and df <= 1e-10 and a.n_scc == b.n_scc


@pytest.mark.parametrize("name", ["h2o_ntpoly", "h2o_nonscc_ntpoly", "ch3oh_ntpoly_pm",
                                  "water_dimer_typed_ntpoly_trs2",
                                  "ch3oh_custom_hubbards_ntpoly_trs2_tight",
                                  "water_box_h5_ntpoly_hpcp", "water_box_mixer_diis_ntpoly"])
@pytest.mark.parametrize("solver", refgen.SPARSE_SOLVERS)
def test_dftbplus_ntpoly_references(name, solver):
    """DFTB+ 24.1's NTPoly (its sparse input, DFTB+'s defaults and settings of the four methods)
    through the sparse path: the frozen references' tolerances and SCC iterations (M7a)."""
    assert refgen.sparse_skip(refgen.CASES_BY_NAME[name]) is None
    r = refgen.compare(refgen.load(name), solver)
    assert r["ok"], r["dev"]


def test_block_potentials_are_skipped_by_refgen():
    for name in ("water_box_ons_ntpoly", "cu4co_plusu_ntpoly"):
        assert "onsite" in refgen.sparse_skip(refgen.CASES_BY_NAME[name])
    assert refgen.sparse_skip(refgen.CASES_BY_NAME["h2o"]) == "not an NTPoly case (M7a)"


def test_truncated_water_box():
    """27 waters with DFTB+'s TRS4 and truncation 1e-6, every product as CSR: the eigensolver's
    SCC within the accuracy of the truncation (P111, 216 waters: 3e-7 Ha, 5e-7 e,
    3e-7 Ha/Bohr); ``last_fill`` the elements per row of X."""
    box = _water_box(27)
    sk = ParamSet(SKF_SETS["ptbp_hcocu"], max_l={"H": 0, "O": 1}).skset(["O", "H"])
    args = (box.get_chemical_symbols(), box.positions / BOHR, box.cell.array / BOHR,
            np.ones(3, bool))
    ref = DFTB(sk, scc_tol=1e-7).calculate(*args, stress=False)
    solver = NTPoly(truncation=1e-6, sparse=True, **CSR)
    res = DFTB(sk, scc_tol=1e-7, solver=solver).calculate(*args, stress=False)
    de, dq, df = _deviations(ref, res)
    assert de <= 2e-6 and dq <= 2e-6 and df <= 2e-6
    assert res.n_scc == ref.n_scc
    assert 0.0 < solver.last_fill <= 2 * len(box)          # 6 orbitals per water


def test_type_switching():
    """A type switch (M5a) needs only the pair elements: the sparse path is the dense one."""
    cu = Octahedron("Cu", 2, cutoff=0, latticeconstant=3.61)
    cu.rattle(0.12, seed=21)
    sk = ParamSet(SKF_SETS["cu_types"]).skset(["Cu", "Cu2"])
    sw = TypeSwitch("Cu", ("Cu2", "Cu"), centre=3.2, width=2.0, cutoff=(4.6, 5.4))
    runs = [DFTB(sk, type_switching=sw, solver=NTPoly(**TIGHT, **extra), scc_tol=1e-10)
            .calculate(["Cu"] * len(cu), cu.positions / BOHR)
            for extra in ({}, dict(sparse=True, **CSR))]
    de, dq, df = _deviations(*runs)
    assert de <= 1e-11 and dq <= 1e-11 and df <= 1e-11 and runs[0].n_scc == runs[1].n_scc


def test_ase_calculator():
    """PyDFTB with NTPoly(sparse=True): the energy and forces of DFTB with the dense NTPoly."""
    box = _water_box(8)
    out = []
    for solver in (NTPoly(**TIGHT), NTPoly(**TIGHT, sparse=True)):
        atoms = box.copy()
        atoms.calc = PyDFTB(SKF_SETS["ptbp_hcocu"], max_l={"H": 0, "O": 1}, scc_tol=1e-10,
                            solver=solver)
        out.append((atoms.get_potential_energy(), atoms.get_forces()))
    assert abs(out[0][0] - out[1][0]) < 1e-9 and np.abs(out[0][1] - out[1][1]).max() < 1e-9


# ---- refusals -----------------------------------------------------------------------------------
def test_refusals(sk):
    """What the sparse path does not have (M7c) and DFTB+'s refusal of two-component spin."""
    w = {"O": -0.028, "H": -0.072, "C": -0.023}
    for opts, msg in (
            (dict(spin_orbit={"O": [0.0, 0.01]}), "two-component spin .*avaible"),
            (dict(spin="noncolinear", spin_constants=w), "two-component spin"),
            (dict(plus_u={"O": [("p", 0.1)]}), "DFTB\\+U and onsite corrections"),
            (dict(analysis={"cm5": True}), "analysis outputs"),
            (dict(force_evaluation="dynamics"), "force_evaluation='dynamics'")):
        with pytest.raises(ValueError, match=r"the sparse NTPoly path \(sparse=True, milestone "
                                             r"M7c\) does not support .*" + msg):
            DFTB(sk, solver=NTPoly(sparse=True), **opts)
    with pytest.raises(ValueError, match="precision='mixed' is not used with sparse=True"):
        NTPoly(sparse=True, precision="mixed")


@TORCH
def test_torch_is_refused(sk):
    with pytest.raises(ValueError, match="the torch backend"):
        DFTB(sk, solver=NTPoly(sparse=True), backend="torch")
