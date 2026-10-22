"""Block potentials (milestone M6b): DFTB+U and onsite corrections.

Comparisons with DFTB+ are the frozen references cu4co_plusu*, cuo_chain_*plusu*,
gdo_spin_plusu, oh_noncolinear_*plusu, cu4co_soc_plusu, *_ons* and
cuo_chain_spin_plusu_bands in test_reference.py; here the building blocks,
self-consistency, limits and errors (no dftb+).
"""
import numpy as np
import pytest

from pydftb import blockpot
from pydftb.api import SCC
from pydftb.ase_calc import PyDFTB
from pydftb.bands import eigenstates
from pydftb.blockpot import ONSITE_PBE, BlockTerms
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.kpoints import monkhorst_pack
from pydftb.params import ParamSet
from pydftb.system import System
from pydftb.validation import refgen
from pydftb.validation.cases import ONS

from .conftest import SKF_HCOCU

U_CU = {"Cu": [("d", 0.15)]}


def _system(name):
    case = refgen.CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    params = ParamSet(refgen.SKF_SETS[case.skf_set], max_l=inp["max_l"],
                      shell_resolved=case.shell_resolved)
    kpts = None if inp["kpoints"] is None else (np.array(inp["kpoints"]), np.array(inp["kweights"]))
    cell = None if inp["cell_angstrom"] is None else np.array(inp["cell_angstrom"]) / BOHR
    opts = dict(case.dftb3 or {}, **({"spin": True, **case.spin} if case.spin else {}),
                **refgen.pauli_options(case, inp["labels"]))
    return (case, params.skset(inp["labels"]), inp["labels"],
            np.array(inp["positions_angstrom"]) / BOHR, cell, kpts, opts)


def _fd_force(calc, labels, pos, cell, atom, comp, h=1e-4):
    pbc = None if cell is None else np.ones(3, bool)
    e = []
    for s in (1, -1):
        p = pos.copy()
        p[atom, comp] += s * h
        e.append(calc.calculate(labels, p, cell, pbc, forces=False).free_energy)
    return -(e[0] - e[1]) / (2 * h)


def _fd_stress(calc, labels, pos, cell, i, j, h=1e-5):
    e = []
    for s in (1, -1):
        eps = np.zeros((3, 3))
        eps[i, j] += 0.5 * s * h
        eps[j, i] += 0.5 * s * h
        m = np.eye(3) + eps
        e.append(calc.calculate(labels, pos @ m.T, cell @ m.T, np.ones(3, bool),
                                forces=False).free_energy)
    return -(e[0] - e[1]) / (2 * h) / abs(np.linalg.det(cell))


# ---- the building blocks ----------------------------------------------------------------
def test_onsite_orbital_matrix():
    """DFTB+ getOnsME: shell blocks spread over orbitals, the s shell's own element unused,
    symmetrised."""
    f = np.array([[0.7, 0.2], [0.4, -0.1]])                 # deliberately not symmetric
    m = blockpot.onsite_orbital_matrix([0, 1], f)
    assert m.shape == (4, 4)
    assert m[0, 0] == 0.0                                   # s-s of the same shell
    assert np.allclose(m[0, 1:], 0.3) and np.allclose(m[1:, 0], 0.3)
    assert np.allclose(m[1:, 1:], -0.1)                     # p-p, diagonal included
    two_s = blockpot.onsite_orbital_matrix([0, 0], f)       # two s shells: cross terms kept
    assert two_s[0, 0] == two_s[1, 1] == 0.0 and two_s[0, 1] == pytest.approx(0.3)


def test_plus_u_specifications():
    _, sk, *_ = _system("cu4co_300K")
    p = blockpot.parse_plus_u
    assert p({"Cu": [("pd", 0.1)]}, sk, ["Cu"]) == {"Cu": [((1, 2), 0.1)]}
    assert p({"Cu": ("d", 0.1)}, sk, ["Cu"]) == {"Cu": [((2,), 0.1)]}
    assert p({"Cu": {"d": 0.1, "s": 0.2}}, sk, ["Cu"]) == {"Cu": [((2,), 0.1), ((0,), 0.2)]}
    assert p({"Cu": [([2, 1], 0.1)]}, sk, ["Cu"]) == {"Cu": [((1, 2), 0.1)]}
    assert p({"Cu": [(2, 0.1)], "Ni": [("d", 0.1)]}, sk, ["Cu", "O"]) == {"Cu": [((2,), 0.1)]}
    for spec, msg in (([("d", -0.1)], "negative"), ([("d", 1e-11)], "too small"),
                      ([("d", 0.1), ("pd", 0.1)], "more than one group"),
                      ([("x", 0.1)], "unknown shell"), ([(3, 0.1)], "out of range")):
        with pytest.raises(ValueError, match=msg):
            p({"Cu": spec}, sk, ["Cu"])
    with pytest.raises(ValueError, match="has no d shell"):
        p({"O": [("d", 0.1)]}, sk, ["O"])


def test_onsite_specifications():
    _, sk, *_ = _system("h2o")
    out = blockpot.parse_onsite(ONSITE_PBE, sk, ["O", "H"])
    assert out["O"][0].shape == (2, 2) and out["H"][1].shape == (1, 1)
    flat = {"O": ([0, 0.1, 0.1, 0.2], [0, 0.3, 0.3, 0.4]), "H": ([0], [0])}
    assert np.allclose(blockpot.parse_onsite(flat, sk, ["O", "H"])["O"][1], [[0, 0.3], [0.3, 0.4]])
    with pytest.raises(ValueError, match="no entry for 'H'"):
        blockpot.parse_onsite({"O": ONSITE_PBE["O"]}, sk, ["O", "H"])
    with pytest.raises(ValueError, match="2 x 2 values"):
        blockpot.parse_onsite({"O": ([0, 1, 2], [0, 1, 2])}, sk, ["O"])
    with pytest.raises(ValueError, match="is for shells spd"):
        blockpot.parse_onsite({"O": ONSITE_PBE["S"]}, sk, ["O"])


def _terms(name, rep="q", **opts):
    _, sk, labels, *_ = _system(name)
    calc = DFTB(sk, **opts)
    st = calc._setup(labels, _system(name)[3], None, None, False, {}, matrices=False)
    return calc, st, BlockTerms.build(st.basis, sk, rep, st.site_of_orb,
                                      blockpot.parse_plus_u(opts.get("plus_u") or {}, sk, labels),
                                      "fll", None if opts.get("onsite_corrections") is None
                                      else blockpot.parse_onsite(opts["onsite_corrections"], sk,
                                                                 labels), False)


def test_mixing_layout_as_dftbplus():
    """Cu4CO with +U on Cu d: per Cu the non-U orbitals (s, p) are one class and the five d
    orbitals five more; C and O one class each; the 10 d-d pairs of each Cu.  Water with
    onsite corrections: every orbital its own class, every pair of an atom."""
    _, _, blk = _terms("cu4co_300K", plus_u=U_CU)
    assert blk.n_class == 4 * 6 + 2 and len(blk.pairs) == 4 * 10 and blk.size == 66
    _, _, ons = _terms("h2o", onsite_corrections=ONS)
    assert ons.n_class == 6 and len(ons.pairs) == 6 and ons.size == 12
    _, _, two = _terms("cu4co_300K", rep="ud", plus_u=U_CU)
    assert two.n_class == 2 * (4 * (2 + 5) + 2 * 2) and len(two.pairs) == 2 * 40


def test_vector_round_trip():
    _, _, blk = _terms("cu4co_300K", plus_u={"Cu": [("pd", 0.15)], "O": [("p", 0.1)]})
    v = np.random.default_rng(1).normal(size=blk.size)
    q, blocks, skew = blk.expand(v)
    assert skew is None
    assert np.allclose(blk.reduce(q, blocks), v, atol=1e-15)


@pytest.mark.parametrize("fll", [True, False])
def test_plus_u_energy_and_potential(fll):
    """E_U equals (U-J)/2 sum_sigma Tr(n - n n) (FLL) or -(U-J)/2 sum_sigma Tr(n n) (pSIC) in
    the spin occupation matrices, and V = dE/dQ."""
    _, sk, labels, *_ = _system("oh_radical")
    st = DFTB(sk)._setup(labels, _system("oh_radical")[3], None, None, False, {}, matrices=False)
    blk = BlockTerms.build(st.basis, sk, "ud", st.site_of_orb,
                           blockpot.parse_plus_u({"O": [("sp", 0.2)]}, sk, labels),
                           "fll" if fll else "psic", None, False)
    rng = np.random.default_rng(2)
    a = rng.normal(size=(2, 4, 4))
    q = [np.stack([a[0] + a[0].T, a[1] + a[1].T])]           # (q, m) blocks of O
    e = blk.energies(q)[0]
    n = [0.5 * (q[0][0] + q[0][1]), 0.5 * (q[0][0] - q[0][1])]
    ref = sum(np.trace(x) * fll - np.trace(x @ x) for x in n) * 0.1
    assert e == pytest.approx(ref, abs=1e-13)
    v = blk.potentials(q)[0][0]
    h = 1e-6
    for s, i, j in ((0, 1, 2), (1, 0, 3), (0, 2, 2)):
        dq = [x.copy() for x in q]
        dq[0][s, i, j] += h
        if i != j:
            dq[0][s, j, i] += h
        de = (blk.energies(dq)[0] - e) / h
        assert de == pytest.approx(v[s, i, j] * (1 if i == j else 2), abs=1e-5)


def test_onsite_potential_is_the_energy_derivative():
    _, _, blk = _terms("h2o", onsite_corrections=ONS)
    rng = np.random.default_rng(3)
    q = []
    for sl in blk.slices:
        a = rng.normal(size=(1, sl.stop - sl.start, sl.stop - sl.start))
        q.append(a + a.transpose(0, 2, 1))
    e = blk.energies(q)[1]
    v = blk.potentials(q)[0]
    h = 1e-6
    for i, j in ((0, 1), (1, 2), (2, 2), (0, 0)):
        dq = [x.copy() for x in q]
        dq[0][0, i, j] += h
        if i != j:
            dq[0][0, j, i] += h
        de = (blk.energies(dq)[1] - e) / h
        assert de == pytest.approx(v[0][0, i, j] * (1 if i == j else 2), abs=1e-5)


# ---- limits ----------------------------------------------------------------------------
def test_zero_onsite_constants_change_nothing():
    _, sk, labels, pos, *_ = _system("ch3oh")
    zero = {lab: {"uu": np.zeros((n, n)), "ud": np.zeros((n, n))}
            for lab, n in (("C", 2), ("O", 2), ("H", 1))}
    plain = DFTB(sk, scc_tol=1e-12).calculate(labels, pos)
    ons = DFTB(sk, scc_tol=1e-12, onsite_corrections=zero).calculate(labels, pos)
    assert ons.e_onsite == 0.0
    assert ons.energy == pytest.approx(plain.energy, abs=1e-11)
    assert np.allclose(ons.forces, plain.forces, atol=1e-10)
    assert np.allclose(ons.charges, plain.charges, atol=1e-10)
    assert ons.terms["onsite"] == 0.0 and plain.terms["+U"] == 0.0


def test_restart_from_the_block_vector():
    _, sk, labels, pos, *_ = _system("cu4co_300K")
    calc = DFTB(sk, smearing=("fermi", 300.0), scc_tol=1e-10, plus_u=U_CU,
                onsite_corrections=ONS)
    r = calc.calculate(labels, pos)
    again = calc.calculate(labels, pos, q_init=r.q_in)
    assert again.n_scc <= 2                  # given charges: no two-iteration minimum
    assert again.energy == pytest.approx(r.energy, abs=1e-9)
    assert r.e_plus_u > 0 and r.e_onsite > 0


# ---- derivatives -------------------------------------------------------------------------
@pytest.mark.parametrize("name, opts", [
    ("cu4co_300K", {"plus_u": {"Cu": [("s", 0.1)]}}),                 # DFTB+ 25.1 crashes on it
    ("cu4co_300K", {"plus_u": U_CU, "plus_u_functional": "psic"}),
    ("oh_radical", {"plus_u": {"O": [("p", 0.1)]}, "onsite_corrections": ONS}),
    ("oh_noncolinear_soc", {"plus_u": {"O": [("p", 0.1)]}, "onsite_corrections": ONS}),
])
def test_molecule_forces_match_energy_differences(name, opts):
    """Collinear spin and the two-component path with dual spin-orbit coupling (imaginary
    block potentials).  Away from DFTB+'s mid-gap Fermi rule: Cu atoms of Cu4CO (see the
    M6b report)."""
    case, sk, labels, pos, _, _, base = _system(name)
    calc = DFTB(sk, smearing=tuple(case.filling), scc_tol=1e-12, max_scc=500, **base, **opts)
    f = calc.calculate(labels, pos).forces
    for comp in range(3):
        assert _fd_force(calc, labels, pos, None, 1, comp) == pytest.approx(f[1, comp], abs=1e-8)


@pytest.mark.parametrize("name, opts", [
    ("cuo_chain_spin_k422", {"plus_u": U_CU}),
    ("cuo_chain_spin_k422", {"onsite_corrections": ONS}),
    ("cuo_chain_soc_k422", {"plus_u": U_CU}),
])
def test_periodic_forces_and_stress_match_energy_differences(name, opts):
    """Non-Gamma k-points: the block potentials' dS terms in forces and stress (1000 K: at
    500 K the +U SCC of the chain on this grid needs over 200 iterations)."""
    _, sk, labels, pos, cell, _, base = _system(name)
    calc = DFTB(sk, kpts=monkhorst_pack((2, 1, 1)), smearing=("fermi", 1000.0),
                scc_tol=1e-11, max_scc=500, **base, **opts)
    r = calc.calculate(labels, pos, cell, np.ones(3, bool))
    assert _fd_force(calc, labels, pos, cell, 0, 0) == pytest.approx(r.forces[0, 0], abs=2e-8)
    assert _fd_force(calc, labels, pos, cell, 1, 2) == pytest.approx(r.forces[1, 2], abs=2e-8)
    for i, j in ((0, 0), (0, 1)):
        assert _fd_stress(calc, labels, pos, cell, i, j) == pytest.approx(r.stress[i, j], abs=1e-10)


# ---- band states --------------------------------------------------------------------
@pytest.mark.parametrize("name, opts", [
    ("cuo_chain_spin_k422", {"plus_u": U_CU}),
    ("h2o", {"onsite_corrections": ONS}),
    ("cuo_chain_soc_k422", {"plus_u": U_CU}),
])
def test_eigenstates_reproduce_the_scc_levels(name, opts):
    """The block potentials (and the onsite corrections, as DFTB+'s band run that reads its
    charges) in H(k) at frozen charges."""
    case, sk, labels, pos, cell, kpts, base = _system(name)
    pbc = None if cell is None else np.ones(3, bool)
    calc = DFTB(sk, kpts=kpts, smearing=tuple(case.filling), scc_tol=1e-11, max_scc=500,
                **base, **opts)
    r = calc.calculate(labels, pos, cell, pbc, forces=False)
    st = eigenstates(calc, labels, pos, cell, pbc, r.q_in)
    ev = np.array(r.eigenvalues).reshape(st.eigenvalues.shape)
    assert np.allclose(st.eigenvalues, ev, atol=1e-9)
    with pytest.raises(ValueError, match="entries"):
        eigenstates(calc, labels, pos, cell, pbc, r.q_in[:-1])


# ---- interfaces ----------------------------------------------------------------------
def test_api_restarts_from_the_block_vector():
    from ase.build import molecule
    params = ParamSet(SKF_HCOCU, max_l={"O": 1, "H": 0})
    mf = SCC(System.from_ase(molecule("H2O")), params, scc_tol=1e-10, onsite_corrections=ONS)
    e = mf.kernel()
    f = mf.forces()                          # restarts from the block vector
    assert mf.result.n_scc <= 2 and mf.e_tot == pytest.approx(e, abs=1e-9)
    assert f.shape == (3, 3) and mf.result.e_onsite > 0


def test_ase_calculator_passes_the_options():
    from ase.build import molecule
    atoms = molecule("H2O")
    atoms.calc = PyDFTB(SKF_HCOCU, max_l={"O": 1, "H": 0}, scc_tol=1e-10, reuse_charges=True,
                        plus_u={"O": [("p", 0.1)]}, plus_u_functional="psic",
                        onsite_corrections=ONS)
    e = atoms.get_potential_energy()
    assert atoms.calc.raw.e_plus_u < 0 and atoms.calc.raw.e_onsite > 0
    atoms.positions[0, 2] += 0.01
    atoms.get_potential_energy()             # reuse_charges: the block vector as q_init
    assert atoms.calc.raw.n_scc < 10 and atoms.get_potential_energy() != e


def test_sp2_solver_with_onsite_corrections():
    _, sk, labels, pos, *_ = _system("h2o")
    ref = DFTB(sk, scc_tol=1e-10, onsite_corrections=ONS).calculate(labels, pos)
    sp2 = DFTB(sk, scc_tol=1e-10, solver="sp2", onsite_corrections=ONS).calculate(labels, pos)
    assert sp2.energy == pytest.approx(ref.energy, abs=1e-8)
    assert np.allclose(sp2.forces, ref.forces, atol=1e-7)


# ---- errors ------------------------------------------------------------------------
def test_option_errors():
    _, sk, labels, pos, *_ = _system("oh_noncolinear_soc")
    with pytest.raises(ValueError, match="need SCC"):
        DFTB(sk, scc=False, plus_u={"O": [("p", 0.1)]})
    with pytest.raises(ValueError, match="need SCC"):
        DFTB(sk, scc=False, onsite_corrections=ONS)
    with pytest.raises(ValueError, match="'fll' or 'psic'"):
        DFTB(sk, plus_u={"O": [("p", 0.1)]}, plus_u_functional="lda")
    with pytest.raises(ValueError, match="dual spin-orbit model"):
        DFTB(sk, plus_u={"O": [("p", 0.1)]}, spin_orbit={"O": [0.0, 0.01]},
             spin_orbit_dual=False)
    with pytest.raises(ValueError, match="too small"):
        DFTB(sk, plus_u={"O": [("p", 0.0)]})
    with pytest.raises(ValueError, match="no entry for 'H'"):
        DFTB(sk, onsite_corrections={"O": ONSITE_PBE["O"]}).calculate(labels, pos)


def test_torch_and_autodiff_refuse():
    pytest.importorskip("torch")
    from pydftb.autodiff import evaluate
    _, sk, labels, pos, *_ = _system("h2o")
    with pytest.raises(ValueError, match="NumPy backend only"):
        DFTB(sk, backend="torch", onsite_corrections=ONS)
    with pytest.raises(NotImplementedError, match="DFTB\\+U or onsite"):
        evaluate(DFTB(sk, onsite_corrections=ONS), labels, pos)
