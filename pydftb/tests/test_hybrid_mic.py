"""MinimumImage and the NeighbourBased Threshold of periodic hybrids at k-points (milestone
M6h-3c).

Comparisons with DFTB+ are the frozen references *_mic*, *_neighbour_t6*, *_neighbour_t4 and
their band runs and restart in test_reference.py, and the screened optimisation in
test_md_opt.py; here the Wigner-Seitz vectors, DFTB+'s screening groups, the explicit-cell
exchange against the grid, self-adjointness (the NeighbourBased MinimumImage is not: B22), the
MatrixBased MinimumImage forces DFTB+ refuses (against differences of the energy), spin, the
screened history and the refusals (no dftb+).
"""
import numpy as np
import pytest
from ase.build import bulk

from pydftb import hybrid as hy
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.kpoints import supercell_folding
from pydftb.validation import dftbplus

from .test_hybrid_kpoints import PBC, SPIN, _diamond, _o2_box, _sk

FCC = bulk("C", "diamond", a=3.567).cell.array / BOHR
MIC = {"type": "lc", "coulomb": "minimum_image"}
MIC_N = {**MIC, "screening": "neighbour"}
NEIGH = {"type": "lc", "screening": "neighbour"}


def _terms(hybrid, n=(2, 2, 2), build=_diamond, **opts):
    labels, pos, cell = build()
    calc = DFTB(_sk(labels), kpts=supercell_folding(n), hybrid=hybrid, **opts)
    return calc._setup(labels, pos, cell, PBC, False, {}).hybrid


def _random_dp(t, seed=1, scale=0.05):
    """A DeltaP(k) of a real, symmetric DeltaP(R) (X(-k) = conj X(k))."""
    lay = t.layout
    b = np.random.default_rng(seed).normal(scale=scale, size=(lay.ncell, t.n, t.n))
    return lay.from_bvk(0.5 * (b + b[lay.cell_index(-lay.cells)].transpose(0, 2, 1)))


def _inner(t, a, b):
    return float(np.einsum("k,kij->", t.layout.kw, a * b.conj()).real)


# ---- Wigner-Seitz vectors and DFTB+'s screening groups --------------------------------------
@pytest.mark.parametrize("n, count", [((2, 2, 2), 19), ((3, 1, 2), 11), ((3, 3, 3), 43)])
def test_wigner_seitz_vectors(n, count):
    """Every class of the supercell, the ties on the cell's surface each counted (B9), symmetric,
    by length; DFTB+'s counts for the FCC cell."""
    ws = hy.wigner_seitz_vectors(n, FCC)
    assert len(ws) == count
    assert len({tuple(np.mod(v, n)) for v in ws}) == np.prod(n)
    assert {tuple(v) for v in ws} == {tuple(-v) for v in ws}
    length = np.linalg.norm(ws @ FCC, axis=1)
    assert np.all(np.diff(length) >= 0.0) and not np.any(ws[0])


def test_wigner_seitz_reduction():
    """N' = max(N - r, 1): the home cell alone at N' = 1; a cubic cell keeps all 27 ties."""
    for r in (1, 2, 7):
        assert hy.wigner_seitz_vectors((2, 2, 2), FCC, r).tolist() == [[0, 0, 0]]
    assert len(hy.wigner_seitz_vectors((3, 2, 2), FCC, 1)) == len(
        hy.wigner_seitz_vectors((2, 1, 1), FCC))
    assert len(hy.wigner_seitz_vectors((2, 2, 2), 5.0 * np.eye(3))) == 27


@pytest.mark.parametrize("tau", [1e-6, 1e-3, 0.05, 0.5, 2.0])
def test_screening_groups_are_dftbplus_quadruples(tau):
    """The groups cover exactly the pairs of pair entries with est_p est_q >= tau
    (getFourLoopCompositeIndex), the on-site with est 1."""
    t = _terms({**NEIGH, "neighbour_threshold": tau}, n=(3, 1, 2))
    groups = t.screening_groups(1.0)
    est = np.append(t._est, 1.0)                                  # last: the on-site
    keep = np.append(t._pairs.dist < t._cam_cutoff, True)
    want = keep[:, None] & keep[None, :] & (est[:, None] * est[None, :] >= tau)
    got = np.zeros_like(want)
    for pp, po, qp, qo in groups:
        p = np.append(pp, po)
        q = np.append(qp, qo)
        assert not (got[p][:, q]).any()                          # each pair of entries once
        got |= p[:, None] & q[None, :]
    assert np.array_equal(got, want)


# ---- the operators ------------------------------------------------------------------------
@pytest.mark.parametrize("hybrid, n", [(MIC, (3, 1, 2)), (MIC_N, (2, 2, 2)),
                                       ({**MIC_N, "wigner_seitz_reduction": 1}, (2, 1, 2)),
                                       ({**MIC_N, "neighbour_threshold": 1e-4}, (2, 2, 2)),
                                       ({**NEIGH, "neighbour_threshold": 1e-4}, (3, 1, 2))])
def test_band_states_on_the_grid_are_the_scc_ones(hybrid, n):
    """fock_at on the SCC's k-points is fock (the explicit cells' own phases and the gauge of the
    folded atoms; with a Threshold one build of all of DeltaP)."""
    t = _terms(hybrid, n)
    dps = [_random_dp(t)]
    t.reset()
    for a, b in zip(t.fock_at(t.layout.kpts, dps), t.fock(dps)):
        assert np.abs(b).max() > 1e-3 and np.abs(a - b).max() < 1e-12


def test_self_adjointness():
    """<H[P], Q> = <H[Q], P> for the truncated operators and the MatrixBased MinimumImage; the
    NeighbourBased MinimumImage restricts only the A-B density of each path to the Wigner-Seitz
    cell, so it is not (B22: its exchange matrix is not the derivative of its energy)."""
    for hybrid, symmetric in (("lc", True), (NEIGH, True), (MIC, True), (MIC_N, False)):
        t = _terms(hybrid)
        p, q = _random_dp(t, 1), _random_dp(t, 2)
        a, b = _inner(t, t.fock([p])[0], q), _inner(t, t.fock([q])[0], p)
        assert (abs(a - b) < 1e-12 * abs(a)) == symmetric, hybrid


def _o2_off_faces():
    """The O2 box with no atom on a face of the cell: there the Wigner-Seitz sums jump as DFTB+
    folds the atom into the cell (as the untruncated kernel's, B19)."""
    labels, pos, cell = _o2_box()
    return labels, pos + 0.3 * cell.sum(0), cell


FD_CASES = {
    "matrix": (_diamond, (3, 1, 2), {"hybrid": MIC}),
    "matrix_ws1": (_diamond, (2, 2, 2), {"hybrid": {**MIC, "wigner_seitz_reduction": 1}}),
    "matrix_spin": (_o2_off_faces, (2, 1, 1), {**SPIN, "hybrid": MIC}),
}


@pytest.mark.parametrize("name", list(FD_CASES))
def test_matrix_mic_forces_are_the_energy_derivative(name):
    """The MatrixBased MinimumImage (forces and spin DFTB+ 25.1 refuses): central differences
    (h = 5e-5 Bohr) of the energy, away from the cell's faces.  The differences themselves are
    good to 1-2e-9 here (the energies' SCC error over h; NumPy builds differ), hence 5e-9."""
    build, n, opts = FD_CASES[name]
    labels, pos, cell = build()
    calc = DFTB(_sk(labels), kpts=supercell_folding(n), scc_tol=1e-12, max_scc=500,
                temperature=300.0 if "spin" in opts else 0.0, **opts)
    r = calc.calculate(labels, pos, cell, PBC)
    h, fd = 5e-5, np.zeros_like(pos)
    for a in range(len(pos)):
        for x in range(3):
            e = []
            for s in (1, -1):
                p = pos.copy()
                p[a, x] += s * h
                e.append(calc.calculate(labels, p, cell, PBC, forces=False).free_energy)
            fd[a, x] = -(e[0] - e[1]) / (2 * h)
    assert np.abs(r.forces - fd).max() < 5e-9


@pytest.mark.parametrize("hybrid", [MIC, {**MIC_N, "wigner_seitz_reduction": 1}])
def test_closed_shell_spin_is_unpolarised(hybrid):
    """Spin with MinimumImage (DFTB+ refuses it): a closed shell gives the unpolarised result."""
    labels, pos, cell = _diamond()
    k = supercell_folding((2, 1, 2))
    plain = DFTB(_sk(labels), kpts=k, hybrid=hybrid, scc_tol=1e-11)
    spin = DFTB(_sk(labels), kpts=k, hybrid=hybrid, scc_tol=1e-11, spin=True,
                spin_constants={"C": -0.023})
    a = plain.calculate(labels, pos, cell, PBC, forces=False)
    b = spin.calculate(labels, pos, cell, PBC, forces=False)
    assert abs(a.energy - b.energy) < 1e-9 and np.abs(b.magmoms).max() < 1e-8


# ---- the screened history -------------------------------------------------------------------
@pytest.mark.parametrize("hybrid", [NEIGH, {**MIC_N, "wigner_seitz_reduction": 1}])
def test_threshold_zero_is_the_unscreened_build(hybrid):
    labels, pos, cell = _diamond()
    k = supercell_folding((3, 1, 2))
    e = [DFTB(_sk(labels), kpts=k, hybrid={**hybrid, **extra}, scc_tol=1e-10).calculate(
        labels, pos, cell, PBC, forces=False).energy
        for extra in ({}, {"neighbour_threshold": 1e-100})]
    assert abs(e[0] - e[1]) < 1e-12


def test_screened_increments():
    """H accumulates the screened builds of the increments: a vanishing threshold gives the
    whole operator, an unchanged DeltaP returns zero (DFTB+, B7), a huge threshold keeps
    nothing, reset starts again."""
    t = _terms({**NEIGH, "neighbour_threshold": 1e-100}, n=(3, 1, 2))
    exact = _terms(NEIGH, n=(3, 1, 2))
    d1, d2 = _random_dp(t, 1), _random_dp(t, 2)
    t.fock([d1])
    h2 = t.fock([d2])[0]
    assert np.abs(h2 - exact.fock([d2])[0]).max() < 1e-12
    assert not np.any(t.fock([d2])[0])                             # B7
    t.reset()
    assert np.abs(t.fock([d2])[0] - h2).max() < 1e-12
    big = _terms({**NEIGH, "neighbour_threshold": 1e10}, n=(3, 1, 2))
    assert not np.any(big.fock([d1])[0])
    screened = _terms({**NEIGH, "neighbour_threshold": 1e-3}, n=(3, 1, 2))
    a = screened.fock([d1])[0]
    b = screened.fock([d2])[0]                                     # history: d1, then d2
    screened.reset()
    c = screened.fock([d2])[0]                                     # d2 alone
    assert np.abs(b - c).max() > 1e-6 and np.abs(a).max() > 1e-3


def test_every_geometry_starts_a_whole_build():
    """DFTB+ updateCoords: a new geometry forgets the screened history and its first build is a
    whole one of the DeltaP it starts from (as a restart from a charge file, the reference
    *_t6_read_charges): taking the step before's q_in equals a fresh calculator started from it,
    and the result depends on that start."""
    labels, pos, cell = _diamond()
    opts = dict(kpts=supercell_folding((3, 1, 2)), scc_tol=1e-7,
                hybrid={**NEIGH, "neighbour_threshold": 1e-6})
    calc = DFTB(_sk(labels), **opts)
    r1 = calc.calculate(labels, pos, cell, PBC, forces=False)
    moved = pos + np.array([[0.02, -0.01, 0.015], [0.0, 0.0, 0.0]])
    r2 = calc.calculate(labels, moved, cell, PBC, forces=False, q_init=r1.q_in)
    r3 = DFTB(_sk(labels), **opts).calculate(labels, moved, cell, PBC, forces=False,
                                             q_init=r1.q_in)
    r4 = DFTB(_sk(labels), **opts).calculate(labels, moved, cell, PBC, forces=False)
    assert r2.energy == r3.energy and r2.n_scc == r3.n_scc
    assert abs(r3.energy - r4.energy) > 1e-9


# ---- inputs and refusals -----------------------------------------------------------------------
def test_hybrid_options():
    with pytest.raises(ValueError, match="non-negative integer"):
        hy.Hybrid.make({**MIC, "wigner_seitz_reduction": -1})
    with pytest.raises(ValueError, match="non-negative integer"):
        hy.Hybrid.make({**MIC, "wigner_seitz_reduction": 1.5})
    with pytest.raises(ValueError, match="MinimumImage block only"):
        hy.Hybrid.make({"type": "lc", "wigner_seitz_reduction": 1})
    with pytest.raises(ValueError, match="neighbour screening"):
        hy.Hybrid.make({"type": "lc", "neighbour_threshold": 1e-6})
    with pytest.raises(ValueError, match="must not be negative"):
        hy.Hybrid.make({**NEIGH, "neighbour_threshold": -1.0})
    for cutoff in ("coulomb_cutoff", "gsum_cutoff"):
        with pytest.raises(ValueError, match="reads neither in the MinimumImage block"):
            hy.Hybrid.make({**MIC, cutoff: 8.0})
    block = dftbplus.hybrid_block({**MIC_N, "wigner_seitz_reduction": 2,
                                   "neighbour_threshold": 1e-6}, kpoints=True)
    assert "MinimumImage {\n      WignerSeitzReduction = 2\n    }" in block
    assert "Threshold = 1e-06" in block or "Threshold = 1.0e-06" in block
    assert "WignerSeitzReduction" not in dftbplus.hybrid_block(MIC, kpoints=True)


def test_no_stop_on_the_unused_cutoff():
    """DFTB+ computes its default Coulomb cutoff for MinimumImage too and stops where that is
    undefined (B3: a2 along z) though the Wigner-Seitz sums never use it; pydftb goes on, with
    the energy of the same system turned so that DFTB+ can run it."""
    bad = np.array([[9.0, 0.0, 0.0], [0.0, 0.0, 9.5], [0.0, -9.2, 0.0]]) / BOHR   # a2 along z
    labels, pos = ["C", "O"], np.array([[0.3, -0.2, 0.1], [1.42, -0.35, 0.25]]) / BOHR
    c, s = np.cos(0.4), np.sin(0.4)
    turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    k = supercell_folding((2, 1, 1))
    with pytest.raises(ValueError, match="positive radius"):
        hy.dftbp_coulomb_cutoff(bad, (2, 1, 1))
    assert hy.dftbp_coulomb_cutoff(bad @ turn.T, (2, 1, 1)) > 1.0
    for hybrid in (MIC, {**MIC_N, "wigner_seitz_reduction": 1}):
        calc = DFTB(_sk(labels), kpts=k, scc_tol=1e-10, hybrid=hybrid)
        e = [calc.calculate(labels, p, a, PBC, forces=False).energy
             for p, a in ((pos, bad), (pos @ turn.T, bad @ turn.T))]
        assert abs(e[0] - e[1]) < 1e-9


def test_refusals():
    labels, pos, cell = _diamond()
    k = supercell_folding((2, 1, 1))
    with pytest.raises(ValueError, match="setting 'MinimumImage'"):
        DFTB(_sk(labels), kpts=k, hybrid=MIC_N).calculate(labels, pos, cell, PBC)
    with pytest.raises(ValueError, match="Truncated\\+Damping"):            # Gamma point
        DFTB(_sk(labels), hybrid=MIC).calculate(labels, pos, cell, PBC, forces=False)
    mol = np.array([[0.0, 0.0, 0.0], [2.2, 0.0, 0.0]])
    with pytest.raises(ValueError, match="Threshold of periodic systems"):
        DFTB(_sk(labels), hybrid={**NEIGH, "neighbour_threshold": 1e-6}).calculate(labels, mol)
