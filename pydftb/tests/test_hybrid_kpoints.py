"""Range-separated hybrid functionals of periodic systems at the k-points of a supercell folding
(milestone M6h-3b).

Comparisons with DFTB+ are the frozen references diamond2_*_k*, co2_crystal_lc_k211*,
cu_fcc_lc_k333_1000K, o2_box_lc_spin_k*, co_triclinic_lc_k222 (with DFTB+'s B20 reproduced by
the validation switch), their restarts and band runs, in test_reference.py, and the dynamics in
test_md_opt.py; here the k-point sets, DFTB+'s default cutoff and folding of the atoms, the BvK
transforms, the operator and the calculation against the supercell's Gamma point,
self-consistency, the forces DFTB+ refuses (against differences of the energy), the charge
files, the band states, ASE and the refusals (no dftb+).
"""
from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk

from pydftb import hybrid as hy
from pydftb.ase_calc import PyDFTB
from pydftb.bands import eigenstates
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.dftbplus_io import read_charges, write_charges
from pydftb.kpoints import folding_of, monkhorst_pack, reduce_time_reversal, supercell_folding
from pydftb.params import ParamSet
from pydftb.validation import dftbplus
from pydftb.validation.cases import SKF_SETS

PBC = np.ones(3, bool)
MAXL = {"H": 0, "C": 1, "O": 1, "Cu": 2}


def _sk(labels, skf="ptbp_hcocu_lc"):
    return ParamSet(SKF_SETS[skf], max_l={lab: MAXL[lab] for lab in set(labels)}).skset(labels)


def _diamond():
    """Diamond's primitive cell, rattled (atom 0 at a negative fractional coordinate)."""
    at = bulk("C", "diamond", a=3.567)
    at.rattle(0.03, seed=3)
    return ["C", "C"], at.positions / BOHR, at.cell.array / BOHR


def _co_box():
    """CO in a box beyond the SK cutoff (15.4 Bohr): no atom overlaps its own images, S_mumu(k)
    = 1, so P0(k) is the supercell's P0 and the k-point calculation the supercell's Gamma
    point."""
    pos = np.array([[0.3, 0.2, 0.1], [1.42, 0.35, 0.25]]) / BOHR
    return ["C", "O"], pos, np.diag([9.0, 9.3, 9.6]) / BOHR


def _o2_box():
    pos = np.array([[0.0, 0.0, 0.0], [1.21, 0.05, 0.03]]) / BOHR
    pos[0] -= 0.1 * np.array([0.0, 5.2, 0.0]) / BOHR                   # outside the cell
    return ["O", "O"], pos, np.diag([5.0, 5.2, 5.4]) / BOHR


SPIN = dict(spin=True, unpaired_electrons=2.0, spin_constants={"O": -0.028})


def _supercell(pos, cell, n):
    """The supercell diag(n) of the atoms, its copies in DFTB+'s cell order (n1 fastest)."""
    cells = hy.bvk_cells(n)
    return (np.concatenate([pos + c @ cell for c in cells]), np.array(n)[:, None] * cell,
            cells)


# ---- k-point sets, DFTB+'s default cutoff and folding --------------------------------------
@pytest.mark.parametrize("n", [(2, 2, 2), (3, 3, 3), (1, 2, 3), (3, 1, 2), (4, 4, 4), (2, 1, 1)])
def test_supercell_folding_is_dftbplus_set(n):
    """DFTB+'s getSuperSampling with inversion: i1 slowest, k and -k merged into the first,
    the time-reversal reduction of the Gamma-centred grid; the folding recognised from it, from
    the whole grid and from any representatives of the pairs."""
    k, w = supercell_folding(n)
    ref = reduce_time_reversal(*monkhorst_pack(n, gamma_centered=True))
    assert np.array_equal(k, ref[0]) and np.allclose(w, ref[1]) and w.sum() == pytest.approx(1)
    assert folding_of(k, w) == n
    assert folding_of(*monkhorst_pack(n, gamma_centered=True)) == n
    flipped = np.where((np.arange(len(k)) % 2 == 1)[:, None], np.mod(-k, 1.0), k)
    assert folding_of(flipped[::-1], w[::-1]) == n


def test_folding_of_refuses_other_sets():
    assert folding_of(*monkhorst_pack((2, 2, 2))) is None                   # shifted
    assert folding_of(np.array([[0.0, 0.0, 0.25]]), np.ones(1)) is None
    assert folding_of(np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), np.ones(2)) is None
    k = supercell_folding((3, 3, 3))[0]
    assert folding_of(k, np.ones(len(k))) is None                           # wrong weights


def test_dftbplus_default_cutoff():
    """getCoulombTruncationCutoff as coded: the Gamma point's half plane spacing; with a
    folding the Cartesian components scaled (B2: FCC 1x2x3 2.88885 Bohr, the supercell's half
    spacing 1.94586); a2 along z stops DFTB+ (B3), at the Gamma point too."""
    _, _, cell = _diamond()
    assert hy.dftbp_coulomb_cutoff(cell) == pytest.approx(hy.default_coulomb_cutoff(cell),
                                                          rel=1e-14)
    assert hy.dftbp_coulomb_cutoff(cell, (1, 2, 3)) == pytest.approx(2.88885111, abs=1e-8)
    assert hy.default_coulomb_cutoff(np.array([1, 2, 3])[:, None] * cell) == pytest.approx(
        1.94586, abs=1e-5)
    assert hy.dftbp_coulomb_cutoff(cell, (2, 2, 2)) == pytest.approx(
        2 * hy.dftbp_coulomb_cutoff(cell), rel=1e-13)
    bad = np.array([[9.0, 0.0, 0.0], [0.0, 0.0, 9.5], [0.0, -9.2, 0.0]]) / BOHR   # a2 along z
    for nk in [(1, 1, 1), (2, 1, 1)]:
        with pytest.raises(ValueError, match="positive radius.*undefined for this cell"):
            hy.dftbp_coulomb_cutoff(bad, nk)
    labels, pos = ["C", "O"], np.array([[0.3, -0.2, 0.1], [1.42, -0.35, 0.25]]) / BOHR
    sk = _sk(labels)
    for kpts in (None, supercell_folding((2, 1, 1))):
        with pytest.raises(ValueError, match="positive radius"):
            DFTB(sk, kpts=kpts, hybrid="lc").calculate(labels, pos, bad, PBC, forces=False)
        DFTB(sk, kpts=kpts, scc_tol=1e-6, hybrid={"type": "lc", "coulomb_cutoff": 8.0}).calculate(
            labels, pos, bad, PBC, forces=False)


def test_dftbplus_folding_of_the_atoms():
    """foldCoordsToCell: floor of the fractional coordinates, and an atom within eps of the far
    face goes to the near one."""
    cell = np.diag([1.0, 6.0, 7.0])
    frac = np.array([[0.2, -0.1, 1.3], [0.999, 0.0, -2.5], [np.nextafter(1.0, 0.0), 0.5, 0.5]])
    t = hy.dftbp_fold_shifts(frac @ cell, cell)
    assert t.tolist() == [[0, -1, 1], [0, 0, -3], [1, 0, 0]]


# ---- the BvK transforms and the mixed vector ----------------------------------------------
@pytest.mark.parametrize("full", [False, True], ids=["reduced", "whole grid"])
def test_bvk_transforms_and_packing(full):
    n = (3, 1, 2)
    k, w = monkhorst_pack(n, gamma_centered=True) if full else supercell_folding(n)
    rng = np.random.default_rng(3)
    nb = 5
    lay = hy.KPointLayout(k, w, n, [0, 0, 0, 1, 1], "matrix")
    b = rng.normal(size=(lay.ncell, nb, nb))
    x = lay.from_bvk(b)
    assert np.abs(lay.to_bvk(x) - b).max() < 1e-13                 # N cells <-> the grid
    assert np.abs(lay.unpack(lay.pack([x, 2 * x]), 2)[1] - 2 * x).max() == 0.0
    neigh = hy.KPointLayout(k, w, n, [0, 0, 0, 1, 1], "neighbour")
    assert neigh.packed_size == lay.ncell * nb * nb and lay.packed_size == 2 * len(k) * nb * nb
    assert np.abs(neigh.unpack(neigh.pack([x]), 1)[0] - x).max() < 1e-13
    d = lay.pack([x]) - lay.pack([x + 1e-3j])
    assert lay.error(d, 0 * d) == pytest.approx(1e-3, rel=1e-12)


def test_dftbplus_records_round_trip():
    """DFTB+'s records are of the folded atoms: D^dag DeltaP(k) D at DFTB+'s k-points, the BvK
    array's cells shifted by T_B - T_A; back to any k-point set of the folding and either
    screening."""
    n = (3, 1, 2)
    rng = np.random.default_rng(4)
    shifts = np.array([[-1, 0, 0], [0, 1, -1]])
    oa = [0, 0, 1, 1, 1]
    b = rng.normal(size=(6, 5, 5))
    for k, w in (supercell_folding(n), monkhorst_pack(n, gamma_centered=True)):
        mat = hy.KPointLayout(k, w, n, oa, "matrix", shifts)
        nei = hy.KPointLayout(k, w, n, oa, "neighbour", shifts)
        x = mat.from_bvk(b)
        rk, rb = mat.to_dftbp([x]), nei.to_dftbp([x])
        assert rk.shape == (1, len(supercell_folding(n)[0]), 5, 5) and rb.shape == (1, 6, 5, 5)
        for lay in (mat, nei):
            assert np.abs(lay.from_dftbp(rk, bvk=False)[0] - x).max() < 1e-13
            assert np.abs(lay.from_dftbp(rb, bvk=True)[0] - x).max() < 1e-13
    assert not np.allclose(rb[0], b)                                  # the gauge moved cells


# ---- the operator and the calculation against the supercell's Gamma point --------------------
@pytest.mark.parametrize("hybrid", [
    {"type": "lc", "coulomb_cutoff": 6.0, "gsum_cutoff": 12.0},
    {"type": "lc", "coulomb": "damped", "coulomb_cutoff": 6.0},
    {"type": "lc", "screening": "neighbour", "cutoff_reduction": 4.0, "coulomb_cutoff": 6.0},
], ids=["truncated", "damped", "neighbour"])
def test_operator_is_the_supercells_gamma_point(hybrid):
    """H_x(k) of a fixed DeltaP(k) is, cell by cell, M6h-3a's Gamma-point operator of the BvK
    supercell with the same density: the convolution over q is the supercell's real-space
    product."""
    labels, pos, cell = _diamond()
    n = (2, 1, 2)
    sk = _sk(labels)
    st = DFTB(sk, kpts=supercell_folding(n), hybrid=hybrid)._setup(labels, pos, cell, PBC, False,
                                                                   {})
    terms, lay = st.hybrid, st.hybrid.layout
    nb = terms.n
    b = np.random.default_rng(1).normal(size=(lay.ncell, nb, nb))
    b = 0.5 * (b + b[lay.cell_index(-lay.cells)].transpose(0, 2, 1))   # a symmetric supercell DP
    hb = lay.to_bvk(terms.fock([lay.from_bvk(b)])[0])
    sc_pos, sc_cell, cells = _supercell(pos, cell, n)
    st_sc = DFTB(sk, hybrid=hybrid)._setup(labels * lay.ncell, sc_pos, sc_cell, PBC, False, {})
    d_sc = np.block([[b[lay.cell_index(c2 - c1)] for c2 in cells] for c1 in cells])
    h_sc = st_sc.hybrid.fock([d_sc])[0]
    assert np.abs(h_sc[:nb].reshape(nb, lay.ncell, nb).transpose(1, 0, 2) - hb).max() < 1e-12


def test_kpoints_are_the_supercells_gamma_point():
    """Without self-image overlaps the whole calculation is the supercell's Gamma point: N times
    the energy, the same forces, charges and exchange energy (DFTB+'s E_Fock of DeltaP_out at
    k-points, of DeltaP_in at Gamma: equal when converged)."""
    labels, pos, cell = _co_box()
    n = (2, 1, 1)
    hybrid = {"type": "lc", "coulomb_cutoff": 7.0, "gsum_cutoff": 14.0}
    sk = _sk(labels)
    calc = DFTB(sk, kpts=supercell_folding(n), hybrid=hybrid, scc_tol=1e-11)
    assert np.abs(calc._setup(labels, pos, cell, PBC, False, {}).hybrid.s_diag - 1.0).max() == 0
    r = calc.calculate(labels, pos, cell, PBC)
    sc_pos, sc_cell, _ = _supercell(pos, cell, n)
    r_sc = DFTB(sk, hybrid=hybrid, scc_tol=1e-11).calculate(labels * 2, sc_pos, sc_cell, PBC)
    assert r_sc.energy == pytest.approx(2 * r.energy, abs=1e-9)
    assert r_sc.e_fock == pytest.approx(2 * r.e_fock, abs=1e-9)
    assert np.abs(r_sc.forces - np.tile(r.forces, (2, 1))).max() < 1e-8
    assert np.abs(r_sc.charges - np.tile(r.charges, 2)).max() < 1e-8


def test_block_populations_follow_deltap_unless_the_dftbplus_switch():
    """+U at k-points takes the blocks of DeltaP_in(k) + P0(k) every iteration, which the
    supercell's Gamma point confirms; DFTB+ 25.1 keeps the starting blocks (B20,
    DFTB._dftbp_stale_blocks): another energy."""
    labels, pos, cell = _co_box()
    n = (2, 1, 1)
    opts = dict(hybrid={"type": "lc", "coulomb_cutoff": 7.0}, plus_u={"O": [("p", 0.1)]},
                scc_tol=1e-11)
    sk = _sk(labels)
    r = DFTB(sk, kpts=supercell_folding(n), **opts).calculate(labels, pos, cell, PBC,
                                                              forces=False)
    sc_pos, sc_cell, _ = _supercell(pos, cell, n)
    r_sc = DFTB(sk, **opts).calculate(labels * 2, sc_pos, sc_cell, PBC, forces=False)
    assert r_sc.energy == pytest.approx(2 * r.energy, abs=1e-9)
    stale = DFTB(sk, kpts=supercell_folding(n), **opts)
    stale._dftbp_stale_blocks = True
    rs = stale.calculate(labels, pos, cell, PBC, forces=False)
    assert abs(rs.energy - r.energy) > 1e-5 and abs(rs.e_plus_u - r.e_plus_u) > 1e-5


def test_screenings_and_kpoint_sets_agree():
    """NeighbourBased without a reduction is the matrix algorithm; the whole grid and DFTB+'s
    set give the same results; a closed-shell spin calculation the unpolarised one."""
    labels, pos, cell = _diamond()
    sk = _sk(labels)
    k, w = supercell_folding((3, 1, 2))
    base = DFTB(sk, kpts=(k, w), hybrid="lc", scc_tol=1e-11).calculate(labels, pos, cell, PBC)
    others = [
        DFTB(sk, kpts=(k, w), hybrid={"type": "lc", "screening": "neighbour"}, scc_tol=1e-11),
        DFTB(sk, kpts=monkhorst_pack((3, 1, 2), gamma_centered=True), hybrid="lc",
             scc_tol=1e-11),
        DFTB(sk, kpts=(k, w), hybrid="lc", scc_tol=1e-11, spin=True,
             spin_constants={"C": [[-0.031, -0.025], [-0.025, -0.023]]}),
    ]
    for calc in others:
        r = calc.calculate(labels, pos, cell, PBC)
        assert r.energy == pytest.approx(base.energy, abs=1e-9)
        assert np.abs(r.forces - base.forces).max() < 1e-8


# ---- forces, also where DFTB+ refuses them ----------------------------------------------------
FD_CASES = {
    "truncated": (_diamond, (2, 2, 2), {}),
    "neighbour": (_diamond, (3, 1, 2), {"hybrid": {"type": "lc", "screening": "neighbour",
                                                    "cutoff_reduction": 4.0}}),
    "damped": (_diamond, (2, 1, 2), {"hybrid": {"type": "lc", "coulomb": "damped"}}),
    "cam": (_diamond, (1, 2, 2), {"hybrid": "cam", "skf": "ptbp_hcocu_cam"}),
    "spin": (_o2_box, (2, 1, 1), SPIN),
    "spin_neighbour": (_o2_box, (1, 2, 1), {**SPIN, "hybrid": {"type": "lc",
                                                               "screening": "neighbour"}}),
    "spin_damped": (_o2_box, (2, 1, 1), {**SPIN, "hybrid": {"type": "lc", "coulomb": "damped"}}),
}


@pytest.mark.parametrize("name", list(FD_CASES))
def test_forces_are_the_energy_derivative(name):
    """Central differences (h = 5e-5 Bohr) of the energy; DFTB+ 25.1 refuses the forces of the
    damped kernel and of spin at k-points."""
    build, n, opts = FD_CASES[name]
    opts = dict(opts)
    skf = opts.pop("skf", "ptbp_hcocu_lc")
    opts.setdefault("hybrid", "lc")
    labels, pos, cell = build()
    calc = DFTB(_sk(labels, skf), kpts=supercell_folding(n), scc_tol=1e-12, max_scc=500,
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
    assert np.abs(r.forces - fd).max() < 2e-9


def test_full_kernel_gamma_gradient():
    """The untruncated kernel (whose SCC does not settle): the gamma part of the gradient at a
    fixed DeltaP and overlap against differences of E_x."""
    labels, pos, cell = _diamond()
    sk = _sk(labels)
    calc = DFTB(sk, kpts=supercell_folding((2, 1, 1)),
                hybrid={"type": "lc", "coulomb": "full", "gsum_cutoff": 10.0})
    terms = calc._setup(labels, pos, cell, PBC, False, {}).hybrid
    lay = terms.layout
    b = np.random.default_rng(2).normal(scale=0.05, size=(lay.ncell, terms.n, terms.n))
    dps = [lay.from_bvk(0.5 * (b + b[lay.cell_index(-lay.cells)].transpose(0, 2, 1)))]

    def e_x(p):
        t = calc._setup(labels, p, cell, PBC, False, {}).hybrid
        t.S_x = terms.S_x
        return t.energy(t.fock(dps), dps)
    h, grad = 1e-5, terms.gamma_gradient(dps)
    fd = np.zeros_like(pos)
    for a in range(len(pos)):
        for x in range(3):
            p1, p2 = pos.copy(), pos.copy()
            p1[a, x] += h
            p2[a, x] -= h
            fd[a, x] = (e_x(p1) - e_x(p2)) / (2 * h)
    assert np.abs(grad).max() > 1e-4 and np.abs(grad - fd).max() < 1e-8


# ---- charge files, restarts and band states ------------------------------------------------
@pytest.mark.parametrize("screening", ["matrix", "neighbour"])
def test_charge_files_and_restarts(tmp_path, screening):
    """DFTB+'s records (DeltaP(k) or the BvK array after the folding diag(N)) written and read
    back, text and binary; a restart takes one iteration; either record serves either
    screening; another folding is DFTB+'s refusal."""
    labels, pos, cell = _diamond()
    sk = _sk(labels)
    k, w = supercell_folding((3, 1, 2))
    calc = DFTB(sk, kpts=(k, w), hybrid={"type": "lc", "screening": screening}, scc_tol=1e-10)
    r = calc.calculate(labels, pos, cell, PBC, forces=False)
    data = calc.charge_data(labels, r, periodic=True, positions=pos, cell=cell)
    assert np.array_equal(data.folding, np.hstack([np.diag([3.0, 1.0, 2.0]), np.zeros((3, 1))]))
    assert (data.rho_bvk is not None) == (screening == "neighbour")
    for name in ("charges.bin", "charges.dat"):
        write_charges(tmp_path / name, data)
        back = read_charges(tmp_path / name, [4, 4])
        q = calc.populations_from_file(labels, back, positions=pos, cell=cell)
        assert np.abs(q - r.q_in).max() < 1e-14
        again = calc.calculate(labels, pos, cell, PBC, forces=False, q_init=back)
        assert again.n_scc == 1 and again.energy == pytest.approx(r.energy, abs=1e-12)
    other = DFTB(sk, kpts=(k, w), scc_tol=1e-10,
                 hybrid={"type": "lc", "screening": "matrix" if screening == "neighbour"
                         else "neighbour"})
    r2 = other.calculate(labels, pos, cell, PBC, forces=False, q_init=back)
    assert r2.n_scc <= 3 and r2.energy == pytest.approx(r.energy, abs=1e-9)
    with pytest.raises(ValueError, match="only identical k-point samplings"):
        DFTB(sk, kpts=supercell_folding((2, 2, 2)), hybrid="lc").populations_from_file(
            labels, back, positions=pos, cell=cell)
    with pytest.raises(ValueError, match="only identical k-point samplings"):
        DFTB(sk, hybrid="lc").populations_from_file(labels, back, positions=pos, cell=cell)
    with pytest.raises(ValueError, match="needs the positions and the cell"):
        calc.charge_data(labels, r, periodic=True)
    with pytest.raises(ValueError, match="passes its positions and cell"):
        calc.populations_from_file(labels, back)


def test_charge_file_errors(tmp_path):
    labels, pos, cell = _diamond()
    calc = DFTB(_sk(labels), kpts=supercell_folding((2, 1, 1)), hybrid="lc", scc_tol=1e-8)
    r = calc.calculate(labels, pos, cell, PBC, forces=False)
    data = calc.charge_data(labels, r, periodic=True, positions=pos, cell=cell)
    with pytest.raises(ValueError, match="needs its supercell folding"):
        write_charges(tmp_path / "c.bin", replace(data, folding=None, kpoint_info=False))
    with pytest.raises(ValueError, match="uniform extension"):
        write_charges(tmp_path / "c.bin", replace(data, folding=np.array(
            [[2.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])))
    with pytest.raises(ValueError, match="one density matrix record"):
        write_charges(tmp_path / "c.bin", replace(data, rho=np.zeros((1, 8, 8))))
    write_charges(tmp_path / "ok.bin", data)                   # a record of another length
    (tmp_path / "c.bin").write_bytes((tmp_path / "ok.bin").read_bytes()[:-8])
    with pytest.raises(ValueError, match="neither DFTB\\+'s MatrixBased"):
        read_charges(tmp_path / "c.bin", [4, 4])


@pytest.mark.parametrize("screening", ["matrix", "neighbour"])
def test_band_states(screening):
    """The exchange at any k from the SCC's DeltaP(q): on the SCC's k-points the SCC's levels
    (fock_at is fock there), and Gamma-only SCCs still refuse other k-points."""
    labels, pos, cell = _o2_box()
    k, w = supercell_folding((2, 1, 2))
    calc = DFTB(_sk(labels), kpts=(k, w), hybrid={"type": "lc", "screening": screening},
                scc_tol=1e-10, **SPIN)
    r = calc.calculate(labels, pos, cell, PBC, forces=False)
    st = eigenstates(calc, labels, pos, cell, PBC, r.q_in, kpts=(k, w))
    ev = np.asarray(r.eigenvalues).reshape(st.eigenvalues.shape)
    assert np.abs(st.eigenvalues - ev).max() < 1e-12
    terms = calc._setup(labels, pos, cell, PBC, False, {}).hybrid
    dps = terms.unpack(r.q_in[-2 * terms.packed_size:], 2)
    for a, b in zip(terms.fock_at(k, dps), terms.fock(dps)):
        assert np.abs(a - b).max() < 1e-13
    path = eigenstates(calc, labels, pos, cell, PBC, r.q_in,
                       kpts=np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.5, 0.0, 0.25]]))
    assert path.eigenvalues.shape == (2, 3, 8)
    gamma = DFTB(_sk(labels), hybrid="lc", scc_tol=1e-8, **SPIN)
    rg = gamma.calculate(labels, pos, cell, PBC, forces=False)
    with pytest.raises(ValueError, match="SCC on a supercell folding"):
        eigenstates(gamma, labels, pos, cell, PBC, rg.q_in, kpts=np.array([[0.0, 0.0, 0.5]]))


# ---- ASE, DFTB+ inputs and refusals -----------------------------------------------------------
def test_ase_gamma_centred_grid():
    """ASE's kpts={"size": n, "gamma": True} is DFTB+'s SupercellFolding without shift, with
    band structures along a path; a shifted grid is DFTB+'s refusal."""
    labels, pos, cell = _diamond()
    ref = DFTB(_sk(labels), kpts=supercell_folding((2, 2, 2)), hybrid="lc",
               scc_tol=1e-10).calculate(labels, pos, cell, PBC, forces=False)
    atoms = Atoms(labels, positions=pos * BOHR, cell=cell * BOHR, pbc=True)
    atoms.calc = PyDFTB(SKF_SETS["ptbp_hcocu_lc"], max_l={"C": 1}, hybrid="lc", scc_tol=1e-10,
                        kpts={"size": (2, 2, 2), "gamma": True})
    assert atoms.get_potential_energy() == pytest.approx(ref.energy * units.Hartree, rel=1e-12)
    bs = atoms.calc.band_structure("GXL", npoints=7)                    # H_x along the path
    assert bs.energies.shape == (1, 7, 8)
    atoms.calc = PyDFTB(SKF_SETS["ptbp_hcocu_lc"], max_l={"C": 1}, hybrid="lc", kpts=(2, 2, 2))
    with pytest.raises(ValueError, match="zero shift"):
        atoms.get_potential_energy()


def test_ase_charge_files(tmp_path):
    """The ASE calculator writes and reads a k-point hybrid's charge file with its geometry (the
    record is of the atoms as DFTB+ folds them): the restart takes one iteration."""
    labels, pos, cell = _diamond()
    atoms = Atoms(labels, positions=pos * BOHR, cell=cell * BOHR, pbc=True)
    opts = dict(max_l={"C": 1}, hybrid="lc", scc_tol=1e-10,
                kpts={"size": (3, 1, 2), "gamma": True})
    atoms.calc = PyDFTB(SKF_SETS["ptbp_hcocu_lc"], **opts)
    energy = atoms.get_potential_energy()
    atoms.calc.write_charges(tmp_path / "charges.bin")
    atoms.calc = PyDFTB(SKF_SETS["ptbp_hcocu_lc"], read_charges=tmp_path / "charges.bin", **opts)
    assert atoms.get_potential_energy() == pytest.approx(energy, abs=1e-10)
    assert atoms.calc.raw.n_scc == 1


def test_dftbplus_input_blocks():
    hsd = dftbplus.make_hsd(".", {"C": 1}, True, kpts={"folding": (3, 1, 2)})
    assert "SupercellFolding {\n    3 0 0\n    0 1 0\n    0 0 2\n    0.0 0.0 0.0\n  }" in hsd
    block = dftbplus.hybrid_block({"type": "lc", "screening": "neighbour"}, kpoints=True)
    assert "Threshold = 1" in block and "e-100" in block
    assert "Threshold" not in dftbplus.hybrid_block({"type": "lc", "screening": "neighbour"})


@pytest.mark.parametrize("hybrid, kpts, match", [
    ("lc", monkhorst_pack((2, 2, 2)), "zero shift"),
    ("lc", (np.array([[0.0, 0.0, 0.25]]), np.ones(1)), "zero shift"),
    ({"type": "lc", "screening": "threshold"}, supercell_folding((2, 1, 1)),
     "threshold algorithm"),
    ({"type": "lc", "coulomb_cutoff": 0.9}, supercell_folding((2, 1, 1)), "positive radius"),
])
def test_refusals(hybrid, kpts, match):
    labels, pos, cell = _diamond()
    with pytest.raises(ValueError, match=match):
        DFTB(_sk(labels), kpts=kpts, hybrid=hybrid).calculate(labels, pos, cell, PBC,
                                                              forces=False)
