"""Range-separated hybrid functionals of periodic systems at the Gamma point (milestone M6h-3a).

Comparisons with DFTB+ are the frozen references co2_crystal_*, diamond8_lc*, cu4_lc_1000K,
o2_box_lc_spin*, co_triclinic_lc and h2o_monoclinic_lc in test_reference.py (with DFTB+'s two
errors B1 and B6 reproduced by the validation switches), the restart files in
test_charges_file.py, and co2_crystal_lc_md and diamond8_lc_opt in test_md_opt.py; here the
truncated Coulomb kernels, the lattice sums, self-consistency, the switches, the charge files
and the refusals (no dftb+).
"""
import dataclasses

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import PropertyNotImplementedError

from pydftb import Hybrid
from pydftb import hybrid as hy
from pydftb.ase_calc import PyDFTB, _hybrid_au
from pydftb.bands import eigenstates
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.dftbplus_io import read_charges, write_charges
from pydftb.hybrid import HybridTerms
from pydftb.validation import dftbplus, refgen
from pydftb.validation.cases import SKF_SETS

LC, CAM, GLOBAL = (0.3, 0.0, 1.0), (0.3, 0.19, 0.46), (0.3, 0.6, 0.0)   # omega, alpha, beta


def _case(name, **options):
    """(result, engine, labels, positions and cell in Bohr) of a reference case; exact SK
    derivatives (no fd_delta) unless ``options`` say otherwise."""
    case = refgen.CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    res, calc = refgen.run_pydftb(case, inp, engine=True, **options)
    cell = None if inp["cell_angstrom"] is None else np.array(inp["cell_angstrom"]) / BOHR
    return res, calc, inp["labels"], np.array(inp["positions_angstrom"]) / BOHR, cell


PBC = np.ones(3, bool)


# ---- the truncated Coulomb kernels ------------------------------------------------------
def test_poly5zero_matches_value_and_derivatives_at_both_ends():
    y, rd, rc, h = (0.3, -0.02, 0.004), 7.0, 8.0, 1e-5

    def f(d, der=False):
        return hy.poly5zero(*y, np.asarray(d, float), rd, rc, der)
    assert f(rd) == pytest.approx(y[0], abs=1e-15) and f(rd, True) == pytest.approx(y[1], abs=1e-15)
    assert (f(rd + h, True) - f(rd - h, True)) / (2 * h) == pytest.approx(y[2], abs=1e-8)
    assert abs(f(rc)) < 1e-15 and abs(f(rc, True)) < 1e-15
    assert abs(f(rc + h, True) - f(rc - h, True)) / (2 * h) < 1e-8
    d = np.linspace(rd + 0.05, rc - 0.05, 9)
    assert np.abs((f(d + h) - f(d - h)) / (2 * h) - f(d, True)).max() < 1e-9


@pytest.mark.parametrize("ua, ub", [(0.4, 0.4), (0.4, 0.4 + 2e-7), (0.4, 0.4001), (0.3, 0.55)])
def test_ddgamma_hf_is_the_derivative(ua, ub):
    """Every branch (equal, DFTB+'s equal-U band, nearly equal and unequal U): the analytical
    second derivative of gamma_HF (DFTB+'s, for the damping) against differences of the
    first."""
    r, h = np.array([1.5, 3.0, 6.0, 9.0]), 1e-5
    fd = (hy.dgamma_hf(ua, ub, r + h) - hy.dgamma_hf(ua, ub, r - h)) / (2 * h)
    assert np.abs(hy.ddgamma_hf(ua, ub, r) - fd).max() < 1e-9
    assert hy.ddgamma_hf(ua, ub, np.zeros(1))[0] == 0.0                     # on site


@pytest.mark.parametrize("params", [LC, CAM, GLOBAL], ids=["lc", "cam", "global"])
def test_truncated_and_damped_kernels(params):
    """Truncated: gamma below the cutoff, zero from it on.  Damped: gamma up to Rc - 1 Bohr,
    then the quintic with gamma's value, slope and curvature at Rc - 1 and zeros at Rc; the
    derivative is the value's (pydftb), DFTB+'s differs where alpha != 0 (B1)."""
    w, a, b = params
    ua, ub, rc, h = 0.45, 0.58, 8.0, 1e-5
    r = np.linspace(0.6, 9.6, 46)
    full = hy.gamma_cam(ua, ub, r, w, a, b)
    trunc = hy.coulomb_gamma(ua, ub, r, w, a, b, rc, "truncated")
    assert np.array_equal(trunc[r < rc], full[r < rc]) and not trunc[r >= rc].any()
    damp = hy.coulomb_gamma(ua, ub, r, w, a, b, rc, "damped")
    inner, win = r <= rc - 1.0, (r > rc - 1.0) & (r < rc)
    assert np.array_equal(damp[inner], full[inner]) and not damp[r >= rc].any()
    assert np.abs(damp[win] - full[win]).max() > 1e-6
    rd = np.array([rc - 1.0 + 1e-3])                     # C2: equal to O(1e-9) just inside
    g_d = hy.coulomb_gamma(ua, ub, rd, w, a, b, rc, "damped")
    assert abs(g_d - hy.gamma_cam(ua, ub, rd, w, a, b))[0] < 1e-8
    for kind in ("truncated", "damped"):
        rr = r[np.abs(r - rc) > 2e-3]
        fd = (hy.coulomb_gamma(ua, ub, rr + h, w, a, b, rc, kind)
              - hy.coulomb_gamma(ua, ub, rr - h, w, a, b, rc, kind)) / (2 * h)
        assert np.abs(hy.coulomb_gamma(ua, ub, rr, w, a, b, rc, kind, derivative=True)
                      - fd).max() < 1e-9
    exact = hy.coulomb_gamma(ua, ub, r, w, a, b, rc, "damped", derivative=True)
    dftbp = hy.coulomb_gamma(ua, ub, r, w, a, b, rc, "damped", derivative=True, dftbp_hf=True)
    assert np.array_equal(exact[~win], dftbp[~win])
    assert (np.abs(exact[win] - dftbp[win]).max() > 1e-6) == (a != 0.0)


def test_plane_spacings_cutoff_and_translations():
    """DFTB+'s defaults: Rc half the smallest plane spacing; the translations a parallelepiped
    that holds every lattice vector within Rg."""
    cell = np.array([[3.0, 0.0, 0.0], [1.2, 2.8, 0.0], [0.5, 0.7, 3.1]]) / BOHR
    h = hy.plane_spacings(cell)
    assert np.allclose(h, 1.0 / np.linalg.norm(np.linalg.inv(cell), axis=0), rtol=1e-14)
    assert hy.default_coulomb_cutoff(cell) == pytest.approx(0.5 * h.min(), rel=1e-15)
    assert hy.default_coulomb_cutoff(np.diag([10.0, 12.0, 11.0])) == pytest.approx(5.0, rel=1e-14)
    rg = 9.0
    g = hy.lattice_translations(cell, rg)
    nmax = np.floor(rg / h).astype(int) + 1
    assert len(g) == np.prod(2 * nmax + 1) and np.any(np.all(g == 0.0, axis=1))
    n = np.array(np.meshgrid(*[np.arange(-8, 9)] * 3, indexing="ij")).reshape(3, -1).T
    near = n @ cell
    near = near[np.linalg.norm(near, axis=1) <= rg]
    have = {tuple(np.round(v, 9)) for v in g}
    assert all(tuple(np.round(v, 9)) in have for v in near)


def test_hybrid_options():
    assert Hybrid.make({"type": "lc", "coulomb": "Damped", "coulomb_cutoff": 8.0}) == Hybrid(
        coulomb="damped", coulomb_cutoff=8.0)
    for bad, match in (({"coulomb": "ewald"}, "coulomb must be one of"),
                       ({"coulomb_cutoff": 0.0}, "coulomb_cutoff must be positive"),
                       ({"gsum_cutoff": -1.0}, "gsum_cutoff must be positive")):
        with pytest.raises(ValueError, match=match):
            Hybrid.make(bad)
    # the ASE calculator's lengths are Angstrom
    au = _hybrid_au({"type": "lc", "coulomb_cutoff": 4.0, "gsum_cutoff": None})
    assert au["coulomb_cutoff"] == pytest.approx(4.0 / BOHR, rel=1e-15)
    assert au["gsum_cutoff"] is None
    au = _hybrid_au(Hybrid(coulomb_cutoff=4.0, gsum_cutoff=8.0))
    assert au.gsum_cutoff == pytest.approx(8.0 / BOHR, rel=1e-15) and au.cutoff_reduction == 0.0


def test_dftbplus_block():
    """DFTB+'s CoulombMatrix only when the settings are not its defaults: the molecular blocks
    stay those of M6h-1."""
    assert "CoulombMatrix" not in dftbplus.hybrid_block("lc")
    text = dftbplus.hybrid_block({"type": "cam", "coulomb": "damped", "coulomb_cutoff": 8.0})
    assert "CoulombMatrix = TruncatedAndDamped {" in text and "CoulombCutoff [Bohr] = 8" in text
    assert "GSummationCutoff" not in text and text.endswith("  }\n")


# ---- the terms of a geometry ------------------------------------------------------------
def test_molecule_in_a_large_box_has_the_molecules_exchange():
    """No image within the cutoff or the SK range: the Gamma-point gt, S(Gamma) and P0 are the
    molecule's gamma, S and P0, and so is H_x of any density."""
    _, calc, labels, pos, _ = _case("h2o_lc")
    cell = np.eye(3) * 30.0
    mol = calc._setup(labels, pos, None, None, False, {}).hybrid
    box = calc._setup(labels, pos, cell, PBC, False, {}).hybrid
    assert box._periodic and box.rc == pytest.approx(15.0, rel=1e-14)
    assert box.rg == pytest.approx(30.0, rel=1e-14)
    assert np.abs(box.G - mol.G).max() < 1e-15 and np.abs(box.S - mol.S).max() < 1e-15
    assert np.array_equal(box.q0_orb, mol.q0_orb)
    rng = np.random.default_rng(5)
    d = rng.normal(size=(mol.n, mol.n))
    d = d + d.T
    assert np.abs(box.fock([d])[0] - mol.fock([d])[0]).max() < 1e-14


def test_lattice_sums_in_blocks(monkeypatch):
    """Large cells sum over the translations in blocks (IMAGE_CHUNK atom pairs x translations):
    the same gt, energy and forces as one block."""
    res, calc, labels, pos, cell = _case("co2_crystal_lc_damped_rc8", scc_tol=1e-12)
    one = calc._setup(labels, pos, cell, PBC, False, {}).hybrid
    assert len(list(one._image_blocks())) == 4                     # C-C, C-O, O-C, O-O
    monkeypatch.setattr(hy, "IMAGE_CHUNK", 1000)                  # 3 to 9 blocks per pair
    blocks = calc._setup(labels, pos, cell, PBC, False, {}).hybrid
    assert len(list(blocks._image_blocks())) == 22
    assert np.abs(blocks.g_atom - one.g_atom).max() < 1e-14
    r = calc.calculate(labels, pos, cell, PBC, forces=True, stress=False)
    assert r.energy == pytest.approx(res.energy, abs=1e-12) and r.n_scc == res.n_scc
    assert np.abs(r.forces - res.forces).max() < 1e-12


def test_reduced_overlap():
    """NeighbourBased keeps the S(Gamma) blocks of pairs within the largest SK cutoff minus the
    reduction (no reduction: S(Gamma) itself)."""
    _, calc, labels, pos, cell = _case("co2_crystal_lc_neighbour_red10_rc8")
    t = calc._setup(labels, pos, cell, PBC, False, {}).hybrid
    assert t._cam_cutoff == pytest.approx(calc.sk.cutoff - 10.0)
    assert np.abs(t.S_cam - t.S).max() > 1e-3 and np.array_equal(np.diag(t.S_cam), np.ones(t.n))
    _, calc0, *_ = _case("co2_crystal_lc_neighbour")
    t0 = calc0._setup(labels, pos, cell, PBC, False, {}).hybrid
    assert np.array_equal(t0.S_cam, t0.S)


def test_reference_density_divides_by_the_gamma_overlap():
    """P0 = diag(q0 / S_mumu(Gamma)) (DFTB+): in the small triclinic cell S(Gamma)'s diagonal
    holds the orbitals' overlaps with their own images, so that tr(P0 S(Gamma)) = sum q0."""
    _, calc, labels, pos, cell = _case("co_triclinic_lc")
    st = calc._setup(labels, pos, cell, PBC, False, {})
    t = st.hybrid
    ls = np.asarray(st.basis.shell_l)
    q0 = (calc._reference_shell_populations(st.basis) / (2 * ls + 1))[st.basis.orb_shell]
    assert np.abs(np.diag(t.S) - 1.0).max() > 1e-3
    assert np.allclose(t.q0_orb * np.diag(t.S), q0, rtol=0, atol=1e-15)
    assert (t.q0_orb * np.diag(t.S)).sum() == pytest.approx(10.0, abs=1e-12)


def test_neighbour_without_reduction_is_the_matrix_algorithm():
    res, *_ = _case("co2_crystal_lc", scc_tol=1e-12)
    nb0, *_ = _case("co2_crystal_lc_neighbour", scc_tol=1e-12)
    assert nb0.energy == pytest.approx(res.energy, abs=1e-12) and nb0.n_scc == res.n_scc
    assert np.abs(nb0.forces - res.forces).max() < 1e-12


# ---- self-consistency ---------------------------------------------------------------------
def _fd_forces(calc, labels, pos, cell, comps, h=5e-5):
    """-dF/dR of the free energy by 4-point central differences."""
    out = []
    for a, c in comps:
        e = {}
        for s in (-2, -1, 1, 2):
            p = pos.copy()
            p[a, c] += s * h
            e[s] = calc.calculate(labels, p, cell, PBC, forces=False).free_energy
        out.append(-(8.0 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12.0 * h))
    return np.array(out)


@pytest.mark.parametrize("name", ["co2_crystal_lc", "co2_crystal_lc_damped_rc8",
                                  "co2_crystal_cam_damped_rc8", "co2_crystal_global_damped_rc8",
                                  "co2_crystal_lc_neighbour_red10_rc8", "cu4_lc_1000K",
                                  "o2_box_lc_spin", "co_triclinic_lc", "co2_crystal_lc_plusu"])
def test_forces_are_the_energy_derivative(name):
    """pydftb's forces (exact SK derivatives) against 4-point differences of the free energy
    (h = 5e-5 Bohr, SCC tolerance 1e-12): the damped kernels with an HF part (B1), the reduced
    overlap of NeighbourBased and a low-symmetry cell's populations (B6) included."""
    _, calc, labels, pos, cell = _case(name, scc_tol=1e-12)
    calc._dftbp_gamma_q_in = False                     # pydftb's populations (co_triclinic_lc)
    res = calc.calculate(labels, pos, cell, PBC, forces=True, stress=False)
    n = len(labels)
    comps = [(0, 0), (n // 2, 1), (n - 1, 2)]
    fd = _fd_forces(calc, labels, pos, cell, comps)
    assert np.abs(np.array([res.forces[a, c] for a, c in comps]) - fd).max() < 5e-8


def test_dftbplus_damped_derivative_switch():
    """HybridTerms._dftbp_hf_damped_derivative (validation only, B1) changes the forces of a
    damped kernel with an HF part by about 7e-7 Ha/Bohr, nothing else; LC (no HF part) not at
    all."""
    res, calc, labels, pos, cell = _case("co2_crystal_cam_damped_rc8")
    lc, calc_lc, *_ = _case("co2_crystal_lc_damped_rc8")
    HybridTerms._dftbp_hf_damped_derivative = True
    try:
        bug = calc.calculate(labels, pos, cell, PBC, forces=True, stress=False)
        bug_lc = calc_lc.calculate(labels, pos, cell, PBC, forces=True, stress=False)
    finally:
        HybridTerms._dftbp_hf_damped_derivative = False
    assert bug.energy == res.energy and np.array_equal(bug.charges, res.charges)
    assert 1e-7 < np.abs(bug.forces - res.forces).max() < 3e-6
    assert np.array_equal(bug_lc.forces, lc.forces)


def test_dftbp_dense_mulliken():
    """DFTB+'s denseMullikenReal of a lower-triangular DeltaP (validation only, B6): the
    Mulliken populations but for the terms of an atom's own block above its diagonal."""
    rng = np.random.default_rng(7)
    orb_atom = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2])
    n = len(orb_atom)
    d, s = rng.normal(size=(2, n, n))
    d, s = d + d.T, s + s.T
    same = orb_atom[:, None] == orb_atom[None, :]
    exact = (d * s).sum(axis=1)
    missed = (np.triu(d * s, 1) * same).sum(axis=0)
    assert np.allclose(hy.dftbp_dense_mulliken(d, s, orb_atom), exact - missed, atol=1e-13)
    s_block = np.where(same & ~np.eye(n, dtype=bool), 0.0, s)        # no intra-atomic terms
    assert np.allclose(hy.dftbp_dense_mulliken(d, s_block, orb_atom), (d * s_block).sum(1),
                       atol=1e-13)


def test_dftbplus_input_populations_switch():
    """DFTB._dftbp_gamma_q_in (validation only, B6): no effect in a cubic cell (symmetry cancels
    the intra-atomic terms of S(Gamma)); in the triclinic CO cell the SCC ends elsewhere, and
    DFTB+'s charges are pydftb's with the switch (test_reference), not its own (1.3e-5 e)."""
    res, calc, labels, pos, cell = _case("diamond8_lc", scc_tol=1e-12)
    calc._dftbp_gamma_q_in = True
    alt = calc.calculate(labels, pos, cell, PBC, forces=True, stress=False)
    assert alt.energy == pytest.approx(res.energy, abs=1e-12) and alt.n_scc == res.n_scc
    case = refgen.CASES_BY_NAME["co_triclinic_lc"]
    doc = refgen.load("co_triclinic_lc")
    exact = refgen.run_pydftb(dataclasses.replace(case, gamma_q_in_compat=False), doc["input"])
    q0 = refgen._neutral_populations(case, doc["input"])
    dq = np.abs((q0 - exact.charges) - np.array(doc["reference"]["charges_net"])).max()
    assert 3e-6 < dq < 1e-4
    assert abs(exact.energy - doc["reference"]["energy"]) < 1e-9          # second order


# ---- results, restarts and charge files --------------------------------------------------
def test_energy_terms_and_no_stress():
    res, calc, labels, pos, cell = _case("co2_crystal_cam")
    assert res.e_fock < 0.0 and res.stress is None
    total = sum(v for k, v in res.terms.items() if k != "-TS")
    assert total == pytest.approx(res.energy, abs=1e-12)
    with pytest.raises(ValueError, match="no stress with a hybrid functional"):
        calc.calculate(labels, pos, cell, PBC, forces=True, stress=True)


@pytest.mark.parametrize("name", ["co2_crystal_lc", "o2_box_lc_spin"])
def test_eigenstates_reproduce_the_scc_levels(name):
    res, calc, labels, pos, cell = _case(name)
    st = eigenstates(calc, labels, pos, cell, PBC, res.q_in)
    ev = np.asarray(res.eigenvalues).reshape(st.eigenvalues.shape)
    assert np.abs(st.eigenvalues - ev).max() < 1e-12
    with pytest.raises(ValueError, match="Gamma point only"):
        eigenstates(calc, labels, pos, cell, PBC, res.q_in, kpts=np.array([[0.0, 0.0, 0.5]]))


@pytest.mark.parametrize("name, text", [("co2_crystal_lc_plusu", False),
                                        ("o2_box_lc_spin", True)])
def test_charge_file_with_the_folding_record(tmp_path, name, text):
    """DFTB+'s layout of a Gamma-point run: the identity folding (coeffsAndShifts) before
    DeltaP_in.  A molecule of the same atoms starts from the file too (DFTB+ reads it one value
    off), and the Gamma point from a molecule's file (DFTB+ stops)."""
    res, calc, labels, pos, cell = _case(name, scc_tol=1e-10)
    data = calc.charge_data(labels, res, periodic=True)
    assert data.kpoint_info and np.array_equal(data.folding, np.hstack([np.eye(3),
                                                                         np.zeros((3, 1))]))
    path = tmp_path / ("charges.dat" if text else "charges.bin")
    write_charges(path, data)
    back = read_charges(path, data.norb_atom)
    assert back.kpoint_info and np.array_equal(back.folding, data.folding)
    assert np.array_equal(back.rho, data.rho)
    q = calc.populations_from_file(labels, back)
    assert np.abs(q - res.q_in).max() < 1e-14
    again = calc.calculate(labels, pos, cell, PBC, forces=False, q_init=back)
    assert again.n_scc <= 2 and again.energy == pytest.approx(res.energy, abs=1e-9)
    mol = calc.calculate(labels, pos, forces=False, q_init=back)
    assert mol.converged
    mol_data = calc.charge_data(labels, mol)                          # a molecule's layout
    assert mol_data.folding is None and not mol_data.kpoint_info
    gamma = calc.calculate(labels, pos, cell, PBC, forces=False, q_init=mol_data)
    assert gamma.energy == pytest.approx(res.energy, abs=1e-9)


def test_charge_file_errors(tmp_path):
    res, calc, labels, *_ = _case("o2_box_lc_spin")
    data = calc.charge_data(labels, res, periodic=True)
    path = tmp_path / "charges.bin"
    write_charges(path, dataclasses.replace(data, folding=np.hstack([2 * np.eye(3),
                                                                     np.zeros((3, 1))])))
    with pytest.raises(ValueError, match="M6h-3b"):
        read_charges(path, data.norb_atom)
    with pytest.raises(ValueError, match="goes with a density matrix"):
        write_charges(path, dataclasses.replace(data, rho=None, density_matrix=False))
    with pytest.raises(ValueError, match="writes only"):
        write_charges(path, dataclasses.replace(data, folding=None))


def test_ase_calculator(tmp_path):
    """Lengths in Angstrom, no stress, and DFTB+'s Gamma-point charge file."""
    res, calc, labels, pos, cell = _case("co2_crystal_lc_damped_rc8", scc_tol=1e-10)
    atoms = Atoms(labels, positions=pos * BOHR, cell=cell * BOHR, pbc=True)
    atoms.calc = PyDFTB(SKF_SETS["ptbp_hcocu_lc"], max_l={"C": 1, "O": 1}, scc_tol=1e-10,
                        hybrid={"type": "lc", "coulomb": "damped", "coulomb_cutoff": 8.0 * BOHR,
                                "gsum_cutoff": 16.0 * BOHR})
    assert atoms.get_potential_energy() == pytest.approx(res.energy * units.Hartree, rel=1e-12)
    with pytest.raises(PropertyNotImplementedError, match="no stress with a hybrid"):
        atoms.get_stress()
    atoms.calc.write_charges(tmp_path / "charges.bin")
    data = read_charges(tmp_path / "charges.bin", calc.charge_data(labels, res).norb_atom)
    assert data.kpoint_info and data.folding is not None and data.rho is not None


# ---- refusals -----------------------------------------------------------------------------
@pytest.mark.parametrize("spec, opts, match", [
    ("lc", {"kpts": (np.array([[0.0, 0.0, 0.25]]), np.ones(1))}, "Gamma point only"),
    ({"type": "lc", "screening": "threshold"}, {}, "threshold algorithm"),
    ("lc", {"spin": "noncolinear", "spin_constants": {"O": -0.028}}, "with molecular"),
    ("lc", {"casida": {"n_excitations": 3}}, "excited states for periodic"),
    ({"type": "lc", "coulomb": "full"}, {}, "Truncated\\+Damping"),
    ({"type": "lc", "coulomb": "minimum_image"}, {}, "Truncated\\+Damping"),
    ({"type": "lc", "coulomb": "damped", "coulomb_cutoff": 1.0}, {}, "positive radius"),
    ({"type": "lc", "coulomb_cutoff": 0.9}, {}, "positive radius"),
])
def test_refusals(spec, opts, match):
    _, calc, labels, pos, cell = _case("o2_box_lc_spin")
    sk = calc.sk
    with pytest.raises(ValueError, match=match):
        DFTB(sk, scc_tol=1e-8, hybrid=spec, **opts).calculate(labels, pos, cell, PBC,
                                                              forces=False)


@pytest.mark.parametrize("spec", [{"coulomb": "damped"}, {"coulomb_cutoff": 8.0},
                                  {"gsum_cutoff": 16.0}])
def test_molecule_refuses_the_coulomb_matrix(spec):
    """DFTB+ reads CoulombMatrix for periodic systems only and stops on it for a molecule
    (P71); the default Hybrid (no settings) runs."""
    _, calc, labels, pos, _ = _case("o2_box_lc_spin")
    with pytest.raises(ValueError, match="CoulombMatrix of periodic systems"):
        DFTB(calc.sk, scc_tol=1e-8, hybrid={"type": "lc", **spec}).calculate(labels, pos,
                                                                             forces=False)
    DFTB(calc.sk, scc_tol=1e-8, hybrid=Hybrid()).calculate(labels, pos, forces=False)
