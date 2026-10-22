"""Two-component DFTB (milestone M6a): non-collinear spin and spin-orbit coupling.

Comparisons with DFTB+ are the frozen references o2_noncolinear_*, oh_noncolinear_*,
cu4co_soc*, gdo_noncolinear_*, cuo_chain_soc_k422, cuo_chain_noncolinear_k422 and
cu_fcc_soc_bands in test_reference.py; here self-consistency, limits and errors (no dftb+).
"""
import numpy as np
import pytest

from pydftb import spinorbit as so
from pydftb.api import SCC
from pydftb.ase_calc import PyDFTB
from pydftb.bands import eigenstates
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.kpoints import monkhorst_pack, reduce_time_reversal
from pydftb.params import ParamSet
from pydftb.system import System
from pydftb.validation import refgen
from pydftb.validation.cases import TILT, W_SPIN

from .conftest import SKF_HCOCU

XI_O = {"O": [0.0, 0.01]}


def _system(name):
    case = refgen.CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    params = ParamSet(refgen.SKF_SETS[case.skf_set], max_l=inp["max_l"],
                      shell_resolved=case.shell_resolved)
    kpts = None if inp["kpoints"] is None else (np.array(inp["kpoints"]), np.array(inp["kweights"]))
    cell = None if inp["cell_angstrom"] is None else np.array(inp["cell_angstrom"]) / BOHR
    opts = dict(case.dftb3 or {}, **refgen.pauli_options(case, inp["labels"]))
    return (case, params.skset(inp["labels"]), inp["labels"],
            np.array(inp["positions_angstrom"]) / BOHR, cell, kpts, opts)


def _rotation(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


# ---- the building blocks ----------------------------------------------------------------
@pytest.mark.parametrize("l", [1, 2, 3])
def test_angular_momentum_operators(l):
    lx, ly, lz = so.l_matrices(l)
    assert np.allclose(lx @ ly - ly @ lx, 1j * lz, atol=1e-14)
    assert np.allclose(lx @ lx + ly @ ly + lz @ lz, l * (l + 1) * np.eye(2 * l + 1), atol=1e-13)
    for m in (lx, ly, lz):                        # purely imaginary in the real orbitals
        assert np.abs(m.real).max() < 1e-15
        assert np.allclose(m.imag, -m.imag.T, atol=1e-15)


def test_p_operators_in_dftbplus_order():
    """p orbitals are (y, z, x): L_z p_x = i p_y, i.e. <p_y|L_z|p_x> = i."""
    _, _, lz = so.l_matrices(1)
    assert lz[0, 2] == pytest.approx(1j)
    assert lz[2, 0] == pytest.approx(-1j)


def test_spinor_components_roundtrip():
    rng = np.random.default_rng(0)
    m = [rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3)) for _ in range(4)]
    m = [0.5 * (x + np.conj(np.swapaxes(x, 1, 2))) for x in m]          # Hermitian components
    h = so.spinor(*m)
    assert np.allclose(h, np.conj(np.swapaxes(h, 1, 2)))
    assert np.allclose(so.components(h), 2 * np.array(m))              # rho = 1/2 sum m_s sigma_s


def test_k_point_set_closure():
    k, w = monkhorst_pack((4, 2, 2))
    assert so.closed_under_inversion(k, w)
    assert not so.closed_under_inversion(*reduce_time_reversal(k, w))
    assert so.closed_under_inversion(np.zeros((1, 3)), np.ones(1))


# ---- limits and invariances ---------------------------------------------------------------
def test_noncolinear_along_z_equals_collinear():
    """Moments along z: the spinor problem splits into the two collinear channels."""
    _, sk, labels, pos, _, _, _ = _system("o2_noncolinear_tilted")
    w = {"O": -0.028}
    col = DFTB(sk, scc_tol=1e-12, spin=True, unpaired_electrons=2.0, relax_total_spin=True,
               spin_constants=w).calculate(labels, pos)
    nc = DFTB(sk, scc_tol=1e-12, spin="noncolinear", spin_constants=w,
              initial_spins=[[0, 0, 1], [0, 0, 1]]).calculate(labels, pos)
    assert nc.energy == pytest.approx(col.energy, abs=1e-11)
    assert np.allclose(nc.forces, col.forces, atol=1e-10)
    assert np.allclose(nc.charges, col.charges, atol=1e-10)
    assert np.allclose(nc.magmoms[:, 2], col.magmoms, atol=1e-10)
    assert np.allclose(nc.magmoms[:, :2], 0.0, atol=1e-12)
    assert nc.e_spin == pytest.approx(col.e_spin, abs=1e-11)


def test_rotated_moments_give_the_same_molecule():
    _, sk, labels, pos, _, _, opts = _system("oh_noncolinear_tilted")
    calc = lambda spins: DFTB(sk, smearing=("fermi", 300.0), scc_tol=1e-12,  # noqa: E731
                              **{**opts, "initial_spins": spins})
    z = calc([[0, 0, 1], [0, 0, 0]]).calculate(labels, pos)
    t = calc([TILT, [0, 0, 0]]).calculate(labels, pos)
    assert t.energy == pytest.approx(z.energy, abs=1e-10)
    assert np.allclose(t.forces, z.forces, atol=1e-9)
    assert np.allclose(np.linalg.norm(t.magmoms, axis=1), np.abs(z.magmoms[:, 2]), atol=1e-9)
    assert np.allclose(np.cross(t.magmoms[0], TILT), 0.0, atol=1e-9)   # along TILT


def test_rotated_density_is_converged_for_a_periodic_cell():
    """Without spin-orbit coupling a rotated converged density is converged: energy, forces
    and stress unchanged (a k-point grid, here where DFTB+'s own SCC diverges)."""
    _, sk, labels, pos, cell, kpts, opts = _system("cuo_chain_noncolinear_k422")
    calc = DFTB(sk, kpts=kpts, smearing=("fermi", 500.0), scc_tol=1e-11, max_scc=500, **opts)
    z = calc.calculate(labels, pos, cell, np.ones(3, bool))
    nsh = len(z.q_spin[0])
    rot = _rotation([1.0, 2.0, 0.5], 1.1)
    q = z.q_in.copy()
    q[nsh:] = (rot @ q[nsh:].reshape(3, nsh)).ravel()
    r = calc.calculate(labels, pos, cell, np.ones(3, bool), q_init=q)
    assert r.n_scc <= 2
    assert r.energy == pytest.approx(z.energy, abs=1e-10)
    assert np.allclose(r.forces, z.forces, atol=1e-9)
    assert np.allclose(r.stress, z.stress, atol=1e-11)
    assert np.allclose(r.magmoms, z.magmoms @ rot.T, atol=1e-8)


def test_spin_orbit_without_spin_has_no_magnetisation():
    _, sk, labels, pos, _, _, opts = _system("cu4co_soc")
    r = DFTB(sk, smearing=("fermi", 300.0), scc_tol=1e-10, **opts).calculate(labels, pos)
    assert np.abs(r.magmoms).max() < 1e-10
    assert r.e_ls < 0
    assert r.terms["LS"] == r.e_ls
    assert sum(len(e) for e in r.eigenvalues) == 2 * 44            # 2N spinor levels
    assert max(o.max() for o in r.occupations) <= 1.0 + 1e-12


def test_dual_energy_from_block_populations():
    """E_LS = Tr(rho H_SO) equals DFTB+'s expression with the skew block populations."""
    _, sk, labels, pos, _, _, opts = _system("oh_noncolinear_soc")
    calc = DFTB(sk, smearing=("fermi", 300.0), scc_tol=1e-12, **opts)
    res, st, sc = calc._run(labels, pos, None, None, False, None, False)
    blocks = so.block_populations(so.components(sc.rho), st.smat, st.kw, st.basis)
    assert so.dual_soc_energy(blocks, st.pauli.lam, st.basis) == pytest.approx(res.e_ls, abs=1e-13)


def test_restart_from_the_mixed_vector():
    _, sk, labels, pos, _, _, opts = _system("oh_noncolinear_soc")
    calc = DFTB(sk, smearing=("fermi", 300.0), scc_tol=1e-10, **opts)
    r = calc.calculate(labels, pos)
    again = calc.calculate(labels, pos, q_init=r.q_spin.ravel())
    assert again.n_scc == 1                      # given charges: no two-iteration minimum
    assert again.energy == pytest.approx(r.energy, abs=1e-9)


# ---- derivatives ----------------------------------------------------------------------
def _fd_force(calc, labels, pos, cell, atom, comp, h=1e-4):
    pbc = None if cell is None else np.ones(3, bool)
    e = []
    for s in (1, -1):
        p = pos.copy()
        p[atom, comp] += s * h
        e.append(calc.calculate(labels, p, cell, pbc, forces=False).free_energy)
    return -(e[0] - e[1]) / (2 * h)


@pytest.mark.parametrize("dual", [True, False])
def test_molecule_forces_match_energy_differences(dual):
    _, sk, labels, pos, _, _, opts = _system("oh_noncolinear_soc")
    calc = DFTB(sk, smearing=("fermi", 300.0), scc_tol=1e-12, max_scc=500,
                **{**opts, "spin_orbit": {"O": [0.0, 0.05]}, "spin_orbit_dual": dual})
    f = calc.calculate(labels, pos).forces
    for comp in range(3):
        assert _fd_force(calc, labels, pos, None, 1, comp) == pytest.approx(f[1, comp], abs=2e-8)


@pytest.mark.parametrize("dual", [True, False])
def test_periodic_forces_and_stress_match_energy_differences(dual):
    """Non-Gamma k-points, the dual model's dS term in forces and stress."""
    _, sk, labels, pos, cell, _, _ = _system("cuo_chain_soc_k422")
    calc = DFTB(sk, kpts=monkhorst_pack((2, 1, 1)), smearing=("fermi", 500.0), scc_tol=1e-12,
                max_scc=500, spin_orbit={"Cu": [0.0, 0.0, 0.03], "O": [0.0, 0.02]},
                spin_orbit_dual=dual)
    r = calc.calculate(labels, pos, cell, np.ones(3, bool))
    assert _fd_force(calc, labels, pos, cell, 1, 0) == pytest.approx(r.forces[1, 0], abs=2e-8)
    assert _fd_force(calc, labels, pos, cell, 0, 2) == pytest.approx(r.forces[0, 2], abs=2e-8)
    vol = abs(np.linalg.det(cell))
    for i, j in ((0, 0), (0, 1)):
        e = []
        for s in (1, -1):
            eps = np.zeros((3, 3))
            eps[i, j] += s * 0.5e-5
            eps[j, i] += s * 0.5e-5
            m = np.eye(3) + eps
            e.append(calc.calculate(labels, pos @ m.T, cell @ m.T, np.ones(3, bool),
                                    forces=False).free_energy)
        assert -(e[0] - e[1]) / 2e-5 / vol == pytest.approx(r.stress[i, j], abs=1e-10)


# ---- band states --------------------------------------------------------------------
def test_eigenstates_reproduce_the_scc_levels():
    _, sk, labels, pos, cell, kpts, opts = _system("cuo_chain_soc_k422")
    calc = DFTB(sk, kpts=kpts, smearing=("fermi", 500.0), scc_tol=1e-10, **opts)
    r = calc.calculate(labels, pos, cell, np.ones(3, bool), forces=False)
    st = eigenstates(calc, labels, pos, cell, np.ones(3, bool), r.q_in, projections=True)
    assert st.two_component and st.nspin == 1
    assert np.allclose(st.eigenvalues[0], np.array(r.eigenvalues), atol=1e-12)
    assert np.allclose(st.projections.sum(axis=2), 1.0, atol=1e-10)
    assert st.occupations.max() <= 1.0 + 1e-12


# ---- interfaces ----------------------------------------------------------------------
def test_api_uses_the_full_grid_for_two_component_runs():
    from ase.build import bulk
    params = ParamSet(SKF_HCOCU, max_l={"Cu": 2})
    system = System.from_ase(bulk("Cu", "fcc", a=3.61))
    soc = SCC(system, params, kpts=(4, 4, 4), spin_orbit={"Cu": [0.0, 0.0, 0.01]})
    plain = SCC(system, params, kpts=(4, 4, 4))
    assert len(soc.kpts[0]) == 64 and len(plain.kpts[0]) < 64
    with pytest.raises(ValueError, match="-k for every k"):
        SCC(system, params, kpts=(4, 4, 4), time_reversal=True,
            spin_orbit={"Cu": [0.0, 0.0, 0.01]}).kernel()


def test_ase_calculator_reads_vector_moments():
    from ase import Atoms
    atoms = Atoms("OH", positions=[[0, 0, 0], [0.1, 0.05, 0.97]])
    atoms.set_initial_magnetic_moments([TILT, [0, 0, 0]])
    atoms.calc = PyDFTB(SKF_HCOCU, max_l={"O": 1, "H": 0}, smearing=("fermi", 300.0),
                        scc_tol=1e-10, spin="noncolinear",
                        spin_constants={k: W_SPIN[k] for k in ("O", "H")})
    m = atoms.get_magnetic_moments()
    assert m.shape == (2, 3)
    assert atoms.get_magnetic_moment() == pytest.approx(np.linalg.norm(m.sum(axis=0)))
    assert np.allclose(np.cross(m.sum(axis=0), TILT), 0.0, atol=1e-8)
    assert atoms.calc.get_number_of_spins() == 1


# ---- errors ------------------------------------------------------------------------
def test_option_errors():
    _, sk, labels, pos, _, _, _ = _system("oh_noncolinear_soc")
    w = {k: W_SPIN[k] for k in ("O", "H")}
    with pytest.raises(ValueError, match="noncolinear' or no spin"):
        DFTB(sk, spin=True, spin_constants=w, spin_orbit=XI_O)
    with pytest.raises(ValueError, match="s shell"):
        DFTB(sk, spin_orbit={"O": [0.1, 0.01]})
    with pytest.raises(ValueError, match="one value per shell"):
        DFTB(sk, spin_orbit={"O": [0.01]})
    with pytest.raises(ValueError, match="collinear spin"):
        DFTB(sk, spin="noncolinear", spin_constants=w, unpaired_electrons=1.0)
    with pytest.raises(ValueError, match="needs spin_constants"):
        DFTB(sk, spin="noncolinear")
    with pytest.raises(ValueError, match="initial_spins needs"):
        DFTB(sk, initial_spins=[[0, 0, 1], [0, 0, 0]])
    with pytest.raises(ValueError, match="SP2"):
        DFTB(sk, solver="sp2", spin_orbit=XI_O)
    with pytest.raises(ValueError, match="spin must be"):
        DFTB(sk, spin="sideways")
    with pytest.raises(ValueError, match="one .x, y, z. vector per atom"):
        DFTB(sk, spin="noncolinear", spin_constants=w,
             initial_spins=[[0, 0, 1]]).calculate(labels, pos)
    _, sk_cu, labels_cu, pos_cu, _, _, _ = _system("cu4co_soc")
    with pytest.raises(ValueError, match="no entry for 'Cu'"):
        DFTB(sk_cu, spin_orbit={"O": [0.0, 0.01]}).calculate(labels_cu, pos_cu)


def test_torch_and_autodiff_refuse():
    pytest.importorskip("torch")
    from pydftb.autodiff import evaluate
    _, sk, labels, pos, _, _, opts = _system("oh_noncolinear_soc")
    with pytest.raises(ValueError, match="NumPy backend only"):
        DFTB(sk, backend="torch", **opts)
    with pytest.raises(NotImplementedError, match="two-component"):
        evaluate(DFTB(sk, **opts), labels, pos)
