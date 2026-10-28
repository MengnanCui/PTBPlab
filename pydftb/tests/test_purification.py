"""Density matrices without eigenvalues (milestone M7a): NTPoly's purifications (PM, TRS2, TRS4,
HPCP) and inverse square root against eigendecompositions; pydftb's NTPoly solver in the SCC
against the eigensolver for each Hamiltonian it takes (a tight TRS2), torch and mixed precision,
DFTB+'s LAPACK drivers, and the refusals.  The comparisons with DFTB+ 24.1 (ELSI) are the frozen
*_ntpoly cases of test_reference.py.
"""
import dataclasses
import importlib.util

import numpy as np
import pytest
import scipy.linalg as sla
from ase import units
from ase.md.verlet import VelocityVerlet

from pydftb import casida as cs
from pydftb import dftbplus_io as io
from pydftb import purification as pu
from pydftb.ase_calc import PyDFTB
from pydftb.backend import ops
from pydftb.bands import eigenstates
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.params import ParamSet
from pydftb.purification import NTPoly
from pydftb.validation import dynamics, refgen
from pydftb.xlbomd import XLBOMD

from .conftest import SKF_HCOCU

TIGHT = dict(method="trs2", tolerance=1e-10, truncation=1e-15)
TORCH = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")


def _gapped(n, n_occ, seed, complex_=False, gap=0.4):
    """A random Hermitian matrix: n_occ levels in [-1, -gap/2], the others in [gap/2, 1]."""
    rng = np.random.default_rng(seed)
    e = np.concatenate([rng.uniform(-1.0, -gap / 2, n_occ),
                        rng.uniform(gap / 2, 1.0, n - n_occ)])
    a = rng.normal(size=(n, n)) + (1j * rng.normal(size=(n, n)) if complex_ else 0.0)
    q = np.linalg.qr(a)[0]
    return (q * e) @ q.conj().T


def _run(name, solver="eigh", backend=None, dtype=None, **over):
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], **over)
    return refgen.run_pydftb(case, refgen._inputs(case), solver=solver, backend=backend,
                             dtype=dtype)


def _host(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _deviations(a, b):
    return (abs(a.energy - float(b.energy)),
            np.abs(np.asarray(a.charges) - _host(b.charges)).max(),
            np.abs(np.asarray(a.forces) - _host(b.forces)).max())


@pytest.fixture(scope="module")
def sk():
    return ParamSet(SKF_HCOCU).skset(["C", "O", "H"])


# ---- the purifications ----------------------------------------------------------------------
@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("method", ["pm", "trs2", "trs4", "hpcp"])
def test_purification_is_the_projector(method, complex_):
    """X is the projector on the n_occ lowest levels, with its band energy 2 tr(X H) and a
    chemical potential in the gap.  Tight (tolerance 1e-12, no flushing): to rounding, but TRS4,
    which stops when |tr G| < 1e-14 (NTPoly), to 1e-9; DFTB+'s defaults (1e-5, 1e-10): 1e-6."""
    h = _gapped(40, 12, 1 + complex_, complex_)
    w, v = np.linalg.eigh(h)
    proj = v[:, :12] @ v[:, :12].conj().T
    for (tol, thr), bound in (((1e-12, 0.0), 1e-9 if method == "trs4" else 1e-12),
                              ((1e-5, 1e-10), 1e-6)):
        x, energy, mu, steps = pu.purify(h, 24.0, method, tol, thr)
        assert np.abs(x - proj).max() <= bound
        assert energy == pytest.approx(2.0 * np.trace(proj @ h).real, abs=10.0 * bound)
        assert w[11] < mu < w[12]
        assert steps < 40


def test_inverse_square_root():
    """NTPoly's S^-1/2 (coupled Newton-Schulz, Taylor order 5): to rounding in a few steps."""
    for complex_ in (False, True):
        s = _gapped(30, 0, 3, complex_, gap=0.1)               # levels in [0.05, 1]
        w, v = np.linalg.eigh(s)
        z, steps = pu.inverse_sqrt(s, 1e-10, 1e-15)
        assert np.abs(z - (v / np.sqrt(w)) @ v.conj().T).max() < 1e-13
        assert steps <= 8


def test_flushing():
    """NTPoly's truncation: the elements smaller than the threshold in magnitude become zero."""
    a = np.array([[1e-11, -2e-10, 1.0], [3e-11j, 0.5, 1e-10]])
    b = pu._drop(a, 1e-10)
    assert b[0, 0] == 0.0 and b[1, 0] == 0.0
    assert b[0, 1] == -2e-10 and b[1, 2] == 1e-10 and b[1, 1] == 0.5
    assert pu._drop(a, 0.0) is a


def test_unconverged_purification_warns():
    with pytest.warns(RuntimeWarning, match="TRS2 purification did not converge in 3 steps"):
        pu.purify(_gapped(20, 5, 4), 10.0, "trs2", 1e-12, 0.0, max_iterations=3)


def test_method_names():
    assert [pu.method_index(m) for m in ("PM", "trs2", "TRS4", "hpcp", 0, 3)] == [0, 1, 2, 3, 0, 3]
    for bad in ("trs3", 4):
        with pytest.raises(ValueError, match="unknown purification method"):
            pu.method_index(bad)
    with pytest.raises(ValueError, match="precision must be"):
        NTPoly(precision="half")
    assert repr(NTPoly("pm", precision="mixed")) == \
        "NTPoly(method='pm', tolerance=1e-05, truncation=1e-10, precision='mixed')"


def test_solver_interface():
    """NTPoly.solve, the closed-shell interface of the eigensolvers, for H C = S C e at two
    k-points: P = 2 C_occ C_occ^H, W = P H P / 2 = 2 C_occ e_occ C_occ^H, and the weighted mean
    of the chemical potentials, each in its gap."""
    hs, ss = [], []
    for k in range(2):
        s = _gapped(24, 0, 10 + k, True, gap=1.0)               # levels in [0.5, 1]
        w, v = np.linalg.eigh(s)
        root = (v * np.sqrt(w)) @ v.conj().T
        hs.append(root @ _gapped(24, 7, 12 + k, True) @ root)   # the levels of the second
        ss.append(s)
    hs, ss = np.stack(hs), np.stack(ss)
    sol = NTPoly(**TIGHT)
    sol.prepare(ss)
    rho, wdm, evals, mu, ts = sol.solve(hs, np.array([0.25, 0.75]), 14.0, None)
    assert evals is None and ts == 0.0
    gaps = []
    for k in range(2):
        e, c = sla.eigh(hs[k], ss[k])
        co = c[:, :7]
        assert np.abs(rho[k] - 2.0 * co @ co.conj().T).max() < 1e-11
        assert np.abs(wdm[k] - 2.0 * (co * e[:7]) @ co.conj().T).max() < 1e-11
        gaps.append((e[6], e[7]))
    assert max(g[0] for g in gaps) < mu < min(g[1] for g in gaps)


# ---- in the SCC -------------------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    "h2o",                                        # closed shell
    "o2_triplet_initial_spins",                   # collinear spin: a purification per channel
    "o2_noncolinear_soc",                         # two-component, spin-orbit coupling
    "water_slab_dipole",                          # k-points, dipole correction, stress
    "h2o_lc", "diamond2_lc_k222", "o2_box_lc_spin_k222",       # hybrids, k-points, spin
    "h2co_lc_plusu", "h2o_ons", "water_dimer_h5",              # +U, onsite, H5
    "water_box_field", "h2o_gb_field_pc"])        # field (stress), point charges, solvation
def test_tight_purification_is_the_eigensolver(name):
    """A tight TRS2 (tolerance 1e-10, truncation 1e-15) gives the eigensolver's SCC: energies,
    charges, forces and stress to 2e-10 in the same SCC iterations, also where DFTB+'s NTPoly
    refuses (spin, k-points with one process) or is wrong (block potentials in dense form,
    B28)."""
    a, b = _run(name), _run(name, NTPoly(**TIGHT))
    de, dq, df = _deviations(a, b)
    assert de <= 2e-10 and dq <= 2e-10 and df <= 1e-10
    assert a.n_scc == b.n_scc
    assert b.eigenvalues is None and b.occupations is None and b.ts == 0.0
    if a.stress is not None:
        assert np.abs(a.stress - b.stress).max() <= 1e-12


def test_eigenstates_after_purification(tmp_path):
    """The band and DOS module diagonalises at the SCC charges (DFTB+'s band-structure run), so
    it works after a purification: the eigensolver's levels to 1e-10.  band.out, written from
    the SCC's own levels, says that there are none."""
    case = refgen.CASES_BY_NAME["water_slab_dipole"]
    inp = refgen._inputs(case)
    pos = np.array(inp["positions_angstrom"]) / BOHR
    cell = np.array(inp["cell_angstrom"]) / BOHR
    levels = []
    for solver in ("eigh", NTPoly(**TIGHT)):
        res, calc = refgen.run_pydftb(case, inp, engine=True, solver=solver)
        st = eigenstates(calc, inp["labels"], pos, cell, np.ones(3, bool), res.q_in,
                         scc_fermi=res.fermi)
        levels.append(np.asarray(st.eigenvalues))
    assert np.abs(levels[0] - levels[1]).max() < 1e-10
    with pytest.raises(ValueError, match="band.out needs the eigenvalues"):
        io.write_band_out(tmp_path / "band.out", res)


def test_xlbomd_and_ase():
    """XL-BOMD through ASE (fast, one SCC iteration per step once the history is full) with the
    purification: the eigensolver's trajectory; ASE's eigenvalues and occupations say that the
    solver has none, its Fermi level is the chemical potential."""
    inp = dynamics.load("water_dimer_xlfast_md")["input"]
    runs = []
    for solver in ("eigh", NTPoly(**TIGHT)):
        atoms = dynamics._atoms(inp)
        atoms.set_velocities(np.array(inp["velocities_aaps"]) / (1000.0 * units.fs))
        calc = PyDFTB(refgen.SKF_SETS["ptbp_hcocu"], max_l=inp["max_l"], scc_tol=1e-10,
                      max_scc=500, h_damping=4.0, xlbomd=XLBOMD(fast=True, scale=0.5),
                      solver=solver)
        atoms.calc = calc
        energies, n_scc = [], []

        def record(atoms=atoms, calc=calc, energies=energies, n_scc=n_scc):
            energies.append(atoms.get_potential_energy() + atoms.get_kinetic_energy())
            n_scc.append(calc.raw.n_scc)

        md = VelocityVerlet(atoms, timestep=0.5 * units.fs, logfile=None)
        md.attach(record, interval=1)
        md.run(20)
        runs.append((np.array(energies), n_scc, atoms.positions.copy()))
    (e0, n0, x0), (e1, n1, x1) = runs
    assert n1 == n0 and n1[-1] == 1
    assert np.abs(e1 - e0).max() < 1e-8 and np.abs(x1 - x0).max() < 1e-9
    calc = atoms.calc
    for get in (calc.get_eigenvalues, calc.get_occupation_numbers):
        with pytest.raises(RuntimeError, match="not available with this solver"):
            get()
    assert calc.get_fermi_level() == calc.raw.fermi * units.Hartree


def test_chemical_potential_is_the_fermi_level():
    """The Fermi level is NTPoly's chemical potential: in the gap, one per spin channel."""
    a, b = _run("h2o"), _run("h2o", "ntpoly")
    assert a.eigenvalues[0][3] < b.fermi < a.eigenvalues[0][4]
    a, b = _run("o2_triplet_initial_spins"), _run("o2_triplet_initial_spins", "ntpoly")
    for ev, mu, n in zip(a.eigenvalues, b.fermi_spin, (7, 5)):    # 7 up and 5 down electrons
        assert ev[n - 1] < mu < ev[n]
    assert b.fermi == b.fermi_spin[0]


def test_spin_channels_share_the_inverse_square_root():
    sol = NTPoly(**TIGHT)
    _run("o2_box_lc_spin_k222", sol)
    assert len(sol.isr_iterations) == 8 and len(sol.z) == 16
    assert all(len(its) == 8 for its in sol.last_iterations)


def test_density_analyses():
    """Atom energies, net populations, CM5 and the bond analyses need only the density."""
    spec = {"atom_energies": True, "net_populations": True, "cm5": True,
            "bond_populations": True, "bond_energies": True, "bond_orders": True}
    a = _run("ch3oh", analysis=spec).analysis
    b = _run("ch3oh", NTPoly(**TIGHT), analysis=spec).analysis
    for key in ("net_populations", "cm5_charges", "bond_populations", "bond_energies",
                "bond_orders"):
        assert np.abs(np.asarray(getattr(a, key)) - np.asarray(getattr(b, key))).max() < 1e-12
    for key, value in a.atom_energies.items():
        assert np.abs(value - b.atom_energies[key]).max() < 1e-12


def test_temperature_is_ignored_with_a_warning():
    """NTPoly gives the zero-temperature density; DFTB+ ignores the temperature, pydftb warns."""
    with pytest.warns(RuntimeWarning, match="zero-temperature density"):
        warm = _run("h2o", "ntpoly", filling=("fermi", 300.0))
    cold = _run("h2o", "ntpoly")
    assert warm.energy == pytest.approx(cold.energy, abs=1e-14)
    assert warm.ts == 0.0


def test_mixed_precision():
    """precision="mixed" (pydftb's): single-precision steps, then at most four in double; at
    the SCC tolerance 1e-5 the charges move by about 1e-6.  A tighter SCC tolerance warns."""
    for name in ("h2o", "ch3oh", "water_box_mixer_diis"):
        a = _run(name, NTPoly(), scc_tol=1e-5)
        b = _run(name, NTPoly(precision="mixed"), scc_tol=1e-5)
        de, dq, df = _deviations(a, b)
        assert de <= 2e-6 and dq <= 1e-5 and df <= 2e-6
    sk = ParamSet(SKF_HCOCU).skset(["O", "H"])
    with pytest.warns(RuntimeWarning, match="below what the mixed-precision purification"):
        DFTB(sk, solver=NTPoly(precision="mixed"), scc_tol=1e-10)


@TORCH
def test_torch_backend():
    """The NTPoly solver on the torch device: float64 is NumPy's to rounding, float32 to 1e-6
    (charges 1e-5)."""
    for name in ("h2o", "ch3oh"):
        a = _run(name, "ntpoly", scc_tol=1e-5)
        de, dq, df = _deviations(a, _run(name, "ntpoly", "torch", "float64", scc_tol=1e-5))
        assert de <= 1e-13 and dq <= 1e-13 and df <= 1e-13
        de, dq, df = _deviations(a, _run(name, "ntpoly", "torch", "float32", scc_tol=1e-5))
        assert de <= 1e-6 and dq <= 1e-5 and df <= 1e-6


# ---- refusals -----------------------------------------------------------------------------------
def test_refusals(sk):
    """DFTB+'s conditions on ELSI's NTPoly (its error texts), and pydftb's own."""
    w = {"O": -0.028, "H": -0.072, "C": -0.023}
    for opts, msg in (
            (dict(solver="omm"), "OMM .* is not implemented"),
            (dict(solver="pexsi"), "PEXSI is not implemented"),
            (dict(solver="lapack"), "unknown solver 'lapack'"),
            (dict(smearing=("mp", 1000.0, 2)), "Fermi function distributed electrons"),
            (dict(smearing=("gaussian", 1000.0)), "Fermi function distributed electrons"),
            (dict(spin=True, spin_constants=w, relax_total_spin=True), "relax_total_spin"),
            (dict(fixed_fermi_level=-0.2), r"eigenvalues \(not the NTPoly solver\)"),
            (dict(independent_k_filling=True), r"eigenvalues \(not the NTPoly solver\)"),
            (dict(analysis={"eigenvectors": True}),
             r"the analysis \(eigenvectors\) needs an eigensolver \(not the NTPoly solver\)"),
            (dict(analysis={"localise": True, "polarisability": True}),
             r"\(localised orbitals, responses\)"),
            (dict(casida=cs.Casida(4)),
             r"Casida linear response needs eigenvectors \(not the NTPoly solver\)")):
        with pytest.raises(ValueError, match=msg):
            DFTB(sk, **{"solver": "ntpoly", **opts})


def test_odd_electron_count_is_refused(sk):
    """Each spin channel needs a whole number of occupied levels: DFTB+'s NTPoly rounds an odd
    electron count up to an even one (B29: H2O+ is computed as neutral water); pydftb refuses
    and points to spin polarisation."""
    case = refgen.CASES_BY_NAME["h2o"]
    pos = np.array(refgen._inputs(case)["positions_angstrom"]) / BOHR
    with pytest.raises(ValueError, match="whole number of occupied levels"):
        DFTB(sk, solver="ntpoly", charge=1.0).calculate(["O", "H", "H"], pos)


@TORCH
def test_torch_refuses_the_lapack_drivers(sk):
    with pytest.raises(ValueError, match="the torch backend has its own eigensolver"):
        DFTB(sk, solver="qr", backend="torch")


# ---- DFTB+'s LAPACK drivers -------------------------------------------------------------------
@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("driver", ["gv", "gvd", "gvr"])
def test_lapack_drivers(driver, complex_):
    """QR, DivideAndConquer and RelativelyRobust: the eigenpairs of H C = S C e, C^H S C = 1."""
    h = _gapped(30, 10, 20 + complex_, complex_)
    s = _gapped(30, 0, 22 + complex_, complex_, gap=1.0)
    e, c = ops.eigh_driver(h, s, driver)
    assert np.abs(e - sla.eigh(h, s, eigvals_only=True)).max() < 1e-13
    assert np.abs(h @ c - (s @ c) * e).max() < 1e-12
    assert np.abs(c.conj().T @ s @ c - np.eye(30)).max() < 1e-12
    with pytest.raises(ValueError, match="unknown LAPACK driver"):
        ops.eigh_driver(h, s, "gvx")


@pytest.mark.parametrize("solver", ["qr", "divideandconquer", "relativelyrobust", "elpa"])
def test_dftbplus_eigensolvers(solver):
    """DFTB+'s Solver names give the eigensolver's results; DivideAndConquer and ELPA make the
    LAPACK call of "eigh" (bitwise)."""
    a, b = _run("ch3oh"), _run("ch3oh", solver)
    if solver in ("divideandconquer", "elpa"):
        assert b.energy == a.energy and np.array_equal(b.charges, a.charges)
    de, dq, df = _deviations(a, b)
    assert de <= 1e-11 and dq <= 1e-11 and df <= 1e-11
    assert b.n_scc == a.n_scc
