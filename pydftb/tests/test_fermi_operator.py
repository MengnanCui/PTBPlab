"""Finite-temperature densities without eigenvalues (milestone M7b): the recursive Fermi-operator
expansion against eigendecompositions (density, entropy, the exact shift of the chemical
potential and its search), pydftb's SCC with solver="foe" against the eigensolver with the exact
electron count, dynamics, the refusals, torch, and a group of DFTB+'s frozen Fermi references (all
of them: ``refgen --check --solver foe``).
"""
import dataclasses
import importlib.util

import numpy as np
import pytest
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet
from scipy.optimize import brentq

from pydftb import casida as cs
from pydftb import dftbplus_io as io
from pydftb import fermi_operator as fo
from pydftb.ase_calc import PyDFTB
from pydftb.bands import eigenstates
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.fermi_operator import FermiOperator
from pydftb.occupations import Smearing, fill, fill_at, fill_independent_k
from pydftb.params import ParamSet
from pydftb.validation import refgen

from .conftest import SKF_HCOCU

TORCH = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
KT = Smearing("fermi", 1000.0).kt


def _hermitian(n, seed, complex_=False, gap=None):
    """A random Hermitian matrix with its eigenpairs: levels in [-1, 1] (``gap``: none in
    [-gap/2, gap/2], a third below it)."""
    rng = np.random.default_rng(seed)
    if gap is None:
        e = np.sort(rng.uniform(-1.0, 1.0, n))
    else:
        e = np.sort(np.concatenate([rng.uniform(-1.0, -gap / 2, n // 3),
                                    rng.uniform(gap / 2, 1.0, n - n // 3)]))
    a = rng.normal(size=(n, n)) + (1j * rng.normal(size=(n, n)) if complex_ else 0.0)
    q = np.linalg.qr(a)[0]
    return (q * e) @ q.conj().T, e, q


def _fermi(e, mu, kt):
    return 0.5 * (1.0 - np.tanh(0.5 * (e - mu) / kt))


def _entropy(f):
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-(np.where(f > 0, f * np.log(f), 0.0)
                       + np.where(f < 1, (1 - f) * np.log(1 - f), 0.0)).sum())


def _run(name, solver="eigh", backend=None, dtype=None, **over):
    """A frozen case with pydftb; the eigensolver with the exact-count Fermi level (the
    Fermi-operator expansion's: DFTB+'s mid-gap rule needs eigenvalues)."""
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], **over)
    opts = {}
    if solver == "eigh":
        opts["smearing"] = Smearing(case.filling[0], case.filling[1], fermi_level="exact")
    return refgen.run_pydftb(case, refgen._inputs(case), solver=solver, backend=backend,
                             dtype=dtype, **opts)


def _host(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _deviations(a, b):
    return (abs(a.energy - float(b.energy)),
            np.abs(np.asarray(a.charges) - _host(b.charges)).max(),
            np.abs(np.asarray(a.forces) - _host(b.forces)).max())


@pytest.fixture(scope="module")
def sk():
    return ParamSet(SKF_HCOCU).skset(["C", "O", "H"])


# ---- the recursion --------------------------------------------------------------------------
@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("temperature", [300.0, 3000.0])
def test_recursion_is_the_fermi_function(temperature, complex_):
    """22 doublings from X0 = 1/2 - T/4: f((H - mu0)/kT) and its entropy, from the
    log-determinants of the steps, to rounding."""
    kt = Smearing("fermi", temperature).kt
    h, e, q = _hermitian(40, 1 + complex_, complex_)
    r = fo.Recursion(h, 0.05, kt)
    p, ld = r.shifted(0.0)
    f = _fermi(e, 0.05, kt)
    assert r.steps == fo.STEPS
    assert np.abs(p - (q * f) @ q.conj().T).max() < 1e-13
    assert abs(r.entropy(p, 0.0, ld) - _entropy(f)) < 1e-11


def test_entropy_of_a_spectrum_symmetric_about_mu():
    """The odd powers of T = (H - mu0)/(kT 2^n) have no trace when the levels are symmetric
    about mu0: the series of tr ln(1 + T/2) goes on to the even ones (the first version
    stopped there, 5e-6 off at the mid-gap level of a k-point)."""
    e = np.array([-0.03, -0.01, 0.01, 0.03])
    q = np.linalg.qr(np.random.default_rng(3).normal(size=(4, 4)))[0]
    r = fo.Recursion((q * e) @ q.T, 0.0, KT)
    p, ld = r.shifted(0.0)
    assert abs(r.entropy(p, 0.0, ld) - _entropy(_fermi(e, 0.0, KT))) < 1e-13


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
def test_exact_shift(complex_):
    """P(mu0 + d) = P [P + e^(-d/kT) (1 - P)]^-1 within the window of 5 kT: the density, the
    electron count and its derivative, the entropy."""
    h, e, q = _hermitian(40, 5 + complex_, complex_)
    r = fo.Recursion(h, 0.05, KT)
    for d in (-5 * KT, -KT, 0.3 * KT, 5 * KT):
        p, ld = r.shifted(d)
        f = _fermi(e, 0.05 + d, KT)
        assert np.abs(p - (q * f) @ q.conj().T).max() < 1e-12
        assert abs(r.entropy(p, d, ld) - _entropy(f)) < 1e-10
        n, dn = fo._count([r], [1.0], d)
        assert abs(n - f.sum()) < 1e-11
        assert abs(dn - (f * (1 - f)).sum() / KT) < 1e-10      # (tr P - tr P^2)/kT: cancellation


def test_chemical_potential():
    """The search over three weighted k-points: the electron count to 1e-12 from a start 100 kT
    away (several recursions) and from its own result (one); in a gap any level holds the count
    to rounding, and the density is the Fermi density there."""
    hs, evals = [], []
    for i in range(3):
        h, e, _ = _hermitian(30, 10 + i, True)
        hs.append(h)
        evals.append(e)
    w, target = [0.5, 0.3, 0.2], 12.3

    def count(mu):
        return sum(wk * _fermi(ek, mu, KT).sum() for wk, ek in zip(w, evals))

    exact = brentq(lambda m: count(m) - target, -1.5, 1.5, xtol=1e-15)
    mu, recs, d, n_rec = fo.chemical_potential(hs, w, target, KT, exact + 100 * KT)
    assert abs(mu - exact) < 1e-12 and n_rec > 1
    again, _, _, n_rec = fo.chemical_potential(hs, w, target, KT, mu)
    assert abs(again - mu) < 1e-14 and n_rec == 1
    h, e, q = _hermitian(30, 20, gap=0.4)
    mu, recs, d, _ = fo.chemical_potential([h], [1.0], 10.0, KT)
    assert e[9] < mu < e[10] and abs(_fermi(e, mu, KT).sum() - 10.0) < 1e-11
    p, _ = recs[0].shifted(d)
    assert np.abs(p - (q * _fermi(e, mu, KT)) @ q.T).max() < 1e-13


def test_low_temperature_takes_more_steps():
    """At 1 K ||Y||_1 exceeds 2^(22 - 8): more doublings, the density still to rounding; T S to
    rounding too (S itself to ||Y|| N eps, the size of its cancelling terms)."""
    h, e, q = _hermitian(30, 7)
    kt = Smearing("fermi", 1.0).kt
    r = fo.Recursion(h, 0.0, kt)
    assert r.steps > fo.STEPS and 2.0 ** (r.steps - 8) >= fo._norm1(r.y)
    p, ld = r.shifted(0.0)
    f = _fermi(e, 0.0, kt)
    assert np.abs(p - (q * f) @ q.T).max() < 1e-12
    assert abs(kt * (r.entropy(p, 0.0, ld) - _entropy(f))) < 1e-13


def _generalised(nk, n, seed):
    """Random complex H(k), S(k) (S positive definite)."""
    rng = np.random.default_rng(seed)
    hs, ss = [], []
    for _ in range(nk):
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        b = 0.05 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        hs.append(0.1 * (a + a.conj().T))
        ss.append(np.eye(n) + 0.5 * (b @ b.conj().T))
    return np.array(hs), np.array(ss)


def _eig_density(h, s, kw, nelec, kt, filling, spin_factor=2.0):
    """rho, W, mu, T S by the eigendecomposition and pydftb's exact-count filling."""
    import scipy.linalg as sla
    es, cs_ = zip(*(sla.eigh(hk, sk_) for hk, sk_ in zip(h, s)))
    es = np.array(es)
    sm = Smearing("fermi", kt / Smearing("fermi", 1.0).kt, fermi_level="exact")
    occ, mu, ts = filling(es, kw, nelec, sm, spin_factor)
    occ = np.asarray(occ)
    rho = np.array([(c * f) @ c.conj().T for c, f in zip(cs_, occ)])
    wdm = np.array([(c * (f * e)) @ c.conj().T for c, e, f in zip(cs_, es, occ)])
    return rho, wdm, mu, ts


def test_solver_interface():
    """FermiOperator on k-point stacks of generalised problems against the eigendecomposition
    (20000 K: about two levels per kT, so the count fixes the chemical potential): the electron
    count shared by the k-points (solve, the closed-shell interface), a fixed chemical
    potential, every k-point on its own, two channels sharing the chemical potential; rho, W,
    the chemical potential and T S."""
    h, s = _generalised(3, 16, 1)
    kw = np.array([0.25, 0.5, 0.25])
    sm = Smearing("fermi", 20000.0)
    kt = sm.kt
    sol = FermiOperator()
    sol.prepare(s)
    rho, wdm, evals, mu, ts = sol.solve(h, kw, 10.0, sm)
    ref = _eig_density(h, s, kw, 10.0, kt, fill)
    assert evals is None and sol.last_occupations is None
    for x, y in zip((rho, wdm, mu, ts), ref):
        assert np.abs(np.asarray(x) - np.asarray(y)).max() < 1e-12
    out = sol.density(h, kw, 5.0, 2.0, True, kt=kt, mu=ref[2] + 0.01)
    ref = _eig_density(h, s, kw, ref[2] + 0.01, kt,
                       lambda e, w, mu, sm_, g: fill_at(e, w, mu, sm_, g))
    for x, y in zip(out, ref):
        assert np.abs(np.asarray(x) - np.asarray(y)).max() < 1e-12
    out = sol.density(h, kw, 5.0, 2.0, True, kt=kt, independent_k=True)
    ref = _eig_density(h, s, kw, 10.0, kt, fill_independent_k)
    for x, y in zip(out, ref):
        assert np.abs(np.asarray(x) - np.asarray(y)).max() < 1e-12
    h2, s2 = np.concatenate([h, h + 0.02 * s]), np.concatenate([s, s])
    sol.prepare(s2)
    assert sol.z[0] is sol.z[3]                     # the spin channels' copy is not redone
    out = sol.density(h2, kw, 10.0, 1.0, True, kt=kt, channels=2)
    ref = _eig_density(h2, s2, np.tile(kw, 2), 10.0, kt, fill, spin_factor=1.0)
    for x, y in zip(out, ref):
        assert np.abs(np.asarray(x) - np.asarray(y)).max() < 1e-12


# ---- in the SCC -------------------------------------------------------------------------------
METALS = {"cu4co_300K", "oh_noncolinear_soc", "cu4_lc_1000K", "cuo_chain_k422", "cu4co_plusu",
          "cu4co_300K_fixed_ef", "cuo_chain_k422_indep_k", "cu_fcc_lc_k333_1000K"}


@pytest.mark.parametrize("name", [
    "cu4co_300K",                                 # no spin
    "oh_radical",                                 # collinear spin: a chemical potential each
    "o2_relaxed_spin", "ch3_lc_relax_spin",       # relax_total_spin: one for both channels
    "oh_noncolinear_soc",                         # two-component, spin-orbit coupling
    "cu4_lc_1000K", "cu_fcc_lc_k333_1000K",       # hybrids, k-points
    "cuo_chain_k422",                             # k-points, stress
    "cu4co_plusu",                                # DFTB+U
    "cu4co_300K_fixed_ef",                        # a fixed Fermi level
    "cuo_chain_k422_indep_k"])                    # every k-point on its own
def test_scc_is_the_eigensolver(name):
    """solver="foe" gives the eigensolver's SCC (exact-count Fermi level): energies, T S,
    charges, forces, stress to 1e-11 in the same SCC iterations; the same Fermi level where the
    density fixes it (not inside a gap, where any level holds the count)."""
    a, b = _run(name), _run(name, "foe")
    de, dq, df = _deviations(a, b)
    assert de <= 1e-11 and dq <= 1e-11 and df <= 1e-11
    assert abs(a.free_energy - b.free_energy) <= 1e-11 and abs(a.ts - b.ts) <= 1e-12
    assert a.n_scc == b.n_scc
    assert b.eigenvalues is None and b.occupations is None
    if a.stress is not None:
        assert np.abs(a.stress - b.stress).max() <= 1e-12
    if name in METALS:
        mine = np.ravel(b.fermi_spin if b.fermi_spin is not None else b.fermi)
        theirs = np.ravel(a.fermi_spin if a.fermi_spin is not None else a.fermi)
        assert np.abs(mine - theirs).max() < 1e-10


def test_forces_are_free_energy_derivatives():
    """The forces with the Fermi-operator expansion are -dF/dR of the Mermin free energy; the
    calculator keeps its FermiOperator from one geometry to the next."""
    case = dataclasses.replace(refgen.CASES_BY_NAME["cu4co_300K"], scc_tol=1e-12)
    inp = refgen._inputs(case)
    res, calc = refgen.run_pydftb(case, inp, engine=True, solver="foe")
    sol = calc._foe
    pos = np.array(inp["positions_angstrom"]) / BOHR
    step = 1e-4
    for atom, axis in ((0, 0), (4, 2)):
        free = []
        for sign in (1.0, -1.0):
            p = pos.copy()
            p[atom, axis] += sign * step
            free.append(calc.calculate(inp["labels"], p, forces=False).free_energy)
        assert abs(-(free[0] - free[1]) / (2 * step) - res.forces[atom, axis]) < 1e-7
    assert isinstance(sol, FermiOperator) and calc._foe is sol and sol._mu


def test_dynamics_carry_the_chemical_potential(tmp_path):
    """Velocity Verlet through ASE at 1000 K: the eigensolver's trajectory; solver="foe" keeps
    one FermiOperator for all steps (ASE builds a DFTB per step), whose chemical potential
    starts the next search: one recursion per density call here.  The eigenvalues,
    occupations and band.out say that the solver has none."""
    case = refgen.CASES_BY_NAME["cu4co_300K"]
    inp = refgen._inputs(case)
    sm = Smearing("fermi", 1000.0, fermi_level="exact")
    runs, counts, starts = [], [], []
    for solver in ("eigh", "foe"):
        atoms = Atoms(inp["labels"], positions=inp["positions_angstrom"])
        atoms.set_velocities(np.random.default_rng(7).normal(scale=0.02, size=(len(atoms), 3)))
        calc = PyDFTB(SKF_HCOCU, max_l=inp["max_l"], scc_tol=1e-10, smearing=sm, solver=solver)
        if solver == "foe":
            sol = calc.engine_opts["solver"]
            assert isinstance(sol, FermiOperator)

            def counted(*args, sol=sol, density=sol.density, **kw):
                starts.append(dict(sol._mu))
                out = density(*args, **kw)
                counts.append(sol.recursions)
                return out

            sol.density = counted
        atoms.calc = calc
        energies = []
        md = VelocityVerlet(atoms, timestep=0.5 * units.fs, logfile=None)
        md.attach(lambda a=atoms, e=energies: e.append(a.get_potential_energy()
                                                         + a.get_kinetic_energy()), interval=1)
        md.run(6)
        runs.append((np.array(energies), atoms.positions.copy()))
    (e0, x0), (e1, x1) = runs
    assert np.abs(e1 - e0).max() < 1e-9 and np.abs(x1 - x0).max() < 1e-10
    assert starts[0] == {} and all(starts[1:])    # from scratch once, then the last mu
    assert max(counts) == 1
    for get in (calc.get_eigenvalues, calc.get_occupation_numbers):
        with pytest.raises(RuntimeError, match="not available with this solver"):
            get()
    with pytest.raises(ValueError, match="band.out needs the eigenvalues"):
        io.write_band_out(tmp_path / "band.out", calc.raw)


def test_eigenstates_after_the_fermi_operator():
    """The band module diagonalises at the SCC charges: the eigensolver's levels."""
    case = refgen.CASES_BY_NAME["cuo_chain_k422"]
    inp = refgen._inputs(case)
    pos = np.array(inp["positions_angstrom"]) / BOHR
    cell = np.array(inp["cell_angstrom"]) / BOHR
    levels = []
    for solver in ("eigh", "foe"):
        res, calc = refgen.run_pydftb(case, inp, engine=True, solver=solver)
        st = eigenstates(calc, inp["labels"], pos, cell, np.ones(3, bool), res.q_in,
                         scc_fermi=res.fermi)
        levels.append(np.asarray(st.eigenvalues))
    assert np.abs(levels[0] - levels[1]).max() < 1e-10


def test_density_analyses():
    """Atom energies, net populations, CM5 and the bond analyses need only the density."""
    spec = {"atom_energies": True, "net_populations": True, "cm5": True,
            "bond_populations": True, "bond_energies": True, "bond_orders": True}
    a = _run("cu4co_300K", analysis=spec).analysis
    b = _run("cu4co_300K", "foe", analysis=spec).analysis
    for key in ("net_populations", "cm5_charges", "bond_populations", "bond_energies",
                "bond_orders"):
        assert np.abs(np.asarray(getattr(a, key)) - np.asarray(getattr(b, key))).max() < 1e-11
    for key, value in a.atom_energies.items():
        assert np.abs(value - b.atom_energies[key]).max() < 1e-11


@pytest.mark.parametrize("name", ["cu4co_300K", "cuo_chain_k422", "o2_relaxed_spin",
                                  "oh_noncolinear_soc", "cu4co_300K_fixed_ef",
                                  "cuo_chain_k422_indep_k", "cu_fcc_lc_k333_1000K",
                                  "cu4co_plusu_atom"])
def test_dftbplus_references(name):
    """DFTB+'s frozen references (made with its eigensolver) with solver="foe": within their
    tolerances, in DFTB+'s SCC iterations, or (cuo_chain_k422_indep_k: DFTB+'s mid-gap level at
    some k-points) in those of the eigensolver with the exact electron count."""
    r = refgen.compare(refgen.load(name), solver="foe")
    assert r["ok"], r["dev"]
    assert r["n_scc"] == r["dev"].get("fermi_rule", r["n_scc_ref"])
    if "fermi_rule" in r["dev"]:
        assert name == "cuo_chain_k422_indep_k" and r["dev"]["fermi_rule_energy"] < 1e-13


# ---- refusals ---------------------------------------------------------------------------------
def test_refusals(sk):
    """Fermi filling at T > 0 only, no eigenpairs; PEXSI points here."""
    for opts, msg in (
            (dict(smearing=("mp", 1000.0, 2)), "Fermi filling only"),
            (dict(smearing=("gaussian", 1000.0)), "Fermi filling only"),
            (dict(temperature=0.0), r"needs an electronic temperature: use solver='ntpoly'"),
            (dict(analysis={"eigenvectors": True}),
             r"the analysis \(eigenvectors\) needs an eigensolver \(not the FOE solver\)"),
            (dict(analysis={"localise": True, "polarisability": True}),
             r"\(localised orbitals, responses\)"),
            (dict(casida=cs.Casida(4)), r"needs eigenvectors \(not the FOE solver\)")):
        with pytest.raises(ValueError, match=msg):
            DFTB(sk, **{"solver": "foe", "temperature": 1000.0, **opts})
    with pytest.raises(ValueError, match="PEXSI is not implemented.*use 'foe'"):
        DFTB(sk, solver="pexsi")
    w = {"O": -0.028, "H": -0.072, "C": -0.023}
    DFTB(sk, solver="foe", temperature=1000.0, spin=True, spin_constants=w,
         relax_total_spin=True)                  # a shared Fermi level
    DFTB(sk, solver="foe", temperature=1000.0, fixed_fermi_level=-0.2)
    DFTB(sk, solver="foe", temperature=1000.0, independent_k_filling=True)


@TORCH
def test_torch_backend():
    """The Fermi-operator expansion on the torch device: float64 is NumPy's to rounding, float32
    to about 1e-6 (charges 1e-5)."""
    for name in ("cu4co_300K", "cuo_chain_k422"):
        a = _run(name, "foe")
        b = _run(name, "foe", "torch", "float64")
        de, dq, df = _deviations(a, b)
        assert de <= 1e-12 and dq <= 1e-12 and df <= 1e-12 and a.n_scc == b.n_scc
        assert abs(a.ts - float(b.ts)) <= 1e-13
        de, dq, df = _deviations(_run(name, "foe", scc_tol=1e-5),
                                 _run(name, "foe", "torch", "float32", scc_tol=1e-5))
        assert de <= 2e-6 and dq <= 1e-5 and df <= 2e-6
