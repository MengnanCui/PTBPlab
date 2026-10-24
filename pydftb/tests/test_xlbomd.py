"""XL-BOMD (milestone M6e): the integrator of the auxiliary populations, the shadow energy and
its forces, the dynamics force evaluations, the ASE interface and the refusals.

The comparisons with DFTB+'s own XL-BOMD (positions, velocities, shadow energies and the SCC
iterations of every step; K = 5, 6, 7, PreSteps, XlbomdFast, DFTB3, k-points at 500 K) are
the frozen *_xl*_md references in test_md_opt.py.  Here: DFTB+'s coefficients and phases,
the ring buffer against the recursion written out, forces and stress as derivatives of the
shadow energy at fixed auxiliary populations (exact SK derivatives), the shadow energy and
the three force evaluations at self-consistency, MD through ASE, and what DFTB+ refuses.
"""
from dataclasses import replace

import numpy as np
import pytest
from ase import units
from ase.md.verlet import VelocityVerlet

from pydftb.ase_calc import PyDFTB
from pydftb.calculator import DFTB, SCCControl
from pydftb.casida import Casida
from pydftb.constants import BOHR
from pydftb.params import ParamSet
from pydftb.validation import dynamics, refgen
from pydftb.validation.cases import CASES_BY_NAME, DFTB3_3OB
from pydftb.xlbomd import COEFFS, ON, XLBOMD, ExtLagrangian, Integrator

DIMER = CASES_BY_NAME["water_dimer_hdamp"]           # DFTB2 with H-X damping, T = 0
DIMER3 = replace(DIMER, dftb3=DFTB3_3OB)             # full third order (ThirdOrderFull)
CUO = CASES_BY_NAME["cuo_chain_k422"]                # k-points, Fermi 500 K
DIMER_H5 = CASES_BY_NAME["water_dimer_h5"]           # H5 (M6f-3; DFTB+'s XL gradient lacks it)
ONE_STEP = SCCControl(1, 1, 1e-5)                    # XlbomdFast: one SCC iteration


def test_coefficients_as_dftbplus():
    """DFTB+ extlagrangian.F90: the dissipation coefficients sum to zero (a constant history
    is kept), kappa and alpha; alpha7 is written there without the _dp kind (single
    precision)."""
    for k, (c, _, _) in COEFFS.items():
        assert len(c) == k + 1 and c.sum() == 0.0
    assert COEFFS[5][1:] == (18e-3, 1.82) and COEFFS[6][1:] == (5.5e-3, 1.84)
    assert COEFFS[7][1] == float(np.float32(1.6e-3)) != 1.6e-3 and COEFFS[7][2] == 1.86


def _recursion(outs, k, n_transient, scale):
    """The XL recursion written out with the whole history (no ring buffer)."""
    c, alpha, kappa = COEFFS[k]
    hist, out = [], []
    for i, q in enumerate(outs):
        if i <= k:                                  # filling up: the outputs as they are
            hist.append(q.copy())
            out.append(q.copy())
            continue
        j = i - k                                   # steps since the history was full
        if j <= n_transient:                        # XlbomdFast: blend in the output
            cc = j / (n_transient + 1)
            hist[-1] = cc * hist[-1] + (1.0 - cc) * q
        nxt = 2.0 * hist[-1] - hist[-2] + kappa * scale * (q - hist[-1])
        nxt = nxt + alpha * sum(c[m] * hist[-1 - m] for m in range(k + 1))
        hist.append(nxt)
        out.append(nxt)
    return out


@pytest.mark.parametrize("k, n_transient, scale", [(5, 0, 1.0), (6, 3, 0.5), (7, 10, 0.25)])
def test_ring_buffer(k, n_transient, scale):
    rng = np.random.default_rng(k)
    outs = [rng.normal(size=4) for _ in range(30)]
    lag = ExtLagrangian(k, scale)
    lag.turn_on(n_transient)
    got = [lag.next_input(q) for q in outs]
    assert lag.phase == ON
    np.testing.assert_allclose(got, _recursion(outs, k, n_transient, scale), rtol=0, atol=1e-13)


@pytest.mark.parametrize("opts, n_bo", [
    (XLBOMD(), 6),
    (XLBOMD(integration_steps=6, pre_steps=3), 10),
    (XLBOMD(fast=True, scale=0.5), 16),
    (XLBOMD(integration_steps=7, fast=True, transient_steps=5, scale=0.25), 13),
    (XLBOMD(min_scc=4, scc_tol=1e-3), 6)])
def test_phases(opts, n_bo):
    """Converged steps with the calculator's own SCC settings first (PreSteps, K + 1 filling
    the history, XlbomdFast's transient), then the XL settings: the counts of DFTB+'s
    references.  The minimum number of iterations only counts on the first step, which is an
    ordinary one: MinSccIterations has no effect (as in DFTB+)."""
    xl = Integrator(opts)
    seen = []
    for _ in range(n_bo + 3):
        seen.append(xl.scc_parameters(500, 1e-10))
        xl.advance(np.zeros(3))
    assert seen[:n_bo] == [(1, 500, 1e-10)] * n_bo
    assert seen[n_bo:] == [(1,) + opts.scc[1:]] * 3


def _engine(case, **opts):
    """A converged ground state and its pydftb.DFTB (exact SK derivatives)."""
    inp = refgen._inputs(case)
    res, calc = refgen.run_pydftb(case, inp, engine=True, scc_tol=1e-12, **opts)
    return res, calc, inp


def _geometry(inp):
    periodic = inp["cell_angstrom"] is not None
    cell = np.array(inp["cell_angstrom"]) / BOHR if periodic else None
    return (np.array(inp["positions_angstrom"]) / BOHR, cell,
            np.ones(3, bool) if periodic else None)


def _fd(energy, x0, idx, h):
    """-dE/dx at x0 for the flat components idx (4-point central differences)."""
    out = []
    for i in idx:
        e = {}
        for s in (-2, -1, 1, 2):
            x = x0.copy()
            x.flat[i] += s * h
            e[s] = energy(x)
        out.append(-(8.0 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12.0 * h))
    return np.array(out)


@pytest.mark.parametrize("case, kind, tol", [(DIMER, "dynamics_t0", 2e-10),
                                             (DIMER3, "dynamics_t0", 2e-10),
                                             (CUO, "dynamics", 2e-9),
                                             (DIMER_H5, "dynamics_t0", 2e-10)],
                         ids=["dftb2_hdamp", "dftb3", "cuo_k_500K", "dftb2_h5"])
def test_forces_are_the_shadow_energy_gradient(case, kind, tol):
    """One XlbomdFast step from populations n off self-consistency: the forces are
    -dE_shadow/dR at fixed n, E_shadow = E_band[D(n)] + 1/2 (2 q_out - n).gamma.n (+ the
    linearised third order), and so is the stress of the periodic chain.  With H5 too (M6f-3):
    DFTB+'s XL gradient leaves H5 out, pydftb's gamma' has it."""
    res, calc, inp = _engine(case, force_evaluation=kind)
    labels = inp["labels"]
    pos, cell, pbc = _geometry(inp)
    rng = np.random.default_rng(3)
    n = res.q_in + rng.normal(scale=3e-3, size=res.q_in.shape)
    n -= n.mean() - res.q_in.mean()                   # the electron count of the system
    xl = calc.calculate(labels, pos, cell, pbc, q_init=n, scc_control=ONE_STEP, xl=True)
    assert np.abs(xl.q_out - n).max() > 1e-3           # far from self-consistency

    def energy(p):
        return calc.calculate(labels, p, cell, pbc, forces=False, q_init=n,
                              scc_control=ONE_STEP, xl=True).free_energy

    # a step well below the SK grid spacing (0.02 Bohr): a stencil across a grid node would
    # mix two of its interpolation polynomials (1.9e-7 Ha/Bohr here with h = 1e-3)
    idx = [0, 4, 8, 10]
    assert np.abs(xl.forces.flat[idx] - _fd(energy, pos, idx, 2e-4)).max() < tol
    if cell is not None:
        def strained(eps):
            strain = np.eye(3) + eps.reshape(3, 3)
            return calc.calculate(labels, pos @ strain, cell @ strain, pbc, forces=False,
                                  q_init=n, scc_control=ONE_STEP, xl=True).free_energy

        vol = abs(np.linalg.det(cell))
        sigma = _fd(strained, np.zeros(9), [0, 1], 1e-4) / vol      # -(1/V) dE/d(strain)
        assert np.abs(xl.stress.flat[[0, 1]] - sigma).max() < 1e-9


@pytest.mark.parametrize("case, kinds", [(DIMER, ("dynamics_t0", "dynamics")),
                                         (DIMER3, ("dynamics_t0",)), (CUO, ("dynamics",))],
                         ids=["dftb2_hdamp", "dftb3", "cuo_k_500K"])
def test_self_consistent_limit(case, kinds):
    """At self-consistency the shadow energy is the energy, and DFTB+'s dynamics force
    evaluations (1/2 DHD at T = 0, 1/2 (DHS^-1 + S^-1HD)) give the ordinary forces."""
    ref, _, inp = _engine(case)
    pos, cell, pbc = _geometry(inp)
    for kind in kinds:
        res, calc, _ = _engine(case, force_evaluation=kind)
        assert np.abs(res.forces - ref.forces).max() < 1e-10
        xl = calc.calculate(inp["labels"], pos, cell, pbc, q_init=res.q_in,
                            scc_control=SCCControl(1, 500, 1e-12), xl=True)
        assert abs(xl.free_energy - ref.free_energy) < 1e-11
        assert np.abs(xl.forces - ref.forces).max() < 1e-9
        if cell is not None:
            assert np.abs(xl.stress - ref.stress).max() < 1e-11


@pytest.mark.parametrize("name, kinds", [("o2_triplet_wmatrix", ("dynamics_t0", "dynamics")),
                                         ("oh_noncolinear_soc", ("dynamics",))])
def test_force_evaluations_with_spin(name, kinds):
    """Per spin channel (or spinor) the occupations are 0 or 1, so the T = 0 form is D H D,
    not 1/2 D H D: DFTB+'s DynamicsT0 forces are wrong there (0.19 Ha/Bohr for this O2),
    pydftb's are the ordinary ones; Dynamics needs the spinor overlap S x 1."""
    ref, _, inp = _engine(CASES_BY_NAME[name])
    for kind in kinds:
        res, _, _ = _engine(CASES_BY_NAME[name], force_evaluation=kind)
        assert np.abs(res.forces - ref.forces).max() < 1e-12


def _dimer_atoms():
    inp = dynamics.load("water_dimer_xlfast_md")["input"]       # 500 K velocities
    atoms = dynamics._atoms(inp)
    atoms.set_velocities(np.array(inp["velocities_aaps"]) / (1000.0 * units.fs))
    return atoms, inp


def test_ase_md():
    """PyDFTB(xlbomd=...): one calculation per MD step, the integrator kept in between;
    converged steps until the history is full and the transient is over, then one SCC
    iteration per step; ASE's energy is the shadow energy; reset() starts again."""
    atoms, inp = _dimer_atoms()
    calc = PyDFTB(refgen.SKF_SETS[DIMER.skf_set], max_l=inp["max_l"], scc_tol=1e-10,
                  max_scc=500, h_damping=4.0, xlbomd=XLBOMD(fast=True, scale=0.5))
    assert calc.engine_opts["force_evaluation"] == "dynamics_t0"       # T = 0: DynamicsT0
    atoms.calc = calc
    n_scc, energies = [], []

    def record():
        n_scc.append(calc.raw.n_scc)
        energies.append(atoms.get_potential_energy() + atoms.get_kinetic_energy())
        assert atoms.get_potential_energy() == pytest.approx(calc.raw.free_energy
                                                             * units.Hartree, abs=1e-12)

    md = VelocityVerlet(atoms, timestep=0.5 * units.fs, logfile=None)
    md.attach(record, interval=1)
    md.run(24)
    assert min(n_scc[:16]) > 1 and n_scc[16:] == [1] * 9
    assert abs(energies[-1] - energies[0]) < 0.02                       # eV
    calc.reset()
    assert calc._xl is None
    atoms.get_forces()
    assert calc.raw.n_scc > 1                          # a converged step again
    hot = PyDFTB(refgen.SKF_SETS[DIMER.skf_set], temperature=300.0, xlbomd={"fast": True})
    assert hot.engine_opts["force_evaluation"] == "dynamics"            # T > 0: Dynamics


@pytest.mark.parametrize("name, opts, match", [
    ("water_dimer_hdamp", {}, "needs force_evaluation"),
    ("o2_triplet_wmatrix", {"force_evaluation": "dynamics_t0"}, "spin polarisation"),
    ("h2o_field", {"force_evaluation": "dynamics_t0"}, "external fields"),
    ("h2o_ons", {"force_evaluation": "dynamics_t0"}, "onsite corrections"),
    ("h2o_dftb3_onsite", {"force_evaluation": "dynamics_t0"}, "on-site third order"),
    ("water_dimer_hdamp", {"force_evaluation": "dynamics", "smearing": ("mp", 300.0, 1)},
     "Fermi filling"),
    ("h2co_exc_forces", {"force_evaluation": "dynamics_t0"}, "excited-state forces")])
def test_xl_refusals(name, opts, match):
    """What DFTB+ refuses with XL-BOMD (initprogram), with the reason."""
    _, calc, inp = _engine(CASES_BY_NAME[name], **opts)
    pos, cell, pbc = _geometry(inp)
    with pytest.raises(ValueError, match=match):
        calc.calculate(inp["labels"], pos, cell, pbc, scc_control=ONE_STEP, xl=True)


def test_option_refusals():
    sk = ParamSet(refgen.SKF_SETS[DIMER.skf_set], max_l={"O": 1, "H": 0}).skset(["O", "H"])
    with pytest.raises(ValueError, match="requires the electronic"):
        DFTB(sk, smearing=("fermi", 300.0), force_evaluation="dynamics_t0")
    with pytest.raises(ValueError, match="needs SCC"):
        DFTB(sk, scc=False, force_evaluation="dynamics")
    with pytest.raises(ValueError, match="force_evaluation must be"):
        DFTB(sk, force_evaluation="dynamics_t1")
    with pytest.raises(ValueError, match="5, 6 or 7"):
        XLBOMD(integration_steps=4)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        XLBOMD(fast=True, scale=1.5)
    with pytest.raises(ValueError, match="fast=True"):
        XLBOMD(scale=0.5)
    with pytest.raises(ValueError, match="min_scc"):
        XLBOMD(min_scc=3, max_scc=2)
    with pytest.raises(TypeError):
        XLBOMD.make(5)
    assert XLBOMD.make({"fast": True}).scc == (1, 1, 1e-5)


def test_torch_refuses_dynamics_force_evaluation():
    pytest.importorskip("torch")
    from pydftb.autodiff.api import evaluate
    sk = ParamSet(refgen.SKF_SETS[DIMER.skf_set], max_l={"O": 1, "H": 0}).skset(["O", "H"])
    with pytest.raises(ValueError, match="traditional"):
        DFTB(sk, backend="torch", force_evaluation="dynamics")
    inp = refgen._inputs(DIMER)
    pos, _, _ = _geometry(inp)
    with pytest.raises(NotImplementedError, match="M6e"):
        evaluate(DFTB(sk, force_evaluation="dynamics"), inp["labels"], pos)


def test_casida_needs_no_state_for_xl():
    """Excitation energies along an XL trajectory are fine (no state of interest)."""
    _, calc, inp = _engine(DIMER, force_evaluation="dynamics_t0", casida=Casida(4))
    pos, cell, pbc = _geometry(inp)
    r = calc.calculate(inp["labels"], pos, cell, pbc, scc_control=ONE_STEP, xl=True)
    assert r.excitations is not None and len(r.excitations.energies) == 4
