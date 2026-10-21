"""Self-consistent-charge DFTB driver (DFTB2, DFTB3, collinear spin, dispersion): energies,
forces, stress.

Reference implementation: clarity first, dense linear algebra, NumPy/SciPy.

A calculation runs in four stages, each a separate method so that new
physics (DFTB3, spin, stress, ...) has an obvious place to plug in:

1. ``_setup``     basis, neighbour pairs, H0/S (real space and folded), Coulomb kernel,
                  dispersion (it depends on the geometry only)
2. ``_scc``       self-consistent charge loop (density matrices from a solver)
3. ``_energies``  energy terms, populations, DFTB+-style energy bookkeeping
4. ``_gradient``  analytic forces (band + SCC + repulsive + dispersion)

The matching DFTB+ routines are listed in ``pydftb/docs/dftbplus_map.md``.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from .backend import ops
from .coulomb import Coulomb
from .dispersion import DispersionTerms, make_dispersion
from .hamiltonian import Basis, PairBlocks, build_pairs, fold, fold_all, onsite_energies, shift_index
from .mixer import make_mixer
from .occupations import Smearing
from .neighbors import neighbor_list
from .solvers import SOLVERS, _density
from .skf import SKSet
from .thirdorder import ThirdOrder


@dataclass
class DFTBResult:
    """Results in atomic units (Hartree, Bohr).

    Energy names follow DFTB+'s ``detailed.out``:
    ``energy`` = "Total energy" (E_H0 + E_SCC + E_3rd + E_spin + E_rep + E_disp),
    ``free_energy`` = "Total Mermin free energy" (Total - TS),
    ``e_zero`` = "Extrapolated to 0" (Total - TS/2).
    """
    energy: float                 # total energy E
    free_energy: float            # Mermin free energy E - TS
    e_band: float                 # Tr(rho H0)
    e_scc: float
    e_rep: float
    ts: float
    fermi: float
    charges: np.ndarray           # Mulliken electron populations per atom
    eigenvalues: list | None
    forces: np.ndarray | None = None
    n_scc: int = 0
    converged: bool = True
    timings: dict = field(default_factory=dict)
    q_shell: np.ndarray | None = None    # Mulliken populations per shell
    q_orbital: np.ndarray | None = None  # Mulliken populations per orbital
    occupations: list | None = None      # per k-point, incl. spin factor 2
    scc_error: float = 0.0               # max |q_out - q_in| of the last iteration
    e_3rd: float = 0.0                   # third-order (DFTB3) energy
    stress: np.ndarray | None = None     # (3, 3) Hartree/Bohr^3, DFTB+ sign convention
    kpoints: np.ndarray | None = None    # fractional k-points used
    kweights: np.ndarray | None = None
    # collinear spin: eigenvalues/occupations are stacked [up k-points..., down k-points...]
    e_spin: float = 0.0
    magmoms: np.ndarray | None = None    # (n_atom,) N_up - N_down per atom
    q_spin: np.ndarray | None = None     # (2, n_shell) up / down populations per shell
    fermi_spin: tuple | None = None      # (mu_up, mu_down); equal with relax_total_spin
    e_disp: float = 0.0                  # dispersion energy

    @property
    def e_total(self) -> float:
        return self.energy

    @property
    def e_mermin(self) -> float:
        return self.free_energy

    @property
    def e_zero(self) -> float:
        return self.energy - 0.5 * self.ts

    @property
    def terms(self) -> dict:
        return {"H0": self.e_band, "SCC": self.e_scc, "3rd": self.e_3rd, "spin": self.e_spin,
                "repulsive": self.e_rep, "dispersion": self.e_disp, "-TS": -self.ts}


@dataclass
class _Setup:
    """Everything that depends only on the geometry (fixed during SCC)."""
    species: list
    pos: np.ndarray
    cell: np.ndarray | None
    pbc: np.ndarray | None
    basis: Basis
    pairs: PairBlocks
    kpts: np.ndarray            # (nk, 3) fractional
    kw: np.ndarray              # (nk,) weights, sum 1
    gamma_only: bool
    h0: np.ndarray              # (nk, N, N)
    smat: np.ndarray            # (nk, N, N)
    site_of_orb: np.ndarray
    site_atom: np.ndarray
    q0: np.ndarray              # reference (neutral atom) populations per site
    nelec: float
    coulomb: Coulomb | None
    third: ThirdOrder | None = None
    spin: "_Spin | None" = None
    disp: DispersionTerms | None = None


@dataclass
class _Spin:
    """Collinear spin: W over all shells (block-diagonal per atom) and electron counts."""
    w: np.ndarray               # (n_shell, n_shell)
    n_up: float
    n_down: float
    shared_ef: bool             # one Fermi level for both channels (DFTB+ RelaxTotalSpin)


@dataclass
class _SCCState:
    """Converged (or last) state of the SCC loop.

    ``q_in``/``q_out`` are the mixed vectors: site populations, or with spin, as in
    DFTB+, [up populations per shell, down populations per shell].  With spin the
    density matrices are stacked [up k-points, down k-points].
    """
    q_in: np.ndarray
    q_out: np.ndarray
    q_orb: np.ndarray           # total (up + down) population per orbital
    v_orb: np.ndarray           # charge (SCC + 3rd order) potential per orbital
    h: np.ndarray               # (n_spin * nk, N, N) final Hamiltonian
    rho: np.ndarray             # (n_spin * nk, N, N)
    wdm: np.ndarray | None      # energy-weighted density
    evals: np.ndarray | None
    occ: list | None
    mu: float | tuple
    ts: float
    n_iter: int
    converged: bool
    error: float
    vs_orb: np.ndarray | None = None   # spin potential per orbital (+ for up, - for down)


class DFTB:
    """SCC-DFTB calculator.

    Parameters
    ----------
    sk : SKSet            Slater-Koster data.
    scc : bool            Self-consistent charges.
    kpts : (nk,3), (nk,)  k-points in fractional reciprocal coords and weights
                          (weights normalised to 1).  ``None`` => Gamma.
    temperature : float   Electronic temperature (K) for Fermi smearing
                          (shortcut for ``smearing=("fermi", T)``).
    smearing : spec       ("fermi", T) | ("gaussian", T) | ("mp", T, order) or
                          an :class:`~pydftb.occupations.Smearing`.  T < 1e-8 Ha
                          is raised to 1e-8 Ha like DFTB+.
    mixer : str           "broyden" (DFTB+ default) or "pulay".
    solver : str | object "eigh" (reference), "batched" or "sp2" (see solvers.py).
    repulsive : str       "skf" (spline, else polynomial), "polynomial"
                          (DFTB+ ``PolynomialRepulsive = Yes``) or "none".
    warn_unconverged : bool  warn (but still return results, like DFTB+
                          ``ConvergentSCCOnly = No``) if the SCC does not converge.
    fd_delta : float      None (default): exact H0/S derivatives.  A step (Bohr)
                          reproduces DFTB+'s finite-difference derivatives
                          (current DFTB+ default 1.22e-4; ParserVersion < 5,
                          e.g. via ASE: 1e-2).
    third_order : str     None (DFTB2), "full" (DFTB3, DFTB+ ``ThirdOrderFull``) or
                          "onsite" (DFTB+ ``ThirdOrder``: R = 0 terms only).
    hubbard_derivs : dict type label -> Hubbard derivative Ud (Hartree/e), or a
                          list with one value per shell for orbital-resolved SCC
                          (DFTB+ ``HubbardDerivs``).
    h_damping : float     exponent zeta of the H-X damping of gamma (and Gamma),
                          DFTB+ ``HCorrection = Damping { Exponent = zeta }``;
                          damped are atoms lighter than 3.5 amu, as in DFTB+.
    spin : bool           collinear spin polarisation (DFTB+ ``SpinPolarisation =
                          Colinear``).
    unpaired_electrons : float  N_up - N_down (DFTB+ ``UnpairedElectrons``); fixed,
                          with one Fermi level per spin, unless ``relax_total_spin``.
    relax_total_spin : bool  one Fermi level for both spins (DFTB+ ``RelaxTotalSpin``).
    spin_constants : dict type label -> W (Hartree): one number for all shell pairs,
                          or a (n_shell, n_shell) matrix (DFTB+ ``SpinConstants``).
    dispersion : dict | Dispersion  dispersion correction (DFTB+ ``Dispersion``), e.g.
                          ``{"model": "dftd3", "damping": "beckejohnson", "a1": 0.5719,
                          "a2": 3.6017, "s6": 1.0, "s8": 0.5883}``; see
                          :mod:`pydftb.dispersion`.
    """

    def __init__(self, sk: SKSet, scc: bool = True, kpts=None, temperature: float = 0.0,
                 scc_tol: float = 1e-8, max_scc: int = 200, mixing: float = 0.2,
                 charge: float = 0.0, solver: str = "eigh", verbose: bool = False,
                 repulsive: str = "skf", smearing=None, mixer: str = "broyden",
                 warn_unconverged: bool = True, fd_delta: float | None = None,
                 third_order: str | None = None, hubbard_derivs: dict | None = None,
                 h_damping: float | None = None, spin: bool = False,
                 unpaired_electrons: float = 0.0, relax_total_spin: bool = False,
                 spin_constants: dict | None = None, dispersion=None):
        if spin and not scc:
            raise ValueError("spin polarisation needs SCC")
        if spin and spin_constants is None:
            raise ValueError("spin polarisation needs spin_constants")
        if spin and solver == "sp2":
            raise ValueError("the SP2 solver does not support spin polarisation")
        if third_order not in (None, "full", "onsite"):
            raise ValueError(f"third_order must be None, 'full' or 'onsite', not {third_order!r}")
        if third_order and not scc:
            raise ValueError("third_order needs SCC")
        if third_order and hubbard_derivs is None:
            raise ValueError("third_order needs hubbard_derivs")
        if third_order == "onsite" and sk.shell_resolved:
            raise ValueError("third_order='onsite' needs atom-resolved SCC (as in DFTB+)")
        self.sk = sk
        self.scc = scc
        self.kpts = kpts
        self.smearing = Smearing.from_spec(smearing if smearing is not None else ("fermi", temperature))
        self.scc_tol = scc_tol
        self.max_scc = max_scc
        self.mixing = mixing
        self.mixer = mixer
        self.charge = charge
        self.solver = solver
        self.verbose = verbose
        self.repulsive = repulsive
        self.warn_unconverged = warn_unconverged
        self.fd_delta = fd_delta
        self.third_order = third_order
        self.hubbard_derivs = hubbard_derivs
        self.h_damping = h_damping
        self.spin = spin
        self.unpaired_electrons = unpaired_electrons
        self.relax_total_spin = relax_total_spin
        self.spin_constants = spin_constants
        self.dispersion = make_dispersion(dispersion)

    @property
    def kt(self) -> float:
        return self.smearing.kt

    # ------------------------------------------------------------------
    def calculate(self, species, positions, cell=None, pbc=None, forces=True,
                  q_init=None, stress: bool | None = None) -> DFTBResult:
        """Run a calculation. Positions/cell in Bohr.

        ``stress`` (default: with forces for periodic systems) adds the stress
        tensor sigma = -(1/V) dE/d(strain), the sign convention of DFTB+.
        """
        t0 = time.perf_counter()
        timings = {}
        periodic = cell is not None and (pbc is None or np.any(pbc))
        if stress is None:
            stress = forces and periodic
        if stress and not periodic:
            raise ValueError("stress needs a periodic system")
        need_derivs = forces or stress
        setup = self._setup(species, positions, cell, pbc, need_derivs, timings)
        state = self._scc(setup, q_init, need_derivs, timings)
        res = self._energies(setup, state)
        if need_derivs:
            t1 = time.perf_counter()
            grad, virial = self._gradient(setup, state, virial=stress)
            if forces:
                res.forces = -grad
            if stress:
                vol = abs(np.linalg.det(setup.cell))
                res.stress = -virial / vol
            timings["forces"] = time.perf_counter() - t1
        timings["total"] = time.perf_counter() - t0
        res.timings = timings
        return res

    # ------------------------------------------------------------------
    def _sites(self, basis: Basis):
        """Charge sites: atoms, or shells when orbital-resolved."""
        sk = self.sk
        if sk.shell_resolved:
            site_of_orb = basis.orb_shell
            site_atom = basis.shell_atom
            u = np.array([sk.hubbard[basis.species[a]][l]
                          for a, l in zip(basis.shell_atom, basis.shell_l)])
            q0 = np.array([sk.occupation[basis.species[a]][l]
                           for a, l in zip(basis.shell_atom, basis.shell_l)])
        else:
            site_of_orb = basis.orb_atom
            site_atom = np.arange(len(basis.species))
            u = np.array([sk.hubbard[s][0] for s in basis.species])
            q0 = np.array([sk.occupation[s].sum() for s in basis.species])
        return site_of_orb, site_atom, u, q0

    def _setup(self, species, positions, cell, pbc, forces, timings) -> _Setup:
        t0 = time.perf_counter()
        sk = self.sk
        pos = np.asarray(positions, float)
        if cell is not None:
            cell = np.asarray(cell, float)
            pbc = np.ones(3, bool) if pbc is None else np.asarray(pbc, bool)
        basis = Basis.build(species, sk)
        norb = basis.norb
        pb = build_pairs(species, pos, cell, pbc, sk, basis, derivatives=forces,
                         fd_delta=self.fd_delta)
        eps0 = onsite_energies(basis, sk)
        site_of_orb, site_atom, hub, q0 = self._sites(basis)
        periodic = cell is not None and np.any(pbc)
        if periodic and self.kpts is not None:
            kpts, kw = self.kpts
            kpts = np.atleast_2d(np.asarray(kpts, float))
            kw = np.asarray(kw, float) / np.sum(kw)
        else:
            kpts, kw = np.zeros((1, 3)), np.ones(1)
        gamma_only = len(kpts) == 1 and not np.any(kpts[0])
        if gamma_only:
            h0 = fold(pb, pb.h, norb, None, eps0)[None]
            smat = fold(pb, pb.s, norb, None, np.ones(norb))[None]
        else:
            h0 = fold_all(pb, pb.h, norb, kpts, eps0)
            smat = fold_all(pb, pb.s, norb, kpts, np.ones(norb))
        timings["build"] = time.perf_counter() - t0
        coul = third = spin = None
        if self.scc:
            t1 = time.perf_counter()
            damped = None
            if self.h_damping is not None:
                damped = np.array([sk.mass[lab] < 3.5 for lab in species])   # H, as in DFTB+
            coul = Coulomb(pos, site_atom, hub, cell if periodic else None,
                           pbc if periodic else None, damped=damped,
                           damp_exp=self.h_damping or 0.0)
            if self.third_order:
                third = ThirdOrder(coul, self._site_hubbard_derivs(basis),
                                   onsite_only=self.third_order == "onsite")
            timings["gamma"] = time.perf_counter() - t1
        nelec = q0.sum() - self.charge
        if self.spin:
            spin = _Spin(self._spin_matrix(basis), 0.5 * (nelec + self.unpaired_electrons),
                         0.5 * (nelec - self.unpaired_electrons), self.relax_total_spin)
        disp = None
        if self.dispersion is not None:
            t1 = time.perf_counter()
            disp = self.dispersion.evaluate(species, pos, cell if periodic else None,
                                            pbc if periodic else None, grad=forces,
                                            charge=self.charge)
            timings["dispersion"] = time.perf_counter() - t1
        return _Setup(list(species), pos, cell, pbc, basis, pb, kpts, kw, gamma_only, h0, smat,
                      site_of_orb, site_atom, q0, nelec, coul, third, spin, disp)

    def _site_hubbard_derivs(self, basis: Basis) -> np.ndarray:
        """Ud of every charge site: one value per type, or per shell (orbital-resolved SCC)."""
        missing = sorted(set(basis.species) - set(self.hubbard_derivs))
        if missing:
            raise ValueError(f"hubbard_derivs has no entry for {missing}")
        ud = {lab: np.atleast_1d(np.asarray(v, float)) for lab, v in self.hubbard_derivs.items()}
        if not self.sk.shell_resolved:
            return np.array([ud[lab][0] for lab in basis.species])
        out = []
        for a, l in zip(basis.shell_atom, basis.shell_l):
            v = ud[basis.species[a]]
            out.append(v[0] if len(v) == 1 else v[self.sk.shells(basis.species[a]).index(l)])
        return np.array(out)

    def _spin_matrix(self, basis: Basis) -> np.ndarray:
        """Spin constants over all shells: block-diagonal, one (n_shell, n_shell) block per atom."""
        missing = sorted(set(basis.species) - set(self.spin_constants))
        if missing:
            raise ValueError(f"spin_constants has no entry for {missing}")
        nsh = len(basis.shell_l)
        w = np.zeros((nsh, nsh))
        for a, lab in enumerate(basis.species):
            idx = np.nonzero(basis.shell_atom == a)[0]
            v = np.asarray(self.spin_constants[lab], float)
            if v.size == 1:                       # DFTB+ ShellResolvedSpin = No
                blk = np.full((len(idx), len(idx)), float(v.ravel()[0]))
            elif v.size == len(idx) ** 2:
                blk = v.reshape(len(idx), len(idx))
            else:
                raise ValueError(f"spin_constants[{lab!r}] needs 1 or {len(idx)}x{len(idx)} values")
            w[np.ix_(idx, idx)] = blk
        return w

    # ---- the mixed vector: site populations, or [up, down] per shell with spin ----
    def _initial_populations(self, st: _Setup) -> np.ndarray:
        """DFTB+ start: reference populations rescaled to the electron count; with spin
        every shell also gets m = q * (N_up - N_down) / N (DFTB+ initializeCharges)."""
        scale = st.nelec / st.q0.sum() if st.q0.sum() > 0 else 1.0
        if st.spin is None:
            return st.q0 * scale
        b = st.basis
        q = np.array([self.sk.occupation[b.species[a]][l]
                      for a, l in zip(b.shell_atom, b.shell_l)]) * scale
        m = q * (st.spin.n_up - st.spin.n_down) / st.nelec
        return np.concatenate([0.5 * (q + m), 0.5 * (q - m)])

    def _charge_and_magnetisation(self, st: _Setup, qvec):
        """Site population deviations dq and shell magnetisations m (None without spin)."""
        if st.spin is None:
            return qvec - st.q0, None
        nsh = len(st.basis.shell_l)
        up, down = qvec[:nsh], qvec[nsh:]
        q = up + down
        if not self.sk.shell_resolved:
            q = ops.segment_sum(q, st.basis.shell_atom, len(st.species))
        return q - st.q0, up - down

    def _potentials(self, st: _Setup, qvec):
        """Charge potential (SCC + third order) and spin potential, per orbital."""
        if not self.scc:
            return np.zeros(st.basis.norb), None
        dq, m = self._charge_and_magnetisation(st, qvec)
        _, v_site = st.coulomb.energy_shift(dq)
        if st.third is not None:
            v_site = v_site + st.third.energy_shift(dq)[1]
        vs_orb = None if m is None else (st.spin.w @ m)[st.basis.orb_shell]
        return v_site[st.site_of_orb], vs_orb

    def _solve(self, solver, st: _Setup, h, need_w):
        """Densities for H, stacked [up k-points, down k-points] with spin; one Fermi
        level per spin channel, or a shared one (DFTB+ RelaxTotalSpin)."""
        if st.spin is None:
            rho, wdm, evals, mu, ts = solver.solve(h, st.kw, st.nelec, self.smearing, need_w=need_w)
            return rho, wdm, evals, mu, ts, getattr(solver, "last_occupations", None)
        e, c = solver.eig(h)
        nk = len(st.kw)
        if st.spin.shared_ef:
            rho, wdm, evals, mu, ts, occ = _density(e, c, np.tile(st.kw, 2), st.nelec,
                                                    self.smearing, need_w, spin_factor=1.0)
            return rho, wdm, evals, (mu, mu), ts, list(occ)
        parts = [_density(e[s], c[s], st.kw, n, self.smearing, need_w, spin_factor=1.0)
                 for s, n in ((slice(0, nk), st.spin.n_up), (slice(nk, 2 * nk), st.spin.n_down))]
        rho = np.concatenate([p[0] for p in parts])
        wdm = np.concatenate([p[1] for p in parts]) if need_w else None
        evals = np.concatenate([p[2] for p in parts])
        occ = list(parts[0][5]) + list(parts[1][5])
        return rho, wdm, evals, (parts[0][3], parts[1][3]), parts[0][4] + parts[1][4], occ

    def _populations(self, st: _Setup, rho):
        """Mulliken populations per orbital (up + down) and the vector to mix."""
        nk = len(st.kw)
        # q_mu = sum_k w_k Re (rho_k S_k)_{mu mu}
        if st.spin is None:
            q_orb = np.einsum("k,kij,kji->i", st.kw, rho, st.smat).real
            return q_orb, ops.segment_sum(q_orb, st.site_of_orb, len(st.q0))
        q_up = np.einsum("k,kij,kji->i", st.kw, rho[:nk], st.smat).real
        q_dn = np.einsum("k,kij,kji->i", st.kw, rho[nk:], st.smat).real
        shells, nsh = st.basis.orb_shell, len(st.basis.shell_l)
        return q_up + q_dn, np.concatenate([ops.segment_sum(q_up, shells, nsh),
                                            ops.segment_sum(q_dn, shells, nsh)])

    def _scc(self, st: _Setup, q_init, forces, timings) -> _SCCState:
        solver = self.solver if not isinstance(self.solver, str) else SOLVERS[self.solver]()
        t1 = time.perf_counter()
        solver.prepare(st.smat if st.spin is None else np.concatenate([st.smat, st.smat]))
        t_diag = time.perf_counter() - t1
        mixer = make_mixer(self.mixer, self.mixing)
        q_in = (np.asarray(q_init, float).copy() if q_init is not None
                else self._initial_populations(st))
        converged = not self.scc
        nit = 0
        diff = 0.0
        for nit in range(1, (self.max_scc if self.scc else 1) + 1):
            v_orb, vs_orb = self._potentials(st, q_in)
            h = st.h0 + 0.5 * st.smat * (v_orb[:, None] + v_orb[None, :])
            if vs_orb is not None:
                hs = 0.5 * st.smat * (vs_orb[:, None] + vs_orb[None, :])
                h = np.concatenate([h + hs, h - hs])          # up, down
            t1 = time.perf_counter()
            rho, wdm, evals, mu, ts, occ = self._solve(solver, st, h, need_w=False)
            t_diag += time.perf_counter() - t1
            q_orb, q_out = self._populations(st, rho)
            if not self.scc:
                break
            diff = np.abs(q_out - q_in).max()
            if self.verbose:
                print(f"SCC {nit:3d}  max dq = {diff:.3e}")
            if diff < self.scc_tol:
                converged = True
                break
            if nit == 1 and st.spin is not None and q_init is None:
                q_in = q_out            # DFTB+: the guessed spin is not mixed in (main.F90)
            else:
                q_in = mixer.mix(q_in, q_out)
        if forces:
            # W is only needed once, for the final density
            t1 = time.perf_counter()
            rho, wdm, evals, mu, ts, occ = self._solve(solver, st, h, need_w=True)
            t_diag += time.perf_counter() - t1
        timings["diag"] = t_diag
        timings["scc_iter"] = nit
        if self.scc and not converged and self.warn_unconverged:
            warnings.warn(f"SCC not converged after {nit} iterations "
                          f"(max |dq| = {diff:.2e} > {self.scc_tol:.1e}); returning the last "
                          "iteration's results", RuntimeWarning, stacklevel=3)
        return _SCCState(q_in, q_out, q_orb, v_orb, h, rho, wdm, evals, occ, mu, ts, nit,
                         converged, diff, vs_orb)

    def _energies(self, st: _Setup, sc: _SCCState) -> DFTBResult:
        nk = len(st.kw)
        e_band = sum(float(np.einsum("k,kij,kji->", st.kw, sc.rho[s:s + nk], st.h0).real)
                     for s in range(0, len(sc.rho), nk))       # Tr(rho H0), summed over spins
        dq, m = self._charge_and_magnetisation(st, sc.q_out)
        e_scc = st.coulomb.energy_shift(dq)[0] if self.scc else 0.0
        e_3rd = st.third.energy_shift(dq)[0] if st.third is not None else 0.0
        e_spin = 0.5 * m @ st.spin.w @ m if m is not None else 0.0
        e_rep = self._repulsive_energy(st)
        e_disp = st.disp.energy if st.disp is not None else 0.0
        energy = e_band + e_scc + e_3rd + e_spin + e_rep + e_disp
        q_atom = ops.segment_sum(sc.q_orb, st.basis.orb_atom, len(st.species))
        q_shell = ops.segment_sum(sc.q_orb, st.basis.orb_shell, len(st.basis.shell_l))
        occ = None if sc.occ is None else [np.asarray(o) for o in sc.occ]
        mu, extra = sc.mu, {}
        if st.spin is not None:
            mu = sc.mu[0]
            extra = dict(e_spin=e_spin, q_spin=sc.q_out.reshape(2, -1), fermi_spin=tuple(sc.mu),
                         magmoms=ops.segment_sum(m, st.basis.shell_atom, len(st.species)))
        return DFTBResult(energy, energy - sc.ts, e_band, e_scc, e_rep, sc.ts, mu, q_atom,
                          None if sc.evals is None else list(sc.evals), n_scc=sc.n_iter,
                          converged=sc.converged, q_shell=q_shell, q_orbital=sc.q_orb,
                          occupations=occ, scc_error=sc.error, kpoints=st.kpts,
                          kweights=st.kw, e_3rd=e_3rd, e_disp=e_disp, **extra)

    def _gradient(self, st: _Setup, sc: _SCCState, virial: bool = False):
        """dE/dR (n_atom, 3) and, if requested, the virial dE/d(strain) (3, 3).

        Every pair term contributes (dE/d r_pair) (x) r_pair to the virial.
        """
        pb = st.pairs
        grad, w_rep = self._repulsive_gradient(st)
        nk = len(st.kw)
        nspin = len(sc.rho) // nk
        kw = np.tile(st.kw, nspin)                        # weights of the stacked channels
        sgn = np.repeat([1.0, -1.0][:nspin], nk)          # up - down: magnetisation density
        # real-space (energy-weighted) density at the pair elements, summed over spins:
        # P_e = Re sum_k w_k rho_k[nu, mu] exp(2 pi i k.T)
        p_m = None
        if st.gamma_only:
            rho_e = sc.rho[:, pb.cols, pb.rows].real
            p_e = kw @ rho_e
            w_e = kw @ sc.wdm[:, pb.cols, pb.rows].real
            if nspin == 2:
                p_m = (kw * sgn) @ rho_e
        else:
            # group by translation T: R_T = sum_k w_k e^{2 pi i k.T} rho_k  (nT GEMM)
            shifts, t_of = shift_index(pb)
            ph = (np.exp(2j * np.pi * (np.tile(st.kpts, (nspin, 1)) @ shifts.T)) * kw[:, None]).T
            norb = st.basis.norb
            n2 = norb * norb
            rho2 = sc.rho.reshape(len(kw), n2)
            r_t = (ph @ rho2).reshape(-1, norb, norb)
            w_t = (ph @ sc.wdm.reshape(len(kw), n2)).reshape(-1, norb, norb)
            p_e = r_t[t_of, pb.cols, pb.rows].real
            w_e = w_t[t_of, pb.cols, pb.rows].real
            if nspin == 2:
                p_m = ((ph * sgn[None, :]) @ rho2).reshape(-1, norb, norb)[t_of, pb.cols, pb.rows].real
        vsum = 0.5 * (sc.v_orb[pb.rows] + sc.v_orb[pb.cols])
        coef_s = p_e * vsum - w_e
        if p_m is not None:
            coef_s = coef_s + p_m * 0.5 * (sc.vs_orb[pb.rows] + sc.vs_orb[pb.cols])
        ge = p_e[:, None] * pb.dh + coef_s[:, None] * pb.ds   # dE/dvec per element
        gp = ops.segment_sum(ge, pb.pair_of, len(pb.i))          # per pair
        natom = len(st.pos)
        grad = grad + ops.segment_sum(gp, pb.j, natom) - ops.segment_sum(gp, pb.i, natom)
        dq_in, _ = self._charge_and_magnetisation(st, sc.q_in)
        if self.scc:
            g_scc, w_scc = st.coulomb.gradient(dq_in, virial=virial)
            grad = grad + g_scc                                    # already per atom
        if st.third is not None:
            g_3rd, w_3rd = st.third.gradient(dq_in, virial=virial)
            grad = grad + g_3rd
        if st.disp is not None:
            grad = grad + st.disp.gradient
        if not virial:
            return grad, None
        w = gp.T @ pb.vec + w_rep                                  # band + SCC shift + repulsive
        if self.scc:
            w = w + w_scc
        if st.third is not None:
            w = w + w_3rd
        if st.disp is not None:
            w = w + st.disp.virial
        return grad, 0.5 * (w + w.T)

    # ------------------------------------------------------------------
    def _repulsive_pairs(self, st: _Setup):
        """Neighbour list and the (mask, Repulsive) of every label pair present."""
        labels = sorted(set(st.species))
        rc = max((r.cutoff for a in labels for b in labels
                  if (r := self.sk.repulsive(a, b, self.repulsive)) is not None), default=0.0)
        if rc == 0.0:
            return None, []
        nl = neighbor_list(st.pos, st.cell, st.pbc, rc)
        i, j = nl[0], nl[1]
        spec = np.array(st.species)
        out = []
        for a in labels:
            for b in labels:
                rep = self.sk.repulsive(a, b, self.repulsive)
                m = (spec[i] == a) & (spec[j] == b)
                if rep is not None and m.any():
                    out.append((m, rep))
        return nl, out

    def _repulsive_energy(self, st: _Setup) -> float:
        nl, terms = self._repulsive_pairs(st)
        e_tot = 0.0
        for m, rep in terms:
            e, _ = rep(nl[4][m])
            e_tot += 0.5 * e.sum()       # ordered pairs: every bond counted twice
        return e_tot

    def _repulsive_gradient(self, st: _Setup):
        """Repulsive dE/dR (n_atom, 3) and virial (3, 3)."""
        grad = np.zeros_like(st.pos)
        w = np.zeros((3, 3))
        nl, terms = self._repulsive_pairs(st)
        for m, rep in terms:
            i, j, _, vec, d = (x[m] for x in nl)
            _, de = rep(d)
            g = (0.5 * de / d)[:, None] * vec          # dE/d(vec) of each ordered pair
            grad = ops.scatter_add(grad, i, -g)
            grad = ops.scatter_add(grad, j, g)
            w += g.T @ vec
        return grad, w
