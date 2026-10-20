"""Self-consistent-charge DFTB (DFTB2) driver: energies, charges, forces.

Reference implementation: clarity first, dense linear algebra, NumPy/SciPy.

A calculation runs in four stages, each a separate method so that new
physics (DFTB3, spin, stress, ...) has an obvious place to plug in:

1. ``_setup``     basis, neighbour pairs, H0/S (real space and folded), Coulomb kernel
2. ``_scc``       self-consistent charge loop (density matrices from a solver)
3. ``_energies``  energy terms, populations, DFTB+-style energy bookkeeping
4. ``_gradient``  analytic forces (band + SCC + repulsive)

The matching DFTB+ routines are listed in ``pydftb/docs/dftbplus_map.md``.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from .backend import ops
from .coulomb import Coulomb
from .hamiltonian import Basis, PairBlocks, build_pairs, fold, fold_all, onsite_energies, shift_index
from .mixer import make_mixer
from .occupations import Smearing
from .neighbors import neighbor_list
from .solvers import SOLVERS
from .skf import SKSet


@dataclass
class DFTBResult:
    """Results in atomic units (Hartree, Bohr).

    Energy names follow DFTB+'s ``detailed.out``:
    ``energy`` = "Total energy" (E_H0 + E_SCC + E_rep),
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
    stress: np.ndarray | None = None     # (3, 3) Hartree/Bohr^3, DFTB+ sign convention
    kpoints: np.ndarray | None = None    # fractional k-points used
    kweights: np.ndarray | None = None

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
        return {"H0": self.e_band, "SCC": self.e_scc, "repulsive": self.e_rep, "-TS": -self.ts}


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


@dataclass
class _SCCState:
    """Converged (or last) state of the SCC loop."""
    q_in: np.ndarray
    q_out: np.ndarray
    q_orb: np.ndarray
    v_orb: np.ndarray           # SCC potential per orbital
    h: np.ndarray               # (nk, N, N) final Hamiltonian
    rho: np.ndarray             # (nk, N, N)
    wdm: np.ndarray | None      # energy-weighted density
    evals: np.ndarray | None
    occ: list | None
    mu: float
    ts: float
    n_iter: int
    converged: bool
    error: float


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
    """

    def __init__(self, sk: SKSet, scc: bool = True, kpts=None, temperature: float = 0.0,
                 scc_tol: float = 1e-8, max_scc: int = 200, mixing: float = 0.2,
                 charge: float = 0.0, solver: str = "eigh", verbose: bool = False,
                 repulsive: str = "skf", smearing=None, mixer: str = "broyden",
                 warn_unconverged: bool = True, fd_delta: float | None = None):
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
        coul = None
        if self.scc:
            t1 = time.perf_counter()
            coul = Coulomb(pos, site_atom, hub, cell if periodic else None,
                           pbc if periodic else None)
            timings["gamma"] = time.perf_counter() - t1
        return _Setup(list(species), pos, cell, pbc, basis, pb, kpts, kw, gamma_only, h0, smat,
                      site_of_orb, site_atom, q0, q0.sum() - self.charge, coul)

    def _scc(self, st: _Setup, q_init, forces, timings) -> _SCCState:
        solver = self.solver if not isinstance(self.solver, str) else SOLVERS[self.solver]()
        t1 = time.perf_counter()
        solver.prepare(st.smat)
        t_diag = time.perf_counter() - t1
        mixer = make_mixer(self.mixer, self.mixing)
        q0 = st.q0
        if q_init is not None:
            q_in = np.asarray(q_init, float).copy()
        else:
            # DFTB+: neutral reference populations rescaled to the electron count
            q_in = q0 * (st.nelec / q0.sum()) if q0.sum() > 0 else q0.copy()
        converged = not self.scc
        nit = 0
        diff = 0.0
        for nit in range(1, (self.max_scc if self.scc else 1) + 1):
            if self.scc:
                _, v_site = st.coulomb.energy_shift(q_in - q0)
            else:
                v_site = np.zeros(len(q0))
            v_orb = v_site[st.site_of_orb]
            h = st.h0 + 0.5 * st.smat * (v_orb[:, None] + v_orb[None, :])
            t1 = time.perf_counter()
            rho, wdm, evals, mu, ts = solver.solve(h, st.kw, st.nelec, self.smearing, need_w=False)
            t_diag += time.perf_counter() - t1
            # Mulliken: q_mu = sum_k w_k Re (rho_k S_k)_{mu mu}
            q_orb = np.einsum("k,kij,kji->i", st.kw, rho, st.smat).real
            q_out = ops.segment_sum(q_orb, st.site_of_orb, len(q0))
            if not self.scc:
                break
            diff = np.abs(q_out - q_in).max()
            if self.verbose:
                print(f"SCC {nit:3d}  max dq = {diff:.3e}")
            if diff < self.scc_tol:
                converged = True
                break
            q_in = mixer.mix(q_in, q_out)
        if forces:
            # W is only needed once, for the final density
            t1 = time.perf_counter()
            rho, wdm, evals, mu, ts = solver.solve(h, st.kw, st.nelec, self.smearing, need_w=True)
            t_diag += time.perf_counter() - t1
        timings["diag"] = t_diag
        timings["scc_iter"] = nit
        if self.scc and not converged and self.warn_unconverged:
            warnings.warn(f"SCC not converged after {nit} iterations "
                          f"(max |dq| = {diff:.2e} > {self.scc_tol:.1e}); returning the last "
                          "iteration's results", RuntimeWarning, stacklevel=3)
        occ = getattr(solver, "last_occupations", None)
        return _SCCState(q_in, q_out, q_orb, v_orb, h, rho, wdm, evals, occ, mu, ts, nit,
                         converged, diff)

    def _energies(self, st: _Setup, sc: _SCCState) -> DFTBResult:
        e_band = float(np.einsum("k,kij,kji->", st.kw, sc.rho, st.h0).real)   # Tr(rho H0)
        dq = sc.q_out - st.q0
        e_scc = st.coulomb.energy_shift(dq)[0] if self.scc else 0.0
        e_rep = self._repulsive_energy(st)
        energy = e_band + e_scc + e_rep
        q_atom = ops.segment_sum(sc.q_out, st.site_atom, len(st.species))
        q_shell = ops.segment_sum(sc.q_orb, st.basis.orb_shell, len(st.basis.shell_l))
        occ = None if sc.occ is None else [np.asarray(o) for o in sc.occ]
        return DFTBResult(energy, energy - sc.ts, e_band, e_scc, e_rep, sc.ts, sc.mu, q_atom,
                          None if sc.evals is None else list(sc.evals), n_scc=sc.n_iter,
                          converged=sc.converged, q_shell=q_shell, q_orbital=sc.q_orb,
                          occupations=occ, scc_error=sc.error, kpoints=st.kpts,
                          kweights=st.kw)

    def _gradient(self, st: _Setup, sc: _SCCState, virial: bool = False):
        """dE/dR (n_atom, 3) and, if requested, the virial dE/d(strain) (3, 3).

        Every pair term contributes (dE/d r_pair) (x) r_pair to the virial.
        """
        pb = st.pairs
        grad, w_rep = self._repulsive_gradient(st)
        # real-space (energy-weighted) density at the pair elements:
        # P_e = Re sum_k w_k rho_k[nu, mu] exp(2 pi i k.T)
        if st.gamma_only:
            p_e = sc.rho[0][pb.cols, pb.rows].real
            w_e = sc.wdm[0][pb.cols, pb.rows].real
        else:
            # group by translation T: R_T = sum_k w_k e^{2 pi i k.T} rho_k  (nT GEMM)
            shifts, t_of = shift_index(pb)
            ph = (np.exp(2j * np.pi * (st.kpts @ shifts.T)) * st.kw[:, None]).T   # (nT, nk)
            norb = st.basis.norb
            n2 = norb * norb
            r_t = (ph @ sc.rho.reshape(len(st.kw), n2)).reshape(-1, norb, norb)
            w_t = (ph @ sc.wdm.reshape(len(st.kw), n2)).reshape(-1, norb, norb)
            p_e = r_t[t_of, pb.cols, pb.rows].real
            w_e = w_t[t_of, pb.cols, pb.rows].real
        vsum = 0.5 * (sc.v_orb[pb.rows] + sc.v_orb[pb.cols])
        coef_s = p_e * vsum - w_e
        ge = p_e[:, None] * pb.dh + coef_s[:, None] * pb.ds   # dE/dvec per element
        gp = ops.segment_sum(ge, pb.pair_of, len(pb.i))          # per pair
        natom = len(st.pos)
        grad = grad + ops.segment_sum(gp, pb.j, natom) - ops.segment_sum(gp, pb.i, natom)
        if self.scc:
            g_scc, w_scc = st.coulomb.gradient(sc.q_in - st.q0, virial=virial)
            grad = grad + g_scc                                    # already per atom
        if not virial:
            return grad, None
        w = gp.T @ pb.vec + w_rep                                  # band + SCC shift + repulsive
        if self.scc:
            w = w + w_scc
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
