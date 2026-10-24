"""Linear-response TD-DFTB excitations (Casida equation, milestone M6d).

The model is DFTB+ 25.1's ``ExcitedState { Casida }`` (``timedep/linresp.F90``,
``linrespgrad.F90``, ``linrespcommon.F90``, ``transcharges.F90``; Niehaus et al.,
Phys. Rev. B 63, 085108 (2001)): vertical excitation energies, oscillator strengths and
transition dipoles of molecules (and Gamma-point cells) from the converged ground state.

Notation: i, j occupied and a, b virtual orbitals (per spin channel), w_ia = e_a - e_i,
n_ia = f_i - f_a (halved without spin polarisation, the occupations then being 0..2) and
q^ia_A the Mulliken transition charges.  The hermitian Casida matrix

    Omega_ia,jb = delta w_ia^2 + c sqrt(n_ia w_ia) K_ia,jb sqrt(n_jb w_jb)

has c = 4 (closed shell) or 2 (collinear spin polarisation) and K = q gamma q (singlets),
q W q with the atomic spin constants W on the diagonal (triplets), or q gamma q + s s' q W q
(spin polarised, s = +-1); onsite corrections add a term with the transition densities of
each atom's orbital block (DFTB+ ``onsiteEner``).  Its eigenvalues are the squared
excitation energies Omega_I^2; X + Y = F sqrt(w / Omega_I) for the normalised eigenvectors F.

Energies are in Hartree and dipoles in e Bohr.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

from .constants import AU_FS, BOHR, ELEC_TOL_MAX, HARTREE

# DFTB+ common/constants.F90: Debye__au = 1e-16 alpha_fs au__fs Coulomb__au AA__Bohr^2
AU_DEBYE = 1.0 / (1e-16 * 0.007297352568 * AU_FS / 1.60217653e-19 / BOHR ** 2)
EPS_SINGLE = float(np.finfo(np.float32).eps)
SYMMETRIES = {"singlet": "S", "triplet": "T", "both": "B"}
SOLVERS = ("auto", "exact", "davidson", "arpack")
EXACT_LIMIT = 500           # "auto": dense Omega up to this many transitions (Davidson beyond)
DAVIDSON_TOL = 1e-12        # residual norm |Omega x - Omega^2 x| (Hartree^2) of every state


@dataclass
class Casida:
    """Options of DFTB+'s ``ExcitedState { Casida }``.

    ``symmetry`` ("singlet", "triplet" or "both") is ignored with spin polarisation, where
    the excitations of both spin channels are coupled.  ``energy_window`` (Hartree) keeps the
    single-particle transitions up to that much above the ``n_excitations``-th one;
    ``oscillator_window`` then adds higher, bright ones (degenerate groups whose summed
    single-particle oscillator strength exceeds it), as DFTB+'s ``EnergyWindow`` and
    ``OscillatorWindow``.  ``state_of_interest`` > 0 reports that state's excitation energy
    (``DFTBResult.excitation_energy``), < 0 the brightest singlet's; forces are then those of
    the excited state, -d(E_0 + Omega)/dR (milestone M6e, :mod:`pydftb.casida_grad`), unless
    ``excited_forces=False`` asks for the ground-state ones (DFTB+ ``ExcitedStateForces =
    No``).  ``excited_charges`` gives the excited state's Mulliken charges and dipole also
    without forces (DFTB+ ``WriteMulliken``); ``transition_charges`` the state of interest's
    atomic transition charges (DFTB+ ``WriteTransitionCharges``, milestone M6f-4b; singlets or
    spin polarisation).  ``solver``: "exact" (dense Omega), "davidson"
    (matrix-free block Davidson with the diagonal w^2 as preconditioner: Omega is that diagonal
    plus a coupling of rank n_atom, so few iterations), "arpack" (matrix-free, SciPy's ARPACK;
    slower, an independent check) or "auto" (exact up to :data:`EXACT_LIMIT` transitions, else
    Davidson).
    """
    n_excitations: int
    symmetry: str = "singlet"
    state_of_interest: int = 0
    energy_window: float | None = None
    oscillator_window: float | None = None
    solver: str = "auto"
    excited_forces: bool = True
    excited_charges: bool = False
    transition_charges: bool = False

    def __post_init__(self):
        if int(self.n_excitations) < 1:
            raise ValueError("n_excitations must be at least 1")
        self.n_excitations = int(self.n_excitations)
        if self.symmetry not in SYMMETRIES:
            raise ValueError(f"symmetry must be one of {sorted(SYMMETRIES)}, got {self.symmetry!r}")
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {self.solver!r}")
        if self.transition_charges and self.symmetry != "singlet":
            raise ValueError("transition charges are not written for triplets (as DFTB+)")
        if self.transition_charges and self.state_of_interest == 0:
            raise ValueError("transition charges need a state of interest "
                             "(state_of_interest != 0; as DFTB+)")
        if self.state_of_interest < 0 and self.symmetry != "singlet":
            raise ValueError("the brightest state (state_of_interest < 0) is only for singlets "
                             "(as in DFTB+)")
        if self.state_of_interest != 0 and self.symmetry == "both":
            raise ValueError("a state of interest needs one symmetry, not 'both' (as in DFTB+)")
        if self.state_of_interest > self.n_excitations:
            raise ValueError("state_of_interest is beyond n_excitations")
        for name in ("energy_window", "oscillator_window"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"{name} must be positive (as in DFTB+)")

    @classmethod
    def make(cls, spec) -> "Casida | None":
        if spec is None or isinstance(spec, cls):
            return spec
        if isinstance(spec, dict):
            return cls(**spec)
        raise TypeError("casida must be a Casida or a dict of its options")


@dataclass
class SingleParticle:
    """The single-particle transitions that take part (after the windows; DFTB+ SPX.DAT),
    ordered by energy; the oscillator strengths are 2/3 w_ia |d_ia|^2."""
    energies: np.ndarray            # w_ia, Hartree
    oscillator_strengths: np.ndarray
    pairs: np.ndarray               # (n, 3) occupied, virtual (0-based, per spin channel), spin
    spin_polarised: bool = False

    def table(self) -> str:
        """The transitions as DFTB+ writes them to SPX.DAT (energies in eV)."""
        lines = ["", "       #      w [eV]       Osc.Str.        Transition", "",
                 " " + "=" * 58, ""]
        for n, (w, f, (i, a, s)) in enumerate(zip(self.energies, self.oscillator_strengths,
                                                  self.pairs), 1):
            sign = ("U" if s == 0 else "D") if self.spin_polarised else " "
            lines.append(f" {n:7d}   {w * HARTREE:8.3f}   {f:13.7f}    {i + 1:5d}   -> "
                         f"{a + 1:5d} {sign}")
        return "\n".join(lines + [""]) + "\n"


@dataclass
class Excitations:
    """Result of a Casida calculation, one row per excited state.

    ``energies`` are sqrt(Omega^2) (NaN where Omega^2 < 0, an unstable ground state);
    ``transitions`` are the dominant single-particle transitions (occupied, virtual, spin;
    0-based) with their ``weights`` (DFTB+: (X+Y)^2 / ||(X+Y)^2||_2) and energies
    ``ks_energies``; ``xpy`` and ``xmy`` hold X + Y and X - Y over the transitions ``pairs``
    (after the windows).
    ``symmetries`` are "S" / "T" per state, None with spin polarisation, where
    ``spin_contamination`` is DFTB+'s D<S^2>.
    """
    omega2: np.ndarray
    oscillator_strengths: np.ndarray
    transition_dipoles: np.ndarray
    symmetries: list | None
    transitions: np.ndarray
    weights: np.ndarray
    ks_energies: np.ndarray
    xpy: np.ndarray
    pairs: np.ndarray
    single_particle: SingleParticle
    spin_contamination: np.ndarray | None = None
    state: int = 0                  # state of interest (1-based, as DFTB+), 0 for none
    xmy: np.ndarray | None = None   # X - Y = F sqrt(Omega / w)
    transition_charges: np.ndarray | None = None   # (n_atom,) of the state of interest (M6f-4b)

    @property
    def energies(self) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.where(self.omega2 >= 0, np.sqrt(np.abs(self.omega2)), np.nan)

    def degenerate_groups(self, tol: float = ELEC_TOL_MAX) -> dict:
        """DFTB+'s tagged output (``exc_energies_sqr``, ``exc_oscillator``, ``exc_transdip``):
        consecutive states whose Omega^2 differ by at most ``tol`` merge into one, with the
        oscillator strengths summed and the transition dipoles as sqrt(sum |mu|^2)."""
        groups, start = [], 0
        w2 = self.omega2
        for k in range(1, len(w2) + 1):
            if k == len(w2) or abs(w2[k] - w2[k - 1]) > tol:
                groups.append((start, k))
                start = k
        return {"omega2": np.array([w2[a] for a, _ in groups]),
                "oscillator": np.array([self.oscillator_strengths[a:b].sum() for a, b in groups]),
                "transdip": np.array([math.sqrt((self.transition_dipoles[a:b] ** 2).sum())
                                      for a, b in groups]),
                "groups": groups}

    def table(self, header: bool = True) -> str:
        """The excitations as DFTB+ writes them to EXC.DAT (energies in eV, DFTB+'s column
        layout).  For symmetry "both" DFTB+ writes one header, then the triplets and the
        singlets: ``xs[0].table() + xs[1].table(header=False)``."""
        spin = self.symmetries is None
        last = "D<S*S>" if spin else "Sym."
        lines = ["", f"     w [eV]       Osc.Str.         Transition         Weight      KS [eV]"
                     f"    {last}", "", " " + "=" * 80, ""] if header else []
        for n in range(len(self.omega2)):
            i, a, _ = self.transitions[n]
            stable = self.omega2[n] > 0
            w = f" {math.sqrt(self.omega2[n]) * HARTREE:10.3f}" if stable else "      < 0  "
            gap = "     " if stable and not spin else "  "
            weight = f"{self.weights[n]:6.3f}" if stable or not spin else "-"
            tail = (f"    {self.spin_contamination[n]:6.3f}" if spin
                    else f"      {self.symmetries[n]}")
            lines.append(f"{w}    {self.oscillator_strengths[n]:14.8f}{gap}{i + 1:5d}   -> "
                         f"{a + 1:5d}       {weight}  {self.ks_energies[n] * HARTREE:10.3f}{tail}")
        return "\n".join(lines) + "\n"


# ---- the ground-state data ------------------------------------------------------------------
@dataclass
class GroundState:
    """What the response needs from the converged ground state, per spin channel."""
    eps: list                       # (N,) orbital energies per channel
    coeffs: list                    # (N, N) S-orthonormal eigenvectors per channel
    occupations: list               # (N,) occupations per channel (0..2 closed shell, 0..1 spin)
    overlap: np.ndarray             # (N, N)
    orb_atom: np.ndarray            # (N,) atom of every orbital
    positions: np.ndarray           # (n_atom, 3) Bohr
    gamma: np.ndarray               # (n_atom, n_atom)
    spin_w: np.ndarray | None       # (n_atom,) spin constant of each atom's HOAO shell
    onsite: list | None = None      # per atom: (f_uu, f_ud) orbital matrices
    shell_ranges: list | None = None   # per atom: local (start, stop) of every shell
    sc: list = field(default_factory=list)   # S C per channel (filled in by prepare)

    @property
    def spin_polarised(self) -> bool:
        return len(self.coeffs) == 2

    def prepare(self):
        self.sc = [self.overlap @ c for c in self.coeffs]
        return self


def hoao_shell(onsite_energies, occupations) -> int:
    """Index of the highest occupied atomic shell: the largest on-site energy among the shells
    with a non-zero reference occupation (DFTB+ initprogram: HOAO for the response U and W)."""
    e = np.asarray(onsite_energies, float)
    occ = np.asarray(occupations, float) > 0
    if not occ.any():
        raise ValueError("an atom without occupied shells has no HOAO")
    return int(np.flatnonzero(occ)[np.argmax(e[occ])])


# ---- single-particle transitions ------------------------------------------------------------
def single_particle_transitions(gs: GroundState):
    """All (i, a, spin) with f_i > f_a + tol (a >= i), DFTB+'s getSPExcitations order, and
    their energies w_ia and sqrt(n_ia)."""
    pairs, w, sqrt_n = [], [], []
    for s, (e, f) in enumerate(zip(gs.eps, gs.occupations)):
        n = len(e)
        ii, aa = np.triu_indices(n)
        keep = f[ii] > f[aa] + ELEC_TOL_MAX
        ii, aa = ii[keep], aa[keep]
        pairs.append(np.stack([ii, aa, np.full(len(ii), s)], axis=1))
        w.append(e[aa] - e[ii])
        docc = f[ii] - f[aa]
        sqrt_n.append(np.sqrt(docc if gs.spin_polarised else 0.5 * docc))
    return np.concatenate(pairs), np.concatenate(w), np.concatenate(sqrt_n)


def transition_charges(gs: GroundState, pairs, chunk: int = 4096) -> np.ndarray:
    """q^ia_A = 1/2 sum_{mu in A} (c_mu,i (S c_a)_mu + c_mu,a (S c_i)_mu), (n_pair, n_atom)."""
    natom = len(gs.positions)
    out = np.empty((len(pairs), natom))
    for start in range(0, len(pairs), chunk):
        p = pairs[start:start + chunk]
        for s in np.unique(p[:, 2]):
            sel = np.flatnonzero(p[:, 2] == s)
            c, sc = gs.coeffs[s], gs.sc[s]
            i, a = p[sel, 0], p[sel, 1]
            dens = 0.5 * (c[:, i] * sc[:, a] + c[:, a] * sc[:, i])          # (N, n)
            q = np.zeros((natom, len(sel)))
            np.add.at(q, gs.orb_atom, dens)
            out[start + sel] = q.T
    return out


def _orbital_slices(orb_atom, natom):
    first = np.searchsorted(orb_atom, np.arange(natom))
    last = np.searchsorted(orb_atom, np.arange(natom), side="right")
    return [slice(int(a), int(b)) for a, b in zip(first, last)]


def _onsite_blocks(gs: GroundState, pairs):
    """Per atom: the transition densities 1/4 (c_mu,i (Sc_a)_nu + c_mu,a (Sc_i)_nu + (mu <->
    nu)) of every pair over the atom's orbital block, (n_pair, n_orb_A^2)."""
    out = []
    for sl in _orbital_slices(gs.orb_atom, len(gs.positions)):
        n = sl.stop - sl.start
        blk = np.zeros((len(pairs), n, n))
        for s in np.unique(pairs[:, 2]):
            sel = np.flatnonzero(pairs[:, 2] == s)
            c, sc = gs.coeffs[s][sl], gs.sc[s][sl]
            i, a = pairs[sel, 0], pairs[sel, 1]
            t = (np.einsum("mp,np->pmn", c[:, i], sc[:, a])
                 + np.einsum("mp,np->pmn", c[:, a], sc[:, i]))
            blk[sel] = 0.25 * (t + np.swapaxes(t, 1, 2))
        out.append(blk.reshape(len(pairs), n * n))
    return out


def _trace_correction(blocks, ranges, n):
    """DFTB+'s rotational-invariance correction: subtract each shell's mean from the
    diagonal of the (n_pair, n, n) blocks."""
    b = blocks.reshape(-1, n, n).copy()
    for lo, hi in ranges:
        idx = np.arange(lo, hi)
        mean = b[:, idx, idx].sum(axis=1) / (hi - lo)
        b[:, idx, idx] -= mean[:, None]
    return b.reshape(len(blocks), n * n)


# ---- the Casida matrix ---------------------------------------------------------------------
class Response:
    """Omega for one symmetry (or spin polarisation) on a set of transitions."""

    def __init__(self, gs: GroundState, pairs, w, sqrt_n, q, sym: str):
        self.gs, self.pairs, self.w, self.sqrt_n, self.q, self.sym = gs, pairs, w, sqrt_n, q, sym
        self.spin = gs.spin_polarised
        self.pref = 2.0 if self.spin else 4.0
        self.v = sqrt_n * np.sqrt(w)                 # sqrt(n w)
        self.sigma = 1.0 - 2.0 * pairs[:, 2]          # +1 up, -1 down
        if (sym == "T" or self.spin) and gs.spin_w is None:
            raise ValueError("triplet and spin-polarised excitations need spin_constants")
        self._ons = None
        if gs.onsite is not None:
            self._ons = self._onsite_factors()

    def _onsite_factors(self):
        """The onsite matrix (DFTB+ onsiteEner) as a sum of products D C^T over atoms and,
        with spin polarisation, over the spins of the row and column pairs: D the pairs'
        transition densities on the atom's orbital block, C the corrected F o D of the column
        pairs (F = f_uu +- f_ud without spin polarisation, f_uu for equal and f_ud for opposite
        spins).  Items (rows, cols, D, C); rows None means all pairs."""
        gs = self.gs
        up = self.pairs[:, 2] == 0
        out = []
        for a, (dq, (fuu, fud)) in enumerate(zip(_onsite_blocks(gs, self.pairs), gs.onsite)):
            n, ranges = fuu.shape[0], gs.shell_ranges[a]
            if not self.spin:
                f = fuu + (-1.0 if self.sym == "T" else 1.0) * fud
                out.append((None, None, dq, _trace_correction(dq * f.ravel(), ranges, n)))
                continue
            for col_up in (True, False):
                cols = up == col_up
                for row_up in (True, False):
                    f = fuu if row_up == col_up else fud
                    out.append((up == row_up, cols, dq[up == row_up],
                                _trace_correction(dq[cols] * f.ravel(), ranges, n)))
        return out

    def kernel(self) -> np.ndarray:
        """K (n_pair, n_pair) including the onsite term, without the prefactors."""
        q, gs = self.q, self.gs
        if self.spin:
            k = q @ gs.gamma @ q.T
            k += np.outer(self.sigma, self.sigma) * ((q * gs.spin_w) @ q.T)
        elif self.sym == "S":
            k = q @ gs.gamma @ q.T
        else:
            k = (q * gs.spin_w) @ q.T
        if self._ons is not None:
            k = k + self._onsite_kernel()
        return k

    def _onsite_kernel(self) -> np.ndarray:
        n_pair = len(self.pairs)
        k = np.zeros((n_pair, n_pair))
        for rows, cols, dq, corrected in self._ons:
            if rows is None:
                k += dq @ corrected.T
            else:
                k[np.ix_(rows, cols)] += dq @ corrected.T
        # the onsite term has the prefactor 4 also with spin polarisation (DFTB+)
        return k * (4.0 / self.pref)

    def _onsite_product(self, vx: np.ndarray) -> np.ndarray:
        """The onsite matrix times vx (n_pair, m), without forming the matrix."""
        out = np.zeros_like(vx)
        for rows, cols, dq, corrected in self._ons:
            if rows is None:
                out += dq @ (corrected.T @ vx)
            else:
                out[rows] += dq @ (corrected.T @ vx[cols])
        return out * (4.0 / self.pref)

    def matrix(self) -> np.ndarray:
        omega = self.pref * np.outer(self.v, self.v) * self.kernel()
        omega[np.diag_indices_from(omega)] += self.w ** 2
        return 0.5 * (omega + omega.T)

    def product(self, x: np.ndarray) -> np.ndarray:
        """Omega @ x without the matrix (x: (n_pair,) or (n_pair, m))."""
        q, gs = self.q, self.gs
        vx = self.v[:, None] * (x if x.ndim == 2 else x[:, None])
        qx = q.T @ vx                                         # (n_atom, m)
        if self.spin:
            y = q @ (gs.gamma @ qx)
            y += self.sigma[:, None] * (q @ (gs.spin_w[:, None] * (q.T @ (self.sigma[:, None] * vx))))
        elif self.sym == "S":
            y = q @ (gs.gamma @ qx)
        else:
            y = q @ (gs.spin_w[:, None] * qx)
        if self._ons is not None:
            y = y + self._onsite_product(vx)
        out = self.pref * self.v[:, None] * y + (self.w ** 2)[:, None] * (x if x.ndim == 2
                                                                           else x[:, None])
        return out if x.ndim == 2 else out[:, 0]

    def solve(self, n: int, solver: str):
        """The lowest n eigenpairs (Omega^2, F) of Omega."""
        size = len(self.w)
        if solver == "auto":
            solver = "exact" if size <= EXACT_LIMIT else "davidson"
        if solver == "exact" or n >= size - 1:
            evals, f = sla.eigh(self.matrix(), subset_by_index=[0, n - 1])
            return evals, f
        if solver == "davidson":
            return self.davidson(n)
        from scipy.sparse.linalg import LinearOperator, eigsh
        op = LinearOperator((size, size), matvec=self.product, matmat=self.product,
                            dtype=float)
        evals, f = eigsh(op, k=n, which="SA", tol=1e-14, ncv=min(size, max(2 * n + 1, 40)))
        order = np.argsort(evals)
        return evals[order], f[:, order]

    def davidson(self, n: int, tol: float = DAVIDSON_TOL, max_iter: int = 1000):
        """Block Davidson for the lowest n eigenpairs: start from the unit vectors of the n + a
        few lowest w^2, add the preconditioned residuals r / (Omega^2 - w^2) of the states not
        yet converged (|r| > tol), and restart from the Ritz vectors when the space is full.
        A few pairs beyond the n-th are tracked, so a degenerate group at the edge converges."""
        diag = self.w ** 2
        size = len(diag)
        m = min(size, n + min(n, 8))
        max_space = min(size, max(8 * m, 200))
        v = np.zeros((size, m))
        v[np.argsort(diag, kind="stable")[:m], np.arange(m)] = 1.0
        av = self.product(v)
        for _ in range(max_iter):
            h = v.T @ av
            theta, s = np.linalg.eigh(0.5 * (h + h.T))
            x, ax = v @ s[:, :m], av @ s[:, :m]
            r = ax[:, :n] - x[:, :n] * theta[:n]
            todo = np.linalg.norm(r, axis=0) > tol
            if not todo.any():
                return theta[:n], x[:, :n]
            denom = theta[:n][todo][None, :] - diag[:, None]
            t = r[:, todo] / np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            if v.shape[1] + t.shape[1] > max_space:
                v, av = x, ax
            t = _new_directions(t, v)
            if t.shape[1] == 0:
                raise RuntimeError("Casida Davidson solver: no new search directions")
            v, av = np.hstack([v, t]), np.hstack([av, self.product(t)])
        raise RuntimeError(f"Casida Davidson solver: no convergence in {max_iter} iterations")


def _new_directions(t, v):
    """The part of the columns of t orthogonal to the orthonormal v, orthonormalised; columns
    that lie (practically) in the span of v or of the others are dropped."""
    t = t / np.linalg.norm(t, axis=0)
    for _ in range(2):
        t = t - v @ (v.T @ t)
    u, sv, _ = np.linalg.svd(t, full_matrices=False)
    u = u[:, sv > 1e-8]
    u = u - v @ (v.T @ u)
    return u / np.linalg.norm(u, axis=0)


# ---- the calculation -------------------------------------------------------------------------
def excitations(cas: Casida, gs: GroundState) -> Excitations | list:
    """Excitations of one ground state; a list [triplets, singlets] for "both" (DFTB+'s
    order)."""
    gs.prepare()
    pairs, w, sqrt_n = single_particle_transitions(gs)
    n_exc = cas.n_excitations
    if n_exc + 1 >= len(w):
        raise ValueError(f"insufficient single-particle excitations, {len(w)}, for "
                         f"{n_exc} excited states (as DFTB+)")
    order = np.argsort(w, kind="stable")       # DFTB+: merge sort with the oscillator window
    pairs, w, sqrt_n = pairs[order], w[order], sqrt_n[order]
    q = transition_charges(gs, pairs)
    dip = q @ gs.positions                                   # single-particle d_ia
    sposz = 2.0 / 3.0 * w * (dip ** 2).sum(axis=1)
    keep = _windows(cas, gs, pairs, w, sposz)
    pairs, w, sqrt_n, q, dip = pairs[keep], w[keep], sqrt_n[keep], q[keep], dip[keep]
    spx = SingleParticle(w.copy(), sposz[keep], pairs.copy(),    # DFTB+ SPX.DAT: these
                         gs.spin_polarised)

    syms = [None] if gs.spin_polarised else {"S": ["S"], "T": ["T"], "B": ["T", "S"]}[
        SYMMETRIES[cas.symmetry]]
    out = []
    for sym in syms:
        resp = Response(gs, pairs, w, sqrt_n, q, sym or " ")
        omega2, f = resp.solve(n_exc, cas.solver)
        out.append(_properties(cas, gs, resp, omega2, f, dip, spx, sym))
    return out[0] if len(out) == 1 else out


def _windows(cas: Casida, gs: GroundState, pairs, w, sposz) -> np.ndarray:
    """Indices of the transitions that take part (DFTB+ dipSelect and the energy window)."""
    n_all, n_exc = len(w), cas.n_excitations
    if cas.energy_window is None and cas.oscillator_window is None:
        keep = np.arange(n_all)
    else:
        if cas.energy_window is None:
            n_r, rest = 0, np.arange(n_all)
        else:
            n_r = int(np.count_nonzero(w <= w[n_exc - 1] + cas.energy_window))
            rest = np.arange(n_r, n_all)
        bright = []
        if cas.oscillator_window is not None and len(rest):
            bright = _bright_groups(gs, pairs[rest], w[rest], sposz[rest], cas.oscillator_window)
            bright = rest[bright]
        keep = np.concatenate([np.arange(n_r), np.asarray(bright, int)])
    # ARPACK needs one more transition than states (DFTB+ with ARPACK compiled in)
    n_min = min(n_exc + 1, n_all)
    if len(keep) < n_min:
        keep = np.union1d(keep, np.arange(n_min))
    return np.sort(keep)


def _bright_groups(gs: GroundState, pairs, w, sposz, threshold) -> np.ndarray:
    """dipSelect: groups of consecutive transitions with the same occupied-level energy and
    transition energy (to single precision) are kept if their summed oscillator strength
    exceeds the threshold."""
    keep, start = [], 0
    e_occ = np.array([gs.eps[0][i] for i in pairs[:, 0]])      # DFTB+ uses spin channel 1
    for k in range(1, len(w) + 1):
        if (k == len(w) or abs(e_occ[k] - e_occ[start]) > EPS_SINGLE
                or abs(w[k] - w[start]) > EPS_SINGLE):
            if sposz[start:k].sum() > threshold:
                keep.extend(range(start, k))
            start = k
    return np.array(keep, int)


def _properties(cas: Casida, gs: GroundState, resp: Response, omega2, f, dip, spx, sym):
    if np.any(omega2 <= 0):
        warnings.warn("negative or zero Omega^2 in the Casida equation: the ground state is "
                      "unstable", RuntimeWarning, stacklevel=3)
    with np.errstate(invalid="ignore", divide="ignore"):
        omega = np.sqrt(np.abs(omega2))
        xpy = f * np.sqrt(resp.w[:, None] / omega[None, :])
        xmy = f * np.sqrt(omega[None, :] / resp.w[:, None])
    closed = not gs.spin_polarised
    if closed and sym == "T":
        tdip = np.zeros((len(omega2), 3))
        osc = np.zeros(len(omega2))
    else:
        tdip = (dip * resp.sqrt_n[:, None]).T @ xpy                      # (3, n)
        tdip = tdip.T * (math.sqrt(2.0) if closed else 1.0)
        osc = 2.0 / 3.0 * omega * (tdip ** 2).sum(axis=1)
    sq = xpy ** 2
    weights_all = sq / np.linalg.norm(sq, axis=0)[None, :]
    dominant = np.argmax(weights_all, axis=0)
    state = 0
    if cas.state_of_interest > 0:
        state = cas.state_of_interest
    elif cas.state_of_interest < 0:
        state = int(np.argmax(osc)) + 1
    s2 = _spin_contamination(gs, resp.pairs, xpy) if gs.spin_polarised else None
    tq = None
    if cas.transition_charges and state > 0:
        # DFTB+ getAndWriteTransitionCharges: kappa sum_ia q^ia (X+Y)_ia, kappa = sqrt(2)
        # closed shell (1 spin polarised); unlike the dipole not weighted by sqrt(n_ia)
        tq = (1.0 if gs.spin_polarised else math.sqrt(2.0)) * (resp.q.T @ xpy[:, state - 1])
    return Excitations(omega2=omega2, oscillator_strengths=osc, transition_dipoles=tdip,
                       symmetries=None if gs.spin_polarised else [sym] * len(omega2),
                       transitions=resp.pairs[dominant], weights=weights_all[dominant,
                                                                             np.arange(len(omega2))],
                       ks_energies=resp.w[dominant], xpy=xpy, pairs=resp.pairs,
                       single_particle=spx, spin_contamination=s2, state=state, xmy=xmy,
                       transition_charges=tq)


def _spin_contamination(gs: GroundState, pairs, xpy) -> np.ndarray:
    """DFTB+'s D<S^2> of every state (linrespcommon getExcSpin, after Garcia's thesis)."""
    n_up = math.ceil(gs.occupations[0].sum() - 1e-12)
    n_dn = math.ceil(gs.occupations[1].sum() - 1e-12)
    o = gs.coeffs[0].T @ gs.sc[1]                  # <phi_p up | phi_q down>
    m_up = o[:, :n_dn] @ o[:, :n_dn].T              # sum_m<n_dn <p|m><q|m>, up indices
    m_dn = o[:n_up, :].T @ o[:n_up, :]              # sum_m<n_up <m|p><m|q>, down indices
    norb = o.shape[0]
    out = np.empty(xpy.shape[1])
    up = pairs[:, 2] == 0
    for k in range(xpy.shape[1]):
        td = xpy[:, k] / np.linalg.norm(xpy[:, k])
        t_up, t_dn = np.zeros((norb, norb)), np.zeros((norb, norb))
        t_up[pairs[up, 0], pairs[up, 1]] = td[up]
        t_dn[pairs[~up, 0], pairs[~up, 1]] = td[~up]
        s_iaja = np.sum(t_up * (m_up @ t_up)) + np.sum(t_dn * (m_dn @ t_dn))
        s_iaib = np.sum(t_up * (t_up @ m_up)) + np.sum(t_dn * (t_dn @ m_dn))
        s_iajb = np.sum(t_up * (o @ t_dn @ o.T))
        out[k] = s_iaja - s_iaib - 2.0 * s_iajb
    return out
