"""Band structures, densities of states and projected DOS at frozen charges.

DFTB+ computes a band structure in two runs: an SCC run on a k-point grid writes the
last input populations to ``charges.bin``; a second run reads them back
(``ReadInitialCharges``) and builds and diagonalises H once at the path points
(``Klines``, one SCC iteration).  :func:`eigenstates` does the second run in-process:
it reuses the calculator's setup (real-space H0/S pair blocks, Coulomb kernel) and its
potentials, then folds and diagonalises H(k) in chunks of k-points.

Projections are those of DFTB+ ``ProjectStates``: the share of state c on orbital mu is
Re[c_mu^* (S c)_mu].  DOS curves follow DFTB+'s ``dp_dos`` (dptools): same broadening
functions, cut-offs and energy grid (:func:`broaden`).
Design notes: ``docs/design/M2e-bands.html``.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial, pi, sqrt

import numpy as np

from .constants import HARTREE
from .hamiltonian import Basis, fold, fold_all
from .occupations import fill
from .solvers import SOLVERS, BatchedSolver
from . import spinorbit as so

# dp_dos defaults: broadening width (eV), cut-off in widths, energy grid (eV)
DOS_WIDTH = {"gauss": 0.1, "fermi": 0.1, "mp": 0.1}
DOS_CUTOFF = {"gauss": 3.5, "fermi": 7.0, "mp": 3.5}
DOS_GRID = 0.01
MATRIX_BUDGET = 2_000_000      # k-chunk size: chunk * n_orb^2 stays below this


def klines(lines):
    """k-points of DFTB+ ``KPointsAndWeights = Klines { ... }`` (fractional coordinates).

    ``lines`` is a sequence of ``(n, k)``.  Each line adds ``n`` evenly spaced points from
    the previous point (excluded) to ``k`` (included); the first line starts at Gamma and
    ``n = 0`` jumps to ``k`` without adding a point.  Returns ``(kpts, weights)``, equal
    weights summing to 1.
    """
    lines = list(lines)
    if not lines:
        raise ValueError("Klines needs at least one line")
    steps = [int(n) for n, _ in lines]
    if any(n < 0 for n in steps):
        raise ValueError("Klines steps must be >= 0")
    if sum(steps) < 1:
        raise ValueError("the Klines steps must add up to at least 1")
    start = np.zeros(3)
    points = []
    for n, (_, k) in zip(steps, lines):
        k = np.asarray(k, float).reshape(3)
        if n:
            step = (k - start) / n                    # as DFTB+: start + j * step
            points += [start + j * step for j in range(1, n + 1)]
        start = k
    kpts = np.array(points)
    return kpts, np.full(len(kpts), 1.0 / len(kpts))


@dataclass
class States:
    """Eigenstates at frozen charges.  Energies in Hartree; ``dos``, ``pdos`` and
    ``band_structure`` return eV.  Spin channels are the first axis (up, down); a
    two-component calculation (M6a) has one channel of 2N spinor states."""
    kpts: np.ndarray                    # (n_k, 3) fractional
    kweights: np.ndarray                # (n_k,), sum 1
    eigenvalues: np.ndarray             # (n_spin, n_k, n_state)
    occupations: np.ndarray             # (n_spin, n_k, n_state); without spin incl. factor 2
                                        # (two-component: at most 1)
    fermi: np.ndarray                   # (n_spin,) Fermi level(s) of the filling at these k-points
    scc_fermi: np.ndarray | None        # (n_spin,) Fermi level(s) of the SCC run (band plot zero)
    basis: Basis
    projections: np.ndarray | None = None   # (n_spin, n_k, n_orb, n_state) Re[c* (S c)]
                                            # (two-component: summed over both spins)

    @property
    def nspin(self) -> int:
        return self.eigenvalues.shape[0]

    @property
    def two_component(self) -> bool:
        """Spinor states (non-collinear spin or spin-orbit coupling)."""
        return self.eigenvalues.shape[2] == 2 * self.basis.norb

    def orbitals(self, atoms=None, shell=None, m=None) -> np.ndarray:
        """Orbital indices of a region, as DFTB+ ``ProjectStates`` regions.

        ``atoms``: atom indices (default all).  ``shell``: index into each atom's shells
        (DFTB+ ``ShellResolved``, file ``<label>.<shell + 1>.out``).  ``m``: with ``shell``,
        one orbital (DFTB+ ``OrbitalResolved``, file ``<label>.<shell + 1>.<m>.out``).
        """
        b = self.basis
        first_orb = np.concatenate([[0], np.cumsum(2 * b.shell_l + 1)[:-1]])
        first_shell = np.searchsorted(b.shell_atom, np.arange(len(b.species)))
        local_shell = b.orb_shell - first_shell[b.orb_atom]
        m_orb = np.arange(b.norb) - first_orb[b.orb_shell] - b.shell_l[b.orb_shell]
        sel = np.ones(b.norb, bool)
        if atoms is not None:
            sel &= np.isin(b.orb_atom, np.atleast_1d(atoms))
        if shell is not None:
            sel &= local_shell == shell
        if m is not None:
            if shell is None:
                raise ValueError("m needs a shell")
            sel &= m_orb == m
        idx = np.nonzero(sel)[0]
        if len(idx) == 0:
            raise ValueError(f"no orbitals in region atoms={atoms}, shell={shell}, m={m}")
        return idx

    def weights(self, atoms=None, shell=None, m=None) -> np.ndarray:
        """(n_spin, n_k, n_state) share of every state on a region's orbitals; the second
        column of DFTB+'s region files.  Needs ``projections``."""
        if self.projections is None:
            raise ValueError("these states have no projections; compute them with projections=True")
        return self.projections[:, :, self.orbitals(atoms, shell, m), :].sum(axis=2)

    def dos(self, width=None, kind: str = "gauss", cutoff=None, grid: float = DOS_GRID,
            mp_order: int = 1, weights=None, occupied: bool = False):
        """DOS as DFTB+'s ``dp_dos`` -> (energies (n_x,) eV, dos (n_spin, n_x) states/eV).

        ``weights`` (n_spin, n_k, n_state) weight every state (``dp_dos -w`` on a region
        file); ``occupied`` weights by occupations (``dp_dos -w`` on ``band.out``).
        """
        if occupied:
            weights = self.occupations
        return broaden(self.eigenvalues * HARTREE, self.kweights, weights, width, kind, cutoff,
                       grid, mp_order)

    def pdos(self, atoms=None, shell=None, m=None, **kw):
        """Projected DOS of a region (see :meth:`orbitals`); keywords as :meth:`dos`."""
        return self.dos(weights=self.weights(atoms, shell, m), **kw)

    def band_structure(self, path, reference=None, ev: float = HARTREE):
        """ASE ``BandStructure`` (eV) for an ASE ``BandPath`` whose points are ``kpts``.

        ``reference`` (eV) defaults to the SCC Fermi level (spin up's if there are two).
        """
        from ase.spectrum.band_structure import BandStructure
        if len(path.kpts) != len(self.kpts) or not np.allclose(path.kpts, self.kpts):
            raise ValueError("the band path's k-points differ from these states' k-points")
        if reference is None:
            reference = 0.0 if self.scc_fermi is None else float(self.scc_fermi[0]) * ev
        return BandStructure(path=path, energies=self.eigenvalues * ev, reference=reference)


def eigenstates(engine, species, positions, cell=None, pbc=None, populations=None, kpts=None,
                projections: bool = False, scc_fermi=None, chunk: int | None = None) -> States:
    """Eigenstates of H built from fixed populations, at any k-points (atomic units).

    ``engine`` is a :class:`pydftb.DFTB`; positions and cell in Bohr.  ``populations`` is the
    vector the SCC loop mixes, normally ``DFTBResult.q_in`` (DFTB+'s ``charges.bin``): site
    populations, or with spin [up, down] per shell (two-component: [q, m_x, m_y, m_z] per
    shell; with DFTB+U or onsite corrections the block vector of :mod:`pydftb.blockpot`);
    None = the reference populations.
    ``kpts``: ``(kpts, weights)``, an (n_k, 3) array (equal weights), or None for the
    engine's own k-points.  Occupations are filled at these k-points as DFTB+ does in its
    band-structure run; ``scc_fermi`` (Hartree, one value or one per spin) is kept as the
    zero of band plots.
    """
    st = engine._setup(species, positions, cell, pbc, False, {}, matrices=False)
    if kpts is None:
        k, kw = st.kpts, st.kw
    else:
        if st.cell is None or not np.any(st.pbc):
            raise ValueError("k-points need a periodic system")
        k, kw = _kpoint_set(kpts)
    q = (engine._initial_populations(st) if populations is None
         else np.asarray(populations, float))
    pauli = st.pauli is not None
    nsh = len(st.basis.shell_l)
    n_expected = 4 * nsh if pauli else len(st.q0) if st.spin is None else 2 * nsh
    if st.block is not None:
        n_expected = st.block.size
    if q.shape != (n_expected,):
        raise ValueError(f"populations need {n_expected} entries (the SCC's mixed vector, "
                         f"DFTBResult.q_in), got shape {q.shape}")
    v_orb, vs_orb = engine._potentials(st, q)
    # block potentials (M6b), with the onsite corrections: DFTB+'s band run reads its charges
    pots = None if st.block is None else engine._block_potentials(st, q)
    nspin = 1 if vs_orb is None or pauli else 2
    norb = st.basis.norb
    nstate = 2 * norb if pauli else norb
    pb = st.pairs
    ones = np.ones(norb)
    gamma = len(k) == 1 and not np.any(k[0])
    solver = engine.solver if not isinstance(engine.solver, str) else SOLVERS[engine.solver]()
    if not hasattr(solver, "eig"):
        solver = BatchedSolver()                      # SP2 has no eigenvectors
    chunk = chunk or max(1, MATRIX_BUDGET // (nstate * nstate))
    evals = np.empty((nspin, len(k), nstate))
    proj = np.empty((nspin, len(k), norb, nstate)) if projections else None
    for c0 in range(0, len(k), chunk):
        kc = k[c0:c0 + chunk]
        if gamma:
            h0 = fold(pb, pb.h, norb, None, st.eps0)[None]
            s = fold(pb, pb.s, norb, None, ones)[None]
        else:
            h0 = fold_all(pb, pb.h, norb, kc, st.eps0)
            s = fold_all(pb, pb.s, norb, kc, ones)
        if pauli:                                                # two-component (M6a)
            p = st.pauli
            hso = None if p.lam is None else so.soc_hamiltonian(p.lam, s if p.dual else None)
            if pots is not None:
                h = engine._block_hamiltonian(st, h0, s, v_orb, vs_orb, pots, hso)
            else:
                h = engine._spinor_hamiltonian(h0, s, v_orb, vs_orb, hso)
            s2 = so.overlap2(s)
            solver.prepare(s2)
            e, c = solver.eig(h)
            n = len(kc)
            evals[0, c0:c0 + n] = e
            if projections:
                share = (np.conj(c) * (s2 @ c)).real
                proj[0, c0:c0 + n] = share[:, :norb] + share[:, norb:]
            continue
        if pots is not None:
            h = engine._block_hamiltonian(st, h0, s, v_orb, vs_orb, pots, None)
            if vs_orb is not None:
                s = np.concatenate([s, s])
        else:
            h = h0 + 0.5 * s * (v_orb[:, None] + v_orb[None, :])     # as in the SCC loop
            if vs_orb is not None:
                hs = 0.5 * s * (vs_orb[:, None] + vs_orb[None, :])
                h = np.concatenate([h + hs, h - hs])                 # up, down
                s = np.concatenate([s, s])
        solver.prepare(s)
        e, c = solver.eig(h)
        n = len(kc)
        evals[:, c0:c0 + n] = e.reshape(nspin, n, norb)
        if projections:
            share = (np.conj(c) * (s @ c)).real                 # Re[c* (S c)] per orbital, state
            proj[:, c0:c0 + n] = share.reshape(nspin, n, norb, norb)
    occ, fermi = _fill(engine, st, evals, kw)
    if scc_fermi is not None:
        scc_fermi = np.broadcast_to(np.atleast_1d(np.asarray(scc_fermi, float)), (nspin,)).copy()
    return States(np.asarray(k, float), np.asarray(kw, float), evals, occ, fermi, scc_fermi,
                  st.basis, proj)


def _kpoint_set(kpts):
    """(kpts, weights) | (n_k, 3) array of points (equal weights) -> arrays, weights sum 1."""
    if isinstance(kpts, np.ndarray) and kpts.ndim == 2:
        k = np.asarray(kpts, float)
        return k, np.full(len(k), 1.0 / len(k))
    k, w = kpts
    k = np.atleast_2d(np.asarray(k, float))
    w = np.asarray(w, float)
    return k, w / w.sum()


def _fill(engine, st, evals, kw):
    """Occupations and Fermi level(s) at these k-points, as the SCC loop fills (``_solve``)."""
    sm = engine.smearing
    if st.pauli is not None:
        occ, mu, _ = fill(evals[0], kw, st.nelec, sm, spin_factor=1.0)
        return np.array(occ)[None], np.array([mu])
    if st.spin is None:
        occ, mu, _ = fill(evals[0], kw, st.nelec, sm)
        return np.array(occ)[None], np.array([mu])
    if st.spin.shared_ef:
        occ, mu, _ = fill(evals.reshape(-1, evals.shape[2]), np.tile(kw, 2), st.nelec, sm,
                          spin_factor=1.0)
        return np.array(occ).reshape(evals.shape), np.array([mu, mu])
    parts = [fill(evals[s], kw, n, sm, spin_factor=1.0)
             for s, n in ((0, st.spin.n_up), (1, st.spin.n_down))]
    return np.array([np.array(p[0]) for p in parts]), np.array([p[1] for p in parts])


def broaden(energies, kweights, weights=None, width=None, kind: str = "gauss", cutoff=None,
            grid: float = DOS_GRID, mp_order: int = 1):
    """Broadened DOS, exactly as DFTB+'s ``dp_dos``.

    ``energies``: (n_spin, n_k, n_state) or (n_k, n_state) in eV; ``kweights``: (n_k,) or
    (n_spin, n_k); ``weights``: one per state (None: 1, the total DOS).  ``kind``: "gauss"
    exp(-x^2/w^2)/(w sqrt(pi)), "fermi" (derivative of the Fermi function, kT = w) or "mp"
    (Methfessel-Paxton of order ``mp_order``).  ``width`` (eV) and ``cutoff`` (in widths)
    default to dp_dos's: 0.1 eV and 3.5 (Fermi 7).  The energy grid starts at the multiple
    of ``grid`` below the lowest level minus the cut-off.  Returns (x (n_x,) eV,
    dos (n_spin, n_x)).  Only the summation order differs from dp_dos (it is vectorised),
    and a level whose window would start one grid point below the grid is clipped to the
    grid instead of being dropped (dp_dos's slice wraps around).
    """
    kind = kind.lower()
    if kind not in DOS_WIDTH:
        raise ValueError(f"kind must be one of {sorted(DOS_WIDTH)}, not {kind!r}")
    e = np.asarray(energies, float)
    if e.ndim == 2:
        e = e[None]
    nspin, nk, nst = e.shape
    kw = np.broadcast_to(np.asarray(kweights, float), (nspin, nk))
    w = np.ones_like(e) if weights is None else np.broadcast_to(np.asarray(weights, float), e.shape)
    sigma = DOS_WIDTH[kind] if width is None else float(width)
    srange = DOS_CUTOFF[kind] if cutoff is None else float(cutoff)
    if kind == "fermi":
        aa, bb = 0.5 / sigma, 1.0 / sigma
        func = lambda x: 1.0 / (1.0 + np.cosh(bb * x))          # noqa: E731
    elif kind == "gauss":
        aa, bb = 1.0 / (sigma * np.sqrt(np.pi)), -1.0 / sigma ** 2
        func = lambda x: np.exp(bb * x * x)                     # noqa: E731
    else:
        if mp_order < 1:
            raise ValueError("Methfessel-Paxton order must be >= 1 (order 0 is 'gauss')")
        aa = bb = 1.0 / sigma
        coefs = np.zeros(2 * (mp_order + 1))
        for i in range(mp_order + 1):
            coefs[2 * i] = (-1) ** i / (factorial(i) * 4 ** i * sqrt(pi))
        func = lambda x: (np.polynomial.hermite.hermval(bb * x, coefs)   # noqa: E731
                          * np.exp(-x * x * bb * bb))
    dsigma = srange * sigma
    xmin = np.floor((e.min() - dsigma) / grid) * grid
    xmax = np.ceil((e.max() + dsigma) / grid) * grid
    x = np.arange(xmin, xmax + grid, grid, dtype=float)
    nx = len(x)
    y = np.zeros((nspin, nx))
    lo = np.floor((e - dsigma - xmin) / grid).astype(int)
    up = np.ceil((e + dsigma - xmin) / grid).astype(int)
    span = np.arange(int((up - lo).max()) + 1)
    pref = aa * kw[:, :, None] * w                              # (aa * k weight) * weight
    per_chunk = max(1, 4_000_000 // len(span))
    for s in range(nspin):
        lo_s, up_s = np.clip(lo[s].ravel(), 0, None), np.minimum(up[s].ravel(), nx - 1)
        e_s, p_s = e[s].ravel(), pref[s].ravel()
        for c0 in range(0, len(e_s), per_chunk):
            sl = slice(c0, c0 + per_chunk)
            idx = lo_s[sl, None] + span[None, :]
            ok = idx <= up_s[sl, None]
            idx = np.where(ok, idx, 0)
            val = p_s[sl, None] * func(e_s[sl, None] - x[idx])
            y[s] += np.bincount(idx[ok], weights=val[ok], minlength=nx)
    return x, y
