"""Derivatives of DFTB eigenvalues with respect to model parameters at frozen charges (A series,
A0 section 4: fitting environment-dependent electronic parameters to band references).

For H c = eps S c with c^H S c = 1, a parameter x that changes H and S gives the
Hellmann-Feynman derivative

    d eps_n / dx = Re c_n^H (dH/dx - eps_n dS/dx) c_n

(within a degenerate level the perturbation c_m^H (dH/dx - eps dS/dx) c_n is diagonalised and
its eigenvalues, sorted, go to the sorted states).  At frozen populations (the SCC's ``q_in``)
the charge response is left out, which is what A0 means by "derivatives at frozen charges,
loss values from the full SCC": the fitter's Jacobian is approximate, its residuals exact.

* :func:`frozen_matrices`: H(k) and S(k) of fixed populations, as :func:`pydftb.bands.eigenstates`
  builds them (shared code, no behaviour change).
* :func:`eigenvalue_derivatives`: generic, for any model vector x (``make_engine(x) -> DFTB``):
  dH/dx and dS/dx by central differences of the frozen matrices, no extra SCC or
  diagonalisation.
* :func:`adaptive_eigenvalue_derivatives`: analytic for :class:`pydftb.adaptive.AdaptiveParams`
  (on-site shifts: dH/dp_A = the diagonal of the shell's orbitals; hopping scales:
  dH/dp_A = (P_A H0' + H0' P_A) / 2 with H0' the scaled off-site H0; S does not change; the
  repulsive channels do not change eigenvalues).

Energies in Hartree per unit of the parameter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bands as _bands
from .hamiltonian import fold, fold_all

DEGENERACY = 1e-7          # Hartree: levels closer than this are treated as degenerate


@dataclass
class FrozenMatrices:
    """H and S at fixed populations: (n_spin, n_k, n_state, n_state) each."""
    h: np.ndarray
    s: np.ndarray
    kpts: np.ndarray
    kweights: np.ndarray

    @property
    def nspin(self) -> int:
        return self.h.shape[0]


def frozen_matrices(engine, species, positions, cell=None, pbc=None, populations=None,
                    kpts=None, chunk: int | None = None) -> FrozenMatrices:
    """H(k) and S(k) of ``engine`` at fixed populations (arguments as
    :func:`pydftb.bands.eigenstates`); collinear spin: one (H, S) per channel; two-component:
    the spinor H and the 2N overlap."""
    fz = _bands._frozen_setup(engine, species, positions, cell, pbc, populations, kpts)
    nspin, nstate, k = fz.nspin, fz.nstate, fz.k
    chunk = chunk or max(1, _bands.MATRIX_BUDGET // (nstate * nstate))
    h_all = s_all = None
    for c0, kc, h, s in _bands._frozen_chunks(engine, fz, chunk):
        n = len(kc)
        h = np.asarray(h).reshape(nspin, n, nstate, nstate)
        s = np.asarray(s)
        if s.shape[0] != nspin * n:                  # one S for both spin channels
            s = np.concatenate([s] * nspin)
        s = s.reshape(nspin, n, nstate, nstate)
        if h_all is None:
            dt = np.result_type(h.dtype, s.dtype)
            h_all = np.empty((nspin, len(k), nstate, nstate), dt)
            s_all = np.empty((nspin, len(k), nstate, nstate), dt)
        elif np.iscomplexobj(h) and not np.iscomplexobj(h_all):
            h_all, s_all = h_all.astype(complex), s_all.astype(complex)
        h_all[:, c0:c0 + n] = h
        s_all[:, c0:c0 + n] = s
    return FrozenMatrices(h_all, s_all, np.asarray(k, float), np.asarray(fz.kw, float))


def _groups(e, tol):
    """Runs of consecutive (sorted) levels closer than ``tol``: list of index arrays."""
    out, start = [], 0
    for n in range(1, len(e) + 1):
        if n == len(e) or e[n] - e[n - 1] > tol:
            out.append(np.arange(start, n))
            start = n
    return out


def hellmann_feynman(c, e, dh, ds=None, degeneracy: float = DEGENERACY, states=None):
    """d eps / dx of the states of one (spin, k) block: ``c`` (n, n_state) eigenvectors
    (c^H S c = 1), ``e`` (n_state,) eigenvalues, ``dh``/``ds`` (n, n) the derivatives of H and S
    (``ds`` None: S fixed).  ``states``: the indices wanted (whole degenerate levels are
    included).  Returns (n_state,) with NaN for states not computed."""
    e = np.asarray(e, float)
    out = np.full(len(e), np.nan)
    groups = _groups(e, degeneracy)
    if states is not None:
        want = np.zeros(len(e), bool)
        want[np.asarray(states, int)] = True
        groups = [g for g in groups if want[g].any()]
    if not groups:
        return out
    sel = np.concatenate(groups)
    cs = c[:, sel]
    a = cs.conj().T @ dh @ cs
    b = None if ds is None else cs.conj().T @ ds @ cs
    pos = {int(n): k for k, n in enumerate(sel)}
    for g in groups:
        idx = np.array([pos[int(n)] for n in g])
        eg = float(e[g].mean())
        w = a[np.ix_(idx, idx)]
        if b is not None:
            w = w - eg * b[np.ix_(idx, idx)]
        if len(g) == 1:
            out[g[0]] = float(np.real(w[0, 0]))
        else:
            out[g] = np.linalg.eigvalsh(0.5 * (w + w.conj().T))
    return out


def eigenvalue_derivatives(make_engine, x, species, positions, cell=None, pbc=None,
                           populations=None, kpts=None, states=None, step: float = 1e-5,
                           degeneracy: float = DEGENERACY, wanted=None) -> np.ndarray:
    """d eps / dx (n_spin, n_k, n_state, n_x), Hartree per unit x, at frozen ``populations``
    for any model vector x: ``make_engine(x)`` builds the :class:`pydftb.DFTB` engine (its
    SCC's ``q_in`` are the populations; positions and cell in Bohr).  dH/dx, dS/dx by central
    differences of :func:`frozen_matrices` (step ``step`` in every x).  ``states``: the
    :class:`pydftb.bands.States` at x with ``vectors`` (computed if None); ``wanted``: state
    indices to compute (default all)."""
    x = np.asarray(x, float)
    if states is None or states.vectors is None:
        states = _bands.eigenstates(make_engine(x), species, positions, cell, pbc, populations,
                                    kpts, vectors=True)
    e, c = states.eigenvalues, states.vectors
    kk = None if kpts is None else (states.kpts, states.kweights)
    out = np.full(e.shape + (len(x),), np.nan)
    for j in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[j] += step
        xm[j] -= step
        mp = frozen_matrices(make_engine(xp), species, positions, cell, pbc, populations, kk)
        mm = frozen_matrices(make_engine(xm), species, positions, cell, pbc, populations, kk)
        dh = (mp.h - mm.h) / (2.0 * step)
        ds = (mp.s - mm.s) / (2.0 * step)
        s_fixed = not np.any(ds)
        for sp in range(e.shape[0]):
            for k in range(e.shape[1]):
                out[sp, k, :, j] = hellmann_feynman(c[sp, k], e[sp, k], dh[sp, k],
                                                    None if s_fixed else ds[sp, k], degeneracy,
                                                    wanted)
    return out


def adaptive_eigenvalue_derivatives(engine, models, species, positions, cell=None, pbc=None,
                                    populations=None, kpts=None, states=None,
                                    degeneracy: float = DEGENERACY, wanted=None) -> np.ndarray:
    """:func:`eigenvalue_derivatives` for :class:`pydftb.adaptive.AdaptiveParams` models
    (``engine`` built with ``adaptive=models``), analytic: (n_spin, n_k, n_state, n_x) with
    x the concatenated model vectors.  Collinear or no spin (not two-component)."""
    from .adaptive import AdaptiveParams, channel_jacobians
    models = tuple(models)
    if not all(isinstance(m, AdaptiveParams) for m in models):
        raise TypeError("adaptive_eigenvalue_derivatives takes AdaptiveParams models")
    if states is None or states.vectors is None:
        states = _bands.eigenstates(engine, species, positions, cell, pbc, populations, kpts,
                                    vectors=True)
    if states.two_component:
        raise NotImplementedError("analytic band derivatives: no two-component calculations")
    st = engine._setup(species, positions, cell, pbc, False, {}, matrices=False)
    terms = st.adaptive
    if terms is None:
        raise ValueError("the engine has no adaptive models")
    norb = st.basis.norb
    sizes = [len(m.vector()) for m in models]
    n_x = int(sum(sizes))
    # dH/dx_j as two diagonals over orbitals: on-site (d_on) and hopping (d_hop) weights
    d_on = np.zeros((n_x, norb))
    d_hop = np.zeros((n_x, norb))
    k0 = 0
    for m, size in zip(models, sizes):
        jac = channel_jacobians(m, terms)          # key -> (atoms, J (n_atom_el, size))
        for key, (atoms, jm) in jac.items():
            kind = key[0]
            if kind not in ("onsite", "hopping"):
                continue
            orbs = terms.shell_of[(key[1], key[2])]
            target = d_on if kind == "onsite" else d_hop
            for a_idx, o in enumerate(orbs):
                target[k0:k0 + size, o] += jm[a_idx][:, None]
        k0 += size
    use_hop = bool(np.any(d_hop))
    e, c = states.eigenvalues, states.vectors
    kp = states.kpts
    gamma = len(kp) == 1 and not np.any(kp[0])
    out = np.full(e.shape + (n_x,), np.nan)
    pb = st.pairs
    zeros = np.zeros(norb)
    for k in range(e.shape[1]):
        hoff = None
        if use_hop:
            hoff = (fold(pb, pb.h, norb, None, zeros) if gamma
                    else fold_all(pb, pb.h, norb, kp[k:k + 1], zeros)[0])
            hoff = np.asarray(hoff)
        for sp in range(e.shape[0]):
            ck = c[sp, k]
            for j in range(n_x):
                dh = np.diag(d_on[j]).astype(ck.dtype)
                if hoff is not None and np.any(d_hop[j]):
                    dh = dh + 0.5 * (d_hop[j][:, None] * hoff + hoff * d_hop[j][None, :])
                out[sp, k, :, j] = hellmann_feynman(ck, e[sp, k], dh, None, degeneracy, wanted)
    return out
