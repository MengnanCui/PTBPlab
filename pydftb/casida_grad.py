"""Excited-state gradients of linear-response TD-DFTB (Z-vector method, milestone M6e).

The model is DFTB+ 25.1's ``ExcitedState { Casida { StateOfInterest; ExcitedStateForces;
WriteMulliken } }`` (``timedep/linrespgrad.F90``; Heringer, Niehaus and Frauenheim,
J. Comput. Chem. 28, 2589 (2007)): the gradient of E_0 + Omega for one excited state of a
molecule with integer occupations, and the relaxed difference density P whose Mulliken
populations are the excited state's charges (DFTB+ XCH.DAT, XREST.DAT).

Notation as in :mod:`pydftb.casida`.  Per spin channel, X+Y and X-Y of the state are
(n_occ, n_virt) matrices; for an atomic potential g, V[g]_pq = sum_A q^pq_A g_A =
1/2 [C^T (G S + S G) C]_pq is its MO matrix, and q[M] are the Mulliken charges of an MO matrix
M.  For a closed shell (singlets; triplets replace gamma by W in the terms with xi)
(DFTB+ getZVectorEqRHS, solveZVectorPrecond, calcWvectorZ, calcPMatrix):

    T_ab = 1/2 [(X+Y)^T (X+Y) + (X-Y)^T (X-Y)]_ab,   T_ij = -1/2 [(X+Y)(X+Y)^T + (X-Y)(X-Y)^T]_ij
    xi = q[X+Y],   R = -4 (X+Y) V[gamma xi]_vv + 4 V[gamma xi]_oo (X+Y) - 4 V[gamma q[T]]_ov
    (A+B) Z = R,   (A+B) Y = w Y + 4 V[gamma q[Y]]_ov        (the singlet A+B also for triplets)
    W_vv = (X+Y)^T e_o (X+Y) + (X-Y)^T e_o (X-Y) + Omega [(X+Y)^T (X-Y) + (X-Y)^T (X+Y)]
    W_oo = -(X+Y) e_v (X+Y)^T - (X-Y) e_v (X-Y)^T + Omega [(X+Y)(X-Y)^T + (X-Y)(X+Y)^T]
           + 4 V[gamma q[T]]_oo + 4 V[gamma q[Z]]_oo
    W_ov = 4 V[gamma xi]_oo (X+Y) + Z e_o,       P = T + 1/2 (Z + Z^T)

With collinear spin polarisation the prefactors 4 become 2 and every gamma term gets its W
partner with the spin signs.  Unlike DFTB+, the Z-vector runs over all occupied-virtual pairs
(DFTB+ only over those inside an energy window), so the gradient stays exact with a window.

The gradient (DFTB+ addGradients) is assembled by the calculator from :func:`pair_terms` (the
coefficients of dH0 and dS at its pair elements) and the gamma-derivative and external terms.

TD-LC (milestone M6h-2): the long-range exchange adds, with the Mulliken exchange operator
K^x (:func:`pydftb.hybrid.four_terms`), R = C_o (X+Y) C_v^T, L = C_o (X-Y) C_v^T and
q+ = C^T K^x[R_s] C, q- = C^T K^x[L_a] C (s, a: symmetric and antisymmetric parts), the orbital
gradient of Omega's exchange -<R_s, K^x R_s> - <L_a, K^x L_a> (X = X+Y, Y = X-Y):

    G_oo = -2 (q+_ov X^T + q-_ov Y^T),   G_vo = -2 (q+_vv X^T + q-_vv Y^T)
    G_ov = -2 (q+_oo X - q-_oo Y),        G_vv = -2 (q+_ov^T X + q-_ov^T Y)

(rhs += G_ov - G_vo^T, W_ov += G_ov, W_oo and W_vv += the symmetric parts of G_oo, G_vv); the
ground state's exchange response -2 K^x[M] to T and Z_s (rhs, W_oo; the exchange in A+B); and
the explicit derivatives of :func:`exchange_terms` through S and gamma_lr.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .backend import ops
from .casida import EXACT_LIMIT, GroundState, _Exchange, transition_charges
from .constants import ELEC_TOL_MAX
from .hybrid import four_terms

Z_RTOL = 1e-14              # Z-vector: |(A+B) Z - R| <= Z_RTOL |R| (conjugate gradients)
DEGENERATE = 1e-6           # Hartree: orbitals this close count as degenerate (window terms)


@dataclass
class Relaxed:
    """The relaxed excited state of one Casida state (atomic units).

    ``p_ao``, ``w_ao``, ``xpy_ao`` per spin channel (N, N): the relaxed difference density P,
    the energy-weighted matrix in the gradient convention (its term is -sum_mu,nu W dS) and
    C (X+Y + (X+Y)^T) C^T; ``dqex`` (n_chan, n_atom) the Mulliken populations of P (excess
    electrons of the excited state per atom and channel); ``xi`` = q[X+Y] summed over the
    channels and ``xi_ds`` their spin difference (spin polarised only); with TD-LC (M6h-2)
    ``xmy_ao`` = C (X-Y - (X-Y)^T) C^T per channel."""
    omega: float
    sym: str                        # "S", "T" or " " (spin polarised)
    p_ao: list
    w_ao: list
    xpy_ao: list
    dqex: np.ndarray
    xi: np.ndarray
    xi_ds: np.ndarray | None
    z_iterations: int               # conjugate-gradient iterations (0: dense solve)
    xmy_ao: list | None = None

    @property
    def dqex_total(self) -> np.ndarray:
        return self.dqex.sum(axis=0)


class _Channel:
    """One spin channel of the ground state with integer occupations."""

    def __init__(self, gs: GroundState, s: int, full: float):
        f = gs.occupations[s]
        occ = np.abs(f - full) < ELEC_TOL_MAX
        if not np.all(occ | (np.abs(f) < ELEC_TOL_MAX)):
            raise ValueError("excited-state properties need integer occupations (as in DFTB+: "
                             "\"Fractional fillings not currently possible for excited state "
                             "property calculations\")")
        n_o = int(np.count_nonzero(occ))
        if not np.all(occ[:n_o]):
            raise ValueError("the occupied orbitals are not the lowest ones")
        self.n_o, self.n_v = n_o, len(f) - n_o
        self.c, self.sc, self.eps = gs.coeffs[s], gs.sc[s], gs.eps[s]
        self.c_o, self.c_v = self.c[:, :n_o], self.c[:, n_o:]
        self.sc_o, self.sc_v = self.sc[:, :n_o], self.sc[:, n_o:]
        self.e_o, self.e_v = self.eps[:n_o], self.eps[n_o:]
        self.w = self.e_v[None, :] - self.e_o[:, None]          # (n_o, n_v) single-particle w_ia
        self.sigma = 1.0 - 2.0 * s


def _segment(gs: GroundState, per_orbital) -> np.ndarray:
    return ops.segment_sum(per_orbital, gs.orb_atom, len(gs.positions))


def mulliken(gs: GroundState, d) -> np.ndarray:
    """Mulliken populations per atom of a symmetric AO matrix, sum_{mu in A} (D S)_mu,mu."""
    return _segment(gs, (d * gs.overlap).sum(axis=1))


def _q_ov(gs: GroundState, ch: _Channel, y) -> np.ndarray:
    """sum_ia Y_ia q^ia_A of an (n_occ, n_virt) matrix Y."""
    rows = (((ch.c_o @ y) * ch.sc_v).sum(axis=1) + ((ch.c_v @ y.T) * ch.sc_o).sum(axis=1))
    return 0.5 * _segment(gs, rows)


def _v_mo(gs: GroundState, ch: _Channel, g_atom) -> np.ndarray:
    """V[g] = 1/2 C^T (G S + S G) C (N, N) of an atomic potential g."""
    a = (g_atom[gs.orb_atom][:, None] * ch.c).T @ ch.sc
    return 0.5 * (a + a.T)


def _v_ov(gs: GroundState, ch: _Channel, g_atom) -> np.ndarray:
    """The occupied-virtual block of V[g], (n_occ, n_virt)."""
    g = g_atom[gs.orb_atom][:, None]
    return 0.5 * ((g * ch.c_o).T @ ch.sc_v + (g * ch.sc_o).T @ ch.c_v)


class _APlusB:
    """The singlet A+B on all occupied-virtual pairs of the channels: w Y + pref V[gamma q[Y]]
    (+ pref s V[W q_s[Y]] with spin polarisation), pref = 4 (closed shell) or 2; with TD-LC
    (M6h-2) also the exchange -[(ij|ab) + (ib|ja)] Y."""

    def __init__(self, gs: GroundState, chans: list):
        self.gs, self.chans = gs, chans
        self.spin = len(chans) == 2
        self.pref = 2.0 if self.spin else 4.0
        self.sizes = [ch.n_o * ch.n_v for ch in chans]
        self.w = np.concatenate([ch.w.ravel() for ch in chans])
        self.exchange = None
        if gs.lr_gamma is not None:
            self.exchange = _Exchange(gs, self._pairs(), np.ones(len(self.w)))

    def split(self, y):
        out, start = [], 0
        for ch, n in zip(self.chans, self.sizes):
            out.append(y[start:start + n].reshape(ch.n_o, ch.n_v))
            start += n
        return out

    def charges(self, ys):
        """q[Y] summed over the channels and (spin polarised) their signed sum."""
        qs = [_q_ov(self.gs, ch, y) for ch, y in zip(self.chans, ys)]
        q = qs[0] if len(qs) == 1 else qs[0] + qs[1]
        return q, (qs[0] - qs[1] if self.spin else None)

    def product(self, y):
        gs = self.gs
        ys = self.split(y)
        q, qds = self.charges(ys)
        out = []
        for ch, yy in zip(self.chans, ys):
            v = self.pref * _v_ov(gs, ch, gs.gamma @ q)
            if self.spin:
                v = v + self.pref * ch.sigma * _v_ov(gs, ch, gs.spin_w * qds)
            out.append((ch.w * yy + v).ravel())
        out = np.concatenate(out)
        if self.exchange is not None:
            out = out + self.exchange.product(y, 1.0)
        return out

    def _pairs(self) -> np.ndarray:
        """(i, a, spin) of all pairs (orbital indices), in the order of the flat vectors."""
        pairs = []
        for s, ch in enumerate(self.chans):
            i, a = np.divmod(np.arange(ch.n_o * ch.n_v), ch.n_v)
            pairs.append(np.stack([i, a + ch.n_o, np.full(len(i), s)], axis=1))
        return np.concatenate(pairs)

    def _transition_charges(self, chunk=None):
        """q^ia_A of all pairs (n_pair, n_atom), in the order of the flat vectors."""
        pairs = self._pairs()
        if chunk is None:
            return transition_charges(self.gs, pairs)
        return (transition_charges(self.gs, pairs[k:k + chunk])
                for k in range(0, len(pairs), chunk))

    def matrix(self) -> np.ndarray:
        gs = self.gs
        q = self._transition_charges()
        m = self.pref * (q @ gs.gamma @ q.T)
        if self.spin:
            sigma = np.concatenate([np.full(n, ch.sigma) for ch, n in zip(self.chans, self.sizes)])
            m += self.pref * np.outer(sigma, sigma) * ((q * gs.spin_w) @ q.T)
        m[np.diag_indices_from(m)] += self.w
        if self.exchange is not None:
            kd, kx = self.exchange.dense()
            m = m - (kd + kx)
        return 0.5 * (m + m.T)

    def diagonal(self, chunk: int = 4096) -> np.ndarray:
        """The diagonal of A+B (DFTB+'s preconditioner), computed in chunks of pairs."""
        gs = self.gs
        parts = []
        for q in self._transition_charges(chunk):
            d = self.pref * ((q @ gs.gamma) * q).sum(axis=1)
            if self.spin:
                d = d + self.pref * ((q * q) * gs.spin_w).sum(axis=1)
            parts.append(d)
        if self.exchange is not None:
            return np.concatenate(parts) + self.w + self.exchange.diagonal()
        return np.concatenate(parts) + self.w

    def solve(self, rhs):
        """Z with (A+B) Z = rhs: dense up to EXACT_LIMIT pairs, else conjugate gradients
        preconditioned with the diagonal (DFTB+ solveZVectorPrecond)."""
        if len(rhs) <= EXACT_LIMIT:
            return sla.solve(self.matrix(), rhs, assume_a="pos"), 0
        return _pcg(self.product, rhs, 1.0 / self.diagonal())


def _pcg(apply, b, prec, rtol: float = Z_RTOL, max_iter: int = 10000):
    """Preconditioned conjugate gradients for the symmetric positive definite operator;
    restarted from the true residual until it is below rtol |b|."""
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        return np.zeros_like(b), 0
    x = prec * b
    it = 0
    for _ in range(5):                       # restarts with the true residual
        r = b - apply(x)
        if np.linalg.norm(r) <= rtol * bnorm:
            return x, it
        z = prec * r
        p = z.copy()
        rz = r @ z
        while it < max_iter:
            ap = apply(p)
            alpha = rz / (p @ ap)
            x = x + alpha * p
            r = r - alpha * ap
            it += 1
            if np.linalg.norm(r) <= 0.1 * rtol * bnorm:
                break
            z = prec * r
            rz_new = r @ z
            p = z + (rz_new / rz) * p
            rz = rz_new
    r = b - apply(x)
    if np.linalg.norm(r) > 10.0 * rtol * bnorm:
        raise RuntimeError(f"Z-vector: no convergence in {it} iterations "
                           f"(|r|/|R| = {np.linalg.norm(r) / bnorm:.1e})")
    return x, it


def _state_vectors(x, k: int, chans: list):
    """X+Y and X-Y of state k (0-based) as (n_occ, n_virt) matrices per channel (zero for the
    pairs outside the windows)."""
    xp = [np.zeros((ch.n_o, ch.n_v)) for ch in chans]
    xm = [np.zeros((ch.n_o, ch.n_v)) for ch in chans]
    for (i, a, s), p, m in zip(x.pairs, x.xpy[:, k], x.xmy[:, k]):
        ch = chans[s]
        xp[s][i, a - ch.n_o] = p
        xm[s][i, a - ch.n_o] = m
    return xp, xm


def relax(gs: GroundState, x, state: int, window_only: bool = False) -> Relaxed:
    """The relaxed difference density and the energy-weighted matrix of ``state`` (1-based)
    of the excitations ``x`` (:class:`pydftb.casida.Excitations` of ``gs``).

    ``window_only`` solves the Z-vector on the pairs inside the energy window only, as DFTB+
    does (test use: it reproduces DFTB+'s gradient, which is then not that of E_0 + Omega)."""
    gs.prepare()
    spin = gs.spin_polarised
    chans = [_Channel(gs, s, 1.0 if spin else 2.0) for s in range(len(gs.coeffs))]
    k = state - 1
    omega = float(np.sqrt(x.omega2[k]))
    sym = " " if spin else x.symmetries[k]
    xp, xm = _state_vectors(x, k, chans)
    apb = _APlusB(gs, chans)
    pref = apb.pref
    gamma = gs.gamma
    if sym == "T":
        kernel = lambda q: gs.spin_w * q                       # noqa: E731
    else:
        kernel = lambda q: gamma @ q                           # noqa: E731

    # unrelaxed difference density T and the MO potential V of the kernel term (X+Y coupled
    # through gamma or W): Omega = sum T F + (c/2) xi.g.xi, c = 4 (closed shell) or 2
    xi, xi_ds = apb.charges(xp)
    ck = 2.0 if spin else 4.0
    lc = apb.exchange is not None
    if lc:                       # TD-LC (M6h-2): K^x, q+ = C^T K^x[R_s] C, q- = C^T K^x[L_a] C
        s_ao, go = gs.overlap, apb.exchange.go
        kx = lambda m: 0.25 * four_terms(s_ao, go, m)          # noqa: E731
        qps, qms = [], []
    t, vk = [], []
    for ch, p, m in zip(chans, xp, xm):
        tm = np.zeros((ch.n_o + ch.n_v,) * 2)
        tm[:ch.n_o, :ch.n_o] = -0.5 * (p @ p.T + m @ m.T)
        tm[ch.n_o:, ch.n_o:] = 0.5 * (p.T @ p + m.T @ m)
        t.append(tm)
        if spin:
            vk.append(_v_mo(gs, ch, gamma @ xi) + ch.sigma * _v_mo(gs, ch, gs.spin_w * xi_ds))
        else:
            vk.append(_v_mo(gs, ch, kernel(xi)))
        if lc:
            r, l_ = ch.c_o @ p @ ch.c_v.T, ch.c_o @ m @ ch.c_v.T
            qps.append(ch.c.T @ kx(0.5 * (r + r.T)) @ ch.c)
            qms.append(ch.c.T @ kx(0.5 * (l_ - l_.T)) @ ch.c)
    windowed = len(x.pairs) < sum(apb.sizes)
    if windowed and not window_only:
        # with an energy window Omega is not invariant under rotations among the occupied
        # (or the virtual) orbitals: the canonical orbitals' response enters through
        # multipliers z of F_pq = 0 there, which act like T (T -> T + z/2)
        t = [tt + 0.5 * _canonical_multipliers(ch, tt, v, p, ck)
             for ch, tt, v, p in zip(chans, t, vk, xp)]

    # the kernel terms of the right-hand side and of W, and the rest of W without the response
    rhs, w_ov, w_oo, w_vv = [], [], [], []
    for k_ch, (ch, tt, v, p, m) in enumerate(zip(chans, t, vk, xp, xm)):
        n_o = ch.n_o
        v_oo, v_ov, v_vv = v[:n_o, :n_o], v[:n_o, n_o:], v[n_o:, n_o:]
        rhs.append(-ck * p @ v_vv + ck * v_oo @ p)
        w_ov.append(ck * v_oo @ p)
        if window_only:
            # DFTB+: W_ov on the window's pairs only, and W_oo, W_vv in a form that assumes
            # the Casida equations on every pair
            w_ov[-1] = w_ov[-1] * _window_mask(x, chans, chans.index(ch))
            cross = p @ m.T
            w_oo.append(-(p * ch.e_v) @ p.T - (m * ch.e_v) @ m.T + omega * (cross + cross.T))
            cross = p.T @ m
            w_vv.append((p.T * ch.e_o) @ p + (m.T * ch.e_o) @ m + omega * (cross + cross.T))
            continue
        # W = 1/2 (G + G^T) of the orbital gradient G of sum T F + (c/2) xi.g.xi
        t_oo, t_vv = tt[:n_o, :n_o], tt[n_o:, n_o:]
        g = v_ov @ p.T
        w_oo.append(ch.e_o[:, None] * t_oo + t_oo * ch.e_o[None, :] + 0.5 * ck * (g + g.T))
        g = v_ov.T @ p
        w_vv.append(ch.e_v[:, None] * t_vv + t_vv * ch.e_v[None, :] + 0.5 * ck * (g + g.T))
        if lc:                   # the exchange's orbital gradient (module docstring)
            qp, qm = qps[k_ch], qms[k_ch]
            g_oo = -2.0 * (qp[:n_o, n_o:] @ p.T + qm[:n_o, n_o:] @ m.T)
            g_vo = -2.0 * (qp[n_o:, n_o:] @ p.T + qm[n_o:, n_o:] @ m.T)
            g_ov = -2.0 * (qp[:n_o, :n_o] @ p - qm[:n_o, :n_o] @ m)
            g_vv = -2.0 * (qp[:n_o, n_o:].T @ p + qm[:n_o, n_o:].T @ m)
            rhs[-1] = rhs[-1] + g_ov - g_vo.T
            w_ov[-1] = w_ov[-1] + g_ov
            w_oo[-1] = w_oo[-1] + 0.5 * (g_oo + g_oo.T)
            w_vv[-1] = w_vv[-1] + 0.5 * (g_vv + g_vv.T)

    # the ground state's response to T (H+[T], always with gamma)
    q_t = [mulliken(gs, ch.c @ tt @ ch.c.T) for ch, tt in zip(chans, t)]
    qt = q_t[0] if not spin else q_t[0] + q_t[1]
    qt_ds = q_t[0] - q_t[1] if spin else None
    for s, ch in enumerate(chans):
        v = _v_mo(gs, ch, gamma @ qt)
        if spin:
            v = v + ch.sigma * _v_mo(gs, ch, gs.spin_w * qt_ds)
        rhs[s] = rhs[s] - (pref * v[:ch.n_o, ch.n_o:])
        w_oo[s] = w_oo[s] + pref * v[:ch.n_o, :ch.n_o]
        if lc:                   # and its exchange response -2 K^x[T]
            kt = ch.c.T @ kx(ch.c @ t[s] @ ch.c.T) @ ch.c
            rhs[s] = rhs[s] + 2.0 * kt[:ch.n_o, ch.n_o:]
            w_oo[s] = w_oo[s] - 2.0 * kt[:ch.n_o, :ch.n_o]

    # Z-vector
    r = np.concatenate([v.ravel() for v in rhs])
    if window_only:
        z, n_it = _window_solve(apb, r, x, chans)
    else:
        z, n_it = apb.solve(r)
    zs = apb.split(z)
    qz, qz_ds = apb.charges(zs)
    p_ao, w_ao, xpy_ao, dqex = [], [], [], []
    xmy_ao = [] if lc else None
    for s, (ch, zz) in enumerate(zip(chans, zs)):
        v = _v_mo(gs, ch, gamma @ qz)
        if spin:
            v = v + ch.sigma * _v_mo(gs, ch, gs.spin_w * qz_ds)
        woo = w_oo[s] + pref * v[:ch.n_o, :ch.n_o]
        if lc:                   # the exchange response -2 K^x[Z_s]
            zao = ch.c_o @ zz @ ch.c_v.T
            woo = woo - 2.0 * (ch.c_o.T @ kx(0.5 * (zao + zao.T)) @ ch.c_o)
        wov = w_ov[s] + zz * ch.e_o[:, None]
        n = ch.n_o + ch.n_v
        wm = np.zeros((n, n))
        wm[:ch.n_o, :ch.n_o] = woo
        wm[ch.n_o:, ch.n_o:] = w_vv[s]
        wm[:ch.n_o, ch.n_o:] = wov
        wm[ch.n_o:, :ch.n_o] = wov.T
        pm = t[s].copy()
        pm[:ch.n_o, ch.n_o:] += 0.5 * zz
        pm[ch.n_o:, :ch.n_o] += 0.5 * zz.T
        xm_full = np.zeros((n, n))
        xm_full[:ch.n_o, ch.n_o:] = xp[s]
        p_ao.append(ch.c @ pm @ ch.c.T)
        w_ao.append(0.5 * (ch.c @ wm @ ch.c.T))          # DFTB+ wcc / 2: the -sum W dS form
        xpy_ao.append(ch.c @ (xm_full + xm_full.T) @ ch.c.T)
        dqex.append(mulliken(gs, p_ao[-1]))
        if lc:
            ym = np.zeros((n, n))
            ym[:ch.n_o, ch.n_o:] = xm[s]
            xmy_ao.append(ch.c @ (ym - ym.T) @ ch.c.T)
    return Relaxed(omega, sym, p_ao, w_ao, xpy_ao, np.array(dqex), xi, xi_ds, n_it, xmy_ao)


def exchange_terms(rel: Relaxed, dps) -> list:
    """The explicit exchange terms of dOmega/dR for TD-LC (M6h-2), as [(c, M1, M2)] for
    :meth:`pydftb.hybrid.HybridTerms.exchange_force`: -1/2 B(P, DeltaP) (closed shell, both
    total; -B(P_s, DeltaP_s) per spin channel with spin polarisation), -B(R_s, R_s) and
    -B(L_a, L_a) with R_s = xpy_ao / 2 and L_a = xmy_ao / 2.  ``dps``: the ground state's
    DeltaP_out per channel."""
    half = 0.5 if len(rel.p_ao) == 1 else 1.0
    out = []
    for p, dp, xp, xm in zip(rel.p_ao, dps, rel.xpy_ao, rel.xmy_ao):
        out += [(-half, p, dp), (-0.25, xp, xp), (-0.25, xm, xm)]
    return out


def _canonical_multipliers(ch: _Channel, t, v, xp, ck: float) -> np.ndarray:
    """Multipliers z of the canonical conditions F_pq = 0 (p != q, both occupied or both
    virtual) for an energy window: z_pq = -(G_pq - G_qp) / (e_p - e_q) with the orbital
    gradient G = 2 e_p T_pq + c (V_ov (X+Y)^T | V_ov^T (X+Y)) of sum T F + (c/2) xi.g.xi
    (zero without a window: Omega is then invariant under these rotations).  (N, N),
    symmetric, zero in the occupied-virtual block and between (nearly) degenerate orbitals."""
    n_o = ch.n_o
    z = np.zeros_like(t)
    v_ov = v[:n_o, n_o:]
    blocks = ((slice(0, n_o), ch.e_o, v_ov @ xp.T), (slice(n_o, None), ch.e_v, v_ov.T @ xp))
    for blk, e, gk in blocks:
        de = e[:, None] - e[None, :]
        g = 2.0 * de * t[blk, blk] + ck * (gk - gk.T)
        ok = np.abs(de) > DEGENERATE
        z[blk, blk] = np.where(ok, -g / np.where(ok, de, 1.0), 0.0)
    return z


def _window_mask(x, chans, s: int) -> np.ndarray:
    """(n_occ, n_virt) bool: the pairs of channel s inside the windows."""
    ch = chans[s]
    mask = np.zeros((ch.n_o, ch.n_v), bool)
    sel = x.pairs[:, 2] == s
    mask[x.pairs[sel, 0], x.pairs[sel, 1] - ch.n_o] = True
    return mask


def _window_solve(apb: _APlusB, r, x, chans):
    """The Z-vector on the window's pairs only (DFTB+), zero elsewhere."""
    keep = []
    offsets = np.concatenate([[0], np.cumsum(apb.sizes)])
    for i, a, s in x.pairs:
        keep.append(offsets[s] + i * chans[s].n_v + (a - chans[s].n_o))
    keep = np.sort(np.array(keep, int))
    m = apb.matrix()[np.ix_(keep, keep)]
    z = np.zeros_like(r)
    z[keep] = sla.solve(m, r[keep], assume_a="pos")
    return z, 0


def pair_terms(rel: Relaxed, gs: GroundState, rows, cols, v_orb, vs_orb=None):
    """Coefficients of dH0 and dS at the pair elements (rows, cols) for dOmega/dR, in the
    calculator's convention (both orderings of every atom pair are elements):

        p_e = P,   c_e = P (V_mu + V_nu)/2 - W + (V^ex_mu + V^ex_nu)/2 D_0
                         + (h_mu + h_nu) (X+Y)_AO
                         [+ (V^s_mu + V^s_nu)/2 (P_up - P_down)
                          + (W dm^ex_mu + W dm^ex_nu)/2 (D_up - D_down)]

    V is the ground state's charge potential per orbital (``v_orb``, with the external
    potentials), V^ex = gamma dq^ex, h = gamma xi (singlets), W xi (triplets) or
    1/2 (gamma xi + s W xi_ds) per channel, V^s the ground state's spin potential (``vs_orb``)
    and dm^ex the spin difference of dq^ex (DFTB+ addGradients: tmp1 ... tmp10)."""
    oa = gs.orb_atom
    spin = gs.spin_polarised
    p = rel.p_ao[0] if not spin else rel.p_ao[0] + rel.p_ao[1]
    w = rel.w_ao[0] if not spin else rel.w_ao[0] + rel.w_ao[1]
    dens = [(c * f) @ c.T for c, f in zip(gs.coeffs, gs.occupations)]
    d0 = dens[0] if not spin else dens[0] + dens[1]
    p_e = p[cols, rows]
    coef = p_e * (0.5 * (v_orb[rows] + v_orb[cols])) - w[cols, rows]
    vex = (gs.gamma @ rel.dqex_total)[oa]
    coef = coef + 0.5 * (vex[rows] + vex[cols]) * d0[cols, rows]
    for s, xa in enumerate(rel.xpy_ao):
        if spin:
            sigma = 1.0 - 2.0 * s
            h = 0.5 * (gs.gamma @ rel.xi + sigma * gs.spin_w * rel.xi_ds)
        elif rel.sym == "T":
            h = gs.spin_w * rel.xi
        else:
            h = gs.gamma @ rel.xi
        h = h[oa]
        coef = coef + (h[rows] + h[cols]) * xa[cols, rows]
    if spin:
        pm = rel.p_ao[0] - rel.p_ao[1]
        coef = coef + 0.5 * (vs_orb[rows] + vs_orb[cols]) * pm[cols, rows]
        wd = (gs.spin_w * (rel.dqex[0] - rel.dqex[1]))[oa]
        dm = dens[0] - dens[1]
        coef = coef + 0.5 * (wd[rows] + wd[cols]) * dm[cols, rows]
    return p_e, coef


def _fortran_e(x: float, width: int = 20, digits: int = 12) -> str:
    """Fortran's Ew.d edit descriptor (0.dddE+xx)."""
    if x == 0.0:
        return f"{'0.' + '0' * digits + 'E+00':>{width}s}"
    exp = int(np.floor(np.log10(abs(x)))) + 1
    mant = abs(x) / 10.0 ** exp
    s = f"{mant:.{digits}f}"
    if s.startswith("1."):                     # rounded up to 1.000...: renormalise
        exp += 1
        s = f"{mant / 10.0:.{digits}f}"
    body = f"{'-' if x < 0 else ''}0{s[1:]}E{'+' if exp >= 0 else '-'}{abs(exp):02d}"
    return f"{body:>{width}s}"


def mulliken_tables(net_charges, dipole, sym: str, state: int) -> tuple[str, str]:
    """The excited state's XCH.DAT and XREST.DAT as DFTB+ writes them (linrespcommon
    writeExcMulliken): net charges per atom (e, i.e. q0 - excited populations) and the dipole
    (e Bohr, printed in Debye with its norm).  ``sym``: "S", "T" or " " (spin polarised)."""
    from .casida import AU_DEBYE
    q = np.asarray(net_charges, float)
    head = f"# MULLIKEN CHARGES of excited state {sym}{state:2d}"
    xch = [head, f"#  Natoms ={len(q):4d}", "# Atom        netCharge", "#" + "=" * 41]
    xch += [f"{k:5d} {v:16.8f}" for k, v in enumerate(q, 1)]
    d = np.asarray(dipole, float) * AU_DEBYE
    xrest = [f"Mulliken analysis of excited state {sym}{state:2d}", "=" * 42, " ",
             "Mulliken exc. state dipole moment [Debye]", "=" * 42,
             "".join(f"{x:14.8f}" for x in d), " ",
             "Norm of exc. state dipole moment [Debye]", "=" * 42,
             _fortran_e(float(np.linalg.norm(d))), ""]
    return "\n".join(xch) + "\n", "\n".join(xrest) + "\n"


def kernel_gradient_factor(rel: Relaxed) -> float:
    """Factor c of the gamma-derivative term c d/dR (1/2 xi.gamma.xi) of Omega: 4 for closed-
    shell singlets, 0 for triplets (W is a constant), 2 with spin polarisation."""
    if rel.sym == " ":
        return 2.0
    return 4.0 if rel.sym == "S" else 0.0
