"""Third-order (DFTB3) charge term, after DFTB+ ``dftb/thirdorder.F90``.

    E3 = 1/3 sum_ab Gamma_ab dQ_A(a) dq_a dq_b,    Gamma_ab = dgamma_ab/dU_a * Ud_a

a, b are charge sites (atoms, or shells with orbital-resolved SCC), dQ_A the
excess electrons of the atom that carries site a and Ud the Hubbard
derivative.  Only the short-range part S of gamma depends on U, so Gamma
decays exponentially and is summed in real space (no Ewald sum).

Like DFTB+ (and the original DFTB3 code the 3ob set was fitted with) Gamma is
the *partial* derivative d/dU_a when U_a != U_b, but the *total* derivative of
S(tau, tau) when |U_a - U_b| < MIN_HUB_DIFF, which is twice as large: Gamma
of a pair doubles when its two U values become equal.

The H-X damping (DFTB+ ``HCorrection = Damping``) multiplies S by
h = exp(-((U_a + U_b)/2)^zeta r^2) for pairs that involve a damped atom; it
enters Gamma through dS/dU and dh/dU (and gamma itself in coulomb.py).
"""
from __future__ import annotations

import numpy as np

from .backend import ops
from .constants import MIN_HUB_DIFF
from .coulomb import NEAR_TAU, Coulomb, _short, _short_near

MIN_SHORT_GAMMA = 1e-10     # DFTB+ minShortGamma: S below this is cut off
TOL_SHORT_GAMMA = 1e-10     # DFTB+ tolShortGamma: bisection tolerance


# ---- DFTB+ thirdorder.F90 helper functions (tau = 3.2 U, r > 0) ----------
def _ff(ta, tb, r):
    d2 = ta * ta - tb * tb
    return 0.5 * ta * tb ** 4 / d2 ** 2 - (tb ** 6 - 3.0 * ta * ta * tb ** 4) / (d2 ** 3 * r)


def _fpT1(ta, tb, r):                       # d f(ta, tb, r) / d ta
    d2 = ta * ta - tb * tb
    return -0.5 * (tb ** 6 + 3.0 * ta * ta * tb ** 4) / d2 ** 3 - 12.0 * ta ** 3 * tb ** 4 / (d2 ** 4 * r)


def _fpT2(tb, ta, r):                       # d f(tb, ta, r) / d ta
    d2 = tb * tb - ta * ta
    return 2.0 * tb ** 3 * ta ** 3 / d2 ** 3 + 12.0 * tb ** 4 * ta ** 3 / (d2 ** 4 * r)


def _fpR(ta, tb, r):
    return (tb ** 6 - 3.0 * tb ** 4 * ta * ta) / (r * r * (ta * ta - tb * tb) ** 3)


def _fpT1pR(ta, tb, r):
    return 12.0 * ta ** 3 * tb ** 4 / (r * r * (ta * ta - tb * tb) ** 4)


def _fpT2pR(tb, ta, r):
    return -12.0 * ta ** 3 * tb ** 4 / (r * r * (ta * ta - tb * tb) ** 4)


def _gg(t, r):
    return (48.0 + 33.0 * t * r + 9.0 * t * t * r * r + t ** 3 * r ** 3) / (48.0 * r)


def _gpT(t, r):
    return (33.0 + 18.0 * t * r + 3.0 * t * t * r * r) / 48.0


def _gpR(t, r):
    return (-48.0 + 9.0 * (t * r) ** 2 + 2.0 * (t * r) ** 3) / (48.0 * r * r)


def _gpTpR(t, r):
    return (3.0 * t + t * t * r) / 8.0


def _short_terms(ta, tb, r, same):
    """S, dS/dtau_a, dS/dr, d2S/dtau_a dr (DFTB+ short_1/2, shortpT_*, shortpR_*, shortpTpR_*).

    For ``same`` pairs (equal U) the tau derivatives are taken w.r.t. the common tau.
    Nearly equal tau use the Hermite-Genocchi form (:func:`pydftb.coulomb._short_near`):
    there DFTB+'s closed form loses all digits of dS/dtau_a.
    """
    s, st, sr, str_ = (np.empty(r.shape) for _ in range(4))
    if same.any():
        t, x = 0.5 * (ta[same] + tb[same]), r[same]
        e = np.exp(-t * x)
        g, gt, gr, gtr = _gg(t, x), _gpT(t, x), _gpR(t, x), _gpTpR(t, x)
        s[same] = e * g
        st[same] = e * (gt - x * g)
        sr[same] = e * (gr - t * g)
        str_[same] = e * ((t * x - 1.0) * g - t * gt + gtr - x * gr)
    near = ~same & (np.abs(ta - tb) < NEAR_TAU * 0.5 * (ta + tb))
    if near.any():
        s[near], sr[near], st[near], str_[near] = _short_near(r[near], ta[near], tb[near],
                                                              tau_derivs=True)
    nd = ~same & ~near
    if nd.any():
        a, b, x = ta[nd], tb[nd], r[nd]
        ea, eb = np.exp(-a * x), np.exp(-b * x)
        fab, fba = _ff(a, b, x), _ff(b, a, x)
        s[nd] = ea * fab + eb * fba
        st[nd] = ea * (_fpT1(a, b, x) - x * fab) + eb * _fpT2(b, a, x)
        sr[nd] = ea * (_fpR(a, b, x) - a * fab) + eb * (_fpR(b, a, x) - b * fba)
        str_[nd] = (ea * (fab * (a * x - 1.0) - a * _fpT1(a, b, x) + _fpT1pR(a, b, x)
                          - x * _fpR(a, b, x))
                    + eb * (_fpT2pR(b, a, x) - b * _fpT2(b, a, x)))
    return s, st, sr, str_


def dgamma_du(ua, ub, r, damped=False, zeta=0.0):
    """dgamma_ab/dU_a and d2gamma_ab/dU_a dr for r > 0 (DFTB+ gamma2pU, gamma2pUpR)."""
    ua, ub, r, damped = np.broadcast_arrays(np.asarray(ua, float), np.asarray(ub, float),
                                            np.asarray(r, float), np.asarray(damped, bool))
    same = np.abs(ua - ub) < MIN_HUB_DIFF
    s, st, sr, str_ = _short_terms(3.2 * ua, 3.2 * ub, r, same)
    g = -3.2 * st
    gr = -3.2 * str_
    if damped.any():
        m = damped
        u, x = 0.5 * (ua[m] + ub[m]), r[m]
        h = np.exp(-u ** zeta * x * x)
        h_u = -0.5 * zeta * x * x * u ** (zeta - 1.0) * h
        h_r = -2.0 * x * u ** zeta * h
        h_ur = zeta * x * u ** (zeta - 1.0) * (x * x * u ** zeta - 1.0) * h
        gr[m] = gr[m] * h - 3.2 * st[m] * h_r - sr[m] * h_u - s[m] * h_ur
        g[m] = g[m] * h - s[m] * h_u
    return g, gr


def dgamma_du_onsite(ua, ub):
    """dgamma_ab/dU_a for two sites on one atom (R = 0); DFTB+ dGdUr0."""
    ua, ub = np.asarray(ua, float), np.asarray(ub, float)
    ta, tb = 3.2 * ua, 3.2 * ub
    inv = 1.0 / (ta + tb)
    general = 1.6 * inv * (tb + inv * (-ta * tb + inv * (2.0 * ta * tb * tb
                                                          + inv * (-3.0 * ta * ta * tb * tb))))
    return np.where(np.abs(ua - ub) < MIN_HUB_DIFF, 0.5, general)


def short_cutoff_dftbplus(u2: float, u1: float) -> float:
    """DFTB+ ``expGammaCutoff``: where S drops below 1e-10 (doubling, then bisection)."""
    def s(r):
        return float(_short(np.array([r]), np.array([3.2 * u2]), np.array([3.2 * u1]))[0][0])
    rab = 1.0
    while s(rab) > MIN_SHORT_GAMMA:
        rab *= 2.0
    min_cut, max_cut = rab + 0.1, 0.5 * rab - 0.1
    decreasing = s(max_cut) >= s(min_cut)
    lower = s(min_cut)
    cut = max_cut + 0.1
    gamma = s(cut)
    while gamma - lower > TOL_SHORT_GAMMA:
        mid = 0.5 * (cut + min_cut)
        if decreasing == (MIN_SHORT_GAMMA >= s(mid)):
            min_cut, lower = mid, s(mid)
        else:
            cut, gamma = mid, s(mid)
    return min_cut


class ThirdOrder:
    """Third-order term for a fixed geometry (DFTB+ ``ThirdOrderFull``).

    Parameters
    ----------
    coulomb : the :class:`~pydftb.coulomb.Coulomb` of the same calculation; its
              sites, Hubbard U, neighbour list and H-X damping are reused.
    hubbard_deriv : (n_site,) Hubbard derivative Ud of every site.
    onsite_only : keep only the R = 0 terms, E3 = sum_A Ud_A/6 dQ_A^3 (DFTB+
                  ``ThirdOrder``, atom-resolved SCC only).
    """

    def __init__(self, coulomb: Coulomb, hubbard_deriv, onsite_only: bool = False):
        c = coulomb
        self.site_atom = c.site_atom
        self.natom, self.nsite = c.natom, c.nsite
        self.periodic = c.periodic
        u = c.tau / 3.2
        ud = np.asarray(hubbard_deriv, float)
        # on-site block: every pair of sites on one atom (including a = b)
        sites = c.atom_sites
        pa = np.repeat(sites, sites.shape[1], axis=1).ravel()
        pb = np.tile(sites, (1, sites.shape[1])).ravel()
        ok = (pa >= 0) & (pb >= 0)
        pa, pb = pa[ok], pb[ok]
        mat = np.zeros((self.nsite, self.nsite))
        mat[pa, pb] = dgamma_du_onsite(u[pa], u[pb]) * ud[pa]
        self.nl = None
        if not onsite_only:
            # DFTB+ cuts atom pairs at the S < 1e-10 distance of the two atoms' smallest U
            umin = np.full(self.natom, np.inf)
            np.minimum.at(umin, self.site_atom, u)
            uniq, inv = np.unique(umin, return_inverse=True)
            table = np.array([[short_cutoff_dftbplus(b, a) for b in uniq] for a in uniq])
            i, j, shift, vec, d = c.nl
            keep = d < table[inv[i], inv[j]]
            self.nl = (i[keep], j[keep], shift[keep], vec[keep], d[keep])
            i, j, d = self.nl[0], self.nl[1], self.nl[4]
            sa = sites[i][:, :, None]
            sb = sites[j][:, None, :]
            p, x, y = np.nonzero((sa >= 0) & (sb >= 0))
            self.sp_pair = p
            self.sp_a, self.sp_b = sites[i[p], x], sites[j[p], y]
            damped = (c.damped[i] | c.damped[j])[p]
            g_ab, self.gr_ab = dgamma_du(u[self.sp_a], u[self.sp_b], d[p], damped, c.damp_exp)
            _, self.gr_ba = dgamma_du(u[self.sp_b], u[self.sp_a], d[p], damped, c.damp_exp)
            self.gr_ab = self.gr_ab * ud[self.sp_a]
            self.gr_ba = self.gr_ba * ud[self.sp_b]
            mat = ops.scatter_add(mat, (self.sp_a, self.sp_b), g_ab * ud[self.sp_a])
        self.gamma3 = mat

    def energy_shift(self, dq):
        """E3 and its potential dE3/dq per site."""
        dq_atom = ops.segment_sum(dq, self.site_atom, self.natom)
        qa = dq_atom[self.site_atom]
        gq = self.gamma3 @ dq
        e = (qa * dq * gq).sum() / 3.0
        v = (qa * gq + self.gamma3.T @ (qa * dq)
             + ops.segment_sum(dq * gq, self.site_atom, self.natom)[self.site_atom]) / 3.0
        return e, v

    def gradient(self, dq, virial: bool = False):
        """dE3/dR per atom at fixed charges and, with ``virial=True``, dE3/d(strain)."""
        grad = np.zeros((self.natom, 3))
        if self.nl is None:                      # on-site terms do not depend on R
            return grad, (np.zeros((3, 3)) if virial else None)
        i, _, _, vec, d = self.nl
        dq_atom = ops.segment_sum(dq, self.site_atom, self.natom)
        qa, qb = dq[self.sp_a], dq[self.sp_b]
        g = qa * qb * (dq_atom[self.site_atom[self.sp_a]] * self.gr_ab
                       + dq_atom[self.site_atom[self.sp_b]] * self.gr_ba) / 3.0
        dedr = ops.segment_sum(g, self.sp_pair, len(i))   # both orderings of each site pair
        w = None
        if virial:
            w = 0.5 * np.einsum("p,pa,pb->ab", dedr / d, vec, vec)
        return ops.scatter_add(grad, i, -(dedr / d)[:, None] * vec), w
