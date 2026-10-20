"""Second-order (gamma) Coulomb kernel of SCC-DFTB, cluster and periodic.

    gamma_AB = sum_T [ 1/|R_AB + T|  -  S(tau_A, tau_B, |R_AB + T|) ],  tau = 16/5 U

The long-range 1/R lattice sum is done with Ewald summation; the short-range
correction S decays exponentially and is summed in real space.  "Sites" are
atoms (standard SCC) or shells (orbital-resolved SCC).
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

from .backend import ops
from .constants import MIN_HUB_DIFF
from .neighbors import neighbor_list

EWALD_TOL = 1e-12
SHORT_TOL = 1e-14
# DFTB+ treats U_a, U_b as equal if |U_a - U_b| < MIN_HUB_DIFF; in tau = 3.2 U:
SAME_TAU = 3.2 * MIN_HUB_DIFF


def _short(r, ta, tb):
    """Short-range S(r) and dS/dr, vectorised; r > 0."""
    s = np.zeros_like(r)
    ds = np.zeros_like(r)
    same = np.abs(ta - tb) < SAME_TAU
    if same.any():
        t, x = 0.5 * (ta[same] + tb[same]), r[same]
        e = np.exp(-t * x)
        p = 1.0 / x + 11.0 * t / 16.0 + 3.0 * t * t * x / 16.0 + t ** 3 * x * x / 48.0
        dp = -1.0 / x ** 2 + 3.0 * t * t / 16.0 + t ** 3 * x / 24.0
        s[same] = e * p
        ds[same] = e * (dp - t * p)
    nd = ~same
    if nd.any():
        a, b, x = ta[nd], tb[nd], r[nd]
        val = np.zeros_like(x)
        der = np.zeros_like(x)
        for t1, t2 in ((a, b), (b, a)):
            d2 = t1 * t1 - t2 * t2
            k = t2 ** 4 * t1 / (2.0 * d2 * d2)
            l = (t2 ** 6 - 3.0 * t2 ** 4 * t1 * t1) / d2 ** 3
            e = np.exp(-t1 * x)
            val += e * (k - l / x)
            der += e * (-t1 * (k - l / x) + l / x ** 2)
        s[nd] = val
        ds[nd] = der
    return s, ds


def _gamma0(ta, tb):
    """gamma at R = 0 (on-site, possibly two different shells)."""
    general = 0.5 * (ta * tb / (ta + tb) + (ta * tb) ** 2 / (ta + tb) ** 3)
    same = np.abs(ta - tb) < SAME_TAU
    return np.where(same, (ta + tb) / 6.4, general)   # = (U_a + U_b) / 2


def _short_cutoff(tau_min: float) -> float:
    r = 1.0
    while True:
        s, _ = _short(np.array([r]), np.array([tau_min]), np.array([tau_min]))
        if abs(s[0]) < SHORT_TOL:
            return r
        r += 0.5


class Coulomb:
    """Gamma matrix and its derivatives for a fixed geometry."""

    def __init__(self, site_pos, site_atom, hubbard, cell=None, pbc=None):
        self.pos = np.asarray(site_pos, float)
        self.atom = np.asarray(site_atom)
        self.tau = 3.2 * np.asarray(hubbard, float)
        self.cell = None if cell is None else np.asarray(cell, float)
        self.pbc = np.zeros(3, bool) if pbc is None else np.asarray(pbc, bool)
        self.periodic = self.cell is not None and self.pbc.any()
        if self.periodic and not self.pbc.all():
            raise NotImplementedError("Only 0D or 3D periodicity implemented")
        n = len(self.pos)
        self.rcut_short = _short_cutoff(self.tau.min())
        if self.periodic:
            vol = abs(np.linalg.det(self.cell))
            self.vol = vol
            natom = len(np.unique(self.atom))
            self.alpha = np.sqrt(np.pi) * (natom / vol ** 2) ** (1.0 / 6.0)
            lt = np.sqrt(-np.log(EWALD_TOL))
            self.rcut_ewald = lt / self.alpha
            gcut = 2.0 * self.alpha * lt
            rec = 2.0 * np.pi * np.linalg.inv(self.cell).T
            nmax = [int(np.ceil(gcut / np.linalg.norm(rec[k]) * 1.0)) + 1 for k in range(3)]
            g = np.array(np.meshgrid(*[np.arange(-m, m + 1) for m in nmax], indexing="ij")).reshape(3, -1).T
            g = g[np.any(g != 0, axis=1)] @ rec
            g2 = (g * g).sum(1)
            keep = g2 < gcut ** 2
            self.gvec = g[keep]
            g2 = g2[keep]
            self.gfac = 4.0 * np.pi / vol * np.exp(-g2 / (4 * self.alpha ** 2)) / g2
        else:
            self.rcut_ewald = 0.0
        rc = max(self.rcut_short, self.rcut_ewald) if self.periodic else self.rcut_short
        self.nl = neighbor_list(self.pos, self.cell, self.pbc, rc)
        self.gamma = self._matrix(n)

    # ------------------------------------------------------------------
    def _matrix(self, n):
        i, j, _, vec, d = self.nl
        g = np.zeros((n, n))
        ta, tb = self.tau[i], self.tau[j]
        s, _ = _short(d, ta, tb)
        # same-atom pairs with T = 0 are not in the neighbour list
        g = ops.scatter_add(g, (i, j), -s)
        diag = _gamma0(self.tau[:, None], self.tau[None, :])
        same_atom = self.atom[:, None] == self.atom[None, :]
        g[same_atom] += diag[same_atom]
        if not self.periodic:
            diff = self.pos[None, :, :] - self.pos[:, None, :]
            dist = np.sqrt((diff * diff).sum(-1))
            with np.errstate(divide="ignore"):
                inv = np.where(same_atom, 0.0, 1.0 / np.where(same_atom, 1.0, dist))
            return g + inv
        # Ewald 1/R lattice sum, excluding the T = 0 self term
        a = self.alpha
        m = d < self.rcut_ewald
        g = ops.scatter_add(g, (i[m], j[m]), erfc(a * d[m]) / d[m])
        cg = np.cos(self.pos @ self.gvec.T)
        sg = np.sin(self.pos @ self.gvec.T)
        g += (cg * self.gfac) @ cg.T + (sg * self.gfac) @ sg.T
        g -= np.pi / (self.vol * a * a)
        # sites of one atom at distance 0: erf(a r)/r -> 2a/sqrt(pi)
        g[same_atom] -= 2.0 * a / np.sqrt(np.pi)
        return g

    def energy_shift(self, dq):
        v = self.gamma @ dq
        return 0.5 * dq @ v, v

    def gradient(self, dq):
        """dE_SCC/dr for every site (n_site, 3) with E = 1/2 dq.gamma.dq."""
        i, j, _, vec, d = self.nl
        n = len(self.pos)
        grad = np.zeros((n, 3))
        _, ds = _short(d, self.tau[i], self.tau[j])
        dg = -ds
        if self.periodic:
            a = self.alpha
            m = d < self.rcut_ewald
            dg[m] += -erfc(a * d[m]) / d[m] ** 2 - 2 * a / np.sqrt(np.pi) * np.exp(-(a * d[m]) ** 2) / d[m]
        else:
            # direct 1/R between different atoms (all pairs)
            diff = self.pos[None, :, :] - self.pos[:, None, :]
            dist = np.sqrt((diff * diff).sum(-1))
            other = self.atom[:, None] != self.atom[None, :]
            fac = np.where(other, -1.0 / np.where(other, dist, 1.0) ** 3, 0.0)
            # dE/dr_A = -sum_B qA qB d(1/R)/dR * R_vec/R
            grad -= ((dq[:, None] * dq[None, :] * fac)[:, :, None] * diff).sum(1)
        w = (dq[i] * dq[j] * dg / d)[:, None] * vec
        grad = ops.scatter_add(grad, i, -w)
        if self.periodic:
            phase = self.pos @ self.gvec.T
            cg, sg = np.cos(phase), np.sin(phase)
            sc = cg.T @ dq
            ss = sg.T @ dq
            # E_rec = 1/2 sum_G f |S(G)|^2 ;  dE/dr_A = q_A sum_G f G (-sin cS + cos sS)
            coef = dq[:, None] * (-sg * sc[None, :] + cg * ss[None, :]) * self.gfac[None, :]
            grad += coef @ self.gvec
        return grad
