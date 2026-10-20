"""Second-order (gamma) Coulomb kernel of SCC-DFTB, cluster and periodic.

    gamma_ab = sum_T [ 1/|R_AB + T|  -  S(tau_a, tau_b, |R_AB + T|) ],  tau = 16/5 U

The long-range 1/R lattice sum is done with Ewald summation; the short-range
correction S decays exponentially and is summed in real space.

"Sites" carry the charges: atoms (standard SCC) or shells (orbital-resolved
SCC).  Several sites may sit on one atom.  Everything that depends only on
positions (neighbour search, 1/R and Ewald terms) is done per *atom* and then
broadcast to the sites; only S depends on the site's Hubbard U.  Each pair of
U values gets its own short-range cut-off, so a soft shell (small U, slowly
decaying S) does not inflate the work for all other shell pairs.
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
    """Short-range S(r) and dS/dr, vectorised (broadcasting); r > 0."""
    r, ta, tb = np.broadcast_arrays(r, ta, tb)
    s = np.zeros(r.shape)
    ds = np.zeros(r.shape)
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


def _short_cutoff(ta: float, tb: float) -> float:
    """Distance beyond which |S(ta, tb, r)| < SHORT_TOL (S decays monotonically)."""
    r = 1.0
    while True:
        s, _ = _short(np.array([r]), np.array([ta]), np.array([tb]))
        if abs(s[0]) < SHORT_TOL:
            return r
        r += 0.5


class Coulomb:
    """Gamma matrix and its derivatives for a fixed geometry.

    Parameters
    ----------
    atom_pos : (n_atom, 3) Bohr
    site_atom : (n_site,) atom index of every charge site
    hubbard : (n_site,) Hubbard U of every site
    """

    def __init__(self, atom_pos, site_atom, hubbard, cell=None, pbc=None):
        self.pos = np.asarray(atom_pos, float)
        self.site_atom = np.asarray(site_atom)
        self.natom = len(self.pos)
        self.nsite = len(self.site_atom)
        self.tau = 3.2 * np.asarray(hubbard, float)
        self.cell = None if cell is None else np.asarray(cell, float)
        self.pbc = np.zeros(3, bool) if pbc is None else np.asarray(pbc, bool)
        self.periodic = self.cell is not None and self.pbc.any()
        if self.periodic and not self.pbc.all():
            raise NotImplementedError("Only 0D or 3D periodicity implemented")
        # sites of each atom, padded to a rectangular table (-1 = no site)
        self.max_sites = int(np.bincount(self.site_atom, minlength=self.natom).max())
        self.atom_sites = -np.ones((self.natom, self.max_sites), int)
        fill = np.zeros(self.natom, int)
        for s, a in enumerate(self.site_atom):
            self.atom_sites[a, fill[a]] = s
            fill[a] += 1
        # per-(tau_a, tau_b) short-range cut-offs
        self._utau = np.unique(self.tau)
        self._cut = np.array([[_short_cutoff(a, b) for b in self._utau] for a in self._utau])
        self.rcut_short = float(self._cut.max())
        if self.periodic:
            vol = abs(np.linalg.det(self.cell))
            self.vol = vol
            self.alpha = np.sqrt(np.pi) * (self.natom / vol ** 2) ** (1.0 / 6.0)
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
        rc = max(self.rcut_short, self.rcut_ewald)
        self.nl = neighbor_list(self.pos, self.cell, self.pbc, rc)      # atom pairs
        self._site_pairs()
        self.gamma = self._matrix()

    # ------------------------------------------------------------------
    def _site_pairs(self):
        """Expand atom pairs (i, j, T) to site pairs (a on i, b on j) within their cut-off."""
        i, j, _, _, d = self.nl
        sa = self.atom_sites[i][:, :, None]              # (np, ms, 1)
        sb = self.atom_sites[j][:, None, :]              # (np, 1, ms)
        valid = (sa >= 0) & (sb >= 0)
        ia = np.searchsorted(self._utau, self.tau)       # tau -> index into the cut-off table
        ka = np.where(sa >= 0, ia[np.maximum(sa, 0)], 0)
        kb = np.where(sb >= 0, ia[np.maximum(sb, 0)], 0)
        within = valid & (d[:, None, None] < self._cut[ka, kb])
        p, x, y = np.nonzero(within)
        self.sp_pair = p                                   # atom-pair index of each site pair
        self.sp_a = self.atom_sites[i[p], x]
        self.sp_b = self.atom_sites[j[p], y]
        self.sp_d = d[p]

    def _expand(self, mat_atoms):
        """Atom-level (n_atom, n_atom) matrix -> site-level."""
        return mat_atoms[np.ix_(self.site_atom, self.site_atom)]

    def _matrix(self):
        same_atom = self.site_atom[:, None] == self.site_atom[None, :]
        g = np.where(same_atom, _gamma0(self.tau[:, None], self.tau[None, :]), 0.0)
        s, _ = _short(self.sp_d, self.tau[self.sp_a], self.tau[self.sp_b])
        g = ops.scatter_add(g, (self.sp_a, self.sp_b), -s)
        if not self.periodic:
            diff = self.pos[None, :, :] - self.pos[:, None, :]
            dist = np.sqrt((diff * diff).sum(-1))
            np.fill_diagonal(dist, 1.0)
            inv = 1.0 / dist
            np.fill_diagonal(inv, 0.0)
            return g + self._expand(inv)
        # Ewald 1/R lattice sum (per atom pair), excluding the T = 0 self term
        a = self.alpha
        i, j, _, _, d = self.nl
        m = d < self.rcut_ewald
        real = ops.scatter_add(np.zeros((self.natom, self.natom)), (i[m], j[m]),
                               erfc(a * d[m]) / d[m])
        cg = np.cos(self.pos @ self.gvec.T)
        sg = np.sin(self.pos @ self.gvec.T)
        rec = (cg * self.gfac) @ cg.T + (sg * self.gfac) @ sg.T
        atom_mat = real + rec - np.pi / (self.vol * a * a)
        # sites on one atom at distance 0: erf(a r)/r -> 2a/sqrt(pi)
        atom_mat[np.diag_indices(self.natom)] -= 2.0 * a / np.sqrt(np.pi)
        return g + self._expand(atom_mat)

    def energy_shift(self, dq):
        v = self.gamma @ dq
        return 0.5 * dq @ v, v

    def gradient(self, dq):
        """dE_SCC/dR per **atom** (n_atom, 3) with E = 1/2 dq.gamma.dq (dq per site)."""
        i, j, _, vec, d = self.nl
        dq_atom = ops.segment_sum(dq, self.site_atom, self.natom)
        # short range: aggregate site pairs into dE/d(distance) per atom pair
        _, ds = _short(self.sp_d, self.tau[self.sp_a], self.tau[self.sp_b])
        dedr = ops.segment_sum(-dq[self.sp_a] * dq[self.sp_b] * ds, self.sp_pair, len(i))
        grad = np.zeros((self.natom, 3))
        if self.periodic:
            a = self.alpha
            m = d < self.rcut_ewald
            dg = -erfc(a * d[m]) / d[m] ** 2 - 2 * a / np.sqrt(np.pi) * np.exp(-(a * d[m]) ** 2) / d[m]
            dedr[m] += dq_atom[i[m]] * dq_atom[j[m]] * dg
            phase = self.pos @ self.gvec.T
            cg, sg = np.cos(phase), np.sin(phase)
            sc = cg.T @ dq_atom
            ss = sg.T @ dq_atom
            # E_rec = 1/2 sum_G f |S(G)|^2 ;  dE/dr_A = q_A sum_G f G (-sin cS + cos sS)
            coef = dq_atom[:, None] * (-sg * sc[None, :] + cg * ss[None, :]) * self.gfac[None, :]
            grad += coef @ self.gvec
        else:
            diff = self.pos[None, :, :] - self.pos[:, None, :]
            dist = np.sqrt((diff * diff).sum(-1))
            np.fill_diagonal(dist, 1.0)
            fac = -1.0 / dist ** 3
            np.fill_diagonal(fac, 0.0)
            # dE/dr_A = -sum_B qA qB d(1/R)/dR * R_vec/R
            grad -= ((dq_atom[:, None] * dq_atom[None, :] * fac)[:, :, None] * diff).sum(1)
        w = (dedr / d)[:, None] * vec
        return ops.scatter_add(grad, i, -w)
