"""Second-order (gamma) Coulomb kernel of SCC-DFTB, cluster and periodic.

    gamma_ab = sum_T [ 1/|R_AB + T|  -  S(tau_a, tau_b, |R_AB + T|) ],  tau = 16/5 U

The long-range 1/R lattice sum is done with Ewald summation; the short-range
correction S decays exponentially and is summed in real space up to DFTB+'s cut-off
(S < 1e-10, :func:`short_cutoff`).  An earlier, tighter cut-off (1e-14) left ~1e-9 Ha
differences in the potentials of periodic systems (M2e report).

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
from .constants import MIN_HUB_DIFF, MIN_SHORT_GAMMA, TOL_SHORT_GAMMA
from .corrections import h5_scale
from .neighbors import neighbor_list

EWALD_TOL = 1e-12
# DFTB+ treats U_a, U_b as equal if |U_a - U_b| < MIN_HUB_DIFF; in tau = 3.2 U:
SAME_TAU = 3.2 * MIN_HUB_DIFF
# The closed form of S for tau_a != tau_b carries 1/(tau_a^2 - tau_b^2)^n terms that
# cancel when tau_a ~ tau_b: at |U_a - U_b| = 1e-5 its relative error is 6e-4 (and 24
# for dS/dtau_a, i.e. DFTB3).  Below this relative tau difference S is computed from
# the Hermite-Genocchi integral instead (see _short_near).
NEAR_TAU = 0.05
_GL_X, _GL_W = np.polynomial.legendre.leggauss(12)
_GL_S = 0.5 * (_GL_X + 1.0)                       # Gauss-Legendre nodes on [0, 1]
_GL_W = 0.5 * _GL_W * _GL_S * (1.0 - _GL_S)       # ... and weights times s (1 - s)


def _short_near(r, ta, tb, tau_derivs=False):
    """S and dS/dr (with ``tau_derivs`` also dS/dtau_a, d2S/dtau_a dr) for any tau.

    S is a divided difference, S = -A^2 B^2 E[A, A, B, B] with A = ta^2, B = tb^2 and
    E(y) = exp(-sqrt(y) r) / (y r); by the Hermite-Genocchi formula

        S = -A^2 B^2 int_0^1 s (1 - s) E'''(A + s (B - A)) ds,

    which is free of the cancellations of the closed form.  With 12 Gauss points S and its
    first derivatives agree with 60-digit mpmath values to < 4e-12 (relative) for
    |tau_a - tau_b| < 10 % and r < 18 Bohr.
    """
    xp = ops.namespace(r, ta, tb)
    s = ops.asarray(_GL_S, like=r)[:, None]
    w = ops.asarray(_GL_W, like=r)[:, None]
    a2, b2 = ta * ta, tb * tb
    q = xp.sqrt(a2 + s * (b2 - a2))
    x = q * r
    e = xp.exp(-x)
    q8 = q ** 8
    e3 = -e * (((x + 9.0) * x + 33.0) * x + 48.0) / (8.0 * r * q8)               # E'''
    e3r = e * ((((x + 7.0) * x + 24.0) * x + 48.0) * x + 48.0) / (8.0 * r * r * q8)
    pre = -a2 * a2 * b2 * b2
    j, jr = (w * e3).sum(0), (w * e3r).sum(0)
    if not tau_derivs:
        return pre * j, pre * jr
    q10 = q8 * q * q
    e4 = e * ((((x + 14.0) * x + 87.0) * x + 279.0) * x + 384.0) / (16.0 * r * q10)  # E''''
    e4r = -e * (((((x + 11.0) * x + 59.0) * x + 192.0) * x + 384.0) * x + 384.0) / (16.0 * r * r * q10)
    w1 = w * (1.0 - s)
    da = -2.0 * a2 * b2 * b2 * j + pre * (w1 * e4).sum(0)                       # dS/dA
    dar = -2.0 * a2 * b2 * b2 * jr + pre * (w1 * e4r).sum(0)
    return pre * j, pre * jr, 2.0 * ta * da, 2.0 * ta * dar


def _short(r, ta, tb):
    """Short-range S(r) and dS/dr, vectorised (broadcasting); r > 0.  NumPy or torch."""
    xp = ops.namespace(r, ta, tb)
    r, ta, tb = xp.broadcast_arrays(r, ta, tb)
    s = ops.zeros(r.shape, like=r)
    ds = ops.zeros(r.shape, like=r)
    same = xp.abs(ta - tb) < SAME_TAU
    if same.any():
        t, x = 0.5 * (ta[same] + tb[same]), r[same]
        e = xp.exp(-t * x)
        p = 1.0 / x + 11.0 * t / 16.0 + 3.0 * t * t * x / 16.0 + t ** 3 * x * x / 48.0
        dp = -1.0 / x ** 2 + 3.0 * t * t / 16.0 + t ** 3 * x / 24.0
        s[same] = e * p
        ds[same] = e * (dp - t * p)
    near = ~same & (xp.abs(ta - tb) < NEAR_TAU * 0.5 * (ta + tb))
    if near.any():
        s[near], ds[near] = _short_near(r[near], ta[near], tb[near])
    nd = ~same & ~near
    if nd.any():
        a, b, x = ta[nd], tb[nd], r[nd]
        val = xp.zeros_like(x)
        der = xp.zeros_like(x)
        for t1, t2 in ((a, b), (b, a)):
            d2 = t1 * t1 - t2 * t2
            k = t2 ** 4 * t1 / (2.0 * d2 * d2)
            l = (t2 ** 6 - 3.0 * t2 ** 4 * t1 * t1) / d2 ** 3
            e = xp.exp(-t1 * x)
            val += e * (k - l / x)
            der += e * (-t1 * (k - l / x) + l / x ** 2)
        s[nd] = val
        ds[nd] = der
    return s, ds


def _gamma0(ta, tb):
    """gamma at R = 0 (on-site, possibly two different shells)."""
    xp = ops.namespace(ta, tb)
    general = 0.5 * (ta * tb / (ta + tb) + (ta * tb) ** 2 / (ta + tb) ** 3)
    same = xp.abs(ta - tb) < SAME_TAU
    return xp.where(same, (ta + tb) / 6.4, general)   # = (U_a + U_b) / 2


def short_cutoff(ta: float, tb: float) -> float:
    """DFTB+ ``expGammaCutoff`` for tau = 3.2 U: the distance where S drops below
    MIN_SHORT_GAMMA, by doubling and then bisection to TOL_SHORT_GAMMA in S.  The same
    cut serves the second-order term (per pair of U values) and the third-order term."""
    def s(r):
        return float(_short(np.array([r]), np.array([ta]), np.array([tb]))[0][0])
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


class Coulomb:
    """Gamma matrix and its derivatives for a fixed geometry.

    Parameters
    ----------
    atom_pos : (n_atom, 3) Bohr
    site_atom : (n_site,) atom index of every charge site
    hubbard : (n_site,) Hubbard U of every site
    damped : (n_atom,) bool, atoms whose pairs get the H-X damping of DFTB+
             ``HCorrection = Damping``: S -> S exp(-((U_a + U_b)/2)^damp_exp r^2)
    h5 : (is_h, k, R, r_scaling, w_scaling), the per-atom H5 parameters of
         :meth:`pydftb.corrections.H5.atom_parameters`: DFTB+'s ``HCorrection = H5`` on pairs
         of one H and an atom with k > 0, S -> S (1 + g) - g / r (milestone M6f-3)
    """

    def __init__(self, atom_pos, site_atom, hubbard, cell=None, pbc=None, damped=None,
                 damp_exp: float = 0.0, h5=None):
        self.pos = np.asarray(atom_pos, float)
        self.site_atom = np.asarray(site_atom)
        self.natom = len(self.pos)
        self.nsite = len(self.site_atom)
        self.tau = 3.2 * np.asarray(hubbard, float)
        self.cell = None if cell is None else np.asarray(cell, float)
        self.pbc = np.zeros(3, bool) if pbc is None else np.asarray(pbc, bool)
        self.periodic = self.cell is not None and self.pbc.any()
        self.damped = np.zeros(self.natom, bool) if damped is None else np.asarray(damped, bool)
        self.damp_exp = float(damp_exp)
        self.h5 = h5
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
        self._cut = np.array([[short_cutoff(a, b) for b in self._utau] for a in self._utau])
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
            gint = g[np.any(g != 0, axis=1)]
            g = gint @ rec
            g2 = (g * g).sum(1)
            keep = g2 < gcut ** 2
            self.gvec = g[keep]
            self.gint = gint[keep]          # integer coordinates (G = gint @ 2 pi inv(cell).T)
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

    def _sp_short(self):
        """S and dS/dr of every site pair, damped where an H-X pair asks for it."""
        s, ds = _short(self.sp_d, self.tau[self.sp_a], self.tau[self.sp_b])
        if self.damped.any():
            i, j = self.nl[0], self.nl[1]
            m = (self.damped[i] | self.damped[j])[self.sp_pair]
            u = (self.tau[self.sp_a[m]] + self.tau[self.sp_b[m]]) / 6.4     # (U_a + U_b) / 2
            r = self.sp_d[m]
            h = np.exp(-u ** self.damp_exp * r * r)
            ds[m] = ds[m] * h - 2.0 * r * u ** self.damp_exp * s[m] * h
            s[m] = s[m] * h
        if self.h5 is not None:                     # DFTB+'s H5 (not with damping; M6f-3)
            is_h, k, radius, r_scaling, w_scaling = self.h5
            i, j = self.nl[0], self.nl[1]
            kp = np.where(is_h[i] != is_h[j], k[np.where(is_h[i], j, i)], -1.0)[self.sp_pair]
            m = kp > 0.0
            if m.any():
                p = self.sp_pair[m]
                s[m], ds[m] = h5_scale(s[m], ds[m], self.sp_d[m], kp[m],
                                       radius[i[p]] + radius[j[p]], r_scaling, w_scaling)
        return s, ds

    def _expand(self, mat_atoms):
        """Atom-level (n_atom, n_atom) matrix -> site-level."""
        return mat_atoms[np.ix_(self.site_atom, self.site_atom)]

    def _matrix(self):
        same_atom = self.site_atom[:, None] == self.site_atom[None, :]
        g = np.where(same_atom, _gamma0(self.tau[:, None], self.tau[None, :]), 0.0)
        s, _ = self._sp_short()
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

    def gradient(self, dq, virial: bool = False, other=None):
        """dE_SCC/dR per **atom** (n_atom, 3) with E = 1/2 dq.gamma.dq (dq per site).

        With ``virial=True`` (periodic systems) the same pass also gives the
        virial dE_SCC/d(strain) (3, 3) at fixed charges; otherwise it is None.
        Pair terms: 1/2 sum over ordered atom pairs of dE/dr * r_a r_b / r.
        Reciprocal Ewald: 1/2 sum_G f|S|^2 [-delta + 2 G G (1/(4 alpha^2) + 1/G^2)].
        Background: + pi Q^2 / (2 V alpha^2) delta.  The Ewald alpha is held
        fixed (the energy does not depend on it).

        ``other`` (per site): the derivative of the bilinear E = 1/2 dq.gamma.other instead
        (excited-state gradients, the XL-BOMD shadow energy; milestone M6e): every product
        x_a x_b becomes 1/2 (x_a y_b + y_a x_b).
        """
        if virial and not self.periodic:
            raise ValueError("the virial/stress needs a periodic system")
        i, j, _, vec, d = self.nl
        dq_atom = ops.segment_sum(dq, self.site_atom, self.natom)
        if other is None:
            def prod(x1, y1, x2, y2):
                return x1 * y2
            o, o_atom = dq, dq_atom
        else:
            def prod(x1, y1, x2, y2):
                return 0.5 * (x1 * y2 + y1 * x2)
            o = np.asarray(other, float)
            o_atom = ops.segment_sum(o, self.site_atom, self.natom)
        # short range: aggregate site pairs into dE/d(distance) per atom pair
        _, ds = self._sp_short()
        dedr = ops.segment_sum(-prod(dq[self.sp_a], o[self.sp_a], dq[self.sp_b], o[self.sp_b]) * ds,
                               self.sp_pair, len(i))
        grad = np.zeros((self.natom, 3))
        w = None
        if self.periodic:
            a = self.alpha
            m = d < self.rcut_ewald
            dg = -erfc(a * d[m]) / d[m] ** 2 - 2 * a / np.sqrt(np.pi) * np.exp(-(a * d[m]) ** 2) / d[m]
            dedr[m] += prod(dq_atom[i[m]], o_atom[i[m]], dq_atom[j[m]], o_atom[j[m]]) * dg
            phase = self.pos @ self.gvec.T
            cg, sg = np.cos(phase), np.sin(phase)
            sc = cg.T @ dq_atom
            ss = sg.T @ dq_atom
            # E_rec = 1/2 sum_G f |S(G)|^2 ;  dE/dr_A = q_A sum_G f G (-sin cS + cos sS)
            coef = dq_atom[:, None] * (-sg * sc[None, :] + cg * ss[None, :]) * self.gfac[None, :]
            if other is not None:
                sco, sso = cg.T @ o_atom, sg.T @ o_atom
                coef = 0.5 * (dq_atom[:, None] * (-sg * sco[None, :] + cg * sso[None, :])
                              * self.gfac[None, :]
                              + o_atom[:, None] * (-sg * sc[None, :] + cg * ss[None, :])
                              * self.gfac[None, :])
            grad += coef @ self.gvec
            if virial:
                if other is None:
                    c = 0.5 * self.gfac * (sc * sc + ss * ss)            # 1/2 f |S(G)|^2
                else:
                    c = 0.5 * self.gfac * (sc * sco + ss * sso)
                g2 = (self.gvec * self.gvec).sum(1)
                w = 0.5 * np.einsum("p,pa,pb->ab", dedr / d, vec, vec)
                w -= c.sum() * np.eye(3)
                w += 2.0 * np.einsum("g,ga,gb->ab", c * (1.0 / (4 * a * a) + 1.0 / g2), self.gvec, self.gvec)
                q = dq_atom.sum()
                qo = q if other is None else o_atom.sum()
                w += np.pi * q * qo / (2.0 * self.vol * a * a) * np.eye(3)
        else:
            diff = self.pos[None, :, :] - self.pos[:, None, :]
            dist = np.sqrt((diff * diff).sum(-1))
            np.fill_diagonal(dist, 1.0)
            fac = -1.0 / dist ** 3
            np.fill_diagonal(fac, 0.0)
            # dE/dr_A = -sum_B qA qB d(1/R)/dR * R_vec/R
            qq = prod(dq_atom[:, None], o_atom[:, None], dq_atom[None, :], o_atom[None, :])
            grad -= ((qq * fac)[:, :, None] * diff).sum(1)
        g = (dedr / d)[:, None] * vec
        return ops.scatter_add(grad, i, -g), w
