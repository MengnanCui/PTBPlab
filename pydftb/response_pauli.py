"""Coupled-perturbed SCC response of two-component calculations (milestone M6g-3b): DFTB+'s
``Polarisability``, ``ResponseKernel`` and the development version's
``ExternalChargeDerivatives`` with non-collinear spin and spin-orbit coupling (onsite and dual
model), with DFTB+U and onsite corrections; the entry points of :mod:`pydftb.response` branch
here when the calculation is two-component.

DFTB+ (``derivs/linearresponse.F90`` dRhoPauli, dRhoFermiChangePauli) builds the first-order
density from the spinor states of every k-point as for complex eigenvectors, at most one
electron per level and one Fermi level.  pydftb writes the response, as in
:mod:`pydftb.response`, as a matrix: the Pauli component b = (q, m_x, m_y, m_z) of the
population of orbital mu has the element

    m^{mu,b}_ai = 1/2 sum_ss' [c*_{s mu,a} sigma^b_ss' (S c_i)_{s' mu} + (S c_a)*_{s mu} sigma^b_ss' c_{s' mu,i}]

(the Mulliken population of Tr_spin(sigma^b rho), :func:`pydftb.spinorbit.components`), and a
unit potential on shell X of component b the matrix element sum_{mu in X} m^{mu,b}_ai, so
chi0 couples all four components: rows [orbital populations] and columns [shell potentials]
of every component, (4 n_orb, 4 n_shell), with the static pair weights of
:meth:`pydftb.response.GroundState.weights` and the rank-one move of the one Fermi level.  The
kernel is gamma (and the full third order) on the charge, the spin constants W on each
magnetisation component.

Block potentials (+U, onsite corrections) add the channels of
:class:`pydftb.response.BlockChannels` in the four components (operator 1/2 (S E_c + E_c S)
x sigma_b).  With dual spin-orbit coupling the ground state also mixes the imaginary (skew)
block populations, and the +U and onsite potentials depend on them: :class:`SkewChannels`
(operator i/2 (S E_c + E_c S) x sigma_b, E_c = E_ij - E_ji, population 2 Qi_ij).  DFTB+ 25.1
leaves them out (perturb.F90 passes only the real block populations to the block potentials),
which makes its polarisability with dual SOC and +U or onsite corrections wrong by about 1e-4
(relative); ``DFTB._response_skew_compat`` (testing) leaves them out too.

Without spin-orbit coupling a global rotation of all moments does not change the energy: the
static response equation (I - A B) x = A V has exact zero modes, the rotations e_a x m of the
ground state's shell (and block) magnetisations.  pydftb solves the bordered system
[[I - A B, N], [N^T, 0]] with an orthonormal basis N of these rotations, i.e. the response
without a global rotation (DFTB+'s purely longitudinal answer for collinear moments); with
spin-orbit coupling it uses the singular value decomposition and leaves out (with a warning)
singular values below max(1e-10 s_max, 10 x the SCC error), e.g. the rotation of a linear
molecule's electrons about its axis.

NumPy engine only.  Coordinate derivatives of two-component calculations (molecules) are in
:func:`coordinate_derivatives`; DFTB+ refuses them.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import response as R
from . import spinorbit as so
from .backend import ops

NULL_REL = 1e-10          # rotation generators: relative norm below which a direction is noise
SVD_REL = 1e-10           # singular values of I - A B left out below SVD_REL s_max (with SOC)
SVD_SCC = 10.0            # ... or below SVD_SCC x the ground state's SCC error


# ---- the unperturbed spinor states ---------------------------------------------------------
class PauliGroundState(R.GroundState):
    """Eigenpairs of the last spinor Hamiltonian: one block per k-point (2N levels, at most one
    electron each, ``spin`` 0), one Fermi level.  ``d_comp`` (4, N) is the Fermi vector of the
    four Pauli components (sum_k w_k sum_{i in W} delta_i m^{mu,b}_ii), ``d_net`` (N,) its
    on-site charge part, ``dos`` (1,) sum_k w_k sum_{i in W} delta_i."""

    def __init__(self, calc, st, sc, degeneracy_tol: float = R.DEGEN_TOL,
                 filling_tol: float | None = None):
        b = st.basis
        self.natom = len(st.species)
        self.norb = n = b.norb
        self.orb_shell = np.asarray(b.orb_shell)
        self.orb_atom = np.asarray(b.orb_atom)
        self.shell_atom = np.asarray(b.shell_atom)
        self.nshell = len(b.shell_l)
        self.shell_start = np.concatenate([[0], np.cumsum(2 * np.asarray(b.shell_l) + 1)[:-1]])
        self.nspin = 1                               # one channel, one Fermi level
        self.max_fill = 1.0
        self.beta = 1.0 / calc.smearing.kt
        self.fixed_fermi = calc.fixed_fermi_level is not None
        self.shared_fermi = False
        mu = float(np.ravel(sc.mu)[0])
        low, high = (R.FILLED_TOL, R.EMPTY_TOL) if filling_tol is None else (filling_tol,) * 2
        self.blocks = []
        for k, hk in enumerate(np.asarray(sc.h)):
            s2 = np.asarray(st.pauli.s2[k])
            evals, c = ops.eigh(hk, s2)
            evals, c = np.asarray(evals), np.asarray(c)
            fill = R._theta(self.beta * (evals - mu))
            above = np.flatnonzero(fill > low)
            n_filled = int(above[-1]) + 1 if above.size else 0
            short = np.flatnonzero(np.abs(fill - 1.0) > high)
            first_empty = int(short[0]) if short.size else 0
            group = np.concatenate([[0], np.cumsum(np.diff(evals) > degeneracy_tol)])
            self.blocks.append(R._Block(0, k, float(st.kw[k]), evals, c, s2 @ c, s2, mu,
                                        n_filled, first_empty, group))
        self.metallic = any(blk.metallic for blk in self.blocks)
        self.fermi_shift = self.metallic and not self.fixed_fermi
        self.dos = np.zeros(1)
        self.d_comp = np.zeros((4, n))
        self.d_net = np.zeros(n)
        for blk in self.blocks:
            win = blk.window
            if win.size:
                dlt = blk.weight * R._delta(blk.evals[win], blk.fermi, self.beta)
                self.dos[0] += dlt.sum()
                self.d_comp += np.real(spinor_elements(blk, win, win)) @ dlt
                cw = blk.c[:, win]
                self.d_net += (np.abs(cw[:n]) ** 2 + np.abs(cw[n:]) ** 2) @ dlt


def _pauli_sum(part):
    """sum_ss' sigma^b_ss' X_ss' for the four Pauli matrices, X_ss' = part(s, s')."""
    uu, dd, ud, du = part(0, 0), part(1, 1), part(0, 1), part(1, 0)
    return np.stack([uu + dd, ud + du, -1j * ud + 1j * du, uu - dd])


def spinor_elements(blk: R._Block, i, a) -> np.ndarray:
    """m^{mu,b}_ai of every orbital and Pauli component: (4, N, pairs)."""
    n = blk.c.shape[0] // 2

    def part(s, t):
        ca, sa = blk.c[s * n:(s + 1) * n, a].conj(), blk.sc[s * n:(s + 1) * n, a].conj()
        ci, si = blk.c[t * n:(t + 1) * n, i], blk.sc[t * n:(t + 1) * n, i]
        return ca * si + sa * ci

    return 0.5 * _pauli_sum(part)


def onsite_elements(blk: R._Block, i, a) -> np.ndarray:
    """The on-site charge products sum_s c_{s mu,a} c*_{s mu,i}: (N, pairs)."""
    n = blk.c.shape[0] // 2
    c = blk.c
    return c[:n, a] * c[:n, i].conj() + c[n:, a] * c[n:, i].conj()


def channel_elements(blk: R._Block, i, a, mu, nu, skew: bool = False) -> np.ndarray:
    """<a| O_c x sigma_b |i> of block channels (mu, nu): real O_c = 1/2 (S E_c + E_c S) with
    E_c = (E_mu,nu + E_nu,mu) / (1 + d_mu,nu), skew O_c = i/2 (S E_c + E_c S) with
    E_c = E_mu,nu - E_nu,mu: (4, n_channel, pairs)."""
    n = blk.c.shape[0] // 2
    sgn = -1.0 if skew else 1.0

    def part(s, t):
        ca, sa = blk.c[s * n:(s + 1) * n, a].conj(), blk.sc[s * n:(s + 1) * n, a].conj()
        ci, si = blk.c[t * n:(t + 1) * n, i], blk.sc[t * n:(t + 1) * n, i]
        return (sa[mu] * ci[nu] + sgn * sa[nu] * ci[mu]
                + ca[mu] * si[nu] + sgn * ca[nu] * si[mu])

    out = 0.5 * _pauli_sum(part)
    if skew:
        return 1j * out
    return out / (1.0 + (mu == nu))[None, :, None]


# ---- block channels ------------------------------------------------------------------------
@dataclass
class SkewChannels:
    """The skew (imaginary) block channels of dual spin-orbit coupling with block potentials:
    the elements i < j of every atom block the potentials touch.  Channel c puts
    Vi = v E_c, E_c = E_ij - E_ji, into a Pauli component (H gets i/2 (S Vi + Vi S) x sigma_b);
    the conjugate population is 2 Qi_ij, Qi the imaginary part of the dual block populations.
    ``kernel`` (4 ns, 4 ns) is d Vi_c / d Qi_d of :meth:`pydftb.blockpot.BlockTerms.potentials`
    (linear: Vi does not depend on the real blocks, nor V on Qi)."""
    k: np.ndarray
    mu: np.ndarray
    nu: np.ndarray
    local_i: np.ndarray
    local_j: np.ndarray
    kernel: np.ndarray

    @property
    def size(self) -> int:
        return len(self.k)

    @property
    def to_q(self) -> np.ndarray:
        return np.full(self.size, 0.5)

    @classmethod
    def build(cls, bt) -> SkewChannels:
        k, mu, nu, li, lj = [], [], [], [], []
        for kk, sl in enumerate(bt.slices):
            n = sl.stop - sl.start
            mask = np.zeros((n, n), bool)
            for _, idx, _ in bt.u_groups[kk]:
                mask[np.ix_(idx, idx)] = True
            if bt.ons[kk] is not None:
                mask[:] = True
            for i in range(n):
                for j in range(i + 1, n):
                    if mask[i, j]:
                        k.append(kk)
                        mu.append(sl.start + i)
                        nu.append(sl.start + j)
                        li.append(i)
                        lj.append(j)
        ns = len(k)
        zero = [np.zeros((4, sl.stop - sl.start, sl.stop - sl.start)) for sl in bt.slices]
        _, vi0 = bt.potentials(zero, [z.copy() for z in zero])
        kernel = np.zeros((4 * ns, 4 * ns))
        for t in range(4):
            for d in range(ns):
                skew = [z.copy() for z in zero]
                skew[k[d]][t, li[d], lj[d]], skew[k[d]][t, lj[d], li[d]] = 1.0, -1.0
                _, vi = bt.potentials(zero, skew)
                for s in range(4):
                    kernel[s * ns:(s + 1) * ns, t * ns + d] = [
                        vi[k[c]][s, li[c], lj[c]] - vi0[k[c]][s, li[c], lj[c]] for c in range(ns)]
        return cls(np.array(k, int), np.array(mu, int), np.array(nu, int), np.array(li, int),
                   np.array(lj, int), kernel)


def block_channels(calc, st) -> tuple:
    """(real channels in the four Pauli components, skew channels) of a two-component
    calculation with block potentials; None where there are none.  The skew channels exist
    with dual spin-orbit coupling (``BlockTerms.imaginary``) unless
    ``calc._response_skew_compat`` (DFTB+ 25.1's response, testing)."""
    if st.block is None:
        return None, None
    real = R.BlockChannels.build(st.block)
    real = real if real.size else None
    skew = None
    if st.block.imaginary and not getattr(calc, "_response_skew_compat", False):
        skew = SkewChannels.build(st.block)
        skew = skew if skew.size else None
    return real, skew


@dataclass
class Layout:
    """Sizes of the response's rows and columns per Pauli component: rows [N orbitals | nb real
    channels | ns skew channels], columns [n_shell shells | nb | ns]."""
    norb: int
    nshell: int
    nb: int
    ns: int

    @property
    def nr(self) -> int:
        return self.norb + self.nb + self.ns

    @property
    def nc(self) -> int:
        return self.nshell + self.nb + self.ns


def _row_elements(gs: PauliGroundState, blk, i, a, real, skew) -> np.ndarray:
    """The population elements of a block's pairs, (4, nr, P)."""
    parts = [spinor_elements(blk, i, a)]
    if real is not None:
        parts.append(channel_elements(blk, i, a, real.mu, real.nu))
    if skew is not None:
        parts.append(channel_elements(blk, i, a, skew.mu, skew.nu, skew=True))
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def _columns(gs: PauliGroundState, lay: Layout, rows: np.ndarray) -> np.ndarray:
    """Row elements (4, nr, ...) -> column elements (4, nc, ...): orbitals summed to shells."""
    sh = np.add.reduceat(rows[:, :lay.norb], gs.shell_start, axis=1)
    if lay.nb + lay.ns == 0:
        return sh
    return np.concatenate([sh, rows[:, lay.norb:]], axis=1)


def fermi_vectors(gs: PauliGroundState, lay: Layout, real, skew) -> np.ndarray:
    """Rows (4, nr): the Fermi vector of every population (orbitals, real and skew channels)."""
    out = np.zeros((4, lay.nr))
    out[:, :lay.norb] = gs.d_comp
    if lay.nb + lay.ns:
        for blk in gs.blocks:
            win = blk.window
            if win.size:
                dlt = blk.weight * R._delta(blk.evals[win], blk.fermi, gs.beta)
                rows = _row_elements(gs, blk, win, win, real, skew)
                out[:, lay.norb:] += np.real(rows[:, lay.norb:]) @ dlt
    return out


# ---- chi0 ----------------------------------------------------------------------------------
@dataclass
class PauliChi0:
    """Non-interacting response of a two-component calculation at one frequency: ``full``
    (4 nr, 4 nc), per Pauli component (q, m_x, m_y, m_z) the rows [orbital populations | real
    block populations P_c | skew populations] per unit potential on the columns [shells | real
    channels | skew channels] of every component (the Fermi-level move included); ``net``
    (N, 4 nc) the on-site charge populations, or None."""
    full: np.ndarray
    net: np.ndarray | None
    layout: Layout


def chi0(gs: PauliGroundState, omegas, eta: float = R.ETA, with_net: bool = False,
         eta_compat: bool = False, real=None, skew=None) -> list:
    """:class:`PauliChi0` at every frequency of ``omegas`` (one pass over the pairs); complex
    weights by their real part (``eta_compat``: their conjugate, as in
    :func:`pydftb.response.chi0`)."""
    lay = Layout(gs.norb, gs.nshell, 0 if real is None else real.size,
                 0 if skew is None else skew.size)
    nr, nc = lay.nr, lay.nc
    nw = len(omegas)
    full = np.zeros((nw, 4 * nr, 4 * nc))
    net = np.zeros((nw, gs.norb, 4 * nc)) if with_net else None
    for blk in gs.blocks:
        i, a, same = gs.pairs(blk)
        if not i.size:
            continue
        step = max(1, R.PAIR_CHUNK // max(4 * nr, 1))
        for lo in range(0, i.size, step):
            ii, aa, ss = i[lo:lo + step], a[lo:lo + step], same[lo:lo + step]
            rows = _row_elements(gs, blk, ii, aa, real, skew)          # (4, nr, P)
            cols = _columns(gs, lay, rows).reshape(4 * nc, -1)          # (4 nc, P)
            rc = rows.reshape(4 * nr, -1).conj()
            on = onsite_elements(blk, ii, aa) if with_net else None
            for w, omega in enumerate(omegas):
                g = gs.weights(blk, ii, aa, ss, float(omega), eta) * blk.weight
                if np.iscomplexobj(g):
                    g = np.conj(g) if eta_compat else g.real
                gc = g[:, None] * cols.T                                # (P, 4 nc)
                full[w] += np.real(rc @ gc)
                if with_net:
                    net[w] += np.real(on @ gc)
    if gs.fermi_shift:
        d_rows = fermi_vectors(gs, lay, real, skew)                     # (4, nr)
        d_cols = _columns(gs, lay, d_rows).ravel()
        for w, omega in enumerate(omegas):
            if float(omega) == 0.0:
                full[w] += np.outer(d_rows.ravel(), d_cols) / gs.dos[0]
                if with_net:
                    net[w] += np.outer(gs.d_net, d_cols) / gs.dos[0]
    return [PauliChi0(full[w], None if net is None else net[w], lay) for w in range(nw)]


# ---- solving -------------------------------------------------------------------------------
def rotation_generators(calc, st, sc, lay: Layout, nq: int, real) -> np.ndarray | None:
    """Orthonormal basis (n_unknown, r) of the global spin rotations of the ground state (its
    input populations, from which H was built): e_a x m of every shell magnetisation and of
    every real block channel's magnetisation, zero charge parts; None for a non-magnetic
    ground state."""
    _, m = calc._charge_and_magnetisation(st, sc.q_in)
    m = np.asarray(m, float)                                            # (3, n_shell)
    nsh, nb = lay.nshell, lay.nb
    qb = None
    if real is not None:
        _, blocks, _ = st.block.expand(sc.q_in)
        qb = np.array([[blocks[k][1 + a, i, j] for k, i, j in
                        zip(real.k, real.local_i, real.local_j)] for a in range(3)])
    nunk = nq + 3 * nsh + 4 * nb + 4 * lay.ns
    gens = np.zeros((nunk, 3))
    for a in range(3):
        e = np.eye(3)[a][:, None]
        gens[nq:nq + 3 * nsh, a] = np.cross(e, m, axis=0).ravel()
        if qb is not None:
            off = nq + 3 * nsh
            gens[off + nb:off + 4 * nb, a] = np.cross(e, qb, axis=0).ravel()
    u, s, _ = np.linalg.svd(gens, full_matrices=False)
    if s[0] <= 1e-12:
        return None
    return u[:, s > NULL_REL * s[0]]


def scc_error(sc) -> float:
    """The ground state's last SCC error, max |q_out - q_in| of the mixed vectors."""
    return float(np.abs(np.asarray(sc.q_out) - np.asarray(sc.q_in)).max(initial=0.0))


class PauliResponse:
    """The SCC response of a two-component calculation at one frequency: unknowns x = [dq on the
    SCC sites | dm_x, dm_y, dm_z on the shells | real block populations Q_c of the four
    components | skew populations Qi_c of the four components], (I - A B) x = A V_ext with
    A = chi0 (potentials -> x) and B the kernel (x -> potentials [shells | real | skew] per
    component).  ``null``: rotation generators (bordered solve); ``cut``: the SVD threshold
    (with spin-orbit coupling); neither: a plain solve (finite frequency)."""

    def __init__(self, gs: PauliGroundState, kernel, x: PauliChi0, scc: bool, real=None,
                 skew=None, null=None, cut: float | None = None):
        self.gs, self.x, self.scc = gs, x, scc
        self.real, self.skew = real, skew
        lay = self.layout = x.layout
        kq, e, w = kernel
        no, nsh, nb, ns, nr, nc = lay.norb, lay.nshell, lay.nb, lay.ns, lay.nr, lay.nc
        nq = e.shape[1]
        full = x.full
        nunk = nq + 3 * nsh + 4 * nb + 4 * ns
        amat = np.zeros((nunk, 4 * nc))
        amat[:nq] = e.T @ gs.shells(full[:no])
        for a in range(3):
            r0 = (a + 1) * nr
            amat[nq + a * nsh:nq + (a + 1) * nsh] = gs.shells(full[r0:r0 + no])
        off = nq + 3 * nsh
        for b in range(4):
            if nb:
                amat[off + b * nb:off + (b + 1) * nb] = (full[b * nr + no:b * nr + no + nb]
                                                         * real.to_q[:, None])
            if ns:
                o2 = off + 4 * nb
                amat[o2 + b * ns:o2 + (b + 1) * ns] = 0.5 * full[b * nr + no + nb:(b + 1) * nr]
        bmat = np.zeros((4 * nc, nunk))
        bmat[:nsh, :nq] = e @ kq
        if w is not None:
            for a in range(3):
                c0 = (a + 1) * nc
                bmat[c0:c0 + nsh, nq + a * nsh:nq + (a + 1) * nsh] = w
        for b in range(4):
            if nb:
                bmat[b * nc + nsh:b * nc + nsh + nb, off:off + 4 * nb] = real.kernel[b * nb:(b + 1) * nb]
            if ns:
                o2 = off + 4 * nb
                bmat[b * nc + nsh + nb:(b + 1) * nc, o2:] = skew.kernel[b * ns:(b + 1) * ns]
        self.amat, self.bmat = amat, bmat
        self.null, self.cut = null, cut
        self.dropped = 0
        self._svd = None
        if scc and null is None and cut is not None:
            lhs = np.eye(nunk) - amat @ bmat
            u, s, vt = np.linalg.svd(lhs)
            keep = s > max(SVD_REL * s[0], cut)
            self.dropped = int(np.count_nonzero(~keep))
            self._svd = (u[:, keep], s[keep], vt[keep])
            if self.dropped:
                warnings.warn(f"two-component response: {self.dropped} near-zero mode(s) of the "
                              f"SCC response left out (singular values "
                              f"{', '.join(f'{v:.1e}' for v in s[~keep])}; minimum-norm "
                              "solution); a soft spin mode is lost when the SCC tolerance is "
                              "loose", RuntimeWarning, stacklevel=4)

    def _solve(self, rhs) -> np.ndarray:
        nunk = self.amat.shape[0]
        if self._svd is not None:
            u, s, vt = self._svd
            return vt.T @ ((u.T @ rhs) / s[:, None])
        lhs = np.eye(nunk) - self.amat @ self.bmat
        if self.null is not None:
            r = self.null.shape[1]
            big = np.zeros((nunk + r, nunk + r))
            big[:nunk, :nunk] = lhs
            big[:nunk, nunk:] = self.null
            big[nunk:, :nunk] = self.null.T
            return np.linalg.solve(big, np.vstack([rhs, np.zeros((r, rhs.shape[1]))]))[:nunk]
        return np.linalg.solve(lhs, rhs)

    def _embed(self, v_ext) -> np.ndarray:
        """External potentials on the charge component's shells (n_shell, n_rhs) in the
        potentials' layout (4 nc, n_rhs)."""
        v = np.asarray(v_ext, float)
        out = np.zeros((4 * self.layout.nc,) + v.shape[1:])
        out[:self.layout.nshell] = v
        return out

    def potentials(self, v_ext) -> np.ndarray:
        """Total potentials (4 nc, n_rhs) for external charge potentials per shell."""
        v = self._embed(v_ext)
        if not self.scc:
            return v
        return v + self.bmat @ self._solve(self.amat @ v)

    def potentials_with(self, x0, v_ext) -> np.ndarray:
        """Total potentials when the populations also change by ``x0`` (unknowns' space) at
        fixed potentials (coordinate perturbations): (I - A B) x = x0 + A v_ext."""
        v = self._embed(v_ext)
        if not self.scc:
            return v
        return v + self.bmat @ self._solve(np.asarray(x0, float) + self.amat @ v)

    def populations(self, v_tot) -> np.ndarray:
        """(4, N, n_rhs) Mulliken populations of the four Pauli components."""
        lay = self.layout
        p = np.stack([self.x.full[b * lay.nr:b * lay.nr + lay.norb] for b in range(4)])
        return p @ v_tot

    def net(self, v_tot) -> np.ndarray:
        """(N, n_rhs) on-site charge populations."""
        return self.x.net @ v_tot


def scc_kernel(calc, st, sc) -> tuple:
    """(K_q, E, W) of :func:`pydftb.response.scc_kernel` with W the two-component spin
    constants (None without spin polarisation)."""
    kq, e, _ = R.scc_kernel(calc, st, sc)
    w = st.pauli.w if calc.scc else None
    return kq, e, None if w is None else np.asarray(w, float)


def make_response(calc, st, sc, gs, x: PauliChi0, kernel, scc: bool, omega: float, real,
                  skew) -> PauliResponse:
    """:class:`PauliResponse` with the solve of the static problem: bordered by the global
    rotations without spin-orbit coupling, SVD-cut with it; a plain solve at finite
    frequency."""
    null = cut = None
    if scc and omega == 0.0:
        if st.pauli.lam is None:
            if kernel[2] is not None:
                null = rotation_generators(calc, st, sc, x.layout, kernel[1].shape[1], real)
        else:
            cut = SVD_SCC * scc_error(sc)
    return PauliResponse(gs, kernel, x, scc, real, skew, null, cut)


# ---- level and Fermi-level derivatives -----------------------------------------------------
def _runs(blk) -> list:
    """The degenerate runs (more than one level) of a block."""
    return [r for r in (np.flatnonzero(blk.group == g) for g in np.unique(blk.group))
            if r.size > 1]


def _level_matrices(gs: PauliGroundState, resp: PauliResponse, blk, v_tot) -> tuple:
    """The potentials' part of M = C^+ dH C of a block for total potentials (4 nc, n_rhs): the
    diagonal (n_rhs, n_level) and M[r, r] (n_rhs, |r|, |r|; rows a, columns i) of every
    degenerate run r."""
    lay = resp.layout
    nrhs = v_tot.shape[1]
    lev = np.arange(len(blk.evals))
    cols = _columns(gs, lay, _row_elements(gs, blk, lev, lev, resp.real, resp.skew))
    diag = np.real(v_tot.T @ cols.reshape(4 * lay.nc, -1))
    runs = []
    for r in _runs(blk):
        ii, aa = np.meshgrid(r, r, indexing="ij")                       # element <a|dH|i>
        el = _columns(gs, lay, _row_elements(gs, blk, ii.ravel(), aa.ravel(), resp.real,
                                             resp.skew)).reshape(4 * lay.nc, -1)
        mrun = (v_tot.T @ el).reshape(nrhs, r.size, r.size)            # [rhs, i, a]
        runs.append(np.transpose(mrun, (0, 2, 1)))
    return diag, runs


def eigenvalue_derivatives(gs: PauliGroundState, resp: PauliResponse, v_tot) -> np.ndarray:
    """d eps of every k-point's 2N levels for total potentials (4 nc, n_rhs): the diagonal of
    M = C^+ dH C, in a degenerate run (Kramers pairs included) the eigenvalues of its block
    (ascending): (n_rhs, n_k, 2N)."""
    nrhs = v_tot.shape[1]
    nlev = max(len(blk.evals) for blk in gs.blocks)
    out = np.zeros((nrhs, len(gs.blocks), nlev))
    for b, blk in enumerate(gs.blocks):
        diag, runs = _level_matrices(gs, resp, blk, v_tot)
        out[:, b, :len(blk.evals)] = diag
        for r, mrun in zip(_runs(blk), runs):
            for c in range(nrhs):
                out[c, b, r] = np.linalg.eigvalsh(mrun[c])
    return out


def fermi_derivative(gs: PauliGroundState, resp: PauliResponse, v_tot) -> np.ndarray:
    """dE_F (n_rhs, 1): sum_k w_k sum_{i in W} delta_i M_ii / sum_k w_k sum_{i in W} delta_i."""
    lay = resp.layout
    d_cols = _columns(gs, lay, fermi_vectors(gs, lay, resp.real, resp.skew)).ravel()
    return (d_cols @ v_tot / gs.dos[0])[:, None]


# ---- the analyses --------------------------------------------------------------------------
def _setup(calc, st, sc, opts, omegas, with_net=False):
    gs = PauliGroundState(calc, st, sc, opts["degeneracy_tol"], opts["filling_tol"])
    real, skew = block_channels(calc, st)
    compat = bool(getattr(calc, "_response_eta_compat", False))
    xs = chi0(gs, omegas, opts["eta"], with_net=with_net, eta_compat=compat, real=real,
              skew=skew)
    return gs, real, skew, xs


def polarisability(calc, st, sc, spec, opts: dict) -> R.PolarisabilityResult:
    """:func:`pydftb.response.polarisability` of a two-component calculation: the populations
    of all four Pauli components respond (``charge_derivatives[w, i, b, A]``, b = q, m_x, m_y,
    m_z); one Fermi level."""
    omegas = R.frequencies(spec, "polarisability")
    gs, real, skew, xs = _setup(calc, st, sc, opts, omegas)
    kernel = scc_kernel(calc, st, sc)
    periodic = st.cell is not None and bool(np.any(st.pbc))
    pos = np.asarray(st.pos, float)
    v_ext = pos[gs.shell_atom]
    nw = len(omegas)
    orb = np.zeros((nw, 3, 4, gs.norb))
    dei = []
    dfermi = np.zeros((nw, 3, 1)) if gs.metallic else None
    for w, x in enumerate(xs):
        resp = make_response(calc, st, sc, gs, x, kernel, calc.scc, float(omegas[w]), real, skew)
        v_tot = resp.potentials(v_ext)
        orb[w] = np.transpose(resp.populations(v_tot), (2, 0, 1))
        dei.append(eigenvalue_derivatives(gs, resp, v_tot))
        if dfermi is not None and float(omegas[w]) == 0.0 and gs.fermi_shift:
            dfermi[w] = fermi_derivative(gs, resp, v_tot)
    atom = R._sum_atoms(orb, gs.orb_atom, gs.natom)
    tensor = None if periodic else -np.einsum("Aj,wiA->wji", pos, atom[:, :, 0, :])
    return R.PolarisabilityResult(omegas, tensor, atom, orb, dei, dfermi,
                                  gs.dos.copy() if gs.metallic else None, gs.metallic)


def response_kernel(calc, st, sc, spec, opts: dict) -> R.ResponseKernelResult:
    """:func:`pydftb.response.response_kernel` of a two-component calculation (four Pauli
    components of the populations; the net populations of the charge)."""
    spec_d = {} if spec is True else dict(spec)
    rpa = bool(spec_d.get("rpa", False)) or not calc.scc
    omegas = R.frequencies(spec, "response_kernel")
    gs, real, skew, xs = _setup(calc, st, sc, opts, omegas, with_net=True)
    kernel = scc_kernel(calc, st, sc)
    v_ext = np.eye(gs.natom)[gs.shell_atom]
    nw = len(omegas)
    orb = np.zeros((nw, gs.natom, 4, gs.norb))
    net = np.zeros((nw, gs.natom, gs.natom))
    dei = []
    dfermi = np.zeros((nw, gs.natom, 1)) if gs.metallic else None
    for w, x in enumerate(xs):
        scc = calc.scc and not rpa
        resp = make_response(calc, st, sc, gs, x, kernel, scc, float(omegas[w]), real, skew)
        v_tot = resp.potentials(v_ext)
        orb[w] = np.transpose(resp.populations(v_tot), (2, 0, 1))
        net[w] = R._sum_atoms(resp.net(v_tot).T, gs.orb_atom, gs.natom)
        dei.append(eigenvalue_derivatives(gs, resp, v_tot))
        if dfermi is not None and float(omegas[w]) == 0.0 and gs.fermi_shift:
            dfermi[w] = fermi_derivative(gs, resp, v_tot)
    atom = R._sum_atoms(orb, gs.orb_atom, gs.natom)
    frontier = [(0, blk.k, blk.n_filled - 1, blk.first_empty) for blk in gs.blocks]
    return R.ResponseKernelResult(omegas, rpa, atom, orb, net, dei, dfermi,
                                  gs.dos.copy() if gs.metallic else None, gs.metallic,
                                  frontier, gs.orb_atom.copy())


def external_charge_derivatives(calc, st, sc, spec, opts: dict) -> R.ExternalChargeDerivativesResult:
    """:func:`pydftb.response.external_charge_derivatives` of a two-component calculation: the
    point charges' potential acts on the charge; all four Pauli components respond (DFTB+'s
    development version prints the charge part)."""
    ncharge = len(st.ext.pc.charges)
    sel = R.external_charge_selection(spec, ncharge)
    charges = np.arange(ncharge) if sel is None else sel
    gs, real, skew, xs = _setup(calc, st, sc, opts, [0.0])
    resp = make_response(calc, st, sc, gs, xs[0], scc_kernel(calc, st, sc), calc.scc, 0.0, real,
                         skew)
    jac = st.ext.pc_jacobian()[:, charges, :]
    v_ext = jac[gs.shell_atom].reshape(gs.nshell, -1)
    orb = resp.populations(resp.potentials(v_ext))                     # (4, N, 3 n_j)
    orb = orb.reshape(4, gs.norb, len(charges), 3).transpose(2, 3, 0, 1)
    return R.ExternalChargeDerivativesResult(charges, R._sum_atoms(orb, gs.orb_atom, gs.natom),
                                             orb, gs.orb_atom.copy())


# ---- coordinate derivatives ----------------------------------------------------------------
def _component_populations(a, sc, n) -> np.ndarray:
    """(4, N) Mulliken populations of the four components of d rho = A C^+ (A = C X):
    Re sum_ss' sigma^b_ss' sum_j A_{s' mu,j} (S c_j)*_{s mu}."""
    return np.real(_pauli_sum(lambda s, t: (a[t * n:(t + 1) * n] * sc[s * n:(s + 1) * n].conj())
                              .sum(1)))


def _block_change(real, skew, bt, comp_rho, a, sc, d_s, lo, hi, n) -> tuple:
    """Real (4, nb) and skew (4, ns) block populations of the bare coordinate response: the dual
    populations Q_b = 1/2 (t_b + t_b^+), t_b = (comp_b(d rho) S + comp_b(rho) S')_AA (S' only
    couples atom K to the others); the real part feeds the real channels, the imaginary part
    the skew channels."""
    q_r = np.zeros((4, 0 if real is None else real.size))
    q_s = np.zeros((4, 0 if skew is None else skew.size))
    for kk, sl in enumerate(bt.slices):
        sr = np.flatnonzero(real.k == kk) if real is not None else np.zeros(0, int)
        ss = np.flatnonzero(skew.k == kk) if skew is not None else np.zeros(0, int)
        if not (sr.size or ss.size):
            continue
        ra = [a[u * n + sl.start:u * n + sl.stop] for u in (0, 1)]
        rs = [sc[s * n + sl.start:s * n + sl.stop].conj().T for s in (0, 1)]
        t = _pauli_sum(lambda s, u, ra=ra, rs=rs: ra[u] @ rs[s])
        if sl.start == lo:                                   # the displaced atom itself
            t = t + comp_rho[:, sl, :] @ d_s.T
        else:
            t = t + comp_rho[:, sl, lo:hi] @ d_s[:, sl]
        q = 0.5 * (t + np.conj(np.transpose(t, (0, 2, 1))))
        if sr.size:
            q_r[:, sr] = np.real(q[:, real.local_i[sr], real.local_j[sr]])
        if ss.size:
            q_s[:, ss] = np.imag(q[:, skew.local_i[ss], skew.local_j[ss]])
    return q_r, q_s


def coordinate_derivatives(calc, st, sc, spec, opts: dict) -> R.CoordDerivativesResult:
    """:func:`pydftb.response.coordinate_derivatives` of a two-component molecule (DFTB+ refuses
    it).  For x = x_K,c the spinor H' has the components
    H'_q = H0' + 1/2 S' (v_mu + v_nu) + 1/2 (S' V^q + V^q S') + i/2 (S' Vi^q + Vi^q S') and
    H'_a = 1/2 S' (v^a_mu + v^a_nu) + 1/2 (S' V^a + V^a S') + i/2 (S' Y^a + Y^a S'),
    Y^a = Vi^a plus, with dual spin-orbit coupling, Lambda^a (H_SO = i/2 (S Lambda^a + Lambda^a S)
    x sigma_a; the onsite model has no S); S'_2 = S' x 1.  M = C^+ H' C, N = C^+ S'_2 C (complex),
    X = W o (M - e_bar N) - f_bar N, d rho = C X C^+, plus rho S' in the four components'
    Mulliken and block populations; then the SCC response as for the field."""
    natom = len(st.species)
    sel = R.coordinate_atoms(spec, natom)
    atoms = np.arange(natom) if sel is None else sel
    gs, real, skew, xs = _setup(calc, st, sc, opts, [0.0])
    kernel = scc_kernel(calc, st, sc)
    resp = make_response(calc, st, sc, gs, xs[0], kernel, calc.scc, 0.0, real, skew)
    lay = resp.layout
    _, e_sh, _ = kernel
    no, nsh = gs.norb, gs.nshell
    pb = R._pair_derivatives(calc, st)
    ai = pb.i[pb.pair_of]
    v_orb, vs_orb = calc._potentials(st, sc.q_in)
    vc = np.zeros((4, no))
    vc[0] = np.asarray(v_orb, float)
    if vs_orb is not None:
        vc[1:] = np.asarray(vs_orb, float)
    vb = y = None                                        # block potentials, antisymmetric parts
    if st.block is not None:
        vblk, viblk = calc._block_potentials(st, sc.q_in)
        vb = np.zeros((4, no, no))
        for kk, sl in enumerate(st.block.slices):
            vb[:, sl, sl] = vblk[kk]
        if viblk is not None:
            y = np.zeros((4, no, no))
            for kk, sl in enumerate(st.block.slices):
                y[:, sl, sl] = viblk[kk]
    if st.pauli.lam is not None and st.pauli.dual:
        y = np.zeros((4, no, no)) if y is None else y
        y[1:] = y[1:] + np.asarray(st.pauli.lam, float)
    dq_in, _ = calc._charge_and_magnetisation(st, sc.q_in)
    jac = R.external_jacobian(st)
    if calc.scc:
        jac = jac + R.coulomb_jacobian(st, np.asarray(dq_in, float))
        if st.third is not None:
            jac = jac + R.third_order_jacobian(st, np.asarray(dq_in, float))
    offsets = np.asarray(st.basis.offsets)
    blk = gs.blocks[0]                                   # a molecule: one k-point
    cmat, scm = blk.c, blk.sc
    f = R._theta(gs.beta * (blk.evals - blk.fermi))
    w = R.static_weights(gs, blk)
    ebar = 0.5 * (blk.evals[:, None] + blk.evals[None, :])
    fbar = 0.5 * (f[:, None] + f[None, :])
    comp_rho = blk.weight * so.components((cmat * f) @ cmat.conj().T)
    win = blk.window
    dlt = R._delta(blk.evals[win], blk.fermi, gs.beta)
    runs = _runs(blk)
    ncol = 3 * len(atoms)
    nb, ns = lay.nb, lay.ns
    dq0 = np.zeros((4, no, ncol))
    dqr0 = np.zeros((4, nb, ncol))
    dqs0 = np.zeros((4, ns, ncol))
    dv = np.zeros((nsh, ncol))
    de0 = np.zeros((ncol, len(blk.evals)))
    runs0 = []
    efnum = np.zeros(ncol)
    for k_idx, K in enumerate(atoms):
        lo, hi = offsets[K], offsets[K] + int(np.count_nonzero(gs.orb_atom == K))
        rows = np.flatnonzero(ai == K)
        r_loc = pb.rows[rows] - lo
        cols = pb.cols[rows]
        idx = np.concatenate([np.arange(lo, hi), no + np.arange(lo, hi)])
        ck = cmat[idx]
        for c in range(3):
            col = 3 * k_idx + c
            d_h = np.zeros((hi - lo, no))
            d_s = np.zeros((hi - lo, no))
            np.add.at(d_h, (r_loc, cols), -pb.dh[rows, c])     # d/dR_K = -d/d(R_B - R_K)
            np.add.at(d_s, (r_loc, cols), -pb.ds[rows, c])
            comp = 0.5 * d_s[None] * (vc[:, lo:hi, None] + vc[:, None, :])
            comp = comp.astype(complex)
            comp[0] += d_h
            if vb is not None:
                comp += 0.5 * (d_s[None] @ vb + vb[:, lo:hi, lo:hi] @ d_s[None])
            if y is not None:
                comp += 0.5j * (d_s[None] @ y + y[:, lo:hi, lo:hi] @ d_s[None])
            z = ck.conj().T @ (so.spinor(*comp) @ cmat)
            m = z + z.conj().T
            zs = ck.conj().T @ (so.spinor(d_s, 0.0, 0.0, 0.0) @ cmat)
            nm = zs + zs.conj().T
            xm = w * (m - ebar * nm) - fbar * nm
            a = blk.weight * (cmat @ xm)
            q = _component_populations(a, scm, no)
            rk = np.real(comp_rho[:, lo:hi, :])
            q += (rk * d_s[None]).sum(1)
            q[:, lo:hi] += (rk * d_s[None]).sum(2)
            dq0[:, :, col] = q
            if nb + ns:
                dqr0[:, :, col], dqs0[:, :, col] = _block_change(real, skew, st.block, comp_rho,
                                                                 a, scm, d_s, lo, hi, no)
            diag = np.real(np.diag(m) - blk.evals * np.diag(nm))
            de0[col] = diag
            runs0.append([(m - ebar * nm)[np.ix_(r, r)] for r in runs])
            if win.size:
                efnum[col] = blk.weight * (dlt * diag[win]).sum()
            dv[:, col] = e_sh @ jac[:, K, c]
    ef0 = np.zeros(ncol)
    if gs.fermi_shift:                                   # the bare part keeps the electron count
        ef0 = efnum / gs.dos[0]
        d_rows = fermi_vectors(gs, lay, real, skew)       # (4, nr)
        dq0 += d_rows[:, :no, None] * ef0
        if nb:
            dqr0 += (d_rows[:, no:no + nb] * real.to_q)[:, :, None] * ef0
        if ns:
            dqs0 += (0.5 * d_rows[:, no + nb:])[:, :, None] * ef0
    # the SCC response: (I - A B) x = x0 + A dV, v_tot = dV + B x
    x0 = np.vstack([e_sh.T @ gs.shells(dq0[0])] + [gs.shells(dq0[b]) for b in (1, 2, 3)]
                   + [dqr0.reshape(4 * nb, ncol), dqs0.reshape(4 * ns, ncol)])
    v_tot = resp.potentials_with(x0, dv)
    orb = dq0 + resp.populations(v_tot)                  # (4, N, ncol)
    orb = orb.reshape(4, no, len(atoms), 3).transpose(2, 3, 0, 1)
    atom = R._sum_atoms(orb, gs.orb_atom, natom)
    diag, mruns = _level_matrices(gs, resp, blk, v_tot)
    de = de0 + diag
    for col in range(ncol):
        for j, r in enumerate(runs):
            de[col, r] = np.linalg.eigvalsh(runs0[col][j] + mruns[j][col])
    dei = de.reshape(len(atoms), 3, 1, -1)
    dfermi = None
    if gs.fermi_shift:
        dfermi = (ef0[:, None] + fermi_derivative(gs, resp, v_tot)).reshape(len(atoms), 3, 1)
    dq_out, _ = calc._charge_and_magnetisation(st, sc.q_out)
    dn_atom = np.zeros(natom)
    np.add.at(dn_atom, np.asarray(st.site_atom), np.asarray(dq_out, float))
    pos = np.asarray(st.pos, float)
    born = -np.einsum("Aj,kiA->kij", pos, atom[:, :, 0, :])
    born[:, np.arange(3), np.arange(3)] -= dn_atom[atoms][:, None]
    return R.CoordDerivativesResult(atoms, born, atom, orb, dei, dfermi,
                                    gs.dos.copy() if gs.metallic else None, gs.metallic,
                                    gs.orb_atom.copy())
