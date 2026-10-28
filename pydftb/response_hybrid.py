"""Responses with a hybrid functional (milestones M6h-4, M6h-4b, M6h-4c): DFTB+'s
``Polarisability``, ``ResponseKernel`` and the development version's
``ExternalChargeDerivatives`` on the real path (molecules and the Gamma point, collinear spin;
:class:`HybridResponse`), at k-points (:class:`KPointHybridResponse`) and with two-component
spin (molecules; :class:`PauliHybridResponse`), where DFTB+ 25.1 crashes (B24, B25); the entry
points of :mod:`pydftb.response` branch here when the calculation has a hybrid functional.

DFTB+ (``derivs/linearresponse.F90`` dRhoReal) adds the exchange of the first-order density
matrix, H_x[P1], to the first-order Hamiltonian and iterates the response SCC on the dense P1
with the ground state's mixer.  The kernel of :mod:`pydftb.response` acts on populations; the
exchange needs all of P1, so pydftb writes the static response in the space of level pairs:

* chi0 of a (spin) block: X = fill w W o (C^T H1 C), P1 = C X C^T, with W <= 0 the static pair
  weights of :func:`pydftb.response.static_weights` (DFTB+'s window and theta weights, the
  intraband -delta, the degenerate runs); a metal without a fixed Fermi level adds
  fill w delta_i dE_F on the window's diagonal, dE_F = sum w delta M_ii / sum w delta, per
  spin channel (or shared, as :func:`pydftb.response.chi0`);
* D = sqrt(-fill w W), T z = C (D o z) C^T and T^T H = D o (C^T H C), whose window diagonal is
  projected orthogonal to u ~ sqrt(w delta) in every Fermi group (that projection is the move
  of the Fermi level): chi0 = -T T^T;
* P1 = -T z with (I + T^T K T) z = T^T H_ext, K the response potentials of P1: gamma and the
  full third order's Hessian on the shell populations, the spin constants on the shell
  magnetisations, the block kernel of :class:`pydftb.response.BlockChannels` on the dual block
  populations 1/2 (P1 S + S P1) (operator 1/2 (S V + V S)), and the exchange.  These are second
  derivatives of the energy, so the system is symmetric; it is positive definite when the
  ground state is a minimum of the SCC, but the (negative) exchange can make it indefinite: a
  hot metal's SCC solution is then a saddle point (Cu4 with LC at 1000 K has five eigenvalues
  near -1.7), whose response DFTB+'s mixing still finds.  pydftb solves it by MINRES (minimal
  residuals, for symmetric systems, definite or not) on all right-hand sides at once (one
  batched exchange build per iteration), to a relative residual of :data:`SOLVE_TOL`, and warns
  when it meets negative curvature.

The exchange is the exact operator, :meth:`pydftb.hybrid.HybridTerms.response_fock`: the
Thresholded screening (whose increments DFTB+ continues from the ground state's, which adds its
screening error, 4e-6 of water's alpha) is not applied.  DFTB+ 25.1 (and the development
version) never updates the block populations in a hybrid's response SCC, so the +U and onsite
potentials do not respond (B26); ``DFTB._response_blocks_compat`` (testing) leaves them out too.
``DFTB._response_pc_compat`` (testing, also without a hybrid: :mod:`pydftb.response`) adds the
point charges' potential to the perturbation, as DFTB+ 25.1 does in every response (B27).

The complex paths (M6h-4b) keep the equation: z is Hermitian per block and the products take
their real part.  At k-points every (spin, k) block's P1 carries w_k and the exchange is the
ground state's k-point operator on dP(k) = P1(k) / w_k (its Thresholded screening's limit; the
MinimumImage NeighbourBased operator is not self-adjoint, B22, and refused).  Two-component: the
populations of the four Pauli components, H1 = sum_b [1/2 S o (V^b_mu + V^b_nu)] x sigma^b, and
the global spin rotations (zero modes without spin-orbit coupling) projected out.

Finite frequencies (M6h-4c; DFTB+ refuses them with a hybrid): for an e^{-iwt} perturbation P1
has a symmetric (Hermitian) part P^s and an antisymmetric (anti-Hermitian) one P^a; populations
see only P^s, the exchange both.  With Omega_ij = w / (e_i - e_j) on the pairs of different
degenerate runs (the others have F = 0 at w != 0; no Fermi-level move, as DFTB+) and
G = 1 / sqrt(1 + Omega^2), z^a = G y:

    [[I + D K_s D, -Omega G], [-G Omega, G (I + D K_x D) G]] [z^s; y] = [D o M_ext; 0],

K_s the static response potentials of P^s, K_x the exchange of P^a (two-component: its upper-
right spin block the negative adjoint of the lower-left one).  Symmetric, indefinite above an
excitation; w = 0 is the static equation.  MINRES, and when it stalls near or among
excitations a dense solve of small systems on the physical space, else a warning.  The
two-component rotations are not projected at w != 0 (no zero modes there) but near-null at
small w (eigenvalues ~ w^2): the right-hand side has no part along them, rounding has, so
MINRES may stall on a residual plateau just above its tolerance (accepted) and the dense solve
amplifies the rounding by 1 / w^2 (so it comes only after MINRES fails).
The eta -> 0 limit of DFTB+'s weights (the populations of M6g's and M6g-3b's finite-frequency
responses without the exchange).

NumPy engine only; coordinate derivatives are milestone M6h-4d.
"""
from __future__ import annotations

import warnings

import numpy as np

from . import response as R

SOLVE_TOL = 1e-13         # relative residual of every right-hand side
SOLVE_MAX = 1000          # iterations
NEGATIVE_CURVATURE = -1e-8   # Rayleigh quotient of a Lanczos vector: an indefinite equation
ROTATION_REL = 1e-8       # spin-rotation zero modes below this of the largest: no rotation
RHS_ELEMENTS = 2_000_000  # right-hand sides per batch: n_rhs n_block n_orb^2 at most this
DENSE_LIMIT = 3000        # finite frequency: a dense solve when MINRES stalls, unknowns
OMEGA_PLATEAU = 1e-11     # finite frequency without the dense solve: a relative residual of
PLATEAU_STEPS = 100       # at most this not halved in this many iterations is accepted


def _real(x) -> np.ndarray:
    """A real matrix of the real path (the Gamma point's are real already)."""
    x = np.asarray(x)
    if np.iscomplexobj(x):
        raise ValueError("hybrid functionals: complex eigenvectors on the real response path "
                         "(k-points and two-component spin have their own classes)")
    return np.asarray(x, float)


def _dot(a, b) -> np.ndarray:
    """Per right-hand side, sum over the blocks of the Frobenius products."""
    return sum(np.einsum("rij,rij->r", x, y) for x, y in zip(a, b))


def _scale(num, den, x) -> np.ndarray:
    """(num / den) x per right-hand side, zero where den is zero."""
    f = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0)
    return f[:, None, None] * x


class HybridResponse:
    """The static response of one hybrid ground state (see the module docstring); ``rpa``: the
    bare chi0 (no SCC, no exchange)."""

    def __init__(self, calc, st, sc, opts: dict, rpa: bool = False):
        self.gs = gs = R.GroundState(calc, st, sc, opts["degeneracy_tol"], opts["filling_tol"])
        self.scc = calc.scc and not rpa
        self.kq, self.e, self.w = R.scc_kernel(calc, st, sc)
        ch = None if getattr(calc, "_response_blocks_compat", False) else R.block_channels(st)
        self.channels = ch if ch is not None and ch.size else None
        self.hybrid = st.hybrid
        self.n, self.ns = gs.norb, gs.nspin
        self.s = _real(gs.blocks[0].s)
        self.c, self.d = [], []
        for blk in gs.blocks:
            self.c.append(_real(blk.c))
            w = R.static_weights(gs, blk)                    # <= 0: F_ia <= 0, -delta
            self.d.append(np.sqrt(np.clip(-gs.max_fill * blk.weight * w, 0.0, None)))
        self.groups = self._fermi_groups()
        self.nqm, self.nmat = self.ns, self.n        # population components, matrix size
        self.iterations = 0                              # most MINRES steps of a solve
        self.indefinite = False

    def _fermi_groups(self) -> list:
        """The Fermi groups (one channel, or both with a shared Fermi level): the window levels
        whose electron count is kept and the unit vector u ~ sqrt(w delta) of the projection."""
        gs, groups = self.gs, []
        if gs.fermi_shift:
            chans = ([list(range(len(gs.blocks)))] if gs.shared_fermi else
                     [[b for b, blk in enumerate(gs.blocks) if blk.spin == s]
                      for s in range(self.ns)])
            for members in chans:
                parts = [(b, gs.blocks[b].window) for b in members if gs.blocks[b].window.size]
                if parts:
                    u = np.concatenate([np.diag(self.d[b])[win] for b, win in parts])
                    groups.append((parts, u / np.linalg.norm(u)))
        return groups

    # ---- T and T^T -----------------------------------------------------------------------
    def t_apply(self, z) -> list:
        """P1 of every spin channel (n_rhs, n, n) from z of every block: -C (D o z) C^T."""
        out = [np.zeros((z[0].shape[0], self.n, self.n)) for _ in range(self.ns)]
        for b, blk in enumerate(self.gs.blocks):
            out[blk.spin] -= self.c[b] @ (self.d[b] * z[b]) @ self.c[b].T
        return out

    def t_adjoint(self, hs) -> list:
        """z of every block from H of every spin channel: D o (C^T H C), projected."""
        z = [self.d[b] * (self.c[b].T @ hs[blk.spin] @ self.c[b])
             for b, blk in enumerate(self.gs.blocks)]
        return self._fermi_project(z)

    def _fermi_project(self, z) -> list:
        """The window diagonal of every Fermi group orthogonal to its u (the Fermi level's
        move), in place."""
        for parts, u in self.groups:
            y = np.concatenate([z[b][:, win, win] for b, win in parts], axis=1)
            y = y - (y @ u)[:, None] * u[None, :]
            o = 0
            for b, win in parts:
                z[b][:, win, win] = y[:, o:o + len(win)]
                o += len(win)
        return z

    # ---- the response potentials ---------------------------------------------------------
    def populations(self, ps) -> list:
        """Mulliken populations diag(P S) of every spin channel, (n_rhs, n)."""
        return [np.einsum("rij,ji->ri", p, self.s) for p in ps]

    def _h_shell(self, v_orb) -> np.ndarray:
        """1/2 S o (v_mu + v_nu) of orbital potentials (n_rhs, n)."""
        return 0.5 * self.s * (v_orb[:, :, None] + v_orb[:, None, :])

    def kernel(self, ps) -> list:
        """H1 of the potentials that P1 induces, per spin channel (n_rhs, n, n)."""
        gs, nr = self.gs, ps[0].shape[0]
        hs = [np.zeros((nr, self.n, self.n)) for _ in range(self.ns)]
        if not self.scc:
            return hs
        qsh = [gs.shells(q.T) for q in self.populations(ps)]          # (n_shell, n_rhs)
        v = self.e @ (self.kq @ (self.e.T @ sum(qsh)))
        vm = self.w @ (qsh[0] - qsh[1]) if self.ns == 2 and self.w is not None else None
        for s in range(self.ns):
            vs = v if vm is None else (v + vm if s == 0 else v - vm)
            hs[s] += self._h_shell(vs[gs.orb_shell].T)
        ch = self.channels
        if ch is not None:                               # block potentials (+U, onsite)
            qb = []
            for p in ps:
                ps_ = p @ self.s
                qb.append(0.5 * (ps_ + np.transpose(ps_, (0, 2, 1)))[:, ch.mu, ch.nu])
            comp = qb if self.ns == 1 else [qb[0] + qb[1], qb[0] - qb[1]]
            vb = ch.kernel @ np.concatenate(comp, axis=1).T              # (n_comp nb, n_rhs)
            vb = [vb[i * ch.size:(i + 1) * ch.size] for i in range(ch.n_comp)]
            for s in range(self.ns):
                vc = vb[0] if self.ns == 1 else (vb[0] + vb[1] if s == 0 else vb[0] - vb[1])
                vmat = np.zeros((nr, self.n, self.n))
                vmat[:, ch.mu, ch.nu] = vc.T
                vmat[:, ch.nu, ch.mu] = vc.T
                sv = self.s @ vmat
                hs[s] += 0.5 * (sv + np.transpose(sv, (0, 2, 1)))
        for s, hx in enumerate(self.hybrid.response_fock(ps)):
            hs[s] += hx
        return hs

    # ---- solving -------------------------------------------------------------------------
    _inner = staticmethod(_dot)                          # the real Frobenius products

    def _project(self, z) -> list:
        """Directions the solution leaves out (none here; the two-component spin rotations)."""
        return z

    def _operator(self, z) -> list:
        """(I + T^T K T) z (t_apply carries the minus sign of P1 = -T z)."""
        k = self.t_adjoint(self.kernel(self.t_apply(z)))
        return [a - b for a, b in zip(z, k)]

    def _minres(self, b, op=None, maxit: int = SOLVE_MAX, strict: bool = True,
                curvature: bool = True, plateau: float | None = None) -> list:
        """MINRES on every right-hand side of ``b`` (z of every block): Paige and Saunders'
        Lanczos process with the QR factorisation of its tridiagonal by Givens rotations
        (SciPy's minres without a preconditioner), each column with its own scalars, until the
        residual estimate is SOLVE_TOL of |b|.  A negative Rayleigh quotient of a Lanczos
        vector marks the equation as indefinite (``curvature``).  ``op``: the operator
        (:meth:`_operator`); after ``maxit`` iterations an error, or with ``strict`` False the
        last iterate (``self._converged`` False, ``self._residual`` the worst relative
        residual).  ``plateau``: a worst relative residual of at most this that has not halved
        in PLATEAU_STEPS iterations ends the iteration as converged (its attainable accuracy;
        ``self._residual`` that residual)."""
        op = self._operator if op is None else op
        self._converged, self._residual = True, 0.0
        hist = []
        nr = b[0].shape[0]
        beta1 = np.sqrt(self._inner(b, b))
        x = [np.zeros_like(v) for v in b]
        w = [np.zeros_like(v) for v in b]
        w2 = [np.zeros_like(v) for v in b]
        r1 = [v.copy() for v in b]
        r2 = [v.copy() for v in b]
        oldb, beta, dbar, epsln = np.zeros(nr), beta1.copy(), np.zeros(nr), np.zeros(nr)
        phibar, cs, sn = beta1.copy(), -np.ones(nr), np.zeros(nr)
        stop = SOLVE_TOL * beta1
        for it in range(1, maxit + 2):
            active = phibar > stop
            if not active.any():
                break
            if plateau is not None:                      # finite frequency (M6h-4c)
                worst = float(np.max(phibar[active] / beta1[active]))
                if (len(hist) >= PLATEAU_STEPS and worst <= plateau
                        and worst > 0.5 * hist[-PLATEAU_STEPS]):
                    self._residual = worst
                    break
                hist.append(worst)
            if it > maxit:
                worst = float(np.max(phibar[active] / beta1[active]))
                if strict:
                    raise RuntimeError(f"the hybrid response did not converge in {maxit} "
                                       f"iterations (relative residual {worst:.1e})")
                self._converged, self._residual = False, worst
                break
            v = [_scale(1.0, beta, y) for y in r2]              # the Lanczos vector
            y = op(v)
            if it >= 2:
                y = [a - _scale(beta, oldb, c) for a, c in zip(y, r1)]
            alfa = self._inner(v, y)
            if curvature and np.any(alfa[active] < NEGATIVE_CURVATURE):
                self.indefinite = True
            y = [a - _scale(alfa, beta, c) for a, c in zip(y, r2)]
            r1, r2 = r2, y
            oldb, beta = beta, np.sqrt(self._inner(r2, r2))
            oldeps = epsln                                     # the previous rotation
            delta = cs * dbar + sn * alfa
            gbar = sn * dbar - cs * alfa
            epsln = sn * beta
            dbar = -cs * beta
            gamma = np.maximum(np.hypot(gbar, beta), np.finfo(float).eps)
            cs, sn = gbar / gamma, beta / gamma                # the next one
            phi = np.where(active, cs * phibar, 0.0)
            phibar = np.where(active, sn * phibar, phibar)
            w1, w2 = w2, w
            w = [(a - oldeps[:, None, None] * c - delta[:, None, None] * d) / gamma[:, None, None]
                 for a, c, d in zip(v, w1, w2)]
            x = [a + phi[:, None, None] * c for a, c in zip(x, w)]
        self.iterations = max(self.iterations, it - 1)
        return x

    def solve(self, v_shell) -> tuple:
        """The response to external shell potentials ``v_shell`` (n_shell, n_rhs), the same
        for both spin channels: populations (n_rhs, n_qm, n_orb) as charge (and
        magnetisation), on-site populations (n_rhs, n_orb) of the charge, eigenvalue
        derivatives (n_rhs, n_block, n_level) and the Fermi level's (n_rhs, n_spin) or None."""
        gs = self.gs
        nrhs = v_shell.shape[1]
        step = max(1, RHS_ELEMENTS // (len(gs.blocks) * self.nmat * self.nmat))
        nlev = max(len(blk.evals) for blk in gs.blocks)
        orb = np.zeros((nrhs, self.nqm, self.n))
        net = np.zeros((nrhs, self.n))
        dei = np.zeros((nrhs, len(gs.blocks), nlev))
        efnum = np.zeros((nrhs, self.ns))
        for lo in range(0, nrhs, step):
            cols = slice(lo, min(lo + step, nrhs))
            h_ext = self._external(v_shell[gs.orb_shell, cols].T)
            b = self._project(self.t_adjoint(h_ext))
            ps = self.t_apply(self._minres(b) if self.scc else b)
            hs = [x + y for x, y in zip(h_ext, self.kernel(ps))]
            orb[cols] = self._orbital_populations(ps)
            net[cols] = self._net(ps)
            for k, blk in enumerate(gs.blocks):
                m = self._block_matrix(hs, k, blk)
                diag = np.einsum("rii->ri", m).real
                de = diag.copy()
                for gid in np.unique(blk.group):
                    idx = np.flatnonzero(blk.group == gid)
                    if idx.size > 1:                     # degenerate run: its block's eigenvalues
                        de[:, idx] = np.linalg.eigvalsh(m[:, idx][:, :, idx])
                dei[cols, k, :len(blk.evals)] = de
                win = blk.window
                if win.size:
                    dlt = blk.weight * R._delta(blk.evals[win], blk.fermi, gs.beta)
                    efnum[cols, blk.spin] += diag[:, win] @ dlt
        if self.indefinite:
            warnings.warn("the hybrid response equation is indefinite: the ground state is a "
                          "saddle point of the SCC (unstable), not a minimum; the response is "
                          "that of this stationary state (as DFTB+'s)", RuntimeWarning,
                          stacklevel=3)
        dfermi = None
        if gs.metallic:
            dfermi = np.zeros((nrhs, self.ns))
            if gs.fermi_shift:
                if gs.shared_fermi:
                    dfermi[:] = efnum.sum(1, keepdims=True) / gs.dos.sum()
                else:
                    for s in range(self.ns):
                        if gs.dos[s] > 0.0:
                            dfermi[:, s] = efnum[:, s] / gs.dos[s]
        return orb, net, dei, dfermi

    # the pieces of solve that depend on the blocks' layout
    def _external(self, v_orb) -> list:
        """H of the external orbital potentials (n_rhs, n), as the kernel's H1."""
        h = self._h_shell(v_orb)
        return [h] * self.ns

    def _orbital_populations(self, ps) -> np.ndarray:
        """(n_rhs, n_qm, n): charge (and magnetisation)."""
        pop = self.populations(ps)
        return np.stack(pop if self.ns == 1 else [pop[0] + pop[1], pop[0] - pop[1]], axis=1)

    def _net(self, ps) -> np.ndarray:
        """On-site populations of the charge (n_rhs, n)."""
        return sum(np.einsum("rii->ri", p) for p in ps)

    def _block_matrix(self, hs, k: int, blk) -> np.ndarray:
        """C^+ H1 C of block k."""
        return self.c[k].T @ hs[blk.spin] @ self.c[k]

    # ---- finite frequencies (milestone M6h-4c) -------------------------------------------
    _complex = False                                     # the blocks' matrices are real

    def _t_d(self, z, d) -> list:
        """:meth:`t_apply` with the weights ``d`` of every block."""
        out = [np.zeros((z[0].shape[0], self.n, self.n)) for _ in range(self.ns)]
        for b, blk in enumerate(self.gs.blocks):
            out[blk.spin] -= self.c[b] @ (d[b] * z[b]) @ self.c[b].T
        return out

    def _tt_d(self, hs, d) -> list:
        """:meth:`t_adjoint` with the weights ``d``, without the Fermi level's move."""
        return [d[b] * (self.c[b].T @ hs[blk.spin] @ self.c[b])
                for b, blk in enumerate(self.gs.blocks)]

    def _exchange_a(self, ps) -> list:
        """H_x of the antisymmetric part of P1 (every spin channel)."""
        return self.hybrid.response_fock(ps)

    def _zero_modes(self) -> bool:
        """Whether the static equation has zero modes (the two-component spin rotations)."""
        return False

    def _omega_weights(self, omega: float) -> tuple:
        """Per block D without the pairs of one degenerate run (the intraband ones included:
        F = 0 there at omega != 0), Omega = omega / (e_i - e_j) and G = 1 / sqrt(1 + Omega^2)."""
        dfin, om, g = [], [], []
        for b, blk in enumerate(self.gs.blocks):
            e = blk.evals
            d = self.d[b].copy()
            d[blk.group[:, None] == blk.group[None, :]] = 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                o = np.where(d > 0.0, omega / (e[:, None] - e[None, :]), 0.0)
            dfin.append(d)
            om.append(o)
            g.append(1.0 / np.sqrt(1.0 + o * o))
        return dfin, om, g

    def _operator_omega(self, x, weights) -> list:
        """[[I + D K_s D, -Omega G], [-G Omega, G (I + D K_x D) G]] on [z^s of every block...,
        y of every block...] (z^a = G y): K_s every response potential of P^s, K_x the exchange
        of P^a."""
        dfin, om, g = weights
        nb = len(dfin)
        zs, za = x[:nb], [g[b] * x[nb + b] for b in range(nb)]
        ks = self._tt_d(self.kernel(self._t_d(zs, dfin)), dfin)
        kx = self._tt_d(self._exchange_a(self._t_d(za, dfin)), dfin)
        return ([zs[b] - ks[b] - om[b] * za[b] for b in range(nb)]
                + [g[b] * (za[b] - kx[b] - om[b] * zs[b]) for b in range(nb)])

    def _pair_unknowns(self, dfin) -> int:
        """Real unknowns of the finite-frequency system: the pairs i < j with D > 0, two parts,
        real and imaginary for complex blocks."""
        pairs = sum(int(np.count_nonzero(np.triu(d > 0.0, 1))) for d in dfin)
        return 2 * pairs * (2 if self._complex else 1)

    def _solve_stacked(self, rhs, weights, omega: float) -> list:
        """MINRES on the finite-frequency system (indefinite above an excitation, so without the
        saddle-point flag).  Small systems (at most DENSE_LIMIT unknowns) without zero modes:
        not converged after as many iterations as unknowns, the dense solve
        (:meth:`_dense_solve`).  With the two-component rotations (near-null at small w:
        rounding along them can stall MINRES just above its tolerance, and the dense solve
        amplifies it by 1 / w^2) and in large systems: MINRES to SOLVE_MAX, a stagnating
        residual of at most OMEGA_PLATEAU accepted; then the dense solve (small systems) or a
        warning and the last iterate."""
        dim = self._pair_unknowns(weights[0])
        small, zero = dim <= DENSE_LIMIT, self._zero_modes()
        op = lambda x: self._operator_omega(x, weights)          # noqa: E731
        x = self._minres(rhs, op, min(dim, SOLVE_MAX) if small and not zero else SOLVE_MAX,
                         strict=False, curvature=False,
                         plateau=OMEGA_PLATEAU if zero or not small else None)
        if self._converged:
            return x
        if small:
            return self._dense_solve(rhs, op, weights[0])
        extra = (" or, with two-component spin, to the omega -> 0 pole of the global spin "
                 "rotations" if zero else "")
        warnings.warn(f"the hybrid response at omega = {omega:g} Ha did not converge in "
                      f"{SOLVE_MAX} MINRES iterations (relative residual {self._residual:.1e}): "
                      f"the frequency is close to or among excitations of the response "
                      f"equation{extra}; this is the last iterate (the dense solve of its {dim} "
                      f"unknowns: pydftb.response_hybrid.DENSE_LIMIT >= {dim})", RuntimeWarning,
                      stacklevel=4)
        return x

    def _dense_solve(self, rhs, op, dfin) -> list:
        """The finite-frequency system as a dense matrix on its physical space -- z^s symmetric
        (Hermitian), y antisymmetric (anti-Hermitian) on the pairs i < j with D > 0, in the
        orthonormal basis (e_ij +- e_ji) / sqrt 2 and, for complex blocks, i (e_ij -+ e_ji) /
        sqrt 2 -- the operator applied to the basis in batches, solved by LU: accurate to the
        condition number times the rounding near an excitation, where MINRES stalls."""
        idx = [np.argwhere(np.triu(d > 0.0, 1)) for d in dfin]
        dim = self._pair_unknowns(dfin)
        a = np.empty((dim, dim))
        step = max(1, RHS_ELEMENTS // (2 * len(idx) * self.nmat * self.nmat))
        for lo in range(0, dim, step):
            hi = min(dim, lo + step)
            e = np.zeros((dim, hi - lo))
            e[np.arange(lo, hi), np.arange(hi - lo)] = 1.0
            a[:, lo:hi] = self._coordinates(op(self._from_coordinates(e, idx)), idx)
        return self._from_coordinates(np.linalg.solve(a, self._coordinates(rhs, idx)), idx)

    def _coordinates(self, x, idx) -> np.ndarray:
        """(n_unknowns, n_rhs) coordinates of [z^s..., y...] in :meth:`_dense_solve`'s basis."""
        nb, out = len(idx), []
        for part, sign in ((0, 1.0), (1, -1.0)):
            for b, ij in enumerate(idx):
                m, i, j = x[part * nb + b], ij[:, 0], ij[:, 1]
                out.append(((m[:, i, j] + sign * m[:, j, i]) * _SQRT_HALF).real)
                if self._complex:                            # i (e_ij - sign e_ji) / sqrt 2
                    out.append(((-1j * m[:, i, j] + sign * 1j * m[:, j, i]) * _SQRT_HALF).real)
        return np.concatenate(out, axis=1).T

    def _from_coordinates(self, c, idx) -> list:
        """[z^s..., y...] of coordinates (n_unknowns, n_rhs) in :meth:`_dense_solve`'s basis."""
        nb, nr = len(idx), c.shape[1]
        dtype = complex if self._complex else float
        out = [np.zeros((nr, self.nmat, self.nmat), dtype) for _ in range(2 * nb)]
        o = 0
        for part, sign in ((0, 1.0), (1, -1.0)):
            for b, ij in enumerate(idx):
                m, i, j, k = out[part * nb + b], ij[:, 0], ij[:, 1], len(ij)
                cr = c[o:o + k].T * _SQRT_HALF
                o += k
                m[:, i, j] += cr
                m[:, j, i] += sign * cr
                if self._complex:
                    ci = c[o:o + k].T * _SQRT_HALF
                    o += k
                    m[:, i, j] += 1j * ci
                    m[:, j, i] -= sign * 1j * ci
        return out

    def solve_omega(self, v_shell, omega: float) -> tuple:
        """:meth:`solve` at a frequency omega != 0 (M6h-4c): the populations, on-site
        populations and level derivatives of the symmetric part P^s (the antisymmetric one has
        none); no Fermi-level move (the Fermi level's derivatives are zero)."""
        gs = self.gs
        weights = self._omega_weights(omega)
        dfin, om = weights[0], weights[1]
        nrhs = v_shell.shape[1]
        step = max(1, RHS_ELEMENTS // (2 * len(gs.blocks) * self.nmat * self.nmat))
        nlev = max(len(blk.evals) for blk in gs.blocks)
        orb = np.zeros((nrhs, self.nqm, self.n))
        net = np.zeros((nrhs, self.n))
        dei = np.zeros((nrhs, len(gs.blocks), nlev))
        for lo in range(0, nrhs, step):
            cols = slice(lo, min(lo + step, nrhs))
            h_ext = self._external(v_shell[gs.orb_shell, cols].T)
            b = self._tt_d(h_ext, dfin)
            if self.scc:
                zs = self._solve_stacked(b + [np.zeros_like(v) for v in b], weights,
                                         omega)[:len(b)]
            else:                                        # the bare response (RPA)
                zs = [v / (1.0 - o * o) for v, o in zip(b, om)]
            ps = self._t_d(zs, dfin)
            hs = [x + y for x, y in zip(h_ext, self.kernel(ps))]
            orb[cols] = self._orbital_populations(ps)
            net[cols] = self._net(ps)
            for k, blk in enumerate(gs.blocks):
                m = self._block_matrix(hs, k, blk)
                de = np.einsum("rii->ri", m).real.copy()
                for gid in np.unique(blk.group):
                    idx = np.flatnonzero(blk.group == gid)
                    if idx.size > 1:                     # degenerate run: its block's eigenvalues
                        de[:, idx] = np.linalg.eigvalsh(m[:, idx][:, :, idx])
                dei[cols, k, :len(blk.evals)] = de
        return orb, net, dei, np.zeros((nrhs, self.ns)) if gs.metallic else None


_SQRT_HALF = 1.0 / np.sqrt(2.0)


def _cdot(a, b) -> np.ndarray:
    """Per right-hand side, sum over the blocks of Re Tr(a^+ b) (Hermitian blocks)."""
    return sum(np.einsum("rij,rij->r", x.conj(), y).real for x, y in zip(a, b))


class KPointHybridResponse(HybridResponse):
    """The static response at k-points (milestone M6h-4b; collinear): one complex block per
    (spin, k), P1 of every block weighted by w_k (D as M6h-4's), z Hermitian per block and the
    real part of the Frobenius products; populations sum_k Re diag(P1(k) S(k)), the block
    potentials of the dual populations sum_k Re 1/2 (P1 S + S P1) with 1/2 (S(k) V + V S(k)),
    and the exchange of dP(k) = P1(k) / w_k by the ground state's k-point operator
    (:meth:`pydftb.hybrid.KPointHybridTerms.response_fock`)."""

    def __init__(self, calc, st, sc, opts: dict, rpa: bool = False):
        self.gs = gs = R.GroundState(calc, st, sc, opts["degeneracy_tol"], opts["filling_tol"])
        self.scc = calc.scc and not rpa
        self.kq, self.e, self.w = R.scc_kernel(calc, st, sc)
        ch = None if getattr(calc, "_response_blocks_compat", False) else R.block_channels(st)
        self.channels = ch if ch is not None and ch.size else None
        self.hybrid = st.hybrid
        self.n, self.ns = gs.norb, gs.nspin
        self.nk = len(st.kw)
        self.c = [np.asarray(blk.c) for blk in gs.blocks]
        self.ch = [c.conj().T for c in self.c]
        self.sk = [np.asarray(blk.s) for blk in gs.blocks]
        self.d = [np.sqrt(np.clip(-gs.max_fill * blk.weight * R.static_weights(gs, blk), 0.0,
                                  None)) for blk in gs.blocks]
        self.groups = self._fermi_groups()
        self.nqm, self.nmat = self.ns, self.n
        self.iterations = 0
        self.indefinite = False

    _inner = staticmethod(_cdot)

    def t_apply(self, z) -> list:
        """P1 of every block from z of every block: -C (D o z) C^+."""
        return [-(self.c[b] @ (self.d[b] * z[b]) @ self.ch[b]) for b in range(len(z))]

    def t_adjoint(self, hs) -> list:
        """z of every block from H of every block: D o (C^+ H C), projected."""
        return self._fermi_project([self.d[b] * (self.ch[b] @ hs[b] @ self.c[b])
                                    for b in range(len(hs))])

    def populations(self, ps) -> list:
        out = [0.0] * self.ns
        for b, blk in enumerate(self.gs.blocks):
            out[blk.spin] = out[blk.spin] + np.einsum("rij,ji->ri", ps[b], self.sk[b]).real
        return out

    def _h_block(self, v_orb, b: int) -> np.ndarray:
        return 0.5 * self.sk[b] * (v_orb[:, :, None] + v_orb[:, None, :])

    def kernel(self, ps) -> list:
        """H1 of every block."""
        gs, nr = self.gs, ps[0].shape[0]
        hs = [np.zeros((nr, self.n, self.n), complex) for _ in gs.blocks]
        if not self.scc:
            return hs
        qsh = [gs.shells(q.T) for q in self.populations(ps)]
        v = self.e @ (self.kq @ (self.e.T @ sum(qsh)))
        vm = self.w @ (qsh[0] - qsh[1]) if self.ns == 2 and self.w is not None else None
        for b, blk in enumerate(gs.blocks):
            vs = v if vm is None else (v + vm if blk.spin == 0 else v - vm)
            hs[b] += self._h_block(vs[gs.orb_shell].T, b)
        ch = self.channels
        if ch is not None:                               # block potentials (+U, onsite)
            qb = [0.0] * self.ns
            for b, blk in enumerate(gs.blocks):
                ps_ = ps[b] @ self.sk[b]
                dual = 0.5 * (ps_ + np.conj(np.transpose(ps_, (0, 2, 1))))
                qb[blk.spin] = qb[blk.spin] + dual[:, ch.mu, ch.nu].real
            comp = qb if self.ns == 1 else [qb[0] + qb[1], qb[0] - qb[1]]
            vb = ch.kernel @ np.concatenate(comp, axis=1).T
            vb = [vb[i * ch.size:(i + 1) * ch.size] for i in range(ch.n_comp)]
            for b, blk in enumerate(gs.blocks):
                vc = vb[0] if self.ns == 1 else (vb[0] + vb[1] if blk.spin == 0 else vb[0] - vb[1])
                vmat = np.zeros((nr, self.n, self.n))
                vmat[:, ch.mu, ch.nu] = vc.T
                vmat[:, ch.nu, ch.mu] = vc.T
                sv = self.sk[b] @ vmat
                hs[b] += 0.5 * (sv + np.conj(np.transpose(sv, (0, 2, 1))))
        for b, hx in enumerate(self._exchange(ps)):
            hs[b] += hx
        return hs

    def _exchange(self, ps) -> list:
        """H_x of every block's P1: dP(k) = P1(k) / w_k of every spin channel through the ground
        state's k-point operator."""
        gs, nk = self.gs, self.nk
        dps = [np.stack([ps[s * nk + k] / gs.blocks[s * nk + k].weight for k in range(nk)],
                        axis=1) for s in range(self.ns)]
        out = [None] * len(ps)
        for s, hx in enumerate(self.hybrid.response_fock(dps)):
            for k in range(nk):
                out[s * nk + k] = hx[:, k]
        return out

    def _external(self, v_orb) -> list:
        return [self._h_block(v_orb, b) for b in range(len(self.gs.blocks))]

    def _net(self, ps) -> np.ndarray:
        return sum(np.einsum("rii->ri", p).real for p in ps)

    def _block_matrix(self, hs, k: int, blk) -> np.ndarray:
        return self.ch[k] @ hs[k] @ self.c[k]

    # finite frequencies (M6h-4c): P^a(-k) = conj P^a(k) too, so the convolution holds
    _complex = True

    def _t_d(self, z, d) -> list:
        return [-(self.c[b] @ (d[b] * z[b]) @ self.ch[b]) for b in range(len(z))]

    def _tt_d(self, hs, d) -> list:
        return [d[b] * (self.ch[b] @ hs[b] @ self.c[b]) for b in range(len(hs))]

    def _exchange_a(self, ps) -> list:
        return self._exchange(ps)


class PauliHybridResponse(HybridResponse):
    """The static response of a two-component (non-collinear, no spin-orbit coupling) hybrid
    ground state (milestone M6h-4b; molecules): one spinor block (2N levels, at most one
    electron each; :class:`pydftb.response_pauli.PauliGroundState`), the populations of the four
    Pauli components P^b = Tr_spin(sigma^b P1) (charge to gamma and the third order, the
    magnetisations to the spin constants), H1 = sum_b [1/2 S o (V^b_mu + V^b_nu)] x sigma^b,
    the block potentials in the four components and the exchange of the spinor P1
    (:meth:`pydftb.hybrid.HybridTerms.response_fock`).  Without spin-orbit coupling the global
    spin rotations -i/2 [sigma_b x 1, P0] are zero modes: their images in z are projected out
    of the right-hand side and every Lanczos vector, i.e. the response without a global
    rotation (as :mod:`pydftb.response_pauli`'s bordered system)."""

    def __init__(self, calc, st, sc, opts: dict, rpa: bool = False):
        from . import response_pauli as RP
        self.gs = gs = RP.PauliGroundState(calc, st, sc, opts["degeneracy_tol"],
                                           opts["filling_tol"])
        self.scc = calc.scc and not rpa
        self.kq, self.e, _ = R.scc_kernel(calc, st, sc)
        self.w = np.asarray(st.pauli.w, float) if calc.scc else None
        ch = (None if getattr(calc, "_response_blocks_compat", False)
              else RP.block_channels(calc, st)[0])
        self.channels = ch if ch is not None and ch.size else None
        self.hybrid = st.hybrid
        self.n, self.ns = gs.norb, 1
        self.s = np.asarray(st.smat_precise, float)[0]
        self.c = [np.asarray(blk.c) for blk in gs.blocks]
        self.ch = [c.conj().T for c in self.c]
        self.d = [np.sqrt(np.clip(-gs.max_fill * blk.weight * R.static_weights(gs, blk), 0.0,
                                  None)) for blk in gs.blocks]
        self.groups = self._fermi_groups()
        self.nqm, self.nmat = 4, 2 * self.n
        self.iterations = 0
        self.indefinite = False
        self.rotations = self._rotations()

    _inner = staticmethod(_cdot)

    def t_apply(self, z) -> list:
        return [-(self.c[b] @ (self.d[b] * z[b]) @ self.ch[b]) for b in range(len(z))]

    def t_adjoint(self, hs) -> list:
        return self._fermi_project([self.d[b] * (self.ch[b] @ hs[b] @ self.c[b])
                                    for b in range(len(hs))])

    def _components(self, ps) -> np.ndarray:
        """(4, n_rhs, n): Re diag(P^b S) of the charge and the three magnetisations."""
        from .spinorbit import components
        return sum(np.einsum("brij,ji->bri", components(p), self.s).real for p in ps)

    def _spinor(self, op) -> np.ndarray:
        """sum_b op[b] x sigma^b, (n_rhs, 2N, 2N), of four (n_rhs, N, N) operators."""
        n, nr = self.n, op.shape[1]
        h = np.zeros((nr, 2 * n, 2 * n), complex)
        h[:, :n, :n] = op[0] + op[3]
        h[:, n:, n:] = op[0] - op[3]
        h[:, :n, n:] = op[1] - 1j * op[2]
        h[:, n:, :n] = op[1] + 1j * op[2]
        return h

    def _h_components(self, v4) -> np.ndarray:
        """H1 of component potentials per orbital v4 (4, n_rhs, n)."""
        return self._spinor(0.5 * self.s * (v4[:, :, :, None] + v4[:, :, None, :]))

    def kernel(self, ps) -> list:
        gs, nr = self.gs, ps[0].shape[0]
        if not self.scc:
            return [np.zeros((nr, 2 * self.n, 2 * self.n), complex) for _ in ps]
        qsh = np.stack([gs.shells(q.T) for q in self._components(ps)])    # (4, n_shell, r)
        v = np.empty_like(qsh)
        v[0] = self.e @ (self.kq @ (self.e.T @ qsh[0]))
        v[1:] = np.einsum("ij,bjr->bir", self.w, qsh[1:])
        h = self._h_components(np.transpose(v[:, gs.orb_shell, :], (0, 2, 1)))
        ch = self.channels
        if ch is not None:                               # block potentials, four components
            from .spinorbit import components
            q = 0.0
            for p in ps:
                cs = components(p) @ self.s
                q = q + (0.5 * (cs + np.conj(np.swapaxes(cs, -1, -2))))[:, :, ch.mu, ch.nu].real
            vb = (ch.kernel @ np.concatenate(list(q), axis=1).T).reshape(4, ch.size, nr)
            e = np.zeros((4, nr, self.n, self.n))
            e[:, :, ch.mu, ch.nu] = np.transpose(vb, (0, 2, 1))
            e[:, :, ch.nu, ch.mu] = np.transpose(vb, (0, 2, 1))
            se = self.s @ e
            h = h + self._spinor(0.5 * (se + np.swapaxes(se, -1, -2)))
        return [h + hx for hx in self.hybrid.response_fock(ps)]

    def _rotations(self) -> list:
        """Orthonormal z of the global spin rotations -i/2 [sigma_b x 1, P0] (zero modes without
        spin-orbit coupling): C^-1 = C^+ S of the spinor overlap, z = -X / D on the pairs."""
        n, out = self.n, []
        sig = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
        for blk, c, d in zip(self.gs.blocks, self.c, self.d):
            f = R._theta(self.gs.beta * (blk.evals - blk.fermi))
            p0 = (c * f) @ c.conj().T
            for sb in sig:
                g = np.kron(sb, np.eye(n))
                x = c.conj().T @ blk.s @ (-0.5j * (g @ p0 - p0 @ g)) @ blk.s @ c
                with np.errstate(divide="ignore", invalid="ignore"):
                    out.append(np.where(d > 0.0, -x / d, 0.0))
        basis = []
        scale = max(np.sqrt(np.vdot(z, z).real) for z in out) if out else 0.0
        for z in out:
            for q in basis:
                z = z - np.vdot(q, z).real * q
            nz = np.sqrt(np.vdot(z, z).real)
            if nz > ROTATION_REL * scale:
                basis.append(z / nz)
        return basis

    def _project(self, z) -> list:
        if not self.rotations:
            return z
        y = z[0].copy()
        for q in self.rotations:
            y = y - np.einsum("ij,rij->r", q.conj(), y).real[:, None, None] * q[None]
        return [y]

    def _operator(self, z) -> list:
        z = self._project(z)
        k = self.t_adjoint(self.kernel(self.t_apply(z)))
        return self._project([a - b for a, b in zip(z, k)])

    def _external(self, v_orb) -> list:
        v4 = np.zeros((4,) + v_orb.shape)
        v4[0] = v_orb
        h = self._h_components(v4)
        return [h] * len(self.gs.blocks)

    def _orbital_populations(self, ps) -> np.ndarray:
        return np.transpose(self._components(ps), (1, 0, 2))

    def _net(self, ps) -> np.ndarray:
        n = self.n
        return sum((np.einsum("rii->ri", p[:, :n, :n]) + np.einsum("rii->ri", p[:, n:, n:])).real
                   for p in ps)

    def _block_matrix(self, hs, k: int, blk) -> np.ndarray:
        return self.ch[k] @ hs[k] @ self.c[k]

    # finite frequencies (M6h-4c): no projection (the rotations are no zero modes at w != 0)
    _complex = True

    def _t_d(self, z, d) -> list:
        return [-(self.c[b] @ (d[b] * z[b]) @ self.ch[b]) for b in range(len(z))]

    def _tt_d(self, hs, d) -> list:
        return [d[b] * (self.ch[b] @ hs[b] @ self.c[b]) for b in range(len(hs))]

    def _exchange_a(self, ps) -> list:
        return self.hybrid.response_fock(ps, antihermitian=True)

    def _zero_modes(self) -> bool:
        return bool(self.rotations)


def _response(calc, st, sc, opts: dict, rpa: bool = False) -> HybridResponse:
    """The response class of the ground state: two-component, k-points or the real path."""
    from .hybrid import KPointHybridTerms
    if st.pauli is not None:
        return PauliHybridResponse(calc, st, sc, opts, rpa)
    if isinstance(st.hybrid, KPointHybridTerms):
        return KPointHybridResponse(calc, st, sc, opts, rpa)
    return HybridResponse(calc, st, sc, opts, rpa)


# ---- the analyses ----------------------------------------------------------------------------
def _solve_all(hr, v, omegas) -> list:
    """(orb, net, dei, dfermi) at every frequency: the static solve at 0, M6h-4c's otherwise."""
    return [hr.solve(v) if float(w) == 0.0 else hr.solve_omega(v, float(w)) for w in omegas]


def _stack(xs):
    return None if xs[0] is None else np.stack(xs)


def polarisability(calc, st, sc, spec, opts: dict) -> R.PolarisabilityResult:
    """DFTB+'s ``Polarisability`` with a hybrid functional: the response to E_x, E_y, E_z at
    every frequency (as :func:`pydftb.response.polarisability`; M6h-4c at omega != 0)."""
    omegas = R.frequencies(spec, "polarisability")
    hr = _response(calc, st, sc, opts)
    gs = hr.gs
    pos = np.asarray(st.pos, float)
    v = pos[gs.shell_atom]                               # (n_shell, 3)
    if getattr(calc, "_response_pc_compat", False):      # DFTB+ 25.1 (B27)
        v = v + R.point_charge_shift(st, gs)[:, None]
    periodic = st.cell is not None and bool(np.any(st.pbc))
    orbs, atoms, deis, dfermis, tensors = [], [], [], [], []
    for orb, _, dei, dfermi in _solve_all(hr, v, omegas):
        atom = R._sum_atoms(orb, gs.orb_atom, gs.natom)  # (3, n_qm, n_atom)
        orbs.append(orb)
        atoms.append(atom)
        deis.append(dei)
        dfermis.append(dfermi)
        tensors.append(None if periodic else -np.einsum("Aj,iA->ji", pos, atom[:, 0, :]))
    return R.PolarisabilityResult(omegas, _stack(tensors), np.stack(atoms), np.stack(orbs), deis,
                                  _stack(dfermis), gs.max_fill * gs.dos if gs.metallic else None,
                                  gs.metallic)


def response_kernel(calc, st, sc, spec, opts: dict) -> R.ResponseKernelResult:
    """DFTB+'s ``ResponseKernel`` with a hybrid functional: the response to a unit potential on
    each atom at every frequency (``rpa``: the bare response, as DFTB+'s zero iterations;
    M6h-4c at omega != 0)."""
    spec_d = {} if spec is True else dict(spec)
    rpa = bool(spec_d.get("rpa", False)) or not calc.scc
    omegas = R.frequencies(spec, "response_kernel")
    hr = _response(calc, st, sc, opts, rpa=rpa)
    gs = hr.gs
    v = np.eye(gs.natom)[gs.shell_atom]                  # (n_shell, n_atom)
    if getattr(calc, "_response_pc_compat", False) and not rpa:      # DFTB+ 25.1 (B27)
        v = v + R.point_charge_shift(st, gs)[:, None]
    out = _solve_all(hr, v, omegas)
    frontier = [(blk.spin, blk.k, blk.n_filled - 1, blk.first_empty) for blk in gs.blocks]
    return R.ResponseKernelResult(
        omegas, rpa, np.stack([R._sum_atoms(o[0], gs.orb_atom, gs.natom) for o in out]),
        np.stack([o[0] for o in out]), np.stack([R._sum_atoms(o[1], gs.orb_atom, gs.natom)
                                                 for o in out]),
        [o[2] for o in out], _stack([o[3] for o in out]),
        gs.max_fill * gs.dos if gs.metallic else None, gs.metallic, frontier, gs.orb_atom.copy())


def external_charge_derivatives(calc, st, sc, spec,
                                opts: dict) -> R.ExternalChargeDerivativesResult:
    """dN / dR_j of the point charges with a hybrid functional (the development version's
    ``ExternalChargeDerivatives``): the response to dV_A / dR_j as
    :func:`pydftb.response.external_charge_derivatives`."""
    ncharge = len(st.ext.pc.charges)
    sel = R.external_charge_selection(spec, ncharge)
    charges = np.arange(ncharge) if sel is None else sel
    hr = _response(calc, st, sc, opts)
    gs = hr.gs
    jac = st.ext.pc_jacobian()[:, charges, :]                   # (n_atom, n_j, 3)
    orb, _, _, _ = hr.solve(jac[gs.shell_atom].reshape(gs.nshell, -1))
    orb = orb.reshape(len(charges), 3, hr.nqm, gs.norb)
    return R.ExternalChargeDerivativesResult(charges, R._sum_atoms(orb, gs.orb_atom, gs.natom),
                                             orb, gs.orb_atom.copy())
