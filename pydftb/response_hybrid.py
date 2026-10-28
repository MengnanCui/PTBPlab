"""Static responses with a hybrid functional (milestone M6h-4): DFTB+'s ``Polarisability``,
``ResponseKernel`` and the development version's ``ExternalChargeDerivatives`` on the real path
(molecules and the Gamma point, collinear spin); the entry points of :mod:`pydftb.response`
branch here when the calculation has a hybrid functional.

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

NumPy engine only; static only (DFTB+ refuses finite frequencies with a hybrid).  Two-component
calculations, k-points and coordinate derivatives are milestone M6h-4b.
"""
from __future__ import annotations

import warnings

import numpy as np

from . import response as R

SOLVE_TOL = 1e-13         # relative residual of every right-hand side
SOLVE_MAX = 1000          # iterations
NEGATIVE_CURVATURE = -1e-8   # Rayleigh quotient of a Lanczos vector: an indefinite equation
RHS_ELEMENTS = 2_000_000  # right-hand sides per batch: n_rhs n_block n_orb^2 at most this


def _real(x) -> np.ndarray:
    """A real matrix of the real path (the Gamma point's are real already)."""
    x = np.asarray(x)
    if np.iscomplexobj(x):
        raise ValueError("hybrid functionals: the response with complex eigenvectors "
                         "(k-points) is milestone M6h-4b")
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
        # the Fermi groups (one channel, or both with a shared Fermi level): the window levels
        # whose electron count is kept and the unit vector u ~ sqrt(w delta) of the projection
        self.groups = []
        if gs.fermi_shift:
            chans = ([list(range(len(gs.blocks)))] if gs.shared_fermi else
                     [[b for b, blk in enumerate(gs.blocks) if blk.spin == s]
                      for s in range(self.ns)])
            for members in chans:
                parts = [(b, gs.blocks[b].window) for b in members if gs.blocks[b].window.size]
                if parts:
                    u = np.concatenate([np.diag(self.d[b])[win] for b, win in parts])
                    self.groups.append((parts, u / np.linalg.norm(u)))
        self.iterations = 0                              # most MINRES steps of a solve
        self.indefinite = False

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
        for parts, u in self.groups:                     # the Fermi level's move
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
    def _operator(self, z) -> list:
        """(I + T^T K T) z (t_apply carries the minus sign of P1 = -T z)."""
        k = self.t_adjoint(self.kernel(self.t_apply(z)))
        return [a - b for a, b in zip(z, k)]

    def _minres(self, b) -> list:
        """MINRES on every right-hand side of ``b`` (z of every block): Paige and Saunders'
        Lanczos process with the QR factorisation of its tridiagonal by Givens rotations
        (SciPy's minres without a preconditioner), each column with its own scalars, until the
        residual estimate is SOLVE_TOL of |b|.  A negative Rayleigh quotient of a Lanczos
        vector marks the equation as indefinite."""
        nr = b[0].shape[0]
        beta1 = np.sqrt(_dot(b, b))
        x = [np.zeros_like(v) for v in b]
        w = [np.zeros_like(v) for v in b]
        w2 = [np.zeros_like(v) for v in b]
        r1 = [v.copy() for v in b]
        r2 = [v.copy() for v in b]
        oldb, beta, dbar, epsln = np.zeros(nr), beta1.copy(), np.zeros(nr), np.zeros(nr)
        phibar, cs, sn = beta1.copy(), -np.ones(nr), np.zeros(nr)
        stop = SOLVE_TOL * beta1
        for it in range(1, SOLVE_MAX + 2):
            active = phibar > stop
            if not active.any():
                break
            if it > SOLVE_MAX:
                worst = float(np.max(phibar[active] / beta1[active]))
                raise RuntimeError(f"the hybrid response did not converge in {SOLVE_MAX} "
                                   f"iterations (relative residual {worst:.1e})")
            v = [_scale(1.0, beta, y) for y in r2]              # the Lanczos vector
            y = self._operator(v)
            if it >= 2:
                y = [a - _scale(beta, oldb, c) for a, c in zip(y, r1)]
            alfa = _dot(v, y)
            if np.any(alfa[active] < NEGATIVE_CURVATURE):
                self.indefinite = True
            y = [a - _scale(alfa, beta, c) for a, c in zip(y, r2)]
            r1, r2 = r2, y
            oldb, beta = beta, np.sqrt(_dot(r2, r2))
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
        step = max(1, RHS_ELEMENTS // (len(gs.blocks) * self.n * self.n))
        nlev = max(len(blk.evals) for blk in gs.blocks)
        orb = np.zeros((nrhs, self.ns, self.n))
        net = np.zeros((nrhs, self.n))
        dei = np.zeros((nrhs, len(gs.blocks), nlev))
        efnum = np.zeros((nrhs, self.ns))
        for lo in range(0, nrhs, step):
            cols = slice(lo, min(lo + step, nrhs))
            h = self._h_shell(v_shell[gs.orb_shell, cols].T)
            h_ext = [h] * self.ns
            b = self.t_adjoint(h_ext)
            ps = self.t_apply(self._minres(b) if self.scc else b)
            hs = [x + y for x, y in zip(h_ext, self.kernel(ps))]
            pop = self.populations(ps)
            orb[cols] = np.stack(pop if self.ns == 1 else [pop[0] + pop[1], pop[0] - pop[1]],
                                 axis=1)
            net[cols] = sum(np.einsum("rii->ri", p) for p in ps)
            for k, blk in enumerate(gs.blocks):
                m = self.c[k].T @ hs[blk.spin] @ self.c[k]
                diag = np.einsum("rii->ri", m)
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


# ---- the analyses ----------------------------------------------------------------------------
def _static(spec, what: str) -> np.ndarray:
    omegas = R.frequencies(spec, what)
    if np.any(omegas != 0.0):                            # _check_hybrid stops this before
        raise ValueError("Finite frequency hybrid xc-functional calculation not currently "
                         "supported (as in DFTB+)")
    return omegas


def polarisability(calc, st, sc, spec, opts: dict) -> R.PolarisabilityResult:
    """DFTB+'s ``Polarisability`` with a hybrid functional: the static response to E_x, E_y,
    E_z (as :func:`pydftb.response.polarisability`)."""
    omegas = _static(spec, "polarisability")
    hr = HybridResponse(calc, st, sc, opts)
    gs = hr.gs
    pos = np.asarray(st.pos, float)
    v = pos[gs.shell_atom]                               # (n_shell, 3)
    if getattr(calc, "_response_pc_compat", False):      # DFTB+ 25.1 (B27)
        v = v + R.point_charge_shift(st, gs)[:, None]
    orb, _, dei, dfermi = hr.solve(v)
    atom = R._sum_atoms(orb, gs.orb_atom, gs.natom)      # (3, n_qm, n_atom)
    periodic = st.cell is not None and bool(np.any(st.pbc))
    tensor = None if periodic else -np.einsum("Aj,iA->ji", pos, atom[:, 0, :])[None]
    return R.PolarisabilityResult(omegas, tensor, atom[None], orb[None], [dei],
                                  None if dfermi is None else dfermi[None],
                                  gs.max_fill * gs.dos if gs.metallic else None, gs.metallic)


def response_kernel(calc, st, sc, spec, opts: dict) -> R.ResponseKernelResult:
    """DFTB+'s ``ResponseKernel`` with a hybrid functional: the static response to a unit
    potential on each atom (``rpa``: the bare response, as DFTB+'s zero iterations)."""
    spec_d = {} if spec is True else dict(spec)
    rpa = bool(spec_d.get("rpa", False)) or not calc.scc
    omegas = _static(spec, "response_kernel")
    hr = HybridResponse(calc, st, sc, opts, rpa=rpa)
    gs = hr.gs
    v = np.eye(gs.natom)[gs.shell_atom]                  # (n_shell, n_atom)
    if getattr(calc, "_response_pc_compat", False) and not rpa:      # DFTB+ 25.1 (B27)
        v = v + R.point_charge_shift(st, gs)[:, None]
    orb, net, dei, dfermi = hr.solve(v)
    frontier = [(blk.spin, blk.k, blk.n_filled - 1, blk.first_empty) for blk in gs.blocks]
    return R.ResponseKernelResult(omegas, rpa, R._sum_atoms(orb, gs.orb_atom, gs.natom)[None],
                                  orb[None], R._sum_atoms(net, gs.orb_atom, gs.natom)[None],
                                  [dei], None if dfermi is None else dfermi[None],
                                  gs.max_fill * gs.dos if gs.metallic else None, gs.metallic,
                                  frontier, gs.orb_atom.copy())


def external_charge_derivatives(calc, st, sc, spec,
                                opts: dict) -> R.ExternalChargeDerivativesResult:
    """dN / dR_j of the point charges with a hybrid functional (the development version's
    ``ExternalChargeDerivatives``): the response to dV_A / dR_j as
    :func:`pydftb.response.external_charge_derivatives`."""
    ncharge = len(st.ext.pc.charges)
    sel = R.external_charge_selection(spec, ncharge)
    charges = np.arange(ncharge) if sel is None else sel
    hr = HybridResponse(calc, st, sc, opts)
    gs = hr.gs
    jac = st.ext.pc_jacobian()[:, charges, :]                   # (n_atom, n_j, 3)
    orb, _, _, _ = hr.solve(jac[gs.shell_atom].reshape(gs.nshell, -1))
    orb = orb.reshape(len(charges), 3, gs.nspin, gs.norb)
    return R.ExternalChargeDerivativesResult(charges, R._sum_atoms(orb, gs.orb_atom, gs.natom),
                                             orb, gs.orb_atom.copy())
