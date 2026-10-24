"""Coupled-perturbed SCC response (milestone M6g-1): DFTB+'s ``Polarisability`` and
``ResponseKernel`` (``Analysis`` block).

DFTB+ (``derivs/perturb.F90``, ``derivs/linearresponse.F90``) perturbs the Hamiltonian by
H'_mu,nu = 1/2 S_mu,nu (V_A + V_B), V_A the coordinate X_A,i for an electric field along i
or a unit potential on one atom for the response kernel, and builds the first-order density
from the unperturbed states of every (k, spin) block (a spectral sum):

* levels i <= n_F (the last with a filling above 1e-7) and a >= n_E (the first whose filling
  differs from the full one by more than the single-precision epsilon); with n_F != n_E - 1
  the block is metallic and W = [n_E, n_F] holds its partly filled levels;
* M = C^+ H' C; inside a run of degenerate levels (gaps <= 1e-9 Ha, chained) M is
  diagonalised and the vectors rotated, M_ii then being the eigenvalue derivatives;
* static: U_ai = M_ai theta_ia F_ia with theta_ia = 1 / (1 + exp(beta (e_i - e_a))) (the two
  orders of a pair inside W share the pair) and F_ia = (f_i - f_a) / (e_i - e_a), -delta_i
  for |e_i - e_a| <= 10 eps (the intraband term, delta = beta f (1 - f)); pairs of one
  degenerate run with i != a are left out; dc_i = sum_a c_a U_ai,
  drho_u = sum_i (dc_i c_i^+ + c_i dc_i^+);
* finite frequency: U_ai = M_ai theta_ia (f_i - f_a) / (e_i - e_a + s omega + i eta) for
  s = +-1, drho_u += 1/2 (dc c^+ + c dc^+) (no intraband term);
* metals (static, no fixed Fermi level): the Fermi level moves to keep every spin channel's
  electron count, dE_F = -sum dq_u / N_F with N_F = maxFill sum_k w_k sum_{i in W} delta_i,
  adding maxFill sum_{i in W} delta_i dE_F c_i c_i^+;
* drho = maxFill drho_u; the response potential of the populations (gamma, the spin
  constants W, the full third order linearised at the ground state's input charges) is
  added and the cycle iterated with the ground state's mixer to ``PerturbSccTol``.

The polarisability is alpha_ji = d mu_j / d E_i = -sum_A R_A,j dN_A / dE_i (N electron
populations, DFTB+'s dipole sum_A (q0 - q)_A R_A).

pydftb uses the same weights but writes the response as matrices: with
m^X_ai = 1/2 sum_{mu in X} [c*_mu,a (S c_i)_mu + (S c_a)*_mu c_mu,i] the matrix element of a unit
potential on shell X, chi0_u(Y, X) = sum_p Re[G_p m^X_p conj(m^Y_p)] (G = 2 theta F static,
the sum over s of the complex weights at finite frequency), summed over k with the weights;
the Fermi-level move is the symmetric rank-one term d d^T / sum_k w_k sum_W delta with
d_Y = sum_k w_k sum_{i in W} delta_i m^Y_ii; a degenerate run inside W contributes the
rotation-free -delta (pairs i != j of the run weighted -delta/2 per order).  The SCC
response is then solved directly, (I - chi0 K) x = chi0 V_ext, without iterations (so
without DFTB+'s convergence error, 2.5e-7 for water at its default tolerance 1e-5).

At a finite frequency the weights are complex (the +i eta of DFTB+'s two terms).  With complex
(k-point) eigenvectors Re[G m^X conj(m^Y)] keeps a term Im(G) Im(m^X conj(m^Y)) of order
eta/Delta that cancels between k and -k: DFTB+ keeps it for k-point sets reduced by time
reversal (the full set gives the eta -> 0 result), pydftb drops it, so that a reduced set gives
the full set's response; ``DFTB._response_eta_compat`` (testing) keeps it with DFTB+'s sign.

NumPy engine only; collinear spin and k-points included; DFTB+U, onsite corrections,
non-collinear spin and spin-orbit coupling are milestone M6g-3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import ops

DEGEN_TOL = 1e-9          # Hartree, DFTB+ PerturbDegenTol (absolute)
ETA = 1e-8                # Hartree, DFTB+ PerturbEta
FILLED_TOL = 1e-7         # DFTB+ elecTolMax: n_F is the last level with a filling above it
EMPTY_TOL = float(np.finfo(np.float32).eps)   # DFTB+ epsilon(1.0): first not fully filled level
SAME_TOL = 10.0 * np.finfo(float).eps         # DFTB+ invDiffStatic: e_i = e_a (intraband)
PAIR_CHUNK = 4_000_000    # orbital x pair elements per batch of the matrix elements
_POL_KEYS = {"static", "frequencies", "frequency_range"}
_KERNEL_KEYS = _POL_KEYS | {"rpa"}
_PERT_KEYS = {"degeneracy_tol", "eta"}


# ---- options -------------------------------------------------------------------------------
def frequencies(spec, what: str) -> np.ndarray:
    """DFTB+'s list of frequencies (Hartree) of ``Polarisability``/``ResponseKernel``: 0 first
    for "static" (default True), then "frequencies", then the points of "frequency_range"
    (start, stop, step): n = int((stop - start) / step) + 1."""
    spec = {} if spec is True else dict(spec)
    unknown = set(spec) - (_KERNEL_KEYS if what == "response_kernel" else _POL_KEYS)
    if unknown:
        raise ValueError(f"analysis {what}: unknown keys {sorted(unknown)}")
    out = [0.0] if spec.get("static", True) else []
    for w in np.atleast_1d(spec.get("frequencies", ())):
        if float(w) < 0.0:
            raise ValueError(f"analysis {what}: negative frequency {float(w)}")
        out.append(float(w))
    if spec.get("frequency_range") is not None:
        start, stop, step = (float(x) for x in spec["frequency_range"])
        if start < 0.0 or stop < 0.0:
            raise ValueError(f"analysis {what}: negative frequency in the range")
        if abs(step) <= np.finfo(float).eps:
            raise ValueError(f"analysis {what}: frequency step too small")
        n = max(int((stop - start) / step) + 1, 0)
        out += [start + j * step for j in range(n)]
    if not out:
        raise ValueError(f"analysis {what}: nothing to calculate (static = False and no "
                         "frequencies; DFTB+ silently does nothing)")
    return np.array(out)


def perturbation_options(spec) -> dict:
    spec = dict(spec or {})
    unknown = set(spec) - _PERT_KEYS
    if unknown:
        raise ValueError(f"analysis perturbation: unknown keys {sorted(unknown)} "
                         f"(use {sorted(_PERT_KEYS)})")
    tol = float(spec.get("degeneracy_tol", DEGEN_TOL))
    eta = float(spec.get("eta", ETA))
    if tol < np.finfo(float).eps or eta < np.finfo(float).eps:
        raise ValueError("analysis perturbation: degeneracy_tol and eta must be at least the "
                         "machine epsilon (DFTB+)")
    return {"degeneracy_tol": tol, "eta": eta}


# ---- results -------------------------------------------------------------------------------
@dataclass
class PolarisabilityResult:
    """Electric-field response (``r.analysis.polarisability``, atomic units).

    ``tensor[w, j, i]`` = d mu_j / d E_i at ``frequencies[w]`` (Hartree), mu = sum_A (q0 - q)_A
    R_A as DFTB+ (None for periodic systems); ``charge_derivatives[w, i, c, A]`` = dN_A / dE_i
    (electron populations, c = 0 charge, 1 magnetisation with collinear spin);
    ``orbital_derivatives[w, i, c, mu]`` the same per orbital; ``eigenvalue_derivatives[w]``
    (3, n_block, n_level) d eps / d E_i of every (spin, k) block (degenerate levels: the
    eigenvalues of M in the run, ascending); for metals ``fermi_derivatives[w, i, channel]``
    = dE_F / dE_i (the physical value; DFTB+ prints it divided by the full filling) and
    ``dos_fermi`` DFTB+'s neFermi per channel (with the full filling)."""
    frequencies: np.ndarray
    tensor: np.ndarray | None
    charge_derivatives: np.ndarray
    orbital_derivatives: np.ndarray
    eigenvalue_derivatives: list
    fermi_derivatives: np.ndarray | None
    dos_fermi: np.ndarray | None
    metallic: bool


@dataclass
class ResponseKernelResult:
    """Response to a unit potential on each atom (``r.analysis.response_kernel``).

    ``atom[w, A, c, B]`` = dN_B / dV_A (c = 0 charge, 1 magnetisation), ``orbital[w, A, c, mu]``
    per orbital, ``net[w, A, B]`` the on-site (net) population of B, ``eigenvalue_derivatives
    [w]`` (n_atom, n_block, n_level); ``rpa``: without SCC screening (the bare response);
    ``frontier``: (spin, k, last filled, first empty level) of every block (DFTB+'s n_F and
    n_E, 0-based); ``orbital_atom``: the atom of every orbital."""
    frequencies: np.ndarray
    rpa: bool
    atom: np.ndarray
    orbital: np.ndarray
    net: np.ndarray
    eigenvalue_derivatives: list
    fermi_derivatives: np.ndarray | None
    dos_fermi: np.ndarray | None
    metallic: bool
    frontier: list
    orbital_atom: np.ndarray


# ---- the unperturbed states ----------------------------------------------------------------
@dataclass
class _Block:
    spin: int                   # collinear channel (0 without spin)
    k: int
    weight: float
    evals: np.ndarray
    c: np.ndarray               # (N, n) eigenvectors
    sc: np.ndarray              # S(k) c
    s: np.ndarray               # S(k)
    fermi: float
    n_filled: int               # levels 0 .. n_filled - 1 may be occupied (DFTB+ nFilled)
    first_empty: int            # index of DFTB+'s nEmpty
    group: np.ndarray           # degenerate run of every level

    @property
    def metallic(self) -> bool:
        return self.n_filled != self.first_empty

    @property
    def window(self) -> np.ndarray:
        """The partly filled levels W (empty for insulating blocks)."""
        return np.arange(self.first_empty, self.n_filled)


def _theta(x):
    """1 / (1 + exp(x)) without overflow warnings."""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(x))


def _delta(e, ef, beta):
    """DFTB+ deltamn: beta / (2 (1 + cosh(beta (e - E_F)))) = beta f (1 - f)."""
    with np.errstate(over="ignore"):
        return beta / (2.0 * (1.0 + np.cosh(beta * (e - ef))))


class GroundState:
    """Eigenpairs of the last Hamiltonian (built from the last input populations, as DFTB+),
    fillings, windows and degenerate runs of every (spin, k) block."""

    def __init__(self, calc, st, sc, degeneracy_tol: float = DEGEN_TOL):
        b = st.basis
        self.natom = len(st.species)
        self.norb = b.norb
        self.orb_shell = np.asarray(b.orb_shell)
        self.orb_atom = np.asarray(b.orb_atom)
        self.shell_atom = np.asarray(b.shell_atom)
        self.nshell = len(b.shell_l)
        self.shell_start = np.concatenate([[0], np.cumsum(2 * np.asarray(b.shell_l) + 1)[:-1]])
        self.nspin = 2 if st.spin is not None else 1
        self.max_fill = 2.0 if self.nspin == 1 else 1.0
        self.beta = 1.0 / calc.smearing.kt
        self.fixed_fermi = calc.fixed_fermi_level is not None
        self.shared_fermi = st.spin is not None and bool(st.spin.shared_ef)
        mu = sc.mu if isinstance(sc.mu, (tuple, list)) else (sc.mu,) * self.nspin
        h = np.asarray(sc.h)
        nk = len(st.kw)
        s_all = np.asarray(st.smat_precise)
        self.blocks = []
        for i, hk in enumerate(h):
            spin, k = divmod(i, nk)
            sk = s_all[k]
            evals, c = ops.eigh(hk, sk)
            evals, c = np.asarray(evals), np.asarray(c)
            fill = self.max_fill * _theta(self.beta * (evals - mu[spin]))
            above = np.flatnonzero(fill > FILLED_TOL)
            n_filled = int(above[-1]) + 1 if above.size else 0
            short = np.flatnonzero(np.abs(fill - self.max_fill) > EMPTY_TOL)
            first_empty = int(short[0]) if short.size else 0           # DFTB+'s default 1
            group = np.concatenate([[0], np.cumsum(np.diff(evals) > degeneracy_tol)])
            self.blocks.append(_Block(spin, k, float(st.kw[k]), evals, c, sk @ c, sk,
                                      float(mu[spin]), n_filled, first_empty, group))
        self.metallic = any(blk.metallic for blk in self.blocks)
        self.fermi_shift = self.metallic and not self.fixed_fermi
        # DFTB+'s neFermi / maxFill per channel and d^mu = sum_k w_k sum_{i in W} delta_i
        # (Mulliken weight, on-site weight) of the Fermi-level move
        self.dos = np.zeros(self.nspin)
        self.d_orb = np.zeros((self.nspin, self.norb))
        self.d_net = np.zeros((self.nspin, self.norb))
        for blk in self.blocks:
            win = blk.window
            if win.size:
                dlt = blk.weight * _delta(blk.evals[win], blk.fermi, self.beta)
                cw = blk.c[:, win]
                self.dos[blk.spin] += dlt.sum()
                self.d_orb[blk.spin] += np.real(cw.conj() * blk.sc[:, win]) @ dlt
                self.d_net[blk.spin] += (np.abs(cw) ** 2) @ dlt

    def shells(self, x_orb):
        """Orbital rows -> shell rows (sum over a shell's orbitals)."""
        return np.add.reduceat(x_orb, self.shell_start, axis=0)

    # ---- pairs and weights -------------------------------------------------------------
    def pairs(self, blk: _Block):
        """DFTB+'s pairs (i <= n_F, a >= n_E) of a block: index arrays (i, a) and whether the
        two levels are in one degenerate run (i != a)."""
        i, a = np.meshgrid(np.arange(blk.n_filled), np.arange(blk.first_empty, len(blk.evals)),
                           indexing="ij")
        i, a = i.ravel(), a.ravel()
        same = (blk.group[i] == blk.group[a]) & (i != a)
        return i, a, same

    def weights(self, blk: _Block, i, a, same, omega: float, eta: float):
        """G_p of every pair: 2 theta F (static), sum_s theta (f_i - f_a)/(e_i - e_a + s omega
        + i eta) at finite frequency; pairs of one degenerate run: -delta/2 per order (static,
        the rotation-free form of DFTB+'s rotated intraband term), 0 at finite frequency."""
        e, beta = blk.evals, self.beta
        th = _theta(beta * (e[i] - e[a]))
        f = _theta(beta * (e - blk.fermi))
        df = f[i] - f[a]
        de = e[i] - e[a]
        if omega == 0.0:
            with np.errstate(divide="ignore", invalid="ignore"):
                fs = np.where(np.abs(de) > SAME_TOL, df / de, -_delta(e[i], blk.fermi, beta))
            g = 2.0 * th * fs
            if same.any():
                d = _delta(e, blk.fermi, beta)
                g = np.where(same, -0.5 * (d[i] + d[a]), g)
            return g
        z = th * df
        g = z / (de + omega + 1j * eta) + z / (de - omega + 1j * eta)
        return np.where(same, 0.0, g)

    def elements(self, blk: _Block, i, a):
        """m^mu_ai = 1/2 [c*_mu,a (S c_i)_mu + (S c_a)*_mu c_mu,i] (orbitals x pairs) and the
        on-site products n^mu_ai = c_mu,a c*_mu,i."""
        c, sc = blk.c, blk.sc
        ca, ci = c[:, a], c[:, i]
        m = 0.5 * (ca.conj() * sc[:, i] + sc[:, a].conj() * ci)
        return m, ca * ci.conj()


# ---- chi0 ----------------------------------------------------------------------------------
@dataclass
class Chi0:
    """Non-interacting response at one frequency in the up/down representation: ``orb``
    (n_spin n_orb, n_spin n_shell), the Mulliken population of every orbital of every channel
    per unit potential on a shell of a channel (full filling and Fermi-level move included);
    ``net`` the same for the on-site populations (or None)."""
    orb: np.ndarray
    net: np.ndarray | None


def chi0(gs: GroundState, omegas, eta: float = ETA, with_net: bool = False,
         eta_compat: bool = False) -> list:
    """Chi0 at every frequency in ``omegas`` (one pass over the pairs).  Complex weights enter
    by their real part (the eta term of complex eigenvectors cancels between k and -k); with
    ``eta_compat`` by their conjugate (DFTB+'s sign of the term for reduced k-point sets)."""
    ns, no, nsh = gs.nspin, gs.norb, gs.nshell
    nw = len(omegas)
    orb = np.zeros((nw, ns, no, nsh))
    net = np.zeros((nw, ns, no, nsh)) if with_net else None
    fill = gs.max_fill
    for blk in gs.blocks:
        i, a, same = gs.pairs(blk)
        if not i.size:
            continue
        step = max(1, PAIR_CHUNK // max(no, 1))
        for lo in range(0, i.size, step):
            ii, aa, ss = i[lo:lo + step], a[lo:lo + step], same[lo:lo + step]
            m, n = gs.elements(blk, ii, aa)
            msh = gs.shells(m)
            for w, omega in enumerate(omegas):
                g = gs.weights(blk, ii, aa, ss, float(omega), eta) * (blk.weight * fill)
                if np.iscomplexobj(g):
                    g = np.conj(g) if eta_compat else g.real
                gm = g[:, None] * msh.T
                orb[w, blk.spin] += np.real(m.conj() @ gm)
                if with_net:
                    net[w, blk.spin] += np.real(n @ gm)
    d_sh = np.array([gs.shells(gs.d_orb[s]) for s in range(ns)])
    out = []
    for w, omega in enumerate(omegas):
        o = _block_diag(orb[w])
        nt = _block_diag(net[w]) if with_net else None
        if gs.fermi_shift and float(omega) == 0.0:
            if gs.shared_fermi:                        # one move for both channels
                o = o + fill * np.outer(gs.d_orb.ravel(), d_sh.ravel()) / gs.dos.sum()
                if with_net:
                    nt = nt + fill * np.outer(gs.d_net.ravel(), d_sh.ravel()) / gs.dos.sum()
            else:                                      # DFTB+: every channel keeps its count
                for s in range(ns):
                    if gs.dos[s] > 0.0:
                        r, c = slice(s * no, (s + 1) * no), slice(s * nsh, (s + 1) * nsh)
                        o[r, c] += fill * np.outer(gs.d_orb[s], d_sh[s]) / gs.dos[s]
                        if with_net:
                            nt[r, c] += fill * np.outer(gs.d_net[s], d_sh[s]) / gs.dos[s]
        out.append(Chi0(o, nt))
    return out


def _block_diag(x) -> np.ndarray:
    """(n_spin, n_rows, n_cols) -> block-diagonal (n_spin n_rows, n_spin n_cols)."""
    ns, nr, nc = x.shape
    out = np.zeros((ns * nr, ns * nc))
    for s in range(ns):
        out[s * nr:(s + 1) * nr, s * nc:(s + 1) * nc] = x[s]
    return out


def to_qm(gs: GroundState, mat: np.ndarray) -> np.ndarray:
    """Up/down response (n_spin n_rows, n_spin n_shell) -> (charge, magnetisation): rows
    q = up + down, m = up - down; columns V_up = V_q + V_m, V_down = V_q - V_m."""
    if gs.nspin == 1:
        return mat
    nr = mat.shape[0] // 2
    nc = mat.shape[1] // 2
    rows = np.block([[np.eye(nr), np.eye(nr)], [np.eye(nr), -np.eye(nr)]])
    cols = np.block([[np.eye(nc), np.eye(nc)], [np.eye(nc), -np.eye(nc)]])
    return rows @ mat @ cols


def shell_rows(gs: GroundState, mat: np.ndarray) -> np.ndarray:
    """Orbital rows of every channel -> shell rows."""
    ns, no = gs.nspin, gs.norb
    return np.vstack([gs.shells(mat[s * no:(s + 1) * no]) for s in range(ns)])


# ---- the SCC kernel ------------------------------------------------------------------------
def scc_kernel(calc, st, sc) -> tuple:
    """K of the response: (K_q on the SCC sites, the shell -> site matrix E, W on the shells or
    None).  K_q = gamma plus the Hessian of the full third order at the ground state's last
    input charges (DFTB+ getdShiftdQ)."""
    nsite = len(st.site_atom)
    b = st.basis
    first_orb = np.concatenate([[0], np.cumsum(2 * np.asarray(b.shell_l) + 1)[:-1]])
    shell_site = np.asarray(st.site_of_orb)[first_orb]
    e = np.zeros((len(b.shell_l), nsite))
    e[np.arange(len(b.shell_l)), shell_site] = 1.0
    if not calc.scc:
        return np.zeros((nsite, nsite)), e, None
    kq = np.array(st.coulomb.gamma, dtype=float, copy=True)
    if st.third is not None:
        dq, _ = calc._charge_and_magnetisation(st, sc.q_in)
        kq = kq + third_order_hessian(st.third, np.asarray(dq, float))
    w = None if st.spin is None else np.asarray(st.spin.w, float)
    return kq, e, w


def third_order_hessian(third, dq) -> np.ndarray:
    """d^2 E3 / d dq^2 of :class:`pydftb.thirdorder.ThirdOrder` (the Jacobian of its
    potential), E3 = 1/3 sum_s (A dq)_s dq_s (G dq)_s with A the same-atom matrix."""
    g = np.asarray(third.gamma3)
    sa = np.asarray(third.site_atom)
    a = (sa[:, None] == sa[None, :]).astype(float)
    gq, aq = g @ dq, a @ dq
    j = (gq[:, None] * a + aq[:, None] * g + g.T @ (dq[:, None] * a) + g.T * aq[None, :]
         + a * gq[None, :] + a @ (dq[:, None] * g))
    return j / 3.0


# ---- solving -------------------------------------------------------------------------------
class Response:
    """The SCC response of one calculation at one frequency: unknowns x = (dq on the SCC
    sites, dm on the shells), (I - A B) x = A V_ext with A = chi0 (shell potentials -> x) and
    B = K (x -> shell potentials)."""

    def __init__(self, gs: GroundState, kernel, x: Chi0, scc: bool):
        self.gs = gs
        self.x = x
        kq, e, w = kernel
        nsh = gs.nshell
        chi_sh = to_qm(gs, shell_rows(gs, x.orb))       # (n_qm nsh, n_qm nsh)
        nq = e.shape[1]
        if gs.nspin == 1:
            amat = e.T @ chi_sh
            bmat = e @ kq
        else:
            proj = np.zeros((nq + nsh, 2 * nsh))
            proj[:nq, :nsh] = e.T
            proj[nq:, nsh:] = np.eye(nsh)
            amat = proj @ chi_sh
            bmat = np.zeros((2 * nsh, nq + nsh))
            bmat[:nsh, :nq] = e @ kq
            if w is not None:
                bmat[nsh:, nq:] = w
        self.amat, self.bmat, self.scc = amat, bmat, scc

    def potentials(self, v_ext) -> np.ndarray:
        """Total shell potentials (n_qm nsh, n_rhs) for external charge potentials per shell
        (n_shell, n_rhs); the magnetisation part of the external potential is zero."""
        v = np.asarray(v_ext, float)
        if self.gs.nspin == 2:
            v = np.vstack([v, np.zeros_like(v)])
        if not self.scc:
            return v
        lhs = np.eye(self.amat.shape[0]) - self.amat @ self.bmat
        xsol = np.linalg.solve(lhs, self.amat @ v)
        return v + self.bmat @ xsol

    def populations(self, v_tot, net: bool = False) -> np.ndarray:
        """(n_qm, n_orb, n_rhs) Mulliken (or on-site) populations for total shell potentials."""
        chi = to_qm(self.gs, self.x.net if net else self.x.orb)
        p = chi @ v_tot
        return p.reshape(self.gs.nspin, self.gs.norb, p.shape[1])


def eigenvalue_derivatives(gs: GroundState, v_tot) -> np.ndarray:
    """d eps of every (spin, k) block for total shell potentials (n_qm nsh,): the diagonal of
    M = C^+ dH C, in a degenerate run the eigenvalues of its block (ascending)."""
    nsh = gs.nshell
    vq = v_tot[:nsh]
    vm = v_tot[nsh:] if gs.nspin == 2 else np.zeros(nsh)
    out = []
    for blk in gs.blocks:
        v = (vq + (vm if blk.spin == 0 else -vm))[gs.orb_shell]
        dh = 0.5 * blk.s * (v[:, None] + v[None, :])
        m = blk.c.conj().T @ dh @ blk.c
        de = np.real(np.diag(m)).copy()
        for gid in np.unique(blk.group):
            idx = np.flatnonzero(blk.group == gid)
            if idx.size > 1:
                de[idx] = np.linalg.eigvalsh(m[np.ix_(idx, idx)])
        out.append(de)
    return np.array(out)


def fermi_derivative(gs: GroundState, v_tot) -> np.ndarray:
    """dE_F per channel (equal for a shared Fermi level): sum_k w_k sum_{i in W} delta_i M_ii
    / sum_k w_k sum_{i in W} delta_i."""
    nsh = gs.nshell
    vq = v_tot[:nsh]
    vm = v_tot[nsh:] if gs.nspin == 2 else np.zeros(nsh)
    num = np.array([gs.shells(gs.d_orb[s]) @ (vq + (vm if s == 0 else -vm))
                    for s in range(gs.nspin)])
    if gs.shared_fermi:
        return np.full(gs.nspin, num.sum() / gs.dos.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(gs.dos > 0.0, num / gs.dos, 0.0)


# ---- the two analyses ----------------------------------------------------------------------
def _sum_atoms(x, orb_atom, natom):
    """Sum the last (orbital) axis over atoms."""
    out = np.zeros(x.shape[:-1] + (natom,))
    for a in range(natom):
        out[..., a] = x[..., orb_atom == a].sum(-1)
    return out


def polarisability(calc, st, sc, spec, opts: dict) -> PolarisabilityResult:
    """DFTB+'s ``Polarisability``: the response to E_x, E_y, E_z (H' from the atoms'
    coordinates) at every frequency of ``spec``."""
    omegas = frequencies(spec, "polarisability")
    gs = GroundState(calc, st, sc, opts["degeneracy_tol"])
    kernel = scc_kernel(calc, st, sc)
    periodic = st.cell is not None and bool(np.any(st.pbc))
    pos = np.asarray(st.pos, float)
    v_ext = pos[gs.shell_atom]                           # (n_shell, 3)
    nw, nqm = len(omegas), gs.nspin
    orb = np.zeros((nw, 3, nqm, gs.norb))
    dei = []
    dfermi = np.zeros((nw, 3, gs.nspin)) if gs.metallic else None
    compat = bool(getattr(calc, "_response_eta_compat", False))
    for w, x in enumerate(chi0(gs, omegas, opts["eta"], eta_compat=compat)):
        resp = Response(gs, kernel, x, calc.scc)
        v_tot = resp.potentials(v_ext)                   # (n_qm nsh, 3)
        orb[w] = np.transpose(resp.populations(v_tot), (2, 0, 1))
        dei.append(np.array([eigenvalue_derivatives(gs, v_tot[:, i]) for i in range(3)]))
        if dfermi is not None and float(omegas[w]) == 0.0 and gs.fermi_shift:
            for i in range(3):
                dfermi[w, i] = fermi_derivative(gs, v_tot[:, i])
    atom = _sum_atoms(orb, gs.orb_atom, gs.natom)
    tensor = None if periodic else -np.einsum("Aj,wiA->wji", pos, atom[:, :, 0, :])
    return PolarisabilityResult(omegas, tensor, atom, orb, dei, dfermi,
                                gs.max_fill * gs.dos if gs.metallic else None, gs.metallic)


def response_kernel(calc, st, sc, spec, opts: dict) -> ResponseKernelResult:
    """DFTB+'s ``ResponseKernel``: the response to a unit potential on each atom (SCC
    screened, or the bare response with ``rpa`` and without SCC)."""
    spec_d = {} if spec is True else dict(spec)
    rpa = bool(spec_d.get("rpa", False)) or not calc.scc
    omegas = frequencies(spec, "response_kernel")
    gs = GroundState(calc, st, sc, opts["degeneracy_tol"])
    kernel = scc_kernel(calc, st, sc)
    v_ext = np.eye(gs.natom)[gs.shell_atom]              # (n_shell, n_atom)
    nw, nqm = len(omegas), gs.nspin
    orb = np.zeros((nw, gs.natom, nqm, gs.norb))
    net = np.zeros((nw, gs.natom, gs.natom))
    dei = []
    dfermi = np.zeros((nw, gs.natom, gs.nspin)) if gs.metallic else None
    compat = bool(getattr(calc, "_response_eta_compat", False))
    for w, x in enumerate(chi0(gs, omegas, opts["eta"], with_net=True, eta_compat=compat)):
        resp = Response(gs, kernel, x, calc.scc and not rpa)
        v_tot = resp.potentials(v_ext)                   # (n_qm nsh, n_atom)
        orb[w] = np.transpose(resp.populations(v_tot), (2, 0, 1))
        pnet = resp.populations(v_tot, net=True)[0]      # (n_orb, n_atom), charge part
        net[w] = _sum_atoms(pnet.T, gs.orb_atom, gs.natom)
        dei.append(np.array([eigenvalue_derivatives(gs, v_tot[:, a])
                             for a in range(gs.natom)]))
        if dfermi is not None and float(omegas[w]) == 0.0 and gs.fermi_shift:
            for a in range(gs.natom):
                dfermi[w, a] = fermi_derivative(gs, v_tot[:, a])
    atom = _sum_atoms(orb, gs.orb_atom, gs.natom)
    frontier = [(blk.spin, blk.k, blk.n_filled - 1, blk.first_empty) for blk in gs.blocks]
    return ResponseKernelResult(omegas, rpa, atom, orb, net, dei, dfermi,
                                gs.max_fill * gs.dos if gs.metallic else None, gs.metallic,
                                frontier, gs.orb_atom.copy())
