"""Range-separated hybrid functionals: LC-DFTB and CAM (milestone M6h-1; DFTB+ ``Hybrid``).

The exchange of the density-matrix difference DeltaP = P - P0 (P0 the diagonal reference
populations q0_mu / S_mumu, half per spin channel), as DFTB+ (``dftb/hybridxc.F90``,
``dftb/rshgamma.F90``):

* gamma = alpha gamma_HF + beta gamma_LR(omega), tau = 3.2 U of the first shell; omega, alpha
  and beta come only from the SK files' ``RangeSep`` section (:meth:`pydftb.skf.SKSet.range_sep`).
  For nearly equal U (tau within 5 %) the closed form of unequal U cancels (DFTB+'s gamma is
  off by 1e-7 at |U_A - U_B| = 1e-4 and by 2 at 1e-6): there, as pydftb's SCC gamma (M2b),
  divided differences by the Hermite-Genocchi integral (gamma_LR = gamma_HF - the Yukawa
  integral of exp(-omega r) / r);
* H_x[DeltaP] = c [(S DP S) o G + ((S DP) o G) S + S (DP o G) S + S ((DP S) o G)], G the gamma
  of the orbitals' atoms, c = -1/8 (-1/4 per spin channel with spin), i.e.
  H_munu = c sum_ab S_mua DP_ab S_bnu (g_munu + g_mub + g_ab + g_anu);
* E_Fock = 1/2 sum_spin sum H_x[DP_in] o DP_in (DFTB+ "Energy Fock").

Non-collinear spin (milestone M6h-1b; DFTB+ ``addCamHamiltonianMatrix_pauli``): DeltaP is the
2N x 2N spinor matrix (Hermitian, complex), P0 half the reference populations on both spin
blocks; H_x applies the operator above, with c = -1/4, to each of its four N x N blocks (S x 1
and G on all four blocks); E_Fock = 1/2 Re sum H_x o conj(DP).

The SCC mixes DeltaP (:meth:`pydftb.calculator.DFTB._scc_hybrid`); the forces take DeltaP_out
(:meth:`HybridTerms.pair_coefficient` for the S derivatives, :meth:`HybridTerms.gamma_gradient`
for the gamma derivatives).  Screening as DFTB+'s molecular algorithms: ``matrix`` (dense,
exact; DFTB+'s default MatrixBased), ``neighbour`` (S blocks beyond the largest SK cutoff minus
``cutoff_reduction`` left out; NeighbourBased) and ``threshold`` (DFTB+'s incremental
Thresholded build).  The forces are always the exact ones, as DFTB+'s.  Molecules only (periodic
systems: milestone M6h-3).

TD-LC (milestone M6h-2, :mod:`pydftb.casida`): the Mulliken exchange operator
K^x[M] = 1/4 :func:`four_terms` (S, G, M) of any square M (K^x[M^T] = K^x[M]^T) gives the
exchange integrals of the Casida matrices, and :func:`bilinear_ds`, :func:`bilinear_dg` the
derivatives of B(M1, M2) = <M1, K^x[M2]> for the excited-state forces
(:meth:`HybridTerms.exchange_force`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coulomb import _GL_S, _GL_W, NEAR_TAU, _short_near

TOL_SAME_DIST = 1.0e-5          # DFTB+ tolSameDist (Bohr): on-site below this distance
MIN_HUB_DIFF = 0.3125e-5        # DFTB+ MinHubDiff, on tau = 3.2 U: "equal U" below this
CAM_EPS = 100 * np.finfo(float).eps   # DFTB+ leaves out an alpha or beta term below this
SCREENINGS = ("matrix", "neighbour", "threshold")
KINDS = ("lc", "cam")


@dataclass(frozen=True)
class Hybrid:
    """The hybrid functional to use (DFTB+ ``Hybrid = LC {...}`` or ``CAM {...}``).

    ``kind``: "lc" needs LC SK files; "cam" takes LC, global (beta = 0) and CAM files.
    ``screening``: "matrix" (DFTB+'s default MatrixBased), "neighbour" (NeighbourBased) or
    "threshold" (Thresholded); ``threshold`` (1e-6) of the latter; ``cutoff_reduction`` (Bohr,
    0) of the latter two: atom pairs beyond the largest SK cutoff minus it have no exchange."""
    kind: str = "lc"
    screening: str = "matrix"
    threshold: float = 1e-6
    cutoff_reduction: float = 0.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"hybrid type must be 'lc' or 'cam' (DFTB+ has no 'Global': a global "
                             f"hybrid is 'cam' with beta = 0 SK files), not {self.kind!r}")
        if self.screening not in SCREENINGS:
            raise ValueError(f"hybrid screening must be one of {SCREENINGS}, not "
                             f"{self.screening!r}")
        if self.cutoff_reduction < 0.0:
            raise ValueError("Cutoff reduction for hybrid xc-functional neighbours should be zero "
                             "or positive (as DFTB+)")
        if self.screening == "matrix" and self.cutoff_reduction != 0.0:
            raise ValueError("the matrix screening has no cutoff_reduction (DFTB+ MatrixBased)")
        if not self.threshold >= 0.0:
            raise ValueError("the hybrid screening threshold must not be negative")

    @classmethod
    def make(cls, spec) -> "Hybrid | None":
        """``None``, "lc", "cam", a dict ({"type": ..., "screening": ..., "threshold": ...,
        "cutoff_reduction": ...}) or a :class:`Hybrid`."""
        if spec is None or isinstance(spec, cls):
            return spec
        if isinstance(spec, str):
            return cls(kind=spec.lower())
        if isinstance(spec, dict):
            d = dict(spec)
            if "type" in d:
                d["kind"] = d.pop("type")
            unknown = set(d) - {"kind", "screening", "threshold", "cutoff_reduction"}
            if unknown:
                raise ValueError(f"hybrid: unknown keys {sorted(unknown)} (use 'type', "
                                 "'screening', 'threshold', 'cutoff_reduction')")
            if "kind" in d:
                d["kind"] = str(d["kind"]).lower()
            if "screening" in d:
                d["screening"] = str(d["screening"]).lower()
            return cls(**d)
        raise TypeError(f"hybrid must be None, 'lc', 'cam', a dict or a Hybrid, not {spec!r}")


# ---- gamma (DFTB+ rshgamma.F90) -------------------------------------------------------------
def _y(ta, tb, r, w):
    """getYGammaSubPart."""
    prefac = ta ** 2 / ((ta - w) * (ta + w))
    tmp = (tb ** 6 - 3.0 * ta ** 2 * tb ** 4 + 2.0 * w ** 2 * tb ** 4) / r
    tmp = tmp * prefac * prefac / (ta ** 2 - tb ** 2) ** 3
    tmp = ta * tb ** 4 * 0.5 * prefac / (tb ** 2 - ta ** 2) ** 2 - tmp
    return tmp * np.exp(-ta * r)


def _dy(ta, tb, r, w):
    """getdYGammaSubPart: d/dr of :func:`_y`."""
    inv = 1.0 / r
    prefac = ta ** 2 / (ta ** 2 - w ** 2)
    tmp = prefac * prefac / (ta ** 2 - tb ** 2) ** 3
    poly = tb ** 6 - 3.0 * ta ** 2 * tb ** 4 + 2.0 * w ** 2 * tb ** 4
    dtmp = tmp * poly * inv ** 2
    tmp = tmp * poly * inv
    tmp2 = ta * tb ** 4 * 0.5 * prefac / (tb ** 2 - ta ** 2) ** 2 - tmp
    return (dtmp - tmp2 * ta) * np.exp(-ta * r)


def _branches(ua, ub, r):
    """(tau_a, tau_b, r, on site, equal U, nearly equal U, unequal U)."""
    ua, ub, r = np.broadcast_arrays(np.asarray(ua, float), np.asarray(ub, float),
                                    np.asarray(r, float))
    ta, tb = 3.2 * ua, 3.2 * ub
    on = r < TOL_SAME_DIST
    eq = ~on & (np.abs(ta - tb) < MIN_HUB_DIFF)
    near = ~on & ~eq & (np.abs(ta - tb) < NEAR_TAU * 0.5 * (ta + tb))
    return ta, tb, r, on, eq, near, ~on & ~eq & ~near


def _yukawa_near(r, ta, tb, w):
    """The Yukawa integral (kernel exp(-w r) / r) of the two densities and its r-derivative:
    -A^2 B^2 E[A, A, B, B] with A = ta^2, B = tb^2 and E(y) = (exp(-w r) - exp(-sqrt(y) r)) /
    (r (y - w^2)), by the Hermite-Genocchi integral as :func:`pydftb.coulomb._short_near`
    (f is the third derivative of E in y, fr its r-derivative)."""
    s, wt = _GL_S[:, None], _GL_W[:, None]
    a2, b2 = ta * ta, tb * tb
    q = np.sqrt(a2 + s * (b2 - a2))
    e, ew = np.exp(-q * r), np.exp(-w * r)
    q2, w2 = q * q, w * w
    d = q2 - w2
    f = (r ** 3 * e * q2 * d ** 3 + 3.0 * r ** 2 * e * q * d ** 2 * (3.0 * q2 - w2)
         + 3.0 * r * e * d * (11.0 * q2 * q2 - 4.0 * q2 * w2 + w2 * w2)
         + 48.0 * q ** 5 * (e - ew)) / (8.0 * r * q ** 5 * d ** 4)
    fr = -(r ** 4 * e * q * d ** 3 + r ** 3 * e * d ** 2 * (7.0 * q2 - w2)
           + 24.0 * r ** 2 * e * q ** 3 * d + 48.0 * r * q ** 3 * (e * q - ew * w)
           + 48.0 * q ** 3 * (e - ew)) / (8.0 * r ** 2 * q ** 3 * d ** 4)
    pre = -a2 * a2 * b2 * b2
    return pre * (wt * f).sum(0), pre * (wt * fr).sum(0)


def gamma_hf(ua, ub, r):
    """The full-range (Hartree-Fock) gamma of Hubbard values ``ua``, ``ub`` at distance ``r``
    (DFTB+ getHfAnalyticalGammaValue_workhorse)."""
    ta, tb, r, on, eq, near, ne = _branches(ua, ub, r)
    out = np.zeros(r.shape)
    out[on] = 0.5 * (ta[on] + tb[on]) * 0.3125
    if eq.any():
        t, d = 0.5 * (ta[eq] + tb[eq]), r[eq]
        inv, x = 1.0 / d, d * t
        out[eq] = inv - (x ** 3 / 48.0 + 0.1875 * x ** 2 + 0.6875 * x + 1.0) * np.exp(-t * d) * inv
    if near.any():                       # 1/r minus the SCC's short-range part
        out[near] = 1.0 / r[near] - _short_near(r[near], ta[near], tb[near])[0]
    if ne.any():
        a, b, d = ta[ne], tb[ne], r[ne]
        out[ne] = 1.0 / d - _y(a, b, d, 0.0) - _y(b, a, d, 0.0)
    return out


def gamma_lr(ua, ub, omega, r):
    """The long-range gamma (DFTB+ getLrAnalyticalGammaValue_workhorse)."""
    ta, tb, r, on, eq, near, ne = _branches(ua, ub, r)
    w = float(omega)
    w2 = w * w
    out = np.zeros(r.shape)
    if near.any():                       # gamma_HF minus the Yukawa integral
        rn, an, bn = r[near], ta[near], tb[near]
        out[near] = (1.0 / rn - _short_near(rn, an, bn)[0]) - _yukawa_near(rn, an, bn, w)[0]
    if on.any():
        t = 0.5 * (ta[on] + tb[on])
        tmp = 5.0 * t ** 6 + 15.0 * t ** 4 * w2 - 5.0 * t ** 2 * w ** 4 + w ** 6
        tmp = tmp * 0.0625 - w * t ** 5
        tmp = tmp * t ** 3 / (t ** 2 - w2) ** 4
        out[on] = t * 0.3125 - tmp
    if eq.any():
        t, d = 0.5 * (ta[eq] + tb[eq]), r[eq]
        inv, x = 1.0 / d, d * t
        tmp2 = (x ** 3 / 48.0 + 0.1875 * x ** 2 + 0.6875 * x + 1.0) * np.exp(-t * d) * inv
        tmp = -t ** 8 / (t ** 2 - w2) ** 4 * (tmp2 + np.exp(-t * d) * (
            d ** 2 * (3.0 * t ** 4 * w ** 4 - 3.0 * t ** 6 * w2 - t ** 2 * w ** 6)
            + d * (15.0 * t ** 3 * w ** 4 - 21.0 * t ** 5 * w2 - 3.0 * t * w ** 6)
            + (15.0 * t ** 2 * w ** 4 - 45.0 * t ** 4 * w2 - 3.0 * w ** 6)) / (48.0 * t ** 5))
        out[eq] = 1.0 / d - tmp2 - (t ** 8 / (t ** 2 - w2) ** 4 * np.exp(-w * d) * inv + tmp)
    if ne.any():
        a, b, d = ta[ne], tb[ne], r[ne]
        inv = 1.0 / d
        prefac = a ** 4 / (a ** 2 - w2) ** 2 * b ** 4 / (b ** 2 - w2) ** 2 * np.exp(-w * d) * inv
        tmp = inv - (prefac - _y(a, b, d, w) - _y(b, a, d, w))
        out[ne] = tmp - _y(a, b, d, 0.0) - _y(b, a, d, 0.0)
    return out


def dgamma_hf(ua, ub, r):
    """d gamma_HF / dr (DFTB+ getdHfAnalyticalGammaValue_workhorse; 0 on site)."""
    ta, tb, r, on, eq, near, ne = _branches(ua, ub, r)
    out = np.zeros(r.shape)
    if near.any():
        out[near] = -1.0 / r[near] ** 2 - _short_near(r[near], ta[near], tb[near])[1]
    if eq.any():
        t, d = 0.5 * (ta[eq] + tb[eq]), r[eq]
        inv2 = 1.0 / d ** 2
        e = np.exp(-t * d)
        dtmp = ((2.0 * d * t ** 3 / 48.0 + 0.1875 * t ** 2 - inv2) * e
                - (d ** 2 * t ** 3 / 48.0 + 0.1875 * d * t ** 2 + 0.6875 * t + 1.0 / d) * t * e)
        out[eq] = -inv2 - dtmp
    if ne.any():
        a, b, d = ta[ne], tb[ne], r[ne]
        out[ne] = -1.0 / d ** 2 - _dy(a, b, d, 0.0) - _dy(b, a, d, 0.0)
    return out


def dgamma_lr(ua, ub, omega, r):
    """d gamma_LR / dr (DFTB+ getdLrAnalyticalGammaValue_workhorse; 0 on site)."""
    ta, tb, r, on, eq, near, ne = _branches(ua, ub, r)
    w = float(omega)
    w2 = w * w
    out = np.zeros(r.shape)
    if near.any():
        rn, an, bn = r[near], ta[near], tb[near]
        out[near] = (-1.0 / rn ** 2 - _short_near(rn, an, bn)[1]) - _yukawa_near(rn, an, bn, w)[1]
    if eq.any():
        t, d = 0.5 * (ta[eq] + tb[eq]), r[eq]
        inv = 1.0 / d
        e, ew = np.exp(-t * d), np.exp(-w * d)
        tmp = (d ** 2 * (3.0 * t ** 4 * w ** 4 - 3.0 * t ** 6 * w2 - t ** 2 * w ** 6)
               + d * (15.0 * t ** 3 * w ** 4 - 21.0 * t ** 5 * w2 - 3.0 * t * w ** 6)
               + (15.0 * t ** 2 * w ** 4 - 45.0 * t ** 4 * w2 - 3.0 * w ** 6))
        dtmp = (2.0 * d * (3.0 * t ** 4 * w ** 4 - 3.0 * t ** 6 * w2 - t ** 2 * w ** 6)
                + (15.0 * t ** 3 * w ** 4 - 21.0 * t ** 5 * w2 - 3.0 * t * w ** 6))
        dtmp = (dtmp * e - tmp * t * e) / (48.0 * t ** 5)
        dtmp2 = ((2.0 * d * t ** 3 / 48.0 + 0.1875 * t ** 2 - inv ** 2) * e
                 - (d ** 2 * t ** 3 / 48.0 + 0.1875 * d * t ** 2 + 0.6875 * t + inv) * t * e)
        out[eq] = (-inv ** 2 - dtmp2 + (t ** 8 / (t ** 2 - w2) ** 4)
                   * (dtmp + dtmp2 + w * ew * inv + ew * inv ** 2))
    if ne.any():
        a, b, d = ta[ne], tb[ne], r[ne]
        inv = 1.0 / d
        ew = np.exp(-w * d)
        prefac = (a ** 4 / (a ** 2 - w2) ** 2 * b ** 4 / (b ** 2 - w2) ** 2
                  * (-w * ew * inv - ew * inv ** 2))
        out[ne] = (-inv ** 2 - prefac + _dy(a, b, d, w) + _dy(b, a, d, w)
                   - _dy(a, b, d, 0.0) - _dy(b, a, d, 0.0))
    return out


def four_terms(s, g, d):
    """(S D S) o G + ((S D) o G) S + S (D o G) S + S ((D S) o G): four times the Mulliken exchange
    operator K^x[D]_munu = sum_kl (mu kappa|nu lambda) D_kl (G the orbital-level gamma), for any
    square D (milestone M6h-1; M6h-2 applies it to transition densities)."""
    sd, ds = s @ d, d @ s
    return (sd @ s) * g + (sd * g) @ s + s @ (d * g) @ s + s @ (ds * g)


def bilinear_ds(s, g, m1, m2):
    """The S coefficients C of B = <M1, K^x[M2]> (M6h-2; S and G symmetric): the derivative
    along a symmetric change of S_munu = S_numu is C_munu + C_numu (C_mumu on the diagonal), as
    the pair terms take both orderings of every element.
    C = 1/4 [(M1 o G) S M2^T + ((M1 S) o G) M2^T + M1 ((S M2^T) o G) + (M1 S)(M2 o G)^T
             + (M1 o G)^T S M2 + M1^T ((S M2) o G) + ((S M1) o G)^T M2 + (S M1)^T (M2 o G)]."""
    c1 = ((m1 * g) @ s @ m2.T + ((m1 @ s) * g) @ m2.T + m1 @ ((s @ m2.T) * g)
          + (m1 @ s) @ (m2 * g).T)
    c2 = ((m1 * g).T @ s @ m2 + m1.T @ ((s @ m2) * g) + ((s @ m1) * g).T @ m2
          + (s @ m1).T @ (m2 * g))
    return 0.25 * (c1 + c2)


def bilinear_dg(s, m1, m2):
    """The orbital-level gamma coefficients of B = <M1, K^x[M2]> (M6h-2), in the same sense as
    :func:`bilinear_ds` (summed over both orderings of an atom pair by :meth:`HybridTerms
    ._gamma_force`): 1/4 [M1 o (S M2 S) + (M1 S) o (S M2) + (S M1) o (M2 S) + M2 o (S M1 S)]."""
    return 0.25 * (m1 * (s @ m2 @ s) + (m1 @ s) * (s @ m2) + (s @ m1) * (m2 @ s)
                   + m2 * (s @ m1 @ s))


def gamma_cam(ua, ub, r, omega, alpha, beta):
    """alpha gamma_HF + beta gamma_LR(omega), a term left out when its coefficient is below
    100 eps (DFTB+ getCamAnalyticalGammaValue_workhorse)."""
    out = np.zeros(np.broadcast(np.asarray(ua), np.asarray(ub), np.asarray(r)).shape)
    if abs(alpha) > CAM_EPS:
        out = out + alpha * gamma_hf(ua, ub, r)
    if abs(beta) > CAM_EPS:
        out = out + beta * gamma_lr(ua, ub, omega, r)
    return out


def dgamma_cam(ua, ub, r, omega, alpha, beta):
    """d/dr of :func:`gamma_cam`."""
    out = np.zeros(np.broadcast(np.asarray(ua), np.asarray(ub), np.asarray(r)).shape)
    if abs(alpha) > CAM_EPS:
        out = out + alpha * dgamma_hf(ua, ub, r)
    if abs(beta) > CAM_EPS:
        out = out + beta * dgamma_lr(ua, ub, omega, r)
    return out


# ---- the terms of one geometry --------------------------------------------------------------
class HybridTerms:
    """The geometry-dependent parts of the exchange (``_Setup.hybrid``) and its Fock builds.

    ``params``: (kind, omega, alpha, beta) of the SK files; ``u_atom``: the first shell's
    Hubbard U of every atom; ``smat``: the dense real overlap; ``q0_orb``: the reference
    population of every orbital; ``sk_cutoff``: the largest SK cutoff (the neighbour and
    threshold screening's CAM cutoff is that minus the reduction); ``pauli``: non-collinear
    spin (M6h-1b, the matrix screening): the densities are one 2N x 2N spinor matrix."""

    # validation only (M6h-1b): the two-component gamma term as DFTB+'s addCamGradientsMatrix_pauli,
    # from the real parts of DeltaP's Pauli components (pauli2components): not the energy's
    # derivative when the components have imaginary parts
    _dftbp_pauli_gamma = False

    def __init__(self, spec: Hybrid, params, u_atom, pos, orb_atom, smat, q0_orb, sk_cutoff,
                 pauli: bool = False):
        self.spec = spec
        self.pauli = pauli
        self.kind, self.omega, self.alpha, self.beta = params
        pos = np.asarray(pos, float)
        nat = len(pos)
        self.vec = pos[:, None, :] - pos[None, :, :]                 # R_A - R_B
        self.dist = np.sqrt((self.vec ** 2).sum(-1))
        u = np.asarray(u_atom, float)
        ua, ub = np.broadcast_arrays(u[:, None], u[None, :])
        self.g_atom = gamma_cam(ua, ub, self.dist, self.omega, self.alpha, self.beta)
        self.dg_atom = dgamma_cam(ua, ub, self.dist, self.omega, self.alpha, self.beta)
        self.orb_atom = np.asarray(orb_atom)
        self.offsets = np.concatenate([[0], np.cumsum(np.bincount(self.orb_atom,
                                                                  minlength=nat))])
        self.G = self.g_atom[np.ix_(self.orb_atom, self.orb_atom)]
        self.S = np.asarray(smat, float)
        self.q0_orb = np.asarray(q0_orb, float)
        self.n = len(self.S)
        self.tril = np.tril_indices(2 * self.n if pauli else self.n)
        cam_cutoff = sk_cutoff - spec.cutoff_reduction
        if cam_cutoff < 0.0:
            raise ValueError("Screening cutoff for hybrid xc-functional neighbours too short "
                             "(as DFTB+)")
        self.neighbours = self.dist < cam_cutoff
        mask = self.neighbours[np.ix_(self.orb_atom, self.orb_atom)]
        self.S_cam = self.S * mask
        self._h_prev = None                    # the threshold screening's state
        self._dp_prev = None
        self._t_max = None

    # -- densities ------------------------------------------------------------------------
    @property
    def packed_size(self) -> int:
        """Length of one spin channel's packed DeltaP (the lower triangle, diagonal included);
        two-component: the real and imaginary parts of the spinor matrix's lower triangle."""
        return 2 * len(self.tril[0]) if self.pauli else len(self.tril[0])

    def reference(self, nspin: int) -> np.ndarray:
        """P0 of one spin channel: the reference populations on the diagonal, half each with
        spin; two-component: half on both spin blocks of the spinor matrix (DFTB+
        denseSubtractDensityOfAtomsPauli)."""
        if self.pauli:
            return np.diag(np.tile(0.5 * self.q0_orb, 2)).astype(complex)
        return np.diag(self.q0_orb * (0.5 if nspin == 2 else 1.0))

    def pack(self, dps) -> np.ndarray:
        """The vector the SCC mixes: every channel's lower triangle (DFTB+ mixes the full
        matrices with a zero upper triangle: the same dot products); two-component: [Re | Im] of
        the spinor matrix's (DFTB+'s complex mixers take the real part of conj(x) y: the same
        mixing)."""
        if self.pauli:
            t = dps[0][self.tril]
            return np.concatenate([t.real, t.imag])
        return np.concatenate([d[self.tril] for d in dps])

    def unpack(self, vec, nspin: int) -> list:
        k = self.packed_size
        nmat = 1 if self.pauli else nspin
        if len(vec) != k * nmat:
            what = ("the packed two-component DeltaP" if self.pauli
                    else f"the packed DeltaP of {nspin} spin channel(s)")
            raise ValueError(f"a hybrid calculation's mixed vector (q_init) has {k * nmat} "
                             f"elements ({what}), not {len(vec)}")
        if self.pauli:
            n2, h = 2 * self.n, k // 2
            m = np.zeros((n2, n2), complex)
            m[self.tril] = vec[:h] + 1j * vec[h:]
            return [m + np.tril(m, -1).conj().T]
        out = []
        for s in range(nspin):
            m = np.zeros((self.n, self.n))
            m[self.tril] = vec[s * k:(s + 1) * k]
            out.append(m + np.tril(m, -1).T)
        return out

    def error(self, x_out, x_in) -> float:
        """The SCC error max|DeltaP_out - DeltaP_in|; two-component: of the complex elements
        (DFTB+ getNextInputDensityPauli)."""
        d = np.asarray(x_out) - np.asarray(x_in)
        if self.pauli:
            h = len(d) // 2
            return float(np.hypot(d[:h], d[h:]).max())
        return float(np.abs(d).max())

    # -- Fock matrices and energy -----------------------------------------------------------
    def reset(self, restart: bool):
        """Start of an SCC: the threshold screening's increments begin again (DFTB+ resets them
        when the geometry changes; with a starting DeltaP the first build takes all of it)."""
        self._h_prev = np.zeros((self.n, self.n)) if restart else None
        self._dp_prev = np.zeros((self.n, self.n)) if restart else None

    def fock(self, dps) -> list:
        """H_x[DeltaP] of every spin channel; two-component: of the spinor matrix, the operator
        on each of its blocks (H_x of the upper right block the adjoint of the lower left's)."""
        if self.pauli:
            n, d = self.n, dps[0]
            h = np.empty_like(d)
            for r, c in ((0, 0), (1, 1), (1, 0)):
                h[r * n:(r + 1) * n, c * n:(c + 1) * n] = -0.25 * self._four_terms(
                    self.S, d[r * n:(r + 1) * n, c * n:(c + 1) * n])
            h[:n, n:] = h[n:, :n].conj().T
            return [h]
        if self.spec.screening == "threshold":
            return [self._fock_threshold(dps[0])]
        s = self.S_cam if self.spec.screening == "neighbour" else self.S
        c = -0.125 if len(dps) == 1 else -0.25
        return [c * self._four_terms(s, d) for d in dps]

    def _four_terms(self, s, d):
        return four_terms(s, self.G, d)

    def _fock_threshold(self, dp):
        """DFTB+ addCamHamiltonianThreshold_cluster: H_prev += the change from DeltaDeltaP =
        DeltaP - the previous DeltaP, the atom pairs (C, D) of the block (A, B <= A) left out
        when max|DDP| max|S_CA| or that times max|S_DB| is below the threshold; restricted.
        Only the atoms that can pass enter the products (the same sums, without the zeros)."""
        n, off = self.n, self.offsets
        if self._h_prev is None:              # checkAndInitScreening: no change at the start
            self._h_prev = np.zeros((n, n))
            self._dp_prev = dp.copy()
        ddp = dp - self._dp_prev
        self._dp_prev = dp.copy()
        t = self._block_overlap_max()
        pbound = np.abs(ddp).max()
        thr = self.spec.threshold
        g, oa, gorb, s = self.g_atom, self.orb_atom, self.G, self.S
        dh = np.zeros((n, n))
        for A in range(len(off) - 1):
            pc = pbound * t[:, A]                                       # per C
            cs = pc >= thr
            if not cs.any():
                continue
            rows = np.nonzero(cs[oa])[0]
            pr = pc[oa[rows]]
            sa = s[rows, off[A]:off[A + 1]]
            top = pc[cs].max()
            for B in range(A + 1):
                ds = top * t[:, B] >= thr                               # D that can pass
                if not ds.any():
                    continue
                cols = np.nonzero(ds[oa])[0]
                m = pr[:, None] * t[oa[cols], B][None, :] >= thr       # (C, D) of the orbitals
                w = (g[A, B] + gorb[rows, off[B]][:, None] + gorb[off[A], cols][None, :]
                     + gorb[np.ix_(rows, cols)]) * m
                dh[off[A]:off[A + 1], off[B]:off[B + 1]] += (
                    sa.T @ (ddp[np.ix_(rows, cols)] * (-0.125 * w)) @ s[cols, off[B]:off[B + 1]])
        dh = np.tril(dh) + np.tril(dh, -1).T
        self._h_prev = self._h_prev + dh
        return self._h_prev.copy()

    def _block_overlap_max(self) -> np.ndarray:
        """max |S| of every atom pair's block, zero beyond the CAM cutoff (DFTB+ testOvr)."""
        if self._t_max is None:
            starts = self.offsets[:-1]
            a = np.abs(self.S)
            t = np.maximum.reduceat(np.maximum.reduceat(a, starts, axis=0), starts, axis=1)
            self._t_max = np.where(self.neighbours, t, 0.0)
        return self._t_max

    def energy(self, focks, dps) -> float:
        """E_Fock = 1/2 sum_spin sum H_x o DeltaP; two-component 1/2 Re sum H_x o conj(DeltaP)
        (DFTB+ evaluateEnergy_pauli)."""
        if self.pauli:
            return float(0.5 * np.sum(focks[0] * dps[0].conj()).real)
        return float(sum(0.5 * np.sum(h * d) for h, d in zip(focks, dps)))

    def _spinor(self):
        """S x 1 and G on all four blocks: the overlap and gamma of the spinor matrices."""
        n = self.n
        s2 = np.zeros((2 * n, 2 * n))
        s2[:n, :n] = s2[n:, n:] = self.S
        return s2, np.tile(self.G, (2, 2))

    # -- forces ---------------------------------------------------------------------------
    def pair_coefficient(self, dps, pairs) -> np.ndarray:
        """dE_x/dS at every pair element (the S derivatives at fixed DeltaP_out; with the band
        terms' coefficient of dS): c [(D o G) S D + ((D S) o G) D + (D S)(D o G) + D ((S D) o G)],
        summed over the spin channels (exact, whatever the screening, as DFTB+).  Two-component:
        the real part of the same on the spinor matrices, its two diagonal blocks summed (S x 1)."""
        if self.pauli:
            s, g = self._spinor()
            d = dps[0]
            ds, sd = d @ s, s @ d
            x = (-0.25 * ((d * g) @ sd + (ds * g) @ d + ds @ (d * g) + d @ (sd * g))).real
            n = self.n
            return (x[:n, :n] + x[n:, n:])[pairs.rows, pairs.cols]
        s, g = self.S, self.G
        c = -0.125 if len(dps) == 1 else -0.25
        x = np.zeros_like(s)
        for d in dps:
            ds, sd = d @ s, s @ d
            x = x + c * ((d * g) @ sd + (ds * g) @ d + ds @ (d * g) + d @ (sd * g))
        return x[pairs.rows, pairs.cols]

    def gamma_gradient(self, dps) -> np.ndarray:
        """dE_x/dR_A through gamma: dE/dG = c [D o (S D S) + (D S) o (S D)] summed over each
        atom pair's orbitals, times gamma'(R_AB) (R_A - R_B) / R_AB, over all atom pairs."""
        if self.pauli:          # Re[conj(D) o (S D S) + (S D)^T o (S D)], the four blocks summed
            s, _ = self._spinor()
            d = dps[0]
            if self._dftbp_pauli_gamma:
                n = self.n
                uu, ud, du, dd = d[:n, :n], d[:n, n:], d[n:, :n], d[n:, n:]
                q, mz = (uu + dd).real, (uu - dd).real
                mx, my = (ud + du).real, (1j * (ud - du)).real
                d = 0.5 * np.block([[q + mz, mx - 1j * my], [mx + 1j * my, q - mz]])
            sd = s @ d
            x = (-0.25 * (d.conj() * (sd @ s) + sd.T * sd)).real
            n = self.n
            de = x[:n, :n] + x[:n, n:] + x[n:, :n] + x[n:, n:]
        else:
            s = self.S
            c = -0.125 if len(dps) == 1 else -0.25
            de = np.zeros_like(s)
            for d in dps:
                ds, sd = d @ s, s @ d
                de = de + c * (d * (sd @ s) + ds * sd)
        return self._gamma_force(de)

    def _gamma_force(self, de) -> np.ndarray:
        """sum_AB dE/dgamma_AB dgamma_AB/dR from the orbital-level dE/dG (n_atom, 3)."""
        starts = self.offsets[:-1]
        g = np.add.reduceat(np.add.reduceat(de, starts, axis=0), starts, axis=1)
        nat = len(starts)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(np.eye(nat, dtype=bool), 0.0, (g + g.T) * self.dg_atom / self.dist)
        return np.einsum("ab,abx->ax", w, self.vec)

    def exchange_force(self, terms, pairs):
        """The explicit derivatives of sum_k c_k B(M1_k, M2_k), B = <M1, K^x[M2]> (TD-LC
        excited-state forces, milestone M6h-2; ``terms`` = [(c, M1, M2), ...]): the dS
        coefficient at every pair element and the gamma part of the gradient (n_atom, 3).  The
        full S and gamma whatever the screening, as DFTB+'s excited-state gradient."""
        s, g = self.S, self.G
        cds = np.zeros_like(s)
        cdg = np.zeros_like(s)
        for c, m1, m2 in terms:
            cds = cds + c * bilinear_ds(s, g, m1, m2)
            cdg = cdg + c * bilinear_dg(s, m1, m2)
        return cds[pairs.rows, pairs.cols], self._gamma_force(cdg)
