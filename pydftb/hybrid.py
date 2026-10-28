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
Thresholded build).  The forces are always the exact ones, as DFTB+'s.

Periodic systems at the Gamma point (milestone M6h-3a; DFTB+ with one k-point, Gamma): the same
H_x with S -> S(Gamma), the image-folded overlap (NeighbourBased: of the pairs within the reduced
cutoff), gamma -> gt(A, B) = sum_g gamma_c(|R_A - R_B - g|) over DFTB+'s parallelepiped of
lattice translations (|n_i| <= floor(Rg / h_i) + 1, h_i the plane spacings) with the truncated
Coulomb kernel gamma_c (``coulomb="truncated"``: zero from the cutoff Rc on; ``"damped"``: a
quintic on (Rc - 1 Bohr, Rc) matching gamma's value, first and second derivative), and P0 =
diag(q0 / S_mumu(Gamma)).  Defaults as DFTB+: Rc half the smallest plane spacing, Rg = 2 Rc.
The forces are the energy's derivative (NeighbourBased: with the reduced overlap, as DFTB+'s
Gamma-point gradient).

k-points (milestone M6h-3b; DFTB+ with a SupercellFolding diag(N1, N2, N3) without shift,
:class:`KPointHybridTerms`): pydftb's Bloch convention M(k) = sum_T M(0, T) exp(2 pi i k.T), the
k-points a Gamma-centred grid of the folding (whole, or one of every k, -k as DFTB+'s set);
H_x(k) = c [C(S DP S) + C(S DP) S(k) + S(k) C(DP S) + S(k) C(DP) S(k)] with the convolution
C(X)(k) = 1/N sum_q X(q) o G(k - q), G(p)_AB = sum_g gamma_c(|R_B + g - R_A|) exp(2 pi i p.n_g):
the exchange of the Born-von Karman supercell at its Gamma point, evaluated as the product of
the supercell's real-space (BvK) matrices.  The lattice sums are those of DFTB+, over its
parallelepiped about the atoms folded into the cell (which only matters for the untruncated
kernel, ``coulomb="full"``), P0(k) = diag(q0 / S_mumu(k)), E_Fock with DeltaP_out (as DFTB+).
The SCC mixes every k-point's complex DeltaP(k) ("matrix") or the real BvK array DeltaP(n)
("neighbour", S then the reduced overlap), as DFTB+.

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
from .hamiltonian import fold, fold_all
from .kpoints import supercell_folding

TOL_SAME_DIST = 1.0e-5          # DFTB+ tolSameDist (Bohr): on-site below this distance
MIN_HUB_DIFF = 0.3125e-5        # DFTB+ MinHubDiff, on tau = 3.2 U: "equal U" below this
CAM_EPS = 100 * np.finfo(float).eps   # DFTB+ leaves out an alpha or beta term below this
SCREENINGS = ("matrix", "neighbour", "threshold")
KINDS = ("lc", "cam")
COULOMBS = ("truncated", "damped", "full", "minimum_image")   # DFTB+ CoulombMatrix (periodic)
DIST_FUDGE = 1.0                # DFTB+ distFudge (Bohr): the damping region is (Rc - 1, Rc)
LR_DD_DELTA = 1e-8              # DFTB+'s central difference of gamma_LR' at the damping start
IMAGE_CHUNK = 2 ** 20           # atom pairs x translations per block of the lattice sums


@dataclass(frozen=True)
class Hybrid:
    """The hybrid functional to use (DFTB+ ``Hybrid = LC {...}`` or ``CAM {...}``).

    ``kind``: "lc" needs LC SK files; "cam" takes LC, global (beta = 0) and CAM files.
    ``screening``: "matrix" (DFTB+'s default MatrixBased), "neighbour" (NeighbourBased) or
    "threshold" (Thresholded); ``threshold`` (1e-6) of the latter; ``cutoff_reduction`` (Bohr,
    0) of the latter two: atom pairs beyond the largest SK cutoff minus it have no exchange.
    Periodic systems only (DFTB+ ``CoulombMatrix``, milestone M6h-3): ``coulomb`` "truncated"
    (default), "damped" (TruncatedAndDamped) or, at k-points only (M6h-3b, as DFTB+), "full"
    (the untruncated lattice sum over the parallelepiped of ``gsum_cutoff``: it grows with that,
    so the SCC does not settle; for comparisons only); ``coulomb_cutoff`` (Bohr, None: DFTB+'s
    default, half the smallest plane spacing at the Gamma point, :func:`dftbp_coulomb_cutoff` of
    the folding at k-points) and ``gsum_cutoff`` (Bohr, None: twice the former).  A molecule
    refuses them (DFTB+ stops on its unread CoulombMatrix).  At k-points (M6h-3c, as DFTB+)
    also ``coulomb`` "minimum_image" (DFTB+ MinimumImage: the untruncated kernel over the Wigner-
    Seitz cell of the supercell folding reduced by ``wigner_seitz_reduction`` cells,
    :func:`wigner_seitz_vectors`, lattice vectors of the atoms as DFTB+ folds them into the cell,
    so the energy jumps when an atom crosses a face of the cell (B23; up to Hartrees in small
    supercells); MatrixBased and NeighbourBased are different approximations, and the latter has
    no forces; no cutoffs, and not DFTB+'s stop on the default cutoff it computes but does not
    use, B3).  The neighbour screening at k-points builds the whole operator
    unless ``neighbour_threshold`` is given (DFTB+'s ``Threshold``, default 1e-6 there: the
    increments of DeltaP_in screened, which shifts the energy by up to about 1e-5 Ha and makes
    it depend on the SCC's history; M6h-3c)."""
    kind: str = "lc"
    screening: str = "matrix"
    threshold: float = 1e-6
    cutoff_reduction: float = 0.0
    coulomb: str = "truncated"
    coulomb_cutoff: float | None = None
    gsum_cutoff: float | None = None
    wigner_seitz_reduction: int = 0
    neighbour_threshold: float | None = None

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
        if self.coulomb not in COULOMBS:
            raise ValueError(f"hybrid coulomb must be one of {COULOMBS}, not {self.coulomb!r}")
        for name in ("coulomb_cutoff", "gsum_cutoff"):
            v = getattr(self, name)
            if v is not None and not v > 0.0:
                raise ValueError(f"hybrid {name} must be positive (Bohr)")
        r = self.wigner_seitz_reduction                   # M6h-3c
        if isinstance(r, bool) or not isinstance(r, (int, np.integer)) or r < 0:
            raise ValueError("hybrid wigner_seitz_reduction must be a non-negative integer (DFTB+ "
                             "WignerSeitzReduction)")
        if r and self.coulomb != "minimum_image":
            raise ValueError("wigner_seitz_reduction goes with coulomb='minimum_image' (DFTB+ "
                             "reads WignerSeitzReduction in the MinimumImage block only)")
        if self.coulomb == "minimum_image" and (self.coulomb_cutoff is not None
                                                or self.gsum_cutoff is not None):
            raise ValueError("coulomb_cutoff and gsum_cutoff do not go with "
                             "coulomb='minimum_image' (DFTB+ reads neither in the MinimumImage "
                             "block: the Wigner-Seitz cell of the folding and the untruncated "
                             "kernel)")
        if self.neighbour_threshold is not None:
            if self.screening != "neighbour":
                raise ValueError("neighbour_threshold is the Threshold of the neighbour screening "
                                 "(DFTB+ NeighbourBased)")
            if not self.neighbour_threshold >= 0.0:
                raise ValueError("the hybrid neighbour_threshold must not be negative")

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
            unknown = set(d) - {"kind", "screening", "threshold", "cutoff_reduction", "coulomb",
                                "coulomb_cutoff", "gsum_cutoff", "wigner_seitz_reduction",
                                "neighbour_threshold"}
            if unknown:
                raise ValueError(f"hybrid: unknown keys {sorted(unknown)} (use 'type', "
                                 "'screening', 'threshold', 'cutoff_reduction', 'coulomb', "
                                 "'coulomb_cutoff', 'gsum_cutoff', 'wigner_seitz_reduction', "
                                 "'neighbour_threshold')")
            for key in ("kind", "screening", "coulomb"):
                if key in d:
                    d[key] = str(d[key]).lower()
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


def _ddy(ta, tb, r, w):
    """getddYGammaSubPart: d^2/dr^2 of :func:`_y`."""
    prefac = ta * ta / (ta ** 2 - w ** 2)
    tmp = prefac * prefac / (ta * ta - tb * tb) ** 3
    poly = tb ** 6 - 3.0 * ta * ta * tb ** 4 + 2.0 * w * w * tb ** 4
    dtmp = tmp * poly / r ** 2
    ddtmp = -2.0 * tmp * poly / r ** 3
    tmp = tmp * poly / r
    tmp2 = ta * tb ** 4 * 0.5 * prefac / (tb * tb - ta * ta) ** 2 - tmp
    e = np.exp(-ta * r)
    return (ddtmp - tmp / r * ta) * e - (dtmp - tmp2 * ta) * ta * e


def ddgamma_hf(ua, ub, r):
    """d^2 gamma_HF / dr^2 (DFTB+ getddHfAnalyticalGammaValue_workhorse; 0 on site), for the
    damped Coulomb kernel (M6h-3a); for nearly equal U the central difference of
    :func:`dgamma_hf` (its Hermite-Genocchi form; DFTB+'s closed form has no precision there)."""
    ta, tb, r, on, eq, near, ne = _branches(ua, ub, r)
    out = np.zeros(r.shape)
    if near.any():
        a, b, d = ta[near] / 3.2, tb[near] / 3.2, r[near]
        h = 1e-5 * d
        out[near] = (dgamma_hf(a, b, d + h) - dgamma_hf(a, b, d - h)) / (2.0 * h)
    if eq.any():
        t, d = 0.5 * (ta[eq] + tb[eq]), r[eq]
        e = np.exp(-t * d)
        ddtmp = ((2.0 * t ** 3 / 48.0 + 2.0 / d ** 3
                  - t * (2.0 * d * t ** 3 / 48.0 + 0.1875 * t ** 2 - 1.0 / d ** 2)) * e
                 + (d ** 2 * t ** 3 / 48.0 + 0.1875 * d * t ** 2 + 0.6875 * t + 1.0 / d)
                 * t ** 2 * e
                 - (2.0 * d * t ** 3 / 48.0 + 0.1875 * t ** 2 - 1.0 / d ** 2) * t * e)
        out[eq] = 2.0 / d ** 3 - ddtmp
    if ne.any():
        a, b, d = ta[ne], tb[ne], r[ne]
        out[ne] = 2.0 / d ** 3 - _ddy(a, b, d, 0.0) - _ddy(b, a, d, 0.0)
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


# ---- periodic systems (milestone M6h-3a) -----------------------------------------------------
def poly5zero(y0, y1, y2, d, rd, rc, derivative: bool = False):
    """DFTB+ poly5zero: the quintic on (rd, rc) with value, first and second derivative y0, y1,
    y2 at rd and zero value, first and second derivative at rc; with ``derivative`` its d/dd."""
    w = rc - rd
    t = (np.asarray(d, float) - rd) / w
    f, e, q = y0, y1 * w, 0.5 * y2 * w * w
    s0, s1, s2 = -(q + e + f), -(2.0 * q + e), -2.0 * q
    a = 0.5 * (s2 + 12.0 * s0 - 6.0 * s1)
    b = s1 - 3.0 * s0 - 2.0 * a
    c = s0 - a - b
    if derivative:
        return (5.0 * a * t ** 4 + 4.0 * b * t ** 3 + 3.0 * c * t ** 2 + 2.0 * q * t + e) / w
    return a * t ** 5 + b * t ** 4 + c * t ** 3 + q * t ** 2 + e * t + f


def _damping_tables(ua, ub, omega, rd):
    """(LR, HF) value, first and second derivative at the damping start rd (DFTB+: the LR second
    derivative a central difference of the analytical first, the HF one analytical)."""
    dl = LR_DD_DELTA
    lr = (float(gamma_lr(ua, ub, omega, rd)), float(dgamma_lr(ua, ub, omega, rd)),
          float((dgamma_lr(ua, ub, omega, rd + dl) - dgamma_lr(ua, ub, omega, rd - dl))
                / (2.0 * dl)))
    hf = (float(gamma_hf(ua, ub, rd)), float(dgamma_hf(ua, ub, rd)),
          float(ddgamma_hf(ua, ub, rd)))
    return lr, hf


def coulomb_gamma(ua, ub, r, omega, alpha, beta, rc, kind, derivative=False, dftbp_hf=False):
    """gamma (or with ``derivative`` d gamma/dr) of one pair of Hubbard values with DFTB+'s
    truncated Coulomb kernel ``kind`` (M6h-3a): "truncated" zero from ``rc`` on; "damped" also
    :func:`poly5zero` on (rc - 1, rc).  ``dftbp_hf`` (validation only): the HF part's derivative
    in the damping region from the LR table, as DFTB+ 25.1
    (getdHfTruncatedAndDampedGammaValue)."""
    r = np.asarray(r, float)
    out = (dgamma_cam if derivative else gamma_cam)(ua, ub, r, omega, alpha, beta)
    if kind == "damped":
        rd = rc - DIST_FUDGE
        win = (r > rd) & (r < rc)
        if win.any():
            lr, hf = _damping_tables(ua, ub, omega, rd)
            part = np.zeros(int(win.sum()))
            if abs(beta) > CAM_EPS:
                part = part + beta * poly5zero(*lr, r[win], rd, rc, derivative)
            if abs(alpha) > CAM_EPS:
                tab = lr if (derivative and dftbp_hf) else hf
                part = part + alpha * poly5zero(*tab, r[win], rd, rc, derivative)
            out = np.array(out, float)
            out[win] = part
    return np.where(r < rc, out, 0.0)


def plane_spacings(cell) -> np.ndarray:
    """The distances between the lattice planes (cell rows: the lattice vectors)."""
    cell = np.asarray(cell, float)
    vol = abs(np.linalg.det(cell))
    return np.array([vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
                     for i in range(3)])


def default_coulomb_cutoff(cell) -> float:
    """DFTB+'s default CoulombCutoff of a Gamma-point run: half the smallest plane spacing (the
    radius of the largest sphere about the cell's centre within it)."""
    return 0.5 * float(plane_spacings(cell).min())


def dftbp_coulomb_cutoff(cell, nk=(1, 1, 1)) -> float:
    """DFTB+ 25.1's default CoulombCutoff (getCoulombTruncationCutoff) as coded, of a k-point run
    with the supercell folding ``nk`` (milestone M6h-3b): the "BvK vectors" are the lattice
    vectors with their Cartesian components scaled by nk (B2: not the vectors scaled, so with an
    anisotropic folding not the half plane spacing of the supercell), the cutoff the distance from
    the crossing of two of their parallelepiped's body diagonals to its nearest face.  DFTB+ finds
    the crossing from the diagonals' x and y components and checks z; where their x-y
    projections are parallel (e.g. a2 along z) it goes on with an unset point and stops on the
    damping region (B3), at the Gamma point too: the same refusal here (a ``coulomb_cutoff``
    avoids it).  For nk = 1 the value is :func:`default_coulomb_cutoff`'s, up to rounding."""
    lat = np.asarray(cell, float).T * np.asarray(nk, float)[:, None]   # columns: BvK vectors
    b1, b2, b3 = lat[:, 0], lat[:, 1], lat[:, 2]
    s1, e1 = b2, b2 + (b1 - b2 + b3)            # the diagonals (start, end)
    s2, e2 = np.zeros(3), b1 + b2 + b3
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = (((s2[0] - s1[0]) * (s2[1] - e2[1]) - (s2[1] - s1[1]) * (s2[0] - e2[0]))
              / ((s2[1] - e2[1]) * (e1[0] - s1[0]) - (s2[0] - e2[0]) * (e1[1] - s1[1])))
        ss = (((e1[1] - s1[1]) * (s2[0] - s1[0]) - (e1[0] - s1[0]) * (s2[1] - s1[1]))
              / ((e1[1] - s1[1]) * (s2[0] - e2[0]) - (e1[0] - s1[0]) * (s2[1] - e2[1])))
        z = (tt * (e1[2] - s1[2]) + ss * (s2[2] - e2[2])) - (s2[2] - s1[2])
    if not z < 1e-10:
        raise ValueError("The beginning of the damping region of the truncated Coulomb kernel "
                         "must be at a positive radius (as DFTB+ 25.1, whose default Coulomb "
                         "cutoff is undefined for this cell: it intersects the cell's body "
                         "diagonals from their x-y projections, which are parallel here; give "
                         "the hybrid's coulomb_cutoff)")
    x1 = s1 + tt * (e1 - s1)
    if not (abs(tt) <= 1.0 and abs(ss) <= 1.0) or np.linalg.norm(s2 + ss * (e2 - s2) - x1) > 1e-10:
        raise ValueError("Error while calculating Coulomb truncation cutoff for given cell "
                         "geometry. (as DFTB+; give the hybrid's coulomb_cutoff)")
    normals = (np.cross(b1, b2), np.cross(b3, b1), np.cross(b2, b3))
    return float(min(abs(x1 @ (v / np.linalg.norm(v))) for v in normals))


def dftbp_fold_shifts(pos, cell) -> np.ndarray:
    """The lattice translations DFTB+ takes off the atoms when it folds them into the cell
    (boundarycond.F90 foldCoordsToCell; milestone M6h-3b): floor of the fractional coordinates,
    one more where |a_i| (1 - the remainder) < eps (an atom within eps of the far face goes to
    the near one); (n_atom, 3) integers.  DFTB+'s lattice sums and k-point charge files are of
    the folded atoms."""
    cell = np.asarray(cell, float)
    frac = np.asarray(pos, float) @ np.linalg.inv(cell)
    t = np.floor(frac)
    vlen = np.linalg.norm(cell, axis=1)
    t = t + (np.abs(vlen * (1.0 - (frac - t))) < np.finfo(float).eps)
    return t.astype(int)


def bvk_cells(nfold) -> np.ndarray:
    """The cells of the Born-von Karman supercell diag(``nfold``) in DFTB+'s order (n1 fastest),
    (N1 N2 N3, 3) integers (milestone M6h-3b)."""
    n1, n2, n3 = (int(x) for x in nfold)
    i = np.arange(n1 * n2 * n3)
    return np.stack([i % n1, (i // n1) % n2, i // (n1 * n2)], axis=1)


def lattice_translations(cell, rg) -> np.ndarray:
    """DFTB+ getCellTranslations: the lattice vectors n.cell with |n_i| <= floor(rg / h_i) + 1
    (a parallelepiped), Cartesian, (n_g, 3)."""
    nmax = np.floor(rg / plane_spacings(cell)).astype(int) + 1
    n = np.array(np.meshgrid(*[np.arange(-m, m + 1) for m in nmax], indexing="ij"))
    return n.reshape(3, -1).T @ np.asarray(cell, float)


def wigner_seitz_vectors(nfold, cell, reduction: int = 0, search: int = 2,
                         tol: float = 1e-8) -> np.ndarray:
    """DFTB+ generateWignerSeitzGrid (math/wignerseitz.F90) for MinimumImage (milestone
    M6h-3c): the integer lattice vectors n (n_ws, 3) of the Wigner-Seitz cell of the supercell
    N' = max(N - reduction, 1): n in [-2N', 2N']^3 whose metric length |n.A| is the smallest of
    n - i N' (i in [-2, 2]^3) within ``tol`` (Bohr^2), the ties on the cell's surface all kept,
    each with weight 1 (B9), by length (a stable sort; DFTB+'s heap sort orders ties otherwise)."""
    nk = np.maximum(np.asarray(nfold, int) - int(reduction), 1)
    lat = np.asarray(cell, float)
    metric = lat @ lat.T
    n_all = np.array(np.meshgrid(*[np.arange(-search * n, search * n + 1) for n in nk],
                                 indexing="ij")).reshape(3, -1).T
    shift = np.array(np.meshgrid(*[np.arange(-search, search + 1)] * 3,
                                 indexing="ij")).reshape(3, -1).T * nk
    centre = int(np.flatnonzero((shift == 0).all(1))[0])
    keep = []
    for i0 in range(0, len(n_all), 4096):
        diff = n_all[i0:i0 + 4096, None, :] - shift[None, :, :]
        dist = np.einsum("nai,ij,naj->na", diff, metric, diff)
        keep.append(np.abs(dist[:, centre] - dist.min(1)) <= tol)
    ws = n_all[np.concatenate(keep)]
    return ws[np.argsort(np.linalg.norm(ws @ lat, axis=1), kind="mergesort")]


def dftbp_dense_mulliken(dp, s, orb_atom) -> np.ndarray:
    """DFTB+ denseMullikenReal on the lower triangle of ``dp`` (validation only, M6h-3a): per
    orbital, every block below the diagonal by its column sums and, for two different atoms, by
    its row sums; the atoms' own blocks only by column sums, so half of their off-diagonal terms
    is lost when S(Gamma) has them (DFTB+'s Gamma-point q_in of a hybrid run)."""
    low = np.tril(dp) * s
    oa = np.asarray(orb_atom)
    return low.sum(axis=0) + (low * (oa[:, None] > oa[None, :])).sum(axis=1)


# ---- the terms of one geometry --------------------------------------------------------------
class HybridTerms:
    """The geometry-dependent parts of the exchange (``_Setup.hybrid``) and its Fock builds.

    ``params``: (kind, omega, alpha, beta) of the SK files; ``u_atom``: the first shell's
    Hubbard U of every atom; ``smat``: the dense real overlap; ``q0_orb``: the reference
    population of every orbital; ``sk_cutoff``: the largest SK cutoff (the neighbour and
    threshold screening's CAM cutoff is that minus the reduction); ``pauli``: non-collinear
    spin (M6h-1b, the matrix screening): the densities are one 2N x 2N spinor matrix.
    Periodic, at the Gamma point (M6h-3a): ``cell`` (Bohr, rows the lattice vectors), ``pairs``
    the pair blocks with their images, ``smat`` then S(Gamma)."""

    # validation only (M6h-1b): the two-component gamma term as DFTB+'s addCamGradientsMatrix_pauli,
    # from the real parts of DeltaP's Pauli components (pauli2components): not the energy's
    # derivative when the components have imaginary parts
    _dftbp_pauli_gamma = False
    # validation only (M6h-3a): the damped HF part's derivative from the LR table, as DFTB+ 25.1
    # (getdHfTruncatedAndDampedGammaValue): not the energy's derivative with alpha != 0
    _dftbp_hf_damped_derivative = False

    def __init__(self, spec: Hybrid, params, u_atom, pos, orb_atom, smat, q0_orb, sk_cutoff,
                 pauli: bool = False, cell=None, pairs=None):
        self.spec = spec
        self.pauli = pauli
        self.kind, self.omega, self.alpha, self.beta = params
        pos = np.asarray(pos, float)
        nat = len(pos)
        u = np.asarray(u_atom, float)
        self.orb_atom = np.asarray(orb_atom)
        self.offsets = np.concatenate([[0], np.cumsum(np.bincount(self.orb_atom,
                                                                  minlength=nat))])
        self.S = np.asarray(smat, float)
        self.q0_orb = np.asarray(q0_orb, float)
        self.n = len(self.S)
        self.tril = np.tril_indices(2 * self.n if pauli else self.n)
        cam_cutoff = sk_cutoff - spec.cutoff_reduction
        if cam_cutoff < 0.0:
            raise ValueError("Screening cutoff for hybrid xc-functional neighbours too short "
                             "(as DFTB+)")
        self._periodic = cell is not None
        self._cam_cutoff = cam_cutoff
        if self._periodic:
            self._periodic_setup(np.asarray(cell, float), pos, u, pairs)
        else:
            self.vec = pos[:, None, :] - pos[None, :, :]                 # R_A - R_B
            self.dist = np.sqrt((self.vec ** 2).sum(-1))
            ua, ub = np.broadcast_arrays(u[:, None], u[None, :])
            self.g_atom = gamma_cam(ua, ub, self.dist, self.omega, self.alpha, self.beta)
            self.dg_atom = dgamma_cam(ua, ub, self.dist, self.omega, self.alpha, self.beta)
            self.neighbours = self.dist < cam_cutoff
            mask = self.neighbours[np.ix_(self.orb_atom, self.orb_atom)]
            self.S_cam = self.S * mask
        self.G = self.g_atom[np.ix_(self.orb_atom, self.orb_atom)]
        self._h_prev = None                    # the threshold screening's state
        self._dp_prev = None
        self._t_max = None

    def _periodic_setup(self, cell, pos, u, pairs):
        """The Gamma point of a periodic cell (M6h-3a, as DFTB+ with one k-point): gt over the
        lattice translations with the truncated kernel, the reduced S(Gamma) of NeighbourBased
        (the pairs within the CAM cutoff), P0 = q0 / S_mumu(Gamma)."""
        spec = self.spec
        if spec.coulomb not in ("truncated", "damped"):
            raise ValueError("Hybrid functionals don't yet support Gamma point calculations for "
                             "CoulombMatrix types other than 'Truncated' or 'Truncated+Damping' "
                             "(as DFTB+)")
        if spec.coulomb_cutoff is None:
            dftbp_coulomb_cutoff(cell)              # DFTB+ stops where it is undefined (B3)
        rc = (spec.coulomb_cutoff if spec.coulomb_cutoff is not None
              else default_coulomb_cutoff(cell))
        if rc - DIST_FUDGE <= 0.0:                  # DFTB+ checks it whatever the kernel
            raise ValueError("The beginning of the damping region of the truncated Coulomb "
                             "kernel must be at a positive radius (as DFTB+; the Coulomb cutoff "
                             f"is {rc:.4f} Bohr)")
        self.cell, self.rc = cell, rc
        self.rg = spec.gsum_cutoff if spec.gsum_cutoff is not None else 2.0 * rc
        self.gvec = lattice_translations(cell, self.rg)
        self.pos, self.u = pos, u
        self.g_atom = np.zeros((len(pos), len(pos)))
        for ia, ib, v, d in self._image_blocks():
            g = coulomb_gamma(self.u[ia[0]], self.u[ib[0]], d, self.omega, self.alpha, self.beta,
                              rc, spec.coulomb)
            self.g_atom[np.ix_(ia, ib)] += g.sum(-1)
        keep = (pairs.dist < self._cam_cutoff)[pairs.pair_of]        # S(Gamma) as folded
        self.S_cam = fold(pairs, pairs.s * keep, self.n, None, np.ones(self.n))
        self.q0_orb = self.q0_orb / np.diag(self.S)

    def _image_blocks(self):
        """Per pair of Hubbard values: the atoms (ia, ib), R_A - R_B - g (n_a, n_b, n_g, 3) and
        its length, over blocks of the lattice translations g (at most IMAGE_CHUNK atom pairs
        times translations each: one block for small cells)."""
        values, inv = np.unique(self.u, return_inverse=True)
        groups = [np.flatnonzero(inv == k) for k in range(len(values))]
        for ia in groups:
            for ib in groups:
                step = max(1, IMAGE_CHUNK // (len(ia) * len(ib)))
                for g0 in range(0, len(self.gvec), step):
                    v = (self.pos[ia][:, None, None, :] - self.pos[ib][None, :, None, :]
                         - self.gvec[None, None, g0:g0 + step, :])
                    yield ia, ib, v, np.sqrt((v * v).sum(-1))

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

    def response_fock(self, dps) -> list:
        """H_x of first-order densities (M6h-4), every channel's with leading batch axes: the
        exact operator, for the threshold screening its limit (the pairs within the CAM cutoff,
        as its increments) without touching the screening's state; two-component (M6h-4b): of
        the spinor matrices (..., 2N, 2N), as :meth:`fock`."""
        if self.pauli:
            n, out = self.n, []
            for d in dps:
                h = np.empty_like(d)
                for r, c in ((0, 0), (1, 1), (1, 0)):
                    h[..., r * n:(r + 1) * n, c * n:(c + 1) * n] = -0.25 * self._four_terms(
                        self.S, d[..., r * n:(r + 1) * n, c * n:(c + 1) * n])
                h[..., :n, n:] = np.conj(np.swapaxes(h[..., n:, :n], -1, -2))
                out.append(h)
            return out
        s = self.S if self.spec.screening == "matrix" else self.S_cam
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
        reduced = self._periodic and self.spec.screening == "neighbour"
        s, g = (self.S_cam if reduced else self.S), self.G
        c = -0.125 if len(dps) == 1 else -0.25
        x = np.zeros_like(s)
        for d in dps:
            ds, sd = d @ s, s @ d
            x = x + c * ((d * g) @ sd + (ds * g) @ d + ds @ (d * g) + d @ (sd * g))
        if reduced:             # M6h-3a: DFTB+'s Gamma-point gradient is the energy's derivative
            return x[pairs.rows, pairs.cols] * (pairs.dist < self._cam_cutoff)[pairs.pair_of]
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
            s = (self.S_cam if self._periodic and self.spec.screening == "neighbour"
                 else self.S)
            c = -0.125 if len(dps) == 1 else -0.25
            de = np.zeros_like(s)
            for d in dps:
                ds, sd = d @ s, s @ d
                de = de + c * (d * (sd @ s) + ds * sd)
        return self._gamma_force(de)

    def _gamma_force(self, de) -> np.ndarray:
        """sum_AB dE/dgamma_AB dgamma_AB/dR from the orbital-level dE/dG (n_atom, 3); periodic
        (M6h-3a): d gt_AB/dR_A = sum_g gamma_c'(d) (R_A - R_B - g) / d over the translations."""
        starts = self.offsets[:-1]
        g = np.add.reduceat(np.add.reduceat(de, starts, axis=0), starts, axis=1)
        nat = len(starts)
        if self._periodic:
            w = g + g.T
            out = np.zeros((nat, 3))
            for ia, ib, v, d in self._image_blocks():
                dg = coulomb_gamma(self.u[ia[0]], self.u[ib[0]], d, self.omega, self.alpha,
                                   self.beta, self.rc, self.spec.coulomb, derivative=True,
                                   dftbp_hf=self._dftbp_hf_damped_derivative)
                with np.errstate(divide="ignore", invalid="ignore"):
                    f = np.where(d > 0.0, dg / d, 0.0)
                t = np.einsum("abg,abgx->abx", f, v)
                t[ia[:, None] == ib[None, :]] = 0.0          # an atom's own images: no force
                out[ia] += np.einsum("ab,abx->ax", w[np.ix_(ia, ib)], t)
            return out
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


# ---- k-points (milestone M6h-3b) -------------------------------------------------------------
class KPointLayout:
    """The k-points of a periodic hybrid run beyond the Gamma point (milestone M6h-3b) and the
    layout of its mixed DeltaP.

    ``kpts``, ``kw``: the calculation's k-points and weights, the Gamma-centred grid of the
    supercell folding ``nfold`` (N = N1 N2 N3 points), whole or one of every k, -k (weights 2/N,
    1/N where k = -k), so that sum_k w_k Re f(k) = 1/N sum_q f(q) over the grid whenever f(-q) =
    conj f(q).  The Born-von Karman (BvK) matrices X(n) = sum_k w_k Re[X(k) exp(-2 pi i k.n)] of
    the N cells n (:func:`bvk_cells`) are the real-space matrices folded into the supercell,
    X(n)_munu = sum over T = n mod N of X(mu in cell 0, nu in cell T).  ``screening``
    "neighbour" mixes those (DFTB+ NeighbourBased, its deltaRhoInCplxHS), the others [Re | Im] of
    every k-point's DeltaP(k) (MatrixBased, deltaRhoInCplx).  ``shifts``
    (:func:`dftbp_fold_shifts`) convert to and from DFTB+'s charge-file records, which are of the
    folded atoms."""

    def __init__(self, kpts, kw, nfold, orb_atom, screening: str, shifts=None):
        self.kpts = np.atleast_2d(np.asarray(kpts, float))
        self.kw = np.asarray(kw, float) / np.sum(kw)
        self.nfold = tuple(int(x) for x in nfold)
        self.orb_atom = np.asarray(orb_atom)
        self.n = len(self.orb_atom)
        self.nk = len(self.kpts)
        self.neighbour = screening == "neighbour"
        self.cells = bvk_cells(self.nfold)
        self.ncell = len(self.cells)
        theta = 2.0 * np.pi * (self.kpts @ self.cells.T)                     # (nk, N)
        self._cos, self._sin = np.cos(theta), np.sin(theta)
        self._cos_w = (self.kw[:, None] * self._cos).T
        self._sin_w = (self.kw[:, None] * self._sin).T
        self.shifts = None if shifts is None else np.asarray(shifts, int)

    def cell_index(self, m) -> np.ndarray:
        """The BvK cell (index into :attr:`cells`) of integer lattice vectors ``m`` (..., 3)."""
        m = np.mod(np.asarray(m, int), self.nfold)
        n1, n2, _ = self.nfold
        return (m[..., 2] * n2 + m[..., 1]) * n1 + m[..., 0]

    def to_bvk(self, x) -> np.ndarray:
        """The BvK matrices (N, n, n) of k-point matrices ``x`` (nk, n, n)."""
        x = np.asarray(x).reshape(self.nk, -1)
        out = self._cos_w @ x.real
        if np.iscomplexobj(x):
            out = out + self._sin_w @ x.imag
        return out.reshape(self.ncell, self.n, self.n)

    def from_bvk(self, b) -> np.ndarray:
        """X(k) = sum_n X(n) exp(2 pi i k.n) at the k-points (nk, n, n)."""
        b = np.asarray(b, float).reshape(self.ncell, -1)
        return (self._cos @ b + 1j * (self._sin @ b)).reshape(self.nk, self.n, self.n)

    # -- the mixed vector ---------------------------------------------------------------------
    @property
    def packed_size(self) -> int:
        """Length of one spin channel's packed DeltaP."""
        n2 = self.n * self.n
        return self.ncell * n2 if self.neighbour else 2 * self.nk * n2

    def pack(self, dps) -> np.ndarray:
        """The vector the SCC mixes: per spin channel [Re | Im] of DeltaP(k) at every k-point
        (DFTB+'s complex mixers take the real part of conj(x) y: the same mixing), or the BvK
        array."""
        if self.neighbour:
            return np.concatenate([self.to_bvk(d).ravel() for d in dps])
        return np.concatenate([np.concatenate([d.real.ravel(), d.imag.ravel()]) for d in dps])

    def unpack(self, vec, nspin: int) -> list:
        k = self.packed_size
        vec = np.asarray(vec, float)
        if len(vec) != k * nspin:
            raise ValueError(f"a k-point hybrid calculation's mixed vector (q_init) has {k * nspin} "
                             f"elements (the packed DeltaP of {nspin} spin channel(s)), not "
                             f"{len(vec)}")
        out = []
        for s in range(nspin):
            v = vec[s * k:(s + 1) * k]
            if self.neighbour:
                out.append(self.from_bvk(v))
            else:
                h = k // 2
                out.append((v[:h] + 1j * v[h:]).reshape(self.nk, self.n, self.n))
        return out

    def error(self, x_out, x_in) -> float:
        """The SCC error: the largest modulus of DeltaP_out(k) - DeltaP_in(k), or of the BvK
        arrays' difference (DFTB+ getNextInputDensityCplx)."""
        d = np.asarray(x_out) - np.asarray(x_in)
        if self.neighbour:
            return float(np.abs(d).max())
        d = d.reshape(-1, 2, self.nk * self.n * self.n)
        return float(np.hypot(d[:, 0], d[:, 1]).max())

    # -- DFTB+'s charge-file records ----------------------------------------------------------
    def _grid_index(self, k) -> np.ndarray:
        n = np.array(self.nfold)
        i = np.mod(np.rint(np.atleast_2d(np.asarray(k, float)) * n).astype(int), n)
        return (i[:, 0] * n[1] + i[:, 1]) * n[2] + i[:, 2]

    def _match(self, k_from, k_to) -> tuple:
        """For every point of ``k_from``: the point of ``k_to`` equal to it mod 1, or else to its
        inverse (then True)."""
        to = {int(f): j for j, f in enumerate(self._grid_index(k_to))}
        idx, inverse = [], []
        for a, b in zip(self._grid_index(k_from), self._grid_index(-np.asarray(k_from))):
            j = to.get(int(a))
            if j is None:
                j = to.get(int(b))
                if j is None:
                    raise ValueError("the k-points do not cover the folding's grid")
            idx.append(j)
            inverse.append(int(a) not in to)
        return np.array(idx, int), np.array(inverse, bool)

    def _atom_phases(self, k) -> np.ndarray:
        """exp(2 pi i k.T) of every orbital's atom, (n_k, n)."""
        if self.shifts is None:
            raise ValueError("DFTB+'s k-point charge records need the atoms' positions (the "
                             "fold shifts)")
        return np.exp(2j * np.pi * (np.atleast_2d(k) @ self.shifts[self.orb_atom].T))

    def _bvk_shifted(self, b, sign: int) -> np.ndarray:
        """out[m][A, B] = b[m + sign (T_B - T_A) mod N][A, B] for every atom pair (A, B)."""
        if self.shifts is None:
            raise ValueError("DFTB+'s k-point charge records need the atoms' positions (the "
                             "fold shifts)")
        t = self.shifts
        off = np.concatenate([[0], np.cumsum(np.bincount(self.orb_atom))])
        out = np.empty_like(b)
        for a in range(len(off) - 1):
            ra = slice(off[a], off[a + 1])
            for c in range(len(off) - 1):
                rc = slice(off[c], off[c + 1])
                perm = self.cell_index(self.cells + sign * (t[c] - t[a]))
                out[:, ra, rc] = b[perm, ra, rc]
        return out

    def to_dftbp(self, dps) -> np.ndarray:
        """DFTB+'s charge-file record of the DeltaP of every spin channel, as the file holds it
        read in C order: MatrixBased (n_spin, nK, n, n) complex, DeltaP(k) at DFTB+'s k-points
        (:func:`pydftb.kpoints.supercell_folding`), NeighbourBased the (n_spin, N, n, n) BvK
        array, both of the folded atoms (D^dag DeltaP(k) D, D = diag(exp(2 pi i k.T_atom));
        R[n]_AB = DeltaP(m)_AB with m = n - T_B + T_A)."""
        if self.neighbour:
            return np.stack([self._bvk_shifted(self.to_bvk(d), -1) for d in dps])
        kd = supercell_folding(self.nfold)[0]
        j, inverse = self._match(kd, self.kpts)
        ph = self._atom_phases(kd).conj()
        out = []
        for d in dps:
            m = np.array(d[j])
            m[inverse] = m[inverse].conj()
            out.append(ph[:, :, None] * m * ph.conj()[:, None, :])
        return np.stack(out)

    def from_dftbp(self, record, bvk: bool) -> list:
        """The DeltaP(k) of every spin channel at the calculation's k-points from DFTB+'s record
        (:meth:`to_dftbp`; ``bvk``: the NeighbourBased array; either layout serves either
        screening here)."""
        record = np.asarray(record)
        if bvk:
            return [self.from_bvk(self._bvk_shifted(np.asarray(r, float), 1)) for r in record]
        kd = supercell_folding(self.nfold)[0]
        if record.shape[1] != len(kd):
            raise ValueError(f"the k-point record has {record.shape[1]} k-points, the folding "
                             f"{self.nfold} {len(kd)}")
        j, inverse = self._match(self.kpts, kd)
        ph = self._atom_phases(kd)
        out = []
        for r in record:
            p = ph[:, :, None] * r * ph.conj()[:, None, :]
            m = p[j]
            m[inverse] = m[inverse].conj()
            out.append(m)
        return out


class KPointHybridTerms:
    """The exchange of a periodic cell at the k-points of a supercell folding (``_Setup.hybrid``,
    milestone M6h-3b; DFTB+ 25.1 addCamHamiltonianMatrix_kpts and
    addCamHamiltonianNeighbour_kpts_ct), with the interface of :class:`HybridTerms`.

    ``smat``: S(k) at the k-points (nk, n, n); ``kpts``, ``kw``, ``nfold``: the k-points as
    :class:`KPointLayout`; the rest as :class:`HybridTerms`.  In the BvK supercell's real space
    (:meth:`KPointLayout.to_bvk`) the convolution C(X)(k) = 1/N sum_q X(q) o G(k - q) is the
    product X(n) o g(n), g(n)_AB = sum_g gamma_c(|R'_B + g - R'_A|) over the translations g of
    DFTB+'s parallelepiped (:func:`lattice_translations`) whose cell n_g + T_A - T_B is n mod N,
    R' = R - T.cell the atoms as DFTB+ folds them (:func:`dftbp_fold_shifts`): the same sums as
    DFTB+'s, which for the untruncated kernel depend on that folding (B19).  Defaults as DFTB+:
    Rc :func:`dftbp_coulomb_cutoff` of the folding, Rg = 2 Rc."""

    pauli = False
    _periodic = True

    def __init__(self, spec: Hybrid, params, u_atom, pos, orb_atom, cell, pairs, smat, q0_orb,
                 sk_cutoff, kpts, kw, nfold):
        self.spec = spec
        self.kind, self.omega, self.alpha, self.beta = params
        cell = np.asarray(cell, float)
        pos = np.asarray(pos, float)
        nat = len(pos)
        self.cell, self.u = cell, np.asarray(u_atom, float)
        self.orb_atom = np.asarray(orb_atom)
        self.offsets = np.concatenate([[0], np.cumsum(np.bincount(self.orb_atom,
                                                                  minlength=nat))])
        self.n = len(self.orb_atom)
        cam_cutoff = sk_cutoff - spec.cutoff_reduction
        if cam_cutoff < 0.0:
            raise ValueError("Screening cutoff for hybrid xc-functional neighbours too short "
                             "(as DFTB+)")
        self._cam_cutoff = cam_cutoff
        if spec.coulomb == "minimum_image":      # M6h-3c: no cutoffs (DFTB+ still computes the
            self.rc = self.rg = None             # default it does not use, and stops on B3)
        else:
            rc = (spec.coulomb_cutoff if spec.coulomb_cutoff is not None
                  else dftbp_coulomb_cutoff(cell, nfold))
            if rc - DIST_FUDGE <= 0.0:              # DFTB+ checks it whatever the kernel
                raise ValueError("The beginning of the damping region of the truncated Coulomb "
                                 "kernel must be at a positive radius (as DFTB+; the Coulomb "
                                 f"cutoff is {rc:.4f} Bohr)")
            self.rc = rc
            self.rg = spec.gsum_cutoff if spec.gsum_cutoff is not None else 2.0 * rc
        self.shifts = dftbp_fold_shifts(pos, cell)
        self.layout = KPointLayout(kpts, kw, nfold, self.orb_atom, spec.screening, self.shifts)
        self.pos_folded = pos - self.shifts @ cell
        if spec.coulomb == "minimum_image":       # M6h-3c: the supercell's Wigner-Seitz cell
            self.ws = wigner_seitz_vectors(nfold, cell, spec.wigner_seitz_reduction)
            self.ncoef = self.ws
            self.gvec = self.ws @ cell
        else:
            self.gvec = lattice_translations(cell, self.rg)
            self.ncoef = np.rint(self.gvec @ np.linalg.inv(cell)).astype(int)
        self.threshold = spec.neighbour_threshold if spec.screening == "neighbour" else None
        self._prev = self._hprev = None
        self._images = self._image_list()
        lay = self.layout
        g = np.zeros(lay.ncell * nat * nat)
        for (a, b, gi, d, gam, _, _) in self._images:
            flat = (lay.cell_index(self.ncoef[gi] + self.shifts[a] - self.shifts[b]) * nat
                    + a) * nat + b
            g += np.bincount(flat, weights=gam, minlength=len(g))
        self.g_bvk = g.reshape(lay.ncell, nat, nat)
        self.G = self.g_bvk[:, self.orb_atom][:, :, self.orb_atom]           # (N, n, n)
        self.S = np.asarray(smat)
        self._pairs = pairs
        if spec.screening == "neighbour":       # S(k) of the pairs within the CAM cutoff
            self._keep = (pairs.dist < cam_cutoff)[pairs.pair_of]
            self.S_x = fold_all(pairs, pairs.s * self._keep, self.n, lay.kpts, np.ones(self.n))
        else:
            self._keep = None
            self.S_x = self.S
        self.q0_orb = np.asarray(q0_orb, float)
        self.s_diag = np.array([np.diag(s).real for s in self.S])              # (nk, n)

    def _gamma(self, ua, ub, d, derivative: bool = False):
        if self.spec.coulomb in ("full", "minimum_image"):
            return (dgamma_cam if derivative else gamma_cam)(ua, ub, d, self.omega, self.alpha,
                                                             self.beta)
        return coulomb_gamma(ua, ub, d, self.omega, self.alpha, self.beta, self.rc,
                             self.spec.coulomb, derivative=derivative)

    def _image_list(self) -> list:
        """The terms of the lattice sums per pair of Hubbard values: atoms a, b, translation
        index, distance |R'_A - R'_B - g| and gamma, those within Rc only for a truncated kernel
        (gamma and its derivative vanish beyond), in blocks of at most IMAGE_CHUNK candidates;
        MinimumImage: the Wigner-Seitz vectors, MatrixBased (getCamGammaFourier over them)."""
        values, inv = np.unique(self.u, return_inverse=True)
        groups = [np.flatnonzero(inv == k) for k in range(len(values))]
        full = self.spec.coulomb in ("full", "minimum_image")
        out = []
        for ia in groups:
            for ib in groups:
                step = max(1, IMAGE_CHUNK // (len(ia) * len(ib)))
                parts = []
                for g0 in range(0, len(self.gvec), step):
                    v = (self.pos_folded[ia][:, None, None, :]
                         - self.pos_folded[ib][None, :, None, :]
                         - self.gvec[None, None, g0:g0 + step, :])
                    d = np.sqrt((v * v).sum(-1))
                    i, j, k = np.nonzero(np.ones(d.shape, bool) if full else d < self.rc)
                    parts.append((ia[i], ib[j], g0 + k, d[i, j, k]))
                a, b, gi, d = (np.concatenate([p[x] for p in parts]) for x in range(4))
                ua, ub = self.u[ia[0]], self.u[ib[0]]
                out.append((a, b, gi, d, self._gamma(ua, ub, d), ua, ub))
        return out

    # -- densities ------------------------------------------------------------------------
    @property
    def packed_size(self) -> int:
        return self.layout.packed_size

    def reference(self, nspin: int) -> np.ndarray:
        """P0(k) = diag(q0 / S_mumu(k)) of one spin channel, half each with spin (DFTB+)."""
        out = np.zeros((self.layout.nk, self.n, self.n), complex)
        i = np.arange(self.n)
        out[:, i, i] = (0.5 if nspin == 2 else 1.0) * self.q0_orb / self.s_diag
        return out

    def pack(self, dps) -> np.ndarray:
        return self.layout.pack(dps)

    def unpack(self, vec, nspin: int) -> list:
        return self.layout.unpack(vec, nspin)

    def error(self, x_out, x_in) -> float:
        return self.layout.error(x_out, x_in)

    def reset(self, restart: bool = True):
        """Forget the screened builds' history (``neighbour_threshold``, M6h-3c): the next
        :meth:`fock` builds all of DeltaP, as DFTB+ at a new geometry."""
        self._prev = self._hprev = None

    # -- Fock matrices and energy -----------------------------------------------------------
    def _conv(self, x) -> np.ndarray:
        """1/N sum_q X(q) o G(k - q) at the k-points: X(n) o g(n) in the BvK supercell."""
        lay = self.layout
        return lay.from_bvk(lay.to_bvk(x) * self.G)

    def fock(self, dps) -> list:
        """H_x[DeltaP](k) of every spin channel (zero for a zero DeltaP, as DFTB+'s first
        iteration); with ``neighbour_threshold`` DFTB+'s screened increments
        (:meth:`_screened_fock`, M6h-3c)."""
        if self.threshold is not None:
            return self._screened_fock(dps)
        c = -0.125 if len(dps) == 1 else -0.25
        s = self.S_x
        out = []
        for d in dps:
            if not np.any(d):
                out.append(np.zeros((self.layout.nk, self.n, self.n), complex))
                continue
            sd, ds = s @ d, d @ s
            h = self._conv(sd @ s) + self._conv(sd) @ s + s @ self._conv(ds) + s @ self._conv(d) @ s
            out.append(c * h)
        return out

    def response_fock(self, dps) -> list:
        """H_x(k) of first-order densities (M6h-4b): every spin channel's (n_rhs, nk, n, n) at the
        k-points, the right-hand sides in one BvK convolution; with ``neighbour_threshold`` the
        screening's limit (the S of the pairs within the CAM cutoff, no screened increments),
        the history of the ground state's builds untouched."""
        c = -0.125 if len(dps) == 1 else -0.25
        s = self.S_x
        out = []
        for d in dps:
            sd, ds = s @ d, d @ s
            h = (self._conv_rhs(sd @ s) + self._conv_rhs(sd) @ s + s @ self._conv_rhs(ds)
                 + s @ self._conv_rhs(d) @ s)
            out.append(c * h)
        return out

    def _conv_rhs(self, x) -> np.ndarray:
        """:meth:`_conv` of (n_rhs, nk, n, n) matrices: the k-points first, the rest as columns."""
        lay, nr = self.layout, x.shape[0]
        xk = np.moveaxis(x, 1, 0).reshape(lay.nk, -1)
        b = lay._cos_w @ xk.real + lay._sin_w @ xk.imag                   # (N, n_rhs n n)
        b = (b.reshape(lay.ncell, nr, self.n, self.n) * self.G[:, None]).reshape(lay.ncell, -1)
        y = (lay._cos @ b + 1j * (lay._sin @ b)).reshape(lay.nk, nr, self.n, self.n)
        return np.moveaxis(y, 0, 1)

    # -- DFTB+'s Threshold of the neighbour screening (M6h-3c) ------------------------------------
    def screening_groups(self, pmax: float) -> list:
        """DFTB+'s kept quadruples (getFourLoopCompositeIndex): pair entries p = (M, A@h) and
        q = (N, B@l) of the pairs within the CAM cutoff and the on-site, kept when pmax est_p
        est_q >= Threshold (est = max |S block|, calculateOverlapEstimates; 1 on-site), as groups
        of p that keep the same q: [(p pairs, p on-site, q pairs, q on-site)], boolean masks of
        the pairs and flags."""
        if getattr(self, "_est", None) is None:
            p = self._pairs
            est = np.zeros(len(p.dist))
            np.maximum.at(est, p.pair_of, np.abs(p.s))
            self._est = est
        est, keep = self._est, self._pairs.dist < self._cam_cutoff
        tau = self.threshold / pmax
        levels = np.unique(np.concatenate([est[keep], [1.0]]))[::-1]
        with np.errstate(divide="ignore"):
            theta = tau / np.where(keep & (est > 0.0), est, np.inf)
        k_pair = np.searchsorted(-levels, -theta, side="right")      # q levels kept by each p
        k_on = int(np.searchsorted(-levels, -tau, side="right"))
        out = []
        for kq in sorted(set(k_pair[keep].tolist()) | {k_on}):
            if kq:
                lev = levels[kq - 1]
                out.append((keep & (k_pair == kq), kq == k_on, keep & (est >= lev), 1.0 >= lev))
        return out

    def _overlap_of(self, pair_mask, onsite: bool, kpts=None) -> np.ndarray:
        """S(k) of the pairs in ``pair_mask`` (and the on-site identity) at ``kpts`` (default:
        the SCC's)."""
        p = self._pairs
        k = self.layout.kpts if kpts is None else np.atleast_2d(kpts)
        return fold_all(p, p.s * pair_mask[p.pair_of], self.n, k,
                        np.ones(self.n) if onsite else np.zeros(self.n))

    def _group_operators(self, groups) -> list:
        return [(self._overlap_of(pp, po), self._overlap_of(qp, qo)) for pp, po, qp, qo in groups]

    def _increment(self, x, sets) -> np.ndarray:
        """The four terms of the BvK array ``x`` over the kept quadruples, at the SCC's k-points:
        sum over the (S_P, S_Q) of C(S_P D S_Q) + C(S_P D) S_Q + S_P C(D S_Q) + S_P C(D) S_Q."""
        d = self.layout.from_bvk(x)
        h = np.zeros((self.layout.nk, self.n, self.n), complex)
        for sp, sq in sets:
            spd, dsq = sp @ d, d @ sq
            h = h + (self._conv(spd @ sq) + self._conv(spd) @ sq + sp @ self._conv(dsq)
                     + sp @ self._conv(d) @ sq)
        return h

    def _screened_fock(self, dps) -> list:
        """DFTB+'s NeighbourBased exchange with a Threshold (addCamHamiltonianNeighbour_kpts_ct
        and _mic): H = H_prev + H[dDeltaP] over the quadruples :meth:`screening_groups` keeps
        for pmax = max |dDeltaP| (BvK arrays, all spin channels), dDeltaP = DeltaP - DeltaP_prev
        (all of DeltaP after :meth:`reset`); an unchanged DeltaP returns zero, as DFTB+ (B7)."""
        c = -0.125 if len(dps) == 1 else -0.25
        lay, n = self.layout, self.n
        bv = [lay.to_bvk(d) for d in dps]
        inc = bv if self._prev is None else [b - q for b, q in zip(bv, self._prev)]
        if self._hprev is None:
            self._hprev = [np.zeros((lay.nk, n, n), complex) for _ in dps]
        self._prev = bv
        pmax = max(float(np.abs(x).max()) for x in inc)
        if pmax < 1e-16:
            return [np.zeros((lay.nk, n, n), complex) for _ in dps]
        sets = self._group_operators(self.screening_groups(pmax))
        for i, x in enumerate(inc):
            self._hprev[i] = self._hprev[i] + c * self._increment(x, sets)
        return [np.array(h) for h in self._hprev]

    def fock_at(self, kpts, dps) -> list:
        """H_x[DeltaP] at any k-points (band structures; DFTB+'s band run from the SCC's
        DeltaP): C(X)(k) = sum_g X(n_g mod N) o gamma(g) exp(2 pi i k.n_g) over the lattice
        sums' terms, the same as :meth:`fock` on the grid; with ``neighbour_threshold`` one
        screened build of all of DeltaP (DFTB+'s band run, not the SCC's accumulated one)."""
        kpts = np.atleast_2d(np.asarray(kpts, float))
        lay, oa, nat = self.layout, self.orb_atom, len(self.u)
        c = -0.125 if len(dps) == 1 else -0.25
        s_band = self.overlap_at(kpts)
        if self.threshold is None:
            sets = [(self.S_x, self.S_x, s_band, s_band)]
        else:
            pmax = max(float(np.abs(lay.to_bvk(d)).max()) for d in dps)
            groups = self.screening_groups(pmax) if pmax >= 1e-16 else []
            sets = [(self._overlap_of(pp, po), self._overlap_of(qp, qo),
                     self._overlap_of(pp, po, kpts), self._overlap_of(qp, qo, kpts))
                    for pp, po, qp, qo in groups]
        parts = []
        for d in dps:
            per = []
            for sp, sq, _, _ in sets:
                spd, dsq = sp @ d, d @ sq
                per.append([lay.to_bvk(x) for x in (spd @ sq, spd, dsq, d)])
            parts.append(per)
        out = [np.zeros((len(kpts), self.n, self.n), complex) for _ in dps]
        for i, k in enumerate(kpts):
            h = np.zeros(lay.ncell * nat * nat, complex)
            for (a, b, gi, d, gam, _, _) in self._images:
                nvec = self.ncoef[gi] + self.shifts[a] - self.shifts[b]
                flat = (lay.cell_index(nvec) * nat + a) * nat + b
                w = gam * np.exp(2j * np.pi * (nvec @ k))
                h += (np.bincount(flat, weights=w.real, minlength=len(h))
                      + 1j * np.bincount(flat, weights=w.imag, minlength=len(h)))
            h = h.reshape(lay.ncell, nat, nat)[:, oa][:, :, oa]
            for j, per in enumerate(parts):
                tot = None
                for (b1, b2, b3, b4), (_, _, spb, sqb) in zip(per, sets):
                    c1, c2, c3, c4 = ((x * h).sum(0) for x in (b1, b2, b3, b4))
                    t = c1 + c2 @ sqb[i] + spb[i] @ c3 + spb[i] @ c4 @ sqb[i]
                    tot = t if tot is None else tot + t
                if tot is not None:
                    out[j][i] = c * tot
        return out

    def overlap_at(self, kpts) -> np.ndarray:
        """The overlap the exchange takes at any k-points (NeighbourBased: of the pairs within
        the CAM cutoff)."""
        p = self._pairs
        values = p.s if self._keep is None else p.s * self._keep
        return fold_all(p, values, self.n, np.atleast_2d(kpts), np.ones(self.n))

    def energy(self, focks, dps) -> float:
        """E_Fock = 1/2 sum_spin sum_k w_k Re sum H_x(k) o conj(DeltaP(k)); DFTB+ takes
        DeltaP_out (B4: DeltaP_in at the Gamma point), as the SCC here."""
        kw = self.layout.kw
        return float(sum(0.5 * np.einsum("k,kij->", kw, h * d.conj()).real
                         for h, d in zip(focks, dps)))

    # -- forces ---------------------------------------------------------------------------
    def pair_coefficient(self, dps, pairs) -> np.ndarray:
        """dE_x/dS at every pair element (T, mu, nu): sum_k w_k Re[X(k)_numu exp(2 pi i k.T)],
        the BvK matrix of X(k) = c [C(D) S D + C(D S) D + D S C(D) + D C(S D)] at cell -T,
        summed over the spin channels (the Gamma-point formula in k; NeighbourBased with the
        reduced overlap and the pairs within the CAM cutoff, as DFTB+'s gradient)."""
        c = -0.125 if len(dps) == 1 else -0.25
        s, lay = self.S_x, self.layout
        x = np.zeros((lay.nk, self.n, self.n), complex)
        for d in dps:
            ds, sd = d @ s, s @ d
            cd = self._conv(d)
            x += c * (cd @ sd + self._conv(ds) @ d + ds @ cd + d @ self._conv(sd))
        xb = lay.to_bvk(x)
        coef = xb[lay.cell_index(-pairs.shift)[pairs.pair_of], pairs.cols, pairs.rows]
        if self._keep is not None:
            coef = coef * (pairs.dist < self._cam_cutoff)[pairs.pair_of]
        return coef

    def gamma_gradient(self, dps) -> np.ndarray:
        """dE_x/dR_A through gamma: dE/dg(n) = c [D(n) o (S D S)(n) + (D S)(n) o (S D)(n)] (BvK
        matrices) summed over each atom pair's orbitals, times gamma_c'(d) (R'_A - R'_B - g) / d
        of every term of the lattice sums, which enters g(n) of (A, B) and of (B, A)."""
        c = -0.125 if len(dps) == 1 else -0.25
        s, lay = self.S_x, self.layout
        w = np.zeros((lay.ncell, self.n, self.n))
        for d in dps:
            ds, sd = d @ s, s @ d
            w += c * (lay.to_bvk(d) * lay.to_bvk(sd @ s) + lay.to_bvk(ds) * lay.to_bvk(sd))
        starts = self.offsets[:-1]
        w_at = np.add.reduceat(np.add.reduceat(w, starts, axis=1), starts, axis=2)
        nat = len(self.u)
        out = np.zeros((nat, 3))
        for (a, b, gi, d, _, ua, ub) in self._images:
            nvec = self.ncoef[gi] + self.shifts[a] - self.shifts[b]
            wsym = (w_at[lay.cell_index(nvec), a, b] + w_at[lay.cell_index(-nvec), b, a])
            dg = self._gamma(ua, ub, d, derivative=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where((d > 0.0) & (a != b), wsym * dg / d, 0.0)
            v = self.pos_folded[a] - self.pos_folded[b] - self.gvec[gi]
            for x in range(3):
                out[:, x] += np.bincount(a, weights=f * v[:, x], minlength=nat)
        return out


class NeighbourMinimumImageTerms(KPointHybridTerms):
    """NeighbourBased exchange with the MinimumImage Coulomb matrix at k-points (DFTB+
    addCamHamiltonianNeighbour_kpts_mic; milestone M6h-3c).  Every path M@0 -> A@h -> B@h+g ->
    N@g+h-l of the neighbour lists (the pairs within the CAM cutoff and the on-site), g a
    Wigner-Seitz vector (:func:`wigner_seitz_vectors`), adds S_MA DeltaP_AB S_BN times
    gamma_MN + gamma_MB + gamma_AN + gamma_AB, the untruncated kernel at the path's own distances;
    only the A-B offset of the density is restricted to the Wigner-Seitz cell.  The sums regroup
    into convolutions of explicit-cell kernels in the frame of the folded atoms: with S the
    overlap kernel, P the density at the WS vectors and g gamma at every cell,
    (S*P*S) o g + ((S*P) o g)*S + S*((P*S) o g) + S*(P o g)*S, evaluated by FFT on a grid wide
    enough not to alias and folded into the supercell.  The operator is not self-adjoint (B22):
    its exchange matrix is not the derivative of its energy, so there are no forces (DFTB+ has
    none either); DFTB+'s Threshold as in :class:`KPointHybridTerms`."""

    def _image_list(self) -> list:
        return []                                  # gamma at the explicit cells instead

    def _kernels(self):
        """The grid (L, the explicit cells m, their BvK cells), gamma(m) and the overlap
        kernel's transform, once."""
        if getattr(self, "_g_l", None) is not None:
            return
        p, t, nat, n = self._pairs, self.shifts, len(self.u), self.n
        keep = p.dist < self._cam_cutoff
        self._hcell = p.shift + t[p.j] - t[p.i]               # pair cells of the folded atoms
        hmax = np.abs(self._hcell[keep]).max(0) if keep.any() else np.zeros(3, int)
        ext = 2 * hmax + np.abs(self.ws).max(0)
        self._grid = tuple(int(x) for x in 2 * ext + 1)
        m = np.array(np.meshgrid(*[np.arange(x) for x in self._grid],
                                 indexing="ij")).reshape(3, -1).T
        m = np.where(m > ext, m - 2 * ext - 1, m)
        self._m, self._cell_of_m = m, self.layout.cell_index(m)
        r = self.pos_folded
        v = r[None, None, :, :] + (m @ self.cell)[:, None, None, :] - r[None, :, None, :]
        dist = np.sqrt((v * v).sum(-1))                        # (m, X, Y): |R'_Y + m - R'_X|
        g = np.zeros_like(dist)
        for a in range(nat):
            for b in range(nat):
                g[:, a, b] = self._gamma(self.u[a], self.u[b], dist[:, a, b])
        oa = self.orb_atom
        self._g_l = g[:, oa][:, :, oa].reshape(self._grid + (n, n))
        self._s_full = self._s_kernel(keep, True)

    def _s_kernel(self, pair_mask, onsite: bool) -> np.ndarray:
        """FFT of the overlap kernel of the pairs in ``pair_mask`` (and the on-site identity)."""
        p, n = self._pairs, self.n
        sl = np.zeros(self._grid + (n, n))
        el = pair_mask[p.pair_of]
        hc = np.mod(self._hcell[p.pair_of[el]], self._grid)
        np.add.at(sl, (hc[:, 0], hc[:, 1], hc[:, 2], p.rows[el], p.cols[el]), p.s[el])
        if onsite:
            sl[0, 0, 0][np.diag_indices(n)] += 1.0
        return np.fft.fftn(sl, axes=(0, 1, 2))

    def _explicit(self, x, sets) -> np.ndarray:
        """The four terms (no prefactor) at the explicit cells, (cells, n, n), of the BvK array
        ``x`` over the (S_P, S_Q) kernel pairs."""
        lay, n = self.layout, self.n
        pf = lay._bvk_shifted(x, -1)                          # the folded atoms' frame
        pl = np.zeros(self._grid + (n, n))
        wc = np.mod(self.ws, self._grid)
        pl[wc[:, 0], wc[:, 1], wc[:, 2]] = pf[lay.cell_index(self.ws)]
        fft, ifft, g = np.fft.fftn, np.fft.ifftn, self._g_l
        p_hat, pg_hat = fft(pl, axes=(0, 1, 2)), fft(pl * g, axes=(0, 1, 2))
        y = np.zeros(self._grid + (n, n))
        for sp, sq in sets:
            spp = ifft(sp @ p_hat, axes=(0, 1, 2)).real
            pps = ifft(p_hat @ sq, axes=(0, 1, 2)).real
            y += ifft(fft(spp, axes=(0, 1, 2)) @ sq, axes=(0, 1, 2)).real * g
            y += ifft(fft(spp * g, axes=(0, 1, 2)) @ sq, axes=(0, 1, 2)).real
            y += ifft(sp @ fft(pps * g, axes=(0, 1, 2)), axes=(0, 1, 2)).real
            y += ifft(sp @ pg_hat @ sq, axes=(0, 1, 2)).real
        return y.reshape(-1, n, n)

    def _on_grid(self, y) -> np.ndarray:
        """Explicit-cell blocks (folded frame) -> H(k) at the SCC's k-points."""
        lay = self.layout
        yf = np.zeros((lay.ncell, self.n, self.n))
        np.add.at(yf, self._cell_of_m, y)
        return lay.from_bvk(lay._bvk_shifted(yf, 1))

    def _group_operators(self, groups) -> list:
        return [(self._s_kernel(pp, po), self._s_kernel(qp, qo)) for pp, po, qp, qo in groups]

    def _increment(self, x, sets) -> np.ndarray:
        return self._on_grid(self._explicit(x, sets))

    def fock(self, dps) -> list:
        self._kernels()
        if self.threshold is not None:
            return self._screened_fock(dps)
        c = -0.125 if len(dps) == 1 else -0.25
        lay = self.layout
        out = []
        for d in dps:
            if not np.any(d):
                out.append(np.zeros((lay.nk, self.n, self.n), complex))
                continue
            out.append(c * self._increment(lay.to_bvk(d), [(self._s_full, self._s_full)]))
        return out

    def response_fock(self, dps) -> list:
        raise ValueError(_MIC_NEIGHBOUR_RESPONSE)

    def fock_at(self, kpts, dps) -> list:
        """H_x at any k-points: the explicit cells' blocks with their own phases (DFTB+'s band
        run builds them at its k-points); with a Threshold one screened build of DeltaP."""
        self._kernels()
        kpts = np.atleast_2d(np.asarray(kpts, float))
        lay = self.layout
        c = -0.125 if len(dps) == 1 else -0.25
        if self.threshold is None:
            sets = [(self._s_full, self._s_full)]
        else:
            pmax = max(float(np.abs(lay.to_bvk(d)).max()) for d in dps)
            sets = (self._group_operators(self.screening_groups(pmax)) if pmax >= 1e-16 else [])
        phase = np.exp(2j * np.pi * (kpts @ self._m.T))                    # (nk, cells)
        dk = np.exp(2j * np.pi * (kpts @ self.shifts[self.orb_atom].T))     # gauge: D(k)
        out = []
        for d in dps:
            y = self._explicit(lay.to_bvk(d), sets).reshape(len(self._m), -1)
            hk = (phase @ y).reshape(len(kpts), self.n, self.n)
            out.append(c * dk[:, :, None] * hk * dk.conj()[:, None, :])
        return out

    def pair_coefficient(self, dps, pairs):
        raise ValueError(_MIC_NEIGHBOUR_FORCES)

    def gamma_gradient(self, dps):
        raise ValueError(_MIC_NEIGHBOUR_FORCES)


_MIC_NEIGHBOUR_RESPONSE = (
    "hybrid functionals: no response analyses at k-points with the MinimumImage NeighbourBased "
    "exchange (M6h-4b): that operator is not self-adjoint (B22), so the response equation is not "
    "symmetric (as its forces, which neither DFTB+ nor pydftb has)")

_MIC_NEIGHBOUR_FORCES = (
    "Hybrid functionals don't yet support gradient calculations for periodic systems beyond the "
    "Gamma point for CoulombMatrix setting 'MinimumImage'. (as DFTB+ for the NeighbourBased "
    "screening; pydftb has none either: that operator is not self-adjoint, so its exchange matrix "
    "is not the derivative of the energy (B22) and forces would need a response calculation; the "
    "MatrixBased MinimumImage has forces)")
