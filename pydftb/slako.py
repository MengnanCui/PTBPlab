"""Slater-Koster two-centre transformation for s, p and d shells.

Instead of hard-coding the 100+ entries of the Slater-Koster table we rotate
the (diagonal) bond-frame matrix with real-spherical-harmonic rotation
matrices.  This is exact, works for any (l1, l2) <= 2 and is vectorised
over all atom pairs:

    B_lab(R) = D^{l1}(Rot) . M_bond(|R|) . D^{l2}(Rot)^T,    Rot z = R/|R|

Orbital order (identical to DFTB+):
    s ;  p: y, z, x ;  d: xy, yz, z2, xz, x2-y2
All operations are complex-analytic, so derivatives can be taken with the
complex-step trick (machine-precision, no truncation error).
"""
from __future__ import annotations

import numpy as np

from .skf import SK_COLUMN

# |m| of each real harmonic in DFTB+ order
M_ABS = {0: np.array([0]), 1: np.array([1, 0, 1]), 2: np.array([2, 1, 0, 1, 2])}

# Cartesian index of each p orbital (y, z, x)
_P_CART = np.array([1, 2, 0])


def _d_tensors() -> np.ndarray:
    """Frobenius-orthonormal traceless symmetric tensors for d orbitals."""
    q = np.zeros((5, 3, 3))
    s2 = 1.0 / np.sqrt(2.0)
    q[0, 0, 1] = q[0, 1, 0] = s2                     # xy
    q[1, 1, 2] = q[1, 2, 1] = s2                     # yz
    q[2] = np.diag([-1.0, -1.0, 2.0]) / np.sqrt(6.0)  # z2
    q[3, 0, 2] = q[3, 2, 0] = s2                     # xz
    q[4] = np.diag([1.0, -1.0, 0.0]) * s2            # x2-y2
    return q


_QD = _d_tensors()


def bond_frames(vec: np.ndarray) -> np.ndarray:
    """Rotation matrices (n, 3, 3) whose 3rd column is the unit bond vector.

    Works for complex input (complex-step differentiation); branch choices
    are made on the real part only.
    """
    r = np.sqrt((vec * vec).sum(-1))
    ez = vec / r[:, None]
    helper = np.zeros_like(vec)
    use_x = np.abs(ez[:, 0].real) < 0.9
    helper[use_x, 0] = 1.0
    helper[~use_x, 1] = 1.0
    ex = np.cross(helper, ez)
    ex = ex / np.sqrt((ex * ex).sum(-1))[:, None]
    ey = np.cross(ez, ex)
    return np.stack([ex, ey, ez], axis=-1)


def rotation_blocks(rot: np.ndarray, lmax: int) -> list[np.ndarray]:
    """D^l matrices (n, 2l+1, 2l+1) for l = 0..lmax."""
    n = rot.shape[0]
    out = [np.ones((n, 1, 1), dtype=rot.dtype)]
    if lmax >= 1:
        out.append(rot[:, _P_CART][:, :, _P_CART])
    if lmax >= 2:
        # D_ab = <Q_a, Rot Q_b Rot^T>_F
        rq = np.einsum("nij,bjk,nlk->nbil", rot, _QD, rot, optimize=True)
        out.append(np.einsum("aij,nbij->nab", _QD, rq, optimize=True))
    return out


def bond_matrix(integrals: np.ndarray, l1: int, l2: int, kind: int) -> np.ndarray:
    """Bond-frame (z-aligned) block (n, 2l1+1, 2l2+1) for l1 <= l2.

    ``integrals`` has the 20 SKF columns; ``kind`` = 0 for H, 10 for S.
    """
    n = integrals.shape[0]
    blk = np.zeros((n, 2 * l1 + 1, 2 * l2 + 1), dtype=integrals.dtype)
    for i in range(2 * l1 + 1):
        m = i - l1
        j = m + l2
        blk[:, i, j] = integrals[:, kind + SK_COLUMN[(l1, l2, abs(m))]]
    return blk


def sk_block(vec: np.ndarray, integ_ab: np.ndarray, integ_ba: np.ndarray,
             shells_a: list[int], shells_b: list[int], kind: int,
             dmats: list[np.ndarray]) -> np.ndarray:
    """Full orbital block <a|O|b> (n, norb_a, norb_b) for pairs a->b.

    ``integ_ab`` are integrals from the A-B file, ``integ_ba`` from B-A.
    For l_a > l_b we use  <a|O|b>(R) = <b|O|a>(-R)^T ; reflecting R flips
    the parity, giving the factor (-1)^(l_a + l_b).
    """
    na = sum(2 * l + 1 for l in shells_a)
    nb = sum(2 * l + 1 for l in shells_b)
    out = np.zeros((vec.shape[0], na, nb), dtype=integ_ab.dtype)
    ia = 0
    for la in shells_a:
        ib = 0
        for lb in shells_b:
            if la <= lb:
                m = bond_matrix(integ_ab, la, lb, kind)
                blk = np.einsum("nij,njk,nlk->nil", dmats[la], m, dmats[lb], optimize=True)
            else:
                m = bond_matrix(integ_ba, lb, la, kind)
                blk = np.einsum("nij,njk,nlk->nil", dmats[lb], m, dmats[la], optimize=True)
                blk = np.swapaxes(blk, 1, 2) * (-1.0) ** (la + lb)
            out[:, ia:ia + 2 * la + 1, ib:ib + 2 * lb + 1] = blk
            ib += 2 * lb + 1
        ia += 2 * la + 1
    return out
