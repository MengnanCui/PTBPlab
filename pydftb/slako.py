"""Slater-Koster two-centre transformation for s, p, d and f shells.

Instead of hard-coding the 100+ entries of the Slater-Koster table (DFTB+'s sk.F90)
we rotate the (diagonal) bond-frame matrix with real-spherical-harmonic rotation
matrices.  This is exact, works for any (l1, l2) <= 3 and is vectorised over all
atom pairs:

    B_lab(R) = D^{l1}(Rot) . M_bond(|R|) . D^{l2}(Rot)^T,    Rot z = R/|R|

A real harmonic of degree l is a traceless symmetric tensor T of rank l
(e.g. xyz <-> T_xyz); with the tensors Frobenius-normalised,
D^l_ab = <T_a, Rot.T_b>, Rot acting on every index.

Orbital order (identical to DFTB+):
    s ;  p: y, z, x ;  d: xy, yz, z2, xz, x2-y2 ;
    f: y(3x2-y2), xyz, yz2, z3, xz2, z(x2-y2), x(x2-3y2)
All operations are complex-analytic, so derivatives can be taken with the
complex-step trick (machine-precision, no truncation error).  The same code runs on torch
tensors (``ops.namespace``; milestone M4b), where autograd gives the derivatives.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .backend import ops
from .skf import SK_COLUMN

# |m| of each real harmonic in DFTB+ order
M_ABS = {0: np.array([0]), 1: np.array([1, 0, 1]), 2: np.array([2, 1, 0, 1, 2]),
         3: np.array([3, 2, 1, 0, 1, 2, 3])}

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


# f orbitals as cubic polynomials {(a, b, c): coefficient of x^a y^b z^c}, signs as DFTB+
_F_POLYNOMIALS = [
    {(2, 1, 0): 3.0, (0, 3, 0): -1.0},                      # y(3x2 - y2)
    {(1, 1, 1): 1.0},                                       # xyz
    {(0, 1, 2): 4.0, (2, 1, 0): -1.0, (0, 3, 0): -1.0},     # y(5z2 - r2)
    {(0, 0, 3): 2.0, (2, 0, 1): -3.0, (0, 2, 1): -3.0},     # z(5z2 - 3r2)
    {(1, 0, 2): 4.0, (3, 0, 0): -1.0, (1, 2, 0): -1.0},     # x(5z2 - r2)
    {(2, 0, 1): 1.0, (0, 2, 1): -1.0},                      # z(x2 - y2)
    {(3, 0, 0): 1.0, (1, 2, 0): -3.0},                      # x(x2 - 3y2)
]


def _f_tensors() -> np.ndarray:
    """Frobenius-orthonormal traceless symmetric rank-3 tensors for f orbitals."""
    t = np.zeros((7, 3, 3, 3))
    for a, poly in enumerate(_F_POLYNOMIALS):
        for idx in itertools.product(range(3), repeat=3):
            power = tuple(idx.count(k) for k in range(3))
            if power in poly:      # spread the coefficient over the index permutations
                t[(a,) + idx] = poly[power] * math.prod(map(math.factorial, power)) / 6.0
        t[a] /= np.sqrt((t[a] ** 2).sum())
    return t


_QD9 = _d_tensors().reshape(5, 9)
_QF27 = _f_tensors().reshape(7, 27)


def bond_frames(vec: np.ndarray) -> np.ndarray:
    """Rotation matrices (n, 3, 3) whose 3rd column is the unit bond vector.

    Works for complex input (complex-step differentiation); branch choices
    are made on the real part only.
    """
    xp = ops.namespace(vec)
    r = xp.sqrt((vec * vec).sum(-1))
    ez = vec / r[:, None]
    helper = xp.zeros_like(vec)
    use_x = xp.abs(ez[:, 0].real) < 0.9
    helper[use_x, 0] = 1.0
    helper[~use_x, 1] = 1.0
    ex = xp.cross(helper, ez)
    ex = ex / xp.sqrt((ex * ex).sum(-1))[:, None]
    ey = xp.cross(ez, ex)
    return xp.stack([ex, ey, ez], axis=-1)


def rotation_blocks(rot: np.ndarray, lmax: int) -> list[np.ndarray]:
    """D^l matrices (n, 2l+1, 2l+1) for l = 0..lmax."""
    n = rot.shape[0]
    out = [ops.ones((n, 1, 1), like=rot)]
    if lmax >= 1:
        out.append(rot[:, _P_CART][:, :, _P_CART])
    if lmax >= 2:
        # D_ab = <Q_a, Rot Q_b Rot^T>_F = Q_a . (Rot (x) Rot) . Q_b over flattened indices
        rr = (rot[:, :, None, :, None] * rot[:, None, :, None, :]).reshape(n, 9, 9)
        q = ops.asarray(_QD9, like=rot)
        out.append(q @ rr @ q.T)
    if lmax >= 3:
        # D_ab = <T_a, Rot.T_b>_F, Rot acting on all three indices
        rrr = (rot[:, :, None, None, :, None, None] * rot[:, None, :, None, None, :, None]
               * rot[:, None, None, :, None, None, :]).reshape(n, 27, 27)
        q = ops.asarray(_QF27, like=rot)
        out.append(q @ rrr @ q.T)
    return out


def bond_matrix(integrals: np.ndarray, l1: int, l2: int, kind: int,
                colmap: np.ndarray | None = None) -> np.ndarray:
    """Bond-frame (z-aligned) block (n, 2l1+1, 2l2+1) for l1 <= l2.

    ``integrals`` has the 40 columns of :meth:`SKFile.evaluate` (``kind`` = 0 for H,
    ``N_INTEGRALS`` = 20 for S), or only a file's stored columns with its ``colmap``
    (integrals the file does not store are zero).
    """
    n = integrals.shape[0]
    blk = ops.zeros((n, 2 * l1 + 1, 2 * l2 + 1), like=integrals)
    for i in range(2 * l1 + 1):
        m = i - l1
        c = kind + SK_COLUMN[(l1, l2, abs(m))]
        if colmap is not None:
            c = colmap[c]
            if c < 0:
                continue
        blk[:, i, m + l2] = integrals[:, c]
    return blk


def sk_block(vec: np.ndarray, integ_ab: np.ndarray, integ_ba: np.ndarray,
             shells_a: list[int], shells_b: list[int], kind: int,
             dmats: list[np.ndarray], colmap_ab: np.ndarray | None = None,
             colmap_ba: np.ndarray | None = None) -> np.ndarray:
    """Full orbital block <a|O|b> (n, norb_a, norb_b) for pairs a->b.

    ``integ_ab`` are integrals from the A-B file, ``integ_ba`` from B-A (with their
    ``colmap`` if they hold only the stored columns).
    For l_a > l_b we use  <a|O|b>(R) = <b|O|a>(-R)^T ; reflecting R flips
    the parity, giving the factor (-1)^(l_a + l_b).
    """
    xp = ops.namespace(integ_ab)
    na = sum(2 * l + 1 for l in shells_a)
    nb = sum(2 * l + 1 for l in shells_b)
    out = ops.zeros((vec.shape[0], na, nb), like=integ_ab)
    ia = 0
    for la in shells_a:
        ib = 0
        for lb in shells_b:
            if la <= lb:
                m = bond_matrix(integ_ab, la, lb, kind, colmap_ab)
                blk = dmats[la] @ m @ xp.swapaxes(dmats[lb], 1, 2)
            else:
                m = bond_matrix(integ_ba, lb, la, kind, colmap_ba)
                blk = dmats[lb] @ m @ xp.swapaxes(dmats[la], 1, 2)
                blk = xp.swapaxes(blk, 1, 2) * (-1.0) ** (la + lb)
            out[:, ia:ia + 2 * la + 1, ib:ib + 2 * lb + 1] = blk
            ib += 2 * lb + 1
        ia += 2 * la + 1
    return out
