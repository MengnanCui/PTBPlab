"""Periodic neighbour list built on a k-d tree (O(N log N))."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .constants import MIN_NEIGH_DIST


class AtomsTooClose(ValueError):
    """Two atoms closer than ``MIN_NEIGH_DIST`` (0.01 Bohr), an atom and its own periodic
    image included: DFTB+ stops there (``periodic.F90:updateNeighbourList``, minNeighDist).
    :class:`pydftb.ase_calc.PyDFTB` reports it as ASE's ``CalculationFailed``."""


def image_range(cell: np.ndarray, pbc: np.ndarray, cutoff: float) -> np.ndarray:
    """Number of periodic images needed along each lattice vector."""
    n = np.zeros(3, dtype=int)
    if cell is None:
        return n
    inv = np.linalg.inv(cell)  # rows of cell are lattice vectors
    # distance between opposite faces = 1 / |reciprocal vector|
    heights = 1.0 / np.linalg.norm(inv, axis=0)
    for k in range(3):
        if pbc[k]:
            n[k] = int(np.ceil(cutoff / heights[k]))
    return n


def pair_vectors(pos: np.ndarray, cell: np.ndarray | None, i: np.ndarray, j: np.ndarray,
                 shift: np.ndarray) -> np.ndarray:
    """Bond vectors r_j + shift.cell - r_i for an integer pair list.

    Separating the (integer) topology from the geometry lets other backends
    and strain/autodiff derivatives recompute vectors without a new search.
    """
    vec = pos[j] - pos[i]
    if cell is not None:
        vec = vec + shift @ cell
    return vec


def neighbor_list(pos: np.ndarray, cell: np.ndarray | None, pbc, cutoff: float):
    """All ordered pairs (i, j, shift) with |r_j + shift.cell - r_i| < cutoff.

    Self-pairs (i == j, shift == 0) are excluded.  Any other pair closer than
    ``MIN_NEIGH_DIST`` raises :class:`AtomsTooClose`, as DFTB+ stops there; this includes
    an atom and its own periodic image.  Returns
    ``i, j, shifts (int, (n,3)), vec (n,3), dist (n,)``.
    """
    natom = len(pos)
    pbc = np.zeros(3, bool) if cell is None else np.asarray(pbc, bool)
    nimg = image_range(cell, pbc, cutoff)
    if cell is not None and natom:
        # atoms outside the unit cell (fractional spread >= 1) can need one more image
        inv = np.linalg.inv(cell)
        frac = pos @ inv
        reach = cutoff * np.linalg.norm(inv, axis=0) + frac.max(0) - frac.min(0)
        nimg = np.where(pbc, np.maximum(nimg, np.floor(reach).astype(int)), 0)
    rng = [np.arange(-n, n + 1) for n in nimg]
    shifts = np.array(np.meshgrid(*rng, indexing="ij")).reshape(3, -1).T
    if cell is not None:
        # wrap positions so images are compact; keep original for vectors
        tvec = shifts @ cell
    else:
        tvec = np.zeros((1, 3))
        shifts = np.zeros((1, 3), dtype=int)
    img_pos = (pos[None, :, :] + tvec[:, None, :]).reshape(-1, 3)
    tree_img = cKDTree(img_pos)
    tree_c = cKDTree(pos)
    sdm = tree_c.sparse_distance_matrix(tree_img, cutoff, output_type="coo_matrix")
    i = sdm.row.astype(np.int64)
    flat = sdm.col.astype(np.int64)
    j = flat % natom
    s = flat // natom
    sh = shifts[s]
    vec = pair_vectors(pos, cell, i, j, sh)
    dist = np.sqrt((vec * vec).sum(1))
    keep = (i != j) | sh.any(1)           # all but each atom with itself, as in DFTB+
    order = np.lexsort((s[keep], j[keep], i[keep]))
    i, j, sh, vec, dist = (x[keep][order] for x in (i, j, sh, vec, dist))
    close = np.flatnonzero(dist < MIN_NEIGH_DIST)
    if len(close):
        k = close[0]
        t = sh[k].tolist()
        if not any(t):
            who = f"atoms {i[k]} and {j[k]}"
        elif i[k] == j[k]:
            who = f"atom {i[k]} and its own periodic image at lattice shift {t}"
        else:
            who = f"atom {i[k]} and the periodic image of atom {j[k]} at lattice shift {t}"
        raise AtomsTooClose(f"{who} are {dist[k]:.3g} Bohr apart, closer than "
                            f"{MIN_NEIGH_DIST:g} Bohr (DFTB+ minNeighDist)")
    return i, j, sh, vec, dist
