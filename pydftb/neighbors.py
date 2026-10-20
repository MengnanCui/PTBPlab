"""Periodic neighbour list built on a k-d tree (O(N log N))."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


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

    Self-pairs (i == j, shift == 0) are excluded.  Returns
    ``i, j, shifts (int, (n,3)), vec (n,3), dist (n,)``.
    """
    natom = len(pos)
    pbc = np.zeros(3, bool) if cell is None else np.asarray(pbc, bool)
    nimg = image_range(cell, pbc, cutoff)
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
    keep = dist > 1e-8
    order = np.lexsort((s[keep], j[keep], i[keep]))
    return (i[keep][order], j[keep][order], sh[keep][order],
            vec[keep][order], dist[keep][order])
