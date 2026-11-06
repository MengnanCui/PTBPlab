"""k-point grids."""
from __future__ import annotations

import numpy as np


def monkhorst_pack(n, gamma_centered: bool = False):
    """Full Monkhorst-Pack grid in fractional coordinates, weights sum to 1."""
    g = [np.arange(k) / k - (0 if gamma_centered else (k - 1) / (2.0 * k)) for k in n]
    k = np.array(np.meshgrid(*g, indexing="ij")).reshape(3, -1).T
    return k, np.full(len(k), 1.0 / len(k))


def reduce_time_reversal(kpts, weights, decimals: int = 8):
    """Merge k and -k (H(-k) = H(k)^*  => identical eigenvalues): the first k of every pair
    {k, -k} in the input order, with the weights of the pair summed in the input order.

    Vectorised (milestone M7e): the same points, order and sums, bit for bit, as the loop of
    before, which took about 10 ms for a 16^3 grid."""
    k = np.asarray(kpts, float)
    w = np.asarray(weights, float)
    if len(k) == 0:
        return k[[]], np.array([])
    # canonical representative of the pair {k, -k}, folded into [0, 1): the lexicographically
    # smaller one (Python's tuple order), compared at the first component where they differ
    a = np.round(np.mod(k, 1.0), decimals) % 1.0
    b = np.round(np.mod(-k, 1.0), decimals) % 1.0
    differ = a != b
    col = np.argmax(differ, axis=1)
    row = np.arange(len(k))
    canon = np.where((~differ.any(axis=1) | (a[row, col] < b[row, col]))[:, None], a, b)
    _, first, group = np.unique(canon, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")          # the pairs in order of first occurrence
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    pair = rank[group.reshape(-1)]                    # the pair of every input point
    weight = w[first[order]].copy()                   # its first point's weight, as the loop
    rest = np.ones(len(k), bool)
    rest[first] = False
    np.add.at(weight, pair[rest], w[rest])            # then the others, in input order
    return k[first[order]], weight


def supercell_folding(n):
    """DFTB+'s k-points of the supercell folding diag(n) with zero shift (``getSuperSampling``
    with ``reduceByInversion``): k = (i1/n1, i2/n2, i3/n3), i1 slowest, k and -k merged into the
    first occurrence, the weights summed.  The set and order of DFTB+'s hybrid charge files."""
    n = tuple(int(x) for x in n)
    if len(n) != 3 or min(n) < 1:
        raise ValueError(f"a supercell folding needs three positive integers, not {n}")
    idx = np.array(np.meshgrid(*[np.arange(m) for m in n], indexing="ij")).reshape(3, -1).T
    full = idx / np.array(n, float)
    keep = np.ones(len(full), bool)
    w = np.full(len(full), 1.0 / len(full))
    for a in range(len(full)):
        if not keep[a]:
            continue
        rr = np.mod(-full[a], 1.0)
        for b in range(a + 1, len(full)):
            if keep[b] and np.all(np.abs(full[b] - rr) < 1e-4):
                keep[b] = False
                w[a] += w[b]
    return full[keep], w[keep]


def folding_of(kpts, weights, tol: float = 1e-8):
    """The folding (n1, n2, n3) whose zero-shift grid the k-points are, the whole grid (equal
    weights) or DFTB+'s inversion-reduced set (any of k, -k), or None."""
    k = np.atleast_2d(np.asarray(kpts, float))
    w = np.asarray(weights, float)
    w = w / w.sum()
    n = []
    for c in k.T:
        m = 1
        while m <= 256 and np.abs(c * m - np.rint(c * m)).max() > tol:
            m += 1
        if m > 256:
            return None
        n.append(m)
    idx = np.mod(np.rint(k * np.array(n)), n).astype(int)
    total = int(np.prod(n))
    flat = (idx[:, 0] * n[1] + idx[:, 1]) * n[2] + idx[:, 2]
    neg = np.mod(-idx, n)
    flat_neg = (neg[:, 0] * n[1] + neg[:, 1]) * n[2] + neg[:, 2]
    if len(set(flat)) != len(flat):
        return None
    if len(k) == total and np.allclose(w, 1.0 / total, rtol=1e-8, atol=0.0):
        return tuple(n)
    covered = set(flat) | set(flat_neg)
    expect = np.where(flat == flat_neg, 1.0, 2.0) / total
    if (len(covered) == total and not (set(flat) & set(flat_neg[flat != flat_neg]))
            and np.allclose(w, expect, rtol=1e-8, atol=0.0)):
        return tuple(n)
    return None
