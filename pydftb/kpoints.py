"""k-point grids."""
from __future__ import annotations

import numpy as np


def monkhorst_pack(n, gamma_centered: bool = False):
    """Full Monkhorst-Pack grid in fractional coordinates, weights sum to 1."""
    g = [np.arange(k) / k - (0 if gamma_centered else (k - 1) / (2.0 * k)) for k in n]
    k = np.array(np.meshgrid(*g, indexing="ij")).reshape(3, -1).T
    return k, np.full(len(k), 1.0 / len(k))


def reduce_time_reversal(kpts, weights, decimals: int = 8):
    """Merge k and -k (H(-k) = H(k)^*  => identical eigenvalues)."""
    k = np.asarray(kpts, float)
    w = np.asarray(weights, float)
    # canonical representative of the pair {k, -k}, folded into [0, 1)
    a = np.round(np.mod(k, 1.0), decimals) % 1.0
    b = np.round(np.mod(-k, 1.0), decimals) % 1.0
    canon = [min(tuple(x), tuple(y)) for x, y in zip(a, b)]
    index, first, weight = {}, [], []
    for idx, c in enumerate(canon):
        j = index.get(c)
        if j is None:
            index[c] = len(first)
            first.append(idx)
            weight.append(w[idx])
        else:
            weight[j] += w[idx]
    return k[first], np.array(weight)


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
