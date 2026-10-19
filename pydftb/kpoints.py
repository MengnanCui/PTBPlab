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
