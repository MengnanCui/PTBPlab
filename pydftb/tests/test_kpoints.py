"""k-point grids: the vectorised time-reversal reduction (milestone M7e) gives the points,
order and weights of the loop it replaced, bit for bit."""
import numpy as np
import pytest

from pydftb.kpoints import monkhorst_pack, reduce_time_reversal


def _loop(kpts, weights, decimals: int = 8):
    """``reduce_time_reversal`` before M7e (a dictionary over the canonical tuples)."""
    k = np.asarray(kpts, float)
    w = np.asarray(weights, float)
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


def _same(a, b):
    assert a[0].shape == b[0].shape and a[1].shape == b[1].shape
    assert a[0].tobytes() == b[0].tobytes()
    assert a[1].tobytes() == b[1].tobytes()


@pytest.mark.parametrize("grid", [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 4, 3),
                                  (14, 14, 14), (16, 16, 16), (12, 12, 6), (1, 7, 2)])
@pytest.mark.parametrize("gamma", [False, True])
def test_grids_as_the_loop(grid, gamma):
    k, w = monkhorst_pack(grid, gamma_centered=gamma)
    _same(reduce_time_reversal(k, w), _loop(k, w))


def test_shifted_shuffled_and_weighted_inputs():
    rng = np.random.default_rng(7)
    for _ in range(40):
        n = tuple(rng.integers(1, 7, size=3))
        k, _ = monkhorst_pack(n, gamma_centered=bool(rng.integers(2)))
        k = k + rng.choice([0.0, 0.5, 0.25, -1.0, 2.0], size=3)       # images and shifts
        k = k[rng.permutation(len(k))]
        if len(k) > 1:                                                # the same point twice
            k = np.concatenate([k, k[rng.integers(len(k), size=3)]])
        w = rng.random(len(k))
        _same(reduce_time_reversal(k, w), _loop(k, w))
        _same(reduce_time_reversal(k, w, decimals=5), _loop(k, w, decimals=5))


def test_signed_zeros_near_one_and_empty_inputs():
    k = np.array([[0.0, -0.0, 0.5], [-0.0, 0.0, -0.5], [1.0 - 1e-12, 0.0, 0.0],
                  [1e-12, 0.0, 0.0], [0.25, -0.25, 0.0], [-0.25, 0.25, -0.0], [0.3, 0.1, 0.9]])
    w = np.array([0.1, -0.0, 0.2, 0.3, 0.15, 0.05, 0.2])
    _same(reduce_time_reversal(k, w), _loop(k, w))
    _same(reduce_time_reversal(k[:1], w[1:2]), _loop(k[:1], w[1:2]))     # a -0.0 weight
    out = reduce_time_reversal(np.zeros((0, 3)), np.zeros(0))
    assert out[0].shape == (0, 3) and out[1].shape == (0,)
