"""Regressors of the A-series environment models: derivatives against central differences,
layouts, the kernel's inducing points, JSON round trips."""
import json

import numpy as np
import pytest

from pydftb.regressors import MLP, Kernel, Linear, regressor_from_dict


def _fd_theta(reg, x, th, h=1e-6):
    out = np.zeros((len(x), reg.n_out, reg.size))
    for k in range(reg.size):
        e = np.zeros(reg.size)
        e[k] = h
        out[:, :, k] = (reg.value(x, th + e) - reg.value(x, th - e)) / (2 * h)
    return out


def _fd_input(reg, x, th, h=1e-6):
    out = np.zeros((len(x), reg.n_out, x.shape[1]))
    for f in range(x.shape[1]):
        e = np.zeros(x.shape[1])
        e[f] = h
        out[:, :, f] = (reg.value(x + e, th) - reg.value(x - e, th)) / (2 * h)
    return out


REGS = {
    "linear": Linear(3, n_out=2, centre=(1.0, 0.0, -1.0), scale=(2.0, 1.0, 0.5)),
    "kernel": Kernel(((0.0, 0.0, 0.0), (1.0, 0.5, -0.5), (2.0, -1.0, 0.3)),
                     length_scale=(0.8, 1.2, 0.6), n_out=2),
    "kernel_nc": Kernel(((0.0, 1.0, 0.0), (1.0, 0.5, -0.5)), length_scale=0.9, constant=False),
    "mlp": MLP(3, n_out=2, hidden=(5,), centre=(0.5, 0.0, 0.0), scale=(1.0, 2.0, 1.0), seed=3),
    "mlp2": MLP(3, n_out=1, hidden=(4, 3), seed=1),
}


@pytest.mark.parametrize("name", sorted(REGS))
def test_derivatives(name):
    reg = REGS[name]
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 3))
    th = rng.normal(size=reg.size) * 0.5
    v = reg.value(x, th)
    assert v.shape == (6, reg.n_out) and len(reg.names) == reg.size
    dth = reg.d_theta(x, th)
    din = reg.d_input(x, th)
    assert np.abs(dth - _fd_theta(reg, x, th)).max() < 1e-7
    assert np.abs(din - _fd_input(reg, x, th)).max() < 1e-7
    assert np.abs(din).max() > 1e-3 and np.abs(dth).max() > 1e-3


@pytest.mark.parametrize("name", sorted(REGS))
def test_initial_is_no_environment_dependence(name):
    reg = REGS[name]
    x = np.random.default_rng(1).normal(size=(4, 3))
    th = reg.initial()
    assert np.abs(reg.value(x, th)).max() == 0.0
    assert np.abs(reg.d_theta(x, th)).max() > 0.0           # a usable starting gradient


@pytest.mark.parametrize("name", sorted(REGS))
def test_json_round_trip(name):
    reg = REGS[name]
    back = regressor_from_dict(json.loads(json.dumps(reg.to_dict())))
    x = np.random.default_rng(2).normal(size=(3, 3))
    th = np.random.default_rng(3).normal(size=reg.size)
    assert np.array_equal(back.value(x, th), reg.value(x, th))


def test_kernel_from_features_and_penalty():
    x = np.random.default_rng(4).normal(size=(40, 2))
    k = Kernel.from_features(x, n_inducing=8, n_out=1, seed=0)
    assert k.n_inducing == 8 and k.n_in == 2 and all(v > 0 for v in k.length_scale)
    p = k.penalty_matrix()
    assert p.shape == (k.size, k.size) and p[0, 0] == 0.0 and np.allclose(p[1:, 1:], p[1:, 1:].T)
    assert np.all(np.linalg.eigvalsh(p[1:, 1:]) > -1e-10)
    # a kernel fit by linear least squares interpolates a smooth target
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1]
    kk = Kernel.from_features(x, n_inducing=30, seed=1)
    a = kk.d_theta(x, kk.initial())[:, 0, :]
    th, *_ = np.linalg.lstsq(a, y, rcond=None)
    assert np.sqrt(np.mean((kk.value(x, th)[:, 0] - y) ** 2)) < 0.05


def test_refusals():
    with pytest.raises(ValueError):
        Kernel(((0.0, 1.0),), length_scale=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        MLP(0)
    with pytest.raises(ValueError):
        MLP(2, hidden=(0,))
    with pytest.raises(ValueError):
        MLP(2).value(np.zeros((1, 3)), MLP(2).initial())
