"""Regressors from descriptor values to model parameters for the A-series environment models
(A0 plan, section 4: linear models on features first, then GPR and small neural networks).

Contract (shared with :mod:`pydftb.adaptive` and :mod:`pydftb.r0mix`): a regressor maps the
descriptor values x (n, n_in) of n atoms to n_out parameters per atom with ``size``
coefficients theta (one flat vector):

* ``value(x, theta)``   -> (n, n_out)
* ``d_theta(x, theta)`` -> (n, n_out, size)   (fitting: dF/dtheta = sum_A dF/dp_A dp_A/dtheta)
* ``d_input(x, theta)`` -> (n, n_out, n_in)   (forces: dp_A/dx_A, then the descriptor pairs)
* ``initial()``         -> theta with no environment dependence (all zeros: p = 0)
* ``names``, ``to_dict()``; :func:`regressor_from_dict` reads one back.

Coefficient layout of :class:`Linear`: output by output, [constant (if any), w_1 .. w_n_in].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class Linear:
    """p_o = c_o + sum_k w_ok (x_k - centre_k) / scale_k (centre 0 and scale 1 by default).
    Standardising the inputs (``centre``, ``scale``: e.g. mean and spread of the training
    features) makes one ridge strength fit all coefficients."""
    n_in: int
    n_out: int = 1
    constant: bool = True
    centre: tuple | None = None
    scale: tuple | None = None
    kind: ClassVar[str] = "linear"

    def __post_init__(self):
        if self.n_in < 0 or self.n_out < 1:
            raise ValueError("Linear needs n_in >= 0 and n_out >= 1")
        if not self.constant and self.n_in == 0:
            raise ValueError("Linear without a constant needs inputs")
        for name in ("centre", "scale"):
            v = getattr(self, name)
            if v is not None:
                v = tuple(float(x) for x in np.ravel(v))
                if len(v) != self.n_in:
                    raise ValueError(f"Linear {name} needs {self.n_in} values")
                if name == "scale" and any(x <= 0.0 for x in v):
                    raise ValueError("Linear scale must be > 0")
                object.__setattr__(self, name, v)

    @property
    def per_output(self) -> int:
        return int(self.constant) + self.n_in

    @property
    def size(self) -> int:
        return self.n_out * self.per_output

    @property
    def names(self) -> list:
        out = []
        for o in range(self.n_out):
            if self.constant:
                out.append(f"out{o}.c")
            out += [f"out{o}.w{k}" for k in range(self.n_in)]
        return out

    def initial(self) -> np.ndarray:
        return np.zeros(self.size)

    def _z(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        if x.shape[1] != self.n_in:
            raise ValueError(f"Linear expects {self.n_in} inputs, got {x.shape[1]}")
        c = np.zeros(self.n_in) if self.centre is None else np.asarray(self.centre)
        s = np.ones(self.n_in) if self.scale is None else np.asarray(self.scale)
        return (x - c) / s, s

    def _features(self, x):
        z, _ = self._z(x)
        if self.constant:
            return np.hstack([np.ones((len(z), 1)), z])
        return z

    def _matrix(self, theta):
        theta = np.asarray(theta, float)
        if theta.shape != (self.size,):
            raise ValueError(f"Linear needs {self.size} coefficients, got {theta.shape}")
        return theta.reshape(self.n_out, self.per_output)

    def value(self, x, theta) -> np.ndarray:
        return self._features(x) @ self._matrix(theta).T

    def d_theta(self, x, theta) -> np.ndarray:
        phi = self._features(x)
        n = len(phi)
        out = np.zeros((n, self.n_out, self.size))
        for o in range(self.n_out):
            out[:, o, o * self.per_output:(o + 1) * self.per_output] = phi
        return out

    def d_input(self, x, theta) -> np.ndarray:
        _, s = self._z(x)
        w = self._matrix(theta)[:, int(self.constant):] / s[None, :]
        n = len(np.atleast_2d(x))
        return np.broadcast_to(w, (n, self.n_out, self.n_in)).copy()

    def to_dict(self) -> dict:
        return {"kind": self.kind, "n_in": self.n_in, "n_out": self.n_out,
                "constant": self.constant,
                "centre": None if self.centre is None else list(self.centre),
                "scale": None if self.scale is None else list(self.scale)}


def _standardised(x, n_in, centre, scale):
    x = np.atleast_2d(np.asarray(x, float))
    if x.shape[1] != n_in:
        raise ValueError(f"expected {n_in} inputs, got {x.shape[1]}")
    c = np.zeros(n_in) if centre is None else np.asarray(centre)
    s = np.ones(n_in) if scale is None else np.asarray(scale)
    return (x - c) / s, s


def _farthest_points(x, m: int, seed: int = 0) -> np.ndarray:
    """Indices of m rows of x chosen by farthest-point sampling (first one at random)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    m = min(int(m), n)
    pick = [int(rng.integers(n))]
    dist = np.linalg.norm(x - x[pick[0]], axis=1)
    while len(pick) < m:
        k = int(np.argmax(dist))
        if dist[k] <= 0.0:
            break
        pick.append(k)
        dist = np.minimum(dist, np.linalg.norm(x - x[k], axis=1))
    return np.array(pick, int)


@dataclass(frozen=True)
class Kernel:
    """Kernel regression in the weight space of a GPR (DOVE's regressor, A0 section 4):
    p_o(x) = c_o + sum_m alpha_om k(x, X_m), k(x, X) = exp(-sum_f (x_f - X_f)^2 / (2 l_f^2)),
    with fixed inducing points X_m (e.g. :meth:`from_features`: farthest-point sampling of the
    training features).  Linear in the coefficients (c_o, alpha_om), so a ridge on them is a
    Gaussian prior; :meth:`penalty_matrix` gives K_mm per output for the RKHS-norm prior
    alpha^T K alpha instead.  Coefficient layout: output by output, [c (if any), alpha_1..M].
    """
    inducing: tuple
    length_scale: object = 1.0
    n_out: int = 1
    constant: bool = True
    kind: ClassVar[str] = "kernel"

    def __post_init__(self):
        xm = np.atleast_2d(np.asarray(self.inducing, float))
        if xm.size == 0:
            raise ValueError("Kernel needs at least one inducing point")
        ls = np.ravel(np.asarray(self.length_scale, float))
        if len(ls) == 1:
            ls = np.full(xm.shape[1], ls[0])
        if len(ls) != xm.shape[1] or np.any(ls <= 0.0):
            raise ValueError("Kernel length_scale: one positive value or one per input")
        if self.n_out < 1:
            raise ValueError("Kernel needs n_out >= 1")
        object.__setattr__(self, "inducing", tuple(tuple(r) for r in xm))
        object.__setattr__(self, "length_scale", tuple(float(v) for v in ls))

    @property
    def n_in(self) -> int:
        return len(self.inducing[0])

    @property
    def n_inducing(self) -> int:
        return len(self.inducing)

    @property
    def per_output(self) -> int:
        return int(self.constant) + self.n_inducing

    @property
    def size(self) -> int:
        return self.n_out * self.per_output

    @property
    def names(self) -> list:
        out = []
        for o in range(self.n_out):
            if self.constant:
                out.append(f"out{o}.c")
            out += [f"out{o}.a{m}" for m in range(self.n_inducing)]
        return out

    def initial(self) -> np.ndarray:
        return np.zeros(self.size)

    def _k(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        if x.shape[1] != self.n_in:
            raise ValueError(f"Kernel expects {self.n_in} inputs, got {x.shape[1]}")
        xm, ls = np.asarray(self.inducing), np.asarray(self.length_scale)
        diff = (x[:, None, :] - xm[None, :, :]) / ls
        k = np.exp(-0.5 * (diff ** 2).sum(axis=2))               # (n, M)
        dk = -k[:, :, None] * diff / ls                          # (n, M, n_in)
        return k, dk

    def _matrix(self, theta):
        theta = np.asarray(theta, float)
        if theta.shape != (self.size,):
            raise ValueError(f"Kernel needs {self.size} coefficients, got {theta.shape}")
        return theta.reshape(self.n_out, self.per_output)

    def _features(self, x):
        k, dk = self._k(x)
        if self.constant:
            return np.hstack([np.ones((len(k), 1)), k]), dk
        return k, dk

    def value(self, x, theta) -> np.ndarray:
        return self._features(x)[0] @ self._matrix(theta).T

    def d_theta(self, x, theta) -> np.ndarray:
        phi, _ = self._features(x)
        out = np.zeros((len(phi), self.n_out, self.size))
        for o in range(self.n_out):
            out[:, o, o * self.per_output:(o + 1) * self.per_output] = phi
        return out

    def d_input(self, x, theta) -> np.ndarray:
        _, dk = self._k(x)
        alpha = self._matrix(theta)[:, int(self.constant):]       # (n_out, M)
        return np.einsum("om,nmf->nof", alpha, dk)

    def penalty_matrix(self) -> np.ndarray:
        """(size, size) block-diagonal K_mm per output (zero for the constants)."""
        k, _ = self._k(np.asarray(self.inducing))
        out = np.zeros((self.size, self.size))
        c = int(self.constant)
        for o in range(self.n_out):
            a = o * self.per_output + c
            out[a:a + self.n_inducing, a:a + self.n_inducing] = k
        return out

    @classmethod
    def from_features(cls, x, n_inducing: int = 50, n_out: int = 1, length_scale="median",
                      constant: bool = True, seed: int = 0) -> "Kernel":
        """Inducing points by farthest-point sampling of the training features x (n, n_in);
        ``length_scale`` "median": the median distance between the inducing points per
        input (floored), or given."""
        x = np.atleast_2d(np.asarray(x, float))
        xm = x[_farthest_points(x, n_inducing, seed)]
        if isinstance(length_scale, str):
            if length_scale != "median":
                raise ValueError("length_scale is 'median' or numbers")
            if len(xm) > 1:
                d = np.abs(xm[:, None, :] - xm[None, :, :])
                iu = np.triu_indices(len(xm), 1)
                ls = np.median(d[iu], axis=0)
            else:
                ls = np.ones(x.shape[1])
            spread = x.std(axis=0)
            ls = np.where(ls > 1e-8, ls, np.where(spread > 1e-8, spread, 1.0))
        else:
            ls = length_scale
        return cls(tuple(map(tuple, xm)), ls, n_out, constant)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "inducing": [list(r) for r in self.inducing],
                "length_scale": list(self.length_scale), "n_out": self.n_out,
                "constant": self.constant}

    @classmethod
    def from_dict(cls, d: dict) -> "Kernel":
        return cls(tuple(tuple(r) for r in d["inducing"]), tuple(d["length_scale"]),
                   d.get("n_out", 1), d.get("constant", True))


@dataclass(frozen=True)
class MLP:
    """A small neural network p = W_L tanh(... tanh(W_1 z + b_1) ...) + b_L, z = (x - centre)
    / scale, with analytic derivatives (A0 section 4: the third regressor; channel E3 with a
    network is A4's Delta-2).  Coefficients: layer by layer, W (row-major, n_next x n_prev)
    then b.  :meth:`initial` draws the hidden weights (seeded, N(0, 1/fan_in)) and zeroes the
    output layer, so p = 0 (no environment dependence) but the gradient is not degenerate.
    The fit is non-convex: start several seeds."""
    n_in: int
    n_out: int = 1
    hidden: tuple = (16,)
    centre: tuple | None = None
    scale: tuple | None = None
    seed: int = 0
    kind: ClassVar[str] = "mlp"

    def __post_init__(self):
        if self.n_in < 1 or self.n_out < 1:
            raise ValueError("MLP needs n_in >= 1 and n_out >= 1")
        hid = tuple(int(h) for h in np.ravel(self.hidden))
        if any(h < 1 for h in hid):
            raise ValueError("MLP hidden layer sizes must be >= 1")
        object.__setattr__(self, "hidden", hid)
        for name in ("centre", "scale"):
            v = getattr(self, name)
            if v is not None:
                v = tuple(float(x) for x in np.ravel(v))
                if len(v) != self.n_in:
                    raise ValueError(f"MLP {name} needs {self.n_in} values")
                if name == "scale" and any(x <= 0.0 for x in v):
                    raise ValueError("MLP scale must be > 0")
                object.__setattr__(self, name, v)

    @property
    def widths(self) -> list:
        return [self.n_in, *self.hidden, self.n_out]

    @property
    def size(self) -> int:
        w = self.widths
        return sum(w[k + 1] * w[k] + w[k + 1] for k in range(len(w) - 1))

    @property
    def names(self) -> list:
        out = []
        w = self.widths
        for k in range(len(w) - 1):
            out += [f"L{k}.W[{r},{c}]" for r in range(w[k + 1]) for c in range(w[k])]
            out += [f"L{k}.b[{r}]" for r in range(w[k + 1])]
        return out

    def _layers(self, theta):
        theta = np.asarray(theta, float)
        if theta.shape != (self.size,):
            raise ValueError(f"MLP needs {self.size} coefficients, got {theta.shape}")
        w, out, k = self.widths, [], 0
        for a in range(len(w) - 1):
            n = w[a + 1] * w[a]
            W = theta[k:k + n].reshape(w[a + 1], w[a])
            k += n
            b = theta[k:k + w[a + 1]]
            k += w[a + 1]
            out.append((W, b))
        return out

    def initial(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        w, parts = self.widths, []
        for a in range(len(w) - 1):
            last = a == len(w) - 2
            W = np.zeros((w[a + 1], w[a])) if last else rng.normal(0.0, 1.0 / np.sqrt(w[a]),
                                                                    (w[a + 1], w[a]))
            parts += [W.ravel(), np.zeros(w[a + 1])]
        return np.concatenate(parts)

    def _forward(self, x, theta):
        z, s = _standardised(x, self.n_in, self.centre, self.scale)
        layers = self._layers(theta)
        acts = [z]
        for W, b in layers[:-1]:
            acts.append(np.tanh(acts[-1] @ W.T + b))
        W, b = layers[-1]
        return acts, acts[-1] @ W.T + b, layers, s

    def value(self, x, theta) -> np.ndarray:
        return self._forward(x, theta)[1]

    def _backward(self, x, theta):
        """d y / d (every layer's input activation), from the output back: list of
        (n, n_out, width) arrays, and the forward pass."""
        acts, y, layers, s = self._forward(x, theta)
        n = len(y)
        grads = [None] * len(layers)
        g = np.broadcast_to(np.eye(self.n_out), (n, self.n_out, self.n_out))    # dy/dy
        for a in range(len(layers) - 1, -1, -1):
            W, _ = layers[a]
            grads[a] = g                                    # dy / d(pre-activation of a+1)
            g = g @ W                                       # dy / d acts[a]
            if a > 0:
                g = g * (1.0 - acts[a] ** 2)[:, None, :]    # through tanh of acts[a]
        return grads, g, acts, s

    def d_theta(self, x, theta) -> np.ndarray:
        grads, _, acts, _ = self._backward(x, theta)
        parts = []
        for a, g in enumerate(grads):
            # pre-activation_{a+1} = W_a acts[a] + b_a
            dW = g[:, :, :, None] * acts[a][:, None, None, :]
            parts += [dW.reshape(len(g), self.n_out, -1), g]
        return np.concatenate(parts, axis=2)

    def d_input(self, x, theta) -> np.ndarray:
        _, g0, _, s = self._backward(x, theta)
        return g0 / s[None, None, :]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "n_in": self.n_in, "n_out": self.n_out,
                "hidden": list(self.hidden),
                "centre": None if self.centre is None else list(self.centre),
                "scale": None if self.scale is None else list(self.scale), "seed": self.seed}

    @classmethod
    def from_dict(cls, d: dict) -> "MLP":
        d = dict(d)
        d["hidden"] = tuple(d.get("hidden", (16,)))
        return cls(**d)


_KINDS = {"linear": Linear, "kernel": Kernel, "mlp": MLP}


def register(kind: str, cls) -> None:
    """Make :func:`regressor_from_dict` know another regressor class (``cls.from_dict(d)`` if
    it has one, else ``cls(**fields)``)."""
    _KINDS[kind] = cls


def regressor_from_dict(d: dict):
    d = dict(d)
    kind = d.pop("kind")
    if kind not in _KINDS:
        raise ValueError(f"unknown regressor kind {kind!r} (known: {sorted(_KINDS)})")
    cls = _KINDS[kind]
    if hasattr(cls, "from_dict"):
        return cls.from_dict(d)
    return cls(**d)
