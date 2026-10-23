"""Charge mixing for the SCC cycle.

DFTB+'s four mixers (``mixer/*.F90``), each an exact port so that the SCC takes the same
iterations as in DFTB+ (milestone M6f):

* :class:`BroydenMixer` - modified Broyden mixing, DFTB+'s default
  (``broydenmixer.F90``; Johnson, PRB 38, 12807 (1988)).
* :class:`SimpleMixer` - linear mixing (``simplemixer.F90``).
* :class:`AndersonMixer` - DFTB+'s Anderson mixer (``andersonmixer.F90``).
* :class:`DIISMixer` - DFTB+'s DIIS mixer (``diismixer.F90``).

and :class:`PulayMixer`, pydftb's original Pulay/DIIS-type mixer (``"pulay"``; it was also
called ``"anderson"`` before M6f, but it is not DFTB+'s Anderson mixer).

``mix(q_in, q_out)`` returns the next input charges.  The mixed quantity is the population
of every SCC site (atom, or shell when shell-resolved), as in DFTB+.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .backend import ops


class BroydenMixer:
    """DFTB+ modified Broyden mixer (defaults = DFTB+ defaults)."""

    def __init__(self, alpha: float = 0.2, omega0: float = 0.01, min_weight: float = 1.0,
                 max_weight: float = 1.0e5, weight_factor: float = 1.0e-2):
        self.alpha = alpha
        self.omega0 = omega0
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight_factor = weight_factor
        self.n = 0
        self.q_last = self.f_last = None
        self.w: list[float] = []          # weights of the stored iterations
        self.df: list[np.ndarray] = []    # normalised dF vectors
        self.u: list[np.ndarray] = []
        self.a: list[list[float]] = []    # overlaps <dF_i|dF_j>

    def mix(self, q_in: np.ndarray, q_out: np.ndarray) -> np.ndarray:
        f = q_out - q_in
        self.n += 1
        if self.n == 1:
            self.q_last, self.f_last = q_in.copy(), f.copy()
            return q_in + self.alpha * f
        norm_f = np.sqrt(f @ f)
        w = self.weight_factor / norm_f if norm_f > self.weight_factor / self.max_weight \
            else self.max_weight
        w = max(w, self.min_weight)
        dfu = f - self.f_last
        inv_norm = 1.0 / max(np.sqrt(dfu @ dfu), np.finfo(float).eps)
        dfu = inv_norm * dfu
        m = len(self.df)                       # previous vectors: nn - 2 in DFTB+
        row = [self.df[i] @ dfu for i in range(m)]
        for i in range(m):
            self.a[i].append(row[i])
        self.a.append(row + [1.0])
        self.w.append(w)
        ww = np.array(self.w)
        aa = np.array(self.a)
        cc = np.array([ww[i] * (self.df[i] @ f) for i in range(m)] + [w * (dfu @ f)])
        beta = ww[:, None] * ww[None, :] * aa
        beta[np.diag_indices_from(beta)] += self.omega0 ** 2
        cc = sla.solve(beta.T, cc)
        self.df.append(dfu)
        u = self.alpha * dfu + inv_norm * (q_in - self.q_last)
        q_new = q_in + self.alpha * f
        for i in range(m):
            q_new = q_new - ww[i] * cc[i] * self.u[i]
        q_new = q_new - w * cc[m] * u
        self.u.append(u)
        self.q_last, self.f_last = q_in.copy(), f.copy()
        return q_new


class SimpleMixer:
    """DFTB+ simple (linear) mixer: q_in + alpha (q_out - q_in)."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def mix(self, q_in: np.ndarray, q_out: np.ndarray) -> np.ndarray:
        return q_in + self.alpha * (q_out - q_in)


class AndersonMixer:
    """DFTB+ Anderson mixer.

    It keeps ``generations - 1`` earlier (input, difference) pairs, newest first.  The first
    ``generations`` calls use ``init_alpha``, the later ones ``alpha``; ``dynamic`` [(threshold,
    alpha), ...] (DFTB+ ``DynMixingParameters``) replaces the mixing parameter with the alpha of
    the last entry, scanning backwards, whose threshold exceeds |F| (not on the first call).
    ``omega0`` (``DiagonalRescaling``) scales the diagonal of the Anderson matrix by
    (1 + omega0^2)."""

    def __init__(self, alpha: float = 0.05, generations: int = 4, init_alpha: float = 0.01,
                 dynamic=None, omega0: float = 0.01):
        if generations < 2:
            raise ValueError("the Anderson mixer needs generations >= 2")
        self.alpha = alpha
        self.init_alpha = init_alpha
        self.dynamic = [(float(t), float(a)) for t, a in (dynamic or [])]
        self.omega02 = omega0 ** 2 if omega0 > 0 else 0.0
        self.m = generations - 1
        self.n_prev = -1
        self.indx = [self.m - i for i in range(self.m)]      # 1-based slots, as in DFTB+
        self.q_prev: dict[int, np.ndarray] = {}
        self.f_prev: dict[int, np.ndarray] = {}

    def _store(self, q: np.ndarray, f: np.ndarray):
        self.indx = [self.indx[-1]] + self.indx[:-1]
        self.q_prev[self.indx[0]] = q.copy()
        self.f_prev[self.indx[0]] = f.copy()

    def mix(self, q_in: np.ndarray, q_out: np.ndarray) -> np.ndarray:
        f = q_out - q_in
        if self.n_prev < self.m:
            self.n_prev += 1
            alpha = self.init_alpha
        else:
            alpha = self.alpha
        norm_f = np.sqrt(f @ f)
        for threshold, a in reversed(self.dynamic):
            if norm_f < threshold:
                alpha = a
                break
        if self.n_prev == 0:
            self._store(q_in, f)
            return q_in + self.init_alpha * f
        n = self.n_prev
        prev_f = [self.f_prev[self.indx[i]] for i in range(n)]
        prev_q = [self.q_prev[self.indx[i]] for i in range(n)]
        aa = np.empty((n, n))
        bb = np.empty(n)
        for i in range(n):
            t1 = f - prev_f[i]
            t2 = f @ t1
            bb[i] = t2
            for j in range(n):
                aa[i, j] = t2 - prev_f[j] @ t1
        if self.omega02 > 0.0:
            aa[np.diag_indices(n)] *= 1.0 + self.omega02
        c = np.linalg.solve(aa, bb)
        rest = 1.0 - c.sum()
        f_mid = np.zeros_like(f)
        q_mid = np.zeros_like(q_in)
        for i in range(n):
            f_mid = f_mid + c[i] * prev_f[i]
        f_mid = f_mid + rest * f
        for i in range(n):
            q_mid = q_mid + c[i] * prev_q[i]
        q_mid = q_mid + rest * q_in
        self._store(q_in, f)
        return q_mid + alpha * f_mid


class DIISMixer:
    """DFTB+ DIIS mixer.

    The last ``generations`` (input, difference) pairs are stored in a circular buffer; the
    new input is sum_i c_i (q_i + F_i) with the DIIS coefficients (sum c_i = 1).  Until the
    buffer is full, ``init_alpha`` F is added, and without ``from_start``
    (``UseFromStart = No``) those first steps are simple mixing only."""

    def __init__(self, init_alpha: float = 0.2, generations: int = 6, from_start: bool = True):
        if generations < 1:
            raise ValueError("the DIIS mixer needs generations >= 1")
        self.init_alpha = init_alpha
        self.from_start = bool(from_start)
        self.m = generations
        self.i_prev = 0
        self.indx = 0
        self.q_prev: dict[int, np.ndarray] = {}
        self.f_prev: dict[int, np.ndarray] = {}

    def mix(self, q_in: np.ndarray, q_out: np.ndarray) -> np.ndarray:
        f = q_out - q_in
        if self.i_prev < self.m:
            self.i_prev += 1
        self.indx = self.indx % self.m + 1
        self.q_prev[self.indx] = q_in.copy()
        self.f_prev[self.indx] = f.copy()
        q_new = q_in
        if self.from_start or self.i_prev == self.m:
            n = self.i_prev
            aa = np.zeros((n + 1, n + 1))
            bb = np.zeros(n + 1)
            for i in range(n):
                for j in range(n):
                    aa[i, j] = self.f_prev[i + 1] @ self.f_prev[j + 1]
            aa[n, :n] = -1.0
            aa[:n, n] = -1.0
            bb[n] = -1.0
            c = np.linalg.solve(aa, bb)
            q_new = np.zeros_like(q_in)
            for i in range(n):
                q_new = q_new + c[i] * (self.q_prev[i + 1] + self.f_prev[i + 1])
        if self.i_prev < self.m:
            q_new = q_new + self.init_alpha * f
        return q_new


class PulayMixer:
    """Pulay/DIIS-type mixing over the last ``history`` iterations (pydftb's original mixer)."""

    def __init__(self, beta: float = 0.2, history: int = 8):
        self.beta = beta
        self.history = history
        self.xs: list[np.ndarray] = []
        self.fs: list[np.ndarray] = []

    def mix(self, q_in: np.ndarray, q_out: np.ndarray) -> np.ndarray:
        f = q_out - q_in
        self.xs.append(q_in.copy())
        self.fs.append(f.copy())
        if len(self.xs) > self.history:
            self.xs.pop(0)
            self.fs.pop(0)
        if len(self.xs) == 1:
            return q_in + self.beta * f
        df = np.array([self.fs[k + 1] - self.fs[k] for k in range(len(self.fs) - 1)])
        dx = np.array([self.xs[k + 1] - self.xs[k] for k in range(len(self.xs) - 1)])
        a = df @ df.T
        a += 1e-12 * np.trace(a) * np.eye(len(a))
        gam = ops.solve(a, df @ f)
        x_bar = q_in - gam @ dx
        f_bar = f - gam @ df
        return x_bar + self.beta * f_bar


# options of each mixer: pydftb name (snake case of DFTB+'s keyword) -> constructor argument
_OPTIONS = {
    "broyden": {"inverse_jacobi_weight": "omega0", "minimal_weight": "min_weight",
                "maximal_weight": "max_weight", "weight_factor": "weight_factor"},
    "simple": {},
    "anderson": {"generations": "generations", "init_mixing": "init_alpha",
                 "dynamic_mixing": "dynamic", "diagonal_rescaling": "omega0"},
    "diis": {"init_mixing": "init_alpha", "generations": "generations",
             "use_from_start": "from_start"},
    "pulay": {"history": "history"},
}
MIXERS = {"broyden": BroydenMixer, "simple": SimpleMixer, "anderson": AndersonMixer,
          "diis": DIISMixer, "pulay": PulayMixer}


def mixer_spec(kind="broyden", mixing: float | None = None, **options) -> tuple[str, dict]:
    """The mixer and its constructor arguments for ``kind`` (a name, or a dict with "kind" and
    options), ``mixing`` (DFTB+ ``MixingParameter``; None: the DFTB+ default) and options
    named after DFTB+'s keywords (e.g. ``generations``, ``init_mixing``)."""
    if isinstance(kind, dict):
        options = {**kind, **options}
        kind = options.pop("kind", "broyden")
        mixing = options.pop("mixing", mixing)
    name = str(kind).lower()
    if name not in MIXERS:
        raise ValueError(f"unknown mixer {kind!r} (use one of {sorted(MIXERS)})")
    unknown = set(options) - set(_OPTIONS[name])
    if unknown:
        raise ValueError(f"mixer {name!r}: unknown options {sorted(unknown)} "
                         f"(allowed: {sorted(_OPTIONS[name])})")
    args = {_OPTIONS[name][k]: v for k, v in options.items()}
    if mixing is not None:
        if name == "diis":
            raise ValueError("the DIIS mixer has no mixing parameter (set init_mixing)")
        args["beta" if name == "pulay" else "alpha"] = mixing
    return name, args


def make_mixer(kind="broyden", mixing: float | None = None, **options):
    """A new mixer (see :func:`mixer_spec`)."""
    name, args = mixer_spec(kind, mixing, **options)
    return MIXERS[name](**args)
