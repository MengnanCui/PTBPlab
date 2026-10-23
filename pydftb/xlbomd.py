"""Extended-Lagrangian Born-Oppenheimer molecular dynamics (XL-BOMD, milestone M6e).

After DFTB+ 25.1 ``md/extlagrangian.F90`` and ``md/xlbomd.F90`` (Niklasson et al.; Aradi,
Niklasson and Frauenheim, J. Chem. Theory Comput. 11, 3357 (2015)).  Auxiliary populations n
(the SCC's mixed vector) are integrated along with the atoms and start every step's SCC:

    n(t+dt) = 2 n(t) - n(t-dt) + kappa s [q_out(t) - n(t)] + alpha sum_{k=0..K} c_k n(t - k dt)

(the last term damps the numerical noise; K = 5, 6 or 7).  The energy is the shadow energy
E_band[D(n)] + 1/2 (2 q_out - n).gamma.n (+ the linearised third order), whose forces at fixed
n need no converged SCC (:meth:`pydftb.DFTB.calculate` with ``xl=True``).

The integration starts after ``pre_steps`` ordinary steps (each starting from the previous
output), fills a ring buffer with K + 1 converged outputs, blends in over ``transient_steps``
(XlbomdFast) and then runs with its own SCC settings: ``min_scc``..``max_scc`` iterations to
``scc_tol`` (DFTB+ ``Xlbomd``), or a single iteration with ``fast=True`` (``XlbomdFast``,
where ``scale`` < 1 keeps metallic or soft systems stable).  As in DFTB+, the minimum number
of iterations counts on the first MD step only.

Use through :class:`pydftb.ase_calc.PyDFTB` (``xlbomd=XLBOMD(...)``), one calculation per MD
step, e.g. with ASE's ``VelocityVerlet``; :class:`Integrator` holds the state between steps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# DFTB+ extlagrangian.F90: dissipation coefficients c_0..c_K, alpha, kappa
COEFFS = {
    5: (np.array([-6.0, 14.0, -8.0, -3.0, 4.0, -1.0]), 18e-3, 1.82),
    6: (np.array([-14.0, 36.0, -27.0, -2.0, 12.0, -6.0, 1.0]), 5.5e-3, 1.84),
    # DFTB+ writes alpha7 = 1.6e-3 without the _dp kind: a single-precision constant
    7: (np.array([-36.0, 99.0, -88.0, 11.0, 32.0, -25.0, 8.0, -1.0]),
        float(np.float32(1.6e-3)), 1.86),
}
OFF, FILLING_UP, INTERPOLATING, ON = 1, 2, 3, 4


class ExtLagrangian:
    """The integrator of the auxiliary vector (DFTB+ ``ExtLagrangian``): a ring buffer of the
    last K + 1 vectors and the phases off -> filling up -> interpolating -> on."""

    def __init__(self, n_time_steps: int, scale: float = 1.0):
        if n_time_steps not in COEFFS:
            raise ValueError("XL-BOMD integration steps must be 5, 6 or 7 (as in DFTB+)")
        self.k = n_time_steps
        self.coeffs, self.alpha, self.kappa = COEFFS[n_time_steps]
        self.scale = float(scale)
        self.phase = OFF
        self.i_step = 1
        self.n_steps = 0
        self.n_transient = 0
        self.ind = 0                # DFTB+ leaves it unset: in a ring buffer any start serves
        self.aux = None

    def turn_on(self, n_transient: int = 0):
        self.phase = FILLING_UP
        self.i_step = 1
        self.n_steps = self.k + 1
        self.n_transient = n_transient

    def _mod(self, ind: int) -> int:
        """DFTB+ modIndex, 0-based."""
        return ind % (self.k + 1)

    def next_input(self, out_last) -> np.ndarray:
        """The next input vector from the last output (DFTB+ getNextInput)."""
        out_last = np.asarray(out_last, float)
        if self.aux is None:
            self.aux = np.zeros((self.k + 1, out_last.size))
        if self.phase == OFF:
            nxt = out_last.copy()
        elif self.phase == FILLING_UP:
            self.ind = self._mod(self.ind + 1)
            self.aux[self.ind] = out_last
            nxt = out_last.copy()
        else:
            i0 = self.ind
            if self.phase == INTERPOLATING:
                cc = self.i_step / (self.n_steps + 1)
                self.aux[i0] = cc * self.aux[i0] + (1.0 - cc) * out_last
            nxt = 2.0 * self.aux[i0] - self.aux[self._mod(i0 - 1)]
            diff = out_last - self.aux[i0]
            nxt = nxt + self.kappa * self.scale * diff
            for ii in range(self.k + 1):
                nxt = nxt + self.alpha * self.coeffs[ii] * self.aux[self._mod(i0 - ii)]
            self.ind = self._mod(self.ind + 1)
            self.aux[self.ind] = nxt
        self._update()
        return nxt

    def needs_converged_values(self) -> bool:
        return self.phase in (OFF, FILLING_UP, INTERPOLATING)

    def _update(self):
        if self.phase in (OFF, ON):
            return
        self.i_step += 1
        if self.phase == FILLING_UP and self.i_step > self.n_steps:
            self.phase = INTERPOLATING
            self.i_step = 1
            self.n_steps = self.n_transient
        if self.phase == INTERPOLATING and self.i_step > self.n_steps:
            self.phase = ON


@dataclass
class XLBOMD:
    """XL-BOMD options (DFTB+ ``Xlbomd`` / ``XlbomdFast`` in the ``VelocityVerlet`` driver).

    ``integration_steps`` K (5, 6, 7), ``pre_steps`` ordinary steps before the integration.
    ``fast=False`` (Xlbomd): ``min_scc``..``max_scc`` SCC iterations to ``scc_tol`` per step
    once the integration runs.  ``fast=True`` (XlbomdFast): one SCC iteration per step after
    ``transient_steps``, the correction scaled by ``scale`` in (0, 1].
    """
    integration_steps: int = 5
    pre_steps: int = 0
    fast: bool = False
    transient_steps: int = 10
    scale: float = 1.0
    min_scc: int = 1
    max_scc: int = 200
    scc_tol: float = 1e-5

    def __post_init__(self):
        if self.integration_steps not in COEFFS:
            raise ValueError("integration_steps must be 5, 6 or 7 (as in DFTB+)")
        if self.pre_steps < 0 or self.transient_steps < 0:
            raise ValueError("pre_steps and transient_steps must not be negative")
        if self.fast:
            if not 0.0 < self.scale <= 1.0:
                raise ValueError("scale must be in (0, 1] (as in DFTB+)")
        else:
            if self.scale != 1.0:
                raise ValueError("scale is for fast=True (DFTB+ XlbomdFast)")
            if self.min_scc < 1 or self.max_scc < self.min_scc:
                raise ValueError("need 1 <= min_scc <= max_scc")
        if self.scc_tol <= 0:
            raise ValueError("scc_tol must be positive")

    @classmethod
    def make(cls, spec) -> "XLBOMD | None":
        if spec is None or isinstance(spec, cls):
            return spec
        if isinstance(spec, dict):
            return cls(**spec)
        raise TypeError("xlbomd must be an XLBOMD or a dict of its options")

    @property
    def scc(self) -> tuple[int, int, float]:
        """(min, max, tolerance) of the SCC once the integration runs; XlbomdFast: one
        iteration, DFTB+'s fixed tolerance 1e-5."""
        return (1, 1, 1e-5) if self.fast else (self.min_scc, self.max_scc, self.scc_tol)


class Integrator:
    """The XL-BOMD state across MD steps (DFTB+ ``TXLBOMD``)."""

    def __init__(self, opts: XLBOMD):
        self.opts = opts
        self.lag = ExtLagrangian(opts.integration_steps, opts.scale if opts.fast else 1.0)
        self.i_step = 1
        # DFTB+: iStartXl = nPreSteps + nFullSCCSteps - K, nFullSCCSteps = K + 1
        self.i_start = opts.pre_steps + 1
        self.n_transient = opts.transient_steps if opts.fast else 0
        self.next = None            # the input populations of the next step

    def scc_parameters(self, default_max: int, default_tol: float) -> tuple[int, int, float]:
        """(min, max, tol) of this step's SCC: the calculator's own until the integration
        runs (DFTB+ getSCCParameters); the minimum applies to the first step only (DFTB+
        getNextInputCharges: iSccIter >= minSccIter .or. iGeoStep > 0)."""
        if self.lag.needs_converged_values():
            mn, mx, tol = 1, default_max, default_tol
        else:
            mn, mx, tol = self.opts.scc
        return (mn if self.i_step == 1 else 1), mx, tol

    def advance(self, q_out) -> np.ndarray:
        """Hand the step's output populations to the integrator; returns the next input."""
        if self.i_step == self.i_start:
            self.lag.turn_on(self.n_transient)
        self.next = self.lag.next_input(q_out)
        self.i_step += 1
        return self.next
