"""PySCF-style high-level interface.

>>> import pydftb
>>> from ase.build import molecule
>>> params = pydftb.ParamSet("path/to/skf")
>>> mf = pydftb.SCC(pydftb.System.from_ase(molecule("H2O")), params)
>>> e = mf.kernel()          # total energy, Hartree
>>> f = mf.forces()          # Hartree/Bohr

The object keeps its state (``e_tot``, ``mulliken``, ``mo_energy`` ...) after
``kernel()``, like a PySCF mean-field object.  All quantities are in atomic
units; use :mod:`pydftb.ase_calc` for an ASE calculator in eV/Angstrom.
"""
from __future__ import annotations

import numpy as np

from .calculator import DFTB, DFTBResult
from .dispersion import make_dispersion
from .kpoints import monkhorst_pack, reduce_time_reversal
from .params import ParamSet
from .system import System


def make_kpoints(kpts, time_reversal: bool = True):
    """None | (n1, n2, n3) Monkhorst-Pack grid | (kpts, weights) -> (kpts, weights) or None."""
    if kpts is None:
        return None
    if len(kpts) == 3 and np.ndim(kpts[0]) == 0:
        k, w = monkhorst_pack(tuple(int(n) for n in kpts))
    else:
        k, w = kpts
        k = np.atleast_2d(np.asarray(k, float))
        w = np.asarray(w, float)
    if time_reversal:
        k, w = reduce_time_reversal(k, w)
    return k, w / w.sum()


class SCC:
    """Self-consistent-charge DFTB calculation for one :class:`System`.

    Keyword arguments are passed to :class:`pydftb.calculator.DFTB`
    (``temperature``, ``scc_tol``, ``max_scc``, ``mixing``, ``solver``,
    ``repulsive``, ``dispersion`` ...).  ``kpts`` may be a Monkhorst-Pack grid ``(n1, n2, n3)``
    or explicit ``(kpoints, weights)``.
    """

    def __init__(self, system: System, params: ParamSet, *, kpts=None, scc: bool = True,
                 time_reversal: bool = True, **options):
        self.system = system
        self.params = params
        self.kpts = make_kpoints(kpts, time_reversal) if system.periodic else None
        self.options = dict(options)
        self.scc = scc
        self.result: DFTBResult | None = None

    # ------------------------------------------------------------------
    def _engine(self) -> DFTB:
        sk = self.params.skset(self.system.unique_labels)
        opts = dict(self.options)
        disp = make_dispersion(opts.pop("dispersion", None))
        if disp is not None:            # elements of the System, also for custom labels
            disp = disp.with_elements(dict(zip(self.system.labels, map(int, self.system.numbers))))
        return DFTB(sk, scc=self.scc, kpts=self.kpts, charge=self.system.charge, dispersion=disp,
                    **opts)

    def _run(self, forces: bool) -> DFTBResult:
        s = self.system
        q_init = None if self.result is None else self._site_populations()
        self.result = self._engine().calculate(s.labels, s.positions, s.cell if s.periodic else None,
                                               s.pbc if s.periodic else None, forces=forces,
                                               q_init=q_init)
        return self.result

    def _site_populations(self):
        """Restart charges from the previous run (the vector the SCC loop mixes)."""
        r = self.result
        if r.q_spin is not None:
            return r.q_spin.ravel()
        return r.q_shell if self.params.shell_resolved else r.charges

    def kernel(self) -> float:
        """Run the SCC cycle; returns the total energy (Hartree)."""
        return self._run(forces=False).energy

    def forces(self) -> np.ndarray:
        """Analytic forces (Hartree/Bohr), restarting from converged charges."""
        res = self.result
        if res is None or res.forces is None:
            res = self._run(forces=True)
        return res.forces

    # ------------------------------------------------------------------
    @property
    def e_tot(self) -> float:
        return self._need().energy

    @property
    def e_free(self) -> float:
        return self._need().free_energy

    @property
    def mulliken(self) -> np.ndarray:
        """Net Mulliken charges per atom (e), positive = electron deficit."""
        r = self._need()
        sk = self.params.skset(self.system.unique_labels)
        q0 = np.array([sk.occupation[lab].sum() for lab in self.system.labels])
        return q0 - r.charges

    @property
    def mo_energy(self):
        return self._need().eigenvalues

    @property
    def fermi(self) -> float:
        return self._need().fermi

    @property
    def converged(self) -> bool:
        return self._need().converged

    def _need(self) -> DFTBResult:
        if self.result is None:
            raise RuntimeError("call kernel() first")
        return self.result
