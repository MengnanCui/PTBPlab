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

from .bands import States, eigenstates
from .calculator import DFTB, DFTBResult
from .dispersion import make_dispersion
from .kpoints import monkhorst_pack, reduce_time_reversal
from .params import ParamSet
from .system import System


def two_component(options: dict) -> bool:
    """Whether :class:`pydftb.DFTB` options ask for a two-component calculation (non-collinear
    spin or spin-orbit coupling, milestone M6a)."""
    spin = options.get("spin")
    return ((isinstance(spin, str) and spin.lower() in ("noncolinear", "noncollinear"))
            or options.get("spin_orbit") is not None)


def block_terms(options: dict) -> bool:
    """Whether :class:`pydftb.DFTB` options ask for block potentials (DFTB+U, onsite
    corrections; milestone M6b)."""
    return options.get("plus_u") is not None or options.get("onsite_corrections") is not None


def make_kpoints(kpts, time_reversal: bool = True):
    """None | (n1, n2, n3) Monkhorst-Pack grid | (kpts, weights) | path points -> (kpts, weights)
    or None.

    Path points, an (n_k, 3) NumPy array or an ASE ``BandPath``, keep their order and get
    equal weights (never time-reversal reduced).
    """
    if kpts is None:
        return None
    if hasattr(kpts, "special_points"):            # ase BandPath
        kpts = np.asarray(kpts.kpts, float)
    if isinstance(kpts, np.ndarray) and kpts.ndim == 2:
        return kpts.astype(float), np.full(len(kpts), 1.0 / len(kpts))
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
    or explicit ``(kpoints, weights)``.  ``time_reversal`` (default None: yes, except for
    two-component calculations, which need the full grid) merges k and -k.
    """

    def __init__(self, system: System, params: ParamSet, *, kpts=None, scc: bool = True,
                 time_reversal: bool | None = None, **options):
        self.system = system
        self.params = params
        if time_reversal is None:
            time_reversal = not two_component(options)
        self.time_reversal = time_reversal
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
        if block_terms(self.options):          # DFTB+U / onsite: the block vector (M6b)
            return r.q_in
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

    def states(self, kpts=None, projections: bool = False, time_reversal: bool | None = None,
               chunk: int | None = None) -> States:
        """Eigenstates at the converged charges (DFTB+'s band-structure run).

        ``kpts``: None (the SCC k-points), a Monkhorst-Pack grid ``(n1, n2, n3)``,
        ``(kpoints, weights)``, or path points (an (n_k, 3) array or an ASE ``BandPath``).
        ``projections`` keeps the orbital shares of every state (PDOS).  See
        :class:`pydftb.bands.States` for DOS, PDOS and band-structure plots.
        """
        r = self._need()
        s = self.system
        if kpts is not None and not s.periodic:
            raise ValueError("k-points need a periodic system")
        if time_reversal is None:
            time_reversal = self.time_reversal
        k = None if kpts is None else make_kpoints(kpts, time_reversal)
        return eigenstates(self._engine(), s.labels, s.positions, s.cell if s.periodic else None,
                           s.pbc if s.periodic else None, r.q_in, kpts=k,
                           projections=projections, chunk=chunk,
                           scc_fermi=r.fermi_spin if r.fermi_spin is not None else r.fermi)

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
        if r.q0_atom is not None:              # custom reference occupations (M6f)
            return r.q0_atom - r.charges
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
