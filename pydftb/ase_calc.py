"""ASE calculator wrapper: drop-in replacement for ase.calculators.dftb.Dftb
in workflows that only need energies, forces and charges."""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes, kptdensity2monkhorstpack

from .calculator import DFTB
from .constants import BOHR, HARTREE
from .kpoints import monkhorst_pack, reduce_time_reversal
from .skf import SKSet


class PyDFTB(Calculator):
    """
    Parameters
    ----------
    skdir : directory containing ``A-B.skf`` files
    max_l : {"C": 1, "H": 0, ...} maximal angular momentum per element
    kpts  : (n1, n2, n3) Monkhorst-Pack grid, or a k-point density (float)
    temperature : electronic temperature (K), Fermi smearing
    """
    implemented_properties = ["energy", "free_energy", "forces", "charges"]

    def __init__(self, skdir, max_l, kpts=None, temperature=0.0, scc=True,
                 scc_tol=1e-6, shell_resolved=False, solver="batched", **kwargs):
        super().__init__(**kwargs)
        self.skdir, self.max_l = skdir, dict(max_l)
        self.kpts_spec = kpts
        self.opts = dict(scc=scc, temperature=temperature, scc_tol=scc_tol, solver=solver)
        self.shell_resolved = shell_resolved
        self._sk = None

    def _skset(self, symbols):
        el = sorted(set(symbols))
        if self._sk is None or not set(el) <= set(self._sk.elements):
            self._sk = SKSet(self.skdir, el, {e: self.max_l[e] for e in el},
                             shell_resolved=self.shell_resolved)
        return self._sk

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        sym = a.get_chemical_symbols()
        periodic = a.pbc.any()
        kpts = None
        if periodic and self.kpts_spec is not None:
            grid = (kptdensity2monkhorstpack(a, self.kpts_spec, even=False)
                    if np.isscalar(self.kpts_spec) else self.kpts_spec)
            kpts = reduce_time_reversal(*monkhorst_pack(grid))
        calc = DFTB(self._skset(sym), kpts=kpts, **self.opts)
        r = calc.calculate(sym, a.positions / BOHR, a.cell.array / BOHR if periodic else None,
                           a.pbc if periodic else None, forces="forces" in properties)
        self.results["energy"] = r.energy * HARTREE
        self.results["free_energy"] = r.free_energy * HARTREE
        self.results["charges"] = np.array([self._sk.occupation[s].sum() for s in sym]) - r.charges
        if r.forces is not None:
            self.results["forces"] = r.forces * HARTREE / BOHR
