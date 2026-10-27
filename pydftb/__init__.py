"""pydftb: a from-scratch, readable SCC-DFTB engine.

Units: everything inside the engine is in atomic units (Bohr, Hartree).
The ASE wrapper (``pydftb.ase_calc.PyDFTB``) converts to/from eV and Angstrom.

High-level (PySCF-style) entry points::

    import pydftb
    params = pydftb.ParamSet("path/to/skf")
    mf = pydftb.SCC(pydftb.System.from_ase(atoms), params, kpts=(4, 4, 4))
    e = mf.kernel(); f = mf.forces()

Design notes, the DFTB+ source map and milestone reports live in
``pydftb/docs/``.
"""
from .analysis import Analysis
from .api import SCC, make_kpoints
from .calculator import DFTB, DFTBResult
from .hybrid import Hybrid
from .params import ParamSet
from .skf import SKFile, SKSet, max_l_from_skf
from .system import System
from .typeswitch import TypeSwitch

__all__ = ["SCC", "make_kpoints", "DFTB", "DFTBResult", "ParamSet", "SKFile", "SKSet",
           "max_l_from_skf", "System", "Analysis", "TypeSwitch", "Hybrid"]
__version__ = "0.1.0.dev0"
