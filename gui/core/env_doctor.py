"""Dependency / calculator environment probe for the Calculator-settings view.

Reports what's installed so the GUI can tell the user *why* a Run button is
disabled (e.g. "DFTB+ not found") without crashing. Everything here degrades
gracefully: a missing package yields a `False` status, never an exception.
"""
from __future__ import annotations

import importlib
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Probe:
    name: str
    ok: bool
    detail: str
    needed_for: str


# (import name, human label, what it unlocks)
_MODULE_PROBES = [
    ("ase", "ASE", "structures, all workflows"),
    ("numpy", "NumPy", "everything"),
    ("scipy", "SciPy", "optimisation inner loop"),
    ("matplotlib", "matplotlib", "plots"),
    ("yaml", "PyYAML", "config save/load"),
    ("skopt", "scikit-optimize", "optimizer: bo / parallel_bo"),
    ("pyswarms", "pyswarms", "optimizer: pso"),
    ("customtkinter", "customtkinter", "the GUI itself"),
    ("hotcent", "hotcent", "SKF band part + optimisation"),
    ("pylibxc", "pylibxc", "XC evaluation for SKF generation"),
]


def _module_version(mod) -> str:
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return "installed"


def probe_modules() -> List[Probe]:
    out: List[Probe] = []
    for import_name, label, needed_for in _MODULE_PROBES:
        try:
            mod = importlib.import_module(import_name)
            out.append(Probe(label, True, _module_version(mod), needed_for))
        except Exception as e:  # ImportError or a broken C-extension import
            out.append(Probe(label, False, f"missing ({type(e).__name__})", needed_for))
    return out


def probe_dftb() -> Probe:
    """Locate the DFTB+ binary: explicit ASE_DFTB_COMMAND wins, else PATH."""
    cmd = os.environ.get("ASE_DFTB_COMMAND")
    if cmd:
        exe = cmd.split()[0] if cmd.split() else cmd
        resolved = shutil.which(exe) or (exe if os.path.exists(exe) else None)
        if resolved:
            return Probe("DFTB+", True, f"ASE_DFTB_COMMAND → {resolved}",
                         "optimisation (energy/forces/bands)")
        return Probe("DFTB+", False,
                     f"ASE_DFTB_COMMAND set but not found: {cmd}",
                     "optimisation (energy/forces/bands)")
    found = shutil.which("dftb+")
    if found:
        return Probe("DFTB+", True, f"on PATH → {found}",
                     "optimisation (energy/forces/bands)")
    return Probe("DFTB+", False, "not found (set ASE_DFTB_COMMAND or add to PATH)",
                 "optimisation (energy/forces/bands)")


@dataclass
class Readiness:
    probes: List[Probe] = field(default_factory=list)

    def by_name(self, name: str) -> Optional[Probe]:
        for p in self.probes:
            if p.name == name:
                return p
        return None

    def _ok(self, name: str) -> bool:
        p = self.by_name(name)
        return bool(p and p.ok)

    @property
    def structures_ready(self) -> bool:
        return self._ok("ASE") and self._ok("NumPy")

    @property
    def skf_ready(self) -> bool:
        return self._ok("hotcent") and self._ok("pylibxc")

    @property
    def optimize_bo_ready(self) -> bool:
        return self.skf_ready and self._ok("DFTB+") and self._ok("scikit-optimize")

    @property
    def optimize_pso_ready(self) -> bool:
        return self.skf_ready and self._ok("DFTB+") and self._ok("pyswarms")


def check() -> Readiness:
    """Run every probe and return a Readiness summary (never raises)."""
    probes = probe_modules()
    probes.append(probe_dftb())
    return Readiness(probes=probes)
