"""DFTB engine selection for PTBP: DFTB+ (through ASE) or pydftb.

Every DFTB calculation in PTBP goes through :func:`make_calculator`
(``utils.calculator.dftbplus_calculator`` is kept as an alias).  The engine
and the electronic smearing are process-wide settings, set once by
``cli/run.py`` via :func:`configure`; forked worker processes inherit them.

What PTBP's DFTB+ runs really use (ASE writes ``ParserVersion = 1``, which
makes DFTB+ apply its legacy defaults; see ``pydftb/docs/design/M1-ptbp-integration.html``):

* shell-resolved SCC (``OrbitalResolvedSCC = Yes`` injected by DFTB+),
* Fermi filling at T = 0 -- the ``Fermi``/``MethfesselPaxton`` blocks written
  by PTBP sit outside ``Filling`` and are silently ignored,
* finite-difference H0/S derivatives with a 0.01 Bohr step,
* ``PolynomialRepulsive = Yes`` with all-zero coefficients, i.e. no
  repulsive energy (PTBP adds it later with physrep).

The pydftb engine reproduces exactly these settings by default
(``pydftb_forces="dftbplus"``); ``pydftb_forces="exact"`` uses exact
derivatives instead of DFTB+'s 0.01 Bohr finite differences.

``smearing`` other than ``"none"`` switches on a real electronic smearing in
*both* engines (for DFTB+ the ``Filling`` block is written correctly and the
temperature/order actually used is checked in ``dftb_pin.hsd``).
"""
from __future__ import annotations

import os
import re
from typing import Any, Mapping

from ase import Atoms

ENGINES = ("dftbplus", "pydftb")
SMEARINGS = ("none", "fermi", "gaussian", "mp")
PYDFTB_FORCES = ("dftbplus", "exact")
KB_HARTREE = 3.16681534524639e-6      # Boltzmann constant in Hartree/K (DFTB+, = pydftb's)
LEGACY_FD_DELTA = 1.0e-2               # DFTB+ finite-difference step for ParserVersion 1

_SETTINGS: dict[str, Any] = {
    "engine": "dftbplus",
    "smearing": "none",               # none = what PTBP has always effectively used (T = 0)
    "smearing_temperature": 2000.0,   # K, used only if smearing != none
    "mp_order": 2,
    "pydftb_forces": "dftbplus",
}


def configure(**kwargs) -> dict:
    """Set process-wide engine options (``engine``, ``smearing``,
    ``smearing_temperature``, ``mp_order``, ``pydftb_forces``)."""
    for key, value in kwargs.items():
        if value is None:
            continue
        if key not in _SETTINGS:
            raise KeyError(f"unknown engine option {key!r}")
        _SETTINGS[key] = value
    if _SETTINGS["engine"] not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {_SETTINGS['engine']!r}")
    if _SETTINGS["smearing"] not in SMEARINGS:
        raise ValueError(f"smearing must be one of {SMEARINGS}, got {_SETTINGS['smearing']!r}")
    if _SETTINGS["pydftb_forces"] not in PYDFTB_FORCES:
        raise ValueError(f"pydftb_forces must be one of {PYDFTB_FORCES}")
    if _SETTINGS["smearing"] == "mp" and int(_SETTINGS["mp_order"]) < 1:
        raise ValueError("Methfessel-Paxton order must be >= 1 (use smearing='gaussian' for 0)")
    return dict(_SETTINGS)


def settings() -> dict:
    return dict(_SETTINGS)


def make_calculator(mole: Atoms, dftb_args: Mapping[str, Any], engine: str | None = None):
    """ASE calculator for one structure.

    ``dftb_args`` keeps PTBP's schema: ``calc_type`` ('static', 'band', 'xtb'),
    ``SCC`` ('Yes'/'No'), ``poly_rep`` ('Yes'/'No'), ``kpts`` (Monkhorst-Pack
    grid ``[n1, n2, n3]`` or an ASE k-path), ``slater_p`` (SKF directory).
    Contract (both engines): energy = DFTB+ "Total energy" (eV), forces (eV/A);
    FileNotFoundError for a missing SKF; results are returned even if the SCC
    does not converge (``ConvergentSCCOnly = No``).
    """
    engine = engine or dftb_args.get("engine") or _SETTINGS["engine"]
    if engine == "dftbplus":
        from utils.calculator import _dftbplus_ase_calculator
        return _dftbplus_ase_calculator(mole, dftb_args, filling=dftbplus_filling_kwargs())
    if engine == "pydftb":
        return _pydftb_calculator(mole, dftb_args)
    raise ValueError(f"unknown engine {engine!r}; choose from {ENGINES}")


# ----------------------------------------------------------------------------
# DFTB+ smearing (ASE keyword form)
# ----------------------------------------------------------------------------
def dftbplus_filling_kwargs() -> dict | None:
    """ASE ``Dftb`` keywords for a correctly placed ``Filling`` block, or None
    to keep PTBP's historical input untouched (smearing='none').

    ASE writes ParserVersion 1; for ParserVersion < 14 DFTB+ replaces a
    Methfessel-Paxton ``Order = n`` by ``n - 1`` (``oldcompat.F90``), so n + 1
    is written to obtain the requested order.  The temperature is given in
    Hartree (no unit modifier).
    """
    kind = _SETTINGS["smearing"]
    if kind == "none":
        return None
    kt = float(_SETTINGS["smearing_temperature"]) * KB_HARTREE
    if kind == "fermi":
        return {"Hamiltonian_Filling_": "Fermi", "Hamiltonian_Filling_Temperature": kt}
    if kind == "gaussian":
        return {"Hamiltonian_Filling_": "Gaussian", "Hamiltonian_Filling_Temperature": kt}
    order = int(_SETTINGS["mp_order"])
    return {"Hamiltonian_Filling_": "MethfesselPaxton",
            "Hamiltonian_Filling_Order": order + 1,           # ParserVersion 1: DFTB+ subtracts 1
            "Hamiltonian_Filling_Temperature": kt}


def expected_filling() -> tuple[str, float, int | None] | None:
    """(kind, kT in Hartree, MP order) that DFTB+ should report, or None."""
    kind = _SETTINGS["smearing"]
    if kind == "none":
        return None
    kt = float(_SETTINGS["smearing_temperature"]) * KB_HARTREE
    name = {"fermi": "Fermi", "gaussian": "Gaussian", "mp": "MethfesselPaxton"}[kind]
    return name, kt, (int(_SETTINGS["mp_order"]) if kind == "mp" else None)


def check_dftbplus_filling(directory: str, expected) -> None:
    """Verify the filling DFTB+ actually used, from its processed ``dftb_pin.hsd``."""
    path = os.path.join(directory or ".", "dftb_pin.hsd")
    with open(path) as fh:
        text = fh.read()
    m = re.search(r"Filling\s*=\s*(\w+)\s*\{(.*?)\}", text, re.S)
    if m is None:
        raise RuntimeError(f"no Filling block in {path}")
    kind, body = m.group(1), m.group(2)
    temp = float(re.search(r"Temperature\s*=\s*(\S+)", body).group(1))
    order = re.search(r"Order\s*=\s*(\d+)", body)
    exp_kind, exp_kt, exp_order = expected
    ok = kind == exp_kind and abs(temp - exp_kt) <= 1e-12 + 1e-9 * exp_kt
    if exp_order is not None:
        # DFTB+ stores the order it will really use after the legacy correction
        ok = ok and order is not None and int(order.group(1)) == exp_order
    if not ok:
        raise RuntimeError(f"DFTB+ used Filling = {kind} {{{body.strip()}}} but "
                           f"{exp_kind} kT={exp_kt:.3e} Ha order={exp_order} was requested")


# ----------------------------------------------------------------------------
# pydftb
# ----------------------------------------------------------------------------
def pydftb_smearing():
    kind = _SETTINGS["smearing"]
    temp = float(_SETTINGS["smearing_temperature"])
    if kind == "none":
        return ("fermi", 0.0)
    if kind == "mp":
        return ("mp", temp, int(_SETTINGS["mp_order"]))
    return (kind, temp)


def _pydftb_calculator(mole: Atoms, dftb_args: Mapping[str, Any]):
    from pydftb.ase_calc import PyDFTB

    calc_type = dftb_args.get("calc_type", "static")
    if calc_type == "xtb":
        raise NotImplementedError("xTB is not available in pydftb; use engine='dftbplus'")
    if calc_type not in ("static", "band"):
        raise ValueError(f"unknown calc_type {calc_type!r}")
    kpts = dftb_args.get("kpts")
    kspec = None
    if kpts is not None:
        if hasattr(kpts, "kpts"):                       # ASE BandPath
            kspec = {"kpts": kpts.kpts}
        elif len(kpts) == 3 and all(int(k) == k for k in kpts):
            kspec = tuple(int(k) for k in kpts)         # Monkhorst-Pack grid
        else:
            kspec = {"kpts": kpts}
    fd_delta = LEGACY_FD_DELTA if _SETTINGS["pydftb_forces"] == "dftbplus" else None
    return PyDFTB(dftb_args.get("slater_p", "./"),
                  kpts=kspec, time_reversal=not isinstance(kspec, dict),
                  smearing=pydftb_smearing(),
                  scc=str(dftb_args.get("SCC", "Yes")).lower() == "yes",
                  scc_tol=1e-5, max_scc=1000,
                  repulsive="polynomial" if str(dftb_args.get("poly_rep", "No")).lower() == "yes"
                  else "skf",
                  shell_resolved=True,          # DFTB+ under ASE (ParserVersion 1) default
                  fd_delta=fd_delta, threads=1)
