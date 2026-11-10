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

``pydftb_backend``/``pydftb_device``/``pydftb_dtype`` (M4d) put pydftb's SCC linear algebra
on PyTorch, on the CPU or a GPU (``pydftb.backend.devices``); the default NumPy/CPU/float64
leaves every calculation as before.  On a GPU the DFTB stage runs in this process, one
structure after the other (:func:`is_gpu`).

Several types of one element (milestone M5b, :func:`set_type_switching`): pydftb mixes them
by the atoms' coordination numbers (pydftb's M5a); the atoms keep their element symbols.  DFTB+
through ASE writes the element symbols only, so it refuses such runs.  Without switches every
calculator is created exactly as before.

Alignment with the DFT reference (A series, :func:`set_alignment`; all off by default, and then
every calculator is created exactly as before): ``kspacing`` gives every periodic structure
VASP's Gamma-centred KSPACING grid instead of PTBP's k-point density rule (both engines; the
isolated atoms of the loss keep their Gamma point); ``energy_kind="sigma0"`` makes the energy
Total - TS/2 (VASP's energy(sigma->0), the A-series scans' energy); ``third_order`` with
``hubbard_derivs`` switches on DFTB3, ``hubbard`` replaces Hubbard values (the last three:
pydftb only).  Smearing as the scans: ``configure(smearing="gaussian",
smearing_temperature=0.05 eV / k_B)``.
"""
from __future__ import annotations

import os
import re
from typing import Any, Mapping

from ase import Atoms

ENGINES = ("dftbplus", "pydftb")
SMEARINGS = ("none", "fermi", "gaussian", "mp")
PYDFTB_FORCES = ("dftbplus", "exact")
PYDFTB_BACKENDS = ("numpy", "torch")
PYDFTB_DTYPES = ("float64", "float32")
KB_HARTREE = 3.16681534524639e-6      # Boltzmann constant in Hartree/K (DFTB+, = pydftb's)
LEGACY_FD_DELTA = 1.0e-2               # DFTB+ finite-difference step for ParserVersion 1

_SETTINGS: dict[str, Any] = {
    "engine": "dftbplus",
    "smearing": "none",               # none = what PTBP has always effectively used (T = 0)
    "smearing_temperature": 2000.0,   # K, used only if smearing != none
    "mp_order": 2,
    "pydftb_forces": "dftbplus",
    "pydftb_backend": "numpy",        # M4d: numpy | torch
    "pydftb_device": "cpu",           # torch device: cpu, cuda[:i], mps, xpu (auto: run.py)
    "pydftb_dtype": "float64",
    # MaxAngularMomentum of every element on both engines: "occupied" (default) = the highest
    # shell occupied in the neutral atom (ASE's read_max_angular_momentum), the basis PTBP's
    # parameters are fitted with; "highest" = every shell the SKF has integrals for
    # (pydftb.skf.max_l_available), e.g. the d of S, I, Sn and the p of Li
    "max_l": "occupied",
}
MAX_L_RULES = ("highest", "occupied")


def configure(**kwargs) -> dict:
    """Set process-wide engine options (``engine``, ``smearing``,
    ``smearing_temperature``, ``mp_order``, ``pydftb_forces``, ``pydftb_backend``,
    ``pydftb_device``, ``pydftb_dtype``, ``max_l``)."""
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
    if _SETTINGS["max_l"] not in MAX_L_RULES:
        raise ValueError(f"max_l must be one of {MAX_L_RULES}, got {_SETTINGS['max_l']!r}")
    if _SETTINGS["smearing"] == "mp" and int(_SETTINGS["mp_order"]) < 1:
        raise ValueError("Methfessel-Paxton order must be >= 1 (use smearing='gaussian' for 0)")
    _check_pydftb_backend()
    return dict(_SETTINGS)


def _check_pydftb_backend() -> None:
    from pydftb.backend.devices import kind_of
    backend, device = _SETTINGS["pydftb_backend"], str(_SETTINGS["pydftb_device"])
    if backend not in PYDFTB_BACKENDS:
        raise ValueError(f"pydftb_backend must be one of {PYDFTB_BACKENDS}, got {backend!r}")
    if _SETTINGS["pydftb_dtype"] not in PYDFTB_DTYPES:
        raise ValueError(f"pydftb_dtype must be one of {PYDFTB_DTYPES}")
    if device != "auto":
        kind_of(device)                               # cpu, cuda[:i], mps, xpu
    if backend == "numpy" and (device != "cpu" or _SETTINGS["pydftb_dtype"] != "float64"):
        raise ValueError("--device other than cpu and --dtype float32 need "
                         "--pydftb_backend torch")
    if backend == "torch" and _SETTINGS["engine"] != "pydftb":
        raise ValueError("--pydftb_backend torch needs --engine pydftb")
    if device.split(":")[0] == "mps" and _SETTINGS["pydftb_dtype"] == "float64":
        raise ValueError("Apple's MPS has no float64: use --dtype float32 with --device mps")


def is_gpu() -> bool:
    """pydftb on a torch device other than the CPU: the DFTB stage then runs in this
    process (a process that uses CUDA cannot hand it to forked workers)."""
    from pydftb.backend.devices import kind_of
    device = str(_SETTINGS["pydftb_device"])
    return (_SETTINGS["engine"] == "pydftb" and _SETTINGS["pydftb_backend"] == "torch"
            and device != "auto" and kind_of(device) != "cpu")


def settings() -> dict:
    return dict(_SETTINGS)


# type switching (M5b): process-wide like the settings (forked workers inherit it), kept out of
# _SETTINGS so that settings() and what is derived from it do not change without switches
_TYPE_SWITCHING: tuple = ()
DFTBPLUS_SWITCHING_ERROR = (
    "several types of one element (type switching, M5b) need --engine pydftb: DFTB+ through ASE "
    "writes only element symbols and gives every atom one fixed type; parameters per element "
    "without declared types work with both engines")


def set_type_switching(switches=()) -> tuple:
    """Set the type switches (``pydftb.typeswitch.TypeSwitch`` objects or dicts) of every pydftb
    calculation of this process; () switches them off."""
    global _TYPE_SWITCHING
    if switches:
        from pydftb.typeswitch import check_switches
        _TYPE_SWITCHING = tuple(check_switches(list(switches)))
    else:
        _TYPE_SWITCHING = ()
    return _TYPE_SWITCHING


def type_switching() -> tuple:
    return _TYPE_SWITCHING


# ----------------------------------------------------------------------------
# Alignment with the DFT reference (A series, A0 plan section 8).  Process-wide like the
# settings (forked workers inherit them) and, like the type switches, kept out of _SETTINGS:
# settings() and everything derived from it (plan.json, the advisor) stay as they were, and
# every calculator is created exactly as before unless an option is set.
# ----------------------------------------------------------------------------
ENERGY_KINDS = ("total", "sigma0")
THIRD_ORDERS = (None, "full", "onsite")
_ALIGN_DEFAULTS = {
    "kspacing": None,          # 1/A: Gamma-centred grid N_i = max(1, ceil(2 pi |b_i| / KSPACING))
    "energy_kind": "total",    # total = DFTB+ "Total energy"; sigma0 = Total - TS/2 (VASP's
                               #   energy(sigma->0), the A-series scans' E), pydftb only
    "third_order": None,       # DFTB3: "full" (ThirdOrderFull) or "onsite", pydftb only
    "hubbard_derivs": None,    # {element: Ud (Hartree/e)} for third_order
    "hubbard": None,           # {element: U (Hartree)}: CustomisedHubbards, pydftb only
}
_ALIGN: dict[str, Any] = dict(_ALIGN_DEFAULTS)
ALIGNMENT_ENGINE_ERROR = ("{name} is a pydftb option (DFTB+ through ASE would need other input "
                          "keywords): use --engine pydftb")


def set_alignment(**kwargs) -> dict:
    """Set the DFT-alignment options (``kspacing``, ``energy_kind``, ``third_order``,
    ``hubbard_derivs``, ``hubbard``); None keeps an option's current value.  Returns them."""
    for key, value in kwargs.items():
        if key not in _ALIGN:
            raise KeyError(f"unknown alignment option {key!r}")
        if value is not None:
            _ALIGN[key] = value
    ks = _ALIGN["kspacing"]
    if ks is not None and not float(ks) > 0.0:
        raise ValueError(f"kspacing must be > 0 (1/A), got {ks!r}")
    if _ALIGN["energy_kind"] not in ENERGY_KINDS:
        raise ValueError(f"energy_kind must be one of {ENERGY_KINDS}, got {_ALIGN['energy_kind']!r}")
    if _ALIGN["third_order"] not in THIRD_ORDERS:
        raise ValueError(f"third_order must be one of {THIRD_ORDERS}, got {_ALIGN['third_order']!r}")
    if _ALIGN["third_order"] and not _ALIGN["hubbard_derivs"]:
        raise ValueError("third_order needs hubbard_derivs ({element: Ud in Hartree/e})")
    for key in ("hubbard_derivs", "hubbard"):
        if _ALIGN[key] is not None:
            _ALIGN[key] = {str(k): float(v) for k, v in dict(_ALIGN[key]).items()}
    return dict(_ALIGN)


def reset_alignment() -> dict:
    """Every alignment option back to its default (PTBP's behaviour)."""
    _ALIGN.clear()
    _ALIGN.update(_ALIGN_DEFAULTS)
    return dict(_ALIGN)


def alignment() -> dict:
    return dict(_ALIGN)


def aligned() -> bool:
    """Whether some alignment option differs from its default."""
    return any(_ALIGN[k] != v for k, v in _ALIGN_DEFAULTS.items())


def kspacing_grid(atoms: Atoms, kspacing: float) -> list:
    """VASP's KSPACING rule (Gamma-centred): N_i = max(1, ceil(2 pi |b_i| / KSPACING)) with
    the reciprocal vectors b_i of the cell (1/A, without 2 pi), 1 along non-periodic
    directions; as ``utils.accuracy.kpoints_for`` (copied: no scan import here)."""
    import numpy as np
    rec = atoms.cell.reciprocal()
    return [max(1, int(np.ceil(2 * np.pi * np.linalg.norm(b) / float(kspacing) - 1e-9)))
            if atoms.pbc[i] else 1 for i, b in enumerate(rec)]


def _aligned_kpts(mole: Atoms, dftb_args: Mapping[str, Any]):
    """dftb_args with the KSPACING grid, Gamma-centred, instead of PTBP's density grid
    (periodic structures with a grid; band paths, molecules and ``kpts_fixed`` calls keep
    theirs)."""
    ks = _ALIGN["kspacing"]
    kpts = dftb_args.get("kpts")
    if ks is None or dftb_args.get("kpts_fixed") or not any(mole.pbc):
        return dftb_args
    if kpts is not None and (hasattr(kpts, "kpts") or isinstance(kpts, dict)
                             or len(kpts) != 3):
        return dftb_args                               # a path or explicit k-points
    return dict(dftb_args, kpts={"size": kspacing_grid(mole, ks), "gamma": True})


def pydftb_alignment_kwargs() -> dict:
    """PyDFTB keywords of the DFTB3 and Hubbard alignment options ({} by default)."""
    out = {}
    if _ALIGN["third_order"]:
        out["third_order"] = _ALIGN["third_order"]
        out["hubbard_derivs"] = dict(_ALIGN["hubbard_derivs"])
    if _ALIGN["hubbard"]:
        out["hubbard"] = dict(_ALIGN["hubbard"])
    return out


def check_alignment_engine(engine: str) -> None:
    """The pydftb-only alignment options refuse the DFTB+ engine."""
    if engine == "pydftb":
        return
    for key, name in (("energy_kind", "--energy-kind sigma0"), ("third_order", "--third-order"),
                      ("hubbard", "--hubbard")):
        if _ALIGN[key] != _ALIGN_DEFAULTS[key]:
            raise ValueError(ALIGNMENT_ENGINE_ERROR.format(name=name))


def make_calculator(mole: Atoms, dftb_args: Mapping[str, Any], engine: str | None = None):
    """ASE calculator for one structure.

    ``dftb_args`` keeps PTBP's schema: ``calc_type`` ('static', 'band', 'xtb'),
    ``SCC`` ('Yes'/'No'), ``poly_rep`` ('Yes'/'No'), ``kpts`` (Monkhorst-Pack
    grid ``[n1, n2, n3]`` or an ASE k-path), ``slater_p`` (SKF directory), optionally
    ``directory`` (DFTB+: run directory, default the current one) and ``k_threads``
    (pydftb: threads over k-points, bitwise identical results).
    Contract (both engines): energy = DFTB+ "Total energy" (eV), forces (eV/A);
    FileNotFoundError for a missing SKF; results are returned even if the SCC
    does not converge (``ConvergentSCCOnly = No``: the loss treats such structures; pydftb's
    own default is to stop, so the pydftb engine passes ``convergent_scc_only=False``).
    Both engines use the ``max_l`` setting for MaxAngularMomentum (default: the occupied
    shells, PTBP's fitting basis), written out explicitly for DFTB+.
    """
    engine = engine or dftb_args.get("engine") or _SETTINGS["engine"]
    if _ALIGN["kspacing"] is not None:               # A series: the DFT's k-point grid
        dftb_args = _aligned_kpts(mole, dftb_args)
    if engine == "dftbplus":
        if _TYPE_SWITCHING:
            raise ValueError(DFTBPLUS_SWITCHING_ERROR)
        check_alignment_engine(engine)
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
        elif isinstance(kpts, dict) and "size" in kpts:  # a Gamma-centred grid (A series)
            kspec = {"size": [int(k) for k in kpts["size"]], "gamma": bool(kpts.get("gamma"))}
        elif len(kpts) == 3 and all(int(k) == k for k in kpts):
            kspec = tuple(int(k) for k in kpts)         # Monkhorst-Pack grid
        else:
            kspec = {"kpts": kpts}
    fd_delta = LEGACY_FD_DELTA if _SETTINGS["pydftb_forces"] == "dftbplus" else None
    backend = {}
    if _SETTINGS["pydftb_backend"] != "numpy":       # M4d; the default calls stay as they were
        backend = dict(backend=_SETTINGS["pydftb_backend"], device=_SETTINGS["pydftb_device"],
                       dtype=_SETTINGS["pydftb_dtype"])
    if _TYPE_SWITCHING:                              # M5b; without switches the call is as before
        backend["type_switching"] = list(_TYPE_SWITCHING)
    time_reversal = not isinstance(kspec, dict)
    cls = PyDFTB
    if aligned():                                    # A series; by default the call is as before
        if isinstance(kspec, dict) and "size" in kspec:
            time_reversal = None                     # a Gamma-centred grid: k and -k merged
        backend.update(pydftb_alignment_kwargs())
        if _ALIGN["energy_kind"] == "sigma0":
            cls = _sigma0_class()
    return cls(dftb_args.get("slater_p", "./"), max_l=_SETTINGS["max_l"],
                  convergent_scc_only=False, warn_unconverged=False,   # PTBP's contract
                  kpts=kspec, time_reversal=time_reversal,
                  smearing=pydftb_smearing(),
                  scc=str(dftb_args.get("SCC", "Yes")).lower() == "yes",
                  scc_tol=1e-5, max_scc=1000,
                  repulsive="polynomial" if str(dftb_args.get("poly_rep", "No")).lower() == "yes"
                  else "skf",
                  shell_resolved=True,          # DFTB+ under ASE (ParserVersion 1) default
                  fd_delta=fd_delta, threads=int(dftb_args.get("blas_threads", 1)),
                  k_threads=int(dftb_args.get("k_threads", 1)), **backend)


_SIGMA0_CLASS = None


def _sigma0_class():
    """PyDFTB whose ``energy`` is E(sigma -> 0) = Total energy - TS/2 (DFTB+'s "Extrapolated
    to 0", VASP's energy(sigma->0), the energy of the A-series scans); free energy and forces
    as before (the forces are the gradient of the free energy, as in VASP)."""
    global _SIGMA0_CLASS
    if _SIGMA0_CLASS is None:
        from ase import units

        from pydftb.ase_calc import PyDFTB

        class Sigma0PyDFTB(PyDFTB):
            def _store(self, r, sk, labels):
                super()._store(r, sk, labels)
                if not self.excited:
                    self.results["energy"] = r.e_zero * units.Hartree

        _SIGMA0_CLASS = Sigma0PyDFTB
    return _SIGMA0_CLASS
