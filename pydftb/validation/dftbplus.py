"""Run DFTB+ on the same input as pydftb (reference generation and timing).

Only needed for validation; pydftb itself never calls DFTB+.

Notes on matching DFTB+ exactly:
* ``ParserVersion = 14`` is always written.  For older parser versions DFTB+
  silently rewrites ``MethfesselPaxton { Order = n }`` to ``n - 1``
  (``dftbplus/oldcompat.F90``), and ASE writes version 1.
* DFTB+ differentiates H0/S numerically.  Its default finite difference is
  used for the references: ``Differentiation = Richardson {}`` was tested and
  is *less* accurate for these tables (H2O force error 1.2e-5 vs 1.2e-6
  Ha/Bohr): the water's H-H distance lies 4e-6 Bohr from SK grid node 153,
  and steps that cross a node mix two pieces of the piecewise interpolation
  polynomial (the default step 1.22e-4 already does).  pydftb's exact
  derivatives agree with its own energy finite differences to ~2e-9.
* Energies come from ``detailed.out`` (10 decimals), everything else from
  ``results.tag`` (full precision).
* DFTB+'s default ``EwaldTolerance`` is 1e-9 (pydftb uses 1e-12): in a 24-atom
  periodic water box this alone gives 1.3e-7 Ha; with ``EwaldTolerance = 1e-12``
  the two codes agree to 6e-11 Ha (1.6e-9 before M2e, when pydftb still summed the
  short-range gamma further than DFTB+'s 1e-10 cut-off).  References keep DFTB+'s
  default, except the band-structure cases (M2e), whose eigenvalues are compared to
  1e-8 Ha.
* Band structures take two runs in one directory (:func:`run_bands`): the SCC run
  writes ``charges.bin`` (the last input charges); the band run reads them back
  (``ReadInitialCharges``) and does one iteration on ``Klines`` points.  For cells
  with more than one atom it needs ``ConvergentSCCOnly = No``, or DFTB+ stops with
  "SCC is NOT converged" (one atom per cell always "converges": its population is the
  electron count).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

DFTBPLUS = os.environ.get("DFTBPLUS", shutil.which("dftb+") or "/opt/mm/dftb/bin/dftb+")
# dptools sources (DFTB+ tools/dptools/src), for dp_dos when freezing DOS curves; dptools is
# LGPL and not part of pydftb, it is only run while regenerating references
DPTOOLS = os.environ.get("DPTOOLS")
PARSER_VERSION = 14


def available() -> bool:
    return bool(DFTBPLUS) and os.path.exists(DFTBPLUS)


def _f(x) -> str:
    """Shortest exact (round-trip) text form of a float."""
    return repr(float(x))


def write_gen(path, labels, pos_ang, cell_ang=None):
    """DFTB+ gen file; species names are the type labels (Type2FileNames)."""
    species = list(dict.fromkeys(labels))
    with open(path, "w") as f:
        f.write(f"{len(labels)} {'S' if cell_ang is not None else 'C'}\n")
        f.write(" ".join(species) + "\n")
        for k, (s, p) in enumerate(zip(labels, pos_ang)):
            f.write(f"{k + 1} {species.index(s) + 1} {_f(p[0])} {_f(p[1])} {_f(p[2])}\n")
        if cell_ang is not None:
            f.write("0.0 0.0 0.0\n")
            for v in cell_ang:
                f.write(f"{_f(v[0])} {_f(v[1])} {_f(v[2])}\n")


def filling_block(filling) -> str:
    """("fermi", T_K) | ("mp", T_K, order) | ("gaussian", T_K) -> HSD."""
    kind = filling[0].lower()
    temp = _f(filling[1])
    if kind == "fermi":
        return f"Fermi {{\n    Temperature [K] = {temp}\n  }}"
    if kind in ("mp", "methfesselpaxton"):
        return (f"MethfesselPaxton {{\n    Order = {int(filling[2])}\n"
                f"    Temperature [K] = {temp}\n  }}")
    if kind == "gaussian":
        # DFTB+ maps Gaussian to Methfessel-Paxton order 0 internally; "Order = 0" is rejected
        return f"Gaussian {{\n    Temperature [K] = {temp}\n  }}"
    raise ValueError(f"unknown filling {filling!r}")


def dftb3_block(opts, labels=None) -> str:
    """HSD lines for pydftb's third_order / hubbard_derivs / h_damping options.

    DFTB+ rejects HubbardDerivs entries for species that are not in the
    structure, so only ``labels`` (if given) are written.
    """
    if not opts:
        return ""
    lines = []
    order = opts.get("third_order")
    if order:
        lines.append("  ThirdOrderFull = Yes" if order == "full" else "  ThirdOrder = Yes")
    if opts.get("hubbard_derivs"):
        lines.append("  HubbardDerivs {")
        lines += [f"    {k} = {' '.join(_f(x) for x in np.atleast_1d(v))}"
                  for k, v in opts["hubbard_derivs"].items() if labels is None or k in labels]
        lines.append("  }")
    if opts.get("h_damping") is not None:
        lines.append(f"  HCorrection = Damping {{ Exponent = {_f(opts['h_damping'])} }}")
    return "\n".join(lines) + "\n"


def spin_block(opts, labels=None, shell_resolved_scc=False) -> str:
    """HSD for pydftb's spin options (unpaired_electrons, relax_total_spin, spin_constants).

    One number per type means DFTB+ ``ShellResolvedSpin = No`` (only allowed, and then
    written, with atom-resolved SCC); matrices are written row by row.
    """
    if not opts:
        return ""
    consts = {k: np.asarray(v, float) for k, v in opts["spin_constants"].items()
              if labels is None or k in labels}
    by_shell = shell_resolved_scc or any(v.size > 1 for v in consts.values())
    lines = ["  SpinPolarisation = Colinear {",
             f"    UnpairedElectrons = {_f(opts.get('unpaired_electrons', 0.0))}",
             f"    RelaxTotalSpin = {'Yes' if opts.get('relax_total_spin') else 'No'}",
             "  }", "  SpinConstants {"]
    if not shell_resolved_scc:
        lines.append(f"    ShellResolvedSpin = {'Yes' if by_shell else 'No'}")
    lines += [f"    {k} = {' '.join(_f(x) for x in v.ravel())}" for k, v in consts.items()]
    lines.append("  }")
    return "\n".join(lines) + "\n"


def spin_constants_block(consts, labels=None) -> str:
    """HSD ``SpinConstants`` alone (one W per type, ``ShellResolvedSpin = No``): triplet
    excitations of a spin-restricted ground state (M6d)."""
    if not consts:
        return ""
    lines = ["  SpinConstants {", "    ShellResolvedSpin = No"]
    lines += [f"    {k} = {_f(float(np.ravel(v)[0]))}" for k, v in consts.items()
              if labels is None or k in labels]
    return "\n".join(lines + ["  }"]) + "\n"


def noncolinear_block(opts, labels=None, shell_resolved_scc=False) -> str:
    """HSD for non-collinear spin (M6a): ``SpinPolarisation = NonColinear`` with every atom's
    initial magnetisation vector, and the spin constants as :func:`spin_block`."""
    if not opts:
        return ""
    rows = "\n".join("        " + " ".join(_f(x) for x in s) for s in opts["initial_spins"])
    consts = {k: np.asarray(v, float) for k, v in opts["spin_constants"].items()
              if labels is None or k in labels}
    by_shell = shell_resolved_scc or any(v.size > 1 for v in consts.values())
    lines = ["  SpinPolarisation = NonColinear {", "    InitialSpins {",
             "      AllAtomSpins = {", rows, "      }", "    }", "  }", "  SpinConstants {"]
    if not shell_resolved_scc:
        lines.append(f"    ShellResolvedSpin = {'Yes' if by_shell else 'No'}")
    lines += [f"    {k} = {' '.join(_f(x) for x in v.ravel())}" for k, v in consts.items()]
    lines.append("  }")
    return "\n".join(lines) + "\n"


def spinorbit_block(opts, labels) -> str:
    """HSD ``SpinOrbit`` (M6a): one constant per shell (Hartree) for every type in
    ``labels`` (DFTB+ needs all of them; types without an entry get zeros), ``Dual``."""
    if not opts:
        return ""
    lines = ["  SpinOrbit {", f"    Dual = {'Yes' if opts.get('dual', True) else 'No'}"]
    for k, nsh in labels.items():
        xi = opts["xi"].get(k, [0.0] * nsh)
        lines.append(f"    {k} [Hartree] = {{ {' '.join(_f(x) for x in xi)} }}")
    lines.append("  }")
    return "\n".join(lines) + "\n"


def orbital_potential_block(opts, shells) -> str:
    """HSD ``OrbitalPotential`` (DFTB+U, M6b) for ``opts`` = {"groups": {label: groups},
    "functional": "fll" | "psic"} (groups as pydftb's ``plus_u``), for the types in ``shells``
    ({label: angular momenta of its shells}); DFTB+ numbers shells from 1."""
    if not opts:
        return ""
    from ..blockpot import plus_u_pairs, shell_indices
    lines = ["  OrbitalPotential {", f"    Functional = {{{opts.get('functional', 'fll')}}}"]
    for lab, spec in opts["groups"].items():
        if lab not in shells:
            continue
        for shell_spec, uj in plus_u_pairs(spec):
            idx = shell_indices(shell_spec, shells[lab], lab)
            lines += [f"    {lab} {{", f"      Shells = {{{' '.join(str(i + 1) for i in idx)}}}",
                      f"      UJ = {_f(uj)}", "    }"]
    lines.append("  }")
    return "\n".join(lines) + "\n"


def onsite_block(onsite, labels) -> str:
    """HSD ``OnSiteCorrection`` (M6b): the f_uu and f_ud matrices of every type in ``labels``
    (DFTB+ needs all), as pydftb's ``onsite_corrections`` entries."""
    if not onsite:
        return ""
    lines = ["  OnSiteCorrection {"]
    for lab in labels:
        e = onsite[lab]
        uu, ud = (e["uu"], e["ud"]) if isinstance(e, dict) else e
        lines.append(f"    {lab}uu = {{{' '.join(_f(x) for x in np.ravel(uu))}}}")
        lines.append(f"    {lab}ud = {{{' '.join(_f(x) for x in np.ravel(ud))}}}")
    lines.append("  }")
    return "\n".join(lines) + "\n"


def electric_field_block(field, point_charges) -> str:
    """HSD ``ElectricField`` (M6c): a field {"strength": V/Angstrom, "direction": (3,)} and
    point charges {"positions": Angstrom, "charges": e, "blur": Angstrom}."""
    if not field and not point_charges:
        return ""
    lines = ["  ElectricField = {"]
    if field:
        lines += ["    External = {", f"      Strength [V/AA] = {_f(field['strength'])}",
                  f"      Direction = {' '.join(_f(x) for x in field['direction'])}", "    }"]
    if point_charges:
        lines += ["    PointCharges = {", "      CoordsAndCharges [Angstrom] = {"]
        for pos, q in zip(point_charges["positions"], point_charges["charges"]):
            lines.append("        " + " ".join(_f(x) for x in (*pos, q)))
        lines.append("      }")
        if point_charges.get("blur"):
            lines.append(f"      GaussianBlurWidth [Angstrom] = {_f(point_charges['blur'])}")
        lines.append("    }")
    lines.append("  }")
    return "\n".join(lines) + "\n"


def site_potential_block(sites) -> str:
    """HSD ``AtomSitePotential`` (M6c): {"gross"/"net": {atom (0-based): V (eV)}}."""
    if not sites:
        return ""
    lines = ["  AtomSitePotential = {"]
    for kind, name in (("gross", "Gross"), ("net", "Net")):
        site = sites.get(kind)
        if site:           # one block per kind (DFTB+ ignores a second block of a kind)
            lines += [f"    {name} = {{",
                      f"      Atoms = {' '.join(str(int(a) + 1) for a in site)}",
                      f"      Vext [eV] = {' '.join(_f(v) for v in site.values())}", "    }"]
    lines.append("  }")
    return "\n".join(lines) + "\n"


def _values(d) -> str:
    """HSD ``Values { X = ...; Y = ... }`` (one assignment per statement)."""
    return "Values { " + "; ".join(f"{k} = {_f(v)}" for k, v in d.items()) + " }"


def _sasa_lines(sasa, indent) -> list:
    out = [f"{indent}ProbeRadius = {_f(sasa['probe_radius'])}",
           f"{indent}SurfaceTension = {_values(sasa['surface_tension'])}"]
    for key, hsd in (("smoothing", "Smoothing"), ("tolerance", "Tolerance"),
                     ("offset", "Offset")):
        if key in sasa:
            out.append(f"{indent}{hsd} = {_f(sasa[key])}")
    if "angular_grid" in sasa:
        out.append(f"{indent}AngularGrid = {int(sasa['angular_grid'])}")
    return out


def solvation_block(spec, rescale: bool = True, data_dir: str = "") -> str:
    """HSD ``Solvation`` (M6c) for a solvation spec of :mod:`pydftb.validation.cases`
    (atomic units: Hartree, Bohr; temperatures in K)."""
    if not spec:
        return ""
    if spec["model"] == "sasa":
        lines = ["  Solvation = SASA {"] + _sasa_lines(spec, "    ")
    else:
        lines = ["  Solvation = GeneralisedBorn {"]
        if "param_file" in spec:
            lines.append(f'    ParamFile = "{os.path.join(data_dir, spec["param_file"])}"')
        if "solvent" in spec:
            sol = spec["solvent"]
            if isinstance(sol, str):
                lines.append(f'    Solvent = fromName {{ "{sol}" }}')
            else:
                lines += ["    Solvent = fromConstants {", f"      Epsilon = {_f(sol['epsilon'])}",
                          f"      MolecularMass [amu] = {_f(sol['molecular_mass'])}",
                          f"      Density [kg/l] = {_f(sol['density'])}", "    }"]
        for key, hsd in (("free_energy_shift", "FreeEnergyShift"), ("born_scale", "BornScale"),
                         ("born_offset", "BornOffset"), ("alpha", "Alpha")):
            if key in spec:
                lines.append(f"    {hsd} = {_f(spec[key])}")
        if "descreening" in spec:
            d = spec["descreening"]
            lines.append("    Descreening = " + ("Unity {}" if d == "unity" else _values(d)))
        if spec.get("kernel"):
            lines.append(f"    Kernel = {spec['kernel']}")
        if spec.get("alpb"):
            lines.append("    ALPB = Yes")
        if spec.get("state"):
            lines.append(f"    State = {spec['state']}")
        if "temperature" in spec:
            lines.append(f"    Temperature [K] = {_f(spec['temperature'])}")
        if spec.get("cm5") is not None:
            lines.append("    CM5 {}")
        if "sasa" in spec:
            lines += ["    SASA {"] + _sasa_lines(spec["sasa"], "      ") + ["    }"]
        if "hbond" in spec:
            lines += ["    HBondCorr = Yes", f"    HBondStrength = {_values(spec['hbond'])}"]
    if not rescale:
        lines.append("    RescaleSolvatedFields = No")
    lines.append("  }")
    return "\n".join(lines) + "\n"


def parse_angular_momenta(text) -> dict:
    """detailed.out's ``Electron``/``Orbital angular momentum (mu_B/hbar)`` tables (M6a):
    {"spin": (n_shell, 3), "orbital": (n_shell, 3)} as printed (8 decimals: -g/4 m and
    -<L> per shell, in atom and shell order), or {} if absent."""
    out = {}
    for key, title in (("spin", "Electron angular momentum"), ("orbital", "Orbital angular momentum")):
        m = re.search(rf"^{title} \(mu_B/hbar\)\n.*?\n((?:\s+\d+\s+\d+\s+\d+\s+\S+ :.*\n)+)",
                      text, re.M)
        if m:
            rows = [ln.split(":")[1].split() for ln in m.group(1).strip().splitlines()]
            out[key] = np.array(rows, float)
    return out


def _yes(flag) -> str:
    return "Yes" if flag else "No"


def dispersion_block(spec, labels) -> str:
    """HSD ``Dispersion = ...`` for a pydftb dispersion spec (``pydftb.dispersion``).

    Values are written in DFTB+'s default units (Bohr, Hartree).  Type labels that are
    not element symbols get ``AtomicNumbers`` (D3, D4) or explicit UFF values, because
    DFTB+ finds parameters by species name.
    """
    from ..dispersion import (DFTD3, DFTD4, LennardJones, SimpleDFTD3, SlaterKirkwood,
                              make_dispersion)
    from ..system import element_of_label

    model = make_dispersion(spec)
    if model is None:
        return ""
    uniq = list(dict.fromkeys(labels))
    plain = all(element_of_label(lab) == lab for lab in uniq)
    numbers = dict(zip(uniq, model.numbers(uniq)))
    atomic = "" if plain else ("    AtomicNumbers {\n" + "".join(
        f"      {lab} = {numbers[lab]}\n" for lab in uniq) + "    }\n")
    if isinstance(model, LennardJones):
        if model.parameters == "uff" and plain:
            body = "    Parameters = UFFParameters {}\n"
        else:
            dist, depth = model.site_parameters(uniq)
            body = "    Parameters {\n" + "".join(
                f"      {lab} {{\n        Distance = {_f(r)}\n        Energy = {_f(d)}\n      }}\n"
                for lab, r, d in zip(uniq, dist, depth)) + "    }\n"
        return f"  Dispersion = LennardJones {{\n{body}  }}\n"
    if isinstance(model, SlaterKirkwood):
        if model.polar_radius_charge is not None:
            rows = "".join("      " + " ".join(_f(x) for x in row) + "\n"
                           for row in model.polar_radius_charge)
            body = f"    PolarRadiusCharge = {{\n{rows}    }}\n"
        else:
            entries = "".join(
                f"      {lab} {{\n        CovalentRadius = {_f(rc)}\n"
                f"        HybridPolarisations = {{ {' '.join(_f(x) for x in vals)} }}\n      }}\n"
                for lab, (rc, vals) in model.hybrid_dependent_pol.items() if lab in uniq)
            body = f"    PolarRadiusCharge = HybridDependentPol {{\n{entries}    }}\n"
        return f"  Dispersion = SlaterKirkwood {{\n{body}  }}\n"
    if isinstance(model, SimpleDFTD3):
        return (f"  Dispersion = SimpleDftD3 {{\n    s6 = {_f(model.s6)}\n    s8 = {_f(model.s8)}\n"
                f"    s10 = {_f(model.s10)}\n    a1 = {_f(model.a1)}\n    a2 = {_f(model.a2)}\n"
                f"    WeightingFactor = {_f(model.weighting_factor)}\n"
                f"    CutoffInter = {_f(model.cutoff_inter)}\n"
                f"    CoordinationNumber = Exp {{ Cutoff = {_f(model.cutoff_cn)} }}\n  }}\n")
    if isinstance(model, DFTD3):
        if model.kind == "bj":
            damping = f"BeckeJohnson {{\n      a1 = {_f(model.a1)}\n      a2 = {_f(model.a2)}\n    }}"
        elif model.kind == "zero":
            damping = (f"ZeroDamping {{\n      sr6 = {_f(model.sr6)}\n"
                       f"      alpha6 = {_f(model.alpha6)}\n    }}")
        else:
            damping = (f"ModifiedZeroDamping {{\n      sr6 = {_f(model.sr6)}\n"
                       f"      beta = {_f(model.beta)}\n      alpha6 = {_f(model.alpha6)}\n    }}")
        return (f"  Dispersion = DftD3 {{\n    Damping = {damping}\n    s6 = {_f(model.s6)}\n"
                f"    s8 = {_f(model.s8)}\n    Threebody = {_yes(model.threebody)}\n"
                f"    HHRepulsion = {_yes(model.hhrepulsion)}\n    Cutoff = {_f(model.cutoff)}\n"
                f"    CutoffCN = {_f(model.cutoff_cn)}\n{atomic}  }}\n")
    if isinstance(model, DFTD4):
        cut = DFTD4.LIBRARY_CUTOFFS        # what pydftb (the dftd4 library) uses
        return (f"  Dispersion = DftD4 {{\n    s6 = {_f(model.s6)}\n    s8 = {_f(model.s8)}\n"
                f"    s9 = {_f(model.s9)}\n    s10 = {_f(model.s10)}\n    a1 = {_f(model.a1)}\n"
                f"    a2 = {_f(model.a2)}\n    alpha = {_f(model.alpha)}\n"
                f"    WeightingFactor = {_f(model.weighting_factor)}\n"
                f"    ChargeSteepness = {_f(model.charge_steepness)}\n"
                f"    ChargeScale = {_f(model.charge_scale)}\n"
                f"    CutoffInter = {_f(cut['cutoff_inter'])}\n"
                f"    CutoffThree = {_f(cut['cutoff_three'])}\n"
                f"    CoordinationNumber = Cov {{ Cutoff = {_f(cut['cutoff_cn'])} }}\n"
                f"    ChargeModel = EEQ {{}}\n{atomic}  }}\n")
    raise TypeError(f"no DFTB+ input for {type(model).__name__}")


def klines_block(lines) -> str:
    """HSD ``KPointsAndWeights = Klines`` for ``[(n, (k1, k2, k3)), ...]`` (see pydftb.bands.klines)."""
    rows = "\n".join(f"    {int(n)} {_f(k[0])} {_f(k[1])} {_f(k[2])}" for n, k in lines)
    return f"  KPointsAndWeights = Klines {{\n{rows}\n  }}\n"


def regions_block(regions) -> str:
    """HSD ``ProjectStates`` for ``[{"label", "atoms" (0-based), "shell_resolved",
    "orbital_resolved"}, ...]``; DFTB+ needs every keyword on its own line."""
    if not regions:
        return ""
    out = ["  ProjectStates {"]
    for r in regions:
        out += ["    Region {", f"      Atoms = {' '.join(str(a + 1) for a in r['atoms'])}"]
        if r.get("shell_resolved"):
            out.append("      ShellResolved = Yes")
        if r.get("orbital_resolved"):
            out.append("      OrbitalResolved = Yes")
        out += [f'      Label = "{r["label"]}"', "    }"]
    out.append("  }")
    return "\n".join(out) + "\n"


def region_files(regions, shells) -> list:
    """(file name, region index, shell or None, m or None) of every file DFTB+ writes;
    ``shells[i]`` = angular momenta of the (single) species of region i (resolved regions)."""
    out = []
    for i, r in enumerate(regions or []):
        if r.get("orbital_resolved"):
            for ish, l in enumerate(shells[i]):
                out += [(f"{r['label']}.{ish + 1}.{m}.out", i, ish, m) for m in range(-l, l + 1)]
        if r.get("shell_resolved"):
            out += [(f"{r['label']}.{ish + 1}.out", i, ish, None) for ish in range(len(shells[i]))]
        if not (r.get("orbital_resolved") or r.get("shell_resolved")):
            out.append((f"{r['label']}.out", i, None, None))
    return out


def md_driver(velocities_aaps, time_step_fs, steps, keep_stationary=False,
              xlbomd=None) -> str:
    """HSD ``Driver = VelocityVerlet`` (NVE) with the given velocities (Angstrom/ps).

    DFTB+ uses given velocities as they are; ``KeepStationary`` (DFTB+ default Yes)
    removes the centre-of-mass motion every step, which ASE's VelocityVerlet does not.
    ``xlbomd``: the options of :class:`pydftb.xlbomd.XLBOMD` as a dict (M6e).
    """
    rows = "\n".join("    " + " ".join(_f(x) for x in v) for v in np.asarray(velocities_aaps))
    return (f"Driver = VelocityVerlet {{\n  TimeStep [fs] = {_f(time_step_fs)}\n"
            f"  Steps = {int(steps)}\n  MovedAtoms = 1:-1\n"
            f"  KeepStationary = {'Yes' if keep_stationary else 'No'}\n  MDRestartFrequency = 1\n"
            f"  Thermostat = None {{}}\n{xlbomd_block(xlbomd)}"
            f"  Velocities [AA/ps] = {{\n{rows}\n  }}\n}}\n")


def xlbomd_block(spec) -> str:
    """``Xlbomd`` / ``XlbomdFast`` of the VelocityVerlet driver (M6e) for the options of
    :class:`pydftb.xlbomd.XLBOMD` (DFTB+ defaults where a key is missing)."""
    if not spec:
        return ""
    lines = [f"    IntegrationSteps = {int(spec.get('integration_steps', 5))}",
             f"    PreSteps = {int(spec.get('pre_steps', 0))}"]
    if spec.get("fast"):
        lines += [f"    TransientSteps = {int(spec.get('transient_steps', 10))}",
                  f"    Scale = {_f(spec.get('scale', 1.0))}"]
        name = "XlbomdFast"
    else:
        lines += [f"    MinSccIterations = {int(spec.get('min_scc', 1))}",
                  f"    MaxSccIterations = {int(spec.get('max_scc', 200))}",
                  f"    SccTolerance = {_f(spec.get('scc_tol', 1e-5))}"]
        name = "Xlbomd"
    return f"  {name} {{\n" + "\n".join(lines) + "\n  }\n"


def opt_driver(optimiser="LBFGS {}", lattice=False, grad_elem=1e-7, max_steps=3000,
               isotropic=False) -> str:
    """HSD ``Driver = GeometryOptimisation``: converged when the largest gradient component
    (atoms, and lattice with ``lattice``) is below ``grad_elem`` and the energy did not rise.
    ``isotropic`` scales the cell uniformly."""
    iso = "  Isotropic = Yes\n" if lattice and isotropic else ""
    return (f"Driver = GeometryOptimisation {{\n  Optimiser = {optimiser}\n"
            f"  LatticeOpt = {'Yes' if lattice else 'No'}\n{iso}  MaxSteps = {int(max_steps)}\n"
            f"  Convergence {{ GradElem = {_f(grad_elem)} }}\n}}\n")


def make_hsd(skdir, max_l, periodic, kpts=None, scc=True, filling=("fermi", 0.0),
             shell_resolved=False, scc_tol=1e-10, max_scc=500, repulsive="skf",
             richardson=False, charge=0.0, extra_ham="", timing=False, klines=None,
             read_charges=False, convergent_only=True, extra_analysis="", driver="",
             extra_top="", forces=True, autotest=False) -> str:
    """The HSD input.  ``extra_top``: further top-level blocks (e.g. ``ExcitedState``);
    ``forces=False`` turns off PrintForces; ``autotest`` writes autotest.tag (M6d)."""
    maxl = "\n".join(f'    {e} = "{"spdf"[l]}"' for e, l in max_l.items())
    kblock = ""
    if periodic and klines is not None:
        kblock = klines_block(klines)
    elif periodic:
        if kpts is None:
            kpts = (np.zeros((1, 3)), np.ones(1))
        k, w = kpts
        lines = "\n".join(f"    {_f(a)} {_f(b)} {_f(c)} {_f(x)}" for (a, b, c), x in zip(np.atleast_2d(k), w))
        kblock = f"  KPointsAndWeights = {{\n{lines}\n  }}\n"
    poly = "  PolynomialRepulsive = SetForAll { Yes }\n" if repulsive == "polynomial" else ""
    diff = "  Differentiation = Richardson {}\n" if richardson else ""
    scc_block = ""
    if scc:   # DFTB+ rejects SCC-only keywords in non-SCC runs
        scc_block = (f"  SCCTolerance = {_f(scc_tol)}\n  MaxSCCIterations = {max_scc}\n"
                     f"  ShellResolvedSCC = {'Yes' if shell_resolved else 'No'}\n")
        if read_charges:
            scc_block += "  ReadInitialCharges = Yes\n"
        if not convergent_only:
            scc_block += "  ConvergentSCCOnly = No\n"
    return f"""{driver}Geometry = GenFormat {{
  <<< "geo.gen"
}}
Hamiltonian = DFTB {{
  SCC = {"Yes" if scc else "No"}
{scc_block}  Charge = {_f(charge)}
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{os.path.abspath(skdir)}/"
    Separator = "-"
    Suffix = ".skf"
  }}
  MaxAngularMomentum {{
{maxl}
  }}
  Filling = {filling_block(filling)}
{kblock}{poly}{diff}{extra_ham}
}}
Analysis {{
{extra_analysis}  PrintForces = {"Yes" if forces else "No"}
  MullikenAnalysis = Yes
}}
Options {{
  WriteDetailedOut = Yes
  WriteResultsTag = Yes
{"  WriteAutotestTag = Yes" + chr(10) if autotest else ""}  TimingVerbosity = {2 if timing else 0}
}}
ParserOptions {{
  ParserVersion = {PARSER_VERSION}
}}
Parallel {{
  UseOmpThreads = Yes
}}
{extra_top}"""


HARTREE_EV = 27.2113845        # DFTB+ Hartree__eV (TRA.DAT, EXC.DAT)


def casida_block(spec, forces: bool = True) -> str:
    """Top-level HSD ``ExcitedState { Casida }`` (M6d) for a spec {"n": NrOfExcitations,
    "symmetry": "singlet" | "triplet" | "both" | None (spin polarised), "solver": "Arpack" |
    "Stratmann", "state": StateOfInterest, "energy_window" (Hartree), "oscillator_window",
    "excited_forces", "excited_charges"}.  With forces and no state of interest DFTB+ needs
    ``ExcitedStateForces = No``; excited forces or charges write the excited state's Mulliken
    charges and dipole (``WriteMulliken``: XCH.DAT, XREST.DAT; M6e)."""
    if not spec:
        return ""
    lines = ["ExcitedState {", "  Casida {", f"    NrOfExcitations = {int(spec['n'])}"]
    if spec.get("symmetry"):
        lines.append(f"    Symmetry = {spec['symmetry']}")
    lines += [f"    StateOfInterest = {int(spec.get('state', 0))}",
              f"    Diagonaliser = {spec.get('solver', 'Stratmann')} {{}}"]
    if spec.get("energy_window") is not None:
        lines.append(f"    EnergyWindow = {_f(spec['energy_window'])}")
    if spec.get("oscillator_window") is not None:
        lines.append(f"    OscillatorWindow = {_f(spec['oscillator_window'])}")
    lines += ["    WriteTransitions = Yes", "    WriteSPTransitions = Yes",
              "    WriteTransitionDipole = Yes"]
    if spec.get("excited_forces") or spec.get("excited_charges"):
        lines.append("    WriteMulliken = Yes")
    if forces:          # the state of interest's forces, or the ground state's
        lines.append(f"    ExcitedStateForces = {'Yes' if spec.get('excited_forces') else 'No'}")
    lines += ["  }", "}"]
    return "\n".join(lines) + "\n"


def parse_casida(workdir, detailed: str) -> dict:
    """Excitation outputs of a Casida run (M6d): ``exc_tags`` (autotest.tag: Omega^2,
    oscillator strengths, |transition dipole| per degenerate group), ``exc_energies`` (every
    state from TRA.DAT, Hartree), ``exc_rows`` (EXC.DAT: w eV, oscillator strength, i, a,
    weight, KS eV, symmetry or D<S^2>), ``spx_rows`` (SPX.DAT: w eV, oscillator strength,
    i, a, spin) and ``excitation_energy`` (detailed.out, the state of interest)."""
    out = {}
    path = os.path.join(workdir, "autotest.tag")
    if os.path.exists(path):
        t = parse_results_tag(path)
        out["exc_tags"] = {k: np.atleast_1d(t[n]).astype(float)
                           for k, n in (("omega2", "exc_energies_sqr"),
                                        ("oscillator", "exc_oscillator"),
                                        ("transdip", "exc_transdip")) if n in t}
        if "mermin_energy" in t:        # with a state of interest: E_Mermin + Omega, full digits
            out["excited_mermin_tag"] = float(t["mermin_energy"])
    path = os.path.join(workdir, "TRA.DAT")
    if os.path.exists(path):
        found = re.findall(r"^\s+Energy\s+\d+\s+(\S+)\s+eV\s*(\S?)", open(path).read(), re.M)
        out["exc_energies"] = np.array([float(e) / HARTREE_EV for e, _ in found])
        out["exc_symmetries"] = [sym for _, sym in found]
    path = os.path.join(workdir, "EXC.DAT")
    if os.path.exists(path):
        rows = []
        for ln in open(path).read().splitlines():
            m = re.match(r"^\s*(\S+)\s+(\S+)\s+(\d+)\s+->\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", ln)
            if m and m.group(1) != "w":
                w = float("nan") if m.group(1) == "<" else float(m.group(1))
                rows.append([w, float(m.group(2)), int(m.group(3)), int(m.group(4)),
                             float(m.group(5)), float(m.group(6)), m.group(7)])
        out["exc_rows"] = rows
    path = os.path.join(workdir, "SPX.DAT")
    if os.path.exists(path):
        rows = re.findall(r"^\s+\d+\s+(\S+)\s+(\S+)\s+(\d+)\s+->\s+(\d+)\s*([UD]?)\s*$",
                          open(path).read(), re.M)
        out["spx_rows"] = [[float(w), float(f), int(i), int(a), s] for w, f, i, a, s in rows]
    m = re.search(r"^Excitation Energy:\s+(\S+)\s+H", detailed, re.M)
    if m:
        out["excitation_energy"] = float(m.group(1))
    path = os.path.join(workdir, "XCH.DAT")                # M6e: excited-state Mulliken
    if os.path.exists(path):
        rows = re.findall(r"^\s*(\d+)\s+(\S+)\s*$", open(path).read(), re.M)
        out["excited_charges_net"] = np.array([float(q) for _, q in rows])
    path = os.path.join(workdir, "XREST.DAT")
    if os.path.exists(path):
        m = re.search(r"dipole moment \[Debye\]\s*\n=+\s*\n\s*(\S+)\s+(\S+)\s+(\S+)",
                      open(path).read())
        if m:
            out["excited_dipole_debye"] = np.array([float(x) for x in m.groups()])
    return out


def parse_results_tag(path) -> dict:
    """Parse DFTB+ ``results.tag`` (name :type:rank:shape, then values)."""
    out = {}
    with open(path) as fh:
        text = fh.read()
    blocks = re.split(r"^(\w+)\s*:(\w+):(\d+):([\d,]*)\s*$", text, flags=re.M)
    for name, typ, rank, shape, body in zip(*[iter(blocks[1:])] * 5):
        vals = np.array([float(x) for x in body.split()]) if typ == "real" else body.split()
        rank = int(rank)
        if typ == "real" and rank == 0:
            out[name] = float(vals[0])
        elif typ == "real":
            dims = [int(x) for x in shape.split(",")]
            out[name] = vals.reshape(dims[::-1])        # Fortran column-major
        else:
            out[name] = vals
    return out


def parse_levels(path):
    """``band.out`` or a ``ProjectStates`` region file -> (k weights (n_spin, n_k), values
    (n_spin, n_k, n_state, 2)): energy (eV) and occupation or region weight, as printed.
    band.out rows are (index, energy, occupation); region rows (energy, weight), or for a
    two-component run (energy, q, m_x, m_y, m_z), of which the charge weight q is kept."""
    blocks = []
    with open(path) as fh:
        text = fh.read()
    for m in re.finditer(r"KPT\s+(\d+)\s+SPIN\s+(\d+)\s+KWEIGHT\s+(\S+)\s*\n(.*?)(?:\n\s*\n|\Z)",
                         text, re.S):
        rows = np.array([[float(x) for x in ln.split()] for ln in m.group(4).strip().splitlines()])
        blocks.append((int(m.group(2)), float(m.group(3)),
                       rows[:, 1:] if rows.shape[1] == 3 else rows[:, :2]))
    nspin = len({b[0] for b in blocks})
    nk = len(blocks) // nspin
    kw = np.array([b[1] for b in blocks]).reshape(nspin, nk)
    vals = np.array([b[2] for b in blocks]).reshape(nspin, nk, -1, 2)
    return kw, vals


def parse_md_out(path) -> dict:
    """``md.out`` -> arrays (one entry per MD step) in Hartree: ``potential`` (Mermin free
    energy), ``kinetic``, ``total``."""
    with open(path) as fh:
        text = fh.read()
    out = {}
    for key, name in (("potential", "Potential Energy"), ("kinetic", "MD Kinetic Energy"),
                      ("total", "Total MD Energy")):
        out[key] = np.array([float(x) for x in re.findall(rf"^{name}:\s+(\S+)\s+H", text, re.M)])
    return out


def parse_xyz_frames(path) -> list:
    """DFTB+ ``.xyz`` trajectory -> frames {symbols, positions (A), charges, velocities (A/ps)}."""
    frames = []
    with open(path) as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        rows = [ln.split() for ln in lines[i + 2:i + 2 + n]]
        vals = np.array([[float(x) for x in r[1:]] for r in rows])
        frames.append({"symbols": [r[0] for r in rows], "positions": vals[:, :3],
                       "charges": vals[:, 3] if vals.shape[1] > 3 else None,
                       "velocities": vals[:, 4:7] if vals.shape[1] >= 7 else None})
        i += 2 + n
    return frames


def read_gen(path):
    """DFTB+ gen file -> (species per atom, positions (A), cell (A) or None)."""
    with open(path) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    n, kind = int(lines[0].split()[0]), lines[0].split()[1].upper()
    names = lines[1].split()
    rows = [ln.split() for ln in lines[2:2 + n]]
    labels = [names[int(r[1]) - 1] for r in rows]
    pos = np.array([[float(x) for x in r[2:5]] for r in rows])
    cell = None
    if kind in ("S", "F"):
        cell = np.array([[float(x) for x in ln.split()] for ln in lines[3 + n:6 + n]])
        if kind == "F":
            pos = pos @ cell
    return labels, pos, cell


def run(labels, pos_ang, cell_ang=None, skdir=".", max_l=None, kpts=None, scc=True,
        filling=("fermi", 0.0), shell_resolved=False, scc_tol=1e-10, max_scc=500,
        repulsive="skf", richardson=False, charge=0.0, extra_ham="", threads=1,
        timing=False, workdir=None, keep=False, klines=None, read_charges=False,
        convergent_only=True, extra_analysis="", driver="", extra_top="", forces=True,
        autotest=False) -> dict:
    """Run DFTB+ and return energies (Ha), forces (Ha/Bohr), charges, eigenvalues (Ha,
    (n_spin, n_k, n_level)), metadata."""
    if not available():
        raise FileNotFoundError("dftb+ binary not found; set $DFTBPLUS")
    own_dir = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="dftbp_")
    try:
        write_gen(os.path.join(workdir, "geo.gen"), labels, np.asarray(pos_ang),
                  None if cell_ang is None else np.asarray(cell_ang))
        hsd = make_hsd(skdir, max_l, cell_ang is not None, kpts, scc, filling, shell_resolved,
                       scc_tol, max_scc, repulsive, richardson, charge, extra_ham, timing,
                       klines, read_charges, convergent_only, extra_analysis, driver,
                       extra_top, forces, autotest)
        with open(os.path.join(workdir, "dftb_in.hsd"), "w") as f:
            f.write(hsd)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
        t0 = time.perf_counter()
        p = subprocess.run([DFTBPLUS], cwd=workdir, env=env, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        if p.returncode != 0:
            raise RuntimeError(p.stdout[-3000:] + p.stderr[-3000:])
        det = open(os.path.join(workdir, "detailed.out")).read()
        tag = parse_results_tag(os.path.join(workdir, "results.tag"))
        res = {"wall": wall, "hsd": hsd, "stdout": p.stdout, "dir": workdir}
        for key, name in [("energy", "Total energy"), ("free_energy", "Total Mermin free energy"),
                          ("e_zero", "Extrapolated to 0"), ("e_rep", "Repulsive energy"),
                          ("e_scc", "Energy SCC"), ("e_band", "Energy H0"), ("ts", "-TS"),
                          ("e_disp", "Dispersion energy"), ("e_spin", "Energy SPIN"),
                          ("e_ls", "Energy L.S"), ("e_3rd", "Energy 3rd"),
                          ("e_plus_u", "Energy DFTB+U"), ("e_onsite", "Energy onsite"),
                          ("e_ext", "Energy ext. field"), ("e_solv", "Solvation energy")]:
            m = re.search(rf"^{re.escape(name)}:\s+(\S+)\s+H", det, re.M)
            if m:
                res[key] = float(m.group(1))
        if "ts" in res:
            res["ts"] = -res["ts"]          # detailed.out prints -TS
        res["fermi"] = float(np.atleast_1d(tag["fermi_level"])[0])
        res["fermi_levels"] = np.atleast_1d(tag["fermi_level"]).astype(float)
        res["eigenvalues"] = tag.get("eigenvalues")
        res["filling"] = tag.get("filling")
        res["forces"] = tag.get("forces")
        if "stress" in tag:
            res["stress"] = tag["stress"]          # Hartree/Bohr^3, DFTB+ convention
        res["charges_net"] = tag["gross_atomic_charges"]
        if "gross_atomic_spins" in tag:
            res["spins"] = tag["gross_atomic_spins"]       # (n_atom,) or (3, n_atom)
        if "orbital_charges" in tag:      # (n_spin, n_atom, max orbitals): electrons per
            res["orbital_charges"] = tag["orbital_charges"]   # orbital; collinear: up, down
        if "forces_ext_charges" in tag:   # M6c
            res["point_charge_forces"] = tag["forces_ext_charges"]
        m = re.search(r"^Dipole moment:\s+(\S+)\s+(\S+)\s+(\S+)\s+au", det, re.M)
        if m:
            res["dipole"] = np.array([float(x) for x in m.groups()])
        res.update({f"angular_{k}": v for k, v in parse_angular_momenta(det).items()})
        if autotest or extra_top:
            res.update(parse_casida(workdir, det))
        res["mermin_tag"] = tag.get("mermin_energy")
        res["extrapolated0_tag"] = tag.get("extrapolated0_energy")
        ver = re.search(r"\|\s+(DFTB\+[^|]*?)\s*$", p.stdout, re.M)
        res["version"] = ver.group(1) if ver else "unknown"
        # one line each (\s would also match the newlines of Casida's Stratmann progress table)
        iters = re.findall(r"^[ \t]+(\d+)[ \t]+[-\d.E+]+[ \t]+[-\d.E+]+[ \t]+[-\d.E+]+[ \t]*$",
                           p.stdout, re.M)
        res["n_scc"] = int(iters[-1]) if iters else (1 if not scc else None)
        if driver:
            res["geometry_steps"] = len(re.findall(r"Geometry step: \d+", p.stdout))
            res["converged_geometry"] = "Geometry converged" in p.stdout
            res["final_geometry"] = read_gen(os.path.join(workdir, "geo_end.gen"))
            res["scc_iterations"] = [int(x) for x in re.findall(
                r"^\s+(\d+)\s+[-\d.E+]+\s+[-\d.E+]+\s+[-\d.E+]+\s*\n\s*\n", p.stdout, re.M)]
            if os.path.exists(os.path.join(workdir, "md.out")):
                res["md"] = parse_md_out(os.path.join(workdir, "md.out"))
                res["trajectory"] = parse_xyz_frames(os.path.join(workdir, "geo_end.xyz"))
        return res
    finally:
        if own_dir and not keep:
            shutil.rmtree(workdir, ignore_errors=True)


def dp_dos(args, infile, workdir) -> np.ndarray:
    """Run dptools' dp_dos on a band.out-like file; returns its columns (x, y...)."""
    if not DPTOOLS:
        raise FileNotFoundError("dp_dos needs $DPTOOLS = the dptools source directory "
                                "(tools/dptools/src in the DFTB+ repository)")
    out = os.path.join(workdir, "dp_dos.dat")
    code = (f"import sys; sys.path.insert(0, {DPTOOLS!r}); "
            "from dptools.scripts.dp_dos import main; main()")
    subprocess.run([sys.executable, "-I", "-c", code, *args, infile, out], check=True,
                   capture_output=True, text=True)
    return np.loadtxt(out, ndmin=2)


def run_bands(labels, pos_ang, cell_ang=None, skdir=".", max_l=None, kpts=None, klines=None,
              regions=None, region_shells=None, dos=None, threads=1, **kw) -> dict:
    """The SCC run (with ``ProjectStates`` regions) and, for ``klines``, DFTB+'s band run
    on the SCC charges, in one directory.  Returns the SCC results with ``regions``
    ({file: (k weights, values)}), ``band_out`` (parsed ``band.out``) and ``dp_dos``
    (``dos`` = [(file, arguments)] -> [(file, arguments, columns)]), plus ``band``: the
    band run's ``eigenvalues``, ``filling`` and ``fermi_levels``."""
    workdir = tempfile.mkdtemp(prefix="dftbp_bands_")
    try:
        res = run(labels, pos_ang, cell_ang, skdir, max_l, kpts, threads=threads, workdir=workdir,
                  extra_analysis=regions_block(regions), **kw)
        res["regions"] = {name: parse_levels(os.path.join(workdir, name))
                          for name, *_ in region_files(regions, region_shells)}
        res["band_out"] = parse_levels(os.path.join(workdir, "band.out"))
        res["dp_dos"] = [(name, list(args), dp_dos(args, os.path.join(workdir, name), workdir))
                         for name, args in (dos or [])]
        if klines is not None:
            band = run(labels, pos_ang, cell_ang, skdir, max_l, kpts, threads=threads,
                       workdir=workdir, klines=klines, read_charges=True,
                       convergent_only=False, **{**kw, "max_scc": 1})
            res["band"] = {k: band[k] for k in ("eigenvalues", "filling", "fermi_levels", "hsd")}
        return res
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
