"""One set of options, two engines (P2): pydftb or DFTB+, switched by one argument.

>>> from pydftb.engines import make_calculator
>>> calc = make_calculator("path/to/skf", engine="dftbplus", kpts=(4, 4, 4),
...                        smearing=("fermi", 300.0), dftbplus={"binary": "/opt/dftb+"})
>>> atoms.calc = calc
>>> common_results(calc, atoms)["energy"]          # eV, the same fields for either engine

``options`` are :class:`pydftb.ase_calc.PyDFTB`'s, in its units, and both engines take them
(:data:`OPTIONS` lists what each engine supports).  ``pydftb={...}`` and ``dftbplus={...}`` hold
options for one engine only, so a set of options switches engines by ``engine`` alone; an
option one engine does not have, given with the common ones, raises :class:`EngineOptionError`
saying where it belongs.  PTBP keeps its own engine switch (``utils/engines.py``).
"""
from __future__ import annotations

import numpy as np
from ase import units

from .dftbplus_calc import (
    DFTBPLUS_DEFAULTS,
    IGNORED,
    OPTION_DEFAULTS,
    REFUSED,
    DFTBPlus,
    EngineOptionError,
)
from .dftbplus_calc import check_options as _check_dftbplus

ENGINES = ("pydftb", "dftbplus")
ALIASES = {"pydftb": "pydftb", "dftbplus": "dftbplus", "dftb+": "dftbplus", "dftb": "dftbplus"}

# notes of the options both engines take (what differs, or what DFTB+ needs)
_NOTES = {
    "kpts": "DFTB+ is given pydftb's reduced k-points; a Gamma-centred grid of a periodic "
            "hybrid as DFTB+'s SupercellFolding",
    "solver": "batched/eigh: DFTB+'s default eigensolver; qr, divideandconquer, "
              "relativelyrobust; elpa and NTPoly need a DFTB+ with ELSI; foe, subsystems, sp2: "
              "pydftb only",
    "repulsive": "'none': pydftb only",
    "fd_delta": "DFTB+ always differentiates H0/S by finite differences (None: its step "
                "2^-13 Bohr); pydftb's None is exact",
    "solvation": "the solvation spec dict on both engines; model objects, radii and CM5 "
                 "parameters: pydftb only (DFTB+ 25.1's CM5 is double_count=True)",
    "dispersion": "the dispersion spec dict on both engines",
    "analysis": "cm5 options (a dict): pydftb only; bond analysis and "
                "external_charge_derivatives need DFTB+'s development version",
    "point_charges": "DFTB+: one Gaussian blur width for all charges",
    "reuse_charges": "DFTB+: the last run's charges.bin and ReadInitialCharges (none for "
                     "non-collinear hybrids: DFTB+ writes no charge file)",
    "read_charges": "the first calculation starts from this DFTB+ charge file",
    "threads": "pydftb: BLAS (and torch) threads; DFTB+: OMP_NUM_THREADS",
    "casida": "pydftb's solver 'arpack': DFTB+'s Arpack, the others: Stratmann",
    "convergent_scc_only": "True (default, both engines): an unconverged SCC raises "
                           "CalculationFailed; False: the last iteration's results",
    "max_l": "DFTB+ MaxAngularMomentum; default on both engines: the occupied shells "
             "(PTBP's basis), warning for unused parametrised ones; 'highest': all of them",
}
_DFTBPLUS_NOTES = {
    "binary": "the DFTB+ executable (default $DFTBPLUS, then dftb+ on PATH)",
    "ewald_tolerance": "1e-12 (pydftb's); None: DFTB+'s own 1e-9",
    "regions": "DFTB+'s ProjectStates in this run (the common way: calc.states(regions=...))",
    "workdir": "the work directory (default: a new temporary one per run)",
    "keep_files": "keep the temporary work directories",
    "extra_hamiltonian": "HSD lines added to the Hamiltonian block",
    "extra_analysis": "HSD lines added to the Analysis block",
    "extra_options": "HSD lines added to the Options block",
    "extra_top": "top-level HSD blocks",
}


def _table() -> dict:
    out = {}
    for name in OPTION_DEFAULTS:
        if name in REFUSED:
            out[name] = {"pydftb": True, "dftbplus": False, "note": REFUSED[name]}
        elif name in IGNORED:
            out[name] = {"pydftb": True, "dftbplus": True,
                         "note": "only pydftb's speed: the DFTB+ engine accepts and ignores it"}
        else:
            out[name] = {"pydftb": True, "dftbplus": True, "note": _NOTES.get(name, "")}
    for name in DFTBPLUS_DEFAULTS:
        out[name] = {"pydftb": False, "dftbplus": True, "note": _DFTBPLUS_NOTES.get(name, "")}
    return out


#: option -> {"pydftb": bool, "dftbplus": bool, "note": str}
OPTIONS = _table()


def engine_name(engine: str) -> str:
    try:
        return ALIASES[str(engine).lower()]
    except KeyError:
        raise ValueError(f"engine must be one of {ENGINES}, not {engine!r}") from None


def check_options(engine: str, options: dict, section: bool = False) -> None:
    """Raise :class:`EngineOptionError` for an option ``engine`` does not run (and TypeError for
    unknown names).  ``section``: the options come from the engine's own section."""
    engine = engine_name(engine)
    unknown = sorted(set(options) - set(OPTIONS))
    if unknown:
        raise TypeError(f"unknown options {unknown}")
    other = "dftbplus" if engine == "pydftb" else "pydftb"
    for name in options:
        if not OPTIONS[name][engine]:
            where = (f"; give it in {other}={{...}}, the {other} engine's own options"
                     if OPTIONS[name][other] and not section else "")
            raise EngineOptionError(f"{name}: not an option of the {engine} engine "
                                    f"({OPTIONS[name]['note']}){where}")
    if engine == "dftbplus":
        try:
            _check_dftbplus(options)
        except EngineOptionError as err:
            hint = "" if section else "; give it in pydftb={...} to use it with pydftb only"
            raise EngineOptionError(f"{err}{hint}") from None


def make_calculator(skdir, engine: str = "pydftb", pydftb: dict | None = None,
                    dftbplus: dict | None = None, **options):
    """:class:`pydftb.ase_calc.PyDFTB` or :class:`pydftb.dftbplus_calc.DFTBPlus` with
    ``options`` (both engines' options, PyDFTB's units), plus the chosen engine's own section."""
    engine = engine_name(engine)
    own = {"pydftb": dict(pydftb or {}), "dftbplus": dict(dftbplus or {})}[engine]
    check_options(engine, options)
    check_options(engine, own, section=True)
    opts = {**options, **own}
    if engine == "pydftb":
        from .ase_calc import PyDFTB
        calc = PyDFTB(skdir, **opts)
        calc.engine = "pydftb"
        return calc
    return DFTBPlus(skdir, **opts)


def engine_of(calc) -> str:
    return getattr(calc, "engine", None) or ("dftbplus" if isinstance(calc, DFTBPlus)
                                             else "pydftb")


def common_results(calc, atoms=None, properties=("energy", "forces")) -> dict:
    """The results both engines give, ASE units, as a flat dict of numbers and lists (JSON):
    energy, free_energy (eV), forces (eV/A), stress (Voigt, eV/A^3, periodic), charges (e),
    dipole (e A, molecules), magmoms, magmom, fermi_levels (eV), eigenvalues and occupations
    (n_spin, n_k, n_state; eV), kpoints, kweights, n_scc, and with them excitation_energies
    (eV) and oscillator_strengths, energies (per atom, eV) and cm5_charges.  Calculates first
    when ``atoms`` is given."""
    if atoms is not None:                 # the most demanding property first: one calculation
        from ase.calculators.calculator import PropertyNotImplementedError
        done = False
        if "stress" in properties and atoms.pbc.any():
            try:
                calc.get_stress(atoms)
                done = True
            except PropertyNotImplementedError:     # e.g. hybrid functionals: no stress
                pass
        if "forces" in properties and not done:
            calc.get_forces(atoms)
        calc.get_potential_energy(atoms)
    r = calc.raw
    res = calc.results
    out = {"engine": engine_of(calc)}
    for key in ("energy", "free_energy", "magmom"):
        if key in res:
            out[key] = float(res[key])
    for key in ("forces", "stress", "charges", "dipole", "magmoms", "energies"):
        if key in res:
            out[key] = np.asarray(res[key], float).tolist()
    out["fermi_levels"] = (np.array(r.fermi_spin if r.fermi_spin is not None else [r.fermi])
                           * units.Hartree).tolist()
    out["n_scc"] = int(r.n_scc)
    if r.kweights is not None:
        out["kpoints"] = np.asarray(r.kpoints, float).tolist()
        out["kweights"] = np.asarray(r.kweights, float).tolist()
    if r.eigenvalues is not None:
        nk = 1 if r.kweights is None else len(r.kweights)
        ev = np.asarray(r.eigenvalues, float)
        out["eigenvalues"] = (ev.reshape(-1, nk, ev.shape[-1]) * units.Hartree).tolist()
        if r.occupations is not None:
            out["occupations"] = np.asarray(r.occupations, float).reshape(
                -1, nk, ev.shape[-1]).tolist()
    exc = r.excitations
    if exc is not None:
        xs = exc if isinstance(exc, (list, tuple)) else [exc]
        out["excitation_energies"] = (np.concatenate([np.asarray(x.energies, float)
                                                      for x in xs]) * units.Hartree).tolist()
        out["oscillator_strengths"] = np.concatenate(
            [np.asarray(x.oscillator_strengths, float) for x in xs]).tolist()
        if r.excitation_energy is not None:
            out["excitation_energy"] = float(r.excitation_energy * units.Hartree)
    cm5 = None
    if getattr(r, "analysis", None) is not None and getattr(r.analysis, "cm5_charges", None) \
            is not None:
        cm5 = r.analysis.cm5_charges
    elif engine_of(calc) == "dftbplus":
        cm5 = ((calc.output or {}).get("analysis") or {}).get("cm5")
    if cm5 is not None:
        out["cm5_charges"] = np.asarray(cm5, float).tolist()
    return out


def compare_results(a: dict, b: dict) -> dict:
    """Largest absolute differences of two :func:`common_results` (keys in both)."""
    out = {}
    for key in sorted(set(a) & set(b)):
        if key == "engine":
            continue
        try:
            x, y = np.asarray(a[key], float), np.asarray(b[key], float)
        except (TypeError, ValueError):
            continue
        if x.shape != y.shape:
            out[key] = float("inf")
        elif x.size:
            out[key] = float(np.abs(x - y).max())
    return out


__all__ = ["ENGINES", "OPTIONS", "EngineOptionError", "check_options", "common_results",
           "compare_results", "engine_name", "make_calculator"]
