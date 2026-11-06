"""DFTB+ as the second engine (P2): an ASE calculator with :class:`pydftb.ase_calc.PyDFTB`'s
options, units and results, which writes DFTB+'s input, runs DFTB+ and reads its outputs back.

>>> from pydftb.dftbplus_calc import DFTBPlus
>>> atoms.calc = DFTBPlus("path/to/skf", kpts=(4, 4, 4), smearing=("fermi", 300.0))
>>> atoms.get_potential_energy()                   # eV, as PyDFTB

or, with one set of options for either engine, :func:`pydftb.engines.make_calculator`.

* The input is assembled from the blocks of :mod:`pydftb.validation.dftbplus`, the writer of
  all the frozen DFTB+ references: every frozen input (517 single points, 34 optimisations and
  MD runs, 37 band and SecondDerivatives runs) is what this calculator writes for the same
  options (``pydftb dual --inputs``).
* Options are converted to Bohr and Hartree with pydftb's own conversions (``ase_calc``) and
  written in DFTB+'s default units, so both engines are given the same numbers.  Defaults are
  pydftb's (SCC tolerance 1e-6, 200 iterations, Fermi at T = 0, Ewald tolerance 1e-12), always
  written out.
* Results: ``results`` in ASE units converted as PyDFTB does, ``raw`` a
  :class:`pydftb.calculator.DFTBResult` (atomic units) with the fields DFTB+ gives, ``output``
  everything parsed, with the parser of the frozen references.
* DFTB+-only options: ``binary`` (default ``$DFTBPLUS``, then ``dftb+`` on PATH),
  ``ewald_tolerance`` (1e-12; None: DFTB+'s 1e-9), ``convergent_scc_only`` (True: DFTB+ stops on
  an unconverged SCC, which raises ``CalculationFailed``), ``regions`` (``ProjectStates`` in the
  run), ``workdir`` and ``keep_files``, and HSD text added as it is: ``extra_hamiltonian``,
  ``extra_analysis``, ``extra_options``, ``extra_top``.
"""
from __future__ import annotations

import gzip
import inspect
import os
import shutil
import subprocess
import tempfile
import time
import weakref
from dataclasses import dataclass, field

import numpy as np
from ase import units
from ase.calculators.calculator import (
    CalculationFailed,
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)

from .analysis import Analysis
from .ase_calc import PyDFTB, _fermi_au, _hybrid_au, _truncation_au, kpoints_of
from .calculator import DFTBResult
from .casida import Casida
from .hybrid import Hybrid
from .kpoints import supercell_folding
from .skf import SKFile, max_l_from_skf, parse_selected_shells
from .validation import dftbplus as _dp

FD_DEFAULT = float(np.finfo(float).eps) ** 0.25    # DFTB+'s FiniteDiff step, 2^-13 Bohr
PRINTED = 1e-9          # detailed.out prints energies with 10 decimals (Hartree)


class EngineOptionError(ValueError):
    """An option the chosen engine cannot run (see :data:`pydftb.engines.OPTIONS`)."""


def _pydftb_defaults() -> dict:
    sig = inspect.signature(PyDFTB.__init__)
    return {name: p.default for name, p in sig.parameters.items()
            if name not in ("self", "skdir", "kwargs")}


#: PyDFTB's options and defaults: the options both engines take
OPTION_DEFAULTS = _pydftb_defaults()
#: the DFTB+ engine's own options
DFTBPLUS_DEFAULTS = {"binary": None, "ewald_tolerance": 1e-12, "convergent_scc_only": True,
                     "regions": None, "workdir": None, "keep_files": False,
                     "extra_hamiltonian": "", "extra_analysis": "", "extra_options": "",
                     "extra_top": ""}
#: pydftb options that change only how fast pydftb is: accepted and not used
IGNORED = ("k_threads",)
#: pydftb options with no DFTB+ counterpart, and why
REFUSED = {
    "type_switching": "smooth type switching is pydftb's (M5a); give the fixed type labels",
    "adaptive": "environment-adaptive parameters are pydftb's (M8a)",
    "backend": "DFTB+ runs its own linear algebra (NumPy-like, float64)",
    "device": "DFTB+ runs on the CPU here",
    "dtype": "DFTB+ computes in float64",
    "xlbomd": "XL-BOMD is an option of DFTB+'s own MD driver: DFTBPlus.md() "
              "(pydftb run md --native)",
    "force_evaluation": "an option of DFTB+'s own MD driver: DFTBPlus.md()",
}


def _not_default(name, value) -> bool:
    d = OPTION_DEFAULTS.get(name)
    if isinstance(value, np.ndarray) or isinstance(d, np.ndarray):
        return True
    return value is not None and value != d


def check_options(options: dict) -> None:
    """TypeError for unknown names, EngineOptionError for pydftb-only options."""
    known = set(OPTION_DEFAULTS) | set(DFTBPLUS_DEFAULTS)
    unknown = sorted(set(options) - known)
    if unknown:
        raise TypeError(f"DFTBPlus: unknown options {unknown}")
    for name, why in REFUSED.items():
        if name in options and _not_default(name, options[name]):
            if name == "backend" and str(options[name]).lower() == "numpy":
                continue
            if name == "dtype" and str(options[name]) == "float64":
                continue
            if name == "device" and str(options[name]) == "cpu":
                continue
            raise EngineOptionError(f"{name}={options[name]!r}: not available with the DFTB+ "
                                    f"engine ({why})")
    if options.get("repulsive", "skf") not in ("skf", "polynomial"):
        raise EngineOptionError(f"repulsive={options['repulsive']!r}: DFTB+ always adds the "
                                "SK repulsive (repulsive 'skf' or 'polynomial')")
    _solvation(options.get("solvation"))


# ---- SK files ----------------------------------------------------------------------------------
def plain_sk_dir(skdir: str, cache: str) -> str:
    """``skdir`` when it has uncompressed ``.skf`` files, else its ``.skf.gz`` files decompressed
    into ``cache`` (DFTB+ reads plain text), once."""
    names = os.listdir(skdir)
    if any(n.endswith(".skf") for n in names):
        return os.path.abspath(skdir)
    os.makedirs(cache, exist_ok=True)
    for n in names:
        if n.endswith(".skf.gz"):
            dst = os.path.join(cache, n[:-3])
            if not os.path.exists(dst):
                tmp = dst + f".{os.getpid()}.tmp"
                with gzip.open(os.path.join(skdir, n), "rb") as fi, open(tmp, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                os.replace(tmp, dst)
    return os.path.abspath(cache)


def _selected(spec) -> dict:
    """SelectedShells as {label: [(file label, letters), ...]}."""
    if not spec:
        return {}
    return {lab: [(f, "".join("spdf"[ll] for ll in ls)) for f, ls in groups]
            for lab, groups in parse_selected_shells(spec).items()}


def _max_l(skdir, species, max_l, sel) -> dict:
    out = {}
    for lab in species:
        if lab in sel:
            continue
        if max_l and lab in max_l:
            out[lab] = int(max_l[lab])
            continue
        path = os.path.join(skdir, f"{lab}-{lab}.skf")
        lmax = max_l_from_skf(path)
        if lmax is None:
            raise ValueError(f"{path}: no occupied shell; give max_l for {lab}")
        out[lab] = lmax
    return out


def neutral_populations(skdir, labels, max_l, sel, reference_occupations=None) -> np.ndarray:
    """The reference populations per atom, as pydftb's (``SKSet.occupation``: the homonuclear
    SK files' shells up to max_l, or the selected shells), with DFTB+ CustomisedOccupations'
    shells replaced (DFTB+'s shell names: s, p, ..., s2 for a second s shell)."""
    from .params import ParamSet
    from .skf import shell_names
    species = list(dict.fromkeys(labels))
    sk = ParamSet(skdir, max_l={k: v for k, v in max_l.items() if k not in sel},
                  selected_shells=sel or None).skset(species)
    occ = {lab: np.asarray(sk.occupation[lab], float) for lab in species}
    out = np.array([occ[lab].sum() for lab in labels])
    for atom, custom in (reference_occupations or {}).items():
        a = int(atom)
        names = shell_names(sk.shells(labels[a]))
        out[a] = sum(float(custom.get(name, occ[labels[a]][n])) for n, name in enumerate(names))
    return out


# ---- option blocks -----------------------------------------------------------------------------
def _efield(spec):
    """{"strength" (V/A), "direction"} of ``electric_field`` ((Ex, Ey, Ez) V/A, or that dict)."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        return {"strength": float(spec["strength"]), "direction": list(spec["direction"])}
    e = np.asarray(spec, float)
    s = float(np.linalg.norm(e))
    return None if s == 0.0 else {"strength": s, "direction": (e / s).tolist()}


def _point_charges(spec):
    """{"positions" (A), "charges", "blur" (A)} of ``point_charges``."""
    if spec is None:
        return None
    from .external import PointCharges
    pc = PointCharges.make(spec)
    blur = np.unique(pc.blur)
    if len(blur) > 1:
        raise EngineOptionError("point_charges: DFTB+ has one Gaussian blur width for all "
                                "point charges")
    return {"positions": pc.positions.tolist(), "charges": pc.charges.tolist(),
            "blur": float(blur[0]) if len(blur) else 0.0}


def _solvation(spec):
    """The solvation spec as DFTB+ is given it, or None: checked as the pydftb engine checks it
    (:func:`pydftb.solvation.solvation_from_spec`), the parameter file an absolute path (DFTB+
    runs in its own directory); what DFTB+'s input cannot say is refused."""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise EngineOptionError("solvation: the DFTB+ engine takes the solvation spec dict "
                                "(pydftb.solvation.solvation_from_spec), not a model object")
    from .solvation import CM5, solvation_from_spec
    spec = dict(spec)
    if "param_file" in spec:
        spec["param_file"] = os.path.abspath(spec["param_file"])
    solvation_from_spec(spec)
    for what, d in (("solvation", spec), ("solvation sasa", spec.get("sasa") or {})):
        if d.get("radii") is not None or str(d.get("radii_table", "d3")).lower() != "d3":
            raise EngineOptionError(f"{what}: radii and radii_table are pydftb's; DFTB+ is "
                                    "given its default (D3) van der Waals radii")
    if spec.get("cm5") is not None and CM5(**spec["cm5"]) != CM5(double_count=True):
        raise EngineOptionError(
            "solvation cm5: DFTB+ 25.1 counts every atom pair of the Born model's CM5 twice, "
            "with CM5's default parameters; give {\"double_count\": True} for the same model on "
            "both engines")
    return spec


def _casida(spec, spin_polarised: bool):
    """``ExcitedState { Casida }`` of pydftb's ``casida``: DFTB+'s Arpack for pydftb's "arpack",
    else Stratmann; no Symmetry for a spin-polarised ground state."""
    c = Casida.make(spec)
    if c is None:
        return None
    return {"n": c.n_excitations, "symmetry": None if spin_polarised else c.symmetry,
            "state": c.state_of_interest,
            "solver": "Arpack" if c.solver == "arpack" else "Stratmann",
            "energy_window": c.energy_window, "oscillator_window": c.oscillator_window,
            "excited_forces": c.excited_forces, "excited_charges": c.excited_charges,
            "transition_charges": c.transition_charges}


def _analysis(spec):
    """The analysis of :func:`pydftb.validation.dftbplus.analysis_block` (Angstrom)."""
    a = Analysis.make(spec)
    if a is None:
        return None
    out = {}
    for key in ("atom_energies", "net_populations", "eigenvectors", "bond_populations",
                "bond_energies", "bond_orders"):
        if getattr(a, key):
            out[key] = True
    if a.cm5 is not False:
        if a.cm5 is not True:
            raise EngineOptionError("analysis cm5 options: not verified against DFTB+ "
                                    "(use cm5=True)")
        out["cm5"] = True
    for key in ("esp", "localise", "polarisability", "response_kernel", "perturbation",
                "coord_derivatives", "external_charge_derivatives"):
        v = getattr(a, key)
        if v not in (None, False):
            out[key] = v
    if (a.perturbation or {}).get("filling_tol") is not None:
        out["perturbation"] = {k: v for k, v in a.perturbation.items() if k != "filling_tol"}
    return out or None


def _solver(spec):
    """DFTB+'s Solver of pydftb's ``solver`` (None: DFTB+'s default eigensolver)."""
    if spec is None or (isinstance(spec, str) and spec in ("batched", "eigh")):
        return None
    if isinstance(spec, str):
        if spec in _dp.SOLVER_NAMES:
            return spec
        raise EngineOptionError(f"solver={spec!r}: no DFTB+ counterpart (pydftb only)")
    if isinstance(spec, dict) and spec.get("name") == "ntpoly":
        return {k: v for k, v in spec.items() if k in ("name", "method", "tolerance",
                                                         "truncation")}
    from .purification import NAMES, NTPoly
    if isinstance(spec, NTPoly):
        return {"name": "ntpoly", "method": NAMES[spec.method], "tolerance": spec.tolerance,
                "truncation": spec.truncation}
    raise EngineOptionError(f"solver {spec!r}: no DFTB+ counterpart (pydftb only)")


def _truncation(spec):
    t = _truncation_au(spec)
    if t is None or isinstance(t, dict):
        return t
    if np.isscalar(t):
        return {"max_distance": float(t), "hard_cutoff": True}
    return {"max_distance": float(t[0]), "hard_cutoff": bool(t[1])}


def two_component(o) -> bool:
    spin = o["spin"]
    return ((isinstance(spin, str) and spin.lower() in ("noncolinear", "noncollinear"))
            or o["spin_orbit"] is not None)


# ---- the input ---------------------------------------------------------------------------------
@dataclass
class Input:
    """What one DFTB+ run reads: the HSD text, the geometry, extra files, and how to run it."""
    hsd: str
    labels: list
    positions: np.ndarray                 # Angstrom
    cell: np.ndarray | None               # Angstrom
    make_hsd: dict                        # the keyword arguments of dftbplus.make_hsd
    files: dict = field(default_factory=dict)   # {name: bytes} written next to dftb_in.hsd
    analysis: dict | None = None
    kpoints: tuple | None = None          # (k, w) DFTB+ uses (folding: its own set)
    excited: bool = False                 # energy and forces of a Casida state of interest
    spin: str = "none"                    # "none", "collinear" or "two-component"
    binary: str | None = None
    threads: int | None = None

    def gen(self) -> str:
        """The geometry file DFTB+ reads (``geo.gen``)."""
        with tempfile.TemporaryDirectory(prefix="pydftb_gen_") as tmp:
            path = os.path.join(tmp, "geo.gen")
            _dp.write_gen(path, self.labels, self.positions, self.cell)
            with open(path) as fh:
                return fh.read()


def write_input(atoms, skdir: str, options: dict, properties=("energy",), *, cache: str,
                charges: tuple | None = None, driver: str = "", band_kpts=None) -> Input:
    """DFTB+'s input for ``atoms`` and PyDFTB-style ``options`` (and the DFTB+-only ones).
    ``charges``: (file name, bytes) of a charge file to start from (``ReadInitialCharges``;
    "charges.dat": text); ``driver``: a Driver block; ``band_kpts``: a band run's k-points
    ({"klines": lines} or (k, w)) at ``charges``: one iteration, no forces, no analysis, no
    excitations."""
    check_options(options)
    o = {**OPTION_DEFAULTS, **DFTBPLUS_DEFAULTS, **options}
    labels = list(o["labels"] or atoms.get_chemical_symbols())
    species = list(dict.fromkeys(labels))
    periodic = bool(atoms.pbc.any())
    plain = plain_sk_dir(skdir, cache)
    sel = _selected(o["selected_shells"])
    max_l = _max_l(plain, species, o["max_l"], sel)
    spin = o["spin"]
    noncolinear = isinstance(spin, str) and spin.lower() in ("noncolinear", "noncollinear")
    collinear = bool(spin) and not noncolinear
    hybrid = Hybrid.make(_hybrid_au(o["hybrid"]))
    lab_set = set(labels)

    # k-points: pydftb's set; a Gamma-centred grid of a periodic hybrid as DFTB+'s folding
    kspec, kpts, kpoints = o["kpts"], None, None
    if periodic:
        if (hybrid is not None and isinstance(kspec, dict) and kspec.get("gamma")
                and "size" in kspec):
            n = tuple(int(x) for x in kspec["size"])
            kpts, kpoints = {"folding": n}, supercell_folding(n)
        else:
            kpoints = kpoints_of(atoms, kspec, o["time_reversal"], two_component(o))
            if kpoints is None:
                kpoints = (np.zeros((1, 3)), np.ones(1))
                kpts = "gamma_folding" if hybrid is not None else None
            else:
                kpts = kpoints
    smearing = o["smearing"]
    if smearing is None:
        smearing = ("fermi", 0.0 if o["temperature"] is None else o["temperature"])

    # initial spins as PyDFTB takes them (the atoms' moments when none are given)
    initial_spins = o["initial_spins"]
    m = np.asarray(atoms.get_initial_magnetic_moments(), float)
    if noncolinear and initial_spins is None:
        initial_spins = m if m.ndim == 2 else np.outer(m, [0.0, 0.0, 1.0])
    elif collinear and initial_spins is None and m.ndim == 1 and np.any(m != 0.0):
        initial_spins = m

    extra = _dp.dftb3_block({k: o[k] for k in ("third_order", "hubbard_derivs", "h_damping")
                             if o[k] is not None}, lab_set)
    if collinear:
        if o["spin_constants"] is None:
            raise ValueError("collinear spin needs spin_constants")
        sp = {"unpaired_electrons": o["unpaired_electrons"],
              "relax_total_spin": o["relax_total_spin"], "spin_constants": o["spin_constants"]}
        if initial_spins is not None:
            sp["initial_spins"] = (dict(initial_spins) if isinstance(initial_spins, dict)
                                   else np.asarray(initial_spins, float).tolist())
        extra += _dp.spin_block(sp, lab_set, o["shell_resolved"])
    if noncolinear:
        if o["spin_constants"] is None:
            raise ValueError("non-collinear spin needs spin_constants")
        extra += _dp.noncolinear_block({"initial_spins": np.asarray(initial_spins, float).tolist(),
                                        "spin_constants": o["spin_constants"]}, lab_set,
                                       o["shell_resolved"])
    if o["spin_orbit"] is not None:
        nsh = {lab: (sum(len(s) for _, s in sel[lab]) if lab in sel else max_l[lab] + 1)
               for lab in species}
        extra += _dp.spinorbit_block({"xi": o["spin_orbit"], "dual": o["spin_orbit_dual"]}, nsh)
    if o["dispersion"] is not None:
        if not isinstance(o["dispersion"], dict):
            raise EngineOptionError("dispersion: the DFTB+ engine takes the dispersion spec "
                                    "dict (pydftb.dispersion.make_dispersion's input)")
        extra += _dp.dispersion_block(o["dispersion"], labels)
    shells = {lab: list(range(max_l[lab] + 1)) for lab in species if lab not in sel}
    if o["plus_u"]:
        extra += _dp.orbital_potential_block({"groups": o["plus_u"],
                                              "functional": o["plus_u_functional"]}, shells)
    if o["onsite_corrections"]:
        extra += _dp.onsite_block(o["onsite_corrections"], species)
    extra += _dp.electric_field_block(_efield(o["electric_field"]),
                                      _point_charges(o["point_charges"]))
    extra += _dp.site_potential_block(o["site_potentials"])
    if o["solvation"] is not None:
        extra += _dp.solvation_block(_solvation(o["solvation"]), o["rescale_solvated_fields"],
                                     "")
    mixer = o["mixer"]
    if o["mixing"] is not None:
        mixer = {**(mixer if isinstance(mixer, dict) else {"kind": mixer}), "mixing": o["mixing"]}
    if not (isinstance(mixer, str) and mixer.lower() == "broyden"):
        extra += _dp.mixer_block(mixer)
    extra += _dp.initial_charges_block(o["initial_charges"])
    extra += _dp.custom_hubbards_block(o["hubbard"], lab_set)
    extra += _dp.custom_occupations_block(o["reference_occupations"])
    extra += _dp.truncate_block(_truncation(o["truncate_sk_range"]))
    if hybrid is not None:
        extra += _dp.hybrid_block(hybrid, kpoints=isinstance(kpts, dict))
    extra += _dp.corrections_block(o["h5"], {} if o["halogen_correction"] else None,
                                   o["dipole_correction"], labels)
    extra += _dp.solver_block(_solver(o["solver"]))
    casida = None if band_kpts is not None else _casida(o["casida"], collinear)
    if casida and o["spin_constants"] is not None and not collinear and not noncolinear:
        extra += _dp.spin_constants_block(o["spin_constants"], lab_set)
    if periodic and o["ewald_tolerance"] is not None:
        extra += f"  EwaldTolerance = {_dp._f(o['ewald_tolerance'])}\n"
    if o["fd_delta"] is not None and float(o["fd_delta"]) != FD_DEFAULT:
        extra += (f"  Differentiation = FiniteDiff {{\n    Delta [Bohr] = "
                  f"{_dp._f(o['fd_delta'])}\n  }}\n")
    extra += o["extra_hamiltonian"]
    if charges is not None and o["scc"]:            # DFTB+ reads no initial spins then
        extra = _dp.strip_initial_spins(extra)

    want_forces = band_kpts is None and ("forces" in properties
                                         or (periodic and "stress" in properties))
    analysis = None if band_kpts is not None else _analysis(o["analysis"])
    extra_analysis = _dp.regions_block(o["regions"]) + o["extra_analysis"]
    autotest = False
    if analysis:
        extra_analysis += _dp.analysis_block(analysis, periodic, o["scc"])
        autotest = bool(analysis.get("esp") or analysis.get("localise"))
    files = {}
    extra_options = o["extra_options"]
    read = charges is not None and o["scc"]
    if read:
        name, blob = charges
        files[name] = blob
        if name.endswith(".dat"):
            extra_options = "  ReadChargesAsText = Yes\n" + extra_options
    extra_top = o["extra_top"]
    if casida:
        extra_top = _dp.casida_block(casida, want_forces) + extra_top
        autotest = True
    max_scc, convergent = o["max_scc"], o["convergent_scc_only"]
    klines = None
    if band_kpts is not None:
        max_scc, convergent = 1, False
        if isinstance(band_kpts, dict) and "klines" in band_kpts:
            klines = [(int(n), tuple(float(x) for x in k)) for n, k in band_kpts["klines"]]
        else:
            kpts = band_kpts
    kw = {"skdir": plain, "max_l": {lab: max_l.get(lab, 0) for lab in species},
          "periodic": periodic, "kpts": kpts, "scc": o["scc"], "filling": tuple(smearing),
          "shell_resolved": o["shell_resolved"], "scc_tol": o["scc_tol"], "max_scc": max_scc,
          "repulsive": o["repulsive"], "charge": float(atoms.get_initial_charges().sum()),
          "extra_ham": extra, "klines": klines, "read_charges": read,
          "convergent_only": convergent, "extra_analysis": extra_analysis, "driver": driver,
          "extra_top": extra_top, "forces": want_forces, "autotest": autotest,
          "extra_options": extra_options,
          "fermi": _fermi(o["fixed_fermi_level"], o["independent_k_filling"], collinear),
          "selected_shells": sel or None}
    c = Casida.make(o["casida"]) if band_kpts is None else None
    excited = bool(c is not None and c.state_of_interest != 0 and c.excited_forces)
    return Input(hsd=_dp.make_hsd(**kw), labels=labels, positions=atoms.positions.copy(),
                 cell=atoms.cell.array.copy() if periodic else None, make_hsd=kw, files=files,
                 analysis=analysis, kpoints=kpoints, excited=excited,
                 spin="two-component" if two_component(o) else
                 ("collinear" if collinear else "none"),
                 binary=o["binary"], threads=o["threads"])


def _fermi(fixed, independent_k, collinear):
    out = {}
    v = _fermi_au(fixed)
    if v is not None:
        v = np.atleast_1d(np.asarray(v, float))
        if collinear and v.size == 1:
            v = np.repeat(v, 2)
        out["fixed"] = v.tolist() if v.size > 1 else float(v[0])
    if independent_k:
        out["independent_k"] = True
    return out or None


# ---- results -----------------------------------------------------------------------------------
def energies(out: dict):
    """(total energy, Mermin free energy, extrapolated to 0 K) in Hartree: detailed.out's (10
    decimals), each replaced by results.tag's full-precision value where that is the same
    quantity (it agrees within the printed digits); the total as 2 E_0 - F when both tags are.
    results.tag's mermin_energy is the force-related energy with a fixed Fermi level, and the
    ELSI build (24.1) writes only total_energy."""
    tot = out.get("energy")
    mer = out.get("free_energy", tot)
    ez = out.get("e_zero", tot)
    t_mer, t_e0, t_tot = out.get("mermin_tag"), out.get("extrapolated0_tag"), out.get("total_tag")
    good_mer = t_mer is not None and mer is not None and abs(t_mer - mer) < PRINTED
    good_e0 = t_e0 is not None and ez is not None and abs(t_e0 - ez) < PRINTED
    if good_mer:
        mer = t_mer
    if good_e0:
        ez = t_e0
    if t_tot is not None and tot is not None and abs(t_tot - tot) < PRINTED:
        tot = t_tot
    elif good_mer and good_e0 and tot is not None and abs(2.0 * t_e0 - t_mer - tot) < PRINTED:
        tot = 2.0 * t_e0 - t_mer
    return tot, mer, ez


@dataclass
class Excitations:
    """DFTB+'s Casida excitations (TRA.DAT, EXC.DAT): energies (Hartree), oscillator strengths
    and symmetries ("S"/"T", or "" with spin polarisation) of every state, in DFTB+'s order."""
    energies: np.ndarray
    oscillator_strengths: np.ndarray
    symmetries: list
    rows: list                         # EXC.DAT rows as parsed


def _excitations(out) -> Excitations | None:
    if "exc_energies" not in out:
        return None
    rows = out.get("exc_rows") or []
    osc = np.array([r[1] for r in rows], float) if rows else np.full(len(out["exc_energies"]),
                                                                     np.nan)
    return Excitations(np.asarray(out["exc_energies"], float), osc,
                       list(out.get("exc_symmetries") or []), rows)


def result_of(out: dict, inp: Input, q0: np.ndarray) -> DFTBResult:
    """DFTB+'s parsed outputs as a :class:`pydftb.calculator.DFTBResult` (atomic units)."""
    tot, mer, _ = energies(out)
    ev, occ = out.get("eigenvalues"), out.get("filling")
    if inp.kpoints is not None:
        k, w = inp.kpoints
    else:
        k, w = np.zeros((1, 3)), np.ones(1)
    magmoms = fermi_spin = None
    if out.get("spins") is not None:
        s = np.asarray(out["spins"], float)
        magmoms = s[0] if s.shape[0] == 1 else s.T
    levels = np.atleast_1d(out["fermi_levels"])
    if inp.spin == "collinear":
        fermi_spin = (float(levels[0]), float(levels[-1]))
    r = DFTBResult(
        energy=tot, free_energy=mer, e_band=out.get("e_band", 0.0), e_scc=out.get("e_scc", 0.0),
        e_rep=out.get("e_rep", 0.0), ts=tot - mer, fermi=float(levels[0]),
        charges=q0 - np.asarray(out["charges_net"], float),
        eigenvalues=None if ev is None else list(np.asarray(ev).reshape(-1, np.shape(ev)[-1])),
        forces=None if out.get("forces") is None else np.asarray(out["forces"], float),
        n_scc=out.get("n_scc") or 0,
        occupations=None if occ is None else list(np.asarray(occ).reshape(-1, np.shape(occ)[-1])),
        e_3rd=out.get("e_3rd", 0.0),
        stress=None if out.get("stress") is None else np.asarray(out["stress"], float),
        kpoints=np.asarray(k, float), kweights=np.asarray(w, float),
        e_spin=out.get("e_spin", 0.0), magmoms=magmoms, fermi_spin=fermi_spin,
        e_disp=out.get("e_disp", 0.0), e_ls=out.get("e_ls", 0.0),
        e_plus_u=out.get("e_plus_u", 0.0), e_onsite=out.get("e_onsite", 0.0),
        e_ext=out.get("e_ext", 0.0), e_solv=out.get("e_solv", 0.0),
        point_charge_forces=(None if out.get("point_charge_forces") is None
                             else np.asarray(out["point_charge_forces"], float)),
        dipole=(None if inp.cell is not None or out.get("dipole_tag") is None
                else np.asarray(out["dipole_tag"], float)),
        excitations=_excitations(out), excitation_energy=out.get("excitation_energy"),
        scc_errors=np.asarray(out["scc_errors"]) if out.get("scc_errors") else None,
        e_halogen=out.get("e_halogen", 0.0), e_fock=out.get("e_fock", 0.0))
    if out.get("e_force_related") is not None:
        r.n_ef = mer - out["e_force_related"]
    if r.excitation_energy is not None:
        r.excited_energy = r.energy + r.excitation_energy
        r.excited_free_energy = (out["excited_mermin_tag"] if out.get("excited_mermin_tag")
                                 is not None else r.free_energy + r.excitation_energy)
    return r


class DFTBPlus(Calculator):
    """DFTB+ with :class:`pydftb.ase_calc.PyDFTB`'s options (see the module docstring); the
    DFTB+-only options are ``binary``, ``ewald_tolerance``, ``convergent_scc_only``,
    ``regions``, ``workdir``, ``keep_files`` and the ``extra_*`` HSD text."""
    implemented_properties = PyDFTB.implemented_properties
    engine = "dftbplus"

    def __init__(self, skdir, **options):
        check_options(options)
        super().__init__()
        self.skdir = skdir
        self.options = dict(options)
        if self.options.get("solvation") is not None:      # a relative parameter file: from here
            self.options["solvation"] = _solvation(self.options["solvation"])
        self._cache = tempfile.mkdtemp(prefix="pydftb_dftbplus_skf_")
        weakref.finalize(self, shutil.rmtree, self._cache, True)
        self.raw = None                  # DFTBResult of the last calculation (atomic units)
        self.output = None               # everything parsed from DFTB+'s outputs
        self.input = None                # the Input of the last run
        self.calls = 0                   # DFTB+ runs
        self._charges = None             # (labels, (file name, bytes)) of the last SCC run
        self._q0 = {}                    # reference populations per atom, by type labels

    # ---- options -------------------------------------------------------------------------
    def _opt(self, name):
        if name in self.options:
            return self.options[name]
        return OPTION_DEFAULTS.get(name, DFTBPLUS_DEFAULTS.get(name))

    @property
    def excited(self) -> bool:
        """Energy and forces of a Casida state of interest (as PyDFTB.excited)."""
        c = Casida.make(self._opt("casida"))
        return c is not None and c.state_of_interest != 0 and c.excited_forces

    def _labels(self, atoms):
        return list(self._opt("labels") or atoms.get_chemical_symbols())

    def _input(self, atoms, properties, overrides=None, **kw) -> Input:
        opts = {k: v for k, v in self.options.items()
                if k not in ("read_charges", "reuse_charges", "workdir", "keep_files")
                and k not in IGNORED}
        opts.update(overrides or {})
        return write_input(atoms, self.skdir, opts, properties, cache=self._cache, **kw)

    # ---- running DFTB+ -------------------------------------------------------------------
    def binary(self, inp: Input | None = None) -> str:
        """The DFTB+ executable: ``binary``, else ``$DFTBPLUS``, else ``dftb+`` on PATH."""
        b = (inp.binary if inp is not None else None) or self._opt("binary") or _dp.DFTBPLUS
        if not (b and (os.path.exists(b) or shutil.which(b))):
            raise FileNotFoundError(
                f"DFTB+ not found ({b!r}): pass binary=..., set $DFTBPLUS or put dftb+ on PATH "
                "(conda install -c conda-forge dftbplus)")
        return b

    def run_input(self, inp: Input, collect=None) -> dict:
        """Write ``inp`` into a work directory, run DFTB+, parse its outputs; ``collect(wd)``
        reads more from the directory before it is removed (into ``out["collected"]``)."""
        workdir, keep = self._opt("workdir"), self._opt("keep_files")
        wd = os.path.abspath(workdir) if workdir else tempfile.mkdtemp(prefix="pydftb_dftbplus_")
        os.makedirs(wd, exist_ok=True)
        try:
            _dp.write_gen(os.path.join(wd, "geo.gen"), inp.labels, inp.positions, inp.cell)
            with open(os.path.join(wd, "dftb_in.hsd"), "w") as fh:
                fh.write(inp.hsd)
            for name, blob in inp.files.items():
                with open(os.path.join(wd, name), "wb") as fh:
                    fh.write(blob)
            env = dict(os.environ)
            if inp.threads is not None:
                env.update(OMP_NUM_THREADS=str(inp.threads), OPENBLAS_NUM_THREADS=str(inp.threads))
            t0 = time.perf_counter()
            p = subprocess.run([self.binary(inp)], cwd=wd, env=env, capture_output=True,
                               text=True, check=False)
            wall = time.perf_counter() - t0
            self.calls += 1
            if p.returncode != 0 or not os.path.exists(os.path.join(wd, "detailed.out")):
                raise CalculationFailed("DFTB+ stopped:\n" + _dftbplus_error(p.stdout, p.stderr))
            kw = inp.make_hsd
            out = _dp.parse_run(wd, p.stdout, kw["scc"], inp.analysis, kw["autotest"],
                                bool(kw["extra_top"]), bool(kw["driver"]))
            out.update(wall=wall, hsd=inp.hsd, charges_file=None, dir=wd if (workdir or keep)
                       else None)
            for name in ("charges.bin", "charges.dat"):   # DFTB+ rewrites the one it read
                path = os.path.join(wd, name)
                if os.path.exists(path):
                    with open(path, "rb") as fh:
                        out["charges_file"] = (name, fh.read())
            if collect is not None:
                out["collected"] = collect(wd)
            return out
        finally:
            if workdir is None and not keep:
                shutil.rmtree(wd, ignore_errors=True)

    def _start_charges(self, labels):
        """The charge file the next SCC starts from: the user's ``read_charges`` for the first
        calculation, else with ``reuse_charges`` the last run's (same type labels)."""
        rc = self._opt("read_charges")
        if self.raw is None and rc is not None:
            with open(rc, "rb") as fh:
                return ("charges.dat" if str(rc).endswith(".dat") else "charges.bin", fh.read())
        if self._opt("reuse_charges") and self._charges is not None and self._charges[0] == labels:
            return self._charges[1]
        return None

    def _keep_charges(self, labels, out):
        if out.get("charges_file") is not None:
            self._charges = (labels, out["charges_file"])
        elif self._opt("scc"):
            self._charges = None          # e.g. none for non-collinear hybrids (DFTB+ 25.1)

    # ---- ASE ---------------------------------------------------------------------------------
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        if a.pbc.any() and "stress" in properties and self._opt("hybrid") is not None:
            raise PropertyNotImplementedError(
                "no stress with a hybrid functional (DFTB+ refuses lattice optimisation and "
                "barostats with one; pydftb neither)")
        labels = self._labels(a)
        inp = self._input(a, properties, charges=self._start_charges(labels))
        self.input = inp
        out = self.run_input(inp)
        self.output = out
        kw = inp.make_hsd
        key = tuple(labels)
        if key not in self._q0:
            self._q0[key] = neutral_populations(kw["skdir"], labels, kw["max_l"],
                                                kw["selected_shells"] or {},
                                                self._opt("reference_occupations"))
        self._store(result_of(out, inp, self._q0[key]), out, inp)
        self._keep_charges(labels, out)

    def _store(self, r: DFTBResult, out: dict, inp: Input):
        """``self.results`` (ASE units) as PyDFTB._store converts a DFTBResult."""
        self.raw = r
        if inp.excited:
            self.results["energy"] = r.excited_energy * units.Hartree
            self.results["free_energy"] = r.excited_free_energy * units.Hartree
        else:
            self.results["energy"] = r.energy * units.Hartree
            self.results["free_energy"] = r.e_force_related * units.Hartree
        self.results["charges"] = np.asarray(out["charges_net"], float).copy()
        if r.forces is not None:
            self.results["forces"] = r.forces * units.Hartree / units.Bohr
        if r.stress is not None:
            self.results["stress"] = (-r.stress * units.Hartree
                                      / units.Bohr ** 3).flat[[0, 4, 8, 5, 2, 1]]
        if r.dipole is not None:
            self.results["dipole"] = r.dipole * units.Bohr
        atom_e = ((out.get("analysis") or {}).get("atom_energies") or {}).get("total")
        if atom_e is not None and not inp.excited:
            self.results["energies"] = np.asarray(atom_e, float) * units.Hartree
        mm = np.zeros(len(inp.labels)) if r.magmoms is None else r.magmoms
        self.results["magmoms"] = mm
        self.results["magmom"] = (float(mm.sum()) if np.ndim(mm) == 1
                                  else float(np.linalg.norm(mm.sum(axis=0))))

    # ---- accessors, as PyDFTB --------------------------------------------------------------
    @property
    def excitations(self):
        """DFTB+'s Casida excitations of the last calculation (:class:`Excitations`)."""
        return None if self.raw is None else self.raw.excitations

    def _raw(self):
        if self.raw is None:
            self.get_potential_energy()
        return self.raw

    def get_number_of_spins(self):
        return 1 if self._raw().fermi_spin is None else 2

    def _index(self, kpt, spin):
        return kpt + spin * len(self._raw().kweights)

    def get_ibz_k_points(self):
        return np.array(self._raw().kpoints)

    def get_k_point_weights(self):
        return np.array(self._raw().kweights)

    def get_eigenvalues(self, kpt=0, spin=0):
        """Eigenvalues (eV) at k-point index ``kpt`` (and ``spin`` 0 = up, 1 = down)."""
        r = self._raw()
        if r.eigenvalues is None:
            raise RuntimeError("eigenvalues are not available with this solver")
        return np.asarray(r.eigenvalues[self._index(kpt, spin)]) * units.Hartree

    def get_occupation_numbers(self, kpt=0, spin=0):
        r = self._raw()
        if r.occupations is None:
            raise RuntimeError("occupations are not available with this solver")
        return np.asarray(r.occupations[self._index(kpt, spin)])

    def get_fermi_level(self):
        """Fermi level (eV); with two spin Fermi levels, that of spin up."""
        return self._raw().fermi * units.Hartree

    def get_fermi_levels(self):
        r = self._raw()
        return np.array(r.fermi_spin if r.fermi_spin is not None else [r.fermi]) * units.Hartree

    def masses(self, labels=None, atoms=None) -> np.ndarray:
        """Masses (amu) of the type labels (default: of ``atoms``, else the last calculation's)
        from their homonuclear SK files, as DFTB+."""
        if labels is None:
            labels = self._labels(self.atoms if atoms is None else atoms)
        plain = plain_sk_dir(self.skdir, self._cache)
        return np.array([float(SKFile.read(os.path.join(plain, f"{lab}-{lab}.skf")).mass)
                         for lab in labels])

    def write_charges(self, path):
        """DFTB+'s charge file of the last SCC (charges.bin, or charges.dat as text)."""
        if self._charges is None:
            raise RuntimeError("no charge file: run an SCC calculation first (DFTB+ writes "
                               "none for non-collinear hybrid functionals)")
        with open(path, "wb") as fh:
            fh.write(self._charges[1][1])

    # ---- second runs ---------------------------------------------------------------------
    def band_input(self, atoms, band_kpts, regions=None, charges=None) -> Input:
        """The input of a band run (:meth:`states`) at ``charges`` (file name, bytes)."""
        return self._input(atoms, ("energy",),
                           overrides={"regions": regions, "analysis": None, "casida": None},
                           charges=charges, band_kpts=band_kpts)

    def states(self, kpts=None, regions=None, atoms=None):
        """Eigenstates at the SCC charges by DFTB+'s band run (ReadInitialCharges, one
        iteration), as :meth:`pydftb.ase_calc.PyDFTB.states`: ``kpts`` None (the SCC k-points),
        any form the ``kpts`` option takes (a band path included), or ``{"klines": [(n, (k1,
        k2, k3)), ...]}`` (DFTB+'s Klines); ``regions`` (DFTB+'s ProjectStates regions) give
        ``region_weights`` {file name: (n_spin, n_k, n_state)}.  Returns a
        :class:`pydftb.bands.States` without eigenvectors."""
        from .bands import States
        from .bands import klines as kline_points
        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise ValueError("no atoms: attach the calculator or pass atoms")
        self.get_potential_energy(atoms)
        a = self.atoms
        periodic = bool(a.pbc.any())
        if kpts is not None and not periodic:
            raise ValueError("k-points need a periodic system")
        labels = self._labels(a)
        scc_fermi = np.array(self.raw.fermi_spin if self.raw.fermi_spin is not None
                             else [self.raw.fermi])
        if not periodic:
            band = k_used = (np.zeros((1, 3)), np.ones(1))
        elif kpts is None:
            band = k_used = self.input.kpoints
        elif isinstance(kpts, dict) and "klines" in kpts:
            band = {"klines": kpts["klines"]}
            pts = kline_points(kpts["klines"])[0]
            k_used = (pts, np.full(len(pts), 1.0 / len(pts)))
        else:
            band = k_used = kpoints_of(a, kpts, self._opt("time_reversal"),
                                       two_component({**OPTION_DEFAULTS, **self.options}))
        charges = self._charges[1] if self._charges is not None else None
        inp = self.band_input(a, band, regions, charges)
        files = [] if not regions else [name for name, *_ in _dp.region_files(
            regions, [self._region_shells(labels, r) for r in regions])]

        def collect(wd):
            return {name: _dp.parse_levels(os.path.join(wd, name))[1][..., 1] for name in files}

        out = self.run_input(inp, collect=collect if files else None)
        st = States(kpts=np.asarray(k_used[0], float), kweights=np.asarray(k_used[1], float),
                    eigenvalues=np.asarray(out["eigenvalues"], float),
                    occupations=np.asarray(out["filling"], float),
                    fermi=np.atleast_1d(out["fermi_levels"]).astype(float), scc_fermi=scc_fermi,
                    basis=None, region_weights=out.get("collected"))
        return st

    def _region_shells(self, labels, region):
        ml = self.input.make_hsd["max_l"]
        return list(range(ml[labels[region["atoms"][0]]] + 1))

    def band_structure(self, path=None, npoints=None, atoms=None):
        """ASE ``BandStructure`` (eV, zero = SCC Fermi level of spin up) along ``path`` (ASE
        ``BandPath`` or special points such as "GXWLGK"), as :meth:`PyDFTB.band_structure`."""
        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise ValueError("no atoms: attach the calculator or pass atoms")
        if path is None and npoints is None:
            return super().band_structure()
        if not hasattr(path, "special_points"):
            path = atoms.cell.bandpath(path, npoints=npoints)
        st = self.states(kpts=path, atoms=atoms)
        return st.band_structure(path, reference=self.get_fermi_level(), ev=units.Hartree)

    def hessian_input(self, atoms, delta: float = 1e-4, derivative_atoms=None,
                      moved_atoms=None, polarisability: bool | None = None) -> Input:
        """The input of DFTB+'s SecondDerivatives run (:meth:`second_derivatives`): DFTB+'s
        defaults are not written (Delta 1e-4 Bohr, all atoms); the analysis is the
        polarisability's alone when there are its derivatives to take, else none."""
        spec = {}
        if float(delta) != 1e-4:
            spec["delta"] = float(delta)
        if derivative_atoms is not None:
            spec["atoms"] = np.unique(derivative_atoms).tolist()
        if moved_atoms is not None:
            spec["moved_atoms"] = np.unique(moved_atoms).tolist()
        pol = polarisability
        if pol is None:
            pol = bool((_analysis(self._opt("analysis")) or {}).get("polarisability"))
        return self._input(atoms, ("energy", "forces"),
                           overrides={"analysis": {"polarisability": True} if pol else None},
                           driver=_dp.hessian_driver(spec))

    def second_derivatives(self, atoms=None, delta: float = 1e-4, derivative_atoms=None,
                           moved_atoms=None, dipole: bool = True,
                           polarisability: bool | None = None):
        """DFTB+'s ``Driver = SecondDerivatives`` (``delta`` in Bohr; atoms 0-based) as
        :class:`pydftb.vibrations.SecondDerivatives`, what
        :meth:`pydftb.ase_calc.PyDFTB.second_derivatives` gives; with ``polarisability`` (None:
        when the analysis has it) also its derivatives (bornderiv.out)."""
        from .vibrations import SecondDerivatives
        a = (self.atoms if atoms is None else atoms).copy()
        n = len(a)
        rows = np.arange(n) if derivative_atoms is None else np.unique(derivative_atoms)
        moved = rows if moved_atoms is None else np.unique(moved_atoms)
        inp = self.hessian_input(a, delta, derivative_atoms, moved_atoms, polarisability)

        def collect(wd):
            with open(os.path.join(wd, "detailed.out")) as fh:
                return _dp.parse_second_derivatives(wd, fh.read())

        out = self.run_input(inp, collect=collect)
        hd = out["collected"]
        born = hd.get("born")
        dmu = None if (born is None or not dipole or a.pbc.any()) else np.asarray(born, float)
        dpol = (None if hd.get("born_derivs") is None     # bornderiv.out: (col, i, j)
                else np.asarray(hd["born_derivs"], float).reshape(-1, 3, 3).transpose(0, 2, 1))
        tot, _, _ = energies(out)
        return SecondDerivatives(rows, moved, float(delta), np.asarray(hd["hessian"], float),
                                 dmu, dpol, float(tot))

    # ---- DFTB+'s own drivers -------------------------------------------------------------
    def relax_input(self, atoms, fmax: float = 0.01, steps: int = 3000, cell: bool = False,
                    isotropic: bool = False, optimiser: str = "LBFGS {}") -> Input:
        """The input of DFTB+'s GeometryOptimisation (:meth:`relax`)."""
        grad = fmax / (units.Hartree / units.Bohr)
        return self._driver_input(atoms, _dp.opt_driver(optimiser, cell, grad, max_steps=steps,
                                                        isotropic=isotropic))

    def relax(self, atoms, fmax: float = 0.01, steps: int = 3000, cell: bool = False,
              isotropic: bool = False, optimiser: str = "LBFGS {}"):
        """DFTB+'s GeometryOptimisation, one DFTB+ run: converged when the largest gradient
        component is below ``fmax`` (eV/A, as DFTB+'s GradElem in Ha/Bohr).  Returns (the
        relaxed copy of ``atoms``, the parsed outputs: ``geometry_steps``,
        ``converged_geometry``, ``scc_iterations``)."""
        out = self._driver_out(self.relax_input(atoms, fmax, steps, cell, isotropic, optimiser))
        _, pos, c = out["final_geometry"]
        new = atoms.copy()
        if c is not None:
            new.set_cell(c, scale_atoms=False)
        new.positions = pos
        return new, out

    def md_input(self, atoms, steps: int, time_step: float, velocities=None, xlbomd=None,
                 keep_stationary: bool = False, force_evaluation: str | None = None) -> Input:
        """The input of DFTB+'s VelocityVerlet (:meth:`md`)."""
        if velocities is None:
            velocities = atoms.get_velocities() * 1000.0 * units.fs
        if force_evaluation is None and xlbomd is not None:    # as PyDFTB's default
            from .constants import MIN_TEMP
            from .occupations import Smearing
            sm = self._opt("smearing") or ("fermi", self._opt("temperature") or 0.0)
            force_evaluation = ("dynamics" if Smearing.from_spec(tuple(sm)).kt > MIN_TEMP
                                else "dynamics_t0")
        extra = ""
        if force_evaluation is not None:
            name = {"traditional": "Traditional", "dynamics_t0": "DynamicsT0",
                    "dynamics": "Dynamics"}[force_evaluation]
            extra = f"  ForceEvaluation = {name}\n"
        drv = _dp.md_driver(velocities, time_step, steps, keep_stationary=keep_stationary,
                            xlbomd=None if xlbomd is None else dict(xlbomd))
        return self._driver_input(atoms, drv, extra)

    def md(self, atoms, steps: int, time_step: float, velocities=None, xlbomd=None,
           keep_stationary: bool = False, force_evaluation: str | None = None) -> dict:
        """DFTB+'s VelocityVerlet (NVE, ``time_step`` fs; ``velocities`` A/ps, default the
        atoms'), with XL-BOMD for ``xlbomd`` (:class:`pydftb.xlbomd.XLBOMD`'s options): one
        DFTB+ run.  Returns the parsed outputs with ``md`` (potential, kinetic and total
        energies, Hartree) and ``trajectory`` (DFTB+'s frames: positions A, velocities A/ps)."""
        return self._driver_out(self.md_input(atoms, steps, time_step, velocities, xlbomd,
                                              keep_stationary, force_evaluation))

    def _driver_input(self, atoms, driver, extra_ham="") -> Input:
        return self._input(atoms, ("energy", "forces"),
                           overrides={"extra_hamiltonian": self._opt("extra_hamiltonian")
                                      + extra_ham},
                           charges=self._start_charges(self._labels(atoms)), driver=driver)

    def _driver_out(self, inp: Input) -> dict:
        out = self.run_input(inp)
        if inp.excited and out.get("excited_mermin_tag") is not None:
            out["free_energy"] = out["excited_mermin_tag"]     # E0 + w, as DFTB+'s md.out
        return out


def _dftbplus_error(stdout: str, stderr: str) -> str:
    """DFTB+'s ERROR block (or the end of its output)."""
    text = stdout + "\n" + stderr
    i = text.find("ERROR!")
    return text[i:i + 3000] if i >= 0 else text[-3000:]
