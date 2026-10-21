"""The rules of the DFTB+ configuration advisor.

Each rule turns one pitfall found while validating pydftb against DFTB+ 25.1
(``docs/dftbplus_map.md``, section 3) into a check of a run: its input, the processed
input DFTB+ wrote (``dftb_pin.hsd``), its screen output, its SKF files and the
environment.  A rule returns findings; it never changes anything.
"""
from __future__ import annotations

import itertools
import math
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .inputs import CURRENT_PARSER, RunFiles

SEVERITIES = ("error", "warning", "info")
PTBP_LEGACY = "ptbp-legacy"     # PTBP's historical DFTB+ input (written by ASE, --smearing none)
PTBP_EXPECTED = ("PTBP's published parameters were fitted with this DFTB+ input; keep it "
                 "for comparable results (pass --smearing for a real electronic temperature)")


@dataclass
class Finding:
    rule: str
    severity: str
    title: str
    message: str
    advice: str = ""
    where: str = ""
    expected: str | None = None       # why this is expected here (context), if it is

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Rule:
    id: str
    severity: str           # usual severity (a finding can be milder or stricter)
    title: str
    source: str             # pitfall number, report, DFTB+ source file
    explain: str
    check: Callable


RULES: dict = {}


def rule(id: str, severity: str, title: str, source: str, explain: str):
    def deco(fn):
        RULES[id] = Rule(id, severity, title, source, textwrap.dedent(explain).strip(), fn)
        return fn
    return deco


def run_rules(run: RunFiles, only=None) -> list:
    """All findings, errors first."""
    out = []
    for r in RULES.values():
        if only and r.id not in only:
            continue
        try:
            out.extend(r.check(run) or [])
        except Exception as err:                      # noqa: BLE001  (a rule must not stop the others)
            out.append(Finding(r.id, "info", f"{r.title}: not checked",
                               f"the rule failed on this input ({type(err).__name__}: {err})"))
    order = {s: i for i, s in enumerate(SEVERITIES)}
    return sorted(out, key=lambda f: (order[f.severity], f.rule))


def _method(node):
    return node.method.name.lower() if node is not None and node.method is not None else None


def _out(run: RunFiles) -> str:
    """Name of the screen output file, for findings' locations."""
    return Path(run.stdout.path).name if run.stdout is not None else ""


# -- parser and input -------------------------------------------------------------------------
@rule("P01", "warning", "old ParserVersion: DFTB+ rewrites the input", "oldcompat.F90; M1",
      """
      DFTB+ converts inputs written for an older parser version (ParserOptions {
      ParserVersion = n }, ASE writes 1) step by step to the current version 14.  Some
      steps change the physics, silently except for a warning on the screen:
      version < 3 with SCC = Yes adds ShellResolvedSCC = Yes; version < 5 adds
      Differentiation = FiniteDiff { Delta = 0.01 } (the default step is 1.22e-4 Bohr;
      with 0.01 Bohr the forces of H2O and CH3OH are off by 4-7e-3 eV/A) and
      ShellResolvedSpin = Yes; version < 7 gives DftD3 the DFTB3-optimised defaults;
      version < 9 gives Driver = LBFGS a line search.  Write ParserVersion = 14 and set
      what you mean explicitly.
      """)
def p01(run: RunFiles):
    v = run.parser_version
    d = run.dftb
    if v >= CURRENT_PARSER or d is None:
        return []
    changes = []
    if v < 3 and run.scc and d.child("OrbitalResolvedSCC") is None and d.child("ShellResolvedSCC") is None:
        changes.append("ShellResolvedSCC = Yes is added (each shell gets its own charge and Hubbard U)")
    if v < 5 and d.child("Differentiation") is None:
        changes.append("Differentiation = FiniteDiff { Delta = 0.01 } is added: H0 and S are "
                       "differentiated with a 0.01 Bohr step instead of 1.22e-4 (H2O, CH3OH: "
                       "forces off by 4-7e-3 eV/A)")
    if v < 5 and d.child("SpinConstants") is not None:
        changes.append("SpinConstants gets ShellResolvedSpin = Yes")
    if v < 7 and _method(d.child("Dispersion")) == "dftd3":
        changes.append("DftD3 takes the DFTB3-optimised defaults (s6 = 1, s8 = 0.5883, a1 = 0.5719, "
                       "a2 = 3.6017) for parameters not given")
    if v < 9 and _method(run.inp.find("Driver")) == "lbfgs":
        changes.append("Driver = LBFGS gets LineSearch = Yes and oldLineSearch = Yes")
    if not changes:
        return []
    node = run.inp.find("ParserOptions/ParserVersion")
    return [Finding("P01", "warning", RULES["P01"].title,
                    f"ParserVersion = {v}: on the way to version {CURRENT_PARSER}, " + "; ".join(changes) + ".",
                    "write ParserVersion = 14 and set these keys to the values you mean",
                    node.where if node is not None else "", PTBP_EXPECTED if run.context == PTBP_LEGACY else None)]


@rule("P02", "warning", "Methfessel-Paxton order is lowered by one", "pitfall 1; oldcompat.F90",
      """
      With ParserVersion < 14, Filling = MethfesselPaxton { Order = n } is converted to
      Order = n - 1: n = 2 runs as first order, n = 1 becomes Gaussian smearing and
      n <= 0 stops DFTB+.  Without Order nothing changes.
      """)
def p02(run: RunFiles):
    d = run.dftb
    if d is None or run.parser_version >= CURRENT_PARSER:
        return []
    filling = d.child("Filling")
    if _method(filling) != "methfesselpaxton":
        return []
    order_node = filling.method.child("Order")
    if order_node is None:
        return []
    n = int(order_node.number())
    if n > 1:
        sev, what = "warning", f"runs as Order = {n - 1}"
    elif n == 1:
        sev, what = "warning", "becomes Gaussian smearing (order 0)"
    else:
        sev, what = "error", f"becomes Order = {n - 1}: DFTB+ stops"
    return [Finding("P02", sev, RULES["P02"].title,
                    f"ParserVersion = {run.parser_version}: MethfesselPaxton Order = {n} {what}.",
                    f"write ParserVersion = 14 (then Order = {n} means order {n})", order_node.where)]


_SMEARING = ("fermi", "methfesselpaxton", "gaussian")


def _filling_used(run: RunFiles) -> str | None:
    pin = run.pin_dftb
    if pin is None:
        return None
    f = pin.child("Filling")
    if f is None or f.method is None:
        return None
    t = f.method.number("Temperature")
    return f"{f.method.name} with T = {t:g} Ha" + (" (zero temperature)" if t == 0 else "")


@rule("P03", "warning", "input ignored by the parser", "M1; hsdutils2.F90",
      """
      With ParserOptions { IgnoreUnprocessedNodes = Yes } (ASE writes it) DFTB+ skips
      every node it does not read, with only a warning on the screen.  A Fermi,
      Gaussian or MethfesselPaxton block placed directly in the Hamiltonian instead of
      in Filling is such a node: the calculation then runs at zero electronic
      temperature (PTBP's historical input does this).  Without the option DFTB+ stops.
      """)
def p03(run: RunFiles):
    d = run.dftb
    if d is None:
        return []
    items = []                                      # (path, lines)
    if run.stdout is not None and run.stdout.ignored:
        items = [(m.path, m.lines) for m in run.stdout.ignored]
    else:
        items = [(c.path, f"{c.line}-{c.end_line}") for c in d.children if c.name.lower() in _SMEARING]
    out = []
    for path, lines in items:
        last = path.split("/")[-1].lower()
        smearing = last in _SMEARING and path.lower().startswith("hamiltonian/dftb/") \
            and "/filling/" not in path.lower()
        where = f"dftb_in.hsd:{lines}" if lines else ""
        sev = "warning" if run.ignore_unprocessed else "error"
        if smearing:
            used = _filling_used(run)
            msg = (f"{path} is not inside Filling and is ignored: its temperature has no effect"
                   + (f"; DFTB+ used Filling = {used}" if used else "; DFTB+ uses its default filling "
                      "(Fermi, zero temperature)"))
            advice = f"move it into Filling: Filling = {path.split('/')[-1]} {{ Temperature [K] = ... }}"
            expected = PTBP_EXPECTED if run.context == PTBP_LEGACY else None
        else:
            msg = f"{path} is not a keyword DFTB+ reads here and is ignored"
            advice = "check the spelling and the block it belongs in"
            expected = None
        if not run.ignore_unprocessed:
            msg += " (without IgnoreUnprocessedNodes = Yes DFTB+ stops)"
        out.append(Finding("P03", sev, RULES["P03"].title, msg, advice, where, expected))
    return out


@rule("F01", "error", "temperature without a unit is read in Hartree", "parser.F90:readFilling",
      """
      Filling = Fermi { Temperature = 2000 } means 2000 Hartree (6e8 K): the electronic
      temperature is in atomic units unless a modifier is given.  Write
      Temperature [K] = 2000.  Values up to about 0.5 are plausible Hartree values and
      only reported.
      """)
def f01(run: RunFiles):
    d = run.dftb
    filling = d.child("Filling") if d is not None else None
    if filling is None or filling.method is None:
        return []
    t = filling.method.child("Temperature")
    if t is None or t.modifier:
        return []
    value = t.number()
    if value is None or value <= 0:
        return []
    kelvin = value / 3.16681534524639e-6
    if value > 0.5:
        return [Finding("F01", "error", RULES["F01"].title,
                        f"Temperature = {t.text()} without a unit is {value:g} Ha = {kelvin:.3g} K.",
                        f"write Temperature [K] = {t.text()}", t.where)]
    return [Finding("F01", "info", RULES["F01"].title,
                    f"Temperature = {t.text()} without a unit is read as {value:g} Ha = {kelvin:.0f} K.",
                    "add the unit ([K] or [eV]) to make it explicit", t.where)]


# -- parameters ---------------------------------------------------------------------------------
# gamma error of DFTB+'s closed form for one atom pair at |U_a - U_b| (Ha), measured
_H01_TABLE = ((5e-6, 3e-3), (1e-5, 1e-4), (3e-5, 4e-6), (1e-4, 5e-8), (3e-4, 3e-9), (1e-3, 3e-10))
MIN_HUB_DIFF = 0.3125e-5


def _gamma_error(du: float) -> float:
    xs = [math.log(a) for a, _ in _H01_TABLE]
    ys = [math.log(b) for _, b in _H01_TABLE]
    x = min(max(math.log(du), xs[0]), xs[-1])
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1:
            return math.exp(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return math.exp(ys[-1])


@rule("H01", "warning", "Hubbard U values nearly equal", "pitfall 6; M2b",
      """
      DFTB+ evaluates the short-range gamma of two different U values with a closed
      formula whose terms cancel when the values are close.  Values closer than 3.1e-6
      Ha count as equal (exact formula); above that the error of gamma per atom pair is
      about 1e-4 Ha at |dU| = 1e-5, 4e-6 Ha at 3e-5, 5e-8 Ha at 1e-4 and 3e-10 Ha at
      1e-3 (measured).  With DFTB3 the U derivative is far worse (relative error 24 at
      1e-5).  Make close U values equal or move them apart.
      """)
def h01(run: RunFiles):
    if run.dftb is None or not run.scc or run.geometry is None:
        return []
    ml = run.max_l()
    values = []                                    # (label, U)
    for s in run.geometry.species:
        info = run.skf.get((s, s))
        if info is None or not info.hubbard:
            continue
        lmax = ml.get(s, max(info.hubbard))
        shells = [l for l in sorted(info.hubbard) if l <= lmax]
        if run.shell_resolved:
            values += [(f"{s} {'spdf'[l]}", info.hubbard[l]) for l in shells]
        elif shells:
            values.append((s, info.hubbard[shells[0]]))
    d = run.dftb
    dftb3 = bool(d.flag("ThirdOrderFull", False) or d.flag("ThirdOrder", False)
                 or d.child("HubbardDerivs") is not None)
    close = []
    for (la, ua), (lb, ub) in itertools.combinations(values, 2):
        du = abs(ua - ub)
        if MIN_HUB_DIFF <= du < (1e-3 if dftb3 else 3e-4):
            close.append((du, la, lb))
    if not close:
        return []
    close.sort()
    du = close[0][0]
    sev = "error" if du < 3e-5 or dftb3 else "warning"
    pairs = ", ".join(f"{a} / {b}: {x:.1e} Ha" for x, a, b in close[:5])
    msg = (f"{len(close)} pair(s) of Hubbard U values differ by less than "
           f"{'1e-3' if dftb3 else '3e-4'} Ha ({pairs}); DFTB+'s gamma is off by about "
           f"{_gamma_error(du):.0e} Ha per atom pair at {du:.1e} Ha"
           + ("; with DFTB3 the U derivative is much worse" if dftb3 else ""))
    return [Finding("H01", sev, RULES["H01"].title, msg,
                    "make the U values equal (|dU| < 3e-6 Ha) or at least 1e-3 Ha apart",
                    "SKF files")]


@rule("K01", "error", "SKF without a Spline repulsive", "pitfall 30; parser.F90:readSKFiles",
      """
      DFTB+ reads the repulsive from the Spline section of every SKF unless
      PolynomialRepulsive is set for that pair; a file without one (hotcent writes such
      files) stops DFTB+.
      """)
def k01(run: RunFiles):
    d = run.dftb
    if d is None:
        return []
    poly = d.child("PolynomialRepulsive")
    def polynomial(a, b) -> bool:
        if poly is None:
            return False
        if _method(poly) == "setforall":
            return bool(poly.flag("", False))
        node = poly.child(f"{a}-{b}") or poly.child(f"{b}-{a}")
        return bool(node is not None and node.flag("", False))
    bad = [Path(info.path).name for (a, b), info in sorted(run.skf.items())
           if info.exists and info.error is None and not info.has_spline and not polynomial(a, b)]
    if not bad:
        return []
    return [Finding("K01", "error", RULES["K01"].title,
                    f"{len(bad)} SKF file(s) without a Spline section and without "
                    f"PolynomialRepulsive for their pair ({', '.join(bad[:6])}"
                    f"{', ...' if len(bad) > 6 else ''}): DFTB+ stops reading the first one",
                    "add a Spline section, or PolynomialRepulsive = SetForAll { Yes } if the "
                    "files' polynomial (possibly all zeros) is what you want", bad[0])]


@rule("B01", "error", "band run without ConvergentSCCOnly = No", "pitfall 16; M2e",
      """
      With KPointsAndWeights = Klines DFTB+ sets MaxSCCIterations to 1.  For more than
      one atom per cell one iteration on the path k-points is never 'converged', and
      DFTB+ stops unless ConvergentSCCOnly = No (one atom per cell always converges).
      """)
def b01(run: RunFiles):
    d = run.dftb
    if d is None or not run.scc or _method(d.child("KPointsAndWeights")) != "klines":
        return []
    maxscc = d.number("MaxSCCIterations")
    if maxscc not in (None, 1):
        return []
    if not d.flag("ConvergentSCCOnly", True):
        return []
    natoms = run.geometry.natoms if run.geometry is not None else None
    if natoms == 1:
        return []
    node = d.child("KPointsAndWeights")
    return [Finding("B01", "error", RULES["B01"].title,
                    f"Klines with MaxSCCIterations = 1 and {natoms or 'several'} atoms: DFTB+ "
                    "stops with 'SCC is NOT converged'",
                    "add ConvergentSCCOnly = No (with ReadInitialCharges = Yes from a converged run)",
                    node.where)]


# -- parallel set-up ---------------------------------------------------------------------------
@rule("T01", "warning", "more threads than cores", "RESEARCH.md 4.4",
      """
      OpenMP threads x MPI processes x runs at the same time should not exceed the cores.
      DFTB+ uses every core when OMP_NUM_THREADS is not set, so several DFTB+ runs (or
      MPI processes) at once oversubscribe the machine and get much slower.
      """)
def t01(run: RunFiles):
    if not run.cores:
        return []
    omp = run.omp_threads()
    if omp is None and not run.threads_from_environment:
        return []                                  # DFTB+ stopped before printing them
    unset = omp is None
    omp = omp or run.cores
    procs = (run.stdout.mpi_procs if run.stdout is not None and run.stdout.mpi_procs else None) \
        or run.mpi_procs or 1
    blas = 1
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        raw = run.environ.get(var, "").strip()
        if raw.isdigit():
            blas = max(blas, int(raw))
    per_process = max(omp, blas)
    total = run.concurrent * procs * per_process
    if total <= run.cores:
        return []
    parts = [f"{per_process} thread(s) per process" + (" (OMP_NUM_THREADS not set: all cores)" if unset else "")]
    if procs > 1:
        parts.append(f"{procs} MPI processes")
    if run.concurrent > 1:
        parts.append(f"{run.concurrent} runs at the same time")
    return [Finding("T01", "warning", RULES["T01"].title,
                    f"{' x '.join(parts)} = {total} threads on {run.cores} cores",
                    "set OMP_NUM_THREADS so that threads x processes x concurrent runs <= cores",
                    _out(run) or "environment")]


@rule("T02", "warning", "MPI groups", "RESEARCH.md 4.2-4.3; mpienv.F90",
      """
      Parallel { Groups = n } distributes k-points and spins over n groups of MPI
      processes.  A DFTB+ built without MPI reads and ignores it; with MPI, n must divide
      the number of processes or DFTB+ stops.
      """)
def t02(run: RunFiles):
    groups_node = run.inp.find("Parallel/Groups")
    groups = int(groups_node.number(default=1)) if groups_node is not None else 1
    out = []
    no_mpi = run.stdout is not None and any("compiled without MPI" in m.text for m in run.stdout.messages)
    if no_mpi and groups > 1:
        out.append(Finding("T02", "warning", RULES["T02"].title,
                           f"Groups = {groups} is ignored: this DFTB+ was built without MPI, so "
                           "k-points are not distributed",
                           "use an MPI build (and mpirun) or OpenMP threads", groups_node.where))
    procs = (run.stdout.mpi_procs if run.stdout is not None else None) or run.mpi_procs
    if procs and groups > 1 and procs % groups:
        out.append(Finding("T02", "error", RULES["T02"].title,
                           f"Groups = {groups} does not divide the {procs} MPI processes: DFTB+ stops",
                           "use a number of processes that is a multiple of Groups",
                           groups_node.where if groups_node is not None else ""))
    return out


@rule("T03", "warning", "OpenMP threads on small matrices", "M3a, M3c",
      """
      On matrices of fewer than about 500 orbitals DFTB+ gets slower with OpenMP threads
      (measured, Cu cells: 36 orbitals 3.3x slower on 4 threads, 288 orbitals 1.2-1.3x
      slower; from 576 orbitals with k-points 1.4-1.7x faster).  Run small systems on one
      thread, several at a time if there are many.
      """)
def t03(run: RunFiles):
    norb = run.norb()
    if norb is None or norb >= 500:
        return []
    omp = run.omp_threads()
    if omp is None and (not run.threads_from_environment or run.environ.get("OMP_NUM_THREADS")):
        return []                                  # unknown
    threads = omp if omp is not None else (run.cores or 1)
    if threads <= 1:
        return []
    how = "" if omp is not None else " (OMP_NUM_THREADS not set: all cores)"
    return [Finding("T03", "warning", RULES["T03"].title,
                    f"{norb} orbitals on {threads} OpenMP threads{how}: slower than one thread "
                    "below about 500 orbitals",
                    "set OMP_NUM_THREADS=1 (and run independent structures side by side)",
                    _out(run) or "environment")]


@rule("T04", "info", "stack size limit", "envcheck.F90",
      """
      DFTB+ recommends an unlimited stack ('ulimit -s unlimited'); with the usual 8 MB
      large systems may crash at random (segmentation faults).
      """)
def t04(run: RunFiles):
    s = run.stdout
    if s is None or s.stack_unlimited or s.stack_mb is None:
        return []
    return [Finding("T04", "info", RULES["T04"].title,
                    f"DFTB+ ran with a stack of {s.stack_mb} MB",
                    "run 'ulimit -s unlimited' (or 'ulimit -s hard') before DFTB+", _out(run))]


# -- results ------------------------------------------------------------------------------------
@rule("X01", "error", "DFTB+ stopped with an error", "screen output",
      """
      The error DFTB+ printed before stopping, with the input line it names.  The other
      rules often explain the cause (K01: SKF without Spline, B01: band run, T02: MPI
      groups).
      """)
def x01(run: RunFiles):
    if run.stdout is None:
        return []
    out = []
    for m in run.stdout.errors:
        where = f"dftb_in.hsd:{m.lines}" if m.lines else _out(run)
        text = m.text + (f" ({m.path})" if m.path else "")
        out.append(Finding("X01", "error", RULES["X01"].title, text, "", where))
    return out



@rule("S01", "warning", "results from an unconverged SCC", "main.F90",
      """
      With ConvergentSCCOnly = No DFTB+ only warns when the SCC loop reaches
      MaxSCCIterations and returns the energy and forces of the last iteration.
      """)
def s01(run: RunFiles):
    d = run.dftb
    if d is None or run.stdout is None or not run.stdout.scc_not_converged:
        return []
    if d.flag("ConvergentSCCOnly", True):
        return []
    maxscc = d.text("MaxSCCIterations") or "the default"
    return [Finding("S01", "warning", RULES["S01"].title,
                    f"DFTB+ reports 'SCC is NOT converged' after {maxscc} iterations, and "
                    "ConvergentSCCOnly = No let it return the last iteration's energy and forces",
                    "check the structure; a finite electronic temperature (Filling) or a smaller "
                    "MixingParameter usually helps metals", _out(run))]


@rule("E01", "info", "default Ewald tolerance", "M2e; parser.F90",
      """
      The default EwaldTolerance of 1e-9 limits periodic SCC energies to about 1e-7 Ha
      (a 24-atom water box: 1.3e-7 Ha against 1e-12).  Set EwaldTolerance = 1e-12 when
      comparing at that level.
      """)
def e01(run: RunFiles):
    d = run.dftb
    if d is None or not run.scc or run.geometry is None or not run.geometry.periodic:
        return []
    if d.child("EwaldTolerance") is not None:
        return []
    return [Finding("E01", "info", RULES["E01"].title,
                    "periodic SCC with the default EwaldTolerance = 1e-9: energies are accurate to "
                    "about 1e-7 Ha",
                    "EwaldTolerance = 1e-12 for comparisons below 1e-7 Ha", "")]


# -- geometry optimisation -----------------------------------------------------------------------
_OLD_RELAX = ("conjugategradient", "steepestdescent", "gdiis", "lbfgs", "fire")


@rule("G01", "warning", "isotropic lattice optimisation of a non-orthogonal cell", "pitfall 25; M2f",
      """
      LatticeOpt = Yes with Isotropic = Yes replaces the stress by the pressure, but the
      lattice gradient p L^-T is not parallel to L when the cell is not orthogonal: the
      optimiser shears the cell while the convergence test only sees the pressure, and
      stops on a sheared cell (bcc primitive cell: 2.7e-7 Ha above the minimum).
      """)
def g01(run: RunFiles):
    drv = run.inp.find("Driver")
    m = drv.method if drv is not None else None
    if m is None or m.name.lower() not in ("geometryoptimisation", "geometryoptimization") + _OLD_RELAX:
        return []
    if not (m.flag("LatticeOpt", False) and m.flag("Isotropic", False)):
        return []
    g = run.geometry
    if g is None or g.orthogonal() is not False:
        return []
    node = m.child("Isotropic")
    return [Finding("G01", "warning", RULES["G01"].title,
                    "Isotropic = Yes on a non-orthogonal cell: the lattice steps shear the cell "
                    "and only the pressure is converged",
                    "optimise the conventional (orthogonal) cell, or drop Isotropic", node.where)]


@rule("G02", "info", "LBFGS without a step limit", "pitfall 24; M2f",
      """
      The LBFGS optimiser of the GeometryOptimisation driver has no step limit: from
      far-off starting points on soft energy surfaces the steps grow until the SCC
      diverges (water dimer: 0.2 -> 0.8 Bohr from step 12, stop at step 23).  Rational
      (the default) is stable but slow.
      """)
def g02(run: RunFiles):
    drv = run.inp.find("Driver")
    m = drv.method if drv is not None else None
    if m is None or m.name.lower() not in ("geometryoptimisation", "geometryoptimization"):
        return []
    opt = m.child("Optimiser") or m.child("Optimizer")
    if _method(opt) != "lbfgs":
        return []
    return [Finding("G02", "info", RULES["G02"].title,
                    "Optimiser = LBFGS has no step-size limit: on soft surfaces far from the minimum "
                    "the steps can grow until the SCC fails",
                    "start from a pre-relaxed structure, or use Rational/FIRE first", opt.where)]


# -- dispersion ----------------------------------------------------------------------------------
def _periodic_d4(run: RunFiles):
    d = run.dftb
    disp = d.child("Dispersion") if d is not None else None
    if _method(disp) != "dftd4" or run.geometry is None or not run.geometry.periodic:
        return None
    return disp.method


@rule("D02", "warning", "periodic D4 with EEQ charges", "pitfall 11; M2d",
      """
      DFTB+'s own D4 gives periodic EEQ charges that depend on the Ewald parameter (1.3e-4
      Ha between EwaldParameter 0.3 and the automatic value) and do not reach the
      molecular limit in large boxes: the EEQ Ewald sum looks wrong.  ChargeScale = 0
      switches the charge dependence off; the dftd4 library is correct.
      """)
def d02(run: RunFiles):
    m = _periodic_d4(run)
    if m is None:
        return []
    cm = m.child("ChargeModel")
    if cm is not None and _method(cm) not in (None, "eeq"):
        return []
    if m.number("ChargeScale", 3.0) == 0:
        return []
    return [Finding("D02", "warning", RULES["D02"].title,
                    "periodic DftD4 with EEQ charges (ChargeScale = "
                    f"{m.text('ChargeScale') or '3.0, the default'}): the charges depend on the "
                    "Ewald parameter and the dispersion energy is unreliable at ~1e-4 Ha",
                    "ChargeScale = 0, or the dftd4 library through another program", m.where)]


@rule("D03", "warning", "periodic D4 three-body term", "pitfall 12; M2d",
      """
      DFTB+'s own periodic D4 counts triples made of images of the same atom with the
      wrong weight: the same crystal in its primitive cell and in a supercell gives
      three-body energies per cell that differ by 10 %.  s9 = 0 switches the term off.
      """)
def d03(run: RunFiles):
    m = _periodic_d4(run)
    if m is None:
        return []
    s9 = m.number("s9")
    if s9 is not None and s9 == 0:
        return []
    return [Finding("D03", "warning", RULES["D03"].title,
                    f"periodic DftD4 with the three-body term (s9 = {m.text('s9') or '?'}): the "
                    "energy per cell depends on the cell chosen (10 % of the three-body energy)",
                    "s9 = 0, or the dftd4 library through another program",
                    m.child("s9").where if m.child("s9") is not None else m.where)]
