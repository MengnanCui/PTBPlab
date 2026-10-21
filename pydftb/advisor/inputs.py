"""Everything the rules look at: input, processed input, screen output, SKFs, environment."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import hsd

CURRENT_PARSER = 14                      # DFTB+ 24.1 / 25.1
L_OF = {"s": 0, "p": 1, "d": 2, "f": 3}
BANNER = re.compile(r"^\|\s+DFTB\+", re.M)


# -- screen output --------------------------------------------------------------------------
@dataclass
class Message:
    kind: str                # "warning" or "error"
    text: str
    path: str | None = None  # HSD path DFTB+ names (without "dftbplusinput/")
    lines: str | None = None # "10-12"


@dataclass
class StdoutFacts:
    """What DFTB+ printed: version, input conversion, ignored nodes, parallel set-up."""
    path: str
    version: str | None = None
    parser_version: int | None = None
    converted_from: int | None = None
    messages: list = field(default_factory=list)
    ignored: list = field(default_factory=list)          # [Message] of ignored nodes
    omp_threads: int | None = None
    mpi_procs: int | None = None
    mpi_groups: int | None = None
    stack_mb: int | None = None                          # None: unlimited or not printed
    stack_unlimited: bool = False
    scc_not_converged: bool = False

    @property
    def warnings(self):
        return [m for m in self.messages if m.kind == "warning"]

    @property
    def errors(self):
        return [m for m in self.messages if m.kind == "error"]


def _strip_root(path: str) -> str:
    return re.sub(r"^dftbplusinput/", "", path.strip())


def parse_stdout(text: str, path: str = "") -> StdoutFacts:
    facts = StdoutFacts(path)
    m = re.search(r"DFTB\+ (release [\w.]+|development version[^\n]*)", text)
    facts.version = m.group(1).strip() if m else None
    m = re.search(r"Parser version:\s*(\d+)", text)
    facts.parser_version = int(m.group(1)) if m else None
    m = re.search(r"Converting input from parser version\s+(\d+)", text)
    facts.converted_from = int(m.group(1)) if m else None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        head = lines[i].strip()
        if head not in ("WARNING!", "ERROR!"):
            i += 1
            continue
        kind = "warning" if head == "WARNING!" else "error"
        # the message: '-> text', indented continuation lines, 'Path:'/'Line:' lines, and
        # for ignored nodes a list of '(n)' / 'Path:' / 'Line:'
        body, j = [], i + 1
        while j < len(lines):
            raw = lines[j]
            s_ = raw.strip()
            if s_.startswith("->") or s_.startswith(("Path:", "Line:")) \
                    or re.fullmatch(r"\(\d+\)", s_) or (raw[:1].isspace() and s_ and body):
                body.append(s_)
                j += 1
                continue
            break
        text_parts, mpath, mline, items = [], None, None, []
        for b in body:
            if re.fullmatch(r"\(\d+\)", b):
                items.append([None, None])
            elif b.startswith("Path:"):
                if items:
                    items[-1][0] = _strip_root(b[5:])
                elif mpath is None:
                    mpath = _strip_root(b[5:])
            elif b.startswith("Line:"):
                lm = re.search(r"Line:\s*([\d-]+)", b)
                if items:
                    items[-1][1] = lm.group(1) if lm else None
                elif mline is None and lm:
                    mline = lm.group(1)
            else:
                text_parts.append(b.lstrip("->").strip())
        msg = Message(kind, " ".join(text_parts), mpath, mline)
        facts.messages.append(msg)
        if "ignored by the parser" in msg.text:
            facts.ignored.extend(Message("warning", "ignored by the parser", p_, l_)
                                 for p_, l_ in items if p_)
        i = j
    m = re.search(r"OpenMP threads:\s*(\d+)", text)
    facts.omp_threads = int(m.group(1)) if m else None
    m = re.search(r"MPI processes:\s*(\d+)(?:\s*\(split into\s*(\d+)\s*groups\))?", text)
    if m:
        facts.mpi_procs = int(m.group(1))
        facts.mpi_groups = int(m.group(2)) if m.group(2) else 1
    m = re.search(r"Current stacksize:\s*(\S+)", text)
    if m:
        if m.group(1).lower().startswith("unlimited"):
            facts.stack_unlimited = True
        elif m.group(1).isdigit():
            facts.stack_mb = int(m.group(1))
    facts.scc_not_converged = "SCC is NOT converged" in text
    return facts


# -- geometry ---------------------------------------------------------------------------------
@dataclass
class Geometry:
    species: list            # species names in order
    atom_species: list       # species name of every atom
    periodic: bool
    cell: np.ndarray | None = None      # Angstrom, rows

    @property
    def natoms(self) -> int:
        return len(self.atom_species)

    def orthogonal(self, tol: float = 1e-6) -> bool | None:
        if self.cell is None:
            return None
        a = self.cell / np.linalg.norm(self.cell, axis=1)[:, None]
        g = a @ a.T
        return bool(np.all(np.abs(g - np.diag(np.diag(g))) < tol))


def _floats(tokens):
    return [float(t.lower().replace("d", "e")) for t in tokens]


def geometry_from(root: hsd.Node) -> Geometry | None:
    """GenFormat (inline or <<<) or explicit TypeNames/TypesAndCoordinates; else None."""
    geo = root.find("Geometry")
    if geo is None:
        return None
    if geo.method is not None and geo.method.name.lower() == "genformat":
        rows = [tok for _, tok in geo.method.data]
        if len(rows) < 2:
            return None
        natom, kind = int(rows[0][0]), rows[0][1].upper()
        species = rows[1]
        atom_species = [species[int(r[1]) - 1] for r in rows[2:2 + natom]]
        cell = None
        if kind in ("S", "F") and len(rows) >= 2 + natom + 4:
            cell = np.array([_floats(r[:3]) for r in rows[3 + natom:6 + natom]])
        return Geometry(species, atom_species, kind in ("S", "F"), cell)
    names = geo.find("TypeNames")
    coords = geo.find("TypesAndCoordinates")
    if names is not None and coords is not None:
        species = (names.text() or "").replace('"', " ").split()
        atom_species = [species[int(tok[0]) - 1] for _, tok in coords.data]
        periodic = bool(geo.flag("Periodic", False))
        cell = None
        lat = geo.find("LatticeVectors")
        if periodic and lat is not None and len(lat.data) >= 3:
            cell = np.array([_floats(tok[:3]) for _, tok in lat.data[:3]])
            if (lat.modifier or "").lower() in ("bohr", "au"):
                cell = cell * 0.529177249
        return Geometry(species, atom_species, periodic, cell)
    return None


# -- Slater-Koster files ---------------------------------------------------------------------
@dataclass
class SKInfo:
    path: str
    exists: bool
    extended: bool = False
    hubbard: dict = field(default_factory=dict)   # l -> U (homonuclear files)
    has_spline: bool = False
    error: str | None = None


def _numbers(line: str) -> list:
    out = []
    for tok in line.replace(",", " ").split():
        if "*" in tok:
            n, v = tok.split("*", 1)
            out.extend([float(v)] * int(n))
        else:
            out.append(float(tok))
    return out


def read_skf_header(path, homonuclear: bool) -> SKInfo:
    info = SKInfo(str(path), Path(path).is_file())
    if not info.exists:
        return info
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
        info.extended = lines[0].lstrip().startswith("@")
        info.has_spline = any(ln.strip().lower().startswith("spline") for ln in lines)
        if homonuclear:
            on = _numbers(lines[2 if info.extended else 1])
            if info.extended:          # Ef Ed Ep Es SPE Uf Ud Up Us ff fd fp fs
                info.hubbard = {3: on[5], 2: on[6], 1: on[7], 0: on[8]}
            else:                      # Ed Ep Es SPE Ud Up Us fd fp fs
                info.hubbard = {2: on[4], 1: on[5], 0: on[6]}
    except (OSError, ValueError, IndexError) as err:
        info.error = f"{type(err).__name__}: {err}"
    return info


def skf_paths(root: hsd.Node, species, base: Path) -> dict:
    """(A, B) -> path, following SlaterKosterFiles (Type2FileNames or an explicit list)."""
    skn = root.find("Hamiltonian/DFTB/SlaterKosterFiles")
    if skn is None:
        return {}
    out = {}
    t2 = skn.method if skn.method is not None and skn.method.name.lower() == "type2filenames" else None
    for a in species:
        for b in species:
            if t2 is not None:
                lower = bool(t2.flag("LowerCaseTypeName", False))
                na, nb = (a.lower(), b.lower()) if lower else (a, b)
                name = (t2.text("Prefix") or "") + na + (t2.text("Separator") or "") + nb \
                    + (t2.text("Suffix") or "")
            else:
                node = skn.find(f"{a}-{b}")
                if node is None:
                    continue
                name = (node.text() or "").split()[0] if node.text() else ""
            p = Path(name)
            out[(a, b)] = p if p.is_absolute() else base / p
    return out


# -- the run -------------------------------------------------------------------------------
@dataclass
class RunFiles:
    """A DFTB+ run: its input (required) and whatever else is there."""
    dir: Path
    inp: hsd.Node
    pin: hsd.Node | None = None
    stdout: StdoutFacts | None = None
    geometry: Geometry | None = None
    skf: dict = field(default_factory=dict)             # (A, B) -> SKInfo
    environ: dict = field(default_factory=dict)
    cores: int | None = None
    mpi_procs: int | None = None                        # --mpi-procs
    concurrent: int = 1                                  # runs at the same time (task farm)
    context: str | None = None                           # "ptbp": PTBP's historical input
    notes: list = field(default_factory=list)            # what could not be read

    # -- convenient views ---------------------------------------------------------------
    @property
    def dftb(self) -> hsd.Node | None:
        ham = self.inp.find("Hamiltonian")
        if ham is not None and ham.method is not None and ham.method.name.lower() == "dftb":
            return ham.method
        return None

    @property
    def pin_dftb(self) -> hsd.Node | None:
        if self.pin is None:
            return None
        ham = self.pin.find("Hamiltonian")
        return ham.method if ham is not None and ham.method is not None else None

    @property
    def parser_version(self) -> int:
        v = self.inp.number("ParserOptions/ParserVersion")
        return int(v) if v is not None else CURRENT_PARSER

    @property
    def ignore_unprocessed(self) -> bool:
        return bool(self.inp.flag("ParserOptions/IgnoreUnprocessedNodes", False))

    @property
    def scc(self) -> bool:
        return bool(self.dftb is not None and self.dftb.flag("SCC", False))

    @property
    def shell_resolved(self) -> bool:
        """Shell-resolved SCC as DFTB+ uses it (pin if there is one, else the input with
        the conversions of old parser versions)."""
        if self.pin_dftb is not None:
            v = self.pin_dftb.flag("ShellResolvedSCC")
            if v is not None:
                return v
        d = self.dftb
        if d is None:
            return False
        for key in ("ShellResolvedSCC", "OrbitalResolvedSCC"):
            v = d.flag(key)
            if v is not None:
                return v
        return self.parser_version < 3 and self.scc

    def max_l(self) -> dict:
        node = self.dftb.find("MaxAngularMomentum") if self.dftb is not None else None
        out = {}
        if node is not None:
            for c in node.children:
                v = (c.text() or "").strip().lower()
                if v in L_OF:
                    out[c.name] = L_OF[v]
        return out

    def norb(self) -> int | None:
        """Orbitals of the structure (MaxAngularMomentum x atoms), None if unknown."""
        if self.geometry is None:
            return None
        ml = self.max_l()
        try:
            return sum((ml[s] + 1) ** 2 for s in self.geometry.atom_species)
        except KeyError:
            return None

    def omp_threads(self) -> int | None:
        """OpenMP threads of the run: printed by DFTB+, or (no screen output yet) from the
        environment; None if unknown or, without screen output, OMP_NUM_THREADS unset."""
        if self.stdout is not None:
            return self.stdout.omp_threads
        raw = self.environ.get("OMP_NUM_THREADS", "").split(",")[0].strip()
        return int(raw) if raw.isdigit() else None

    @property
    def threads_from_environment(self) -> bool:
        """No screen output: the thread count is a guess from this environment."""
        return self.stdout is None


def _find_stdout(d: Path) -> Path | None:
    cands = []
    for p in d.glob("*.out"):
        if p.name in ("detailed.out", "band.out", "md.out", "results.tag"):
            continue
        try:
            head = p.read_text(errors="replace")[:4000]
        except OSError:
            continue
        if BANNER.search(head):
            cands.append(p)
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def collect(path, stdout=None, environ=None, cores=None, mpi_procs=None,
            concurrent: int = 1, context: str | None = None) -> RunFiles:
    """Read a run directory (or a dftb_in.hsd) and what belongs to it."""
    path = Path(path)
    inp_path = path / "dftb_in.hsd" if path.is_dir() else path
    d = inp_path.parent
    run = RunFiles(d, hsd.read(inp_path), environ=dict(os.environ if environ is None else environ),
                   cores=cores, mpi_procs=mpi_procs, concurrent=max(1, int(concurrent)),
                   context=context)
    if run.cores is None and hasattr(os, "sched_getaffinity"):
        run.cores = len(os.sched_getaffinity(0))
    pin_path = d / "dftb_pin.hsd"
    if pin_path.is_file():
        try:
            run.pin = hsd.read(pin_path)
        except hsd.HSDError as err:
            run.notes.append(f"dftb_pin.hsd not read: {err}")
    out_path = Path(stdout) if stdout else _find_stdout(d)
    if out_path is not None and out_path.is_file():
        run.stdout = parse_stdout(out_path.read_text(errors="replace"), str(out_path))
    try:
        run.geometry = geometry_from(run.inp)
        if run.geometry is None and run.pin is not None:
            run.geometry = geometry_from(run.pin)
    except (ValueError, IndexError) as err:
        run.notes.append(f"geometry not read: {err}")
    if run.geometry is not None and run.dftb is not None:
        for (a, b), p in skf_paths(run.inp, run.geometry.species, d).items():
            run.skf[(a, b)] = read_skf_header(p, a == b)
    return run
