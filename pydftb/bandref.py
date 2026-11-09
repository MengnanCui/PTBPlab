"""Reference band structures and orbital energies for band-structure fitting (A series, A0).

A :class:`BandRef` holds what :mod:`pydftb.bandloss` compares DFTB with: the eigenvalues of one
structure, from a DFT code, at a set of k-points, with occupations, the Fermi level and the
number of semicore bands (per spin) that the DFTB basis does not have.

``kind`` is one of:

* ``"path"``: a crystal at its special points, at points next to them (level 1: a fraction
  0.05 and 0.10 of each adjoining segment) and along the segments between them (level 2);
* ``"ibz"``: a disordered cell (rattled, interface, MD frame) at the irreducible k-points of its
  SCF run, with their weights;
* ``"molecule"``: a molecule or cluster at the Gamma point (HOMO, LUMO and the levels around them).

Energies are in eV with the DFT code's own zero; the band loss fits a free shift per structure.
One JSON file per structure (:meth:`BandRef.to_json`, :meth:`BandRef.from_json`).

The module also builds the k-point sets (:func:`special_kpoints`), writes the k-points for a
VASP non-self-consistent run (:func:`vasp_kpoints_text`), reads VASP results
(:func:`from_vasprun`, :func:`from_eigenval`), counts semicore bands (:func:`count_semicore`,
:func:`zval_from_vasprun`, :func:`dftb_valence_electrons`) and reads ASE band structures
(:func:`from_ase_bandstructure`).  It only needs NumPy and ASE.
Design: ``docs/design/A0-accuracy-programme.html`` (section 3).
"""
from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

KINDS = ("path", "ibz", "molecule")
SCHEMA = 1
METAL_GAP = 0.05          # eV: a smaller gap between occupied and empty states means a metal


@dataclass
class BandRef:
    """Reference eigenvalues of one structure (see the module docstring)."""
    name: str
    kind: str
    kpts: np.ndarray                       # (n_k, 3) fractional (reciprocal-lattice basis)
    kweights: np.ndarray                   # (n_k,), sum 1
    energies: np.ndarray                   # (n_spin, n_k, n_band) eV
    occupations: np.ndarray | None = None  # (n_spin, n_k, n_band) fractions 0..1
    fermi: float | None = None             # eV, same zero as ``energies``
    n_semicore: int = 0                    # lowest bands per spin that DFTB does not have
    labels: list = field(default_factory=list)     # per k-point: special point name or ""
    level: np.ndarray | None = None        # per k-point: 1 near special points, 2 path, 0 other
    segments: list = field(default_factory=list)   # [[label_a, label_b, i0, i1]] (inclusive)
    numbers: np.ndarray | None = None      # the structure (positions in Angstrom)
    positions: np.ndarray | None = None
    cell: np.ndarray | None = None
    pbc: np.ndarray | None = None
    meta: dict = field(default_factory=dict)       # code, functional, settings, source file

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        self.kpts = np.atleast_2d(np.asarray(self.kpts, float))
        self.kweights = np.asarray(self.kweights, float)
        self.energies = np.asarray(self.energies, float)
        if self.energies.ndim == 2:                       # (n_k, n_band): one spin channel
            self.energies = self.energies[None]
        if self.occupations is not None:
            occ = np.asarray(self.occupations, float)
            if occ.ndim == 2:
                occ = occ[None]
            if occ.size and occ.max() > 1.0 + 1e-6:        # ISPIN 1 codes may give 0..2
                occ = occ / 2.0
            self.occupations = occ
        n_k = len(self.kpts)
        if self.energies.shape[1] != n_k or len(self.kweights) != n_k:
            raise ValueError(f"{self.name}: {n_k} k-points, energies {self.energies.shape}, "
                             f"weights {self.kweights.shape}")
        if self.occupations is not None and self.occupations.shape != self.energies.shape:
            raise ValueError(f"{self.name}: occupations {self.occupations.shape} differ from "
                             f"energies {self.energies.shape}")
        w = self.kweights.sum()
        if w <= 0:
            raise ValueError(f"{self.name}: the k-point weights sum to {w}")
        self.kweights = self.kweights / w
        self.level = (np.zeros(n_k, int) if self.level is None
                      else np.asarray(self.level, int).reshape(n_k))
        if not self.labels:
            self.labels = [""] * n_k
        if len(self.labels) != n_k:
            raise ValueError(f"{self.name}: {len(self.labels)} labels for {n_k} k-points")
        if self.n_semicore < 0 or self.n_semicore >= self.energies.shape[2]:
            raise ValueError(f"{self.name}: n_semicore {self.n_semicore} with "
                             f"{self.energies.shape[2]} bands")

    # ---- structure ---------------------------------------------------------------------
    @property
    def nspin(self) -> int:
        return self.energies.shape[0]

    @property
    def nbands(self) -> int:
        """Bands without the semicore ones."""
        return self.energies.shape[2] - self.n_semicore

    def atoms(self):
        """The structure as ASE Atoms (None if not stored)."""
        if self.numbers is None:
            return None
        from ase import Atoms
        return Atoms(numbers=self.numbers, positions=self.positions,
                     cell=self.cell if self.cell is not None else np.zeros((3, 3)),
                     pbc=self.pbc if self.pbc is not None else False)

    def set_atoms(self, atoms) -> "BandRef":
        self.numbers = np.array(atoms.numbers)
        self.positions = np.array(atoms.positions, float)
        self.cell = np.array(atoms.cell.array, float)
        self.pbc = np.array(atoms.pbc, bool)
        return self

    def valence(self) -> tuple:
        """(energies, occupations) without the semicore bands."""
        e = self.energies[:, :, self.n_semicore:]
        o = None if self.occupations is None else self.occupations[:, :, self.n_semicore:]
        return e, o

    def edges(self, n_valence_bands: int | None = None) -> dict:
        """VBM, CBM, gap and whether this is a metal (eV).

        With ``n_valence_bands`` (bands per spin below the gap, semicore removed) the edges are
        the highest of those and the lowest above them; otherwise they come from the
        occupations (occupied: fraction > 0.5).  A gap below ``METAL_GAP`` or partly occupied
        states make a metal, whose VBM = CBM = the Fermi level (if known)."""
        e, o = self.valence()
        if n_valence_bands is not None:
            nv = int(n_valence_bands)
            if not 0 < nv <= e.shape[2]:
                raise ValueError(f"{self.name}: {nv} valence bands, {e.shape[2]} available")
            vbm = float(e[:, :, :nv].max())
            cbm = float(e[:, :, nv:].min()) if nv < e.shape[2] else np.nan
        elif o is not None:
            occ = o > 0.5
            vbm = float(e[occ].max()) if occ.any() else np.nan
            cbm = float(e[~occ].min()) if (~occ).any() else np.nan
        else:
            raise ValueError(f"{self.name}: edges need occupations or n_valence_bands")
        partial = o is not None and bool(np.any((o > 0.02) & (o < 0.98)))
        gap = cbm - vbm if np.isfinite(cbm) and np.isfinite(vbm) else np.nan
        metal = partial or (np.isfinite(gap) and gap < METAL_GAP)
        if n_valence_bands is not None and self.fermi is not None:
            # bands that cross the Fermi level: a metal, whatever the index gap says
            nv = int(n_valence_bands)
            metal |= bool(e[:, :, :nv].max() > self.fermi + 2 * METAL_GAP)
            if nv < e.shape[2]:
                metal |= bool(e[:, :, nv:].min() < self.fermi - 2 * METAL_GAP)
        if metal and self.fermi is not None:
            vbm = cbm = float(self.fermi)
        return {"vbm": vbm, "cbm": cbm, "gap": 0.0 if metal else gap, "metal": bool(metal)}

    def select(self, levels) -> np.ndarray:
        """Indices of the k-points of the given levels (all for ``ibz`` and ``molecule``)."""
        if self.kind != "path":
            return np.arange(len(self.kpts))
        return np.nonzero(np.isin(self.level, list(levels)))[0]

    # ---- JSON ----------------------------------------------------------------------------
    def to_dict(self) -> dict:
        def arr(x):
            return None if x is None else np.asarray(x).tolist()
        return {"schema": SCHEMA, "name": self.name, "kind": self.kind, "kpts": arr(self.kpts),
                "kweights": arr(self.kweights), "energies": arr(self.energies),
                "occupations": arr(self.occupations), "fermi": self.fermi,
                "n_semicore": int(self.n_semicore), "labels": list(self.labels),
                "level": arr(self.level), "segments": [list(s) for s in self.segments],
                "numbers": arr(self.numbers), "positions": arr(self.positions),
                "cell": arr(self.cell), "pbc": arr(self.pbc), "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "BandRef":
        if d.get("schema", SCHEMA) > SCHEMA:
            raise ValueError(f"BandRef schema {d['schema']} is newer than this reader ({SCHEMA})")
        keys = ("name", "kind", "kpts", "kweights", "energies", "occupations", "fermi",
                "n_semicore", "labels", "level", "segments", "numbers", "positions", "cell",
                "pbc", "meta")
        kw = {k: d.get(k) for k in keys if d.get(k) is not None}
        for k in ("numbers", "positions", "cell", "pbc"):
            if k in kw:
                kw[k] = np.asarray(kw[k])
        return cls(**kw)

    def to_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh)
        return path

    @classmethod
    def from_json(cls, path: str) -> "BandRef":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


def load_refs(paths) -> list:
    """BandRef objects from JSON files or glob patterns, sorted by name within each pattern.
    JSON files that are not BandRefs (no ``energies``/``kind``, e.g. a generator's
    ``energies.json`` next to them) are skipped with a message; a repeated name is an error."""
    import glob
    out = []
    for p in ([paths] if isinstance(paths, str) else paths):
        files = sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
        for f in files:
            with open(f) as fh:
                d = json.load(fh)
            if not (isinstance(d, dict) and "energies" in d and "kind" in d):
                print(f"[bandref] skipped {f}: not a BandRef")
                continue
            out.append(BandRef.from_dict(d))
    names = [r.name for r in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"repeated BandRef names: {dup[:10]}")
    return out


# ---- k-point sets ------------------------------------------------------------------------
def special_kpoints(atoms, fractions=(0.05, 0.10), points_per_segment: int = 30, path=None):
    """k-points of a crystal's band reference: ``(kpts, labels, level, segments)``.

    Level 1: every special point of ASE's standard path for this cell (``path`` overrides it,
    e.g. ``"GXWKGLUWLK,UX"``) and, on every segment of the path, the points at each fraction of
    the segment from both of its ends.  Level 2: ``points_per_segment`` evenly spaced points on
    every segment, ends included; ``segments`` gives their index ranges.  Coordinates are
    fractional in the reciprocal basis of ``atoms.cell`` (straight lines there are straight
    lines in Cartesian space)."""
    from ase.dft.kpoints import parse_path_string
    bp = atoms.cell.bandpath(path=path, npoints=0)
    sp = {k: np.asarray(v, float) for k, v in bp.special_points.items()}
    runs = parse_path_string(bp.path)
    kpts, labels, level, segments = [], [], [], []

    def add(k, lab, lev):
        kpts.append(np.asarray(k, float))
        labels.append(lab)
        level.append(lev)

    seen = set()
    for run in runs:
        for lab in run:
            if lab not in seen:
                seen.add(lab)
                add(sp[lab], lab, 1)
    for run in runs:
        for a, b in zip(run[:-1], run[1:]):
            for f in fractions:
                add(sp[a] + f * (sp[b] - sp[a]), f"{a}+{f:g}{b}", 1)
                add(sp[b] + f * (sp[a] - sp[b]), f"{b}+{f:g}{a}", 1)
    n = max(2, int(points_per_segment))
    for run in runs:
        for a, b in zip(run[:-1], run[1:]):
            i0 = len(kpts)
            for t in np.linspace(0.0, 1.0, n):
                add(sp[a] + t * (sp[b] - sp[a]), a if t == 0 else (b if t == 1 else ""), 2)
            segments.append([a, b, i0, len(kpts) - 1])
    return np.array(kpts), labels, np.array(level, int), segments


def vasp_kpoints_text(kpts, comment: str = "explicit k-points (A series band reference)") -> str:
    """A VASP KPOINTS file listing these fractional k-points explicitly (weight 1 each), for a
    non-self-consistent run (ICHARG = 11) on the SCF's charge density."""
    k = np.atleast_2d(np.asarray(kpts, float))
    lines = [comment, str(len(k)), "Reciprocal"]
    lines += [f"{a:.10f} {b:.10f} {c:.10f} 1.0" for a, b, c in k]
    return "\n".join(lines) + "\n"


# ---- readers -----------------------------------------------------------------------------
def from_vasprun(path: str, name: str | None = None, kind: str = "ibz", n_semicore: int = 0,
                 labels=None, level=None, segments=None, meta: dict | None = None) -> BandRef:
    """A BandRef from a VASP ``vasprun.xml`` (the last ionic step): its k-points and weights,
    eigenvalues, occupations, Fermi level and structure.  For a non-SCF band run on explicit
    k-points pass the ``labels``, ``level`` and ``segments`` of :func:`special_kpoints`
    (``kind="path"``); its k-point weights are then equal."""
    import ase.io
    at = ase.io.read(path, index=-1, format="vasp-xml")
    calc = at.calc
    kpts = np.asarray(calc.get_ibz_k_points(), float)
    w = np.asarray(calc.get_k_point_weights(), float)
    ns = calc.get_number_of_spins()
    e = np.array([[calc.get_eigenvalues(kpt=i, spin=s) for i in range(len(kpts))]
                  for s in range(ns)])
    o = np.array([[calc.get_occupation_numbers(kpt=i, spin=s) for i in range(len(kpts))]
                  for s in range(ns)])
    if kind == "path":
        w = np.ones(len(kpts))
    info = {"code": "VASP", "source": os.path.abspath(path)}
    info.update(meta or {})
    ref = BandRef(name=name or os.path.basename(os.path.dirname(os.path.abspath(path))),
                  kind=kind, kpts=kpts, kweights=w, energies=e, occupations=o,
                  fermi=float(calc.get_fermi_level()), n_semicore=n_semicore,
                  labels=list(labels) if labels is not None else [],
                  level=level, segments=segments or [], meta=info)
    return ref.set_atoms(at)


def from_eigenval(path: str, atoms=None, name: str | None = None, kind: str = "ibz",
                  fermi: float | None = None, n_semicore: int = 0, **kw) -> BandRef:
    """A BandRef from a VASP ``EIGENVAL`` file (VASP 5 and 6; occupations if written).
    ``fermi`` (eV) is not in the file: take it from OUTCAR (``E-fermi``) or vasprun."""
    with open(path) as fh:
        lines = [ln for ln in fh.read().splitlines()]
    ispin = int(lines[0].split()[3])
    nel, nk, nb = (int(float(x)) for x in lines[5].split()[:3])
    pos = 6
    kpts, w, e, o = [], [], [], []
    for _ in range(nk):
        while not lines[pos].strip():
            pos += 1
        vals = lines[pos].split()
        kpts.append([float(v) for v in vals[:3]])
        w.append(float(vals[3]))
        pos += 1
        ek, ok = [], []
        for _ in range(nb):
            cols = lines[pos].split()
            pos += 1
            ek.append([float(c) for c in cols[1:1 + ispin]])
            ok.append([float(c) for c in cols[1 + ispin:1 + 2 * ispin]] if len(cols) >= 1 + 2 * ispin
                      else [np.nan] * ispin)
        e.append(ek)
        o.append(ok)
    e = np.transpose(np.array(e), (2, 0, 1))       # (spin, k, band)
    o = np.transpose(np.array(o), (2, 0, 1))
    occ = None if np.isnan(o).any() else o
    ref = BandRef(name=name or os.path.basename(os.path.dirname(os.path.abspath(path))),
                  kind=kind, kpts=np.array(kpts), kweights=np.array(w), energies=e,
                  occupations=occ, fermi=fermi, n_semicore=n_semicore,
                  meta={"code": "VASP", "source": os.path.abspath(path), "nelect": nel}, **kw)
    return ref.set_atoms(atoms) if atoms is not None else ref


def outcar_fermi(path: str) -> float:
    """The last ``E-fermi`` of a VASP OUTCAR (eV)."""
    val = None
    with open(path) as fh:
        for line in fh:
            if "E-fermi" in line:
                val = float(line.split()[2])
    if val is None:
        raise ValueError(f"no E-fermi in {path}")
    return val


def from_ase_bandstructure(bs, atoms=None, name: str = "bands", n_semicore: int = 0,
                           meta: dict | None = None) -> BandRef:
    """A path BandRef from an ASE ``BandStructure`` (object or JSON file): its path points are
    level 2, its special points level 1; energies keep their zero (``bs.reference`` is the
    Fermi level)."""
    if isinstance(bs, str):
        from ase.spectrum.band_structure import read_band_structure
        bs = read_band_structure(bs)
    kpts = np.asarray(bs.path.kpts, float)
    x, xs, labs = bs.path.get_linear_kpoint_axis()
    labels, level = [""] * len(kpts), np.full(len(kpts), 2)
    for xv, lab in zip(xs, labs):
        i = int(np.argmin(np.abs(x - xv)))
        labels[i], level[i] = lab, 1
    ref = BandRef(name=name, kind="path", kpts=kpts, kweights=np.ones(len(kpts)),
                  energies=np.asarray(bs.energies, float), fermi=float(bs.reference),
                  n_semicore=n_semicore, labels=labels, level=level,
                  meta={"code": "ASE BandStructure", **(meta or {})})
    return ref.set_atoms(atoms) if atoms is not None else ref


# ---- semicore bands ------------------------------------------------------------------------
def zval_from_vasprun(path: str) -> dict:
    """{element: valence electrons of its POTCAR} from vasprun.xml's atomtypes table."""
    root = ET.parse(path).getroot()
    out = {}
    for arr in root.iter("array"):
        if arr.get("name") != "atomtypes":
            continue
        fields = [f.text.strip() for f in arr.findall("field")]
        ie, iv = fields.index("element"), fields.index("valence")
        for rc in arr.find("set").findall("rc"):
            cols = [c.text.strip() for c in rc.findall("c")]
            out[cols[ie]] = float(cols[iv])
    if not out:
        raise ValueError(f"no atomtypes table in {path}")
    return out


def zval_from_potcar(path: str) -> dict:
    """{element: ZVAL} of a concatenated POTCAR (element from the TITEL line)."""
    out, element = {}, None
    with open(path) as fh:
        for line in fh:
            if "TITEL" in line:
                element = line.split("=")[1].split()[1].split("_")[0]
            m = re.search(r"ZVAL\s*=\s*([-\d.]+)", line)
            if m and element is not None:
                out[element] = float(m.group(1))
                element = None
    return out


def dftb_valence_electrons(skdir: str, symbols, max_l=None) -> dict:
    """{symbol: neutral-atom valence electrons} of the homonuclear SK files (their occupations,
    within the basis ``max_l`` uses)."""
    from .params import ParamSet
    sk = ParamSet(skdir, max_l=max_l).skset(list(dict.fromkeys(symbols)))
    return {s: float(np.sum(sk.occupation[s])) for s in dict.fromkeys(symbols)}


def count_semicore(atoms, zval: dict, dftb_valence: dict) -> int:
    """Bands per spin below the DFTB valence: (sum over atoms of ZVAL - DFTB valence) / 2, which
    must be a non-negative integer (e.g. Li_sv with DFTB's 2s: one 1s band per Li atom)."""
    extra = 0.0
    for s in atoms.get_chemical_symbols():
        if s not in zval or s not in dftb_valence:
            raise ValueError(f"no ZVAL or DFTB valence for {s}")
        extra += zval[s] - dftb_valence[s]
    if extra < -1e-6 or abs(extra / 2 - round(extra / 2)) > 1e-6:
        raise ValueError(f"semicore electrons {extra} are not a non-negative even number")
    return int(round(extra / 2))
