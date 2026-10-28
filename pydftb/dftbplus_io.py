"""DFTB+'s files (milestone M6f).

Charge restart files ``charges.bin`` / ``charges.dat`` (DFTB+ ``dftb/sccinit.F90``,
``writeQToFile`` and ``initQFromFile``; restart format 8, read back to format 4):

* the header: the format number; five logicals (blocks, imaginary blocks, density matrix,
  k-point information, multipoles); the number of atoms and of spin channels; one checksum per
  channel (the sum of its populations);
* the populations of every orbital, per channel and atom: channel 1 the charge q, channel 2
  the magnetisation m (collinear spin), channels 2-4 m_x, m_y, m_z (two-component).  Where
  orbitals are mixed together (an atom, or a shell) the class's total sits on its first
  orbital, as DFTB+'s ``OrbitalEquiv_expand`` leaves it;
* the numbers of atomic dipoles and quadrupoles (always written, ``0 0`` in DFTB);
* with DFTB+U or onsite corrections, every atom's block populations, row by row per channel
  and atom (and the imaginary blocks with dual spin-orbit coupling);
* with a hybrid functional (milestone M6h-1, molecules), the density-matrix difference
  DeltaP_in = P - P0 of every spin channel (up, down), the whole N x N matrix column by column,
  one value per record; DFTB+'s matrices hold the lower triangle (the upper one zero).  A
  periodic run writes its supercell folding (coefficients and shifts, 12 reals) before it: the
  identity at the Gamma point (M6h-3a), diag(N1, N2, N3) for k-points (M6h-3b), whose DeltaP_in
  is one record: MatrixBased the complex DeltaP(k) of every spin channel at DFTB+'s k-points of
  the folding (:func:`pydftb.kpoints.supercell_folding`), NeighbourBased the real array of the
  N1 N2 N3 cells of the Born-von Karman supercell, both of the atoms as DFTB+ folds them into
  the cell (:class:`pydftb.hybrid.KPointLayout` converts).

``charges.bin`` is a Fortran stream file (no record markers, 4-byte integers and logicals);
``charges.dat`` is the same in Fortran list-directed text, written here exactly as gfortran
writes it.  How a calculation's mixing vector maps onto these arrays is
:meth:`pydftb.DFTB.charge_data` and :meth:`pydftb.DFTB.populations_from_file`.

Analysis outputs (milestone M6f-4a; DFTB+ ``dftbplus/mainio.F90``, from
:class:`pydftb.analysis.AnalysisResult`): ``band.out`` (:func:`write_band_out`),
``eigenvec.out``/``eigenvec.bin`` (:func:`write_eigenvectors`), ``ESP.dat``
(:func:`write_esp`) and the analysis blocks of ``detailed.out`` (:func:`detailed_tables`),
each in DFTB+'s format; readers for the tests and the validation tools.  Milestone M6f-4b:
``localOrbs.out``/``.bin`` (``write_eigenvectors(..., localised=True)``), ``bond_pop.dat``,
``bond_energy.dat``, ``bond_order.dat`` (:func:`write_bond_files`) and the Casida transition
charges ``ATQ.DAT`` (:func:`write_transition_charges`).
"""
from __future__ import annotations

import math
import os
import re
import struct
from dataclasses import dataclass, field

import numpy as np

from .constants import BOHR, HARTREE
from .kpoints import supercell_folding

RESTART_FORMAT = 8
CHECKSUM_TOL = 1e-7          # DFTB+ elecTolMax: the checksums must match their arrays
ELECTRON_TOL = 1e-3          # DFTB+: electrons and magnetisation of the file vs the system


@dataclass
class ChargeData:
    """The contents of a DFTB+ charge restart file."""
    q: np.ndarray                         # (n_spin, N) populations of every orbital, atoms in turn
    norb_atom: np.ndarray                 # (n_atom,) orbitals of every atom
    blocks: list | None = None            # per atom (n_spin, n, n) block populations, or None
    iblocks: list | None = None           # imaginary blocks (dual spin-orbit coupling), or None
    checksum: np.ndarray | None = None    # (n_spin,) as read; None: computed when written
    version: int = RESTART_FORMAT
    multipolar: bool = True               # DFTB+ always writes the multipole record
    density_matrix: bool = False          # hybrid functionals' restart data (M6h-1)
    kpoint_info: bool = False
    n_dipole: int = 0
    n_quadrupole: int = 0
    extra: dict = field(default_factory=dict)
    rho: np.ndarray | None = None         # (n_spin, N, N) DeltaP_in of a hybrid run (M6h-1)
    folding: np.ndarray | None = None     # (3, 4) k-point folding and shifts of a periodic
                                          # hybrid run, before rho (M6h-3a: the Gamma point)
    # M6h-3b, a hybrid run at the k-points of the folding diag(N1, N2, N3), the record as read in
    # C order: MatrixBased (n_spin, nK, N, N) complex DeltaP_in(k) at DFTB+'s k-points (pydftb's
    # Bloch convention), NeighbourBased (n_spin, N1 N2 N3, N, N) the BvK array (cells n1 fastest,
    # [n, mu, nu] = DeltaP_in(mu in cell 0, nu in cell n)), both of the folded atoms
    rho_kpoints: np.ndarray | None = None
    rho_bvk: np.ndarray | None = None

    @property
    def n_spin(self) -> int:
        return self.q.shape[0]

    @property
    def n_atom(self) -> int:
        return len(self.norb_atom)

    def sums(self) -> np.ndarray:
        """The checksums as DFTB+ computes them: sum(sum(qq, dim=1), dim=1), i.e. every atom's
        orbitals in turn, then the atoms, in order."""
        out = np.zeros(self.n_spin)
        edges = np.concatenate([[0], np.cumsum(self.norb_atom)])
        for s in range(self.n_spin):
            total = 0.0
            for a in range(self.n_atom):
                part = 0.0
                for x in self.q[s, edges[a]:edges[a + 1]]:
                    part += float(x)
                total += part
            out[s] = total
        return out


# ---- reading ----------------------------------------------------------------------------
class _Stream:
    """Sequential reads of a Fortran stream file."""

    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def start(self):
        """A new read statement (nothing to skip in a stream file)."""

    def take(self, fmt: str):
        size = struct.calcsize(fmt)
        if self.pos + size > len(self.data):
            raise ValueError("the charge file ends too early (wrong orbital counts?)")
        out = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return out

    def ints(self, n):
        return [int(x) for x in self.take(f"<{n}i")]

    def logicals(self, n):
        return [x != 0 for x in self.take(f"<{n}i")]

    def reals(self, n):
        return np.array(self.take(f"<{n}d"), float)

    def complexes(self, n):
        return self.reals(2 * n).view(complex)

    def remaining(self) -> int:
        return len(self.data) - self.pos


_COMPLEX = re.compile(r"\(\s*([^,()\s]+)\s*[,;]\s*([^,()\s]+)\s*\)")


def _fortran_float(x: str) -> float:
    return float(x.replace("D", "E").replace("d", "e"))


class _Records:
    """Fortran list-directed reads: every read statement (:meth:`start`) begins on a new line
    and takes as many values (over as many lines) as it needs; the rest of its last line is
    skipped."""

    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.i = 0
        self.pending: list[str] = []

    def start(self):
        self.pending = []

    def items(self, n):
        while len(self.pending) < n:
            if self.i >= len(self.lines):
                raise ValueError("the charge file ends too early (wrong orbital counts?)")
            self.pending += self.lines[self.i].replace(",", " ").split()
            self.i += 1
        out, self.pending = self.pending[:n], self.pending[n:]
        return out

    def ints(self, n):
        return [int(x) for x in self.items(n)]

    def logicals(self, n):
        out = []
        for x in self.items(n):
            v = x.strip(".").upper()
            if v not in ("T", "F", "TRUE", "FALSE"):
                raise ValueError(f"not a logical in the charge file: {x!r}")
            out.append(v.startswith("T"))
        return out

    def reals(self, n):
        return np.array([_fortran_float(x) for x in self.items(n)])

    def complexes(self, n):
        """n complex values "(re,im)" (M6h-3b), over as many lines as they take."""
        out = []
        while len(out) < n:
            if self.i >= len(self.lines):
                raise ValueError("the charge file ends too early (wrong orbital counts?)")
            out += [complex(_fortran_float(a), _fortran_float(b))
                    for a, b in _COMPLEX.findall(self.lines[self.i])]
            self.i += 1
        return np.array(out[:n], complex)

    def next_is_complex(self) -> bool:
        return self.i < len(self.lines) and "(" in self.lines[self.i]


def read_charges(path, norb_atom, text: bool | None = None) -> ChargeData:
    """Read DFTB+'s ``charges.bin`` (or ``charges.dat`` with ``text``; default: by the file
    name's extension) for a system whose atoms have ``norb_atom`` orbitals (the binary file
    does not store them).  Formats 4 to 8; files with fewer atoms than the system are read as
    DFTB+ reads them (the rest zero), more atoms is an error."""
    norb_atom = np.asarray(norb_atom, int)
    if text is None:
        text = str(path).endswith(".dat")
    if text:
        with open(path) as f:
            rd = _Records(f.read())
    else:
        with open(path, "rb") as f:
            rd = _Stream(f.read())
    rd.start()
    version = rd.ints(1)[0]
    multipolar = kpoint_info = False
    rd.start()                                   # the header is one read statement
    if version == 4:
        block, iblock, rho = rd.logicals(3)
        n_atom_file = len(norb_atom)
        n_spin = rd.ints(1)[0]
    elif version == 5:
        block, iblock, rho = rd.logicals(3)
        n_atom_file, n_spin = rd.ints(2)
    elif version == 6:
        block, iblock, rho, multipolar = rd.logicals(4)
        n_atom_file, n_spin = rd.ints(2)
    elif version in (7, 8):
        block, iblock, rho, kpoint_info, multipolar = rd.logicals(5)
        n_atom_file, n_spin = rd.ints(2)
    else:
        raise ValueError(f"unknown charge file format {version} (DFTB+ 'Incompatible file type')")
    if n_spin not in (1, 2, 4):
        raise ValueError(f"the charge file has {n_spin} spin channels")
    if n_atom_file > len(norb_atom):
        raise ValueError(f"the charge file has {n_atom_file} atoms, the system "
                         f"{len(norb_atom)} (DFTB+: more atoms than the system)")
    checksum = rd.reals(n_spin)
    edges = np.concatenate([[0], np.cumsum(norb_atom)])
    q = np.zeros((n_spin, edges[-1]))
    for s in range(n_spin):
        for a in range(n_atom_file):
            rd.start()
            q[s, edges[a]:edges[a + 1]] = rd.reals(int(norb_atom[a]))
    n_dip = n_quad = 0
    if multipolar:
        rd.start()
        n_dip, n_quad = rd.ints(2)
        if n_dip or n_quad:
            raise ValueError("the charge file has atomic multipoles (not used by DFTB)")
    blocks = iblocks = None
    if block:
        blocks = _read_blocks(rd, norb_atom, n_atom_file, n_spin)
    if iblock:
        iblocks = _read_blocks(rd, norb_atom, n_atom_file, n_spin)
    dmat = folding = rho_k = rho_bvk = None
    if rho:                                      # hybrid functionals (M6h-1)
        if kpoint_info:                          # periodic (M6h-3a): coeffsAndShifts, in order
            rd.start()
            folding = np.array(rd.reals(12), float).reshape(4, 3).T
        if n_atom_file != len(norb_atom):
            raise ValueError("the charge file's density matrix needs all atoms of the system")
        n = int(edges[-1])
        nfold = folding_diagonal(folding)
        rd.start()
        if nfold is None or nfold == (1, 1, 1):  # deltaRhoIn(kk, jj, ii), one value per record
            dmat = rd.reals(n_spin * n * n).reshape(n_spin, n, n).transpose(0, 2, 1).copy()
        else:                                    # k-points (M6h-3b): one record
            ncell = int(np.prod(nfold))
            nk = len(supercell_folding(nfold)[0])
            if text:
                is_complex = rd.next_is_complex()
            else:
                left = rd.remaining()
                if left not in (16 * n * n * nk * n_spin, 8 * n * n * ncell * n_spin):
                    raise ValueError(f"the charge file's k-point density record has {left} "
                                     "bytes, neither DFTB+'s MatrixBased nor its NeighbourBased "
                                     f"one for {n} orbitals and the folding {nfold}")
                is_complex = left == 16 * n * n * nk * n_spin
            if is_complex:
                rho_k = rd.complexes(n_spin * nk * n * n).reshape(n_spin, nk, n, n)
            else:
                rho_bvk = rd.reals(n_spin * ncell * n * n).reshape(n_spin, ncell, n, n)
    return ChargeData(q, norb_atom, blocks, iblocks, checksum, version, multipolar, rho,
                      kpoint_info, n_dip, n_quad, rho=dmat, folding=folding, rho_kpoints=rho_k,
                      rho_bvk=rho_bvk)


def folding_diagonal(folding):
    """(N1, N2, N3) of a charge file's supercell folding record (coefficients and shifts,
    (3, 4)), None without one; DFTB+'s refusals of other foldings in its words (pydftb also
    refuses the non-diagonal ones its check passes, B12)."""
    if folding is None:
        return None
    coeffs, shifts = np.asarray(folding, float)[:, :3], np.asarray(folding, float)[:, 3]
    if abs(np.linalg.det(coeffs)) - 1.0 < -1e-6:
        raise ValueError("Determinant of the supercell matrix must be greater than 1. (the "
                         "charge file's folding, as DFTB+)")
    if np.any(np.abs(np.mod(coeffs + 0.5, 1.0) - 0.5) > 1e-6):
        raise ValueError("The components of the supercell matrix must be integers. (the charge "
                         "file's folding, as DFTB+)")
    if np.any(np.abs(coeffs - np.diag(np.diag(coeffs))) > 1e-6) or np.any(np.diag(coeffs) < 0.5):
        raise ValueError("Hybrid functionals using integration with k-points requires a "
                         "Monkhorst-Pack-like sampling, i.e., a uniform extension of the lattice. "
                         "(the charge file's folding, as DFTB+; pydftb takes diagonal foldings "
                         "only)")
    if np.any(np.abs(shifts) > 1e-6):
        raise ValueError("Hybrid functionals using integration with k-points requires a "
                         "Monkhorst-Pack-like sampling with a zero shift. (the charge file's "
                         "folding, as DFTB+)")
    return tuple(int(np.rint(x)) for x in np.diag(coeffs))


def _read_blocks(rd, norb_atom, n_atom_file, n_spin) -> list:
    """Per atom (n_spin, n, n): line ii of an atom is column ii, qBlock(1:n, ii, atom, spin)."""
    out = [np.zeros((n_spin, int(n), int(n))) for n in norb_atom]
    for s in range(n_spin):
        for a in range(n_atom_file):
            n = int(norb_atom[a])
            for ii in range(n):
                rd.start()
                out[a][s, :, ii] = rd.reals(n)
    return out


# ---- writing ----------------------------------------------------------------------------
def write_charges(path, data: ChargeData, text: bool | None = None) -> None:
    """Write ``data`` in restart format 8 (binary, or with ``text`` the list-directed
    ``charges.dat`` byte for byte as DFTB+ built with gfortran writes it); the checksums are
    computed from the populations."""
    if text is None:
        text = str(path).endswith(".dat")
    if data.n_dipole or data.n_quadrupole or (data.kpoint_info and data.folding is None):
        raise ValueError("pydftb writes only populations, blocks and a density matrix (of a "
                         "molecule, the Gamma point or a supercell folding)")
    dens = [x for x in (data.rho, data.rho_kpoints, data.rho_bvk) if x is not None]
    if len(dens) > 1:
        raise ValueError("one density matrix record only (rho, rho_kpoints or rho_bvk)")
    if data.density_matrix and not dens:
        raise ValueError("density_matrix is set but there is no density matrix (rho)")
    if data.folding is not None and not dens:
        raise ValueError("the k-point folding record goes with a density matrix (DFTB+ writes "
                         "it before one)")
    kpoint_record = data.rho_kpoints is not None or data.rho_bvk is not None
    nfold = folding_diagonal(data.folding)
    if kpoint_record and (nfold is None or nfold == (1, 1, 1)):
        raise ValueError("a k-point density record needs its supercell folding (DFTB+ writes "
                         "it before)")
    if data.rho is not None and nfold not in (None, (1, 1, 1)):
        raise ValueError("a real density matrix goes with a molecule or the Gamma point's "
                         "folding (k-points: rho_kpoints or rho_bvk)")
    norb = np.asarray(data.norb_atom, int)
    edges = np.concatenate([[0], np.cumsum(norb)])
    head_flags = [data.blocks is not None, data.iblocks is not None, bool(dens),
                  data.folding is not None, True]
    checksum = data.sums()
    rows = []                                   # (kind, values) records in order
    rows.append(("i", [RESTART_FORMAT]))
    rows.append(("h", (head_flags, [data.n_atom, data.n_spin], checksum)))
    for s in range(data.n_spin):
        for a in range(data.n_atom):
            rows.append(("r", data.q[s, edges[a]:edges[a + 1]]))
    rows.append(("m", [0, 0]))
    for blocks in (data.blocks, data.iblocks):
        if blocks is None:
            continue
        for s in range(data.n_spin):
            for a in range(data.n_atom):
                for ii in range(int(norb[a])):
                    rows.append(("r", np.asarray(blocks[a][s, :, ii], float)))
    if data.folding is not None:                # periodic hybrids (M6h-3a): coeffsAndShifts
        rows.append(("r", np.asarray(data.folding, float).ravel(order="F")))
    if data.rho is not None:                    # column by column, one value per record
        rho = np.asarray(data.rho, float)
        if rho.shape != (data.n_spin, edges[-1], edges[-1]):
            raise ValueError(f"the density matrix has shape {rho.shape}, not "
                             f"{(data.n_spin, int(edges[-1]), int(edges[-1]))}")
        rows.extend(("r", [x]) for x in rho.transpose(0, 2, 1).ravel())
    n = int(edges[-1])
    if data.rho_kpoints is not None:            # M6h-3b: one record, as held (C order)
        r = np.asarray(data.rho_kpoints, complex)
        nk = len(supercell_folding(nfold)[0])
        if r.shape != (data.n_spin, nk, n, n):
            raise ValueError(f"the k-point density record has shape {r.shape}, not "
                             f"{(data.n_spin, nk, n, n)}")
        rows.append(("c", r.ravel()))
    if data.rho_bvk is not None:
        r = np.asarray(data.rho_bvk, float)
        ncell = int(np.prod(nfold))
        if r.shape != (data.n_spin, ncell, n, n):
            raise ValueError(f"the BvK density record has shape {r.shape}, not "
                             f"{(data.n_spin, ncell, n, n)}")
        rows.append(("r", r.ravel()))
    tmp = f"{path}.tmp"
    if text:
        with open(tmp, "w") as f:
            for kind, vals in rows:
                f.write(_text_record(kind, vals) + "\n")
    else:
        with open(tmp, "wb") as f:
            for kind, vals in rows:
                f.write(_binary_record(kind, vals))
    os.replace(tmp, path)


def _binary_record(kind, vals) -> bytes:
    if kind in ("i", "m"):
        return struct.pack(f"<{len(vals)}i", *vals)
    if kind == "c":
        return np.asarray(vals, "<c16").tobytes()
    if kind == "h":
        flags, ints, sums = vals
        return (struct.pack("<5i", *[1 if f else 0 for f in flags])
                + struct.pack("<2i", *ints) + struct.pack(f"<{len(sums)}d", *sums))
    return struct.pack(f"<{len(vals)}d", *[float(x) for x in vals])


def _text_record(kind, vals) -> str:
    if kind == "i":
        return "".join(ld_integer(v) for v in vals)
    if kind == "m":                              # write(unit, "(2I2)") nDipole, nQuadrupole
        return "".join(f"{v:2d}" for v in vals)
    if kind == "h":
        flags, ints, sums = vals
        return ("".join(ld_logical(f) for f in flags) + "".join(ld_integer(v) for v in ints)
                + "".join(ld_real(x) for x in sums))
    if kind == "c":
        return "".join(ld_complex(z) for z in vals)
    return "".join(ld_real(x) for x in vals)


# gfortran list-directed output: every item has a leading blank; INTEGER(4) in 11 columns,
# LOGICAL as one letter, REAL(8) as 1PG25.17E3 (17 significant digits): fixed point
# F20.(16-e) followed by 5 blanks for 0.1 <= |x| < 1e17 (and zero), otherwise exponential
# with one digit before the point and a three-digit exponent, right-justified in 25 columns.
def ld_integer(n: int) -> str:
    return " " + f"{int(n):d}".rjust(11)


def ld_logical(v: bool) -> str:
    return " T" if v else " F"


def ld_real(x: float) -> str:
    x = float(x)
    if x == 0.0:
        return " " + f"{x:.16f}".rjust(20) + " " * 5
    if not math.isfinite(x):
        return " " + ("NaN" if math.isnan(x) else ("-Infinity" if x < 0 else "Infinity")).rjust(25)
    mant, exp = f"{x:.16E}".split("E")
    e = int(exp)
    if -1 <= e <= 16:
        return " " + f"{x:.{16 - e}f}".rjust(20) + " " * 5
    sign = "-" if e < 0 else "+"
    return " " + f"{mant}E{sign}{abs(e):03d}".rjust(25)


def ld_complex(z: complex) -> str:
    """COMPLEX(8) as gfortran's list-directed output writes it (M6h-3b): "(re,im)" right-justified
    in 53 columns after the blank, each part as :func:`ld_real` without its blanks except that the
    exponential form has 17 digits after the point (18 significant)."""
    return " " + f"({_complex_part(z.real)},{_complex_part(z.imag)})".rjust(53)


def _complex_part(x: float) -> str:
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return ld_real(x).strip()
    e = int(f"{x:.16E}".split("E")[1])
    if -1 <= e <= 16:
        return ld_real(x).strip()
    mant, exp = f"{x:.17E}".split("E")
    e = int(exp)
    return f"{mant}E{'-' if e < 0 else '+'}{abs(e):03d}"


# ---- analysis outputs (milestone M6f-4a) ------------------------------------------------------
PAULI_COLUMNS = ("   Atom   Orb         up spin coefficients   down spin coefficients  "
                 "  charge    x         y         z")


def fortran_e(x: float, width: int, digits: int) -> str:
    """Fortran Ew.d as gfortran writes it: 0.ddd...E+xx (E dropped for |exponent| > 99)."""
    x = float(x)
    if x == 0.0:
        body = "0." + "0" * digits + "E+00"
    else:
        mant, exp = f"{abs(x):.{digits - 1}E}".split("E")
        e = int(exp) + 1
        body = "0." + mant.replace(".", "") + (f"E{e:+03d}" if abs(e) <= 99 else f"{e:+04d}")
        if x < 0:
            body = "-" + body
    return body.rjust(width)


def _write_text(path, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def write_band_out(path, res) -> None:
    """DFTB+'s band.out (``WriteBandOut``): for every spin channel and k-point the
    list-directed header "KPT k SPIN s KWEIGHT w", lines (I6, F12.4, F9.5) of every level (eV)
    and its occupation (with the factor 2 without spin, not times the k weight), then a blank
    line.  ``res`` is a :class:`pydftb.DFTBResult`."""
    if res.eigenvalues is None:                  # M7a: the purification has no eigenvalues
        raise ValueError("band.out needs the eigenvalues: this solver gives none (use "
                         "pydftb.bands.eigenstates at the SCC charges)")
    kw = np.ones(1) if res.kweights is None else np.asarray(res.kweights, float)
    nk = len(kw)
    lines = []
    for b, (e, f) in enumerate(zip(res.eigenvalues, res.occupations)):
        s, k = divmod(b, nk)
        lines.append(" KPT " + ld_integer(k + 1) + "  SPIN " + ld_integer(s + 1) + "  KWEIGHT "
                     + ld_real(kw[k]))
        lines.extend(f"{i + 1:6d}{x * HARTREE:12.4f}{y:9.5f}" for i, (x, y)
                     in enumerate(zip(np.asarray(e, float), np.asarray(f, float).ravel())))
        lines.append("")
    _write_text(path, "\n".join(lines) + "\n")


def read_band_out(path) -> list:
    """The blocks of a band.out: [{"k", "spin", "weight", "energies" (eV), "occupations"}]."""
    blocks = []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0] == "KPT":
                blocks.append({"k": int(t[1]), "spin": int(t[3]), "weight": float(t[5]),
                               "energies": [], "occupations": []})
            else:
                blocks[-1]["energies"].append(float(t[1]))
                blocks[-1]["occupations"].append(float(t[2]))
    for b in blocks:
        b["energies"], b["occupations"] = np.array(b["energies"]), np.array(b["occupations"])
    return blocks


def _analysis_of(res, what: str):
    a = getattr(res, "analysis", None)
    if a is None or getattr(a, what) is None:
        raise ValueError(f"no {what} in this result: calculate with DFTB(analysis=...) asking "
                         f"for them")
    return a


def write_eigenvectors(path, res, text: bool | None = None, run_id: int = 0,
                      localised: bool = False) -> None:
    """DFTB+'s eigenvec.out (``EigenvectorsAsText``) or eigenvec.bin from a result calculated
    with ``analysis={"eigenvectors": True}``.  The text file lists every eigenvector (per spin
    channel and k-point) with the coefficient and Mulliken weight of every orbital, in DFTB+'s
    real, complex (k-points) or two-component format; the binary file is a stream of the int32
    ``run_id`` (DFTB+ draws a random one for waveplot) and the eigenvectors column by column,
    k-points fastest.  ``text`` defaults to whether ``path`` ends with ".out".  ``localised``:
    the Pipek-Mezey orbitals of ``analysis={"localise": ...}`` instead, as DFTB+'s
    localOrbs.out/.bin (M6f-4b)."""
    if localised:
        a = _analysis_of(res, "localisation")
        vectors, weights = a.localisation.orbitals, a.localisation.weights
    else:
        a = _analysis_of(res, "eigenvectors")
        vectors, weights = a.eigenvectors, a.mulliken_weights
    if text is None:
        text = str(path).endswith(".out")
    if not text:
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            f.write(struct.pack("<i", int(run_id)))
            for c in vectors:
                c = np.asarray(c)
                dtype = "<f8" if a.eigenvector_kind == "real" else "<c16"
                for i in range(c.shape[1]):
                    f.write(np.ascontiguousarray(c[:, i], dtype=dtype).tobytes())
        os.replace(tmp, path)
        return
    kind, nk = a.eigenvector_kind, a.n_kpoints
    lines = ["Coefficients and Mulliken populations of the atomic orbitals", ""]
    if kind == "pauli":
        lines += [PAULI_COLUMNS, ""]
    heads = []
    for atom, lab, ish, name in a.orbitals:
        heads.append((atom, (f"{atom + 1:5d} {lab[:2]:<2s}  " if ish == 0 else " " * 10) + name))
    norb = len(heads)
    for b, (c, w) in enumerate(zip(vectors, weights)):
        c, w = np.asarray(c), np.asarray(w)
        s, k = divmod(b, nk)
        spin = ("up", "down")[s] if kind != "pauli" else ""
        for i in range(c.shape[1]):
            if kind == "real":
                lines += [f"Eigenvector:{i + 1:4d}    ({spin})", ""]
            elif kind == "complex":
                lines += [f"K-point: {k + 1:4d}    Eigenvector: {i + 1:4d}    ({spin})", ""]
            else:
                lines.append(f"K-point: {k + 1:4d}    Eigenvector: {i + 1:4d}")
            for mu, (atom, head) in enumerate(heads):
                lead = head[:21].ljust(21)
                if kind == "real":
                    lines.append(f"{lead} {c[mu, i]:10.6f} {w[mu, i]:10.6f}")
                elif kind == "complex":
                    lines.append(f"{lead} ({c[mu, i].real:10.6f},{c[mu, i].imag:10.6f} ) "
                                 f"{w[mu, i]:10.6f}")
                else:
                    up, dn = c[mu, i], c[mu + norb, i]
                    lines.append(f"{lead} ({up.real:10.6f},{up.imag:10.6f})"
                                 f"({dn.real:10.6f},{dn.imag:10.6f}) "
                                 + "".join(f"{x:10.6f}" for x in w[:, mu, i]))
                if mu + 1 == norb or heads[mu + 1][0] != atom:
                    lines.append("")
    _write_text(path, "\n".join(lines) + "\n")


def read_eigenvec_out(path) -> tuple[str, list]:
    """(kind, [(header line, weights)]) of an eigenvec.out: the Mulliken weight of every
    orbital ((n_orb,), two-component (n_orb, 4): charge, x, y, z)."""
    out, cur, pauli = [], None, False
    with open(path) as fh:
        for line in fh:
            if line.startswith("   Atom   Orb"):
                pauli = True
            elif line.startswith(("Eigenvector:", "K-point:")):
                cur = [line.strip(), []]
                out.append(cur)
            elif cur is not None and line.strip():
                t = line.replace("(", " ").replace(")", " ").replace(",", " ").split()
                cur[1].append([float(x) for x in t[-4:]] if pauli else float(t[-1]))
    kind = "pauli" if pauli else ("complex" if out and out[0][0].startswith("K-point") else "real")
    return kind, [(h, np.array(w)) for h, w in out]


def write_esp(path, esp, geo_step: int | None = None, append: bool = False) -> None:
    """DFTB+'s ESP.dat from a :class:`pydftb.analysis.ESPResult`: the header
    ('# ', L2, 3I6, 1x, I0) (external column?, grid points or 0 0 0, number of points), for a
    grid the origin and the three step vectors (A, '#' then 3E20.12), the column titles, then
    the potentials in V (E20.12; points with their coordinates in A, 3E12.4).  The grid header
    holds the correct origin and steps also for rotated grids (DFTB+ writes them wrongly when
    ``Directions`` is not the identity).  ``append`` adds to an existing file without a new
    header, ``geo_step`` labels the block (DFTB+ ``AppendFile`` during MD)."""
    ext = esp.external is not None
    grid = esp.shape is not None
    pts = np.asarray(esp.points, float)
    lines = []
    if not (append and os.path.exists(path) and os.path.getsize(path) > 0):
        shape = esp.shape if grid else (0, 0, 0)
        lines.append("# " + (" T" if ext else " F") + "".join(f"{int(n):6d}" for n in shape)
                     + f" {len(pts)}")
        if grid:
            lines.append("#" + "".join(fortran_e(x * BOHR, 20, 12) for x in esp.origin))
            for row in np.asarray(esp.axes, float):
                lines.append("#" + "".join(fortran_e(x * BOHR, 20, 12) for x in row))
    tag = "" if geo_step is None else f" Geo {int(geo_step)}"
    if grid:
        lines.append(("# Internal (V)        External (V)" if ext else "# Internal (V)") + tag)
    else:
        lines.append(("#           Location (AA)             Internal (V)        External (V)"
                      if ext else "#           Location (AA)             Internal (V)") + tag)
    for i, p in enumerate(pts):
        row = "" if grid else "".join(fortran_e(x * BOHR, 12, 4) for x in p)
        row += fortran_e(esp.internal[i] * HARTREE, 20, 12)
        if ext:
            row += fortran_e(esp.external[i] * HARTREE, 20, 12)
        lines.append(row)
    text = "\n".join(lines) + "\n"
    if append and os.path.exists(path):
        with open(path, "a") as f:
            f.write(text)
    else:
        _write_text(path, text)


def read_esp(path) -> dict:
    """An ESP.dat: {"external" (bool), "shape", "n", "origin", "axes" (A, grids), "points"
    (A, point lists), "internal", "external_values" (V)}."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    head = lines[0][2:].split()
    out = {"external": head[0] == "T", "shape": tuple(int(x) for x in head[1:4]),
           "n": int(head[4]), "origin": None, "axes": None, "points": None}
    grid = all(x > 0 for x in out["shape"])
    comments = [ln for ln in lines if ln.startswith("#")]
    if grid:
        out["origin"] = np.array(comments[1][1:].split(), float)
        out["axes"] = np.array([c[1:].split() for c in comments[2:5]], float)
    data = np.array([ln.split() for ln in lines if ln.strip() and not ln.startswith("#")], float)
    data = data.reshape(len(data), -1)
    if not grid:
        out["points"], data = data[:, :3], data[:, 3:]
    out["internal"] = data[:, 0]
    out["external_values"] = data[:, 1] if out["external"] else None
    return out


def detailed_tables(res) -> str:
    """The analysis blocks of DFTB+'s detailed.out for ``res.analysis``: net populations and
    hybridisation ratios (I5, 1X, F16.8, F16.8), CM5 charges (I5, 1X, F16.8) and the
    atom-resolved electronic, repulsive, total and (with dispersion) dispersion energies
    (I5, F16.8, ' H', F16.6, ' eV'), each followed by a blank line."""
    a = res.analysis
    if a is None:
        raise ValueError("no analysis in this result: calculate with DFTB(analysis=...)")
    lines = []
    if a.net_populations is not None:
        lines += [" Atomic net (on-site) populations and hybridisation ratios",
                  " Atom       Population         Hybrid."]
        lines += [f"{i + 1:5d} {q:16.8f}{h:16.8f}"
                  for i, (q, h) in enumerate(zip(a.net_populations, a.hybridisation))]
        lines.append("")
    if a.cm5_charges is not None:
        lines += [" CM5 corrected atomic gross charges (e)", " Atom           Charge"]
        lines += [f"{i + 1:5d} {q:16.8f}" for i, q in enumerate(a.cm5_charges)]
        lines.append("")
    if a.atom_energies is not None:
        titles = [("electronic", "Atom resolved electronic energies"),
                  ("repulsive", "Atom resolved repulsive energies"),
                  ("total", "Atom resolved total energies")]
        if np.any(a.atom_energies["dispersion"] != 0.0):
            titles.append(("dispersion", "Atom resolved dispersion energies "))
        for key, title in titles:
            lines.append(title)
            lines += [f"{i + 1:5d}{e:16.8f} H{e * HARTREE:16.6f} eV"
                      for i, e in enumerate(a.atom_energies[key])]
            lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


# ---- M6f-4b: bond matrices and transition charges ---------------------------------------------
BOND_FILES = {
    "bond_populations": ("bond_pop.dat", "# Mulliken bond populations: B_AB = sum_{mu in A, "
                                         "nu in B} P_{mu nu} S_{mu nu}"),
    "bond_energies": ("bond_energy.dat", "# Non-SCC bond energies: E_AB = sum_{mu in A, nu in B}"
                                         " P_{mu nu} H0_{mu nu} (a.u.)"),
    "bond_orders": ("bond_order.dat", "# Mayer bond orders: B_AB = sum_{mu in A, nu in B} "
                                      "(PS)_{mu nu} (PS)_{nu mu}")}


def write_bond_matrix(path, matrix, kind: str) -> None:
    """One of DFTB+'s bond_pop.dat, bond_energy.dat, bond_order.dat (``kind`` the analysis
    field "bond_populations", "bond_energies" or "bond_orders"): two comment lines, then a row
    per atom, ``(*(ES16.8,1x))`` (gfortran writes no trailing blank)."""
    m = np.atleast_2d(np.asarray(matrix, float))
    lines = [BOND_FILES[kind][1], f"# nAtom = {len(m)}"]
    lines += [" ".join(f"{x:16.8E}" for x in row) for row in m]
    _write_text(path, "\n".join(lines) + "\n")


def write_bond_files(res, directory=".") -> list:
    """The bond files of a result calculated with ``analysis={"bond_populations": True, ...}``
    (those it has), in ``directory`` under DFTB+'s names; returns the names."""
    a = getattr(res, "analysis", None)
    names = []
    for kind, (name, _) in BOND_FILES.items():
        m = None if a is None else getattr(a, kind)
        if m is not None:
            write_bond_matrix(os.path.join(directory, name), m, kind)
            names.append(name)
    if not names:
        raise ValueError("no bond analysis in this result: calculate with "
                         "DFTB(analysis={'bond_populations': True, ...})")
    return names


def read_bond_matrix(path) -> np.ndarray:
    """The matrix of a bond_*.dat file."""
    with open(path) as fh:
        return np.array([ln.split() for ln in fh if ln.strip() and not ln.startswith("#")],
                        float)


def write_transition_charges(path, res) -> None:
    """DFTB+'s ATQ.DAT from a result with ``Casida(..., transition_charges=True)``: three
    comment lines, then ``(2x,i5,5x,f12.9)`` per atom."""
    exc = getattr(res, "excitations", None)
    q = None if exc is None else getattr(exc, "transition_charges", None)
    if q is None:
        raise ValueError("no transition charges in this result: use "
                         "Casida(..., transition_charges=True) with a state of interest")
    lines = ["#", "#  atom     transition charge", "#"]
    lines += [f"  {i:5d}     {x:12.9f}" for i, x in enumerate(np.asarray(q, float), 1)]
    _write_text(path, "\n".join(lines) + "\n")


def read_transition_charges(path) -> np.ndarray:
    """The charges of an ATQ.DAT file."""
    with open(path) as fh:
        return np.array([float(ln.split()[1]) for ln in fh
                         if ln.strip() and not ln.startswith("#")])


# ---- M6g-1: polarisability and response kernel ------------------------------------------------
_CART = "xyz"


def _omega_line(label: str, omega: float) -> str:
    """DFTB+ 25.1's (A, ':', T32, F18.10, T51, A, T54, F16.4, T71, A) frequency line."""
    head = (label + ":").ljust(31)
    return f"{head}{omega:18.10f}  H {omega * HARTREE:16.4f}  eV"


def response_tables(res, part: str | None = None, dftbplus_fermi: bool = False) -> str:
    """DFTB+'s detailed.out blocks of the response analyses (``res.analysis.polarisability``,
    ``res.analysis.response_kernel``) in DFTB+'s order: per atom and frequency of the kernel the frontier orbital derivatives
    (1X,I2,I4,2F14.6) and the Mulliken and on-site population derivatives (I5, 2F12.6)
    (written during the perturbation); then the density of states at the Fermi energy
    (metals), the atomic charge derivatives d q / d E_i of the last frequency ((I4,1X,E20.12),
    with collinear spin "u"/"d" rows of 1/2 (q +- m), two-component "q"/"x"/"y"/"z" rows of the
    four Pauli components, M6g-3b), the Fermi-level derivatives (metals; the physical values,
    DFTB+ prints them divided by the full filling, 2 without spin: ``dftbplus_fermi``) and the
    polarisability of every frequency (one row per field direction, 3E20.12).  ``part``:
    "kernel" or "main" (the rest) only; the kernel's blocks hold the charge part."""
    a = res.analysis
    pol = None if a is None else a.polarisability
    ker = None if a is None else a.response_kernel
    if pol is None and ker is None:
        raise ValueError("no response analysis in this result: calculate with "
                         "DFTB(analysis={'polarisability': True}) or 'response_kernel'")
    lines = []
    if ker is not None and part != "main":
        natom = ker.atom.shape[1]
        for at in range(natom):
            for w, omega in enumerate(ker.frequencies):
                lines.append("Static response:" if omega == 0.0
                             else _omega_line("Response at omega = ", omega))
                lines.append(f"Derivatives wrt. a potential at atom {at + 1}")
                lines.append(" Frontier orbital energy derivatives (a.u.)")
                lines.append(" Spin Kpt    Last filled   First empty")
                for (s, k, lo, hi), de in zip(ker.frontier, ker.eigenvalue_derivatives[w][at]):
                    lines.append(f" {s + 1:2d}{k + 1:4d}{de[lo]:14.6f}{de[hi]:14.6f}")
                lines.append(" Atomic population derivatives (a.u.)")
                lines.append(" Atom    Mulliken    On-site")
                for b in range(natom):
                    lines.append(f"{b + 1:5d}{ker.atom[w, at, 0, b]:12.6f}{ker.net[w, at, b]:12.6f}")
                lines.append("")
    if part == "kernel":
        return "\n".join(lines) + "\n"
    dos = (pol or ker).dos_fermi
    if dos is not None:
        vals = np.atleast_1d(dos)
        if vals.size == 2:
            lines.append("Density of states at the Fermi energy (a.u.): "
                         f"{fortran_e(vals[0], 13, 6)} (up) {fortran_e(vals[1], 13, 6)} (down)")
        else:
            lines.append("Density of states at the Fermi energy (a.u.): "
                         + fortran_e(vals[0], 13, 6))
    if pol is not None:
        dq = pol.charge_derivatives[-1]                      # (3, n_qm, n_atom)
        for i in range(3):
            lines.append(f"Atomic charge derivatives (a.u.), d q / d E_{_CART[i]}:")
            for at in range(dq.shape[2]):
                if dq.shape[1] == 1:
                    lines.append(f"{at + 1:4d} " + fortran_e(dq[i, 0, at], 20, 12))
                elif dq.shape[1] == 4:                   # two-component (M6g-3b)
                    lines.append(f"{at + 1:4d} q" + fortran_e(dq[i, 0, at], 20, 12))
                    lines += [f"     {c}" + fortran_e(dq[i, b, at], 20, 12)
                              for b, c in enumerate("xyz", 1)]
                else:
                    q, m = dq[i, 0, at], dq[i, 1, at]
                    lines.append(f"{at + 1:4d} u" + fortran_e(0.5 * (q + m), 20, 12))
                    lines.append("     d" + fortran_e(0.5 * (q - m), 20, 12))
            lines.append("")
        if pol.fermi_derivatives is not None:
            lines.append("Derivative of Fermi energy with respect to electric field")
            for i in range(3):
                vals = pol.fermi_derivatives[-1, i]
                if dftbplus_fermi:                       # the full filling: 2 without spin
                    vals = vals / (2.0 if dq.shape[1] == 1 else 1.0)
                lines.append(f" d E_f / d E_{_CART[i]}:" + "".join(fortran_e(v, 20, 12)
                                                              for v in vals))
        if pol.tensor is not None:
            lines += ["", " Electric field polarisability (a.u.)"]
            for w, omega in enumerate(pol.frequencies):
                lines.append("")
                lines.append(" Static polarisability:" if omega == 0.0
                             else _omega_line("Polarisability at omega = ", omega))
                for i in range(3):                           # one row per field direction
                    lines.append("".join(fortran_e(v, 20, 12) for v in pol.tensor[w, :, i]))
            lines.append("")
    return "\n".join(lines) + "\n"


def write_band_derivatives(path, res, kind: str = "field") -> None:
    """DFTB+'s dE_band.out (``kind="field"``: for every direction, spin channel and k-point a
    list-directed header "DIR x KPT k SPIN s KWEIGHT w", then (I6, E16.6) rows of the level
    derivatives with the levels in eV, per unit field in atomic units), dV_band.out
    (``kind="potential"``: per atom "ATOM n KPT ..." blocks, per unit potential in Hartree;
    two-component: SPIN 1 and zero blocks for SPIN 2-4, as DFTB+), of the last frequency of the
    field and of every frequency of the kernel (DFTB+ keeps only the last frequency of atom 1),
    or dBand_dx.out (``kind="coordinate"``, M6g-2: "Atom n dx
    KPT ..." blocks of d eps / dx in eV/Bohr, KWEIGHT 1)."""
    a = res.analysis
    kw = np.ones(1) if res.kweights is None else np.asarray(res.kweights, float)
    nk = len(kw)
    lines = []

    def block(head, de):
        for b, d in enumerate(de):
            s, k = divmod(b, nk)
            lines.append(head + " KPT " + ld_integer(k + 1) + "  SPIN " + ld_integer(s + 1)
                         + "  KWEIGHT " + ld_real(kw[k]))
            lines.extend(f"{i + 1:6d}" + fortran_e(x * HARTREE, 16, 6) for i, x in enumerate(d))
            lines.append("")

    if kind == "field":
        pol = None if a is None else a.polarisability
        if pol is None:
            raise ValueError("no polarisability in this result")
        for i in range(3):
            block(f" DIR {_CART[i]}", pol.eigenvalue_derivatives[-1][i])
    elif kind == "potential":
        ker = None if a is None else a.response_kernel
        if ker is None:
            raise ValueError("no response kernel in this result")
        pauli = ker.atom.shape[2] == 4
        for w in range(len(ker.frequencies)):
            for at, de in enumerate(ker.eigenvalue_derivatives[w]):
                if pauli:                                # DFTB+'s spin slots 2-4 (M6g-3b)
                    de = np.concatenate([de, np.zeros((3 * len(de),) + de.shape[1:])])
                block(f"ATOM {at + 1} ", de)
    elif kind == "coordinate":                           # dBand_dx.out (M6g-2)
        cd = None if a is None else a.coord_derivatives
        if cd is None:
            raise ValueError("no coordinate derivatives in this result")
        kw = np.ones(1)                                  # DFTB+ writes KWEIGHT 1
        for k, at in enumerate(cd.atoms):
            for i in range(3):
                block(f"Atom {at + 1} d{_CART[i]} ", cd.eigenvalue_derivatives[k, i])
    else:
        raise ValueError(f"kind must be 'field', 'potential' or 'coordinate', not {kind!r}")
    _write_text(path, "\n".join(lines) + "\n")


# ---- coordinate derivatives and second derivatives (milestone M6g-2) ------------------------
def _fortran_t(prefix: str, col: int, rest: str) -> str:
    """``prefix`` then ``rest`` from column ``col`` (1-based), overwriting what is there (the
    Fortran T edit descriptor: DFTB+'s (1X,I0,T4,...) loses digits from atom 100 on)."""
    buf = list(prefix.ljust(col - 1))
    buf[col - 1:] = list(rest)
    return "".join(buf)


def coord_derivative_tables(res, dftbplus_sign: bool = True) -> str:
    """DFTB+ 25.1's detailed.out blocks of ``CoordDerivatives``: "Derivatives of atomic Mulliken
    charges with atom positions" (per displaced atom K a header (A,I0,T10,A,T26,A,T42,A), then
    per atom B the row (1X,I0,T4,3F16.8) of -dN_B/dx_K in e/Angstrom) and "Born effective
    charges (a.u.)" (per atom three rows 3F16.8, row i the derivative along x_K,i, separated by
    blank lines; DFTB+'s sign -d mu/dx with ``dftbplus_sign``, else the physical one)."""
    cd = _analysis_of(res, "coord_derivatives").coord_derivatives
    natom = cd.atom.shape[-1]
    lines = ["", "", " Derivatives of atomic Mulliken charges with atom positions"]
    for k, at in enumerate(cd.atoms):
        lines.append(_fortran_t(_fortran_t(_fortran_t(f"At{at + 1}", 10, "x"), 26, "y"), 42, "z"))
        for b in range(natom):
            vals = -cd.atom[k, :, 0, b] / BOHR
            lines.append(_fortran_t(f" {b + 1}", 4, "".join(f"{v:16.8f}" for v in vals)))
        lines.append("")
    lines += ["", " Born effective charges (a.u.)", ""]
    sign = -1.0 if dftbplus_sign else 1.0
    for k in range(len(cd.atoms)):
        for i in range(3):
            lines.append("".join(f"{sign * v:16.8f}" for v in cd.born[k, i]))
        lines.append("")
    lines.append("")
    return "\n".join(lines) + "\n"


def external_charge_tables(res) -> str:
    """The development DFTB+'s detailed.out block of ``ExternalChargeDerivatives`` (M6g-3a):
    "Derivatives of atomic charges w.r.t. coordinates of external charges", then per point
    charge j a header (A,I0,T10,A,T26,A,T42,A) and per atom B the row (A,I0,T4,3F16.8) of
    dN_B/dR_j in e/Angstrom (electron populations, whatever the title says), each block closed
    by an empty record.  The development version prints the opposite sign for periodic systems
    (its Ewald derivative is taken with respect to the atom, not the charge)."""
    ex = _analysis_of(res, "external_charge_derivatives").external_charge_derivatives
    natom = ex.atom.shape[-1]
    lines = ["Derivatives of atomic charges w.r.t. coordinates of external charges"]
    for k, j in enumerate(ex.charges):
        lines.append(_fortran_t(_fortran_t(_fortran_t(f"Chg#{j + 1}", 10, "x"), 26, "y"), 42, "z"))
        for b in range(natom):
            vals = ex.atom[k, :, 0, b] / BOHR
            lines.append(_fortran_t(f"At{b + 1}", 4, "".join(f"{v:16.8f}" for v in vals)))
        lines.append("")
    return "\n".join(lines) + "\n"


def _deriv_path(path, sd, n_rows: int) -> str:
    """DFTB+'s file name: ``<path>.<first>-<last>`` (1-based moved atoms) for a partial matrix."""
    if n_rows > len(sd.moved_atoms) * 3:
        return f"{path}.{int(sd.moved_atoms[0]) + 1}-{int(sd.moved_atoms[-1]) + 1}"
    return str(path)


def _rows(values, per_line: int, fmt) -> list:
    """Fortran's reuse of a format with ``per_line`` items per record."""
    vals = list(values)
    return ["".join(fmt(v) for v in vals[i:i + per_line]) for i in range(0, len(vals), per_line)]


def write_hessian(path, sd) -> str:
    """hessian.out of :func:`pydftb.vibrations.second_derivatives`: every column (one displaced
    coordinate) as 4f16.10 records; a partial Hessian goes to ``<path>.<first>-<last>``.
    Returns the file written."""
    out = _deriv_path(path, sd, sd.hessian.shape[0])
    lines = []
    for col in range(sd.hessian.shape[1]):
        lines += _rows(sd.hessian[:, col], 4, lambda v: f"{v:16.10f}")
    _write_text(out, "\n".join(lines) + "\n")
    return out


def write_born(path, sd) -> str:
    """born.out: d mu / d x of every displaced coordinate (3f16.10); a partial set of moved
    atoms goes to ``<path>.<first>-<last>``."""
    if sd.dipole_derivatives is None:
        raise ValueError("no dipole derivatives (periodic system or dipole=False)")
    out = _deriv_path(path, sd, 3 * len(sd.atoms))
    lines = ["".join(f"{v:16.10f}" for v in row) for row in sd.dipole_derivatives]
    _write_text(out, "\n".join(lines) + "\n")
    return out


def write_born_derivatives(path, sd) -> str:
    """bornderiv.out: d alpha / d x of every displaced coordinate, the 3 x 3 polarisability in
    Fortran order (9E16.8)."""
    if sd.polarisability_derivatives is None:
        raise ValueError("no polarisability derivatives: the calculator needs "
                         "analysis={'polarisability': True}")
    out = _deriv_path(path, sd, 3 * len(sd.atoms))
    lines = ["".join(fortran_e(v, 16, 8) for v in m.T.ravel())
             for m in sd.polarisability_derivatives]
    _write_text(out, "\n".join(lines) + "\n")
    return out


def born_table(sd) -> str:
    """The SecondDerivatives block of detailed.out: " Born charges/dipole derivatives wrt. atom
    positions (e)", then per moved atom "Atom n" (counted from 1 over the moved atoms, as
    DFTB+) and three rows 3F12.6."""
    if sd.dipole_derivatives is None:
        raise ValueError("no dipole derivatives (periodic system or dipole=False)")
    lines = [" Born charges/dipole derivatives wrt. atom positions (e)"]
    for k in range(len(sd.moved_atoms)):
        lines.append(f"Atom {k + 1}")
        lines += ["".join(f"{v:12.6f}" for v in row)
                  for row in sd.dipole_derivatives[3 * k:3 * k + 3]]
    return "\n".join(lines) + "\n"


def _tag_real(x: float) -> str:
    """E24.15E003, DFTB+'s tagged real format."""
    x = float(x)
    if x == 0.0:
        return ("0." + "0" * 15 + "E+000").rjust(24)
    mant, exp = f"{abs(x):.14E}".split("E")
    body = "0." + mant.replace(".", "") + f"E{int(exp) + 1:+04d}"
    return ("-" + body if x < 0 else body).rjust(24)


def format_tag(label: str, values, kind: str = "real") -> str:
    """One entry of a DFTB+ tag file: "label(20):kind:rank:dims", then the values in column-major
    order, 3 x E24.15E003 (real) or 2 x I33 (integer) per line; ``values`` with the Fortran
    array's shape."""
    a = np.asarray(values)
    if a.ndim == 0:
        head, data = f"{label:<20s}:{kind}:0:", [a.item()]
    else:
        head = f"{label:<20s}:{kind}:{a.ndim}:" + ",".join(str(d) for d in a.shape)
        data = a.ravel(order="F")
    if kind == "real":
        lines = _rows(data, 3, _tag_real)
    elif kind == "integer":
        lines = _rows(data, 2, lambda v: f"{int(v):33d}")
    else:
        raise ValueError(f"tag kind {kind!r}: 'real' or 'integer'")
    return "\n".join([head] + lines) + "\n"


def write_vibrations_tag(path, nm, modes=None, remove_rigid: bool = False) -> None:
    """vibrations.tag of DFTB+'s ``modes``: saved_modes, eigenmodes, eigenmodes_scaled (the
    ``modes`` columns, 1-based, default all), frequencies (Hartree), the infrared intensities and
    the Raman scattering summed over degenerate groups (scattering without the first group when
    rigid motions were removed, as ``modes``)."""
    sel = np.arange(len(nm.frequencies)) if modes is None else np.asarray(modes, int) - 1
    parts = [format_tag("saved_modes", sel + 1, kind="integer"),
             format_tag("eigenmodes", nm.modes[:, sel]),
             format_tag("eigenmodes_scaled", nm.mass_weighted[:, sel]),
             format_tag("frequencies", nm.frequencies)]
    if nm.ir is not None:
        parts.append(format_tag("intensities", nm.grouped(nm.ir)))
    if nm.raman is not None:
        g = nm.grouped(nm.raman)
        parts.append(format_tag("scattering", g[1:] if remove_rigid else g))
    _write_text(path, "".join(parts))
