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
  and atom (and the imaginary blocks with dual spin-orbit coupling).

``charges.bin`` is a Fortran stream file (no record markers, 4-byte integers and logicals);
``charges.dat`` is the same in Fortran list-directed text, written here exactly as gfortran
writes it.  How a calculation's mixing vector maps onto these arrays is
:meth:`pydftb.DFTB.charge_data` and :meth:`pydftb.DFTB.populations_from_file`.
"""
from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field

import numpy as np

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
    density_matrix: bool = False          # hybrid functionals' restart data (not supported)
    kpoint_info: bool = False
    n_dipole: int = 0
    n_quadrupole: int = 0
    extra: dict = field(default_factory=dict)

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
        return np.array([float(x.replace("D", "E").replace("d", "e")) for x in self.items(n)])


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
    if rho:
        raise ValueError("the charge file has density-matrix data (hybrid functionals), which "
                         "pydftb does not read")
    return ChargeData(q, norb_atom, blocks, iblocks, checksum, version, multipolar, rho,
                      kpoint_info, n_dip, n_quad)


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
    if data.density_matrix or data.kpoint_info or data.n_dipole or data.n_quadrupole:
        raise ValueError("pydftb writes only populations and blocks")
    norb = np.asarray(data.norb_atom, int)
    edges = np.concatenate([[0], np.cumsum(norb)])
    head_flags = [data.blocks is not None, data.iblocks is not None, False, False, True]
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
