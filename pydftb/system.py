"""Geometry container used by the high-level API.

Every atom carries an *element* (chemistry: nuclear charge, dispersion,
output) and a *type label* (which parameter set to use).  Normally the two
are identical; giving the same element several labels, e.g. ``Li@el`` and
``Li@cat``, is how one element gets several parameter sets (milestone M5).

Internally everything is in atomic units (Bohr); ``from_ase``/``to_ase``
convert from/to Angstrom.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from .constants import BOHR

LABEL_FORBIDDEN = "- /\\"   # labels end up in SKF file names "A-B.skf"


def element_of_label(label: str) -> str:
    """Element symbol of a type label.

    Accepted forms: ``"O"``, ``"O2"``/``"Li1"`` (element + digits, the DFTB+/ASE
    ``Type2FileNames`` convention), ``"Li_el"`` and ``"Li@el"``.
    """
    for n in (2, 1):
        el, rest = label[:n], label[n:]
        if el in atomic_numbers and (rest == "" or rest[0].isdigit() or rest[0] in "_@"):
            return el
    raise ValueError(f"cannot infer an element from type label {label!r}; "
                     "use e.g. 'Li', 'Li2', 'Li_el' or pass elements explicitly")


@dataclass
class System:
    labels: list[str]                      # parameter type of each atom
    positions: np.ndarray                  # (n, 3) Bohr
    cell: np.ndarray | None = None         # (3, 3) Bohr, rows = lattice vectors
    pbc: np.ndarray = field(default_factory=lambda: np.zeros(3, bool))
    elements: list[str] | None = None      # chemical element of each atom
    charge: float = 0.0                    # total charge (e)

    def __post_init__(self):
        self.labels = list(self.labels)
        for lab in self.labels:
            if any(c in lab for c in LABEL_FORBIDDEN):
                raise ValueError(f"type label {lab!r} must not contain any of {LABEL_FORBIDDEN!r}")
        self.positions = np.asarray(self.positions, float).reshape(-1, 3)
        if len(self.labels) != len(self.positions):
            raise ValueError("labels and positions differ in length")
        if self.elements is None:
            self.elements = [element_of_label(lab) for lab in self.labels]
        if self.cell is not None:
            self.cell = np.asarray(self.cell, float).reshape(3, 3)
        self.pbc = np.broadcast_to(np.asarray(self.pbc, bool), (3,)).copy()
        if self.cell is None and self.pbc.any():
            raise ValueError("periodic boundary conditions need a cell")

    # ------------------------------------------------------------------
    @property
    def natoms(self) -> int:
        return len(self.labels)

    @property
    def periodic(self) -> bool:
        return self.cell is not None and bool(self.pbc.any())

    @property
    def numbers(self) -> np.ndarray:
        return np.array([atomic_numbers[e] for e in self.elements])

    @property
    def unique_labels(self) -> list[str]:
        return list(dict.fromkeys(self.labels))

    @classmethod
    def from_ase(cls, atoms: Atoms, labels: list[str] | None = None,
                 charge: float | None = None) -> "System":
        """Convert an ASE ``Atoms`` (Angstrom).  ``labels`` default to elements."""
        elements = atoms.get_chemical_symbols()
        periodic = bool(atoms.pbc.any())
        if charge is None:
            charge = float(atoms.get_initial_charges().sum())
        return cls(labels=list(labels) if labels is not None else list(elements),
                   positions=atoms.positions / BOHR,
                   cell=atoms.cell.array / BOHR if periodic else None,
                   pbc=atoms.pbc.copy(), elements=elements, charge=charge)

    def to_ase(self) -> Atoms:
        atoms = Atoms(self.elements, positions=self.positions * BOHR,
                      cell=None if self.cell is None else self.cell * BOHR, pbc=self.pbc)
        return atoms
