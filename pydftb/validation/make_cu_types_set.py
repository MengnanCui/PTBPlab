"""Create the ``cu_types`` test set: two parameter sets of copper, ``Cu`` and ``Cu2``, and the
cross files ``Cu-Cu2.skf`` and ``Cu2-Cu.skf``, for smooth switching between the parameter sets
of one element (milestone M5a, :mod:`pydftb.types`).

Both types follow PTBP's recipe (``utils/hotpy.py``): PBE, scalar-relativistic atoms with
self-consistent power confinement (p = 2) of every valence subshell (r0_w) and of the density
(r0_d), density superposition, grid 0.4 + 0.02 n Bohr (700 points), and PTBP's table values for
the on-site energies, Hubbard values and occupations.  So the two types differ only in their H
and S tables and their repulsives, as types made by PTBP do:

* ``Cu``: PTBP's published confinement of Cu (r0_w 2.749, r0_d 12.036 Bohr);
* ``Cu2``: looser orbitals (r0_w 3.0, r0_d 12.036 Bohr): integrals 2-20 % larger at 3-5.5
  Bohr, fcc lattice constant 3.77 instead of 3.65 A, 0.03-0.05 Ha per atom less bound.

hotcent 2.1 assumes one atom per element: for two atoms with the same symbol it computes one
direction only, takes the wave-function range of the first atom, and writes the files under the
element symbol with the on-site line.  The cross table is therefore run for both directions with
the range of both atoms (:class:`CrossTable`), and its two files are written with hotcent's
``write_skf`` in the heteronuclear format (no on-site line: DFTB+ decides the format from the
species names, and a hetero-label file with an on-site line would be misread silently), named
after the labels.

Repulsive: physrep (``tools/physrep.py``, PTBP's repulsive, with k_xc = 0 as PTBP uses it):
V(r) = Z^2 / r - Z^2 erf(r / sqrt(a_t^2 + a_u^2)) / r, a = sigma_rep x the covalent radius, cut
where |V| first exceeds 1e-8 Ha from outside (+ 0.1 Bohr), a natural cubic spline on physrep's
grid (0.5-20 Bohr, 700 points), written as physrep writes it.  sigma_rep is 0.42 for Cu and
0.45 for Cu2 (chosen here: PTBP's published 0.299 does not suit these tables, it gives a Cu
dimer without a minimum above 1.4 A and compressed fcc below the equilibrium energy, so clusters
collapse in MD; PTBP's own search at these r0 with the repository's data now gives 0.403, and
0.42 puts the dimer minimum at 2.2 A); the cross pair uses both sizes, so the repulsive is mixed
as well.

    python -m pydftb.validation.make_cu_types_set     # needs hotcent and pylibxc

Deterministic; for testing only, not a physical parametrisation.
"""
from __future__ import annotations

import gzip
import io
import math
import multiprocessing as mp
import os

import numpy as np

from .cases import DATA

XC = "GGA_X_PBE+GGA_C_PBE"
RMIN, DR, NPOINTS = 0.4, 0.02, 700            # utils/hotcent_parallel.py
CONFIGURATION = "[Ar] 3d10 4s1 4p0"
VALENCE = ["3d", "4s", "4p"]
OCCUPATIONS = {"d": 10, "p": 0, "s": 1}
# PTBP's Cu table values (utils/parameters_set.py, PTBP set), Hartree
EIGENVALUES = {"d": -0.1851306924, "p": -0.0247482516, "s": -0.1691679514}
HUBBARD = {"d": 0.5095012987, "p": 0.0799802838, "s": 0.2403269013}
TYPES = {
    "Cu": {"r0_w": 2.749, "r0_d": 12.036, "sigma_rep": 0.42},          # PTBP's published r0
    "Cu2": {"r0_w": 3.0, "r0_d": 12.036, "sigma_rep": 0.45},           # looser orbitals
}
# physrep's grid and cut-off (tools/physrep.py, as PTBP calls it)
REP_RMIN, REP_RMAX, REP_NPOINTS, REP_ECUT = 0.5, 20.0, 700, 1e-8
Z_CU = 29


def confined_atom(r0_w: float, r0_d: float):
    """PTBP's confined atom (``hotpy.hot_hamiltonian``, self-consistent confinement)."""
    from hotcent.atomic_dft import AtomicDFT
    from hotcent.confinement import PowerConfinement
    atom = AtomicDFT("Cu", xc=XC, configuration=CONFIGURATION, valence=VALENCE, scalarrel=True,
                     confinement=PowerConfinement(r0=r0_d, s=2.0),
                     wf_confinement={nl: PowerConfinement(r0=r0_w, s=2.0) for nl in VALENCE},
                     perturbative_confinement=False, txt=None)
    atom.run()
    return atom


def _cross_table_class():
    from hotcent.offsite_twocenter import Offsite2cTable

    class CrossTable(Offsite2cTable):
        """``Offsite2cTable`` of two atoms with the same symbol but different confinements:
        both directions and the wave-function range of both atoms, as hotcent does for two
        elements."""

        def __init__(self, ela, elb, **kwargs):
            super().__init__(ela, elb, **kwargs)
            self.nel, self.pairs, self.elements = 2, [(ela, elb), (elb, ela)], [ela, elb]

    return CrossTable


def _run_table(task):
    """Worker (forked, atoms inherited): the tables of one run, {(i, j): table} and the grid."""
    a, b = task
    from hotcent.offsite_twocenter import Offsite2cTable
    atom_a, atom_b = _ATOMS[a], _ATOMS[b]
    table = (Offsite2cTable if a == b else _cross_table_class())(atom_a, atom_b, txt=None)
    table.run(RMIN, DR, NPOINTS, superposition="density", xc=XC)
    out = {(a, b): table.tables[(0, 0, 0)]}
    if a != b:
        out[(b, a)] = table.tables[(1, 0, 0)]
    return out, table.Rgrid


_ATOMS: dict = {}


def physrep_spline(sigma_a: float, sigma_b: float):
    """physrep's repulsive spline (scipy ``CubicSpline``) for two atom sizes, k_xc = 0."""
    from ase.data import covalent_radii
    from ase.units import Bohr
    from scipy import interpolate
    from scipy.special import erf
    r = np.linspace(REP_RMIN, REP_RMAX, REP_NPOINTS)
    a1 = (covalent_radii[Z_CU] / Bohr) * sigma_a        # physrep's atom size (ASE radius)
    a2 = (covalent_radii[Z_CU] / Bohr) * sigma_b
    gamma = 1.0 / np.sqrt(a1 ** 2 + a2 ** 2)
    zz = Z_CU * Z_CU
    v = -zz * np.array([erf(gamma * x) / x for x in r]) + zz * np.array([1.0 / x for x in r])
    rcut = r[-1] + 0.1
    for ind in range(len(r) - 1, -1, -1):          # the largest r with |V| > Ecut, + 0.1
        if abs(v[ind]) > REP_ECUT:
            rcut = r[ind] + 0.1
            break
    keep = r <= rcut
    return interpolate.CubicSpline(r[keep], v[keep], bc_type="natural")


def spline_text(spl) -> str:
    """The spline block as physrep's ``write_skf`` writes it."""
    nint, coef, x = spl.c.shape[1], spl.c, spl.x
    lines = ["Spline \n", f"{nint} {x[-1]}\n",
             f"1.0 0.0 {coef[3, 0] - math.exp(-1.0 * x[0] + 0.0) - 0.0} \n"]
    for i in range(nint - 1):
        lines.append(f"{x[i]} {x[i + 1]} {coef[3, i]} {coef[2, i]} {coef[1, i]} {coef[0, i]} \n")
    i = nint - 1
    lines.append(f"{x[i]} {x[i + 1]} {coef[3, i]} {coef[2, i]} {coef[1, i]} {coef[0, i]} "
                 f"{0.0} {0.0} \n")
    return "".join(lines)


def skf_text(rgrid, table, homo: bool, spl) -> str:
    from ase.data import atomic_masses, atomic_numbers
    from hotcent.slako import write_skf
    buf = io.StringIO()
    write_skf(buf, rgrid, table, homo, False, EIGENVALUES if homo else {},
              HUBBARD if homo else {}, OCCUPATIONS if homo else {}, 0.0,
              atomic_masses[atomic_numbers["Cu"]], False, {}, {})
    return buf.getvalue() + spline_text(spl)


def main(out: str | None = None, workers: int = 3):
    out = out or os.path.join(DATA, "skf", "cu_types")
    os.makedirs(out, exist_ok=True)
    for lab, spec in TYPES.items():
        _ATOMS[lab] = confined_atom(spec["r0_w"], spec["r0_d"])
    runs = [("Cu", "Cu"), ("Cu2", "Cu2"), ("Cu", "Cu2")]
    with mp.get_context("fork").Pool(min(workers, len(runs))) as pool:
        results = pool.map(_run_table, runs)
    tables, rgrid = {}, None
    for tab, grid in results:
        tables.update(tab)
        rgrid = grid
    for (a, b), table in sorted(tables.items()):
        spl = physrep_spline(TYPES[a]["sigma_rep"], TYPES[b]["sigma_rep"])
        text = skf_text(rgrid, table, a == b, spl)
        path = os.path.join(out, f"{a}-{b}.skf.gz")
        with gzip.GzipFile(path, "wb", mtime=0) as gz:
            gz.write(text.encode())
        print("wrote", path)


if __name__ == "__main__":
    main()
