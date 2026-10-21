"""Create the ``hotcent_gdo`` test set (Gd with an s, p, d, f basis, and O) with hotcent.

The set exists only to compare pydftb with DFTB+ for f orbitals (milestone M2g); it is not a
physical parametrisation and has no repulsive energy (all polynomial coefficients are zero,
so DFTB+ needs ``PolynomialRepulsive = SetForAll { Yes }``).

The electronic part follows PTBP's recipe (``utils/hotpy.py``): PBE, scalar-relativistic
atoms, density superposition, power confinement, grid 0.4 + 0.02 n Bohr.  O uses PTBP's
published confinement and atomic values, so ``O-O.skf`` has exactly the integrals of
``ptbp_hcocu/O-O.skf``.  PTBP has no Gd parameters: its confinement radii are chosen here,
and its on-site energies, Hubbard values and spin constants are those of the free atom
(hotcent's analytical U and W).  Files with Gd are in DFTB+'s extended format (``@``),
``O-O.skf`` in the simple format.

    python -m pydftb.validation.make_f_set     # needs hotcent and pylibxc; ~2 min

Prints the Gd spin constants used by ``validation/cases.py``.
"""
import gzip
import os
import tempfile

import numpy as np

from .cases import DATA

XC = "GGA_X_PBE+GGA_C_PBE"
RMIN, DR, NPOINTS = 0.4, 0.02, 700
ATOMS = {
    "Gd": {"configuration": "[Xe] 4f7 5d1 6s2 6p0", "valence": ["6s", "6p", "5d", "4f"],
           "occupations": {"6s": 2, "6p": 0, "5d": 1, "4f": 7},
           "r0_w": 6.0, "r0_d": 9.0, "p": 2.0},
    # PTBP's published O parameters (utils/parameters_set.py, PTBP set)
    "O": {"configuration": "[He] 2s2 2p4", "valence": ["2s", "2p"],
          "occupations": {"2s": 2, "2p": 4},
          "r0_w": 2.011, "r0_d": 2.664, "p": 2.0,
          "eigenvalues": {"2s": -0.8802104881, "2p": -0.3320261219},
          "hubbardvalues": {"2s": 0.5830843799, "2p": 0.4889023431}},
}


def free_atom_values(symbol: str, spec: dict):
    """Eigenvalues, Hubbard values and spin constants (Ha) of the free atom."""
    from hotcent.atomic_dft import AtomicDFT
    from hotcent.confinement import PowerConfinement
    val = spec["valence"]
    # a confinement this wide changes nothing; hotcent needs one per subshell for U and W
    atom = AtomicDFT(symbol, xc=XC, configuration=spec["configuration"], valence=val,
                     scalarrel=True, perturbative_confinement=True, txt=None,
                     wf_confinement={nl: PowerConfinement(r0=1e6, s=2) for nl in val})
    atom.run()
    eps = {nl: atom.get_eigenvalue(nl) for nl in val}
    hub = {nl: atom.get_analytical_hubbard_value(nl) for nl in val}
    w = np.array([[atom.get_analytical_spin_constant(a, b) for b in val] for a in val])
    return eps, hub, w


def confined_atom(symbol: str, spec: dict):
    from hotcent.atomic_dft import AtomicDFT
    from hotcent.confinement import PowerConfinement
    atom = AtomicDFT(symbol, xc=XC, configuration=spec["configuration"], valence=spec["valence"],
                     scalarrel=True, confinement=PowerConfinement(r0=spec["r0_d"], s=spec["p"]),
                     wf_confinement={nl: PowerConfinement(r0=spec["r0_w"], s=spec["p"])
                                     for nl in spec["valence"]},
                     txt=None)
    atom.run()
    return atom


def main(out: str | None = None):
    from hotcent.offsite_twocenter import Offsite2cTable
    out = out or os.path.join(DATA, "skf", "hotcent_gdo")
    os.makedirs(out, exist_ok=True)
    spin = {}
    for el, spec in ATOMS.items():
        if "eigenvalues" not in spec:
            spec["eigenvalues"], spec["hubbardvalues"], spin[el] = free_atom_values(el, spec)
    atoms = {el: confined_atom(el, spec) for el, spec in ATOMS.items()}
    with tempfile.TemporaryDirectory() as tmp:
        for a, b in (("Gd", "Gd"), ("Gd", "O"), ("O", "O")):
            table = Offsite2cTable(atoms[a], atoms[b], txt=None)
            table.run(RMIN, DR, NPOINTS, superposition="density", xc=XC)
            spec = ATOMS[a]
            table.write(eigenvalues=spec["eigenvalues"], hubbardvalues=spec["hubbardvalues"],
                        occupations=spec["occupations"], spe=0.0,
                        filename_template=os.path.join(tmp, "{el1}-{el2}.skf"))
        for name in sorted(os.listdir(tmp)):
            with open(os.path.join(tmp, name), "rb") as fh, \
                    gzip.GzipFile(os.path.join(out, name + ".gz"), "wb", mtime=0) as gz:
                gz.write(fh.read())
            print("wrote", os.path.join(out, name + ".gz"))
    for el, w in spin.items():
        print(f"{el} spin constants (shells {' '.join(ATOMS[el]['valence'])}):")
        print(np.array2string(w, precision=6, separator=", "))


if __name__ == "__main__":
    main()
