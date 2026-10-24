"""DFTB+'s corrections of milestone M6f-3: the H5 hydrogen-bond correction of gamma, the
halogen correction of DFTB3 and the dipole correction of slabs.

Ports of DFTB+ (041e723; the first three files are the same in 25.1):

* ``dftb/vdwdata.F90``: the van der Waals radii of Mantina et al. (:data:`VDW_PM`);
* ``dftb/h5correction.F90``: H5 scales the short-range part S of gamma = 1/r - S for pairs
  with exactly one H, S' = S (1 + g) - g / r (that is gamma' = gamma (1 + g)), with
  g = k exp(-(r - r0)^2 / (2 c^2)), r0 = RScaling (R_H + R_X), c = WScaling (R_H + R_X) /
  (2 sqrt(2 ln 2)) (:class:`H5`; :class:`pydftb.coulomb.Coulomb` applies it);
* ``dftb/halogenx.F90``: a pair term between O/N and Cl/Br/I, switched on between 0.7 and
  0.8 of the van der Waals radius sum (:func:`halogen_terms`);
* ``dftb/dipolecorr.F90`` (DFTB+ after 25.1): a dipole layer in the vacuum of a slab along z,
  E = 2 pi D^2 / V with D = sum_A dq_A (z_f,A - z_c) (:class:`DipoleCorrection`); written
  for the normal of the slab plane, which is z in every geometry DFTB+ accepts, so that the
  strain derivative (DFTB+ has none) is defined in all directions.

pydftb identifies H, the H5 acceptors and the halogen pairs by element
(:func:`pydftb.system.element_of_label`); DFTB+ by species name, which differs only for type
labels with a suffix ("O2" is O here, no H5 acceptor in DFTB+).  Atomic units unless stated.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .constants import BOHR, KCAL_MOL
from .neighbors import neighbor_list
from .system import element_of_label

AA_BOHR = 1.0 / BOHR             # DFTB+ AA__Bohr

# van der Waals radii (pm) of DFTB+'s vdwdata.F90 (Mantina et al.); elements without a radius
# are left out
VDW_PM = {
    "H": 120, "He": 140, "Li": 182, "Be": 153, "B": 192, "C": 170, "N": 155, "O": 152,
    "F": 147, "Ne": 154, "Na": 227, "Mg": 173, "Al": 184, "Si": 210, "P": 180, "S": 180,
    "Cl": 175, "Ar": 188, "K": 275, "Ca": 231, "Sc": 211, "Ni": 163, "Cu": 140, "Zn": 139,
    "Ga": 187, "Ge": 211, "As": 185, "Se": 190, "Br": 185, "Kr": 202, "Rb": 303, "Sr": 249,
    "Pd": 163, "Ag": 172, "Cd": 158, "In": 193, "Sn": 217, "Sb": 206, "Te": 206, "I": 198,
    "Xe": 216, "Cs": 343, "Ba": 268, "Pt": 175, "Au": 166, "Hg": 155, "Tl": 196, "Pb": 202,
    "Bi": 207, "Po": 197, "At": 202, "Rn": 220, "Fr": 348, "Ra": 283, "U": 186}


def vdw_radius(element: str) -> float | None:
    """The van der Waals radius (Bohr) of an element, as DFTB+ converts it; None if unknown."""
    pm = VDW_PM.get(element)
    return None if pm is None else 0.01 * pm * AA_BOHR


# ---- H5 -----------------------------------------------------------------------------------
H5_SCALING = {"O": 0.06, "N": 0.18, "S": 0.21}     # DFTB+'s defaults (by species name there)
GAUSS_WIDTH = 0.5 / np.sqrt(2.0 * np.log(2.0))     # c = WScaling (R_H + R_X) * this


@dataclass(frozen=True)
class H5:
    """DFTB+ ``HCorrection = H5 { RScaling; WScaling; H5Scaling {...} }``: ``scaling`` maps
    elements to k (added to the defaults O 0.06, N 0.18, S 0.21); k <= 0 switches an element
    off."""
    r_scaling: float = 0.714
    w_scaling: float = 0.25
    scaling: dict = field(default_factory=dict)

    @classmethod
    def make(cls, spec) -> "H5 | None":
        """None/False, True (the paper's values), a dict of the fields, or an H5."""
        if spec is None or spec is False:
            return None
        if spec is True:
            h5 = cls()
        elif isinstance(spec, H5):
            h5 = spec
        elif isinstance(spec, dict):
            unknown = set(spec) - {"r_scaling", "w_scaling", "scaling"}
            if unknown:
                raise ValueError(f"h5: unknown keys {sorted(unknown)} "
                                 "(use 'r_scaling', 'w_scaling', 'scaling')")
            h5 = cls(float(spec.get("r_scaling", 0.714)), float(spec.get("w_scaling", 0.25)),
                     {str(k): float(v) for k, v in dict(spec.get("scaling", {})).items()})
        else:
            raise TypeError(f"h5 must be None, True, a dict or an H5, not {type(spec).__name__}")
        if not h5.w_scaling > 0.0:
            raise ValueError(f"h5 w_scaling must be positive, not {h5.w_scaling!r}")
        return h5

    def atom_parameters(self, labels) -> tuple:
        """(is_h, k, R) per atom for :class:`pydftb.coulomb.Coulomb`'s ``h5`` with
        (r_scaling, w_scaling): k <= 0 means no correction for the atom's pairs with H, R the
        van der Waals radius (Bohr).  An acceptor without a radius warns and is left out, as
        in DFTB+."""
        elements = [element_of_label(lab) for lab in labels]
        is_h = np.array([el == "H" for el in elements])
        if not is_h.any():
            raise ValueError("H5 correction used without H atoms present (as DFTB+)")
        scal = {**H5_SCALING, **self.scaling}
        k = np.full(len(elements), -1.0)
        radius = np.zeros(len(elements))
        missing = set()
        for a, el in enumerate(elements):
            r = vdw_radius(el)
            radius[a] = 0.0 if r is None else r
            if el == "H":
                continue
            kk = float(scal.get(el, -1.0))
            if kk > 0.0 and r is None:
                missing.add(el)
                kk = -1.0
            k[a] = kk
        for el in sorted(missing):
            warnings.warn(f"The van der Waals radius for {el} is required for the H5 correction "
                          f"but is not available. H-{el} contributions therefore neglected "
                          "(as DFTB+)", RuntimeWarning, stacklevel=3)
        return is_h, k, radius, self.r_scaling, self.w_scaling


def h5_scale(s, ds, r, k, sum_vdw, r_scaling, w_scaling):
    """S' and dS'/dr of DFTB+'s scaleShortGamma / scaleShortGammaDeriv for pairs with
    scaling k (> 0) and van der Waals radius sum ``sum_vdw``."""
    r0 = r_scaling * sum_vdw
    cc = w_scaling * sum_vdw * GAUSS_WIDTH
    gauss = np.exp(-0.5 * (r - r0) ** 2 / cc ** 2) * k
    dgauss = -gauss * (r - r0) / cc ** 2
    ds_new = (s * dgauss + ds * (1.0 + gauss)) - (dgauss / r - gauss / r ** 2)
    return s * (1.0 + gauss) - gauss / r, ds_new


# ---- halogen correction ---------------------------------------------------------------------
HALOGEN_ACCEPTORS = ("O", "N")
HALOGENS = ("Cl", "Br", "I")
HALOGEN_D = {("O", "Cl"): 1.237, ("O", "Br"): 1.099, ("O", "I"): 1.313,     # Angstrom
             ("N", "Cl"): 1.526, ("N", "Br"): 1.349, ("N", "I"): 1.521}
HALOGEN_C = (7.761, 0.050, 4.518)       # kcal/mol, 1/A^c3, -
_MIN_INTERACTION = 1.0e-14              # kcal/mol: the pair energy at the cut-off
_R0, _R1 = 0.7, 0.8                     # switching region, in units of the radius sum


@dataclass
class HalogenTerms:
    """Energy (Ha), gradient dE/dR (n_atom, 3), virial dE/d(strain) (3, 3; periodic) and the
    energy per atom (half of every pair, as DFTB+'s atomHalogenX)."""
    energy: float
    gradient: np.ndarray | None
    virial: np.ndarray | None
    atom_energies: np.ndarray


def halogen_pair_types(labels) -> set:
    """The (acceptor, halogen) element pairs present; an error if there are none (DFTB+ reads
    ``HalogenXCorr`` only when the structure has O/N and Cl/Br/I)."""
    elements = {element_of_label(lab) for lab in labels}
    present = {(a, x) for a in HALOGEN_ACCEPTORS if a in elements
               for x in HALOGENS if x in elements}
    if not present:
        raise ValueError("No suitable (O-X or N-X) halogen combinations present for this "
                         "correction (as DFTB+)")
    return present


def halogen_cutoff(pair_types) -> float:
    """DFTB+'s cut-off (Bohr): the largest d present plus the distance where a pair falls to
    2e-14 kcal/mol."""
    c1, c2, c3 = HALOGEN_C
    d_max = max(HALOGEN_D[p] for p in pair_types)
    return (d_max + (-np.log(2.0 * _MIN_INTERACTION / c1) / c2) ** (1.0 / c3)) * AA_BOHR


def halogen_terms(labels, pos, cell=None, pbc=None, grad: bool = True,
                  compat: bool = False) -> HalogenTerms:
    """DFTB+'s halogen correction (``HalogenXCorr``) of a geometry (positions in Bohr).

    E_pair = c1 [sigma e^(-c2 (r - d)^c3) + (1 - sigma) e^(-c2 (0.7 R - d)^c3)] (kcal/mol, A),
    R the sum of the two van der Waals radii, sigma the 7th-order switch from 0 at 0.7 R to 1
    at 0.8 R.  Below 0.7 R the term is constant and f(r) is not evaluated (DFTB+ multiplies
    sigma = 0 by f(r), NaN for r < d).  The gradient is the derivative of this energy;
    ``compat`` (test use only) flips the sign of the d(sigma)/dr terms in the gradient and the
    virial as DFTB+'s getEnergyDeriv does."""
    pair_types = halogen_pair_types(labels)
    elements = [element_of_label(lab) for lab in labels]
    periodic = cell is not None and pbc is not None and np.any(pbc)
    i, j, _, vec, dist = neighbor_list(np.asarray(pos, float), cell if periodic else None,
                                       pbc if periodic else None, halogen_cutoff(pair_types))
    keep = np.array([(elements[a], elements[b]) in HALOGEN_D for a, b in zip(i, j)], bool)
    i, j, vec, dist = i[keep], j[keep], vec[keep], dist[keep]
    natom = len(elements)
    c1, c2, c3 = HALOGEN_C
    d = np.array([HALOGEN_D[(elements[a], elements[b])] for a, b in zip(i, j)])
    radius = {el: vdw_radius(el) for el in set(elements) if el in HALOGEN_ACCEPTORS + HALOGENS}
    rvdw = np.array([(radius[elements[a]] + radius[elements[b]]) * BOHR for a, b in zip(i, j)])
    r = dist * BOHR                                       # Angstrom, as DFTB+
    x = (r - _R0 * rvdw) / ((_R1 - _R0) * rvdw)
    inside = (r > _R0 * rvdw) & (r < _R1 * rvdw)
    sig = np.where(r < _R0 * rvdw, 0.0, 1.0)
    sig[inside] = (-20.0 * x[inside] ** 7 + 70.0 * x[inside] ** 6 - 84.0 * x[inside] ** 5
                   + 35.0 * x[inside] ** 4)
    dsig = np.zeros_like(r)
    dsig[inside] = ((-140.0 * x[inside] ** 6 + 420.0 * x[inside] ** 5 - 420.0 * x[inside] ** 4
                     + 140.0 * x[inside] ** 3) / ((_R1 - _R0) * rvdw[inside]))
    on = r >= _R0 * rvdw                                  # f(r) needed (and r > d there)
    fx = np.zeros_like(r)
    fx[on] = 0.5 * c1 * np.exp(-c2 * (r[on] - d[on]) ** c3)
    dfx = np.zeros_like(r)
    dfx[on] = fx[on] * c2 * c3 * (r[on] - d[on]) ** (c3 - 1.0)    # = -d fx/dr (DFTB+'s dfx)
    fx0 = 0.5 * c1 * np.exp(-c2 * (_R0 * rvdw - d) ** c3)
    half = (sig * fx + (1.0 - sig) * fx0) * KCAL_MOL      # each atom of the pair gets this
    atom_e = np.zeros(natom)
    np.add.at(atom_e, i, half)
    np.add.at(atom_e, j, half)
    gradient = virial = None
    if grad:
        s = -1.0 if compat else 1.0
        dedr = 2.0 * KCAL_MOL * BOHR * (s * dsig * (fx - fx0) - sig * dfx)   # dE_pair/dr (Bohr)
        g = (dedr / dist)[:, None] * vec                  # dE/dR_j; -g on i
        gradient = np.zeros((natom, 3))
        np.add.at(gradient, j, g)
        np.add.at(gradient, i, -g)
        if periodic:
            virial = np.einsum("p,pa,pb->ab", dedr / dist, vec, vec)
    return HalogenTerms(float(atom_e.sum()), gradient, virial, atom_e)


# ---- dipole correction ----------------------------------------------------------------------
ELEC_TOL_MAX = 1.0e-7                    # DFTB+ elecTolMax: the largest net charge allowed
_CELL_TOL = 1.0e-12                      # DFTB+'s test for zero lattice-vector components


@dataclass(frozen=True)
class DipoleCorrection:
    """DFTB+ ``DipoleCorrection { DipoleLayerPos = z0 }`` (after 25.1): a dipole layer at
    z0 (Bohr; None: -h/2, equivalent to h/2) in the vacuum of a slab along z."""
    layer_position: float | None = None

    @classmethod
    def make(cls, spec) -> "DipoleCorrection | None":
        """None/False, True, a dict {"layer_position": z0} or a DipoleCorrection."""
        if spec is None or spec is False:
            return None
        if spec is True:
            return cls()
        if isinstance(spec, DipoleCorrection):
            return spec
        if isinstance(spec, dict):
            unknown = set(spec) - {"layer_position"}
            if unknown:
                raise ValueError(f"dipole_correction: unknown keys {sorted(unknown)} "
                                 "(use 'layer_position')")
            z0 = spec.get("layer_position")
            return cls(None if z0 is None else float(z0))
        raise TypeError("dipole_correction must be None, True, a dict or a DipoleCorrection, "
                        f"not {type(spec).__name__}")

    def setup(self, pos, cell, pbc, check: bool = True) -> "DipoleSetup":
        """The folded heights of this geometry along the slab normal n = a_p x a_q / |...|,
        a_p and a_q the two lattice vectors in the slab plane: the period along n is
        h = V / |a_p x a_q|, the layer at height z0 along n.

        ``check`` (default) asks for DFTB+'s geometry: exactly one lattice vector with a z
        component (the slab normal vector) and that vector along z, so n = e_z and h = a_z.
        DFTB+ only checks that the third vector lies along z, and runs meaningless cells with
        a tilted first or second vector.  Without the check (tests of the strain derivative)
        the normal vector is the one with the largest z component."""
        if cell is None or pbc is None or not np.all(pbc):
            raise ValueError("Dipole correction can only be applied to periodic systems "
                             "(as DFTB+)")
        cell = np.asarray(cell, float)
        zcomp = np.abs(cell[:, 2])
        if check:
            if np.count_nonzero(zcomp > _CELL_TOL) != 1:
                raise ValueError("Dipole correction only applicable if only one lattice vector "
                                 "(the slab normal vector) has a non-zero z component (as "
                                 "DFTB+, which checks the third lattice vector only)")
            normal_vec = int(np.argmax(zcomp))
            if np.any(np.abs(cell[normal_vec, :2]) > _CELL_TOL):
                raise ValueError("Dipole correction only applicable if the slab normal vector "
                                 "has vanishing components apart from z (as DFTB+)")
        else:
            normal_vec = int(np.argmax(zcomp))
        p, q = [k for k in range(3) if k != normal_vec]
        m = np.cross(cell[p], cell[q])
        n = m / np.linalg.norm(m)
        if n[2] < 0.0:
            n = -n
        vol = abs(np.linalg.det(cell))
        h = vol / np.linalg.norm(m)
        z0 = -0.5 * h if self.layer_position is None else self.layer_position
        u = np.asarray(pos, float) @ n
        uf = np.mod(u - z0, h) + z0
        return DipoleSetup(uf - (z0 + 0.5 * h), n, h, vol, z0)


@dataclass
class DipoleSetup:
    """The dipole correction of one geometry: ``height_rel`` = z_f - z_c of every atom along
    the slab normal ``normal``, ``period`` the cell's period along it (Bohr)."""
    height_rel: np.ndarray
    normal: np.ndarray
    period: float
    volume: float
    layer_position: float

    def dipole(self, dq_atom) -> float:
        """D = sum_A dq_A (z_f,A - z_c) (e Bohr; dq = electrons in excess); every update
        requires a neutral system, as in DFTB+."""
        dq_atom = np.asarray(dq_atom, float)
        if abs(dq_atom.sum()) > ELEC_TOL_MAX:
            raise ValueError("Dipole correction only applicable for charge neutral systems "
                             f"(as DFTB+; net charge {-dq_atom.sum():.3e})")
        return float(np.sum(self.height_rel * dq_atom))

    def potential(self, dq_atom) -> np.ndarray:
        """V_A = (4 pi / V) D (z_f,A - z_c) per atom (Hartree/e)."""
        return 4.0 * np.pi / self.volume * self.dipole(dq_atom) * self.height_rel

    def energy(self, dq_atom) -> float:
        """E = 1/2 sum_A dq_A V_A = 2 pi D^2 / V (DFTB+ books it in E_SCC)."""
        return 2.0 * np.pi * self.dipole(dq_atom) ** 2 / self.volume

    def gradient(self, dq_atom) -> np.ndarray:
        """dE/dR at fixed charges: (4 pi / V) D dq_A n."""
        dq_atom = np.asarray(dq_atom, float)
        return (4.0 * np.pi / self.volume * self.dipole(dq_atom) * dq_atom)[:, None] * self.normal

    def virial(self, dq_atom) -> np.ndarray:
        """dE/d(strain) at fixed charges, E (2 n n - 1): a strain eps scales the heights along
        n and the period by 1 + n.eps.n (the folding fixed; z_c drops out of D for a neutral
        system) and the volume by 1 + tr(eps).  DFTB+ has no stress term for the correction:
        its stress lacks (E/V) diag(1, 1, -1)."""
        e = self.energy(dq_atom)
        return e * (2.0 * np.outer(self.normal, self.normal) - np.eye(3))
