"""Second derivatives by finite differences and normal modes (milestone M6g-2): DFTB+'s
``Driver = SecondDerivatives`` (hessian.out, born.out, bornderiv.out) and its ``modes``
program (frequencies, infrared and Raman intensities).

:func:`second_derivatives` differentiates the energy gradient by central differences exactly
as DFTB+ (``derivs/numderivs2.F90``): the moved atoms in ascending order, x, y, z of each, every
coordinate first displaced by -delta, then by +delta (6 n_moved single points; the undisplaced
geometry is computed once more here, for the charges every displaced SCC starts from).  Column
``col`` of the Hessian is d g / d x_col for the gradient components of the derivative atoms
(rows); the dipole (molecules) and, when the calculator has the ``polarisability`` analysis,
the first frequency's polarisability are differentiated in the same pass.  Any calculation
that gives forces can be differentiated (any backend, periodic systems, spin, DFTB3, fields).

:func:`normal_modes` follows ``modes`` (``app/modes``): the Hessian divided by
sqrt(m_I m_J), optionally projected (P = 1 - V V^T, V the mass-weighted translations and
rotations about the principal axes; zero moments, linear molecules, skipped), diagonalised;
frequencies sign(lambda) sqrt|lambda| in Hartree; the displacement of a mode is
M^-1/2 times its eigenvector, normalised; every eigenvector's first component above 100 eps
is made positive; intensities |sum_K Z_K d_K|^2 (infrared) and sum (sum_K dalpha/dx_K d_K)^2
(Raman: the polarisability change) for lambda > eps only, summed over modes closer than
1e-2 cm^-1 for vibrations.tag.  ``modes`` symmetrises nothing: it reads the upper triangle,
or the lower one when it projects; pydftb averages the two by default (``triangle="modes"``
reproduces it).  ``modes`` also gives the projected-out rigid motions intensities whenever
their lambda (zero up to rounding) comes out positive; pydftb gives them none.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import AMU_AU, HARTREE_CM

DELTA = 1e-4                 # Bohr, DFTB+'s default SecondDerivatives Delta
DEGENERATE_CM = 1e-2         # cm^-1: modes merged in vibrations.tag (modes)


@dataclass
class SecondDerivatives:
    """Finite-difference derivatives (atomic units).  ``hessian[row, col]`` = d g_row / d x_col
    with the gradient components of ``atoms`` as rows and the coordinates of ``moved_atoms``
    as columns (DFTB+'s hessian.out holds one column per line group); ``dipole_derivatives
    [col, j]`` = d mu_j / d x_col (born.out; molecules); ``polarisability_derivatives[col, j,
    i]`` = d alpha_ji / d x_col with alpha_ji = d mu_j / d E_i (bornderiv.out); ``energy``
    the undisplaced geometry's energy (Mermin free energy)."""
    atoms: np.ndarray
    moved_atoms: np.ndarray
    delta: float
    hessian: np.ndarray
    dipole_derivatives: np.ndarray | None
    polarisability_derivatives: np.ndarray | None
    energy: float

    @property
    def partial(self) -> bool:
        """Rows and columns differ (DFTB+ writes hessian.out.<first>-<last>)."""
        return len(self.atoms) != len(self.moved_atoms)


@dataclass
class NormalModes:
    """Vibrational modes in ascending order of the eigenvalues.  ``frequencies`` in Hartree
    (imaginary modes negative; ``wavenumbers`` in cm^-1); ``modes[:, j]`` the normalised
    Cartesian displacements (vibrations.tag eigenmodes), ``mass_weighted[:, j]`` the
    eigenvectors of the mass-weighted Hessian (eigenmodes_scaled); ``ir``, ``raman`` per mode
    (0 where lambda <= eps and for projected-out rigid motions); ``groups`` the modes closer
    than 1e-2 cm^-1."""
    frequencies: np.ndarray
    modes: np.ndarray
    mass_weighted: np.ndarray
    ir: np.ndarray | None
    raman: np.ndarray | None
    groups: list

    @property
    def wavenumbers(self) -> np.ndarray:
        return self.frequencies * HARTREE_CM

    def grouped(self, values) -> np.ndarray:
        """Per-mode values summed over every group of (near) degenerate modes."""
        values = np.asarray(values)
        return np.array([values[g].sum() for g in self.groups])


def masses_of(calc, labels) -> np.ndarray:
    """Masses (amu) of the labels from their homonuclear SK files (DFTB+, ``modes``)."""
    return np.array([float(calc.sk.files[(lab, lab)].mass) for lab in labels])


def _atom_list(sel, natom: int, what: str) -> np.ndarray:
    if sel is None:
        return np.arange(natom)
    out = np.unique(np.asarray(sel, dtype=int).ravel())
    if out.size == 0 or out.min() < 0 or out.max() >= natom:
        raise ValueError(f"second_derivatives: invalid {what} {sel}")
    return out


def second_derivatives(calc, labels, positions, cell=None, pbc=None, *, atoms=None,
                       moved_atoms=None, delta: float = DELTA,
                       dipole: bool = True) -> SecondDerivatives:
    """DFTB+'s ``Driver = SecondDerivatives``: central differences of the energy gradient
    (positions and ``delta`` in Bohr).  ``atoms`` (default all) are the derivative atoms,
    ``moved_atoms`` (default ``atoms``, a subset of them) the displaced ones; both 0-based."""
    pos0 = np.asarray(positions, float)
    natom = len(labels)
    rows = _atom_list(atoms, natom, "atoms")
    moved = _atom_list(moved_atoms, natom, "moved_atoms") if moved_atoms is not None else rows
    if not set(moved) <= set(rows):
        raise ValueError("second_derivatives: moved_atoms must be among atoms (as DFTB+)")
    if not delta > 0.0:
        raise ValueError("second_derivatives: delta must be positive")
    periodic = cell is not None and pbc is not None and bool(np.any(pbc))
    ref = calc.calculate(labels, pos0, cell, pbc)
    want_pol = (calc.analysis is not None and calc.analysis.polarisability is not False)
    want_mu = dipole and not periodic and ref.dipole is not None
    ncol = 3 * len(moved)
    hess = np.zeros((3 * len(rows), ncol))
    dmu = np.zeros((ncol, 3)) if want_mu else None
    dpol = np.zeros((ncol, 3, 3)) if want_pol else None
    q_init = getattr(ref, "q_out", None) if calc.scc else None
    for k, atom in enumerate(moved):
        for c in range(3):
            col = 3 * k + c
            for sign in (-1.0, 1.0):
                pos = pos0.copy()
                pos[atom, c] += sign * delta
                r = calc.calculate(labels, pos, cell, pbc, q_init=q_init)
                hess[:, col] += sign * (-np.asarray(r.forces)[rows]).ravel()
                if want_mu:
                    dmu[col] += sign * np.asarray(r.dipole)
                if want_pol:
                    dpol[col] += sign * r.analysis.polarisability.tensor[0]
    scale = 0.5 / delta
    return SecondDerivatives(rows, moved, float(delta), hess * scale,
                             None if dmu is None else dmu * scale,
                             None if dpol is None else dpol * scale, float(ref.energy))


def normal_modes(hessian, masses, positions=None, *, born=None, born_derivatives=None,
                 remove_translation: bool = False, remove_rotation: bool = False,
                 triangle: str = "average") -> NormalModes:
    """Normal modes of a (square) Hessian (Hartree/Bohr^2, the moved atoms' coordinates) with
    ``masses`` in amu; ``born`` = :attr:`SecondDerivatives.dipole_derivatives` and
    ``born_derivatives`` = :attr:`SecondDerivatives.polarisability_derivatives` give the
    intensities; ``positions`` (Bohr) are needed to remove the rotations.  ``triangle``:
    "average" (default) symmetrises the Hessian, "modes" reads the triangle ``modes`` reads."""
    h = np.asarray(hessian, float)
    m = np.asarray(masses, float) * AMU_AU
    n = len(m)
    if h.shape != (3 * n, 3 * n):
        raise ValueError(f"normal_modes: the Hessian must be square, 3 x {n} coordinates "
                         f"(as modes; a partial Hessian of DFTB+ has more rows than columns)")
    if triangle not in ("average", "modes"):
        raise ValueError("normal_modes: triangle must be 'average' or 'modes'")
    sq = np.repeat(np.sqrt(m), 3)
    d = h / np.outer(sq, sq)
    project = remove_translation or remove_rotation
    if triangle == "average":
        d = 0.5 * (d + d.T)
    elif project:                               # modes: adjointLowerTriangle, then heev 'U'
        d = np.tril(d) + np.tril(d, -1).T
    else:                                       # modes: heev 'U' reads the upper triangle
        d = np.triu(d) + np.triu(d, 1).T
    if project:
        vecs = []
        if remove_translation:
            for i in range(3):
                v = np.zeros(3 * n)
                v[i::3] = 1.0
                vecs.append(v)
        if remove_rotation:
            if positions is None:
                raise ValueError("normal_modes: remove_rotation needs the positions")
            pos = np.asarray(positions, float)
            com = (m[:, None] * pos).sum(0) / m.sum()
            rel = pos - com
            inertia = np.zeros((3, 3))
            for a in range(n):
                inertia += m[a] * (np.dot(rel[a], rel[a]) * np.eye(3) - np.outer(rel[a], rel[a]))
            moments, axes = np.linalg.eigh(inertia)
            for i in range(3):
                if moments[i] < np.finfo(float).eps:
                    continue
                vecs.append(np.cross(axes[:, i], rel).ravel())
        proj = np.eye(3 * n)
        for v in vecs:
            v = v * sq
            norm2 = (v * v).sum()
            if norm2 > np.finfo(float).eps:
                v = v / np.sqrt(norm2)
                proj -= np.outer(v, v)
        d = proj.T @ d @ proj
    lam, vec = np.linalg.eigh(d)
    rigid = np.zeros(len(lam), bool)
    if project:                                 # modes in the projected-out space
        rigid = ((vec - proj @ vec) ** 2).sum(0) > 0.5
    for j in range(vec.shape[1]):               # modes' setEigvecGauge
        big = np.flatnonzero(np.abs(vec[:, j]) > 1e2 * np.finfo(float).eps)
        if big.size and vec[big[0], j] < 0.0:
            vec[:, j] = -vec[:, j]
    freq = np.sign(lam) * np.sqrt(np.abs(lam))
    disp = vec / sq[:, None]
    disp = disp / np.sqrt((disp * disp).sum(0))[None, :]
    active = (freq > np.finfo(float).eps) & ~rigid
    ir = raman = None
    if born is not None:
        z = np.asarray(born, float)             # (3n, 3): d mu_j / d x_col
        dmu = disp.T @ z                         # (modes, 3)
        ir = np.where(active, (dmu * dmu).sum(1), 0.0)
    if born_derivatives is not None:
        dz = np.asarray(born_derivatives, float).reshape(3 * n, 9)
        dq = disp.T @ dz
        raman = np.where(active, (dq * dq).sum(1), 0.0)
    groups = [[0]]
    for j in range(1, len(freq)):
        if abs(freq[j] - freq[j - 1]) * HARTREE_CM >= DEGENERATE_CM:
            groups.append([j])
        else:
            groups[-1].append(j)
    return NormalModes(freq, disp, vec, ir, raman, [np.array(g) for g in groups])
