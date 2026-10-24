"""External potentials (milestone M6c), as DFTB+'s ``ElectricField`` (``External``,
``PointCharges``) and ``AtomSitePotential`` (``Gross``, ``Net``).

All of them are fixed potentials V_A on the atoms.  The field, the point charges and the
gross site potentials act on the Mulliken charge deviations dq_A = q_A - q0_A
(electrons): they enter H like the SCC shift, 1/2 S_mu,nu (V_A + V_B), and the energy
as sum_A dq_A V_A.  A net site potential acts on the onsite ("net") population of its
atom only, N_A = sum_{mu on A} rho_mu,mu (the onsite overlap is the unit matrix): it is
added to the diagonal of H, energy (N_A - q0_A) V_A.

* field: V_A = r_A . E; periodic systems use the coordinates folded into the cell (a
  saw-tooth potential, as DFTB+); energy "Energy ext. field".
* point charges Q_j at R_j: V_A = -sum_j Q_j phi(r_A - R_j) (the minus sign: dq counts
  electrons), phi = 1/r or erf(r/sigma)/r with a Gaussian blur (within 10 sigma, as
  DFTB+); periodic systems sum phi by Ewald with the SCC kernel's alpha, G vectors and
  a neutralising background.  The energy is part of DFTB+'s "Energy SCC".
* site potentials: given per atom, energy "Energy ext. field".
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import erf, erfc

from .neighbors import cross_pairs

ERF_ARG_LIMIT = 10.0      # DFTB+ erfArgLimit_: no blur beyond 10 sigma
SAME_POSITION = 1e-8      # Bohr: a point charge this close to an atom is refused
_SQRT_PI = np.sqrt(np.pi)


@dataclass
class PointCharges:
    """External point charges: positions (M, 3) in Bohr, charges (M,) in e (positive for
    a positive charge) and Gaussian blur widths sigma (M,) in Bohr (0: point-like)."""
    positions: np.ndarray
    charges: np.ndarray
    blur: np.ndarray

    @classmethod
    def make(cls, spec) -> "PointCharges":
        """From a PointCharges, (positions, charges[, blur]) or a dict with those keys."""
        if isinstance(spec, PointCharges):
            pos, q, blur = spec.positions, spec.charges, spec.blur
        elif isinstance(spec, dict):
            unknown = set(spec) - {"positions", "charges", "blur"}
            if unknown:
                raise ValueError(f"point_charges: unknown keys {sorted(unknown)}")
            pos, q, blur = spec["positions"], spec["charges"], spec.get("blur", 0.0)
        else:
            if len(spec) not in (2, 3):
                raise ValueError("point_charges must be (positions, charges) or "
                                 "(positions, charges, blur)")
            pos, q = spec[0], spec[1]
            blur = spec[2] if len(spec) == 3 else 0.0
        pos = np.atleast_2d(np.asarray(pos, float))
        q = np.atleast_1d(np.asarray(q, float))
        if pos.ndim != 2 or pos.shape[1] != 3 or len(q) != len(pos):
            raise ValueError(f"point_charges: positions must be (M, 3) and charges (M,), got "
                             f"{pos.shape} and {q.shape}")
        blur = np.broadcast_to(np.asarray(blur, float), q.shape).copy()
        if np.any(blur < 0):
            raise ValueError("point_charges: the Gaussian blur width may not be negative "
                             "(as in DFTB+)")
        return cls(pos, q, blur)


def parse_field(field) -> np.ndarray:
    e = np.asarray(field, float).reshape(-1)
    if e.shape != (3,):
        raise ValueError(f"electric_field must be a vector (E_x, E_y, E_z), got {field!r}")
    return e


def parse_site_potentials(spec, natom: int):
    """{"gross": {atom: V}, "net": {atom: V}} (atoms 0-based, V in Hartree; a sequence of
    n_atom values also works) -> (gross, net) per atom, each None if absent."""
    if not isinstance(spec, dict):
        raise ValueError("site_potentials must be a dict {'gross': {...}, 'net': {...}}")
    unknown = set(spec) - {"gross", "net"}
    if unknown:
        raise ValueError(f"site_potentials: unknown kinds {sorted(unknown)} (use 'gross', 'net')")
    out = []
    for kind in ("gross", "net"):
        val = spec.get(kind)
        if val is None:
            out.append(None)
            continue
        v = np.zeros(natom)
        if isinstance(val, dict):
            for atom, pot in val.items():
                a = int(atom)
                if not 0 <= a < natom:
                    raise ValueError(f"site_potentials[{kind!r}]: atom {atom} outside "
                                     f"0..{natom - 1}")
                v[a] = float(pot)
        else:
            arr = np.asarray(val, float).reshape(-1)
            if arr.shape != (natom,):
                raise ValueError(f"site_potentials[{kind!r}]: need {natom} values, got "
                                 f"{arr.shape[0]}")
            v[:] = arr
        out.append(v)
    return tuple(out)


def fold_into_cell(pos: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Positions folded into the cell, fractional coordinates in [0, 1) (DFTB+
    ``foldCoordsToCell``; rows of ``cell`` are the lattice vectors)."""
    frac = pos @ np.linalg.inv(cell)
    frac = frac - np.floor(frac)
    lengths = np.linalg.norm(cell, axis=1)
    frac[np.abs(lengths * (1.0 - frac)) < np.finfo(float).eps] = 0.0
    return frac @ cell


def _blurred(d, sigma):
    """phi(d) = 1/d, or erf(d/sigma)/d within 10 sigma, and dphi/dd."""
    phi = 1.0 / d
    dphi = -phi * phi
    m = d < ERF_ARG_LIMIT * sigma
    if np.any(m):
        dm, rs = d[m], d[m] / sigma[m]
        e = erf(rs)
        phi[m] = e / dm
        dphi[m] = -(e - 2.0 / _SQRT_PI * rs * np.exp(-rs * rs)) / (dm * dm)
    return phi, dphi


@dataclass
class ExternalTerms:
    """Fixed external potentials of one geometry (see the module docstring)."""
    v_field: np.ndarray | None        # (n_atom,) r_A . E
    v_pc: np.ndarray | None           # (n_atom,) potential of the point charges
    v_gross: np.ndarray | None        # (n_atom,) gross site potentials
    v_net: np.ndarray | None          # (n_atom,) net site potentials
    field: np.ndarray | None          # (3,) the field acting (after solvent scaling)
    pos_field: np.ndarray | None      # positions of the field potential (folded if periodic)
    pc: PointCharges | None
    pos: np.ndarray                   # atom positions
    cell: np.ndarray | None           # periodic systems
    pbc: np.ndarray | None
    coulomb: object | None            # the SCC kernel (Ewald alpha and G vectors)

    @classmethod
    def build(cls, pos, cell, pbc, field=None, pc: PointCharges | None = None, gross=None,
              net=None, coulomb=None, pairs=None) -> "ExternalTerms":
        pos = np.asarray(pos, float)
        periodic = cell is not None and np.any(pbc)
        v_field = pos_field = None
        if field is not None:
            pos_field = fold_into_cell(pos, cell) if periodic else pos
            v_field = pos_field @ field
            if periodic:
                _sawtooth_warnings(pos, cell, field, pairs)
        terms = cls(v_field, None, gross, net, field, pos_field, pc, pos,
                    cell if periodic else None, pbc if periodic else None, coulomb)
        if pc is not None:
            terms.v_pc = terms._pc_potential()
        return terms

    @property
    def v_atom(self) -> np.ndarray | None:
        """The potential on the Mulliken charges (field + point charges + gross)."""
        parts = [v for v in (self.v_field, self.v_pc, self.v_gross) if v is not None]
        if not parts:
            return None
        v = parts[0]
        for p in parts[1:]:
            v = v + p
        return v

    # ---- energies ------------------------------------------------------------
    def energy_ext(self, dq_atom, dn_net=None) -> float:
        """DFTB+ "Energy ext. field": field and site potentials (dn_net = N_A - q0_A)."""
        e = 0.0
        for v in (self.v_field, self.v_gross):
            if v is not None:
                e += float(dq_atom @ v)
        if self.v_net is not None:
            e += float(dn_net @ self.v_net)
        return e

    def energy_pc(self, dq_atom) -> float:
        """The point charges' energy (part of DFTB+'s "Energy SCC")."""
        return 0.0 if self.v_pc is None else float(dq_atom @ self.v_pc)

    def atom_energies(self, dq_atom, dn_net=None) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`energy_ext` and :meth:`energy_pc` atom by atom (milestone M6f-4a): the
        field and gross terms dq_A V_A, the net term (N_A - q0_A) V_A, the point charges
        dq_A V^pc_A (not halved, as DFTB+'s addEnergyPerAtom)."""
        e_ext = np.zeros(len(self.pos))
        for v in (self.v_field, self.v_gross):
            if v is not None:
                e_ext = e_ext + dq_atom * v
        if self.v_net is not None:
            e_ext = e_ext + dn_net * self.v_net
        e_pc = np.zeros(len(self.pos)) if self.v_pc is None else dq_atom * self.v_pc
        return e_ext, e_pc

    # ---- derivatives -------------------------------------------------------------
    def gradient(self, dq_atom) -> np.ndarray:
        """Explicit dE/dr_A at fixed charges (n_atom, 3); the S-derivative part comes with
        the potentials through H."""
        grad = np.zeros_like(self.pos)
        if self.field is not None:
            grad += dq_atom[:, None] * self.field[None, :]
        if self.pc is not None:
            grad += self._pc_derivatives(dq_atom)[0]
        return grad

    def virial(self, dq_atom) -> np.ndarray:
        """dE/d(strain) of the field term, sum_A dq_A E_i r_A,j (folded coordinates);
        point charges have none (no stress, as DFTB+)."""
        if self.field is None:
            return np.zeros((3, 3))
        return np.outer(self.field, dq_atom @ self.pos_field)

    def charge_forces(self, dq_atom) -> np.ndarray | None:
        """Forces on the point charges, -dE/dR_j (M, 3)."""
        if self.pc is None:
            return None
        return self._pc_derivatives(dq_atom)[1]

    # ---- point charges -----------------------------------------------------------
    def _pc_pairs(self):
        """Atom - point charge pairs: (atom, charge, r_A - R_j [- T], distance)."""
        pc = self.pc
        if self.cell is None:
            vec = self.pos[:, None, :] - pc.positions[None, :, :]
            d = np.sqrt((vec * vec).sum(-1))
            if d.min(initial=np.inf) < SAME_POSITION:
                raise ValueError("a point charge sits on an atom")
            a, c = np.nonzero(np.ones_like(d, bool))
            return a, c, vec.reshape(-1, 3), d.reshape(-1)
        self._check_periodic_overlap()
        rc = max(self.coulomb.rcut_ewald, ERF_ARG_LIMIT * float(pc.blur.max(initial=0.0)))
        a, c, vec, d = cross_pairs(self.pos, pc.positions, self.cell, self.pbc, rc)
        return a, c, -vec, d

    def _check_periodic_overlap(self):
        frac = self.pos @ np.linalg.inv(self.cell)
        fq = self.pc.positions @ np.linalg.inv(self.cell)
        for f in frac:                                 # minimum image, atom by atom
            df = fq - f
            df -= np.round(df)
            if np.sqrt(((df @ self.cell) ** 2).sum(1)).min(initial=np.inf) < SAME_POSITION:
                raise ValueError("a point charge sits on an atom (or one of its images)")

    def _pc_kernel(self, d, sigma):
        """phi and dphi/dd per pair (real-space part for periodic systems)."""
        if self.cell is None:
            return _blurred(d, sigma)
        a = self.coulomb.alpha
        phi = erfc(a * d) / d
        dphi = -phi / d - 2.0 * a / _SQRT_PI * np.exp(-(a * d) ** 2) / d
        m = d < ERF_ARG_LIMIT * sigma                 # blur: [erf(d/s) - erf(a d)] / d
        if np.any(m):
            dm, rs = d[m], d[m] / sigma[m]
            ec = erfc(rs)
            phi[m] -= ec / dm
            dphi[m] += ec / (dm * dm) + 2.0 / (_SQRT_PI * sigma[m]) * np.exp(-rs * rs) / dm
        return phi, dphi

    def _pc_potential(self) -> np.ndarray:
        pc, n = self.pc, len(self.pos)
        a, c, _, d = self._pc_pairs()
        phi, _ = self._pc_kernel(d, pc.blur[c])
        v = np.zeros(n)
        np.add.at(v, a, -pc.charges[c] * phi)
        if self.cell is not None:
            co = self.coulomb
            qn = -pc.charges
            cq, sq = np.cos(pc.positions @ co.gvec.T), np.sin(pc.positions @ co.gvec.T)
            ca, sa = np.cos(self.pos @ co.gvec.T), np.sin(self.pos @ co.gvec.T)
            v += (ca * (co.gfac * (qn @ cq))[None, :]).sum(1) + (sa * (co.gfac * (qn @ sq))[None, :]).sum(1)
            v -= np.pi / (co.vol * co.alpha ** 2) * qn.sum()
        return v

    def _pc_derivatives(self, dq_atom):
        """(dE/dr_A, forces on the charges) of E = sum_A dq_A V_A at fixed dq."""
        pc, n = self.pc, len(self.pos)
        a, c, rvec, d = self._pc_pairs()
        _, dphi = self._pc_kernel(d, pc.blur[c])
        # dE/dr_A per pair: dq_A (-Q_j) phi'(d) (r_A - R_j - T) / d
        g = (dq_atom[a] * -pc.charges[c] * dphi / d)[:, None] * rvec
        grad = np.zeros((n, 3))
        fq = np.zeros((len(pc.charges), 3))
        np.add.at(grad, a, g)
        np.add.at(fq, c, g)                 # -dE/dR_j = +g (d/dR_j = -d/dr_A)
        if self.cell is not None:
            co = self.coulomb
            qn = -pc.charges
            ph_a, ph_q = self.pos @ co.gvec.T, pc.positions @ co.gvec.T
            ca, sa, cq, sq = np.cos(ph_a), np.sin(ph_a), np.cos(ph_q), np.sin(ph_q)
            # d/dr_A cos(G (r_A - R_j)) = -sin(G (r_A - R_j)) G
            cqs, sqs = co.gfac * (qn @ cq), co.gfac * (qn @ sq)
            grad += (dq_atom[:, None] * (-sa * cqs[None, :] + ca * sqs[None, :])) @ co.gvec
            cad, sad = co.gfac * (dq_atom @ ca), co.gfac * (dq_atom @ sa)
            fq += (qn[:, None] * (-cq * sad[None, :] + sq * cad[None, :])) @ co.gvec
        return grad, fq


def _sawtooth_warnings(pos, cell, field, pairs):
    """DFTB+'s warnings for a field in a periodic system: the saw-tooth needs vacuum, and
    SK neighbour pairs should not cross its discontinuity."""
    warnings.warn("saw-tooth potential used for a periodic geometry: make sure there is a "
                  "vacuum region along the field (as DFTB+ warns)", RuntimeWarning, stacklevel=4)
    if pairs is None or len(pairs.i) == 0 or not np.any(field):
        return
    n_img = np.floor(pos @ np.linalg.inv(cell)).astype(int)    # folding shifts of the atoms
    shift = pairs.shift + n_img[pairs.j] - n_img[pairs.i]     # pair shifts between folded atoms
    t = shift @ cell
    if np.any(np.abs(t @ field) > np.finfo(float).eps * np.linalg.norm(field)):
        warnings.warn("interactions between atoms cross the saw-tooth discontinuity in the "
                      "electric field (as DFTB+ warns)", RuntimeWarning, stacklevel=4)
