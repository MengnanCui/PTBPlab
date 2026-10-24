"""Analysis outputs of DFTB+'s ``Analysis`` block (milestone M6f-4a).

* **Atom-resolved energies** (``AtomResolvedEnergies``): every term of the total energy split
  over the atoms as DFTB+ does (``dftb/getenergies.F90``), from the last output populations:
  H0 by the Mulliken split of sum_k w_k Re[rho_k H0_k] (an atom's diagonal block to it, an
  A-B block half to each), SCC 1/2 dq_A (gamma dq)_A (the Ewald constants included) plus
  dq_A V^pc_A (not halved) and the dipole correction 1/2 dq_A V_A, the full third order
  1/3 dq_A dE3/dq_A, spin 1/2 m (W m) per shell, spin-orbit, DFTB+U and onsite terms on their
  atom, field and site potentials dq_A V_A (net: (N_A - q0_A) V_A), GB solvation
  1/2 q'_A (B q')_A + surface terms + shift/N, the repulsive and halogen pairs half to each
  atom, dispersion by :meth:`pydftb.dispersion.Dispersion.atom_energies`.  "electronic" is the
  sum of the first nine, "total" adds repulsive, dispersion and halogen; sum(total) = energy
  (DFTB+'s "Total energy", not the Mermin free energy).  DFTB+ spreads the energy of its D3
  library evenly (E/N); here D3 is resolved by atom pairs like D4 (DFTB+'s E/N with the
  test-only class attribute ``DFTB._atom_energy_compat``).  DFTB+ books the on-site third
  order in E_SCC, pydftb in ``e_3rd``: the components follow pydftb's result fields.
* **Net populations** (``WriteNetCharges``): q^net_A = sum_{mu on A} rho_mu,mu(T = 0) and the
  hybridisation ratio 1 - q^net_A / q0_A.
* **CM5 charges** (``CM5``): q0 - q plus the CM5 pair correction (pairs of one type skipped,
  periodic images within the cut-off included).
* **Electrostatic potential** (``ElectrostaticPotential``): the physical potential (Hartree/e)
  of the output atomic charges q0 - q as point charges (softened 1/r for molecules, Ewald for
  periodic systems) plus the dipole correction's, and an external part (point charges, -E.r)
  when there is a field or point charges; ESP.dat holds both in V.
* **Eigenvectors** (``WriteEigenvectors``): of the last Hamiltonian (built from the last input
  populations, as DFTB+), with their Mulliken weights for eigenvec.out/.bin.

The files are written by :mod:`pydftb.dftbplus_io`.  NumPy engine only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from ase.data import atomic_numbers
from scipy.special import erf, erfc

from . import spinorbit as so
from .backend import ops
from .external import ERF_ARG_LIMIT
from .neighbors import cross_pairs
from .solvation import CM5, GeneralizedBorn, cm5_charges
from .system import element_of_label

TOL_SAME_DIST = 1e-5           # Bohr, DFTB+ tolSameDist: coinciding point and charge
ESP_SOFTENING = 1e-6           # Bohr, DFTB+'s default Softening
ESP_CHUNK = 4096               # points per batch of the potential
ENERGY_TERMS = ("h0", "scc", "third_order", "spin", "spin_orbit", "plus_u", "onsite",
                "external", "solvation")
SHELL_NAMES = "spdf"
ORBITAL_NAMES = {0: ("",), 1: ("y", "z", "x"), 2: ("xy", "yz", "z2", "xz", "x2-y2"),
                 3: ("y(3x2-y2)", "xyz", "yz2", "z3", "xz2", "z(x2-y2)", "x(x2-3y2)")}
_ESP_KEYS = {"points", "grid", "fractional", "softening"}
_GRID_KEYS = {"spacing", "origin", "points", "directions"}
_CM5_KEYS = {"alpha", "radii", "cutoff"}


@dataclass(frozen=True)
class Analysis:
    """What to analyse.  ``cm5``: True or {"alpha" (1/Bohr), "radii" {label: Bohr},
    "cutoff" (Bohr)}; ``esp``: {"points": (M, 3)} or {"grid": {"spacing" (3,), "origin" (3,),
    "points" (3 ints), "directions" (3, 3) rows, optional}}, with "fractional": True for
    fractional coordinates of a periodic system and "softening" (Bohr, default 1e-6);
    ``eigenvectors``: keep the eigenvectors and their Mulliken weights (for eigenvec.out)."""
    atom_energies: bool = False
    net_populations: bool = False
    cm5: bool | dict | CM5 = False
    esp: dict | None = None
    eigenvectors: bool = False

    def __post_init__(self):
        if not (isinstance(self.cm5, (bool, dict, CM5))):
            raise TypeError("analysis cm5 must be True/False, a dict or a solvation.CM5")
        if isinstance(self.cm5, dict) and set(self.cm5) - _CM5_KEYS:
            raise ValueError(f"analysis cm5: unknown keys {sorted(set(self.cm5) - _CM5_KEYS)} "
                             f"(use {sorted(_CM5_KEYS)})")
        if self.esp is not None:
            _check_esp(self.esp)

    @classmethod
    def make(cls, spec) -> "Analysis | None":
        """None/False, True (atom energies, net populations, CM5), a dict of the arguments
        or an Analysis."""
        if spec is None or spec is False:
            return None
        if spec is True:
            return cls(atom_energies=True, net_populations=True, cm5=True)
        if isinstance(spec, Analysis):
            return spec
        if isinstance(spec, dict):
            known = {"atom_energies", "net_populations", "cm5", "esp", "eigenvectors"}
            unknown = set(spec) - known
            if unknown:
                raise ValueError(f"analysis: unknown keys {sorted(unknown)} (use {sorted(known)})")
            return cls(**spec)
        raise TypeError(f"analysis must be None, True, a dict or an Analysis, "
                        f"not {type(spec).__name__}")

    @property
    def cm5_model(self) -> CM5 | None:
        if self.cm5 is False:
            return None
        if self.cm5 is True:
            return CM5()
        if isinstance(self.cm5, CM5):
            return replace(self.cm5, double_count=False)
        return CM5(**self.cm5)


@dataclass
class ESPResult:
    """The electrostatic potential (Hartree/e, physical sign) at ``points`` (Bohr):
    ``internal`` from the atoms' charges (and the dipole correction), ``external`` from point
    charges and the field (None without them).  A grid has ``shape`` (n1, n2, n3), the third
    index running fastest, ``origin`` and the step vectors ``axes`` (rows, Bohr)."""
    points: np.ndarray
    internal: np.ndarray
    external: np.ndarray | None = None
    shape: tuple | None = None
    origin: np.ndarray | None = None
    axes: np.ndarray | None = None


@dataclass
class AnalysisResult:
    """``DFTBResult.analysis``: what :class:`Analysis` asked for (atomic units)."""
    atom_energies: dict | None = None        # term -> (n_atom,), see the module docstring
    net_populations: np.ndarray | None = None
    hybridisation: np.ndarray | None = None
    cm5_charges: np.ndarray | None = None
    esp: ESPResult | None = None
    eigenvectors: list | None = None         # per (spin, k) block, as the eigenvalues
    mulliken_weights: list | None = None     # real c (S c) / Re[c* (S c)]; Pauli (4, 2N)
    eigenvector_kind: str | None = None      # "real", "complex" or "pauli"
    orbitals: list | None = None             # (atom, label, shell, name) of every orbital
    n_kpoints: int | None = None


def _check_esp(spec):
    if not isinstance(spec, dict):
        raise TypeError("analysis esp must be a dict with 'points' or 'grid'")
    unknown = set(spec) - _ESP_KEYS
    if unknown:
        raise ValueError(f"analysis esp: unknown keys {sorted(unknown)} (use {sorted(_ESP_KEYS)})")
    if ("points" in spec) == ("grid" in spec):
        raise ValueError("analysis esp: give either 'points' or 'grid' (as DFTB+)")
    if float(spec.get("softening", ESP_SOFTENING)) < 0.0:
        raise ValueError("analysis esp: softening must be >= 0")
    if "points" in spec:
        pts = np.asarray(spec["points"], float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            raise ValueError("analysis esp: points must be an (M, 3) array")
        return
    grid = spec["grid"]
    if not isinstance(grid, dict) or set(grid) - _GRID_KEYS or \
            not {"spacing", "origin", "points"} <= set(grid):
        raise ValueError(f"analysis esp grid: a dict with 'spacing', 'origin', 'points' and "
                         f"optionally 'directions', not {grid!r}")
    n = np.asarray(grid["points"])
    if n.shape != (3,) or np.any(n < 1) or not np.all(np.equal(np.mod(n, 1), 0)):
        raise ValueError("analysis esp grid: points must be three integers >= 1 (as DFTB+)")
    spacing = np.asarray(grid["spacing"], float)
    if spacing.shape != (3,) or np.asarray(grid["origin"], float).shape != (3,):
        raise ValueError("analysis esp grid: spacing and origin need three values each")
    if np.any((np.abs(spacing) < np.finfo(float).eps) & (n > 1)):
        raise ValueError("analysis esp grid: grid spacings must be non-zero (as DFTB+)")
    if "directions" in grid:
        if spec.get("fractional", False):
            raise ValueError("analysis esp grid: directions are not read for fractional grids "
                             "(as DFTB+)")
        d = np.asarray(grid["directions"], float)
        if d.shape != (3, 3) or abs(np.linalg.det(d)) < np.finfo(float).eps:
            raise ValueError("analysis esp grid: directions must be three independent vectors "
                             "(as DFTB+: 'Dependent axis directions')")


def esp_points(spec: dict, cell, periodic: bool):
    """The points (M, 3) of an ``esp`` specification, and for a grid (shape, origin, axes)."""
    frac = bool(spec.get("fractional", False))
    if frac and not periodic:
        raise ValueError("analysis esp: fractional coordinates need a periodic system (as DFTB+)")
    cell = None if cell is None else np.asarray(cell, float)
    if "points" in spec:
        pts = np.asarray(spec["points"], float).reshape(-1, 3)
        return (pts @ cell if frac else pts), None
    grid = spec["grid"]
    n = tuple(int(x) for x in grid["points"])
    spacing = np.asarray(grid["spacing"], float)
    origin = np.asarray(grid["origin"], float)
    idx = np.stack(np.meshgrid(*[np.arange(m) for m in n], indexing="ij"), -1).reshape(-1, 3)
    coords = idx * spacing + origin                 # the last index runs fastest (DFTB+)
    if frac:
        basis = cell
    else:
        d = np.asarray(grid.get("directions", np.eye(3)), float)
        basis = d / np.linalg.norm(d, axis=1)[:, None]
    return coords @ basis, (n, origin @ basis, spacing[:, None] * basis)


def electrostatic_potential(points, charges, positions, coulomb=None,
                            softening: float = ESP_SOFTENING, blur=None) -> np.ndarray:
    """The potential (Hartree/e) at ``points`` (Bohr) of point charges ``charges`` (e) at
    ``positions``, as DFTB+'s getPotential: molecules sum q / sqrt(d^2 + eps^2) (times
    erf(d/sigma) within 10 sigma of a blurred charge); periodic systems (``coulomb`` the SCC
    kernel, whose Ewald alpha and G vectors are used) sum the real-space terms
    erfc(alpha d) / sqrt(d^2 + eps^2) of the charges and their images (those closer than
    1e-5 Bohr left out), the reciprocal terms and -pi/(V alpha^2) sum q, and leave out the
    self term 2 alpha/sqrt(pi) q of a charge (or image) on the point.  DFTB+ does the last
    only for the central cell's charges."""
    pts = np.asarray(points, float).reshape(-1, 3)
    src = np.asarray(positions, float).reshape(-1, 3)
    q = np.asarray(charges, float)
    eps2 = float(softening) ** 2
    sigma = None if blur is None else np.broadcast_to(np.asarray(blur, float), q.shape)
    out = np.empty(len(pts))
    periodic = coulomb is not None and coulomb.periodic
    if periodic:
        a, vol = coulomb.alpha, coulomb.vol
        cell, pbc = coulomb.cell, coulomb.pbc
        inv = np.linalg.inv(cell)
        rc = coulomb.rcut_ewald
        if sigma is not None and np.any(sigma > 0.0):
            rc = max(rc, ERF_ARG_LIMIT * float(sigma.max()))
        ph = src @ coulomb.gvec.T
        sq_c = coulomb.gfac * (q @ np.cos(ph))
        sq_s = coulomb.gfac * (q @ np.sin(ph))
        background = np.pi / (vol * a * a) * q.sum()
    for lo in range(0, len(pts), ESP_CHUNK):
        p = pts[lo:lo + ESP_CHUNK]
        if not periodic:
            d2 = ((p[:, None, :] - src[None, :, :]) ** 2).sum(-1)
            kern = 1.0 / np.sqrt(d2 + eps2)
            if sigma is not None:
                s = np.broadcast_to(sigma, d2.shape)
                m = (s > 0.0) & (d2 < (ERF_ARG_LIMIT * s) ** 2)
                kern[m] *= erf(np.sqrt(d2[m]) / s[m])
            out[lo:lo + len(p)] = kern @ q
            continue
        ia, ib, _, d = cross_pairs(p, src, cell, pbc, rc)
        keep = d >= TOL_SAME_DIST
        ia, ib, d = ia[keep], ib[keep], d[keep]
        den = np.sqrt(d * d + eps2)
        kern = erfc(a * d) / den
        if sigma is not None:
            s = sigma[ib]
            m = (s > 0.0) & (d < ERF_ARG_LIMIT * s)
            kern[m] = (erf(d[m] / s[m]) - erf(a * d[m])) / den[m]
        v = np.zeros(len(p))
        np.add.at(v, ia, q[ib] * kern)
        php = p @ coulomb.gvec.T
        v += np.cos(php) @ sq_c + np.sin(php) @ sq_s - background
        df = (p[:, None, :] - src[None, :, :]) @ inv
        df -= np.round(df)
        on = np.sqrt(((df @ cell) ** 2).sum(-1)) < TOL_SAME_DIST
        if on.any():
            v -= 2.0 * a / np.sqrt(np.pi) * (on * q[None, :]).sum(1)
        out[lo:lo + len(p)] = v
    return out


def orbital_labels(basis) -> list:
    """(atom, type label, shell index on the atom, name) of every orbital with DFTB+'s names
    (eigenvec.out): the shell name s, p, d, f (a second shell of one l gets a counter, "s2"),
    for l > 0 "_" and the orbital (p: y z x; d: xy yz z2 xz x2-y2; f: ...)."""
    out = []
    for a, lab in enumerate(basis.species):
        seen = []
        for ish, l in enumerate(basis.shells[a]):
            name = SHELL_NAMES[l]
            count = sum(1 for prev in seen if prev[0] == name)
            seen.append(name)
            shell = name + (str(count + 1) if count else "")
            for orb in ORBITAL_NAMES[l]:
                out.append((a, lab, ish, shell + ("_" + orb if l > 0 else "")))
    return out


# ----------------------------------------------------------------------------------------
def evaluate(calc, analysis: Analysis, st, sc, res, xl: bool = False) -> AnalysisResult:
    """The analysis of one calculation (``calc`` a :class:`pydftb.DFTB`, ``st``/``sc`` its
    setup and SCC state, ``res`` its result)."""
    out = AnalysisResult()
    natom = len(st.species)
    periodic = st.cell is not None and bool(np.any(st.pbc))
    cell = st.cell if periodic else None
    pbc = st.pbc if periodic else None
    dq, _ = calc._charge_and_magnetisation(st, sc.q_out)
    dq_atom = ops.segment_sum(dq, st.site_atom, natom)
    if analysis.atom_energies:
        if xl:
            raise ValueError("atom-resolved energies are not defined for the XL-BOMD shadow "
                             "energy")
        out.atom_energies = atom_energies(calc, st, sc, dq_atom,
                                          dftbplus=calc._atom_energy_compat)
    if analysis.net_populations:
        q_net = calc._net_populations(st, sc.rho)
        q0_atom = ops.segment_sum(st.q0, st.site_atom, natom)
        out.net_populations = q_net
        with np.errstate(divide="ignore", invalid="ignore"):
            out.hybridisation = 1.0 - q_net / q0_atom
    model = analysis.cm5_model
    if model is not None:
        numbers = [atomic_numbers[element_of_label(lab)] for lab in st.species]
        out.cm5_charges = -dq_atom + cm5_charges(st.pos, list(st.species), numbers, model,
                                                 cell, pbc)
    if analysis.esp is not None:
        out.esp = electrostatic_potential_of(calc, analysis.esp, st, dq_atom, cell, periodic)
    if analysis.eigenvectors:
        _eigenvectors(st, sc, out, periodic)
    return out


def atom_energies(calc, st, sc, dq_atom, dftbplus: bool = False) -> dict:
    """Every energy term per atom (module docstring); sums equal the result's fields."""
    natom = len(st.species)
    b = st.basis
    nk = len(st.kw)
    zero = np.zeros(natom)
    out = {k: zero.copy() for k in ENERGY_TERMS}
    comp = None
    if st.pauli is not None:
        comp = so.components(sc.rho)
        e_orb = np.einsum("k,kij,kji->i", st.kw, comp[0], st.h0).real
    else:
        e_orb = sum(np.einsum("k,kij,kji->i", st.kw, sc.rho[s:s + nk], st.h0).real
                    for s in range(0, len(sc.rho), nk))
    out["h0"] = ops.segment_sum(e_orb, b.orb_atom, natom)
    dq, m = calc._charge_and_magnetisation(st, sc.q_out)
    if calc.scc:
        _, v = st.coulomb.energy_shift(dq)
        out["scc"] = ops.segment_sum(0.5 * dq * v, st.site_atom, natom)
    if st.third is not None:
        _, v3 = st.third.energy_shift(dq)
        out["third_order"] = ops.segment_sum(dq * v3 / 3.0, st.site_atom, natom)
    if m is not None:
        if st.pauli is not None:
            if st.pauli.w is not None:
                e_sh = 0.5 * np.einsum("as,st,at->s", m, st.pauli.w, m)
                out["spin"] = ops.segment_sum(e_sh, b.shell_atom, natom)
        else:
            out["spin"] = ops.segment_sum(0.5 * m * (st.spin.w @ m), b.shell_atom, natom)
    if st.pauli is not None and st.pauli.hso is not None:
        if st.pauli.dual:
            blocks = so.block_populations(comp, st.smat, st.kw, b)
            out["spin_orbit"] = so.dual_soc_atom_energies(blocks, st.pauli.lam, b)
        else:
            x = np.einsum("k,kij,kji->i", st.kw, sc.rho,
                          np.broadcast_to(st.pauli.hso, sc.rho.shape)).real
            n = b.norb
            out["spin_orbit"] = ops.segment_sum(x[:n] + x[n:], b.orb_atom, natom)
    if st.block is not None:
        blk = st.block
        _, blocks, skew = blk.expand(sc.q_out)
        out["plus_u"], out["onsite"] = blk.atom_energies(blk.components(blocks),
                                                         blk.components(skew))
    if st.ext is not None:
        dn = None
        if st.ext.v_net is not None:
            q0_atom = ops.segment_sum(st.q0, st.site_atom, natom)
            dn = calc._net_populations(st, sc.rho) - q0_atom
        out["external"], e_pc = st.ext.atom_energies(dq_atom, dn)
        out["scc"] = out["scc"] + e_pc
    if st.solv is not None:
        out["solvation"] = st.solv.atom_energies(dq_atom)
    if st.dipole_corr is not None:
        out["scc"] = out["scc"] + st.dipole_corr.atom_energies(dq_atom)
    electronic = zero.copy()
    for k in ENERGY_TERMS:
        electronic = electronic + out[k]
    out["electronic"] = electronic
    rep = zero.copy()
    nl, terms = calc._repulsive_pairs(st)
    for msk, f in terms:
        e, _ = f(nl[4][msk])
        np.add.at(rep, nl[0][msk], 0.5 * e)
    out["repulsive"] = rep
    out["dispersion"] = zero.copy()
    if calc.dispersion is not None:
        periodic = st.cell is not None and bool(np.any(st.pbc))
        out["dispersion"] = calc.dispersion.atom_energies(
            st.species, st.pos, st.cell if periodic else None, st.pbc if periodic else None,
            charge=calc.charge, dftbplus=dftbplus)
    out["halogen"] = zero.copy() if st.halogenx is None else st.halogenx.atom_energies.copy()
    out["total"] = electronic + rep + out["dispersion"] + out["halogen"]
    return out


def electrostatic_potential_of(calc, spec: dict, st, dq_atom, cell, periodic) -> ESPResult:
    """The ``esp`` analysis of a calculation (as DFTB+'s TElStatPotentials%evaluate)."""
    if not calc.scc:
        raise ValueError("the electrostatic potential needs SCC (as DFTB+)")
    if isinstance(calc.solvation, GeneralizedBorn):
        raise ValueError("the electrostatic potential is not available with a solvent that "
                         "changes the electrostatics (Generalized Born; as DFTB+)")
    pts, grid = esp_points(spec, cell, periodic)
    eps = float(spec.get("softening", ESP_SOFTENING))
    internal = electrostatic_potential(pts, -dq_atom, st.pos, st.coulomb if periodic else None,
                                       eps)
    if st.dipole_corr is not None:
        internal = internal - st.dipole_corr.point_potential(pts, dq_atom)
    external = None
    ext = st.ext
    if ext is not None and (ext.pc is not None or ext.field is not None):
        external = np.zeros(len(pts))
        if ext.pc is not None:
            external += electrostatic_potential(pts, ext.pc.charges, ext.pc.positions,
                                                st.coulomb if periodic else None, eps,
                                                blur=ext.pc.blur)
        if ext.field is not None:
            external -= pts @ ext.field
    if grid is None:
        return ESPResult(pts, internal, external)
    shape, origin, axes = grid
    return ESPResult(pts, internal, external, shape, origin, axes)


def _eigenvectors(st, sc, out: AnalysisResult, periodic: bool):
    """Eigenvectors of the last Hamiltonian and their Mulliken weights."""
    h = np.asarray(sc.h)
    nk = len(st.kw)
    pauli = st.pauli is not None
    s_all = np.asarray(st.pauli.s2 if pauli else st.smat_precise)
    vecs, weights = [], []
    for i, hk in enumerate(h):
        sk = s_all[i % nk]
        _, c = ops.eigh(hk, sk)
        c = np.asarray(c)
        sc_ = sk @ c
        if pauli:
            n = sk.shape[0] // 2
            cu, cd, su, sd = c[:n], c[n:], sc_[:n], sc_[n:]
            w = np.stack([np.real(cu.conj() * su + cd.conj() * sd),
                          np.real(cd.conj() * su + cu.conj() * sd),
                          np.imag(cu.conj() * sd - cd.conj() * su),
                          np.real(cu.conj() * su - cd.conj() * sd)])
        else:
            w = np.real(c.conj() * sc_)
        vecs.append(c)
        weights.append(w)
    out.eigenvectors, out.mulliken_weights = vecs, weights
    out.eigenvector_kind = ("pauli" if pauli else
                            "real" if not periodic or st.gamma_only else "complex")
    out.orbitals = orbital_labels(st.basis)
    out.n_kpoints = nk
