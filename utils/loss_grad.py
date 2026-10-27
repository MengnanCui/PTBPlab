"""Gradient of PTBP's loss with respect to the confinement radii (milestone M4c).

The loss of a BO step is mu*(r0) = min_sigma mu(r0, sigma).  At the inner minimum
sigma* (design sections 3-4):

    dmu*/dr0 = sum_i dmu/dE_i Edot_i + sum_ic dmu/dF_ic Fdot_ic + sum_Z dmu/dE0_Z E0dot_Z
               + mu dlnP/dr0                                  (smooth minimum, or a bound)

with Edot = <dE_band/dtheta, t>, t = dtheta/dr0 the hotcent tangent
(:mod:`utils.hotcent_tangent`) and the band derivatives from pydftb.autodiff
(:mod:`utils.dftb_stage`, passes 1 and 2).  When sigma* sits on the kink of a term
|r_k|, sigma* follows r_k = 0 and the derivative gets the kink correction.

Every loss is P(r0) sum_j w_j |r_j(r0, sigma)| with constant weights w_j:

* energygeometry: the relative energies per atom of each EOS group (weights
  (1 - c) b_g / (L B_g)) and the equilibrium volume ratios v0 / v0_DFT - 1
  (c b_g / L), c = 0.5;
* dataset / reaction: the formation energies per atom (1 - c) and the force
  components (c / N_F), c = 0.1.

:func:`mu` recomputes the loss of :class:`utils.loss.Loss` bit for bit from the step's
files (tests); :func:`terms` gives the residuals with their derivatives with respect to
sigma and the energies; :func:`refine_sigma` finds sigma* near PTBP's sigma_hat;
:class:`GradRequest` is the hook :class:`utils.loss.Loss` calls when a gradient
optimiser (:mod:`utils.gradient_opt`) asks for one.
"""
from __future__ import annotations

import contextlib
import io
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from math import floor

import numpy as np

EG_C = 0.5                    # energygeometry: mu = (c mu_V + (1 - c) mu_E) P
DS_C = 0.1                    # dataset/reaction: mu = (mu_E (1 - c) + mu_F c) P
SIGMA_BOUNDS = (0.1, 0.8)     # Loss.Repulsion_optimization
FAILED_EOS = 1000             # Loss.eos_atoms on a failed fit
ZERO_RESIDUAL = 1e-7          # |r| below which a residual with ~zero slope is identically 0
ZERO_SLOPE = 1e-9
KINK_DISTANCE = 1e-7          # |r / (dr/dsigma)| below which sigma* is on the kink of r


class GradientUnavailable(RuntimeError):
    """The loss has no derivative at this point (failure value, EOS failure, SCC ...)."""


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# the step's data
@dataclass
class StepData:
    """What mu depends on at fixed r0, besides sigma: the files of a BO step as
    :class:`utils.loss.Loss` reads them, and the loss settings.

    With parameters per type (M5b, ``loss.typed``) ``x`` holds the radii of every parameter set
    and sigma is a vector, one sigma_rep per set (:class:`utils.typemodel.TypeModel`)."""
    loss: object                         # utils.loss.Loss (helpers, settings, E0 cache)
    mode: str                            # energygeometry | dataset | reaction
    dft: list                            # dft_final.xyz
    band: list                           # dftb_band.xyz
    bolt11: list                         # Boltzmann factor of every structure kept
    bolt_f: np.ndarray                   # factor of every EOS group (energygeometry)
    eos: int
    r0_w: float
    r0_d: float
    groups: list = field(default_factory=list)    # energygeometry: (numbers, exclusive) per group
    numbers: list = field(default_factory=list)   # dataset: atomic numbers (physrep order)
    model: object = None                 # utils.typemodel.TypeModel (parameters per type)
    x: list | None = None                # the radii of every parameter set (model's order)

    @classmethod
    def from_files(cls, loss, r0_w: float | None = None, r0_d: float | None = None,
                   directory: str = ".", x=None):
        from utils.loss import _read_step_file
        mode = loss.args_input.optimization_option.lower()
        band = _read_step_file(os.path.join(directory, "dftb_band.xyz"))
        dft = _read_step_file(os.path.join(directory, "dft_final.xyz"))
        with open(os.path.join(directory, "search_index.txt")) as fh:
            search_index = fh.read().strip().split(",")
        bolt11 = [loss.bolt_f11[int(i)] for i in search_index]
        eos = int(loss.args_input.eos_points)
        bolt_f = loss.bolt_f11[floor(eos / 2)::eos]
        if getattr(loss, "typed", False):
            x = [float(v) for v in (x if x is not None else (r0_w, r0_d))]
            return cls(loss, mode, dft, band, bolt11, bolt_f, eos, float("nan"), float("nan"),
                       model=loss.types, x=x)
        step = cls(loss, mode, dft, band, bolt11, bolt_f, eos, float(r0_w), float(r0_d))
        if mode == "energygeometry":
            ng = int(len(dft) / eos)
            step.groups = [loss.get_elements_from_dataset(dft[n * eos:(n + 1) * eos],
                                                          loss.opt_elem)[:2] for n in range(ng)]
        else:
            step.numbers = loss.get_elements_from_dataset(dft, loss.opt_elem)[0]
        return step

    @property
    def multi_element(self) -> bool:
        return self.loss.args_input.multi_element is not None

    # parameters per type (M5b) ------------------------------------------------------
    def _typed_physrep(self, sigma, structures):
        """(physrep of the types of ``structures`` at the sigma vector, {set index: [Z ...]}):
        a physrep of atomic numbers without switching (the elements are the types), a
        switched_physrep of the type labels with it."""
        from ase.data import atomic_numbers

        from tools.physrep import switched_physrep
        from utils.loss import _physrep
        model = self.model
        sig = model.sigma(sigma)
        elements = sorted({s for a in structures for s in a.get_chemical_symbols()},
                          key=lambda s: atomic_numbers[s])
        if not model.switches:
            numbers = [atomic_numbers[s] for s in elements]
            pr = _physrep(numbers, [sig[s] for s in elements], [0.0] * len(numbers))
            varying = {}
            for s, z in zip(elements, numbers):
                k = model.set_index(s)
                if k is not None:
                    varying.setdefault(k, []).append(z)
            return pr, varying
        labels = [lab for s in elements for lab in model.labels_of_element(s)]
        number_of = model.number_of
        pr = _physrep(labels, [sig[lab] for lab in labels], [0.0] * len(labels),
                      numbers={lab: number_of[lab] for lab in labels})
        return switched_physrep(pr, model.switches), None

    def physrep_all(self, sigma):
        """The physrep of the types of every structure with results (Loss writes the SKF files
        with it): as before M5b, the elements of dft_final.xyz, so a type whose tables hotcent
        could not make (its structures failed) gets no repulsive written."""
        if self.model is None:
            raise ValueError("physrep_all is for parameters per type")
        return self._typed_physrep(sigma, self.dft)[0]

    # physrep of this step at sigma, as the loss builds it ---------------------------
    def physrep(self, sigma: float, group: int | None = None):
        """(physrep, element numbers whose covalent radius scales with sigma)."""
        from utils.loss import _physrep
        from utils.parameters_set import get_PTBP
        from ase.data import chemical_symbols
        if self.model is not None:
            if group is None or self.mode != "energygeometry":
                return self._typed_physrep(sigma, self.dft)
            return self._typed_physrep(sigma, self.dft[group * self.eos:(group + 1) * self.eos])
        sigma = float(sigma)
        if self.mode == "energygeometry":
            numbers, exclusive = self.groups[group]
            if self.multi_element:
                scale = [get_PTBP()[chemical_symbols[n]]['sigma_rep'] for n in exclusive] + [sigma]
                varying = [z for z in numbers if z not in exclusive]
            else:                  # every element of the group: the shared sigma (M5b-3)
                scale = [sigma] * len(numbers)
                varying = list(numbers)
            return _physrep(numbers, scale, [0.0] * len(numbers)), varying
        scale = [sigma] * len(self.numbers)
        return _physrep(self.numbers, scale, [0.0] * len(self.numbers)), list(self.numbers)

    def penalty(self) -> float:
        if self.model is not None:
            return self.model.penalty(self.x)
        from utils.loss import r0_penalty
        return r0_penalty(self.r0_w, self.r0_d, self.loss.cov_radius)


# ---------------------------------------------------------------------------
# mu: Loss's arithmetic, bit for bit
def _full(band, dft, pr):
    import utils.data as data
    with _quiet():
        return data.get_full_dftb(band, dft, pr, True)


def mu(step: StepData, sigma, parts: dict | None = None) -> float:
    """The loss at sigma, as Loss.parallel_repulsion / unparallel_repulsion compute it
    (bit for bit; no files are written).  With parameters per type (M5b) sigma is the vector
    of the parameter sets and this is the loss of Loss.Repulsion_optimization_types.
    ``parts`` (a dict) receives the terms the log shows (mu_E, mu_V or mu_F, mu_logb, the
    penalty; in dataset/reaction mode also the per-structure |dE|)."""
    if step.mode == "energygeometry":
        return _mu_energygeometry(step, sigma, parts)
    return _mu_dataset(step, sigma, parts)


def _group_rel_energies(loss, dft_sub, atoms_total, bolt):
    with _quiet():
        return loss.get_energy_per_atom(dft_sub, atoms_total, bolt)


def _eos_v0(phase) -> float:
    """v0 of Loss.eos_atoms (FAILED_EOS when the fit fails)."""
    from utils.loss import _eos_fit
    try:
        return _eos_fit(phase)[1]
    except RuntimeError:
        return FAILED_EOS


class _Energies:
    """Atoms stand-ins for the EOS fit: volume and energy as the dftb_sub file gives them."""
    def __init__(self, atoms, energy):
        self._a, self._e = atoms, energy

    def get_volume(self):
        return self._a.get_volume()

    def get_potential_energy(self):
        return self._e

    def __len__(self):
        return len(self._a)


def _eos_logb(phase_b, phase_d) -> float:
    """Loss.eos_cal's |log10 B / log10 B_DFT - 1| + 1 of one group (logged, not in mu)."""
    from ase.units import kJ

    from utils.loss import _eos_fit

    def bulk(phase):
        try:
            return _eos_fit(phase)[2]
        except RuntimeError:
            return FAILED_EOS
    b_dftb, b_dft = bulk(phase_b) * 1e24 / kJ, bulk(phase_d) * 1e24 / kJ
    try:
        mu_logb = abs(np.log10(b_dftb) / np.log10(b_dft) - 1) + 1
    except ZeroDivisionError:
        mu_logb = 100
    return 100 if math.isnan(mu_logb) else mu_logb


def _mu_energygeometry(step: StepData, sigma: float, parts: dict | None = None) -> float:
    eos = step.eos
    ng = int(len(step.dft) / eos)
    mu_es, mu_vs, mu_bs = [], [], []
    for num in range(ng):
        sl = slice(num * eos, (num + 1) * eos)
        pr, _ = step.physrep(sigma, num)
        total = _full(step.band[sl], step.dft[sl], pr)
        _, _, delta = _group_rel_energies(step.loss, step.dft[sl], total, step.bolt11[sl])
        # dftb_sub_<num>.xyz keeps energies, cells and mu_E exactly (repr / f'{x}')
        v0 = _eos_v0([_Energies(a, a.info['energy']) for a in total])
        v0_dft = _eos_v0(step.dft[sl])
        b = step.bolt_f[num]
        mu_v = abs(v0 / v0_dft - 1)
        if math.isnan(mu_v):
            mu_v = 100
        mu_vs.append(0.0 if b < 0.1 else 0.0 + mu_v * b)
        mu_es.append(delta * b)
        if parts is not None:
            parts.setdefault("v0", []).append((v0, v0_dft))
            if b >= 0.1:
                mu_bs.append(_eos_logb([_Energies(a, a.info['energy']) for a in total],
                                       step.dft[sl]) * b)
    bolt_left = np.sum([step.bolt_f[i] for i in range(len(step.bolt_f)) if step.bolt_f[i] >= 0.1])
    mu_e, mu_v = np.sum(mu_es) / bolt_left, np.sum(mu_vs) / bolt_left
    c = EG_C
    penalty = step.penalty()
    out = (c * mu_v + (1 - c) * mu_e) * penalty
    if parts is not None:
        parts.update(mu_E=mu_e, mu_V=mu_v, mu_logb=np.sum(mu_bs) / bolt_left,
                     bolt_left=bolt_left, penalty=penalty)
    return out


def _mu_dataset(step: StepData, sigma: float, parts: dict | None = None) -> float:
    loss = step.loss
    pr, _ = step.physrep(sigma)
    full = _full(step.band, step.dft, pr)
    with _quiet():
        e0_dftb = loss.get_atomic_energy(step.dft)
        _, rel_b = loss.get_formation_energy(full, e0_dftb, '_')
        _, rel_d = loss.get_formation_energy(step.dft, loss.args_input.E0s, '_')
    diffs = np.abs(np.array(rel_b) - np.array(rel_d))
    mu_e = np.sum(diffs)
    f_d = loss.get_forces(step.dft, '_')
    f_b = loss.get_forces(full, '_')
    mu_f = np.mean(np.abs(np.array(f_d) - np.array(f_b)))
    c = DS_C
    penalty = step.penalty()
    out = (mu_e * (1 - c) + mu_f * c) * penalty
    if parts is not None:
        parts.update(mu_E=mu_e, mu_F=mu_f, e0=e0_dftb, diffs=diffs, penalty=penalty)
    return out


# ---------------------------------------------------------------------------
# derivatives of the repulsion and of the EOS fit
_DSPLINES: dict = {}


def _dsigma_physrep(pr, varying):
    """A copy of ``pr`` whose splines are dV/dsigma on the same knots: its evaluate_pbc
    gives (dE_rep/dsigma, dF_rep/dsigma), the derivative of physrep's splines with the
    knots fixed (the cut-off moves with sigma in steps of ~1e-8 Ha, ignored).
    V = Z1 Z2 erfc(g r)/r (k_xc = 0), g = 1/sqrt(a1^2 + a2^2), a = R_cov sigma."""
    import copy

    from ase.data import covalent_radii
    from ase.units import Bohr
    from scipy import interpolate
    key = (id(pr), tuple(varying))
    hit = _DSPLINES.get(key)
    if hit is not None and hit[0] is pr:
        return hit[1]
    if any(k != 0.0 for k in pr.k_xc.values()):
        raise GradientUnavailable("physrep with k_xc != 0 has no sigma derivative here")
    out = copy.copy(pr)
    out.Vreps = {}
    for key1, spline in pr.Vreps.items():
        z1, z2 = (int(z) for z in key1.split("_"))
        a1, a2 = pr.atsize[z1], pr.atsize[z2]
        g = 1.0 / np.sqrt(a1 ** 2 + a2 ** 2)
        da = [covalent_radii[z] / Bohr if z in varying else 0.0 for z in (z1, z2)]
        dg = -g ** 3 * (a1 * da[0] + a2 * da[1])
        r = spline.x
        dv = -z1 * z2 * (2.0 / np.sqrt(np.pi)) * np.exp(-(g * r) ** 2) * dg
        out.Vreps[key1] = interpolate.CubicSpline(r, dv, bc_type='natural')
    _DSPLINES.clear()                     # one physrep at a time is enough
    _DSPLINES[key] = (pr, out)
    return out


def repulsion_sigma_derivative(pr, varying, atoms):
    """(dE_rep/dsigma in eV, dF_rep/dsigma in eV/A) of one structure."""
    return _dsigma_physrep(pr, varying).evaluate_pbc(atoms)


def birch_murnaghan(v, e0, b0, bp, v0):
    """ase.eos.birchmurnaghan (any array namespace)."""
    eta = (v0 / v) ** (1 / 3)
    return e0 + 9 * b0 * v0 / 16 * (eta ** 2 - 1) ** 2 * (6 + bp * (eta ** 2 - 1) - 4 * eta ** 2)


def eos_v0_sensitivity(vols, ens, params) -> np.ndarray:
    """dv0/de_j of the least-squares Birch-Murnaghan fit (e_j the energies per atom) by
    the implicit function theorem: p minimises S = sum_j (f(V_j; p) - e_j)^2, so
    dp/de_j = H^-1 grad_p f_j, H = sum_j (grad f_j grad f_j^T + r_j hess f_j).
    ``params`` = (e0, B, Bp, v0) in the function's order."""
    import torch
    v = torch.as_tensor(np.asarray(vols, dtype=float))
    e = torch.as_tensor(np.asarray(ens, dtype=float))
    p = torch.as_tensor(np.asarray(params, dtype=float))

    def f(q):
        return birch_murnaghan(v, q[0], q[1], q[2], q[3])

    jac = torch.autograd.functional.jacobian(f, p)                # (n, 4)
    res = (f(p) - e).detach()
    hess = torch.autograd.functional.hessian(lambda q: (res * f(q)).sum(), p)
    h = jac.T @ jac + hess
    dp = torch.linalg.solve(h, jac.T)                             # (4, n)
    return dp[3].numpy()


# ---------------------------------------------------------------------------
# the residuals and their derivatives
@dataclass
class Terms:
    """mu = P sum_j w_j |r_j|, with dr/dsigma and the linear map from the per-structure
    energy derivatives (and E0, forces) to dr/dr0."""
    sigma: float
    penalty: float
    w: np.ndarray
    r: np.ndarray
    drds: np.ndarray
    kind: list                   # "E" | "V" | "F"
    a_e: np.ndarray              # (n_terms, n_structures): dr/dE_i (E in eV)
    a_e0: np.ndarray             # (n_terms, n_elements): dr/dE0_Z (dataset)
    elements: list               # Z of the columns of a_e0
    force: list                  # "F" terms: (structure, atom, component)
    natoms: list
    notes: list = field(default_factory=list)     # warnings on the data (EOS fits ...)

    @property
    def zero(self) -> np.ndarray:
        """Residuals that vanish identically near sigma (symmetry): no sign, no kink."""
        return (np.abs(self.r) <= ZERO_RESIDUAL) & (np.abs(self.drds) <= ZERO_SLOPE)

    def value(self) -> float:
        return float(self.penalty * np.sum(self.w * np.abs(self.r)))

    def kinks(self) -> np.ndarray:
        """Indices of the residuals sigma sits on the kink of (|r / r'| <= KINK_DISTANCE)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = np.abs(self.r) / np.abs(self.drds)
        return np.nonzero((dist <= KINK_DISTANCE) & ~self.zero)[0]

    def slopes(self, kinks=()) -> tuple:
        """One-sided dmu/dsigma (left, right) with the residuals ``kinks`` at zero."""
        s = np.sign(self.r)
        s[self.zero] = 0.0
        inside = np.zeros(len(self.r), bool)
        inside[list(kinks)] = True
        base = np.sum((self.w * s * self.drds)[~inside])
        spread = np.sum(np.abs(self.w * self.drds)[inside])
        return self.penalty * (base - spread), self.penalty * (base + spread)


def terms(step: StepData, sigma: float) -> Terms:
    """The residuals of the loss at sigma with their derivatives."""
    return _terms_energygeometry(step, sigma) if step.mode == "energygeometry" \
        else _terms_dataset(step, sigma)


def _terms_energygeometry(step: StepData, sigma: float) -> Terms:
    from utils.loss import _eos_fit
    eos, n = step.eos, len(step.dft)
    ng = int(n / eos)
    bolt_left = np.sum([step.bolt_f[i] for i in range(len(step.bolt_f)) if step.bolt_f[i] >= 0.1])
    c = EG_C
    w, r, drds, kind, rows, notes = [], [], [], [], [], []
    natoms = [len(a) for a in step.dft]
    for num in range(ng):
        idx = list(range(num * eos, (num + 1) * eos))
        pr, varying = step.physrep(sigma, num)
        total = _full(step.band[num * eos:(num + 1) * eos], step.dft[num * eos:(num + 1) * eos], pr)
        der = [repulsion_sigma_derivative(pr, varying, step.dft[i]) for i in idx]
        e_b = np.array([a.info['energy'] / a.get_global_number_of_atoms() for a in total])
        e_d = np.array([step.dft[i].get_total_energy() / step.dft[i].get_global_number_of_atoms()
                        for i in idx])
        de = np.array([d[0] for d in der])                    # dE_rep/dsigma, eV
        b_g = step.bolt_f[num]
        bolt = [step.bolt11[i] for i in idx]
        mask = [x >= 0.1 for x in bolt]
        if any(mask):
            big_b = np.sum([x for x in bolt if x >= 0.1])
            m0 = int(np.argmin(e_d))
            for k, i in enumerate(idx):
                if not mask[k] or k == m0:          # the reference structure: identically 0
                    continue
                row = np.zeros(n)
                row[i] += 1.0 / natoms[i]
                row[idx[m0]] -= 1.0 / natoms[idx[m0]]
                w.append((1 - c) * b_g / (bolt_left * big_b))
                r.append((e_b[k] - e_b[m0]) - (e_d[k] - e_d[m0]))
                drds.append(de[k] / natoms[i] - de[m0] / natoms[idx[m0]])
                kind.append("E")
                rows.append(row)
        if b_g >= 0.1:
            ens = [a.info['energy'] / len(a) for a in total]
            vols = [a.get_volume() / len(a) for a in total]
            try:
                e0, v0, bm, bp = _eos_fit([_Energies(a, a.info['energy']) for a in total])
                v0_dft = _eos_fit(step.dft[num * eos:(num + 1) * eos])[1]
            except RuntimeError as err:
                raise GradientUnavailable(f"the EOS fit of group {num} failed ({err})")
            ratio = v0 / v0_dft - 1
            if not np.isfinite(ratio):
                raise GradientUnavailable(f"the volume term of group {num} is not finite")
            # a fit without a minimum (B <= 0) or one outside the sampled volumes: the loss
            # can jump nearby (another least-squares solution, a failed fit; M4c report)
            if bm <= 0:
                notes.append(f"group {num}: the EOS fit has no minimum (B = {bm:.3g})")
            elif not min(vols) <= v0 <= max(vols):
                notes.append(f"group {num}: EOS minimum {v0:.3f} outside the volumes "
                             f"{min(vols):.3f}-{max(vols):.3f}")
            dv0 = eos_v0_sensitivity(vols, ens, (e0, bm, bp, v0))     # per energy per atom
            row = np.zeros(n)
            for k, i in enumerate(idx):
                row[i] = dv0[k] / (natoms[i] * v0_dft)
            w.append(c * b_g / bolt_left)
            r.append(ratio)
            drds.append(float(np.sum([row[i] * de[k] for k, i in enumerate(idx)])))
            kind.append("V")
            rows.append(row)
    a_e = np.array(rows) if rows else np.zeros((0, n))
    return Terms(float(sigma), step.penalty(), np.array(w), np.array(r), np.array(drds), kind,
                 a_e, np.zeros((len(w), 0)), [], [], natoms, notes)


def _terms_dataset(step: StepData, sigma: float) -> Terms:
    loss = step.loss
    n = len(step.dft)
    pr, varying = step.physrep(sigma)
    full = _full(step.band, step.dft, pr)
    with _quiet():
        e0_dftb = loss.get_atomic_energy(step.dft)
        _, rel_b = loss.get_formation_energy(full, e0_dftb, '_')
        _, rel_d = loss.get_formation_energy(step.dft, loss.args_input.E0s, '_')
    elements = [int(z) for z in e0_dftb.keys()]
    natoms = [len(a) for a in step.dft]
    c = DS_C
    der = [repulsion_sigma_derivative(pr, varying, a) for a in step.dft]
    w, r, drds, kind, force = [], [], [], [], []
    a_e = np.zeros((n, n))
    a_e0 = np.zeros((n, len(elements)))
    for i, a in enumerate(step.dft):
        w.append(1 - c)
        r.append(rel_b[i] - rel_d[i])
        drds.append(der[i][0] / natoms[i])
        kind.append("E")
        a_e[i, i] = 1.0 / natoms[i]
        numbers = a.get_atomic_numbers().tolist()
        for k, z in enumerate(elements):
            a_e0[i, k] = -numbers.count(z) / natoms[i]
    f_d = np.array(loss.get_forces(step.dft, '_'))
    f_b = np.array(loss.get_forces(full, '_'))
    n_f = len(f_d)
    dfs = np.concatenate([np.asarray(d[1]).ravel() for d in der]) if der else np.zeros(0)
    pos = 0
    for i, a in enumerate(step.dft):
        for atom in range(len(a)):
            for comp in range(3):
                w.append(c / n_f)
                r.append(f_b[pos] - f_d[pos])
                drds.append(dfs[pos])
                kind.append("F")
                force.append((i, atom, comp))
                pos += 1
    nt = len(w)
    a_e_full = np.zeros((nt, n))
    a_e_full[:n] = a_e
    a_e0_full = np.zeros((nt, len(elements)))
    a_e0_full[:n] = a_e0
    return Terms(float(sigma), step.penalty(), np.array(w), np.array(r), np.array(drds), kind,
                 a_e_full, a_e0_full, elements, [None] * n + force, natoms)


# ---------------------------------------------------------------------------
# sigma*: the inner minimum near PTBP's sigma_hat
@dataclass
class InnerMinimum:
    sigma: float                 # sigma*
    case: str                    # "smooth" | "kink" | "bound"
    terms: Terms                 # at sigma*
    kinks: tuple = ()            # residuals at their kink (case "kink")
    mu: float = 0.0              # mu(sigma*)
    evaluations: int = 0
    note: str = ""


def refine_sigma(step: StepData, sigma_hat: float, bounds=SIGMA_BOUNDS, width: float = 1e-3,
                 xatol: float = 1e-12) -> InnerMinimum:
    """The local minimum sigma* of mu(sigma) next to ``sigma_hat`` (PTBP's L-BFGS-B result,
    ftol 1e-3) and its kind: a smooth minimum, the kink of one residual (or of several
    that vanish together), or a bound of the search.  Bounded Brent on the exact mu
    in a window around sigma_hat (widened while the minimum sits on its edge), then the
    kink, if any, is located exactly by Newton steps on its residual."""
    from scipy.optimize import minimize_scalar
    lo, hi = bounds
    s0 = min(max(float(sigma_hat), lo), hi)
    count = [0]

    def f(s):
        count[0] += 1
        return float(mu(step, s))

    for _ in range(8):
        a, b = max(lo, s0 - width), min(hi, s0 + width)
        res = minimize_scalar(f, bounds=(a, b), method="bounded",
                              options={"xatol": xatol, "maxiter": 500})
        s = float(res.x)
        # bounded Brent stops ~sqrt(eps)|s| short of an edge: an edge minimum widens the window
        edge = 1e-4 * (b - a)
        at_edge = (s - a < edge and a > lo) or (b - s < edge and b < hi)
        if not at_edge:
            break
        s0, width = s, width * 4
    for bound, inward in ((lo, 1.0), (hi, -1.0)):           # a minimum on a bound of the search
        if abs(s - bound) < 1e-6:
            t = terms(step, bound)
            count[0] += 1
            left, right = t.slopes(t.kinks())
            if (right if inward > 0 else -left) >= 0:
                return InnerMinimum(bound, "bound", t, (), t.value(), count[0])
    t = terms(step, s)
    kinks = tuple(t.kinks())
    if kinks:
        # Newton on the residual nearest to its kink (they vanish together)
        k = min(kinks, key=lambda j: abs(t.r[j] / t.drds[j]))
        for _ in range(4):
            step_s = -t.r[k] / t.drds[k]
            if not np.isfinite(step_s) or abs(step_s) < 1e-15:
                break
            s = min(max(s + step_s, lo), hi)
            t = terms(step, s)
            count[0] += 1
        kinks = tuple(j for j in t.kinks()) or (k,)
        left, right = t.slopes(kinks)
        note = "" if left <= 0 <= right else (f"one-sided slopes {left:.3g}, {right:.3g} do not "
                                              "bracket zero")
        return InnerMinimum(s, "kink", t, kinks, t.value(), count[0], note)
    return InnerMinimum(s, "smooth", t, (), t.value(), count[0])


# ---------------------------------------------------------------------------
# the gradient
def dlnp(step: StepData) -> dict:
    """d ln P / d r0 (1/Bohr)."""
    r = step.loss.cov_radius
    a = step.r0_w / (2 * r) - 1
    b = step.r0_d / (4 * r) - 1
    return {"r0_w": 2 * a / (r * (a * a + 1)), "r0_d": b / (r * (b * b + 2))}


def force_weights(t: Terms, exclude=()) -> dict:
    """Pass-2 weights {structure: (n_atoms, 3)} = P w_j sign(r_j) of the force residuals
    (zero for the residuals in ``exclude`` and those vanishing identically)."""
    s = np.sign(t.r)
    s[t.zero] = 0.0
    s[list(exclude)] = 0.0
    out = {}
    for j, kind in enumerate(t.kind):
        if kind != "F":
            continue
        i, atom, comp = t.force[j]
        arr = out.setdefault(i, np.zeros((t.natoms[i], 3)))
        arr[atom, comp] = t.penalty * t.w[j] * s[j]
    return out


def assemble(step: StepData, inner: InnerMinimum, edot: dict, e0dot: dict | None = None,
             fdot: dict | None = None, kink_fdot: dict | None = None,
             parts: dict | None = None) -> dict:
    """dmu*/dr0 per direction.  ``edot[name]``: Edot of every structure (eV/Bohr, the
    order of the step's files); ``e0dot[name]``: {Z: E0dot}; ``fdot[name]``: the pass-2
    sum of P w s Fdot over the force residuals; ``kink_fdot[name]``: Fdot of the force
    residual at the kink (if it is one).  ``parts`` (a dict) receives, per direction, the
    energy, force, penalty and kink-correction parts; all but the last make the derivative
    at fixed sigma*."""
    t = inner.terms
    s = np.sign(t.r)
    s[t.zero] = 0.0
    kinks = list(inner.kinks)
    s_tilde = s.copy()
    s_tilde[kinks] = 0.0
    weights = t.penalty * t.w * s_tilde
    dl = dlnp(step)
    out = {}
    for name, ed in edot.items():
        ed = np.asarray(ed, dtype=float)
        e0 = np.array([(e0dot or {}).get(name, {}).get(z, 0.0) for z in t.elements])
        drdr0 = t.a_e @ ed + (t.a_e0 @ e0 if len(e0) else 0.0)        # E and V residuals
        part = {"energies": float(np.sum(weights * drdr0)),
                "forces": float((fdot or {}).get(name, 0.0)),
                "penalty": inner.mu * dl[name], "kink": 0.0}
        if inner.case == "kink":
            k = kinks[0]
            dmu_ds = float(np.sum(weights * t.drds))
            dk = (kink_fdot or {}).get(name, 0.0) if t.kind[k] == "F" else float(drdr0[k])
            part["kink"] = dmu_ds * (-dk / t.drds[k])
        out[name] = part["energies"] + part["forces"] + part["penalty"] + part["kink"]
        if parts is not None:
            parts[name] = part
    return out


# ---------------------------------------------------------------------------
# parameters per type (M5b): sigma is a vector, one sigma_rep per parameter set.
#
# mu*(x) = min over sigma of mu(x, sigma).  At a joint inner minimum sigma* the envelope
# theorem still holds: smooth (grad_sigma mu = 0) or on a bound of some components (they stay
# there), dmu*/dx = dmu/dx at sigma*.  On kinks: the residuals K vanish at sigma*
# (r_K(x, sigma*) = 0); sigma* is a minimum of mu~ = P sum_{j not in K} w_j |r_j| on that set,
# so grad mu~ + J_K^T lambda = 0 on the free components (J_K = dr_K/dsigma) and
# dmu*/dx = dmu~/dx + lambda . dr_K/dx (the multipliers' envelope theorem).  For one sigma and
# one kink lambda = -(dmu~/dsigma)/(dr_k/dsigma): M4c's kink correction.
KINK_DISTANCE_N = 1e-7            # |r| / |dr/dsigma| below which sigma* is on the kink of r


@dataclass
class TermsN:
    """:class:`Terms` with a sigma vector: ``drds`` (n_terms, n_sigma)."""
    sigma: np.ndarray
    penalty: float
    w: np.ndarray
    r: np.ndarray
    drds: np.ndarray
    kind: list
    a_e: np.ndarray
    a_e0: np.ndarray
    elements: list
    force: list
    natoms: list
    notes: list = field(default_factory=list)

    @property
    def zero(self) -> np.ndarray:
        return (np.abs(self.r) <= ZERO_RESIDUAL) & (np.abs(self.drds).max(axis=1) <= ZERO_SLOPE)

    def value(self) -> float:
        return float(self.penalty * np.sum(self.w * np.abs(self.r)))

    def kinks(self, free=None) -> np.ndarray:
        """Residuals sigma sits on the kink of (|r| / |dr/dsigma| <= KINK_DISTANCE_N over the
        free components)."""
        cols = slice(None) if free is None else np.asarray(free, bool)
        slope = np.linalg.norm(self.drds[:, cols], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = np.abs(self.r) / slope
        return np.nonzero((dist <= KINK_DISTANCE_N) & ~self.zero)[0]

    def smooth_gradient(self, exclude=()) -> np.ndarray:
        """d mu~/d sigma with the residuals ``exclude`` (and the zero ones) left out."""
        s = np.sign(self.r)
        s[self.zero] = 0.0
        s[list(exclude)] = 0.0
        return self.penalty * ((self.w * s) @ self.drds)


def terms_typed(step: StepData, sigma) -> TermsN:
    """The residuals of the loss at the sigma vector with their sigma derivatives."""
    sigma = np.asarray([float(v) for v in sigma])
    if step.mode == "energygeometry":
        return _terms_energygeometry_typed(step, sigma)
    return _terms_dataset_typed(step, sigma)


def _sigma_derivatives(pr, varying, structures, ns) -> list:
    """(dE_rep/dsigma_k (n, ns), dF_rep/dsigma_k [(natoms, 3, ns)]) of every structure."""
    de = np.zeros((len(structures), ns))
    df = [np.zeros((len(a), 3, ns)) for a in structures]
    for k, zs in (varying or {}).items():
        for i, a in enumerate(structures):
            e, f = repulsion_sigma_derivative(pr, zs, a)
            de[i, k] = e
            df[i][:, :, k] = f
    return de, df


def _terms_energygeometry_typed(step: StepData, sigma) -> TermsN:
    from utils.loss import _eos_fit
    eos, n, ns = step.eos, len(step.dft), step.model.n_sigma
    ng = int(n / eos)
    bolt_left = np.sum([step.bolt_f[i] for i in range(len(step.bolt_f)) if step.bolt_f[i] >= 0.1])
    c = EG_C
    w, r, drds, kind, rows, notes = [], [], [], [], [], []
    natoms = [len(a) for a in step.dft]
    for num in range(ng):
        idx = list(range(num * eos, (num + 1) * eos))
        sub = step.dft[num * eos:(num + 1) * eos]
        pr, varying = step.physrep(sigma, num)
        total = _full(step.band[num * eos:(num + 1) * eos], sub, pr)
        de, _ = _sigma_derivatives(pr, varying, sub, ns)
        e_b = np.array([a.info['energy'] / a.get_global_number_of_atoms() for a in total])
        e_d = np.array([a.get_total_energy() / a.get_global_number_of_atoms() for a in sub])
        b_g = step.bolt_f[num]
        bolt = [step.bolt11[i] for i in idx]
        mask = [x >= 0.1 for x in bolt]
        if any(mask):
            big_b = np.sum([x for x in bolt if x >= 0.1])
            m0 = int(np.argmin(e_d))
            for k, i in enumerate(idx):
                if not mask[k] or k == m0:
                    continue
                row = np.zeros(n)
                row[i] += 1.0 / natoms[i]
                row[idx[m0]] -= 1.0 / natoms[idx[m0]]
                w.append((1 - c) * b_g / (bolt_left * big_b))
                r.append((e_b[k] - e_b[m0]) - (e_d[k] - e_d[m0]))
                drds.append(de[k] / natoms[i] - de[m0] / natoms[idx[m0]])
                kind.append("E")
                rows.append(row)
        if b_g >= 0.1:
            ens = [a.info['energy'] / len(a) for a in total]
            vols = [a.get_volume() / len(a) for a in total]
            try:
                e0, v0, bm, bp = _eos_fit([_Energies(a, a.info['energy']) for a in total])
                v0_dft = _eos_fit(sub)[1]
            except RuntimeError as err:
                raise GradientUnavailable(f"the EOS fit of group {num} failed ({err})")
            ratio = v0 / v0_dft - 1
            if not np.isfinite(ratio):
                raise GradientUnavailable(f"the volume term of group {num} is not finite")
            if bm <= 0:
                notes.append(f"group {num}: the EOS fit has no minimum (B = {bm:.3g})")
            elif not min(vols) <= v0 <= max(vols):
                notes.append(f"group {num}: EOS minimum {v0:.3f} outside the volumes "
                             f"{min(vols):.3f}-{max(vols):.3f}")
            dv0 = eos_v0_sensitivity(vols, ens, (e0, bm, bp, v0))
            row = np.zeros(n)
            for k, i in enumerate(idx):
                row[i] = dv0[k] / (natoms[i] * v0_dft)
            w.append(c * b_g / bolt_left)
            r.append(ratio)
            drds.append(np.sum([row[i] * de[k] for k, i in enumerate(idx)], axis=0))
            kind.append("V")
            rows.append(row)
    a_e = np.array(rows) if rows else np.zeros((0, n))
    return TermsN(np.asarray(sigma), step.penalty(), np.array(w), np.array(r),
                  np.array(drds).reshape(len(w), ns), kind, a_e, np.zeros((len(w), 0)), [], [],
                  natoms, notes)


def _terms_dataset_typed(step: StepData, sigma) -> TermsN:
    loss = step.loss
    n, ns = len(step.dft), step.model.n_sigma
    pr, varying = step.physrep(sigma)
    full = _full(step.band, step.dft, pr)
    with _quiet():
        e0_dftb = loss.get_atomic_energy(step.dft)
        _, rel_b = loss.get_formation_energy(full, e0_dftb, '_')
        _, rel_d = loss.get_formation_energy(step.dft, loss.args_input.E0s, '_')
    elements = [int(z) for z in e0_dftb.keys()]
    natoms = [len(a) for a in step.dft]
    c = DS_C
    de, df = _sigma_derivatives(pr, varying, step.dft, ns)
    w, r, drds, kind, force = [], [], [], [], []
    a_e = np.zeros((n, n))
    a_e0 = np.zeros((n, len(elements)))
    for i, a in enumerate(step.dft):
        w.append(1 - c)
        r.append(rel_b[i] - rel_d[i])
        drds.append(de[i] / natoms[i])
        kind.append("E")
        a_e[i, i] = 1.0 / natoms[i]
        numbers = a.get_atomic_numbers().tolist()
        for k, z in enumerate(elements):
            a_e0[i, k] = -numbers.count(z) / natoms[i]
    f_d = np.array(loss.get_forces(step.dft, '_'))
    f_b = np.array(loss.get_forces(full, '_'))
    n_f = len(f_d)
    pos = 0
    for i, a in enumerate(step.dft):
        for atom in range(len(a)):
            for comp in range(3):
                w.append(c / n_f)
                r.append(f_b[pos] - f_d[pos])
                drds.append(df[i][atom, comp])
                kind.append("F")
                force.append((i, atom, comp))
                pos += 1
    nt = len(w)
    a_e_full = np.zeros((nt, n))
    a_e_full[:n] = a_e
    a_e0_full = np.zeros((nt, len(elements)))
    a_e0_full[:n] = a_e0
    return TermsN(np.asarray(sigma), step.penalty(), np.array(w), np.array(r),
                  np.array(drds).reshape(nt, ns), kind, a_e_full, a_e0_full, elements,
                  [None] * n + force, natoms)


@dataclass
class InnerMinimumN:
    sigma: np.ndarray            # sigma*
    case: str                    # "smooth" | "kink" | "bound" (bound: some components)
    terms: TermsN                # at sigma*
    kinks: tuple = ()            # residuals at their kink
    bound: tuple = ()            # components on a bound of the search
    mu: float = 0.0
    evaluations: int = 0
    note: str = ""


def refine_sigma_typed(step: StepData, sigma_hat, bounds=SIGMA_BOUNDS, scale: float = 1e-3,
                       xatol: float = 1e-11) -> InnerMinimumN:
    """The local minimum sigma* of mu(sigma) next to PTBP's L-BFGS-B result (ftol 1e-3):
    bounded Nelder-Mead on the exact mu from a small simplex around sigma_hat, then Newton
    steps that put the residuals sigma* sits on the kink of exactly at zero (least squares in
    the free components); the case and the bound components are classified at sigma*."""
    from scipy.optimize import minimize as _minimize
    lo, hi = bounds
    s0 = np.clip(np.asarray([float(v) for v in sigma_hat]), lo, hi)
    n = len(s0)
    count = [0]

    def f(s):
        count[0] += 1
        return float(mu(step, np.clip(s, lo, hi)))

    simplex = [s0] + [np.clip(s0 + scale * np.eye(n)[k] * (1 if s0[k] + scale <= hi else -1),
                              lo, hi) for k in range(n)]
    res = _minimize(f, s0, method="Nelder-Mead", bounds=[(lo, hi)] * n,
                    options={"xatol": xatol, "fatol": 1e-15, "maxfev": 400 * n,
                             "initial_simplex": np.array(simplex)})
    s = np.clip(np.asarray(res.x, float), lo, hi)
    t = terms_typed(step, s)
    count[0] += 1
    best = (t.value(), s, t)

    def bound_components(t, s):
        g = t.smooth_gradient()
        return tuple(k for k in range(n)
                     if (abs(s[k] - lo) < 1e-9 and g[k] >= 0) or (abs(s[k] - hi) < 1e-9 and g[k] <= 0))

    bnd = bound_components(t, s)
    free = np.array([k not in bnd for k in range(n)])
    kinks = tuple(t.kinks(free))
    for _ in range(4):                       # put the kink residuals exactly at zero
        if not kinks or not free.any():
            break
        jac = t.drds[np.ix_(list(kinks), np.flatnonzero(free))]
        step_s = np.zeros(n)
        step_s[free] = -np.linalg.lstsq(jac, t.r[list(kinks)], rcond=None)[0]
        if not np.all(np.isfinite(step_s)) or np.max(np.abs(step_s)) < 1e-15:
            break
        s_new = np.clip(s + step_s, lo, hi)
        t_new = terms_typed(step, s_new)
        count[0] += 1
        if t_new.value() > best[0] + 1e-12 * max(abs(best[0]), 1.0):
            break
        s, t = s_new, t_new
        best = (t.value(), s, t)
        kinks = tuple(t.kinks(free)) or kinks
    note = ""
    if kinks:
        g = t.smooth_gradient(exclude=kinks)[free]
        jac = t.drds[np.ix_(list(kinks), np.flatnonzero(free))]
        lam = np.linalg.lstsq(jac.T, -g, rcond=None)[0]
        resid = np.linalg.norm(jac.T @ lam + g)
        if resid > 1e-6 * max(np.linalg.norm(g), 1e-12):
            note = f"the kink multipliers leave {resid:.3g} of the sigma gradient"
        case = "kink"
    else:
        case = "bound" if bnd else "smooth"
    return InnerMinimumN(s, case, t, kinks, bnd, t.value(), count[0], note)


def kink_multipliers(inner: InnerMinimumN) -> np.ndarray:
    """lambda of the kink residuals: J_K^T lambda = -d mu~/d sigma on the free components."""
    t = inner.terms
    n = len(inner.sigma)
    free = np.array([k not in inner.bound for k in range(n)])
    if not inner.kinks or not free.any():
        return np.zeros(len(inner.kinks))
    g = t.smooth_gradient(exclude=inner.kinks)[free]
    jac = t.drds[np.ix_(list(inner.kinks), np.flatnonzero(free))]
    return np.linalg.lstsq(jac.T, -g, rcond=None)[0]


def assemble_typed(step: StepData, inner: InnerMinimumN, edot: dict, e0dot: dict | None = None,
                   fdot: dict | None = None, kink_fdot: dict | None = None,
                   parts: dict | None = None) -> dict:
    """dmu*/dx for every outer parameter (the model's dimension names).  ``kink_fdot[name]``:
    the pass-2 value of sum_k lambda_k Fdot_k over the force residuals at a kink."""
    t = inner.terms
    s = np.sign(t.r)
    s[t.zero] = 0.0
    kinks = list(inner.kinks)
    s[kinks] = 0.0
    weights = t.penalty * t.w * s
    lam = kink_multipliers(inner)
    names = step.model.dimension_names()
    dl = dict(zip(names, step.model.dlnp(step.x)))
    out = {}
    for name in names:
        ed = np.asarray(edot[name], dtype=float)
        e0 = np.array([(e0dot or {}).get(name, {}).get(z, 0.0) for z in t.elements])
        drdr0 = t.a_e @ ed + (t.a_e0 @ e0 if len(e0) else 0.0)
        part = {"energies": float(np.sum(weights * drdr0)),
                "forces": float((fdot or {}).get(name, 0.0)),
                "penalty": inner.mu * dl[name], "kink": 0.0}
        if kinks:
            part["kink"] = float(sum(lam[i] * drdr0[k] for i, k in enumerate(kinks)
                                     if t.kind[k] != "F")
                                 + (kink_fdot or {}).get(name, 0.0))
        out[name] = part["energies"] + part["forces"] + part["penalty"] + part["kink"]
        if parts is not None:
            parts[name] = part
    return out


# ---------------------------------------------------------------------------
# the hook of Loss.EnergyGeometry
@dataclass
class GradResult:
    """Value and gradient of one loss evaluation."""
    r0_w: float
    r0_d: float
    mu: float | None                 # the loss PTBP returns (at sigma_hat)
    grad: dict | None                # {"r0_w": dmu/dr0_w, "r0_d": ...} (1/Bohr); None: unavailable
    sigma_hat: float | None = None
    sigma_star: float | None = None
    case: str = ""                   # smooth | kink | bound
    gap: float = 0.0                 # mu(sigma_hat) - mu(sigma*)
    reason: str = ""                 # why there is no gradient
    tangent_zero: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)  # s: tangent, sigma, forces, other (grad.out: sum)
    note: str = ""
    parts: dict = field(default_factory=dict)     # see assemble
    x: list | None = None            # parameters per type (M5b): the outer parameters
    names: list | None = None        # and their names (the model's dimension names)

    def line(self) -> str:
        """A line of grad.out."""
        g = self.grad or {}
        val = (lambda x: f"{x:20.10e}" if x is not None else f"{'nan':>20s}")
        if self.x is not None:                       # M5b: every parameter, sigma vectors
            vec = (lambda v: "[" + " ".join(f"{s:.10f}" for s in v) + "]" if v is not None
                   else "nan")
            return (", ".join(f"{v:15.8f}" for v in self.x) + f", {val(self.mu)}, "
                    + ", ".join(val(g.get(n)) for n in self.names)
                    + f", {vec(self.sigma_hat)}, {vec(self.sigma_star)}, {self.case or '-':>6s}, "
                    f"{self.gap:12.3e}, {sum(self.timings.values()):8.1f}, "
                    f"{self.reason or self.note or '-'}")
        return (f"{self.r0_w:15.8f}, {self.r0_d:15.8f}, {val(self.mu)}, {val(g.get('r0_w'))}, "
                f"{val(g.get('r0_d'))}, {val(self.sigma_hat)}, {val(self.sigma_star)}, "
                f"{self.case or '-':>6s}, {self.gap:12.3e}, "
                f"{sum(self.timings.values()):8.1f}, {self.reason or self.note or '-'}")


GRAD_HEADER = ("#           r0_w,            r0_d,                   mu,           dmu/dr0_w,"
               "           dmu/dr0_d,            sigma_hat,           sigma_star,   case,"
               "          gap,     time, note")


def grad_header(names) -> str:
    """The header of grad.out for parameters per type (M5b)."""
    return ("#" + ", ".join(names) + ", mu, " + ", ".join(f"dmu/d{n}" for n in names)
            + ", sigma_hat, sigma_star, case, gap, time, note")


class GradRequest:
    """Set as ``Loss.grad_request``: the next ``Loss_EnergyGeometry`` evaluation also
    computes the gradient (``result``).  Needs engine='pydftb' and torch."""

    def __init__(self, steps=None):
        from utils.hotcent_tangent import DEFAULT_STEPS
        self.steps = tuple(steps or DEFAULT_STEPS)
        self.result: GradResult | None = None
        self.grad_pass = None
        self._tangents = None
        self._skf = None
        self._failure = ""
        self._t0 = 0.0
        self.timings = {}
        self._names = None                   # parameters per type (M5b): dimension names

    @staticmethod
    def directory_name(r0_w: float, r0_d: float) -> str:
        """The evaluation directory: more digits than BO's opt_<.3f>, the steps are small."""
        return f"opt_{r0_w:.6f}_{r0_d:.6f}"

    # ---- hooks ---------------------------------------------------------------
    def after_hotcent(self, loss, conf_parameters) -> None:
        """Tangents at (r0_w, r0_d) and the pass-1 work of the DFTB stage (cwd: the
        evaluation directory with the step's SKF files)."""
        import utils.engines as engines
        from utils import dftb_stage, hotcent_tangent
        from utils.parallel import current_plan
        typed = getattr(loss, "typed", False)                    # M5b: x of every set
        if typed:
            self._names = loss.types.dimension_names()
        else:
            r0_w, r0_d, p = conf_parameters
        self.result, self.timings = None, {}
        self._t0 = time.perf_counter()
        self._skf = os.path.abspath("grad_skf")                 # physrep rewrites the files
        os.makedirs(self._skf, exist_ok=True)
        for name in os.listdir("."):
            if name.endswith(".skf"):
                shutil.copy2(name, os.path.join(self._skf, name))
        labels = sorted({s for a in loss.dft_input for s in a.get_chemical_symbols()})
        sup = loss.args_input.superposition
        hp = {"xc": loss.args_input.xc, "superposition": sup, "opt_elem": loss.opt_elem,
              "label": "ptbp"}
        plan = current_plan()
        workers = plan.stage("hotcent").workers if plan is not None else 1
        try:
            from hotcent import __version__ as version
        except ImportError:                          # hotcent missing: the tangent fails below
            version = None
        from utils.hotpy import confinement
        key = (version, confinement(), tuple(labels), loss.args_input.xc, sup,
               tuple(loss.opt_elem), str(loss.args_input.multi_element))
        self._failure, self.grad_pass = "", None
        try:
            if typed:
                if loss.types.switches:
                    raise GradientUnavailable("no gradient with type switching (pydftb.autodiff "
                                              "has no mixed blocks)")
                self._tangents = hotcent_tangent.compute_types(
                    loss.types, conf_parameters, hp, os.getcwd(), steps=self.steps,
                    workers=workers, probe_key=key + (tuple(loss.types.labels),))
            else:
                self._tangents = hotcent_tangent.compute(
                    lambda w, d: loss.set_parameters([w, d, p]), hp, os.getcwd(), labels, r0_w,
                    r0_d, steps=self.steps, workers=workers, probe_key=key)
        except Exception as err:                    # noqa: BLE001  the loss goes on without
            self._failure = f"no hotcent tangent: {type(err).__name__}: {err}"
        else:
            smearing = engines.settings()["smearing"]
            self.grad_pass = dftb_stage.GradPass(
                self._tangents, mode="hf" if smearing == "none" else "implicit")
        self.timings["tangent"] = time.perf_counter() - self._t0

    def after_step(self, loss, conf_parameters, sigma_hat, mu_hat) -> None:
        """After the sigma search: sigma*, pass 2 (forces), E0 and the gradient."""
        t0 = time.perf_counter()
        zeros = {n: t.zero for n, t in (self._tangents or {}).items()}
        if getattr(loss, "typed", False):                      # M5b
            res = GradResult(float("nan"), float("nan"), float(mu_hat), None,
                             sigma_hat=[float(s) for s in np.atleast_1d(sigma_hat)],
                             tangent_zero=zeros, x=[float(v) for v in conf_parameters],
                             names=loss.types.dimension_names())
        else:
            r0_w, r0_d, _ = conf_parameters
            res = GradResult(float(r0_w), float(r0_d), float(mu_hat), None,
                             sigma_hat=float(np.atleast_1d(sigma_hat)[0]), tangent_zero=zeros)
        try:
            self._finish(loss, res)
        except GradientUnavailable as err:
            res.grad, res.reason = None, str(err)
        except Exception as err:                    # noqa: BLE001  pydftb.autodiff, SCC ...
            res.grad, res.reason = None, f"{type(err).__name__}: {err}"
        finally:                                    # sigma and forces are parts of this
            rest = time.perf_counter() - t0 - sum(self.timings.get(k, 0.0)
                                                  for k in ("sigma", "forces"))
            res.timings = dict(self.timings, other=rest)
            self.result = res
            self._cleanup()

    def fail(self, conf_parameters, mu, reason: str) -> None:
        if len(conf_parameters) == 3:
            r0_w, r0_d, _ = conf_parameters
            self.result = GradResult(float(r0_w), float(r0_d), float(mu), None,
                                     reason=f"the loss failed: {reason}", timings=dict(self.timings))
        else:                                                  # M5b: x of every set
            self.result = GradResult(float("nan"), float("nan"), float(mu), None,
                                     reason=f"the loss failed: {reason}", timings=dict(self.timings),
                                     x=[float(v) for v in conf_parameters], names=self._names)
        self._cleanup()

    def _cleanup(self):
        if self._skf:
            shutil.rmtree(self._skf, ignore_errors=True)
        self._skf, self.grad_pass = None, None

    # ---- the gradient ----------------------------------------------------------
    def _finish(self, loss, res: GradResult) -> None:
        from utils import dftb_stage
        gp = self.grad_pass
        if gp is None:
            raise GradientUnavailable(self._failure or "no tangent (the hotcent hook did not run)")
        if res.x is not None:                                  # M5b
            return self._finish_typed(loss, res)
        step = StepData.from_files(loss, res.r0_w, res.r0_d)
        with open("search_index.txt") as fh:
            kept = [int(i) for i in fh.read().strip().split(",")]
        if self._failure:
            raise GradientUnavailable(self._failure)
        missing = [i for i in kept if gp.out.get(i) is None]
        if missing:
            why = gp.errors.get(missing[0], "")
            raise GradientUnavailable(f"no energy gradient for structures {missing[:5]}"
                                      + (f" ({why})" if why else ""))
        t0 = time.perf_counter()
        inner = refine_sigma(step, res.sigma_hat)
        self.timings["sigma"] = time.perf_counter() - t0
        res.sigma_star, res.case = inner.sigma, inner.case
        res.note = "; ".join(x for x in [inner.note] + inner.terms.notes if x)
        res.gap = float(mu(step, res.sigma_hat) - mu(step, inner.sigma))
        names = list(gp.tangents)
        edot = {n: np.array([gp.out[i][n] for i in kept]) for n in names}
        e0dot = fdot = kink_fdot = None
        t = inner.terms
        if step.mode in ("dataset", "reaction"):
            t0 = time.perf_counter()
            e0dot = self._e0_gradient(loss, t.elements)
            weights = force_weights(t, exclude=inner.kinks)
            gp.weights = {kept[i]: wv for i, wv in weights.items() if np.any(wv)}
            gp.q_init = gp.q_in
            out = dftb_stage.run_force_gradient(loss.dft_input, loss.bolt_f11,
                                                loss.args_input.kpt_density, gp, self._skf)
            fdot = {n: float(sum(out[i][n] for i in sorted(out))) for n in names}
            if inner.case == "kink" and t.kind[inner.kinks[0]] == "F":
                i, atom, comp = t.force[inner.kinks[0]]
                unit = np.zeros((t.natoms[i], 3))
                unit[atom, comp] = 1.0
                gp.weights = {kept[i]: unit}
                one = dftb_stage.run_force_gradient(loss.dft_input, loss.bolt_f11,
                                                    loss.args_input.kpt_density, gp, self._skf)
                kink_fdot = {n: one[kept[i]][n] for n in names}
            self.timings["forces"] = time.perf_counter() - t0
        res.grad = assemble(step, inner, edot, e0dot, fdot, kink_fdot, parts=res.parts)

    def _finish_typed(self, loss, res: GradResult) -> None:
        """The gradient for parameters per type (M5b): sigma* of the vector, the kink
        multipliers, one pass 2 for the force terms and one for the force kinks."""
        from utils import dftb_stage
        gp = self.grad_pass
        step = StepData.from_files(loss, x=res.x)
        with open("search_index.txt") as fh:
            kept = [int(i) for i in fh.read().strip().split(",")]
        if self._failure:
            raise GradientUnavailable(self._failure)
        missing = [i for i in kept if gp.out.get(i) is None]
        if missing:
            why = gp.errors.get(missing[0], "")
            raise GradientUnavailable(f"no energy gradient for structures {missing[:5]}"
                                      + (f" ({why})" if why else ""))
        t0 = time.perf_counter()
        inner = refine_sigma_typed(step, res.sigma_hat)
        self.timings["sigma"] = time.perf_counter() - t0
        res.sigma_star, res.case = [float(s) for s in inner.sigma], inner.case
        res.note = "; ".join(x for x in [inner.note] + inner.terms.notes if x)
        res.gap = float(mu(step, res.sigma_hat) - mu(step, inner.sigma))
        names = step.model.dimension_names()
        edot = {n: np.array([gp.out[i][n] for i in kept]) for n in names}
        e0dot = fdot = kink_fdot = None
        t = inner.terms
        if step.mode in ("dataset", "reaction"):
            t0 = time.perf_counter()
            e0dot = self._e0_gradient(loss, t.elements)
            weights = force_weights(t, exclude=inner.kinks)
            gp.weights = {kept[i]: wv for i, wv in weights.items() if np.any(wv)}
            gp.q_init = gp.q_in
            out = dftb_stage.run_force_gradient(loss.dft_input, loss.bolt_f11,
                                                loss.args_input.kpt_density, gp, self._skf)
            fdot = {n: float(sum(out[i][n] for i in sorted(out))) for n in names}
            fk = [(j, k) for j, k in enumerate(inner.kinks) if t.kind[k] == "F"]
            if fk:                                     # sum_k lambda_k Fdot_k in one pass
                lam = kink_multipliers(inner)
                kw = {}
                for j, k in fk:
                    i, atom, comp = t.force[k]
                    arr = kw.setdefault(i, np.zeros((t.natoms[i], 3)))
                    arr[atom, comp] += lam[j]
                gp.weights = {kept[i]: wv for i, wv in kw.items() if np.any(wv)}
                one = dftb_stage.run_force_gradient(loss.dft_input, loss.bolt_f11,
                                                    loss.args_input.kpt_density, gp, self._skf)
                kink_fdot = {n: float(sum(one[i][n] for i in sorted(one))) for n in names}
            self.timings["forces"] = time.perf_counter() - t0
        res.grad = assemble_typed(step, inner, edot, e0dot, fdot, kink_fdot, parts=res.parts)

    def _e0_gradient(self, loss, elements) -> dict:
        """{name: {Z: dE0/dr0}} of the isolated atoms as Loss.get_atomic_energy sets them
        up (20 A box, Gamma point, the step's SKF files)."""
        import ase
        import torch
        from ase.data import chemical_symbols
        from ase.units import Hartree

        from utils.calculator import dftbplus_calculator
        from utils.dftb_stage import _grad_dict, _leaves, torch_one_thread
        from utils.hotcent_tangent import contract
        active = self.grad_pass.active()
        out = {n: {z: 0.0 for z in elements} for n in self.grad_pass.tangents}
        if not active:
            return out
        torch_one_thread()
        args = {'kpts': [1, 1, 1], 'poly_rep': 'No', 'SCC': 'Yes',
                'slater_p': os.path.join(self._skf, ""), 'calc_type': 'static'}
        for z in elements:
            atoms = ase.Atoms(chemical_symbols[int(z)], positions=[[0, 0, 0]])
            atoms.set_cell(np.eye(3) * 20.0)
            atoms.set_pbc([True, True, True])
            calc = dftbplus_calculator(atoms, args)
            r, params = calc.evaluate(atoms, requires_grad=self.grad_pass.kinds(),
                                      mode=self.grad_pass.mode)
            if not r.energy.requires_grad:
                continue
            gd = _grad_dict(params, torch.autograd.grad(r.energy, _leaves(params),
                                                        allow_unused=True))
            for n, t in active.items():
                out[n][z] = contract(gd, t) * Hartree
        return out
