"""Derivatives of :mod:`pydftb.autodiff` against finite differences of the engine (M4b, G3).

    python -m pydftb.validation.gradcheck --check               # the standard set; exit 1 on failure
    python -m pydftb.validation.gradcheck h2o --kinds tables hubbard --mode implicit

Every case is a frozen reference structure (``refgen``) converged to SCC tolerance 1e-13.
For each parameter kind (SK tables, on-site energies, Hubbard U and derivatives, spin
constants, repulsive coefficients) a random direction t is drawn and the autograd
derivative grad . t is compared with the central difference (X(theta + h t) -
X(theta - h t)) / 2h of the NumPy engine, extrapolated from h and h/2 (Richardson): with f
states near the Fermi level the h^2 error of one difference is 3e-3 at h = 1e-4 (GdO).
The spread of the extrapolations from (2h, h) and (h, h/2) is the difference's error bar.

* mode "hf": X = F, the engine run with the smearing's ``fermi_level="exact"`` and
  ``ts_convention="variational"`` (the fixed-N Mermin free energy of mode "hf");
  tolerance 1e-7;
* mode "implicit": X = E, F, random weighted sums of the charges, eigenvalues and forces
  (and the stress), and the Fermi levels, with DFTB+'s conventions; tolerance 1e-6.

Added to the tolerance are that error bar and a floor for the noise of the difference:
the round-off of X (~30 eps |X|) and,
but for the variational F, the SCC tolerance (one error of ~1e-13 per charge site,
eigenvalue, force component ... of X), both divided by h.  The directions are scaled by
the magnitude of the values; they are symmetric for the spin constants W, and values that
are one number in DFTB+ move together: the integrals with l1 = l2 (ss, pp, dd, ff) that the
files A-B and B-A both store (each matrix triangle uses one copy), and Hubbard values that
are equal (DFTB+'s formulas branch on |U_a - U_b| < MIN_HUB_DIFF, so only such derivatives
exist there).
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time

import numpy as np
import torch

from ..autodiff import KINDS, Params, evaluate
from ..autodiff.api import _engine_for
from ..constants import BOHR, MIN_HUB_DIFF
from ..occupations import Smearing
from ..skf import N_INTEGRALS, SK_COLUMN
from . import refgen
from .cases import CASES_BY_NAME

# the standard set: molecules, metals at T ~ 0 and with smearing, k-points, spin, DFTB3,
# shell-resolved SCC, H-X damping, f orbitals, dispersion
CHECK_CASES = ["h2o", "ch3oh_dftb3", "water_dimer_hdamp", "cu4co_shell_resolved",
               "o2_triplet_wmatrix", "cu8_distorted_k355", "cu8_distorted_k355_mp2_2000K",
               "cuo_chain_spin_k422", "li_bcc_k666_1000K", "gdo_rocksalt_k444",
               "water_box_uff"]
STEP = 2e-5          # relative finite-difference step (with 2 STEP and STEP / 2)
SCC_TOL = 1e-13      # of the reference runs (the tightest all cases reach)
# pass: |ad - fd| <= TOL max(|ad|, |fd|) + noise of the difference, with the noise from the
# round-off of the values (~NOISE eps |X|) and, for all but the variational F, from the SCC
# tolerance (an error ~SCC_TOL per charge site, eigenvalue, force component ... in X)
TOL = {"hf": 1e-7, "implicit": 1e-6}
NOISE = 30.0
# full-layout columns of the l1 = l2 integrals (H and S), stored in both A-B and B-A
SHARED_COLUMNS = {c + o for (l1, l2, _), c in SK_COLUMN.items() if l1 == l2
                  for o in (0, N_INTEGRALS)}


def entries(params: Params, kind: str) -> list:
    """(key, tensor) of one parameter kind; repulsive keys are (pair, coefficient name)."""
    if kind == "repulsive":
        return [((k, n), v) for k, d in params.repulsive.items() for n, v in d.items()]
    return list((getattr(params, kind) or {}).items())


def tangent(params: Params, kind: str, rng) -> dict:
    """A random direction for the tensors of ``kind`` (key -> NumPy array)."""
    out = {}
    ent = entries(params, kind)
    if kind == "hubbard":
        vals = np.concatenate([v.detach().cpu().numpy() for _, v in ent]) if ent else []
        reps = []                                # one random number per group of equal U
        for x in np.sort(vals):
            if not reps or x - reps[-1][0] >= MIN_HUB_DIFF:
                reps.append((x, rng.standard_normal()))
        for key, v in ent:
            a = v.detach().cpu().numpy()
            r = [next(z for x, z in reps if abs(x - u) < MIN_HUB_DIFF * (1 + 1e-9)) for u in a]
            out[key] = np.array(r) * np.abs(a)
        return out
    for key, v in ent:
        a = v.detach().cpu().numpy()
        r = rng.standard_normal(a.shape)
        if kind == "spin_constants" and a.size > 1:
            n = int(round(np.sqrt(a.size)))
            r = r.reshape(n, n)
            r = (0.5 * (r + r.T)).reshape(a.shape)
        out[key] = r * np.abs(a)
    if kind == "tables":
        _tie_shared(params, out)
    return out


def _tie_shared(params: Params, t: dict):
    """Give the l1 = l2 integrals of B-A the direction of the same integrals of A-B."""
    files = params.sk.files
    for (a, b) in t:
        fab, fba = files[(a, b)], files[(b, a)]
        if a >= b or fab.dr != fba.dr:
            continue
        n = min(fab.ngrid, fba.ngrid)
        for c in SHARED_COLUMNS:
            kab, kba = fab.colmap[c], fba.colmap[c]
            if kab >= 0 and kba >= 0:
                t[(b, a)][:n, kba] = t[(a, b)][:n, kab]


def shifted(params: Params, kind: str, t: dict, h: float) -> Params:
    """A copy of ``params`` (detached) with the tensors of ``kind`` moved by h t."""
    def det(d):
        return None if d is None else {k: v.detach().clone() for k, v in d.items()}

    p = Params(params.sk, det(params.tables), det(params.onsite), det(params.hubbard),
               {k: det(d) for k, d in params.repulsive.items()}, params.repulsive_mode,
               det(params.hubbard_derivs), det(params.spin_constants))
    for key, v in entries(p, kind):
        new = v + h * torch.as_tensor(t[key], dtype=v.dtype, device=v.device)
        if kind == "repulsive":
            p.repulsive[key[0]][key[1]] = new
        else:
            getattr(p, kind)[key] = new
    return p


def available(calc) -> list:
    """The parameter kinds a calculation depends on."""
    kinds = ["tables", "onsite"]
    if calc.repulsive != "none":
        kinds.append("repulsive")
    if calc.scc:
        kinds.append("hubbard")
    if calc.third_order:
        kinds.append("hubbard_derivs")
    if calc.spin:
        kinds.append("spin_constants")
    return kinds


def _structure(inp):
    pos = np.array(inp["positions_angstrom"]) / BOHR
    periodic = inp["cell_angstrom"] is not None
    cell = np.array(inp["cell_angstrom"]) / BOHR if periodic else None
    return inp["labels"], pos, cell, (np.ones(3, bool) if periodic else None)


def reference_engine(name: str, variational: bool = True, **backend):
    """The case's calculator at SCC tolerance SCC_TOL; ``variational``: with the smearing's
    ``fermi_level="exact"`` and ``ts_convention="variational"`` (F is then the fixed-N
    Mermin free energy of mode "hf"), else with DFTB+'s conventions (mode "implicit"
    differentiates them).  ``backend``: ``backend``/``device``/``dtype`` of the engine."""
    doc = refgen.load(name)
    case = CASES_BY_NAME[doc["case"]]
    sm = Smearing.from_spec(tuple(case.filling))
    if variational:
        sm = dataclasses.replace(sm, ts_convention="variational", fermi_level="exact")
    _, calc = refgen.run_pydftb(case, doc["input"], engine=True, scc_tol=SCC_TOL, smearing=sm,
                                **backend)
    return calc, _structure(doc["input"])


def _richardson(fn, step):
    """Central differences D of ``fn(h)`` (a dict of numbers) at 2 step, step and step / 2,
    extrapolated in pairs, R(h) = (4 D(h/2) - D(h)) / 3: returns R(step) and, as its error
    bar, |R(2 step) - R(step)| (the truncation error grows with h, the noise as 1/h)."""
    d = []
    for h in (2.0 * step, step, 0.5 * step):
        fp, fm = fn(h), fn(-h)
        d.append({k: (fp[k] - fm[k]) / (2 * h) for k in fp})
    r_big = {k: (4.0 * d[1][k] - d[0][k]) / 3.0 for k in d[0]}
    r = {k: (4.0 * d[2][k] - d[1][k]) / 3.0 for k in d[0]}
    return r, {k: abs(r_big[k] - r[k]) for k in r}


def _quantities(energy, free, charges, eigenvalues, fermi, forces, stress, w):
    """The scalars compared in mode "implicit" (NumPy arrays or torch tensors)."""
    out = {"energy": energy, "free_energy": free, "charges": (w["q"] * charges).sum(),
           "eigenvalues": (w["e"] * eigenvalues).sum(), "fermi": fermi.sum(),
           "forces": (w["f"] * forces).sum()}
    if stress is not None:
        out["stress"] = (w["s"] * stress).sum()
    return out


def _engine_quantities(eng, species, pos, cell, pbc, w):
    res, st, _ = eng._run(species, pos, cell, pbc, True, None, cell is not None)
    fermi = _fermi_levels(res, st)
    return _quantities(res.energy, res.free_energy, res.charges, np.array(res.eigenvalues),
                       fermi, res.forces, res.stress, w)


def check_case(name: str, kinds=None, step: float = STEP, seed: int = 0,
               mode: str = "hf") -> list:
    """Rows (case, mode, kind, quantity, autograd, finite difference, round-off of the
    difference) for one case.  Mode "hf": dF/dtheta; mode "implicit": E, F, charges,
    eigenvalues, Fermi levels, forces and stress (random weighted sums)."""
    calc, (species, pos, cell, pbc) = reference_engine(name, variational=mode == "hf")
    params = Params.from_skset(calc.sk, calc.repulsive, calc.hubbard_derivs,
                               calc.spin_constants)
    kinds = [k for k in (kinds or KINDS) if k in available(calc) and entries(params, k)]
    params.requires_grad_(*kinds)
    implicit = mode == "implicit"
    res = evaluate(calc, species, pos, cell, pbc, params, mode=mode, forces=implicit)
    rng = np.random.default_rng(seed)
    if implicit:
        w = {"q": rng.normal(size=res.charges.shape), "e": rng.normal(size=res.eigenvalues.shape),
             "f": rng.normal(size=res.forces.shape), "s": rng.normal(size=(3, 3))}
        ad_q = _quantities(res.energy, res.free_energy, res.charges, res.eigenvalues, res.fermi,
                           res.forces, res.stress, {k: torch.as_tensor(v) for k, v in w.items()})
        ref = _engine_quantities(calc, species, pos, cell, pbc, w)
        scale = {"energy": abs(ref["energy"]), "free_energy": abs(ref["free_energy"]),
                 "charges": np.abs(w["q"] * res.raw.charges).sum(),
                 "eigenvalues": np.abs(w["e"] * np.array(res.raw.eigenvalues)).sum(),
                 "fermi": np.abs(res.fermi.detach().numpy()).sum(),
                 "forces": np.abs(w["f"]).sum(), "stress": np.abs(w["s"]).sum()}
        nsite = len(res.raw.q_in)
        terms = {"energy": nsite, "free_energy": 0.0, "charges": np.abs(w["q"]).sum(),
                 "eigenvalues": np.abs(w["e"]).sum(), "fermi": len(res.fermi),
                 "forces": np.abs(w["f"]).sum(), "stress": np.abs(w["s"]).sum()}
    else:
        ad_q = {"free_energy": res.free_energy}
        scale = {"free_energy": abs(res.free_energy.item())}
        terms = {"free_energy": 0.0}
    rows = []
    for kind in kinds:
        ent = entries(params, kind)
        t = tangent(params, kind, rng)

        def fd_fn(h):
            eng = _engine_for(calc, shifted(params, kind, t, h))
            if implicit:
                return _engine_quantities(eng, species, pos, cell, pbc, w)
            return {"free_energy": eng._run(species, pos, cell, pbc, False, None, False)[0].free_energy}

        fd, spread = _richardson(fd_fn, step)
        for qty, val in ad_q.items():
            grads = torch.autograd.grad(val, [v for _, v in ent], retain_graph=True,
                                        allow_unused=True)
            ad = sum(float((g * torch.as_tensor(t[k])).sum()) for (k, _), g in zip(ent, grads)
                     if g is not None)
            noise = (NOISE * np.finfo(float).eps * max(scale[qty], 1.0)
                     + SCC_TOL * terms[qty]) / step + spread[qty]
            rows.append((name, mode, kind, qty, ad, fd[qty], noise))
    return rows


def _fermi_levels(res, st):
    from ..autodiff.api import _fermi_levels as fl
    return fl(res, st)


def passes(row, tol: dict = TOL) -> bool:
    ad, fd, noise = row[-3:]
    return abs(ad - fd) <= tol[row[1]] * max(abs(ad), abs(fd)) + noise


def check(names=None, kinds=None, step: float = STEP, tol: dict = TOL,
          verbose: bool = True, modes=("hf", "implicit")) -> bool:
    ok = True
    for mode in modes:
        for name in names or CHECK_CASES:
            why = refgen.missing_requirement(refgen.load(name))
            if why:
                print(f"{name:32s} skipped: {why}")
                continue
            t0 = time.perf_counter()
            for row in check_case(name, kinds, step, mode=mode):
                good = passes(row, tol)
                ok &= good
                if verbose or not good:
                    _, _, kind, qty, ad, fd, noise = row
                    rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-300)
                    print(f"{name:30s} {mode:8s} {kind:14s} {qty:11s} ad {ad: .9e}  "
                          f"fd {fd: .9e}  rel {rel:.1e}  (round-off {noise:.0e})"
                          f"{'' if good else '  FAIL'}")
            if verbose:
                print(f"{'':30s} ({time.perf_counter() - t0:.1f} s)")
    print("gradcheck:", "PASS" if ok else "FAIL")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help="case names (default: the standard set)")
    ap.add_argument("--check", action="store_true", help="exit 1 if any check fails")
    ap.add_argument("--kinds", nargs="*", choices=KINDS, help="parameter kinds (default: all)")
    ap.add_argument("--mode", choices=("hf", "implicit", "both"), default="both")
    ap.add_argument("--step", type=float, default=STEP, help="relative finite-difference step")
    a = ap.parse_args(argv)
    modes = ("hf", "implicit") if a.mode == "both" else (a.mode,)
    ok = check(a.cases, a.kinds, a.step, modes=modes)
    if a.check:
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
