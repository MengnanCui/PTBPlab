"""DFTB parameters as torch tensors (milestone M4b).

:class:`Params` holds, for one :class:`pydftb.skf.SKSet`, the numbers a calculation
depends on as torch tensors, any of which may require gradients:

* ``tables[(a, b)]`` -- the stored columns of the ``a-b.skf`` table (H and S integrals);
* ``onsite[a]``, ``hubbard[a]`` -- per shell (s, p, d, f up to the label's max_l), as in
  ``SKSet.onsite`` / ``SKSet.hubbard`` (atom-resolved SCC uses the first entry);
* ``repulsive[(a, b)]`` -- the coefficients of the pair repulsive the engine uses
  (``{"coeffs", "exp_coeffs"}`` of the spline, or ``{"poly"}``);
* ``hubbard_derivs[a]`` (DFTB3) and ``spin_constants[a]`` when the engine uses them.

:meth:`Params.to_skset` gives a host copy of the set with the current values, which the
NumPy (or torch) engine converges the SCC with.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import torch

from ..skf import SKSet

KINDS = ("tables", "onsite", "hubbard", "repulsive", "hubbard_derivs", "spin_constants")


@dataclasses.dataclass
class Params:
    sk: SKSet
    tables: dict
    onsite: dict
    hubbard: dict
    repulsive: dict
    repulsive_mode: str = "skf"
    hubbard_derivs: dict | None = None
    spin_constants: dict | None = None

    @classmethod
    def from_skset(cls, sk: SKSet, repulsive: str = "skf", hubbard_derivs=None,
                   spin_constants=None, requires_grad=(), dtype=torch.float64,
                   device="cpu") -> "Params":
        """Tensors from ``sk`` (and the engine's DFTB3 / spin settings); ``requires_grad``:
        names from :data:`KINDS` whose tensors are leaves that require gradients."""
        def t(a):
            return torch.as_tensor(np.array(a, dtype=float), dtype=dtype, device=device)

        tables = {k: t(f.table) for k, f in sk.files.items()}
        reps = {}
        for k in sk.files:
            rep = sk.repulsive(*k, repulsive)
            if rep is None:
                continue
            if rep.poly is not None:
                reps[k] = {"poly": t(rep.poly)}
            elif rep.knots is not None:
                reps[k] = {"coeffs": t(rep.coeffs), "exp_coeffs": t(rep.exp_coeffs)}
        p = cls(sk, tables, {a: t(v) for a, v in sk.onsite.items()},
                {a: t(v) for a, v in sk.hubbard.items()}, reps, repulsive,
                None if hubbard_derivs is None else
                {a: t(np.atleast_1d(v)) for a, v in hubbard_derivs.items()},
                None if spin_constants is None else
                {a: t(np.atleast_1d(v)) for a, v in spin_constants.items()})
        p.requires_grad_(*requires_grad)
        return p

    # ------------------------------------------------------------------
    def groups(self) -> dict:
        """Name -> list of the tensors of that kind."""
        out = {"tables": list(self.tables.values()), "onsite": list(self.onsite.values()),
               "hubbard": list(self.hubbard.values()),
               "repulsive": [v for d in self.repulsive.values() for v in d.values()]}
        out["hubbard_derivs"] = list((self.hubbard_derivs or {}).values())
        out["spin_constants"] = list((self.spin_constants or {}).values())
        return out

    def requires_grad_(self, *names) -> "Params":
        for name in names:
            if name not in KINDS:
                raise ValueError(f"unknown parameter kind {name!r} (one of {KINDS})")
            for x in self.groups()[name]:
                x.requires_grad_(True)
        return self

    def leaves(self) -> list:
        """The tensors that require gradients."""
        return [x for xs in self.groups().values() for x in xs if x.requires_grad]

    # ------------------------------------------------------------------
    def to_skset(self) -> SKSet:
        """A host copy of the SK set with the current (detached) values.  The SKF files
        themselves are never modified (they are shared through the parsing cache)."""
        def h(x):
            return x.detach().cpu().numpy().astype(float)

        sk = copy.copy(self.sk)
        files = {}
        for k, f in self.sk.files.items():
            rep, poly_rep = f.repulsive, f.poly_repulsive
            values = self.repulsive.get(k)
            if values is not None:
                if "poly" in values:
                    poly_rep = dataclasses.replace(poly_rep, poly=h(values["poly"]))
                    if f.repulsive is f.poly_repulsive:
                        rep = poly_rep
                else:
                    rep = dataclasses.replace(rep, coeffs=h(values["coeffs"]),
                                              exp_coeffs=h(values["exp_coeffs"]))
            files[k] = dataclasses.replace(f, table=h(self.tables[k]), repulsive=rep,
                                           poly_repulsive=poly_rep, _coef=None)
        sk.files = files
        sk.onsite = {a: h(v) for a, v in self.onsite.items()}
        sk.hubbard = {a: h(v) for a, v in self.hubbard.items()}
        return sk

    def host_hubbard_derivs(self):
        if self.hubbard_derivs is None:
            return None
        return {a: v.detach().cpu().numpy().astype(float) for a, v in self.hubbard_derivs.items()}

    def host_spin_constants(self):
        if self.spin_constants is None:
            return None
        return {a: v.detach().cpu().numpy().astype(float) for a, v in self.spin_constants.items()}
