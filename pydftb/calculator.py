"""Self-consistent-charge DFTB (DFTB2) driver: energies, charges, forces.

Reference implementation: clarity first, dense linear algebra, NumPy/SciPy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .coulomb import Coulomb
from .constants import KB_HARTREE
from .hamiltonian import Basis, build_pairs, fold, fold_all, onsite_energies, shift_index
from .mixer import AndersonMixer
from .solvers import SOLVERS
from .skf import SKSet


@dataclass
class DFTBResult:
    energy: float                 # total energy E (Hartree)
    free_energy: float            # Mermin free energy E - TS
    e_band: float
    e_scc: float
    e_rep: float
    ts: float
    fermi: float
    charges: np.ndarray           # Mulliken electron populations per atom
    eigenvalues: list
    forces: np.ndarray | None = None
    n_scc: int = 0
    converged: bool = True
    timings: dict = field(default_factory=dict)


class DFTB:
    """SCC-DFTB calculator.

    Parameters
    ----------
    sk : SKSet            Slater-Koster data.
    scc : bool            Self-consistent charges.
    kpts : (nk,3), (nk,)  k-points in fractional reciprocal coords and weights
                          (weights normalised to 1).  ``None`` => Gamma.
    temperature : float   Electronic temperature (K) for Fermi smearing.
    solver : str | object "eigh" (reference), "batched" or "sp2" (see solvers.py).
    """

    def __init__(self, sk: SKSet, scc: bool = True, kpts=None, temperature: float = 0.0,
                 scc_tol: float = 1e-8, max_scc: int = 200, mixing: float = 0.2,
                 charge: float = 0.0, solver: str = "eigh", verbose: bool = False):
        self.sk = sk
        self.scc = scc
        self.kpts = kpts
        self.kt = temperature * KB_HARTREE
        self.scc_tol = scc_tol
        self.max_scc = max_scc
        self.mixing = mixing
        self.charge = charge
        self.solver = solver
        self.verbose = verbose

    # ------------------------------------------------------------------
    def _sites(self, basis: Basis):
        """Charge sites: atoms, or shells when orbital-resolved."""
        sk = self.sk
        if sk.shell_resolved:
            site_of_orb = basis.orb_shell
            site_atom = basis.shell_atom
            u = np.array([sk.hubbard[basis.species[a]][l]
                          for a, l in zip(basis.shell_atom, basis.shell_l)])
            q0 = np.array([sk.occupation[basis.species[a]][l]
                           for a, l in zip(basis.shell_atom, basis.shell_l)])
        else:
            site_of_orb = basis.orb_atom
            site_atom = np.arange(len(basis.species))
            u = np.array([sk.hubbard[s][0] for s in basis.species])
            q0 = np.array([sk.occupation[s].sum() for s in basis.species])
        return site_of_orb, site_atom, u, q0

    def calculate(self, species, positions, cell=None, pbc=None, forces=True,
                  q_init=None) -> DFTBResult:
        """Run a calculation. Positions/cell in Bohr."""
        t = {}
        t0 = time.perf_counter()
        sk = self.sk
        pos = np.asarray(positions, float)
        if cell is not None:
            cell = np.asarray(cell, float)
            pbc = np.ones(3, bool) if pbc is None else np.asarray(pbc, bool)
        basis = Basis.build(species, sk)
        norb = basis.norb
        pb = build_pairs(species, pos, cell, pbc, sk, basis, derivatives=forces)
        eps0 = onsite_energies(basis, sk)
        site_of_orb, site_atom, hub, q0 = self._sites(basis)
        nelec = q0.sum() - self.charge
        periodic = cell is not None and np.any(pbc)
        if periodic and self.kpts is not None:
            kpts, kw = self.kpts
            kpts = np.atleast_2d(kpts)
            kw = np.asarray(kw, float) / np.sum(kw)
        else:
            kpts, kw = np.zeros((1, 3)), np.ones(1)
        gamma_only = len(kpts) == 1 and not np.any(kpts[0])
        if gamma_only:
            h0k = [fold(pb, pb.h, norb, None, eps0)]
            sk_ = [fold(pb, pb.s, norb, None, np.ones(norb))]
        else:
            h0k = fold_all(pb, pb.h, norb, kpts, eps0)
            sk_ = fold_all(pb, pb.s, norb, kpts, np.ones(norb))
        t["build"] = time.perf_counter() - t0

        coul = None
        if self.scc:
            t1 = time.perf_counter()
            coul = Coulomb(pos[site_atom], site_atom, hub,
                           cell if periodic else None, pbc if periodic else None)
            t["gamma"] = time.perf_counter() - t1

        solver = self.solver if not isinstance(self.solver, str) else SOLVERS[self.solver]()
        h0 = np.array(h0k)
        smat = np.array(sk_)
        t1 = time.perf_counter()
        solver.prepare(smat)
        t_diag = time.perf_counter() - t1
        mixer = AndersonMixer(self.mixing)
        q_in = q0.copy() if q_init is None else np.asarray(q_init, float).copy()
        converged = not self.scc
        nit = 0
        for nit in range(1, (self.max_scc if self.scc else 1) + 1):
            if self.scc:
                _, v_site = coul.energy_shift(q_in - q0)
            else:
                v_site = np.zeros(len(q0))
            v_orb = v_site[site_of_orb]
            h = h0 + 0.5 * smat * (v_orb[:, None] + v_orb[None, :])
            t1 = time.perf_counter()
            rho, wdm, evals, mu, ts = solver.solve(h, kw, nelec, self.kt, need_w=False)
            t_diag += time.perf_counter() - t1
            # Mulliken: q_mu = sum_k w_k Re (rho_k S_k)_{mu mu}
            q_orb = np.einsum("k,kij,kji->i", kw, rho, smat).real
            q_out = np.bincount(site_of_orb, q_orb, minlength=len(q0))
            if not self.scc:
                break
            diff = np.abs(q_out - q_in).max()
            if self.verbose:
                print(f"SCC {nit:3d}  max dq = {diff:.3e}")
            if diff < self.scc_tol:
                converged = True
                break
            q_in = mixer.mix(q_in, q_out)
        if forces:
            # W is only needed once, for the final density
            t1 = time.perf_counter()
            rho, wdm, evals, mu, ts = solver.solve(h, kw, nelec, self.kt, need_w=True)
            t_diag += time.perf_counter() - t1
        t["diag"] = t_diag
        t["scc_iter"] = nit

        # ---------------- energies ----------------
        e_band = float(np.einsum("k,kij,kji->", kw, rho, h0).real)   # Tr(rho H0)
        dq = q_out - q0
        e_scc = coul.energy_shift(dq)[0] if self.scc else 0.0
        e_rep, rep_grad = self._repulsive(species, pos, cell, pbc, forces)
        energy = e_band + e_scc + e_rep
        q_atom = np.bincount(site_atom, q_out, minlength=len(species))
        res = DFTBResult(energy, energy - ts, e_band, e_scc, e_rep, ts, mu, q_atom,
                         None if evals is None else list(evals), n_scc=nit, converged=converged)

        # ---------------- forces ----------------
        if forces:
            t1 = time.perf_counter()
            grad = rep_grad.copy()
            # real-space (energy-weighted) density at the pair elements:
            # P_e = Re sum_k w_k rho_k[nu, mu] exp(2 pi i k.T)
            if gamma_only:
                p_e = rho[0][pb.cols, pb.rows].real
                w_e = wdm[0][pb.cols, pb.rows].real
            else:
                # group by translation T: R_T = sum_k w_k e^{2 pi i k.T} rho_k  (nT GEMM)
                shifts, t_of = shift_index(pb)
                ph = (np.exp(2j * np.pi * (kpts @ shifts.T)) * kw[:, None]).T   # (nT, nk)
                n2 = norb * norb
                r_t = (ph @ rho.reshape(len(kw), n2)).reshape(-1, norb, norb)
                w_t = (ph @ wdm.reshape(len(kw), n2)).reshape(-1, norb, norb)
                p_e = r_t[t_of, pb.cols, pb.rows].real
                w_e = w_t[t_of, pb.cols, pb.rows].real
            vsum = 0.5 * (v_orb[pb.rows] + v_orb[pb.cols])
            coef_s = p_e * vsum - w_e
            ge = p_e[:, None] * pb.dh + coef_s[:, None] * pb.ds   # dE/dvec per element
            gp = np.stack([np.bincount(pb.pair_of, ge[:, k], minlength=len(pb.i))
                           for k in range(3)], axis=1)
            for k in range(3):
                grad[:, k] += np.bincount(pb.j, gp[:, k], minlength=len(pos))
                grad[:, k] -= np.bincount(pb.i, gp[:, k], minlength=len(pos))
            if self.scc:
                gs = coul.gradient(q_in - q0)
                np.add.at(grad, site_atom, gs)
            res.forces = -grad
            t["forces"] = time.perf_counter() - t1
        t["total"] = time.perf_counter() - t0
        res.timings = t
        return res

    def _repulsive(self, species, pos, cell, pbc, forces):
        from .neighbors import neighbor_list
        sk = self.sk
        rc = max((f.repulsive.cutoff for f in sk.files.values() if f.repulsive is not None),
                 default=0.0)
        grad = np.zeros_like(pos)
        if rc == 0.0:
            return 0.0, grad
        i, j, _, vec, d = neighbor_list(pos, cell, pbc, rc)
        spec = np.array(species)
        e_tot = 0.0
        for a in sorted(set(species)):
            for b in sorted(set(species)):
                m = (spec[i] == a) & (spec[j] == b)
                rep = sk.repulsive(a, b)
                if not m.any() or rep is None:
                    continue
                e, de = rep(d[m])
                e_tot += 0.5 * e.sum()
                if forces:
                    g = (0.5 * de / d[m])[:, None] * vec[m]
                    np.add.at(grad, i[m], -g)
                    np.add.at(grad, j[m], g)
        return e_tot, grad
