"""Divide-and-conquer densities: subsystems with buffers (milestone M7d).

The density-matrix divide and conquer of Yang and Lee (J. Chem. Phys. 103, 5674, 1995), as in
Nakai's DC-DFTB.  The atoms are cut into cores: whole bonded fragments on a grid of cells of
about ``core`` Angstrom.  A core and the fragments within ``buffer`` Angstrom of it (periodic
images included) form a subsystem.  Its H and S are the principal submatrices of the global
H(k) and S(k) on the pattern of the neighbour blocks (milestone M7c), so a subsystem of all the
atoms is the exact calculation, at the Gamma point and at k-points alike.  The eigenstates of
the subsystems fill one density matrix through the partition matrix
p_ij = (c_i + c_j) / 2 (c_i = 1 for an orbital of the subsystem's core):

    P_ij = sum_a p^a_ij g sum_n f_n C_in C_jn*        W_ij: the same with eps_n f_n

on the entries of the pattern.  The occupations share one chemical potential, found from the
core weights w_n = sum_{i in core} Re(C_in* (S C)_in) of the states (the Mulliken populations
of the cores, whose sum is the electron count of P) by DFTB+'s rules (mid-gap, then the
bisection), so that the exact limit has the eigensolver's Fermi level.

The subsystems are independent.  ``processes=None`` diagonalises them in this process, n forks
n worker processes that keep their subsystems from one SCC iteration to the next, "mpi" uses
the ranks of MPI.COMM_WORLD (mpi4py; rank 0 runs pydftb, the others :func:`serve`).  BLAS runs
on one thread for the subsystems in every case, and every entry of the pattern gets at most two
contributions (from the cores of its two atoms), so the results are the same, bit for bit, for
any number of processes.
"""
from __future__ import annotations

import atexit
import contextlib
import os
import re
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.special import expit

from .constants import BOHR, ELEC_TOL_MAX
from .neighbors import neighbor_list

DEFAULT_BUFFER = 6.0          # Angstrom
DEFAULT_CORE = 6.0            # Angstrom
MAX_UNIT = 64                 # atoms: larger bonded fragments (networks, metals) go by atoms
BOND_FACTOR = 1.2             # bonded: d < 1.2 (r_cov(a) + r_cov(b))
ELEC_TOL = 1.0e-15            # DFTB+ Efilling (pydftb.occupations)
CUM_WEIGHT_TOL = 1.0e-9
DEGENERATE = 1.0e-12          # Ha: copies of one level from several subsystems
EPS2 = 2.0 * np.finfo(float).eps
EPS_SINGLE = float(np.finfo(np.float32).eps)
M_EXP_ARG = 36.043653389117154


# ---- partition --------------------------------------------------------------------------------
def _element(label: str) -> str:
    m = re.match(r"[A-Z][a-z]?", label)
    return m.group(0) if m else label


def bonded_fragments(species, pos, cell, pbc):
    """Connected bonded fragments (d < 1.2 (r_a + r_b), ASE's covalent radii, periodic images):
    a list of sorted atom-index arrays, in the order of their first atom."""
    from ase.data import atomic_numbers, covalent_radii
    n = len(species)
    rad = np.array([covalent_radii[atomic_numbers.get(_element(s), 0)] or 1.5
                    for s in species]) / BOHR
    i, j, _, _, d = neighbor_list(pos, cell, pbc, BOND_FACTOR * 2.0 * rad.max())
    keep = d < BOND_FACTOR * (rad[i] + rad[j])
    g = sp.coo_matrix((np.ones(int(keep.sum())), (i[keep], j[keep])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    order = np.argsort(lab, kind="stable")
    cuts = np.flatnonzero(np.diff(lab[order])) + 1
    frags = np.split(order, cuts)
    return sorted(frags, key=lambda f: int(f[0]))


def grid_cores(pos, cell, periodic: bool, units, core: float):
    """Cores: the units (atom arrays) in a grid of cells of about ``core`` Bohr along the lattice
    vectors (molecules: the bounding box), placed by their first atom; sorted atom arrays."""
    if periodic:
        lengths = np.linalg.norm(cell, axis=1)
        frac = np.linalg.solve(np.asarray(cell, float).T, np.asarray(pos, float).T).T % 1.0
    else:
        lo, hi = pos.min(0), pos.max(0)
        lengths = hi - lo
        frac = (pos - lo) / np.maximum(lengths, 1e-9)
    nb = np.maximum(1, np.floor(lengths / core + 0.5)).astype(int)
    cells: dict = {}
    for u in units:
        key = tuple(np.minimum((frac[u[0]] * nb).astype(int), nb - 1))
        cells.setdefault(key, []).append(u)
    return [np.sort(np.concatenate(cells[k])) for k in sorted(cells)]


def subsystem_atoms(pos, cell, pbc, cores, unit_of, units, buffer: float):
    """Every core with the units of the atoms within ``buffer`` Bohr of one of its atoms
    (neighbour list with periodic images; an atom counts once however many of its images are
    near).  ``buffer`` = inf: all atoms."""
    n = len(pos)
    if not np.isfinite(buffer):
        return [np.arange(n) for _ in cores]
    i, j, _, _, _ = neighbor_list(pos, cell, pbc, buffer)
    adj = sp.csr_matrix((np.ones(len(i), bool), (i, j)), shape=(n, n))
    subs = []
    for c in cores:
        near = np.unique(np.concatenate([c, adj[c].indices]))
        near_units = np.unique(unit_of[near])
        subs.append(np.sort(np.concatenate([units[u] for u in near_units])))
    return subs


def partition(species, pos, cell, pbc, buffer=DEFAULT_BUFFER, core=DEFAULT_CORE,
              units="auto", max_unit=MAX_UNIT):
    """(cores, subsystems, units): atom arrays of the cores, of the subsystems (core + buffer)
    and the units the cut keeps whole; ``buffer`` and ``core`` in Angstrom."""
    n = len(species)
    periodic = cell is not None and pbc is not None and bool(np.any(pbc))
    if units == "atoms":
        frags = [np.array([a]) for a in range(n)]
    else:
        frags = bonded_fragments(species, pos, cell if periodic else None,
                                 pbc if periodic else None)
        if units == "auto" and max(len(f) for f in frags) > max_unit:
            frags = [np.array([a]) for a in range(n)]
        elif units not in ("auto", "fragments"):
            raise ValueError(f"units must be 'auto', 'fragments' or 'atoms', not {units!r}")
    unit_of = np.empty(n, int)
    for k, f in enumerate(frags):
        unit_of[f] = k
    cores = grid_cores(pos, cell, periodic, frags, core / BOHR)
    subs = subsystem_atoms(pos, cell if periodic else None, pbc if periodic else None, cores,
                           unit_of, frags, buffer / BOHR)
    return cores, subs, frags


# ---- the subsystems on the pattern ------------------------------------------------------------
def subsystem_index(indptr, indices, n, offsets, norb_atom, core, atoms):
    """Orbitals of a subsystem and its pattern entries: dict with ``orbs``, ``core`` (bool per
    orbital), ``r``, ``c``, ``g`` (local row, column and pattern position of every entry in the
    subsystem's block) and ``kr``, ``kc``, ``kg``, ``kp`` (the entries with partition p > 0)."""
    nnz = len(indices)
    idx = sp.csr_matrix((np.arange(1, nnz + 1, dtype=np.float64), indices, indptr), shape=(n, n))
    return _index(idx, offsets, norb_atom, core, atoms)


def _index(idx, offsets, norb_atom, core, atoms):
    orbs = np.concatenate([np.arange(offsets[a], offsets[a] + norb_atom[a]) for a in atoms])
    is_core = np.repeat(np.isin(atoms, core), norb_atom[atoms])
    blk = idx[orbs][:, orbs].tocoo()
    r, c = blk.row.astype(np.int32), blk.col.astype(np.int32)
    g = blk.data.astype(np.int64) - 1
    p = 0.5 * (is_core[r].astype(float) + is_core[c].astype(float))
    keep = p > 0
    return dict(orbs=orbs, core=is_core, r=r, c=c, g=g, kr=r[keep], kc=c[keep], kg=g[keep],
                kp=p[keep])


def _dense(data, sub, dtype):
    m = len(sub["orbs"])
    a = np.zeros((m, m), dtype=dtype)
    a[sub["r"], sub["c"]] = data[sub["g"]]
    return a


def eig_subsystems(subs, h_data, s_data, dtype):
    """For every subsystem: (eps, core weights, C) of H_a C = S_a C eps."""
    out = []
    for sub in subs:
        hs, ss = _dense(h_data, sub, dtype), _dense(s_data, sub, dtype)
        e, c = sla.eigh(hs, ss, driver="gvd")
        w = (c.conj() * (ss @ c)).real[sub["core"]].sum(0)
        out.append((e, w, c))
    return out


def density_subsystems(subs, states, fills, need_w, nnz, dtype):
    """P and W contributions (without the spin factor) of the subsystems on the pattern, with the
    occupations ``fills`` (one array per subsystem)."""
    rho = np.zeros(nnz, dtype=dtype)
    wdm = np.zeros(nnz, dtype=dtype) if need_w else None
    for sub, (e, _, c), f in zip(subs, states, fills):
        occ = f > 0.0
        if not occ.any():
            continue
        co = c[:, occ]
        cf = co * f[occ]
        d = cf @ co.conj().T
        rho[sub["kg"]] += sub["kp"] * d[sub["kr"], sub["kc"]]
        if need_w:
            d = (cf * e[occ]) @ co.conj().T
            wdm[sub["kg"]] += sub["kp"] * d[sub["kr"], sub["kc"]]
    return rho, wdm


# ---- the chemical potential of weighted levels (DFTB+ Efilling, pydftb.occupations.fill) ------
def _f(e, mu, kt):
    x = (e - mu) / kt
    return np.where(x <= M_EXP_ARG, expit(-x), 0.0)


def _count(mu, blocks, kt):
    return float(sum((w * _f(e, mu, kt)).sum() for e, w in blocks))


def _fill(mu, blocks, kt):
    """Occupations (max 1) per block with DFTB+'s cut (zero from the first sorted level with
    f <= 1e-15 on), and the weighted entropy sum_n w_n s(f_n)."""
    fs, ent = [], 0.0
    for e, w in blocks:
        f = _f(e, mu, kt)
        cut = np.cumsum(f <= ELEC_TOL) > 0
        f = np.where(cut, 0.0, f)
        mid = (f > np.finfo(float).eps) & (f < 1.0 - np.finfo(float).eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(mid, -(f * np.log(f) + (1.0 - f) * np.log(1.0 - f)), 0.0)
        ent += float((w * s).sum())
        fs.append(f)
    return fs, ent


def _middle_gap(blocks, nfill):
    """DFTB+'s middleGap on the weighted levels; the copies of one level (several subsystems)
    count as that level."""
    e = np.concatenate([b[0] for b in blocks])
    w = np.concatenate([b[1] for b in blocks])
    order = np.argsort(e, kind="stable")
    e, cum = e[order], np.cumsum(w[order])
    i = min(int(np.searchsorted(cum, nfill - CUM_WEIGHT_TOL, side="left")), len(e) - 1)
    above = e[e > e[i] + DEGENERATE]
    return 0.5 * (e[i] + (above[0] if len(above) else e[i]))


def chemical_potential(blocks, nfill, kt, fermi_level="dftb+"):
    """mu for the levels ``blocks`` [(eps ascending, weights)] holding ``nfill`` (DFTB+ Efilling
    with weighted levels): (mu, occupations per block, weighted entropy)."""
    total = float(sum(b[1].sum() for b in blocks))
    e_all = np.concatenate([b[0] for b in blocks])
    if nfill < np.finfo(float).eps:
        mu = e_all.min() - 1000.0 * (kt + EPS_SINGLE)
        return mu, [np.zeros(len(b[0])) for b in blocks], 0.0
    if total <= nfill + CUM_WEIGHT_TOL:
        mu = e_all.max() + 1000.0 * (kt + EPS_SINGLE)
        fs, ent = _fill(mu, blocks, kt)
        return mu, fs, ent
    if fermi_level == "dftb+" and abs(nfill - round(nfill)) <= ELEC_TOL:
        mu = _middle_gap(blocks, nfill)
        if abs(nfill - _count(mu, blocks, kt)) <= ELEC_TOL_MAX:
            fs, ent = _fill(mu, blocks, kt)
            return mu, fs, ent
    upper, lower = e_all.max() + 0.01, e_all.min() - 0.01
    n_max, n_min = _count(upper, blocks, kt), _count(lower, blocks, kt)
    while n_min > nfill:
        lower = 2.0 * (lower - upper) + lower
        n_min = _count(lower, blocks, kt)
    while n_max < nfill:
        upper = 2.0 * (upper - lower) + lower
        n_max = _count(upper, blocks, kt)
    mu = 0.5 * (upper + lower)
    n_el = _count(mu, blocks, kt)
    while abs(nfill - n_el) > ELEC_TOL and abs(upper - lower) >= max(abs(mu) * EPS2, EPS2):
        if (n_max >= n_min) == (nfill >= n_el):
            lower, n_min = mu, n_el
        else:
            upper, n_max = mu, n_el
        mu = 0.5 * (upper + lower)
        n_el = _count(mu, blocks, kt)
    if abs(nfill - n_el) > ELEC_TOL_MAX:
        raise RuntimeError("Fermi level search of the subsystems did not converge")
    fs, ent = _fill(mu, blocks, kt)
    if nfill > EPS2 and abs(nfill - n_el) > EPS2:
        fs = [f * nfill / n_el for f in fs]
    return mu, fs, ent


# ---- executors ----------------------------------------------------------------------------------
class _Serial:
    """The subsystems in this process, BLAS on one thread."""

    def __init__(self, subs):
        self.subs = subs
        self.states = None
        self.s_data = None

    def set_overlap(self, s_data):
        self.s_data = s_data

    def eig(self, h_data, offset, dtype):
        from threadpoolctl import threadpool_limits
        s_data = self.s_data[offset:offset + len(h_data)]
        with threadpool_limits(1, user_api="blas"):
            self.states = [eig_subsystems(self.subs, h, s, dtype) for h, s in zip(h_data, s_data)]
        return [[(e, w) for e, w, _ in st] for st in self.states]

    def density(self, fills, need_w, nnz, dtype):
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1, user_api="blas"):
            return [density_subsystems(self.subs, st, f, need_w, nnz, dtype)
                    for st, f in zip(self.states, fills)]

    def close(self):
        self.states = None
        self.s_data = None


def _worker_main(conn, subs):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1, user_api="blas"):
        local = _Serial(subs)
        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                break
            if msg[0] == "overlap":
                local.set_overlap(msg[1])
            elif msg[0] == "eig":
                conn.send(local.eig(*msg[1:]))
            elif msg[0] == "density":
                conn.send(local.density(*msg[1:]))
    conn.close()


class _Workers:
    """Forked worker processes, each with a fixed set of subsystems (inherited at the fork)."""

    def __init__(self, groups):
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        self.procs = []
        for subs in groups:
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker_main, args=(b, subs), daemon=True)
            p.start()
            b.close()
            self.procs.append((p, a))

    def set_overlap(self, s_data):
        for _, conn in self.procs:
            conn.send(("overlap", s_data))

    def eig(self, h_data, offset, dtype):
        for _, conn in self.procs:
            conn.send(("eig", h_data, offset, dtype))
        return [conn.recv() for _, conn in self.procs]

    def density(self, fills_per_worker, need_w, nnz, dtype):
        for (_, conn), fills in zip(self.procs, fills_per_worker):
            conn.send(("density", fills, need_w, nnz, dtype))
        return [conn.recv() for _, conn in self.procs]

    def close(self):
        for p, conn in self.procs:
            try:
                conn.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
        for p, conn in self.procs:
            p.join(5.0)
            if p.is_alive():
                p.terminate()
            conn.close()
        self.procs = []


def _comm():
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError("Subsystems(processes='mpi') needs mpi4py (pip install "
                          "pydftb[mpi]) and an MPI library") from exc
    return MPI.COMM_WORLD


class _MPI:
    """Ranks 1..n-1 of MPI.COMM_WORLD own the subsystems; this is rank 0."""

    def __init__(self, groups, pattern_args):
        self.comm = _comm()
        self.comm.bcast(("geometry", pattern_args), root=0)
        self.comm.scatter([None] + groups, root=0)

    def set_overlap(self, s_data):
        self.comm.bcast(("overlap", s_data), root=0)

    def eig(self, h_data, offset, dtype):
        self.comm.bcast(("eig", h_data, offset, dtype), root=0)
        return self.comm.gather(None, root=0)[1:]

    def density(self, fills_per_worker, need_w, nnz, dtype):
        self.comm.bcast(("density", need_w, nnz, dtype), root=0)
        self.comm.scatter([None] + fills_per_worker, root=0)
        return self.comm.gather(None, root=0)[1:]

    def close(self):
        pass


def serve() -> bool:
    """MPI ranks > 0: serve the subsystems of rank 0's Subsystems(processes="mpi") until rank 0
    stops (at its exit), then return True; rank 0 (and runs without MPI ranks) return False at
    once.  Use as ``if serve(): sys.exit()`` at the top of a script, or run the script with
    ``mpiexec -n N python -m pydftb.subsystems script.py``."""
    comm = _comm()
    if comm.Get_rank() == 0:
        if comm.Get_size() > 1:
            atexit.register(_stop_workers, comm)
        return False
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1, user_api="blas"):
        local = None
        while True:
            msg = comm.bcast(None, root=0)
            if msg[0] == "stop":
                break
            if msg[0] == "geometry":
                indptr, indices, n, offsets, norb_atom = msg[1]
                idx = sp.csr_matrix((np.arange(1, len(indices) + 1, dtype=np.float64),
                                     indices, indptr), shape=(n, n))
                mine = comm.scatter(None, root=0)
                local = _Serial([_index(idx, offsets, norb_atom, c, a) for c, a in mine])
            elif msg[0] == "overlap":
                local.set_overlap(msg[1])
            elif msg[0] == "eig":
                comm.gather(local.eig(*msg[1:]), root=0)
            elif msg[0] == "density":
                fills = comm.scatter(None, root=0)
                comm.gather(local.density(fills, *msg[1:]), root=0)
    return True


def _stop_workers(comm):
    with contextlib.suppress(Exception):          # at exit: nothing to report to
        comm.bcast(("stop",), root=0)


# ---- the solver ---------------------------------------------------------------------------------
class Subsystems:
    """Divide-and-conquer densities (DC-DFTB) on the pattern of the neighbour blocks.

    ``buffer``, ``core``: Angstrom (buffer = inf: every subsystem holds all atoms, the exact
    limit); ``units``: "auto" (bonded fragments, atoms when a fragment has more than
    ``max_unit`` atoms), "fragments" or "atoms"; ``processes``: None (this process), an integer
    (forked worker processes) or "mpi" (the other ranks of MPI.COMM_WORLD, see :func:`serve`).
    Fermi filling only (T = 0 is kT = 1e-8 Ha, as everywhere in pydftb)."""

    name = "subsystems"
    eigenpairs = False
    sparse = True                 # the calculator builds the pattern of M7c
    subsystems = True

    def __init__(self, buffer: float = DEFAULT_BUFFER, core: float = DEFAULT_CORE,
                 units: str = "auto", processes=None, max_unit: int = MAX_UNIT):
        if not buffer >= 0.0 or not core > 0.0:
            raise ValueError("buffer must be >= 0 and core > 0 (Angstrom)")
        if units not in ("auto", "fragments", "atoms"):
            raise ValueError(f"units must be 'auto', 'fragments' or 'atoms', not {units!r}")
        if processes not in (None, "mpi") and not (isinstance(processes, int)
                                                    and processes >= 0):
            raise ValueError("processes must be None, an integer >= 0 or 'mpi'")
        self.buffer, self.core, self.units = float(buffer), float(core), units
        self.processes, self.max_unit = processes, int(max_unit)
        self._exec = None
        self.timings: dict = {}
        self.last_count_error = None
        self.n_subsystems = 0
        self.sizes = None

    def __repr__(self):
        return (f"Subsystems(buffer={self.buffer!r}, core={self.core!r}, units={self.units!r}, "
                f"processes={self.processes!r})")

    # -- geometry ------------------------------------------------------------------------------
    def set_geometry(self, species, pos, cell, pbc, basis, pattern):
        """Partition (cores, subsystems) and the subsystems' entries of ``pattern``; starts or
        tells the worker processes."""
        t0 = time.perf_counter()
        self.close()
        cores, atoms, _ = partition(species, np.asarray(pos, float), cell, pbc, self.buffer,
                                    self.core, self.units, self.max_unit)
        self.cores, self.atoms = cores, atoms
        offsets, norb_atom = np.asarray(basis.offsets), np.asarray(basis.norb_atom)
        self.sizes = np.array([int(norb_atom[a].sum()) for a in atoms])
        self.n_subsystems = len(atoms)
        self.nnz, self.norb = pattern.nnz, pattern.n
        nworkers = self._nworkers()
        self.owner = self._assign(nworkers) if nworkers else [list(range(len(atoms)))]
        if self.processes == "mpi" and nworkers:
            pattern_args = (pattern.indptr, pattern.indices, pattern.n, offsets, norb_atom)
            groups = [[(cores[a], atoms[a]) for a in o] for o in self.owner]
            self._exec = _MPI(groups, pattern_args)
        else:
            idx = sp.csr_matrix((np.arange(1, pattern.nnz + 1, dtype=np.float64),
                                 pattern.indices, pattern.indptr), shape=(pattern.n, pattern.n))
            subs = [_index(idx, offsets, norb_atom, c, a) for c, a in zip(cores, atoms)]
            if nworkers:
                self._exec = _Workers([[subs[a] for a in o] for o in self.owner])
            else:
                self._exec = _Serial(subs)
        self.timings["partition"] = self.timings.get("partition", 0.0) + time.perf_counter() - t0

    def _nworkers(self) -> int:
        if self.processes == "mpi":
            return _comm().Get_size() - 1
        if isinstance(self.processes, int) and self.processes > 0:
            import multiprocessing as mp
            return self.processes if "fork" in mp.get_all_start_methods() else 0
        return 0

    def _assign(self, nworkers):
        """Subsystems to workers: the largest (m^3) first to the least loaded."""
        cost = self.sizes.astype(float) ** 3
        load = np.zeros(nworkers)
        owner = [[] for _ in range(nworkers)]
        for a in np.argsort(-cost, kind="stable"):
            j = int(np.argmin(load))
            owner[j].append(int(a))
            load[j] += cost[a]
        return [sorted(o) for o in owner]

    def close(self):
        if self._exec is not None:
            self._exec.close()
            self._exec = None

    def __del__(self):
        with contextlib.suppress(Exception):      # interpreter shutdown
            self.close()

    # -- SCC -----------------------------------------------------------------------------------
    def prepare(self, s_list):
        """The overlap of every k-point (and spin channel): CSR matrices on the pattern."""
        if self._exec is None:
            raise RuntimeError("Subsystems.set_geometry must come before prepare")
        self.s_data = [np.asarray(m.data) for m in s_list]
        self._exec.set_overlap(self.s_data)              # once per geometry

    def density(self, h, kw, n_occ: float, spin_factor: float, need_w: bool, offset: int = 0,
                kt: float | None = None, mu: float | None = None, independent_k: bool = False,
                channels: int = 1, fermi_level: str = "dftb+"):
        """(P, W or None as (len(h), nnz) values on the pattern, mu, T S) of the CSR matrices
        ``h`` (``channels`` copies of the k-points ``kw``, from ``offset`` of the prepared
        overlaps), ``n_occ`` levels of ``spin_factor`` electrons; ``mu``: a fixed chemical
        potential; ``independent_k``: every k-point holds ``n_occ`` on its own."""
        if kt is None:
            raise ValueError("the subsystems need kT (Fermi filling)")
        t0 = time.perf_counter()
        nh = len(h)
        dtype = np.result_type(h[0].dtype, self.s_data[0].dtype)
        h_data = [np.asarray(m.data) for m in h]
        w = np.tile(np.asarray(kw, float) / np.sum(kw), channels)
        got = self._exec.eig(h_data, offset, dtype)
        ew = self._by_subsystem(got, nh)                 # [ik][subsystem] = (eps, w)
        t1 = time.perf_counter()
        if independent_k:
            groups = [([ik], 1.0) for ik in range(nh)]
        else:
            groups = [(list(range(nh)), None)]
        fills = [[None] * self.n_subsystems for _ in range(nh)]
        mus, ts, count = [], 0.0, 0.0
        for iks, wfix in groups:
            blocks, keys = [], []
            for ik in iks:
                wk = w[ik] if wfix is None else wfix
                for a in range(self.n_subsystems):
                    e, wn = ew[ik][a]
                    blocks.append((e, wn * wk))
                    keys.append((ik, a))
            if mu is not None:
                m = float(mu)
                fs, ent = _fill(m, blocks, kt)
            else:
                m, fs, ent = chemical_potential(blocks, n_occ, kt, fermi_level)
            for (ik, a), f in zip(keys, fs):
                fills[ik][a] = f
            count += float(sum((b[1] * f).sum() for b, f in zip(blocks, fs)))
            scale = 1.0 if wfix is None else w[iks[0]]
            mus.append(m)
            ts += scale * spin_factor * kt * ent
        per_worker = [[[fills[ik][a] for a in o] for ik in range(nh)] for o in self.owner]
        parts = self._exec.density(per_worker if len(self.owner) > 1 or
                                   not isinstance(self._exec, _Serial) else per_worker[0],
                                   need_w, self.nnz, dtype)
        if isinstance(self._exec, _Serial):
            parts = [parts]
        rho = np.zeros((nh, self.nnz), dtype=dtype)
        wdm = np.zeros((nh, self.nnz), dtype=dtype) if need_w else None
        for res in parts:                                # at most two contributions per entry
            for ik, (r, wd) in enumerate(res):
                rho[ik] += r
                if need_w:
                    wdm[ik] += wd
        rho *= spin_factor
        if need_w:
            wdm *= spin_factor
        if independent_k:
            self.last_count_error = spin_factor * (count - n_occ * nh) / max(nh, 1)
            mu_out = float(np.dot(w[:nh] / w[:nh].sum(), mus))
        else:
            self.last_count_error = spin_factor * (count - n_occ)
            mu_out = mus[0]
        self.timings["eig"] = self.timings.get("eig", 0.0) + t1 - t0
        self.timings["density"] = self.timings.get("density", 0.0) + time.perf_counter() - t1
        return rho, wdm, mu_out, float(ts)

    def _by_subsystem(self, got, nh):
        ew = [[None] * self.n_subsystems for _ in range(nh)]
        if isinstance(self._exec, _Serial):
            got = [got]
        for o, res in zip(self.owner, got):
            for ik, per_k in enumerate(res):
                for a, ewa in zip(o, per_k):
                    ew[ik][a] = ewa
        return ew


def main(argv=None) -> int:
    """``mpiexec -n N python -m pydftb.subsystems script.py [args]``: rank 0 runs the script,
    the other ranks serve its Subsystems(processes="mpi")."""
    import runpy
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: mpiexec -n N python -m pydftb.subsystems script.py [args]",
              file=sys.stderr)
        return 2
    if serve():
        return 0
    sys.argv = argv
    sys.path.insert(0, os.path.dirname(os.path.abspath(argv[0])))
    runpy.run_path(argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
