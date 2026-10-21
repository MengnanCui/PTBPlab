"""Cost model of one DFTB single point (energy and forces), for planning parallel runs.

Milestone M3c: PTBP's recommender (``utils/recommender.py``) decides how many processes
and threads a BO step uses from these estimates.  Everything is computed from what is
known before a calculation runs:

* the orbitals N of the structure and its irreducible k-points nk;
* the Slater-Koster pairs within the table cut-off (``r_sk``), weighted by their
  orbital blocks: building H(k), S(k) and the band forces cost nk such pairs;
* the Coulomb pairs of the SCC.  The short-range part of gamma decays as
  exp(-3.2 U r) and is summed until it drops below 1e-10 (DFTB+'s rule), i.e. up to
  96 Bohr for the 4p shell of Cu (U = 0.08 Ha) or 168 Bohr for the 4d shell of Ga.  In
  a periodic cell the pairs grow as natoms^2 / V * rc^3, so from a few tens of atoms
  on, the Coulomb set-up and its forces take more time and memory than the
  eigensolver;
* a typical number of SCC iterations (``N_SCC``; it cannot be known in advance).

The constants were fitted to calibration runs on the development machine (single
thread; Cu dataset of PTBP, Cu/C/CuO/Li/GdO cells of 2-216 atoms; M3c report).
:func:`probe` measures how fast this machine is relative to that one.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from pydftb.constants import BOHR       # Angstrom per Bohr, as the calculator uses it
from pydftb.coulomb import EWALD_TOL
from pydftb.coulomb import short_cutoff as _gamma_cutoff

SK_CUTOFF = 14.38 + 1.0          # Bohr: 719 points x 0.02 Bohr (hotcent/PTBP) + DFTB+'s tail
# SCC iterations assumed (they cannot be known in advance): PTBP's Cu dataset needs a
# median of 7 (pydftb) and 6 (DFTB+); metallic cells of 32-108 atoms 10-23
N_SCC = {"pydftb": 8, "dftbplus": 7}
_FOUR_PI_3 = 4.0 * math.pi / 3.0

# l of the valence shells by block: H, He s; main group sp; d block spd; f block spdf
_D_BLOCK = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81)) | set(range(104, 113))
_F_BLOCK = set(range(57, 72)) | set(range(89, 104))
DEFAULT_HUBBARD = 0.08           # Ha, for shells of unknown U: a long (96 Bohr) cut-off


def block_shells(z: int) -> tuple[int, ...]:
    """Angular momenta of the valence shells of element ``z`` by its block."""
    z = int(z)
    if z <= 2:
        return (0,)
    if z in _F_BLOCK:
        return (0, 1, 2, 3)
    if z in _D_BLOCK:
        return (0, 1, 2)
    return (0, 1)


def basis_size(shells) -> int:
    """Orbitals of an atom with valence shells of angular momenta ``shells``."""
    return int(sum(2 * l + 1 for l in shells))


def irreducible_kpoints(grid) -> int:
    """k-points of a Monkhorst-Pack grid after the time-reversal reduction: k and -k
    pair up; only Gamma is its own partner, and it is on the grid when every n is odd."""
    if grid is None or len(grid) != 3:
        return 1
    n = [int(k) for k in grid]
    return (n[0] * n[1] * n[2] + all(k % 2 for k in n)) // 2


@dataclass(frozen=True)
class Species:
    """Valence shells of an element: angular momentum and Hubbard U (Ha) of each."""
    shells: tuple = ((0, DEFAULT_HUBBARD),)

    @property
    def norb(self) -> int:
        return basis_size(l for l, _ in self.shells)

    @classmethod
    def from_block(cls, z: int, hubbard: float = DEFAULT_HUBBARD) -> "Species":
        return cls(tuple((l, float(hubbard)) for l in block_shells(z)))


@lru_cache(maxsize=None)
def short_cutoff(u_a: float, u_b: float) -> float:
    """Reach (Bohr) of the short-range gamma between shells of Hubbard U ``u_a``, ``u_b``
    (pydftb.coulomb.short_cutoff, the rule of DFTB+)."""
    return float(_gamma_cutoff(3.2 * float(u_a), 3.2 * float(u_b)))


@dataclass
class Shape:
    """What the cost of one calculation depends on."""
    natoms: int
    norb: int
    nk: int = 1
    gamma: bool = True            # Gamma point only: real matrices
    sk_pairs: float = 0.0         # sum over SK pairs (i, j) of norb_i * norb_j
    site_pairs: float = 0.0       # shell pairs within their short-range cut-off
    coulomb_pairs: float = 0.0    # atom pairs within the Coulomb (short or Ewald) cut-off
    periodic: bool = True

    def to_json(self) -> dict:
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def shape(numbers, species: dict, *, cell=None, pbc=None, positions=None, grid=None,
          r_sk: float = SK_CUTOFF, shell_resolved: bool = True) -> Shape:
    """Cost features of a structure.

    numbers   : atomic numbers (or anything ``species`` is keyed by), one per atom
    species   : number -> :class:`Species`
    cell, pbc : lattice vectors (Angstrom, rows) and periodicity; periodic cells use the
                mean density (exact when the cut-offs reach well beyond the cell)
    positions : Angstrom; needed for clusters (pairs are counted)
    grid      : Monkhorst-Pack grid (periodic) or None (Gamma)
    """
    numbers = list(numbers)
    kinds = sorted(set(numbers), key=str)
    count = {k: numbers.count(k) for k in kinds}
    sp = {k: species[k] for k in kinds}
    norb = sum(count[k] * sp[k].norb for k in kinds)
    periodic = cell is not None and (pbc is None or bool(np.all(pbc)))
    if periodic:
        nk = irreducible_kpoints(grid)
        gamma = grid is None or all(int(g) == 1 for g in grid)
    else:
        nk, gamma = 1, True

    def shells(k):
        if shell_resolved:
            return [u for _, u in sp[k].shells]
        return [min(u for _, u in sp[k].shells)]          # one site per atom

    cuts = {(a, b): [short_cutoff(ua, ub) for ua in shells(a) for ub in shells(b)]
            for a in kinds for b in kinds}
    rc_short = max(max(c) for c in cuts.values())
    natoms = len(numbers)
    if periodic:
        vol = abs(float(np.linalg.det(np.asarray(cell, float)))) / BOHR ** 3
        alpha = math.sqrt(math.pi) * (natoms / vol ** 2) ** (1.0 / 6.0)
        rc = max(rc_short, math.sqrt(-math.log(EWALD_TOL)) / alpha)
        dens = {(a, b): count[a] * count[b] / vol * _FOUR_PI_3 for a in kinds for b in kinds}
        sk_pairs = sum(dens[a, b] * r_sk ** 3 * sp[a].norb * sp[b].norb for a, b in dens)
        site_pairs = sum(dens[a, b] * sum(c ** 3 for c in cuts[a, b]) for a, b in dens)
        coulomb_pairs = natoms * natoms / vol * _FOUR_PI_3 * rc ** 3
    else:
        pos = np.asarray(positions, float) / BOHR
        sk_pairs = site_pairs = coulomb_pairs = 0.0
        num = np.array(numbers, dtype=object)
        for a in kinds:
            for b in kinds:
                pa, pb = pos[num == a], pos[num == b]
                d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1))
                off = d[d > 1e-8]
                sk_pairs += np.count_nonzero(off < r_sk) * sp[a].norb * sp[b].norb
                site_pairs += sum(np.count_nonzero(off < c) for c in cuts[a, b])
        coulomb_pairs = float(natoms * (natoms - 1))
    return Shape(natoms, norb, nk, gamma, float(sk_pairs), float(site_pairs), float(coulomb_pairs),
                 periodic)


# ---------------------------------------------------------------------------------------
# Fitted constants (development machine, one thread).  Time in seconds, memory in MB.
# Features: c0, sk_pairs, site_pairs, coulomb_pairs, then per SCC iteration and k-point
# (n_scc + 2 passes: the SCC iterations, the energy and the forces) N^3 for complex
# (k-points) and real (Gamma) matrices, N^2 and a constant, and nk * sk_pairs (H(k),
# S(k) and the band forces).
PYDFTB_TIME = {"c0": 1.791e-02, "sk_pairs": 1.244e-06, "site_pairs": 1.183e-06,
               "coulomb_pairs": 2.872e-07, "N3 complex": 1.857e-09, "N3 real": 2.782e-10,
               "N2": 1.405e-07, "per k-iter": 7.604e-06, "nk*sk_pairs": 5.517e-09}
PYDFTB_MEMORY = {"m0": 4.322, "sk_pairs": 1.752e-04, "site_pairs": 1.571e-04,
                 "coulomb_pairs": 7.387e-05, "matrix bytes": 8.408e-06}
# DFTB+ 25.1 through ASE as PTBP runs it (process start, input and output files
# included).  Its Coulomb sums need far less memory than pydftb's pair lists.
DFTBPLUS_TIME = {"c0": 4.366e-02, "sk_pairs": 0.0, "site_pairs": 0.0,
                 "coulomb_pairs": 1.004e-06, "N3 complex": 6.213e-10, "N3 real": 2.718e-10,
                 "N2": 2.261e-07, "per k-iter": 5.490e-05, "nk*sk_pairs": 4.692e-08}
DFTBPLUS_MEMORY = {"m0": 29.46, "sk_pairs": 0.0, "site_pairs": 0.0,
                   "coulomb_pairs": 9.684e-05, "matrix bytes": 1.071e-06}
# k-point threads (pydftb.parallel): the per-k work of the SCC loop runs on
# 1 + KTHREAD_EFFICIENCY * (T - 1) cores' worth; the rest stays serial.  Fitted to the
# whole-calculation speed-ups of M3a (1.11-1.17x on 2 threads, 1.27-1.43x on 4).
KTHREAD_EFFICIENCY = 0.33
# DFTB+ OpenMP threads, same form, fitted from 576 orbitals up (1.2-1.74x on 4 threads).
# On smaller matrices threads make DFTB+ slower (36 orbitals: 3.3x slower on 4 threads,
# 288 orbitals: 1.2-1.3x slower), so they are not modelled there and never planned.
OMP_EFFICIENCY = 0.2
OMP_MIN_ORBITALS = 500


def _terms(s: Shape, n_scc: int, c: dict) -> tuple[float, float]:
    """(time of the per-k SCC work, time of the rest) of shape ``s`` with constants ``c``."""
    it = (n_scc + 2) * s.nk
    n2, n3 = float(s.norb) ** 2, float(s.norb) ** 3
    kwork = it * (c["N3 real" if s.gamma else "N3 complex"] * n3 + c["N2"] * n2 + c["per k-iter"])
    rest = (c["c0"] + c["sk_pairs"] * s.sk_pairs + c["site_pairs"] * s.site_pairs
            + c["coulomb_pairs"] * s.coulomb_pairs + c["nk*sk_pairs"] * s.nk * s.sk_pairs)
    return kwork, rest


def _amdahl(kwork: float, rest: float, threads: int, efficiency: float) -> float:
    return rest + kwork / (1.0 + efficiency * (max(1, int(threads)) - 1))


def effective_kthreads(s: Shape, k_threads: int) -> int:
    """Threads pydftb really uses for ``s``: k chunks of at least MIN_CHUNK_WORK (N^3 units)."""
    from pydftb.parallel import MIN_CHUNK_WORK
    min_size = max(1, -(-MIN_CHUNK_WORK // max(1, s.norb) ** 3))
    return max(1, min(int(k_threads), s.nk // min_size))


@dataclass
class CostModel:
    """Time (s) and peak memory (bytes) of one energy-and-forces calculation.

    ``speed``: this machine relative to the calibration machine (:func:`probe`);
    ``n_scc``: SCC iterations assumed (default: ``N_SCC`` of the engine).
    """
    speed: float = 1.0
    n_scc: int | None = None

    def _n_scc(self, engine: str, n_scc: int | None) -> int:
        if n_scc is not None:
            return int(n_scc)
        return int(self.n_scc) if self.n_scc is not None else N_SCC[engine]

    def pydftb_time(self, s: Shape, k_threads: int = 1, n_scc: int | None = None) -> float:
        kwork, rest = _terms(s, self._n_scc("pydftb", n_scc), PYDFTB_TIME)
        return _amdahl(kwork, rest, effective_kthreads(s, k_threads), KTHREAD_EFFICIENCY) / self.speed

    def kthread_speedup(self, s: Shape, k_threads: int, n_scc: int | None = None) -> float:
        return self.pydftb_time(s, 1, n_scc) / self.pydftb_time(s, k_threads, n_scc)

    def dftbplus_time(self, s: Shape, omp_threads: int = 1, n_scc: int | None = None) -> float:
        """DFTB+ time; OpenMP threads count from OMP_MIN_ORBITALS on (below, see above)."""
        kwork, rest = _terms(s, self._n_scc("dftbplus", n_scc), DFTBPLUS_TIME)
        threads = omp_threads if s.norb >= OMP_MIN_ORBITALS else 1
        return _amdahl(kwork, rest, threads, OMP_EFFICIENCY) / self.speed

    def omp_speedup(self, s: Shape, threads: int, n_scc: int | None = None) -> float:
        return self.dftbplus_time(s, 1, n_scc) / self.dftbplus_time(s, threads, n_scc)

    def time(self, engine: str, s: Shape, threads: int = 1, n_scc: int | None = None) -> float:
        if engine == "pydftb":
            return self.pydftb_time(s, threads, n_scc)
        return self.dftbplus_time(s, threads, n_scc)

    def memory(self, engine: str, s: Shape) -> float:
        """Peak memory a calculation adds to its process, bytes."""
        c = PYDFTB_MEMORY if engine == "pydftb" else DFTBPLUS_MEMORY
        matrix = s.nk * float(s.norb) ** 2 * (8 if s.gamma else 16)
        mb = (c["m0"] + c["sk_pairs"] * s.sk_pairs + c["site_pairs"] * s.site_pairs
              + c["coulomb_pairs"] * s.coulomb_pairs + c["matrix bytes"] * matrix)
        return mb * 2 ** 20


# ---------------------------------------------------------------------------------------
# Probe: how fast is this machine compared with the calibration machine?
# Three small kernels of a DFTB calculation, BLAS on one thread, best of five runs; the
# reference is the best of seven probes on the calibration machine.  Timings on a shared
# machine scatter by about 20 %, as much as the model's own error.
PROBE_REFERENCE = {"batched_eigh": 0.048, "gamma_eigh": 0.028, "elementwise": 0.027}


def _kernels():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((256, 36, 36)) + 1j * rng.standard_normal((256, 36, 36))
    herm = a @ np.conj(np.swapaxes(a, 1, 2)) + 36 * np.eye(36)
    b = rng.standard_normal((500, 500))
    sym = b @ b.T + 500 * np.eye(500)
    r = rng.uniform(1.0, 90.0, 2_000_000)

    def batched_eigh():                        # k-point eigensolver (complex, small N)
        np.linalg.eigh(herm)

    def gamma_eigh():                          # Gamma eigensolver (real, larger N)
        np.linalg.eigh(sym)

    def elementwise():                         # short-range gamma of many shell pairs
        t = 0.9
        e = np.exp(-t * r)
        (e * (1.0 / r + 11.0 * t / 16.0 + 3.0 * t * t * r / 16.0 + t ** 3 * r * r / 48.0)).sum()

    return {"batched_eigh": batched_eigh, "gamma_eigh": gamma_eigh, "elementwise": elementwise}


def probe(repeat: int = 5) -> dict:
    """Run the kernels (about 0.5 s): {"timings": {kernel: s}, "speed": relative speed}."""
    try:
        from threadpoolctl import threadpool_limits
        one_thread = threadpool_limits(1)        # applied now, restored on exit
    except Exception:                           # noqa: BLE001  (threadpoolctl missing)
        one_thread = nullcontext()
    timings = {}
    with one_thread:
        for name, fn in _kernels().items():
            fn()                                # warm-up
            best = math.inf
            for _ in range(max(1, int(repeat))):
                t0 = time.perf_counter()
                fn()
                best = min(best, time.perf_counter() - t0)
            timings[name] = best
    return {"timings": timings, "speed": speed_from_timings(timings)}


def speed_from_timings(timings: dict) -> float:
    """Geometric mean of reference / measured time over the kernels both know."""
    ratios = [PROBE_REFERENCE[k] / v for k, v in timings.items() if k in PROBE_REFERENCE and v > 0]
    if not ratios:
        return 1.0
    return float(math.exp(sum(math.log(x) for x in ratios) / len(ratios)))


def machine_fingerprint() -> str:
    """CPU model, cores, Python, NumPy and BLAS: what makes probe results reusable."""
    cpu = platform.processor() or ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    blas = ""
    try:
        cfg = np.show_config(mode="dicts")       # NumPy >= 1.26
        blas = json.dumps(cfg.get("Build Dependencies", {}).get("blas", {}), sort_keys=True)
    except Exception:                           # noqa: BLE001
        pass
    key = "|".join([cpu, str(os.cpu_count()), platform.python_version(), np.__version__, blas])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def load_probe(path, run: bool = False) -> float | None:
    """Relative speed of this machine from the cache ``path`` (JSON keyed by
    :func:`machine_fingerprint`); with ``run`` the probe runs when there is no entry
    and its result is stored.  None: nothing cached and ``run`` is False."""
    path = Path(path)
    key = machine_fingerprint()
    try:
        cache = json.loads(path.read_text())
    except (OSError, ValueError):
        cache = {}
    entry = cache.get(key)
    if entry is not None and "timings" in entry:
        return speed_from_timings(entry["timings"])
    if not run:
        return None
    result = probe()
    cache[key] = {"timings": result["timings"], "date": time.strftime("%Y-%m-%d")}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1) + "\n")
        tmp.replace(path)
    except OSError:
        pass
    return result["speed"]
