"""Threads over k-points.

For a periodic structure the expensive steps of a calculation (the reduction to
standard form, the eigensolver and the density matrices) are independent for every
k-point, and NumPy releases the GIL inside BLAS and LAPACK.  :class:`KThreads` cuts
the k stack into contiguous chunks and runs them on a thread pool.  Every matrix
goes through the same LAPACK/BLAS call as in the whole-stack version, so the results
are bitwise identical for any number of threads, as long as BLAS itself runs on one
thread (:class:`pydftb.DFTB` limits it while k threads are in use).  Sums over k (the
Fermi level, populations, energies, forces) are never split.

A pool is shut down before the process forks (``os.register_at_fork``), so worker
processes forked by a task farm (PTBP's ``utils.parallel``) start without stale
threads and build their own pool on first use.
"""
from __future__ import annotations

import os
import weakref
from concurrent.futures import ThreadPoolExecutor

# A chunk is worth handing to a thread once it holds this many N^3 operations
# (about 16 k-points of a 9-orbital Cu atom; one k-point from N = 26 orbitals on).
MIN_CHUNK_WORK = 16 * 9 ** 3

_LIVE: weakref.WeakSet = weakref.WeakSet()


class KThreads:
    """Thread pool over chunks of k-points; ``n = 1`` runs everything inline."""

    def __init__(self, n: int = 1, min_work: int = MIN_CHUNK_WORK):
        self.n = max(1, int(n))
        self.min_work = int(min_work)
        self._pool = None
        self._pid = None

    def chunks(self, nk: int, norb: int) -> list[slice]:
        """Contiguous k ranges: at most one per thread, each worth a hand-off."""
        min_size = max(1, -(-self.min_work // max(1, norb) ** 3))
        nchunk = max(1, min(self.n, nk // min_size))
        bounds = [nk * i // nchunk for i in range(nchunk + 1)]
        return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]

    def map(self, fn, nk: int, norb: int) -> list:
        """``[fn(chunk) for chunk in self.chunks(nk, norb)]``, the chunks run in threads."""
        parts = self.chunks(nk, norb)
        if len(parts) == 1:
            return [fn(parts[0])]
        return list(self._executor().map(fn, parts))

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None or self._pid != os.getpid():
            self._pool = ThreadPoolExecutor(self.n, thread_name_prefix="pydftb-k")
            self._pid = os.getpid()
            _LIVE.add(self)
        return self._pool

    def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None and self._pid == os.getpid():
            pool.shutdown(wait=True)

    def __getstate__(self):                     # pickled (e.g. with a calculator): no pool
        return {"n": self.n, "min_work": self.min_work, "_pool": None, "_pid": None}

    def __repr__(self) -> str:
        return f"KThreads({self.n})"


def _close_all_before_fork() -> None:
    for kt in list(_LIVE):
        try:
            kt.close()
        except Exception:                       # never block a fork
            pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_close_all_before_fork)


def as_kthreads(k_threads) -> KThreads:
    """``KThreads`` from an int, ``"auto"`` (the CPUs this process may use) or a KThreads."""
    if isinstance(k_threads, KThreads):
        return k_threads
    if k_threads is None:
        return KThreads(1)
    if k_threads == "auto":
        n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        return KThreads(n or 1)
    if int(k_threads) < 1:
        raise ValueError(f"k_threads must be >= 1 or 'auto', not {k_threads!r}")
    return KThreads(int(k_threads))
