"""Run a list of independent tasks in forked worker processes.

:func:`map_tasks` returns one :class:`TaskResult` per task, in task order, whatever
order the tasks ran in.  Why not ``multiprocessing.Pool``:

* Pool workers are daemonic and may not start processes of their own; PTBP's workers
  must (DFTB+ runs, hotcent inside a parallel_bo particle);
* a Pool hangs when a worker dies (segmentation fault, out-of-memory kill);
* a Pool pickles the function, and PTBP's functions hang off large objects.

:class:`ForkBackend` forks non-daemonic workers for each call; they inherit the
function and the tasks, and only task numbers and results cross the pipes.  The parent
hands the next task (in ``order``: largest first gives the best balance) to whichever
worker finishes first, watches results and process ends together, records the task
of a worker that died as :class:`WorkerDied`, and returns exceptions with their type
(the nearest built-in type if they cannot be pickled) and the worker's traceback.
Each worker leads its own process group, so the cleanup in ``finally`` (SIGTERM,
then SIGKILL) also ends the DFTB+ processes it started, on errors and on Ctrl-C.

A future MPI backend only needs ``map(fn, tasks, order, init) -> list[TaskResult]``;
tasks that cross process boundaries must then be picklable (the DFTB stage's are).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import wait
from typing import Any, Callable, Sequence

from .budget import apply_env, applied_env, thread_env

TERM_GRACE = 5.0        # seconds between SIGTERM and SIGKILL at cleanup


class WorkerDied(RuntimeError):
    """A worker process ended (crash, kill, out of memory) while it had this task."""


@dataclass
class TaskResult:
    index: int
    ok: bool
    value: Any = None
    error: BaseException | None = None
    wall: float = 0.0             # seconds spent in fn
    pid: int | None = None

    def get(self):
        if self.ok:
            return self.value
        raise self.error


def _portable(exc: BaseException):
    """(exception that survives pickling, traceback text) for the trip to the parent."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        clone = pickle.loads(pickle.dumps(exc))
        if type(clone) is not type(exc):
            clone = None
    except Exception:
        clone = None
    if clone is None:
        base = next((c for c in type(exc).__mro__
                     if c.__module__ == "builtins" and issubclass(c, Exception)), RuntimeError)
        try:
            clone = base(f"{type(exc).__qualname__}: {exc}")
        except Exception:
            clone = RuntimeError(f"{type(exc).__qualname__}: {exc}")
    return clone, tb


def _with_traceback(err: BaseException, tb: str, pid: int) -> BaseException:
    err.remote_traceback = tb
    if hasattr(err, "add_note"):
        err.add_note(f"[raised in worker process {pid}]\n{tb.rstrip()}")
    return err


def _limit_blas(blas_threads: int):
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return None
    return threadpool_limits(blas_threads)


# ---------------------------------------------------------------------------
class SerialBackend:
    """All tasks in this process, in index order."""
    name = "serial"

    def map(self, fn, tasks, order=None, threads: int = 1, blas_threads: int = 1):
        results = []
        with applied_env(thread_env(threads)):
            limits = _limit_blas(blas_threads)
            try:
                for i, task in enumerate(tasks):
                    t0 = time.perf_counter()
                    try:
                        results.append(TaskResult(i, True, fn(task), wall=time.perf_counter() - t0,
                                                  pid=os.getpid()))
                    except Exception as exc:
                        results.append(TaskResult(i, False, error=exc,
                                                  wall=time.perf_counter() - t0, pid=os.getpid()))
            finally:
                if limits is not None:
                    limits.restore_original_limits()
        return results


def _die_with_parent(parent: int) -> None:
    """SIGTERM this worker when the parent process ends (Linux), e.g. killed by SIGKILL."""
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)    # PR_SET_PDEATHSIG
    except Exception:
        pass
    if os.getppid() != parent:
        os._exit(1)


def _end_group(signum, frame):
    """SIGTERM: end the whole process group (this worker and the DFTB+ it started)."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.killpg(0, signal.SIGTERM)


def _worker(fn, tasks, conn, others, parent, threads, blas_threads):
    try:
        os.setpgid(0, 0)
    except OSError:
        pass
    for c in others:                                    # the parent's ends of other pipes
        c.close()
    _die_with_parent(parent)
    signal.signal(signal.SIGTERM, _end_group)
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # Ctrl-C is the parent's to handle
    apply_env(thread_env(threads))
    _limits = _limit_blas(blas_threads)                 # noqa: F841  (kept for the worker's life)
    pid = os.getpid()
    while True:
        try:
            i = conn.recv()
        except EOFError:
            break
        if i is None:
            break
        t0 = time.perf_counter()
        try:
            msg = (i, True, fn(tasks[i]), time.perf_counter() - t0, pid)
        except Exception as exc:
            msg = (i, False, _portable(exc), time.perf_counter() - t0, pid)
        try:
            conn.send(msg)
        except Exception as exc:                        # the result does not pickle
            conn.send((i, False, _portable(exc), msg[3], pid))
    conn.close()
    sys.stdout.flush()
    sys.stderr.flush()


class ForkBackend:
    """Forked, non-daemonic workers fed one task at a time (Linux)."""
    name = "fork"

    def __init__(self, workers: int):
        self.workers = max(1, int(workers))

    def map(self, fn, tasks, order=None, threads: int = 1, blas_threads: int = 1):
        n = len(tasks)
        queue = list(range(n)) if order is None else [int(i) for i in order]
        if sorted(queue) != list(range(n)):
            raise ValueError("order must be a permutation of the task indices")
        results: list[TaskResult | None] = [None] * n
        if n == 0:
            return []
        ctx = mp.get_context("fork")
        procs, conns, current = [], [], []
        try:
            for w in range(min(self.workers, n)):
                here, there = ctx.Pipe(duplex=True)
                p = ctx.Process(target=_worker, name=f"ptbp-worker-{w}", daemon=False,
                                args=(fn, tasks, there, list(conns) + [here], os.getpid(),
                                      threads, blas_threads))
                p.start()
                there.close()
                try:
                    os.setpgid(p.pid, p.pid)            # also here: no race with cleanup
                except OSError:
                    pass
                procs.append(p)
                conns.append(here)
                current.append(None)
            pos = 0

            def feed(w):
                nonlocal pos
                nxt = queue[pos] if pos < n else None
                pos += nxt is not None
                current[w] = nxt
                try:
                    conns[w].send(nxt)
                except OSError:
                    pass                                # dead: noticed through its sentinel

            def receive(w):
                i, ok, payload, wall, pid = conns[w].recv()
                if ok:
                    results[i] = TaskResult(i, True, payload, wall=wall, pid=pid)
                else:
                    err, tb = payload
                    results[i] = TaskResult(i, False, error=_with_traceback(err, tb, pid),
                                            wall=wall, pid=pid)
                current[w] = None

            for w in range(len(procs)):
                feed(w)
            alive = set(range(len(procs)))
            while alive:
                ready = wait([conns[w] for w in alive] + [procs[w].sentinel for w in alive])
                for w in sorted(alive):
                    if conns[w] in ready:
                        try:
                            receive(w)
                            feed(w)
                            continue
                        except (EOFError, OSError):
                            pass                        # pipe closed: the worker ended
                    if procs[w].sentinel in ready or not procs[w].is_alive():
                        try:
                            while conns[w].poll():      # results sent just before the end
                                receive(w)
                        except (EOFError, OSError):
                            pass
                        procs[w].join()
                        _kill_group(procs[w].pid, signal.SIGKILL)
                        if current[w] is not None:
                            i = current[w]
                            results[i] = TaskResult(i, False, error=WorkerDied(
                                f"worker process {procs[w].pid} ended (exit code "
                                f"{procs[w].exitcode}) while running task {i}"))
                            current[w] = None
                        alive.discard(w)
        finally:
            _cleanup(procs)
            for c in conns:
                c.close()
        for i in range(n):
            if results[i] is None:
                results[i] = TaskResult(i, False, error=WorkerDied(
                    f"task {i} was not run: all worker processes ended"))
        return results


def _kill_group(pgid: int, sig) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _cleanup(procs) -> None:
    """End every worker that is still running, with the programs it started."""
    running = [p for p in procs if p.is_alive()]
    for p in running:
        _kill_group(p.pid, signal.SIGTERM)
        p.terminate()
    deadline = time.monotonic() + TERM_GRACE
    for p in running:
        p.join(max(0.0, deadline - time.monotonic()))
        if p.is_alive():
            _kill_group(p.pid, signal.SIGKILL)
            p.kill()
            p.join()


def default_backend(workers: int):
    """Fork on Linux with more than one worker; ``PTBP_BACKEND=serial|fork`` overrides."""
    choice = os.environ.get("PTBP_BACKEND", "").strip().lower()
    if choice not in ("", "serial", "fork"):
        raise ValueError(f"PTBP_BACKEND must be 'serial' or 'fork', not {choice!r}")
    if choice == "serial" or workers <= 1:
        return SerialBackend()
    if choice == "fork" or sys.platform.startswith("linux"):
        return ForkBackend(workers)
    return SerialBackend()


def map_tasks(fn: Callable, tasks: Sequence, workers: int = 1, *, order=None,
              threads: int = 1, blas_threads: int = 1, backend=None) -> list[TaskResult]:
    """``fn(task)`` for every task with up to ``workers`` processes; results in task order.

    ``order``: the order tasks are handed out in (default: as given).  ``threads``: the
    OpenMP/BLAS threads of programs the tasks start (``OMP_NUM_THREADS`` ...);
    ``blas_threads``: BLAS threads inside the worker processes.
    """
    backend = backend or default_backend(workers)
    return backend.map(fn, list(tasks), order=order, threads=threads, blas_threads=blas_threads)
