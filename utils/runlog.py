"""Run logs for long computations (A series): everything a run did, so that a computation done
on a remote server can be checked and repeated later.

Each run gets ``<root>/<name>/<YYYYmmdd-HHMMSS>-<host>-<pid>/`` with

* ``run.log``      everything written to stdout and stderr (tee: the terminal still shows it)
                   plus Python warnings and the traceback of an uncaught error;
* ``meta.json``    command line, working directory, arguments, git commit and dirty flag,
                   host, CPU count, Slurm job, thread variables, Python and package versions;
* ``events.jsonl`` one JSON object per line from :meth:`RunLog.event` and :meth:`RunLog.step`
                   (with wall time), so progress can be followed with ``tail -f``;
* ``status.json``  written at the end: ``ok`` or ``failed``, elapsed seconds, error text.

    from utils.runlog import session
    with session("k1_numerics", vars(args)) as log:
        with log.step("hotcent", element="Li"):
            ...
        log.event("result", mae=0.012)
        log.save_json("summary.json", summary)      # next to run.log

``root`` defaults to ``$PTBP_LOG_ROOT``, else ``./logs``.  Nothing here changes a result.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import warnings

PACKAGES = ("numpy", "scipy", "ase", "pydftb", "hotcent", "skopt", "torch", "mace", "gpaw",
            "tblite", "dftd3", "dscribe")
ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "PTBP_", "PYDFTB_", "VASP_", "SLURM_JOB_ID",
            "SLURM_JOB_NAME", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK", "SLURM_JOB_PARTITION",
            "CONDA_DEFAULT_ENV", "VIRTUAL_ENV", "GPAW_SETUP_PATH")


class _Tee:
    """Write to the original stream and to the log file."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, text):
        self.stream.write(text)
        self.fh.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False

    def fileno(self):                     # some libraries ask; give the terminal's
        return self.stream.fileno()


def _version(name: str):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:                     # noqa: BLE001
        pass
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "imported (no version)")
    except Exception:                     # noqa: BLE001
        return None


def git_state(path: str | None = None) -> dict:
    """Commit, branch and dirty flag of the repository holding ``path`` (default: this file)."""
    path = os.path.dirname(os.path.abspath(path or __file__))
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=path, capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:                 # noqa: BLE001
            return ""
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status), "dirty_files": status.splitlines()[:50]}


def environment() -> dict:
    """Host, CPU, Slurm and thread settings, Python and package versions."""
    env = {k: v for k, v in os.environ.items()
           if any(k == key or (key.endswith("_") and k.startswith(key)) for key in ENV_KEYS)}
    return {"host": socket.gethostname(), "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0], "executable": sys.executable,
            "packages": {p: _version(p) for p in PACKAGES}, "env": env}


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_jsonable(v) for v in obj]
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return repr(obj)


class RunLog:
    """Handle of one run (see the module docstring)."""

    def __init__(self, directory: str, name: str):
        self.dir = directory
        self.name = name
        self.t0 = time.perf_counter()
        self._events = open(os.path.join(directory, "events.jsonl"), "a", buffering=1)

    def path(self, *parts) -> str:
        """A file next to run.log."""
        return os.path.join(self.dir, *parts)

    def event(self, kind: str, **data) -> None:
        rec = {"t": round(time.perf_counter() - self.t0, 3),
               "time": _dt.datetime.now().isoformat(timespec="seconds"), "kind": kind}
        rec.update(_jsonable(data))
        self._events.write(json.dumps(rec) + "\n")
        print(f"[{self.name} {rec['t']:9.1f}s] {kind} "
              + " ".join(f"{k}={v}" for k, v in data.items() if not isinstance(v, (dict, list))),
              flush=True)

    @contextlib.contextmanager
    def step(self, kind: str, **data):
        """Time a block; the event is written at its end, with ``seconds`` and ``ok``."""
        t = time.perf_counter()
        ok = False
        try:
            yield
            ok = True
        finally:
            self.event(kind, seconds=round(time.perf_counter() - t, 3), ok=ok, **data)

    def save_json(self, name: str, obj) -> str:
        path = self.path(name)
        with open(path, "w") as fh:
            json.dump(_jsonable(obj), fh, indent=2)
        return path

    def close(self):
        self._events.close()


@contextlib.contextmanager
def session(name: str, args=None, root: str | None = None, tee: bool = True):
    """Run inside a logged session (see the module docstring); yields a :class:`RunLog`."""
    root = root or os.environ.get("PTBP_LOG_ROOT") or os.path.join(os.getcwd(), "logs")
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = os.path.join(root, name, f"{stamp}-{socket.gethostname().split('.')[0]}-{os.getpid()}")
    os.makedirs(directory, exist_ok=True)
    meta = {"name": name, "start": _dt.datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv, "cwd": os.getcwd(), "args": _jsonable(args or {}),
            "git": git_state(), **environment()}
    with open(os.path.join(directory, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    fh_log = open(os.path.join(directory, "run.log"), "a", buffering=1)
    out, err, showwarning = sys.stdout, sys.stderr, warnings.showwarning
    if tee:
        sys.stdout, sys.stderr = _Tee(out, fh_log), _Tee(err, fh_log)

    def _warn(message, category, filename, lineno, file=None, line=None):
        fh_log.write(f"WARNING {category.__name__}: {message} ({filename}:{lineno})\n")
        showwarning(message, category, filename, lineno, file, line)
    warnings.showwarning = _warn
    log = RunLog(directory, name)
    print(f"[runlog] {name}: logs in {directory}", flush=True)
    status = {"status": "failed", "error": None}
    try:
        yield log
        status["status"] = "ok"
    except BaseException as exc:          # noqa: BLE001  (logged, then re-raised)
        status["error"] = f"{type(exc).__name__}: {exc}"
        fh_log.write(traceback.format_exc())
        raise
    finally:
        status.update(end=_dt.datetime.now().isoformat(timespec="seconds"),
                      seconds=round(time.perf_counter() - log.t0, 1))
        with open(os.path.join(directory, "status.json"), "w") as fh:
            json.dump(status, fh, indent=2)
        log.close()
        sys.stdout, sys.stderr, warnings.showwarning = out, err, showwarning
        fh_log.close()
        print(f"[runlog] {name}: {status['status']} after {status['seconds']} s ({directory})",
              flush=True)
