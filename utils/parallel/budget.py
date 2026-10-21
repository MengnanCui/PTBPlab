"""How many cores (and how much memory) a PTBP run may use.

The core budget is the smallest of

* the CPU affinity of the process (``taskset``, SLURM/cgroup cpusets),
* the cgroup CPU quota (containers; v2 ``cpu.max`` or v1 ``cpu.cfs_quota_us``,
  rounded up, the tightest along the cgroup hierarchy),
* ``SLURM_CPUS_PER_TASK``,
* the environment variable ``PTBP_NCORES``,
* ``--ncores`` (``ptbp optimize``; ``auto`` or absent = no limit of its own).
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

CGROUP_ROOT = "/sys/fs/cgroup"
PROC_CGROUP = "/proc/self/cgroup"

# OpenMP (DFTB+) and BLAS threads of programs started by a worker
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")
# Thread binding: several DFTB+ processes pinned to the same cores would share them
BINDING_VARS = ("OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY")


def parse_ncores(value) -> int | None:
    """``None``, ``""`` or ``"auto"`` -> None (no limit); a positive integer -> int."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "auto")):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"ncores must be a positive integer or 'auto', not {value!r}") from None
    if n < 1:
        raise ValueError(f"ncores must be a positive integer or 'auto', not {value!r}")
    return n


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _cgroup_dirs(controller: str | None, root: Path, proc_cgroup: Path) -> list[Path]:
    """The process's cgroup directory for ``controller`` (None: v2) and its ancestors."""
    text = _read(proc_cgroup) or ""
    rel = None
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if (controller is None and parts[0] == "0" and parts[1] == "") or \
                (controller is not None and controller in parts[1].split(",")):
            rel = parts[2]
            break
    if rel is None:
        return []
    bases = [root] if controller is None else [root / d for d in (controller, f"{controller},cpuacct",
                                                                      f"cpuacct,{controller}")]
    if controller is None and (root / "unified").is_dir():
        bases.append(root / "unified")               # hybrid hierarchy
    for base in bases:
        if not base.is_dir():
            continue
        start = base / rel.lstrip("/")
        if not start.is_dir():                       # namespaced: we see our own cgroup as root
            start = base
        dirs = [start]
        while dirs[-1] != base and base in dirs[-1].parents:
            dirs.append(dirs[-1].parent)
        return dirs
    return []


def cgroup_cpu_limit(root=CGROUP_ROOT, proc_cgroup=PROC_CGROUP) -> float | None:
    """CPU quota in cores (None: unlimited), the tightest along the hierarchy."""
    root, proc_cgroup = Path(root), Path(proc_cgroup)
    limits = []
    for d in _cgroup_dirs(None, root, proc_cgroup):                    # cgroup v2
        quota = (_read(d / "cpu.max") or "").split()
        if len(quota) == 2 and quota[0] != "max":
            limits.append(int(quota[0]) / int(quota[1]))
    for d in _cgroup_dirs("cpu", root, proc_cgroup):                   # cgroup v1
        quota, period = _read(d / "cpu.cfs_quota_us"), _read(d / "cpu.cfs_period_us")
        if quota and period and int(quota) > 0:
            limits.append(int(quota) / int(period))
    return min(limits) if limits else None


def cgroup_memory_limit(root=CGROUP_ROOT, proc_cgroup=PROC_CGROUP) -> int | None:
    """Memory still available under the cgroup limit, bytes (None: unlimited)."""
    root, proc_cgroup = Path(root), Path(proc_cgroup)
    free = []
    for ctrl, lim_name, use_name in ((None, "memory.max", "memory.current"),
                                     ("memory", "memory.limit_in_bytes", "memory.usage_in_bytes")):
        for d in _cgroup_dirs(ctrl, root, proc_cgroup):
            lim, use = _read(d / lim_name), _read(d / use_name)
            if lim and lim != "max" and int(lim) < 1 << 60:
                free.append(int(lim) - int(use or 0))
    return max(0, min(free)) if free else None


def available_memory(meminfo="/proc/meminfo", **cgroup) -> int | None:
    """Bytes that may still be allocated: MemAvailable, capped by the cgroup limit."""
    avail = None
    for line in (_read(Path(meminfo)) or "").splitlines():
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) * 1024
    limit = cgroup_memory_limit(**cgroup)
    values = [v for v in (avail, limit) if v is not None]
    return min(values) if values else None


def affinity_cores() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@dataclass
class CpuBudget:
    """The cores a run may use and where each limit comes from."""
    ncores: int
    sources: dict = field(default_factory=dict)   # source -> cores it allows (None: no limit)

    @property
    def limited_by(self) -> list[str]:
        return [k for k, v in self.sources.items() if v is not None and math.ceil(v) == self.ncores]

    def describe(self) -> str:
        set_ = [f"{k} {v:g}" for k, v in self.sources.items() if v is not None]
        return f"{self.ncores} core{'s' * (self.ncores != 1)} ({', '.join(set_)})"

    def to_json(self) -> dict:
        return {"ncores": self.ncores, "sources": dict(self.sources)}


def detect_cpu(ncores=None, environ=None, *, affinity: int | None = None,
               cgroup_root=CGROUP_ROOT, proc_cgroup=PROC_CGROUP) -> CpuBudget:
    """The core budget (see the module docstring); ``ncores`` is ``--ncores``."""
    env = os.environ if environ is None else environ
    slurm = env.get("SLURM_CPUS_PER_TASK")
    sources = {
        "CPU affinity": affinity if affinity is not None else affinity_cores(),
        "cgroup quota": cgroup_cpu_limit(cgroup_root, proc_cgroup),
        "SLURM_CPUS_PER_TASK": int(slurm) if slurm and slurm.strip().isdigit() else None,
        "PTBP_NCORES": parse_ncores(env.get("PTBP_NCORES")),
        "--ncores": parse_ncores(ncores),
    }
    n = min(math.ceil(v) for v in sources.values() if v is not None)
    return CpuBudget(max(1, n), sources)


def thread_env(threads: int) -> dict[str, str | None]:
    """Environment of a worker whose programs (DFTB+) may use ``threads`` threads:
    OpenMP and BLAS thread counts set, thread binding removed (None)."""
    env: dict[str, str | None] = {v: str(int(threads)) for v in THREAD_VARS}
    env.update({v: None for v in BINDING_VARS})
    return env


def apply_env(env: dict[str, str | None]) -> None:
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@contextmanager
def applied_env(env: dict[str, str | None]):
    """``env`` in ``os.environ`` for the duration of the block (serial stages)."""
    saved = {k: os.environ.get(k) for k in env}
    apply_env(env)
    try:
        yield
    finally:
        apply_env(saved)
