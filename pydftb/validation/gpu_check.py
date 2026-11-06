"""GPU self-check of pydftb (milestones M4d, M7e): correctness and timings on your device.

    pydftb gpu-check --dtype both --json gpu_check.json    # everything, the JSON to send back
    pydftb gpu-check --device cuda:1 --json gpu_check.json
    pydftb gpu-check --quick                  # a subset, a few minutes
    pydftb gpu-check --device cpu --quick     # the script itself, no GPU
    pydftb gpu-check --timings ptbp,stacked   # only these timings

(``python -m pydftb.validation.gpu_check ...`` is the same.)

Correctness:

* references: frozen DFTB+ cases on the device, with the tolerances of ``refgen --check``;
* cpu: the same cases with torch on the CPU, float64: |dE|, |dF|, |dq|, |deps| <= 1e-10;
* determinism: one calculation twice, bit for bit;
* autodiff: dF/dtheta on the device in modes "hf" and "implicit" and on the CPU, <= 1e-9;
* fork (CUDA): a process that used CUDA forks a CPU-only NumPy calculation, which must end.

``--dtype float32`` (or ``both``) adds float32 runs, reported without a pass/fail.

Timings (reported only; ``--timings`` picks them, ``--skip-timing`` runs none), in the
precisions of ``--dtype``: float64 times float64 and complex128 matrices, float32 times float32
and complex64; a type the device lacks is skipped.

* ``eigh``: N = 9-288 orbitals and batches of 1-2048 matrices, the device against NumPy;
* ``cu``: Cu cells computed as PTBP computes them, NumPy on one thread and with k-point
  threads against the device;
* ``ptbp``: PTBP's Cu DFTB stage (:mod:`pydftb.validation.ptbp_workload`: 66 structures, 18
  with ``--quick``) structure by structure, NumPy on one thread, with k-point threads, in a
  pool of ``--workers`` processes (default: the usable cores) and on the device; the work the
  device run leaves on the CPU, and the bound a pipeline of CPU processes and the device
  could reach (milestone M7e);
* ``stacked``: the whole solve of an SCC iteration (Cholesky factor, eigh, Fermi level,
  densities, populations) for the structures of a group (equal N, real or complex), stacked
  against one structure at a time, with their deviations; an eigh-only column as in M4d.

The JSON (``--json``) is rewritten after every section: a section that fails records its
error and traceback and the next one runs, and an interrupted run (Ctrl-C) leaves what was
measured with ``"complete": false``.  Exit status: 0 everything passed, 1 a check failed or a
section could not run, 2 interrupted.  Send the JSON back: it holds the environment, every
number and the verdicts.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import heapq
import json
import os
import platform
import subprocess
import sys
import textwrap
import time
import traceback

import numpy as np

SCHEMA = 2                   # the JSON layout (1: M4d)
CPU_TOL = 1e-10              # device vs CPU, float64 (Ha, Ha/Bohr, e)
F32_SCC_TOL = 1e-5           # float32 (M4a); at 1e-6 the mixer of some DFTB3 cases diverges
GRAD_TOL = 1e-9              # relative, gradients
QUICK_CASES = ["h2o", "ch3oh_dftb3", "cu_fcc_k6_T0", "cu8_distorted_k355_mp2_2000K",
               "cuo_chain_spin_k422", "gdo_rocksalt_k444", "water_box_d3bj_3body"]
AUTODIFF_CASES = ["h2o", "cu4co_shell_resolved", "cuo_chain_spin_k422"]
DETERMINISM_CASES = ["ch3oh_dftb3", "cu8_distorted_k355_mp2_2000K"]
SKIP = None                  # a check that does not apply (e.g. fork on the CPU)
TIMING_SECTIONS = ("eigh", "cu", "ptbp", "stacked")
KINDS = {"float64": ("float64", "complex128"), "float32": ("float32", "complex64")}


@dataclasses.dataclass
class Check:
    name: str
    ok: bool | None          # None: not applicable / report only
    details: dict

    def line(self) -> str:
        mark = {True: "ok", False: "FAILED", None: "-"}[self.ok]
        info = ", ".join(f"{k} {_fmt(v)}" for k, v in self.details.items()
                         if not isinstance(v, (dict, list)))
        return f"  {self.name:44s} {mark:7s} {info}"


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.2e}" if v and (abs(v) < 1e-2 or abs(v) >= 1e4) else f"{v:.4g}"
    return str(v)


def _silent(*args, **kw):
    pass


def _sync(device: str) -> None:
    import torch
    kind = device.split(":")[0]
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()
    elif kind == "xpu":
        torch.xpu.synchronize()


def _usable_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


# ---- the report: written after every section ------------------------------------------------
def _jsonable(x):
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def write_report(report: dict, path) -> None:
    """``report`` as JSON at ``path``, atomically (a temporary file next to it, with the
    permissions of a new file, then ``os.replace``): an interrupted write leaves the previous
    version."""
    if not path:
        return
    path = os.path.abspath(path)
    tmp = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as fh:
            json.dump(report, fh, indent=1, default=_jsonable)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _is_oom(err) -> bool:
    return type(err).__name__ == "OutOfMemoryError" or "out of memory" in str(err).lower()


def _empty_cache(device: str) -> None:
    import torch
    kind = device.split(":")[0]
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        torch.mps.empty_cache()


def _memory_reset(device: str) -> None:
    if device.startswith("cuda"):
        import torch
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def _memory_peak(device: str) -> dict:
    out = {}
    with contextlib.suppress(Exception):
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss      # KiB (bytes on macOS)
        out["max_rss_gb"] = rss / (2 ** 30 if sys.platform == "darwin" else 2 ** 20)
    if device.startswith("cuda"):
        import torch
        out["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated(torch.device(device)) / 2 ** 30
    return out


def _guard(name: str, fn, device: str, log):
    """(fn(), record): the section's wall time and memory peaks, and on an error its message
    and traceback instead of the result (None); out of device memory: the cache is emptied.
    KeyboardInterrupt passes."""
    rec = {"section": name}
    with contextlib.suppress(Exception):
        _memory_reset(device)
    t0 = time.perf_counter()
    try:
        out = fn()
        rec["ok"] = True
    except Exception as err:                    # noqa: BLE001  recorded, the next section runs
        out = None
        rec.update(ok=False, error=f"{type(err).__name__}: {err}",
                   traceback=traceback.format_exc(limit=30))
        if _is_oom(err):
            rec["out_of_memory"] = True
            with contextlib.suppress(Exception):
                _empty_cache(device)
        log(f"[gpu_check] {name} could not run: {rec['error']}")
    rec["wall_s"] = time.perf_counter() - t0
    with contextlib.suppress(Exception):
        rec.update(_memory_peak(device))
    return out, rec


# ---- environment ----------------------------------------------------------------------------
def _physical_cores():
    """Distinct (physical id, core id) pairs of /proc/cpuinfo; None elsewhere."""
    try:
        cores, phys, core = set(), None, None
        with open("/proc/cpuinfo") as fh:
            for line in list(fh) + [""]:
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip():
                    if core is not None:
                        cores.add((phys, core))
                    phys = core = None
        return len(cores) or None
    except OSError:
        return None


def _threadpools():
    """The BLAS and OpenMP libraries threadpoolctl sees, with their threads."""
    try:
        from threadpoolctl import threadpool_info
        keys = ("user_api", "internal_api", "num_threads", "version", "threading_layer",
                "architecture", "prefix")
        return [{k: p.get(k) for k in keys if k in p} for p in threadpool_info()]
    except Exception as err:                    # noqa: BLE001
        return f"{type(err).__name__}: {err}"


def _git_commit():
    """(commit, uncommitted changes in the package) of the git checkout that tracks this
    file; (None, None) for an installed copy, even inside another repository, or without git."""
    here = os.path.abspath(__file__)
    where = os.path.dirname(here)

    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=where,
                              timeout=10, check=False)
    try:
        if git("ls-files", "--error-unmatch", here).returncode != 0:
            return None, None
        rev = git("rev-parse", "--short", "HEAD").stdout.strip() or None
        dirty = git("status", "--porcelain", "--", os.path.dirname(where)).stdout.strip()
        return rev, bool(dirty)
    except Exception:                           # noqa: BLE001  no git
        return None, None


def _nvidia_driver():
    try:
        run = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30, check=False)
        lines = run.stdout.strip().splitlines()
        return lines[0].strip() if run.returncode == 0 and lines else None
    except Exception:                           # noqa: BLE001  not installed
        return None


def environment(device: str, argv=None) -> dict:
    import torch

    from .. import __version__
    from ..backend import devices
    env = {"python": platform.python_version(), "numpy": np.__version__,
           "torch": torch.__version__, "cuda_version": torch.version.cuda,
           "hip_version": getattr(torch.version, "hip", None), "platform": platform.platform(),
           "cpu": devices._cpu_name(), "cpu_count": os.cpu_count(),
           "usable_cpus": _usable_cpus(), "physical_cores": _physical_cores(), "device": device,
           "capabilities": devices.capabilities(device).to_json(), "pydftb": __version__,
           "pydftb_path": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
           "command_line": list(sys.argv if argv is None else argv)}
    for name in ("scipy", "ase", "threadpoolctl"):
        try:
            env[name] = __import__(name).__version__
        except Exception:                       # noqa: BLE001
            env[name] = None
    env["threadpools"] = _threadpools()
    env["torch_threads"] = torch.get_num_threads()
    with contextlib.suppress(Exception):
        env["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    env["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    if device.startswith("cuda"):
        dev = torch.device(device)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        env.update(gpu=prop.name, gpu_memory_gb=round(prop.total_memory / 2 ** 30, 2),
                   compute_capability=f"{prop.major}.{prop.minor}",
                   multiprocessors=prop.multi_processor_count,
                   cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        with contextlib.suppress(Exception):
            env["gpu_free_memory_gb"] = round(torch.cuda.mem_get_info(idx)[0] / 2 ** 30, 2)
        with contextlib.suppress(Exception):
            env["linalg_library"] = str(torch.backends.cuda.preferred_linalg_library())
        with contextlib.suppress(Exception):
            env["allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        env["driver"] = None if env["hip_version"] else _nvidia_driver()
    env["pydftb_commit"], env["pydftb_dirty"] = _git_commit()
    return env


# ---- correctness ----------------------------------------------------------------------------
def _cases(names):
    from . import refgen
    from .cases import CASES
    out = []
    for case in CASES:
        if names and case.name not in names:
            continue
        if not case.torch:                      # NumPy-only physics (M6a, M6b, M6c, M6d)
            continue
        if not os.path.exists(os.path.join(refgen.REF_DIR, f"{case.name}.json")):
            continue
        doc = refgen.load(case.name)
        if refgen.missing_requirement(doc) is None:
            out.append(doc)
    return out


def _diffs(a, b) -> dict:
    d = {"energy": abs(a.energy - b.energy), "forces": float(np.abs(a.forces - b.forces).max()),
         "charges": float(np.abs(a.charges - b.charges).max())}
    if a.eigenvalues is not None and b.eigenvalues is not None:
        d["eigenvalues"] = float(np.abs(np.array(a.eigenvalues) - np.array(b.eigenvalues)).max())
    return d


def compare_case(doc, device: str, dtype: str) -> tuple[dict, dict, object]:
    """(deviations from the frozen reference with refgen's verdict, device - CPU torch,
    the device result)."""
    from . import refgen
    tol = {"scc_tol": F32_SCC_TOL} if dtype == "float32" else {}
    r_dev = refgen.compare(doc, backend="torch", device=device, dtype=dtype, **tol)
    r_cpu = refgen.compare(doc, backend="torch", device="cpu", dtype=dtype, **tol)
    return ({"ok": r_dev["ok"], "wall": r_dev["wall"], "n_scc": r_dev["n_scc"],
             "n_scc_ref": r_dev["n_scc_ref"], **r_dev["dev"]},
            _diffs(r_dev["result"], r_cpu["result"]), r_dev["result"])


def check_references(device: str, dtype: str, names) -> list[Check]:
    """Frozen DFTB+ references on the device, and the device against the CPU."""
    out = []
    for doc in _cases(names):
        name = f"{doc['case']} [{dtype}]"
        try:
            ref, cpu, _ = compare_case(doc, device, dtype)
        except Exception as err:                # noqa: BLE001  e.g. an operation missing
            out.append(Check(f"references {name}", False if dtype == "float64" else SKIP,
                             {"error": f"{type(err).__name__}: {err}"}))
            continue
        if dtype == "float64":
            out.append(Check(f"references {name}", bool(ref["ok"]),
                             {k: ref[k] for k in ("energy", "forces", "charges", "n_scc",
                                                  "n_scc_ref", "wall")}))
            out.append(Check(f"cpu {name}", max(cpu.values()) <= CPU_TOL, cpu))
        else:                                   # float32: reported only
            out.append(Check(f"references {name}", SKIP,
                             {k: ref[k] for k in ("energy", "forces", "charges", "wall")}))
            out.append(Check(f"cpu {name}", SKIP, cpu))
    return out


def check_determinism(device: str, dtype: str, names=None) -> list[Check]:
    """One calculation twice on the device, bit for bit (``names``: DETERMINISM_CASES)."""
    from . import refgen
    names = DETERMINISM_CASES if names is None else names
    out = []
    for doc in _cases(names) if names else []:
        case = refgen.CASES_BY_NAME[doc["case"]]
        tol = {"scc_tol": F32_SCC_TOL} if dtype == "float32" else {}
        runs = [refgen.run_pydftb(case, doc["input"], backend="torch", device=device,
                                  dtype=dtype, **tol) for _ in range(2)]
        same = {f: np.asarray(getattr(runs[0], f)).tobytes()
                == np.asarray(getattr(runs[1], f)).tobytes()
                for f in ("energy", "forces", "charges", "eigenvalues")}
        out.append(Check(f"determinism {doc['case']} [{dtype}]", all(same.values()),
                         {k: "same" if v else "differs" for k, v in same.items()}))
    return out


def _grads(calc, structure, mode, device):
    import torch

    from ..autodiff import Params, evaluate
    from . import gradcheck
    species, pos, cell, pbc = structure
    params = Params.from_skset(calc.sk, calc.repulsive, calc.hubbard_derivs,
                               calc.spin_constants, device=device)
    params.requires_grad_(*[k for k in gradcheck.available(calc)
                            if gradcheck.entries(params, k)])
    res = evaluate(calc, species, pos, cell, pbc, params, mode=mode, forces=False)
    leaves = params.leaves()
    g = torch.autograd.grad(res.free_energy, leaves, allow_unused=True)
    return [(torch.zeros_like(x) if y is None else y).detach().cpu() for x, y in zip(leaves, g)]


def _rel(a, b) -> float:
    worst = 0.0
    for x, y in zip(a, b):
        scale = max(float(x.abs().max()), 1e-12)
        worst = max(worst, float((x - y).abs().max()) / scale)
    return worst


def check_autodiff(device: str, names=AUTODIFF_CASES) -> list[Check]:
    """dF/dtheta of every parameter kind: the engine and the graph on the device, modes hf
    and implicit, against the NumPy engine with the graph on the CPU."""
    from . import gradcheck
    out = []
    for name in names:
        try:
            calc_dev, structure = gradcheck.reference_engine(name, backend="torch",
                                                             device=device)
            calc_cpu, _ = gradcheck.reference_engine(name)
            hf = _grads(calc_dev, structure, "hf", device)
            impl = _grads(calc_dev, structure, "implicit", device)
            cpu = _grads(calc_cpu, structure, "hf", "cpu")
            d = {"hf_vs_implicit": _rel(hf, impl), "device_vs_cpu": _rel(hf, cpu)}
            out.append(Check(f"autodiff {name}", max(d.values()) <= GRAD_TOL, d))
        except Exception as err:                # noqa: BLE001
            out.append(Check(f"autodiff {name}", False,
                             {"error": f"{type(err).__name__}: {err}"}))
    return out


_FORK_PROBE = textwrap.dedent("""
    import os, sys, time
    import numpy as np
    from pydftb.validation import refgen
    doc = refgen.load("h2o")
    case = refgen.CASES_BY_NAME[doc["case"]]
    refgen.run_pydftb(case, doc["input"], backend="torch", device=sys.argv[1])  # CUDA in use
    pid = os.fork()
    if pid == 0:                                       # the child: CPU only, NumPy
        try:
            refgen.run_pydftb(case, doc["input"])
            os._exit(0)
        except BaseException:
            os._exit(3)
    t0 = time.time()
    while time.time() - t0 < float(sys.argv[2]):
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            sys.exit(0 if os.waitstatus_to_exitcode(status) == 0 else 4)
        time.sleep(0.2)
    os.kill(pid, 9)
    sys.exit(5)                                        # the child hung
""")


def check_fork(device: str, timeout: float = 60.0) -> Check:
    """After CUDA has been used, a forked child that only computes on the CPU must end
    (what PTBP does: hotcent workers forked by a process that runs the DFTB stage on a GPU)."""
    if not device.startswith("cuda") or not hasattr(os, "fork"):
        return Check("fork after CUDA", SKIP, {"reason": "CUDA only (fork)"})
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        run = subprocess.run([sys.executable, "-c", _FORK_PROBE, device, str(timeout)],
                             capture_output=True, text=True, timeout=timeout + 120, env=env)
        code = run.returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    meaning = {0: "the child ended", 3: "the child failed", 4: "the child failed",
               5: "the child hung"}.get(code, "the probe failed")
    return Check("fork after CUDA", code == 0, {"result": meaning, "code": code})


# ---- timings: eigh and Cu cells -------------------------------------------------------------
def timing_kinds(cap, dtypes) -> tuple[list, list]:
    """(matrix types to time, skipped ones with the reason) for the ``--dtype`` choice:
    float64 -> float64, complex128; float32 -> float32, complex64."""
    kinds, skipped = [], []
    for d in dtypes:
        for k in KINDS[d]:
            if k in ("float64", "complex128") and not cap.float64:
                skipped.append({"dtype": k, "reason": "no float64 on the device"})
            elif k.startswith("complex") and not cap.complex_:
                skipped.append({"dtype": k, "reason": "no complex arithmetic on the device"})
            else:
                kinds.append(k)
    return kinds, skipped


def _hermitian(rng, batch, n, complex_):
    a = rng.normal(size=(batch, n, n))
    if complex_:
        a = a + 1j * rng.normal(size=(batch, n, n))
    return 0.5 * (a + np.conj(np.swapaxes(a, 1, 2)))


def _best(fn, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def time_eigh(device: str, sizes, batches, dtypes, max_bytes: float = 2 ** 30,
              repeat: int = 3) -> list[dict]:
    """``eigh`` of random Hermitian stacks on the device and with NumPy (all BLAS threads);
    a row that fails records its error."""
    import torch
    rng = np.random.default_rng(0)
    tdt = {"float64": torch.float64, "complex128": torch.complex128,
           "float32": torch.float32, "complex64": torch.complex64}
    ndt = {"float64": np.float64, "complex128": np.complex128, "float32": np.float32,
           "complex64": np.complex64}
    rows = []
    for name in dtypes:
        for n in sizes:
            for batch in batches:
                if batch * n * n * np.dtype(ndt[name]).itemsize * 4 > max_bytes:
                    continue
                row = {"dtype": name, "n": n, "batch": batch}
                try:
                    h = _hermitian(rng, batch, n, name.startswith("complex")).astype(ndt[name])
                    t = torch.as_tensor(h).to(device=device, dtype=tdt[name])

                    def dev():
                        torch.linalg.eigh(t)
                        _sync(device)
                    dev()                               # warm-up (kernels, handles)
                    row["device_s"] = _best(dev, repeat)
                    row["numpy_s"] = _best(lambda: np.linalg.eigh(h), repeat)
                    row["speedup"] = row["numpy_s"] / max(row["device_s"], 1e-12)
                except Exception as err:                # noqa: BLE001
                    row["error"] = f"{type(err).__name__}: {err}"
                    if _is_oom(err):
                        _empty_cache(device)
                rows.append(row)
    return rows


CU_CELLS = [("Cu1 k14", "prim", (1, 1, 1), (14, 14, 14)),
            ("Cu2 k12x12x6", "prim", (1, 1, 2), (12, 12, 6)),
            ("Cu4 k8", "cubic", (1, 1, 1), (8, 8, 8)),
            ("Cu8 k6", "prim", (2, 2, 2), (6, 6, 6)),
            ("Cu32 k3", "cubic", (2, 2, 2), (3, 3, 3))]
TIMING_MAX_SCC = 100         # enough to time; larger cells at T = 0 may not converge in 1000


def time_cu(device: str, dtypes=("float64",), quick: bool = False, repeat: int = 1) -> list[dict]:
    """Cu cells with PTBP's settings (PyDFTB's batched solver, T = 0, shell-resolved SCC to
    1e-5, the 0.01 Bohr finite-difference forces, polynomial repulsive) but at most
    TIMING_MAX_SCC iterations: NumPy on one thread, NumPy with k-point threads over the
    usable cores, and the device in each of ``dtypes`` (tags ``device_float64`` ...).  The SCC
    iteration counts may differ between them (metals at T = 0), hence also the time per
    iteration."""
    from ase.build import bulk
    from threadpoolctl import threadpool_limits

    from .. import DFTB, ParamSet
    from ..constants import BOHR
    from ..kpoints import monkhorst_pack, reduce_time_reversal
    from .cases import SKF_SETS
    dtypes = (dtypes,) if isinstance(dtypes, str) else tuple(dtypes)
    params = ParamSet(SKF_SETS["ptbp_hcocu"], shell_resolved=True)
    sk = params.skset(["Cu"])
    rows = []
    for label, kind, rep, grid in (CU_CELLS[:3] if quick else CU_CELLS):
        atoms = bulk("Cu", "fcc", a=3.61, cubic=kind == "cubic") * rep
        kpts = None
        if grid is not None:
            k, w = reduce_time_reversal(*monkhorst_pack(grid))
            kpts = (k, w / w.sum())
        common = dict(kpts=kpts, smearing=("fermi", 0.0), scc_tol=1e-5,
                      max_scc=TIMING_MAX_SCC, warn_unconverged=False, solver="batched",
                      repulsive="polynomial", fd_delta=0.01)
        sym = atoms.get_chemical_symbols()
        args = (sym, atoms.positions / BOHR, atoms.cell.array / BOHR, np.ones(3, bool))
        row = {"cell": label, "atoms": len(atoms), "kpoints": 1 if kpts is None else len(kpts[0])}
        res = {}
        tags = [("numpy_1", {}), ("numpy_kthreads", {"k_threads": _usable_cpus()})]
        tags += [(f"device_{d}", {"backend": "torch", "device": device, "dtype": d})
                 for d in dtypes]
        for tag, opts in tags:
            try:
                calc = DFTB(sk, **common, **opts)

                def run():
                    if tag == "numpy_1":
                        with threadpool_limits(1):
                            res[tag] = calc.calculate(*args)
                    else:
                        res[tag] = calc.calculate(*args)
                    if tag.startswith("device"):
                        _sync(device)
                row[f"{tag}_s"] = _best(run, repeat)
                row[f"{tag}_n_scc"] = res[tag].n_scc
                row[f"{tag}_converged"] = bool(res[tag].converged)
                row[f"{tag}_per_scc_s"] = row[f"{tag}_s"] / max(res[tag].n_scc, 1)
            except Exception as err:             # noqa: BLE001
                res.pop(tag, None)
                row[f"{tag}_error"] = f"{type(err).__name__}: {err}"
                if _is_oom(err):
                    _empty_cache(device)
        for d in dtypes:
            if "numpy_1" in res and f"device_{d}" in res:
                row[f"energy_diff_{d}"] = abs(res["numpy_1"].energy - res[f"device_{d}"].energy)
        rows.append(row)
    return rows


# ---- timings: PTBP's DFTB stage (M7e) -------------------------------------------------------
STAGE_MODES = ("numpy_1", "numpy_kthreads", "numpy_procs")    # then device_<dtype>


@contextlib.contextmanager
def _fill_timer():
    """The time spent in the Fermi level and the occupations: :func:`pydftb.occupations.fill`
    as :mod:`pydftb.solvers` calls it, on the host also with the torch backend."""
    from .. import solvers
    inner, acc = solvers.fill, [0.0]

    def timed(*args, **kw):
        t0 = time.perf_counter()
        try:
            return inner(*args, **kw)
        finally:
            acc[0] += time.perf_counter() - t0
    solvers.fill = timed
    try:
        yield acc
    finally:
        solvers.fill = inner


def _run_structure(s, sync: str | None = None, **overrides) -> dict:
    """One structure as PTBP's DFTB stage computes it: a new calculator (``overrides``: e.g.
    ``backend``, ``device``, ``dtype``), then ``atoms.get_forces()`` (energy, forces and,
    periodic, the stress); ``sync``: the device to synchronise before the clock stops.  Its
    wall time, the engine's own timings, the time in ``fill`` and the results (atomic units)."""
    from .ptbp_workload import calculator
    atoms = s.atoms.copy()
    with _fill_timer() as fill:
        t0 = time.perf_counter()
        atoms.calc = calculator(s, **overrides)
        atoms.get_forces()
        if sync is not None:
            _sync(sync)
        wall = time.perf_counter() - t0
    r = atoms.calc.raw
    return {"wall_s": wall, "fill_s": fill[0],
            "nk": 1 if r.kweights is None else len(r.kweights),
            "n_scc": int(r.n_scc), "converged": bool(r.converged),
            "timings": {k: (float(v) if isinstance(v, float) else v) for k, v in r.timings.items()},
            "energy": float(r.energy), "forces": np.array(r.forces),
            "stress": None if r.stress is None else np.array(r.stress)}


def _pool_init(barrier, warm) -> None:
    """A worker of the pool: modules imported and one small structure computed (the SK files
    read), then it waits for the other workers, so the timed part starts with all of them."""
    with contextlib.suppress(Exception):        # the timed runs record an error
        if warm is not None:
            _run_structure(warm)
    if barrier is not None:
        barrier.wait(timeout=900)


def _pool_ping() -> int:
    return os.getpid()


def _pool_task(s) -> dict:
    try:
        row = _run_structure(s)
    except Exception as err:                    # noqa: BLE001
        row = {"error": f"{type(err).__name__}: {err}"}
    row["pid"] = os.getpid()
    return row


def _estimate(s) -> float:
    """PTBP's order without measured times: k-points x orbitals^3."""
    return float(np.prod(s.kpts)) * float(s.norb) ** 3


def _lpt(times, workers: int) -> float:
    """The makespan of list scheduling, largest first, on ``workers`` processes."""
    loads = [0.0] * max(1, int(workers))
    for t in sorted(times, reverse=True):
        heapq.heapreplace(loads, loads[0] + t)
    return max(loads)


def _pool_stage(structs, workers: int, order) -> tuple[dict, float]:
    """The structures in ``order`` on a pool of ``workers`` spawned processes (one BLAS
    thread each, as PTBP's parallel stage), after every worker has warmed up; ({index: row},
    the makespan from the first submission to the last result)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(workers)
    warm = min(structs, key=_estimate)
    rows = {}
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_pool_init,
                             initargs=(barrier, warm)) as pool:
        for f in [pool.submit(_pool_ping) for _ in range(workers)]:
            f.result()
        t0 = time.perf_counter()
        futures = {pool.submit(_pool_task, structs[i]): i for i in order}
        for f in as_completed(futures):
            rows[futures[f]] = f.result()
        makespan = time.perf_counter() - t0
    return rows, makespan


def _stage_order(structs, rows) -> list:
    """Largest first: the measured NumPy times if every structure has one, else estimates."""
    walls = [r["modes"].get("numpy_1", {}).get("wall_s") for r in rows]
    keys = walls if all(w is not None for w in walls) else [_estimate(s) for s in structs]
    return sorted(range(len(structs)), key=lambda i: -keys[i])


def time_ptbp_stage(device: str, dtypes=("float64",), quick: bool = False, workers=None,
                    log=None, structs=None, modes=None) -> dict:
    """PTBP's Cu DFTB stage (:mod:`pydftb.validation.ptbp_workload`) structure by structure,
    each as PTBP computes it (a new calculator, energy, forces, stress):

    * ``numpy_1``: NumPy on one thread, one structure after the other;
    * ``numpy_kthreads``: NumPy with ``workers`` k-point threads;
    * ``numpy_procs``: a pool of ``workers`` spawned processes, one thread each, largest
      structure first, every worker warmed up before the clock starts;
    * ``device_<dtype>``: the torch backend on ``device`` (one torch thread, as PTBP's stage on
      a GPU), after one untimed structure (its time is ``cold_start_s``).

    Per structure and mode: wall time, the engine's timings, SCC iterations, convergence, the
    time in ``fill`` and |dE|, |dF|, |dS| against ``numpy_1`` (atomic units).  The summary
    adds per mode the sums and, for the device, the work it leaves on the CPU, Σ(wall - diag +
    fill), the device's share Σ(diag - fill), the bound max(CPU work / workers, device solve) of a
    pipeline of the two, and the ratios to the pool and to ``numpy_1`` / workers.  The pool
    uses the spawn start method, so a script that calls this guards its main code with
    ``if __name__ == "__main__":`` (``pydftb gpu-check`` does)."""
    from ..backend import devices
    from .ptbp_workload import structures
    log = log or _silent
    structs = structures(quick=quick) if structs is None else list(structs)
    workers = int(workers or _usable_cpus())
    cap = devices.capabilities(device)
    every = list(STAGE_MODES) + [f"device_{d}" for d in dtypes]
    modes = every if modes is None else [m for m in every if m in modes]
    rows = [{"name": s.name, "phase": s.phase, "eos": s.eos, "atoms": len(s.atoms),
             "norb": s.norb, "grid": list(s.kpts), "modes": {}} for s in structs]
    results, pool, cold = {}, {}, {}

    def store(i, mode, out):
        res = (out.pop("energy", None), out.pop("forces", None), out.pop("stress", None))
        if "wall_s" in out:
            out["energy"] = res[0]
            results[mode, i] = res
            ref = results.get(("numpy_1", i))
            if mode != "numpy_1" and ref is not None:
                out["dE"] = abs(res[0] - ref[0])
                out["dF"] = float(np.abs(res[1] - ref[1]).max())
                if res[2] is not None and ref[2] is not None:
                    out["dS"] = float(np.abs(res[2] - ref[2]).max())
        rows[i]["modes"][mode] = out

    for mode in modes:
        log(f"[gpu_check] PTBP stage, {mode}: {len(structs)} structures ...")
        if mode == "numpy_procs":
            order = _stage_order(structs, rows)
            try:
                got, makespan = _pool_stage(structs, workers, order)
                pool = {"makespan_s": makespan, "order": [structs[i].name for i in order]}
            except Exception as err:            # noqa: BLE001  e.g. spawn not possible
                got, pool = {}, {"error": f"{type(err).__name__}: {err}"}
            for i, out in got.items():
                store(i, mode, out)
            continue
        dev, opts = None, {}
        if mode == "numpy_kthreads":
            opts = {"k_threads": workers}
        elif mode.startswith("device_"):
            dev = device
            opts = {"backend": "torch", "device": device, "dtype": mode[len("device_"):]}
        warm = next((i for i, s in enumerate(structs) if s.gamma), 0)
        try:                                    # kernels, handles, the SK files
            cold[mode] = _run_structure(structs[warm], dev, **opts)["wall_s"]
        except Exception as err:                # noqa: BLE001  the timed runs record it
            cold[mode] = f"{type(err).__name__}: {err}"
        for i, s in enumerate(structs):
            if dev is not None and not s.gamma and not cap.complex_:
                rows[i]["modes"][mode] = {"skipped": "no complex arithmetic on the device"}
                continue
            try:
                out = _run_structure(s, dev, **opts)
            except Exception as err:            # noqa: BLE001
                out = {"error": f"{type(err).__name__}: {err}"}
                if _is_oom(err):
                    _empty_cache(device)
            store(i, mode, out)
    summary = _stage_summary(rows, modes, workers, pool, cold)
    for mode in modes:
        s = summary[mode]
        log(f"  {mode:16s} {s['structures']:3d} structures  {s['sum_s']:8.2f} s"
            + (f"  makespan {s['makespan_s']:.2f} s" if "makespan_s" in s else "")
            + (f"  CPU work {s['cpu_work_s']:.2f} s, device solve {s['device_solve_s']:.2f} s"
               if "cpu_work_s" in s else "")
            + (f"  errors {s['errors']}" if s["errors"] else ""))
    return {"workers": workers, "modes": modes, "structures": rows, "pool": pool,
            "summary": summary}


def _stage_summary(rows, modes, workers: int, pool: dict, cold: dict) -> dict:
    out = {"workers": workers}
    for mode in modes:
        done = [r["modes"][mode] for r in rows if "wall_s" in r["modes"].get(mode, {})]
        s = {"structures": len(done),
             "errors": sum("error" in r["modes"].get(mode, {}) for r in rows),
             "skipped": sum("skipped" in r["modes"].get(mode, {}) for r in rows),
             "sum_s": sum(d["wall_s"] for d in done),
             "mean_s": sum(d["wall_s"] for d in done) / max(len(done), 1),
             "n_scc": sum(d["n_scc"] for d in done),
             "unconverged": sum(not d["converged"] for d in done),
             "fill_s": sum(d["fill_s"] for d in done)}
        for key in ("dE", "dF", "dS"):
            vals = [d[key] for d in done if key in d]
            if vals:
                s[f"max_{key}"] = max(vals)
        if mode in cold:
            s["cold_start_s"] = cold[mode]
        if mode == "numpy_procs":
            s.update({k: v for k, v in pool.items() if k != "order"})
            if done:
                s["lpt_makespan_s"] = _lpt([d["wall_s"] for d in done], workers)
        if mode == "numpy_1" and done:
            s["lpt_makespan_s"] = _lpt([d["wall_s"] for d in done], workers)
        if mode.startswith("device_") and done:
            cpu = sum(d["wall_s"] - d["timings"].get("diag", 0.0) + d["fill_s"] for d in done)
            solve = sum(d["timings"].get("diag", 0.0) - d["fill_s"] for d in done)
            s.update(cpu_work_s=cpu, device_solve_s=solve,
                     cpu_work_share=cpu / max(s["sum_s"], 1e-12),
                     pipeline_bound_s=max(cpu / workers, solve))
        out[mode] = s
    makespan = out.get("numpy_procs", {}).get("makespan_s")
    n1 = out.get("numpy_1", {}).get("sum_s")
    for mode in modes:
        s = out[mode]
        if not mode.startswith("device_") or not s["structures"]:
            continue
        if makespan:
            s["vs_pool"] = s["sum_s"] / makespan
            s["pipeline_vs_pool"] = s["pipeline_bound_s"] / makespan
        if n1:
            s["vs_numpy_1_over_workers"] = s["sum_s"] / (n1 / workers)
    return out


# ---- timings: the whole solve, stacked over structures (M7e) --------------------------------
STACK_GROUPS = (("gamma N=18 real", ("dimer", "AA-graphite", "diamond")),
                ("beta-Sn N=18 complex", ("beta-Sn",)),
                ("bcc+fcc N=9 complex", ("bcc", "fcc")))
STACK_COPIES = 8             # EOS copies of every Cu cell of the N scan
STACK_TOL = 1e-12            # stacked against one at a time, float64 (Ha, e)


def _cu_scan(quick: bool) -> list:
    """(name, structures): STACK_COPIES EOS volumes of every cell of CU_CELLS."""
    from ase.build import bulk

    from .ptbp_workload import Structure, volume_ratio
    out = []
    for label, kind, rep, grid in (CU_CELLS[:3] if quick else CU_CELLS):
        base = bulk("Cu", "fcc", a=3.61, cubic=kind == "cubic") * rep
        group = []
        for i in range(1, STACK_COPIES + 1):
            a = base.copy()
            a.set_cell(base.cell.array * volume_ratio(i) ** (1.0 / 3.0), scale_atoms=True)
            group.append(Structure(f"{label} {i:02d}", label, i, a, tuple(grid), 9 * len(a)))
        out.append((f"{label} N={9 * len(base)} complex", group))
    return out


def _setups(structs, device: str, dtype: str) -> list:
    """(engine, geometry setup) of every structure with the torch backend: H0 and S on the
    device as the SCC of :class:`pydftb.DFTB` uses them.  Built without SCC: the same H0, S,
    k weights, electrons and orbital sites, without the Ewald pair tables (about 330 MB for one
    shell-resolved Cu32 cell); the neighbour blocks are dropped too."""
    from ..constants import BOHR
    from .ptbp_workload import calculator
    out = []
    for s in structs:
        calc = calculator(s, backend="torch", device=device, dtype=dtype, scc=False)
        eng, labels, _ = calc._engine(s.atoms)
        a = s.atoms
        st = eng._setup(labels, a.positions / BOHR, a.cell.array / BOHR, np.asarray(a.pbc),
                        False, {})
        st.pairs = None
        out.append((eng, st))
    return out


def _stack_bytes(structs, dtype: str) -> float:
    """The memory of a group at its peak, a little over: 16 stacks of N x N matrices over the
    k-points (half the grid and one, as time reversal leaves them) in the working precision
    and two in double precision (P132: the Cu32 group of 8 cells peaks at 2.2 GiB)."""
    complex_ = any(not s.gamma for s in structs)
    item = {"float64": 8, "float32": 4}[dtype] * (2 if complex_ else 1)
    nk = sum(1.0 if s.gamma else float(np.prod(s.kpts)) / 2 + 1 for s in structs)
    n = max(s.norb for s in structs)
    return nk * n * n * item * (16 + (4 if dtype == "float32" else 0))


def _memory_budget(device: str) -> float:
    if device.startswith("cuda"):
        import torch
        dev = torch.device(device)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        return 0.5 * torch.cuda.mem_get_info(idx)[0]
    return 4.0 * 2 ** 30


def _eigh(b, hp):
    import torch
    return torch.linalg.eigh(hp if b.linalg_device is None else hp.to(b.linalg_device))


def _stack_group(setups, device: str, repeat: int) -> dict:
    """One structure at a time (:class:`~pydftb.backend.torch_solver.TorchSolver` prepare,
    solve and populations, as the SCC) against all of them in one stack (one prepare, one eig,
    one copy to the host, the Fermi level of each structure, one density, the populations as
    one product with the block matrix of the k weights)."""
    import torch

    from ..backend import devices
    from ..backend.torch_solver import TorchSolver
    from ..occupations import fill
    engs = [e for e, _ in setups]
    sts = [st for _, st in setups]
    b, sm = engs[0].backend, engs[0].smearing
    cpu = devices.kind_of(device) == "cpu"
    with devices.deterministic(b), b.threads(1 if cpu else None):
        solvers = [TorchSolver(b) for _ in sts]
        one, stacked = {}, {}

        def prepare_one():
            for sv, st in zip(solvers, sts):
                sv.prepare(st.smat_precise)
            _sync(device)

        def solve_one(need_w):
            out = []
            for eng, sv, st in zip(engs, solvers, sts):
                rho, _, ev, mu, _ = sv.solve(st.h0, st.kw, st.nelec, sm, need_w=need_w)
                out.append((ev, mu, eng._populations(st, rho)[0]))
            _sync(device)
            one["res"] = out

        h = torch.cat([st.h0 for st in sts])
        s_precise = torch.cat([st.smat_precise for st in sts])
        s_work = torch.cat([st.smat for st in sts])
        cuts = np.cumsum([0] + [len(st.kw) for st in sts])
        w = np.zeros((len(sts), int(cuts[-1])))
        for i, st in enumerate(sts):
            w[i, cuts[i]:cuts[i + 1]] = st.kw
        w_dev = b.asarray(w, precise=True)
        stack = TorchSolver(b)

        def prepare_stack():
            stack.prepare(s_precise)
            _sync(device)

        def solve_stack(need_w):
            e, c = stack.eig(h)
            ev = b.to_host(e)
            occ, mus = [], []
            for i, st in enumerate(sts):
                o, mu, _ = fill(ev[cuts[i]:cuts[i + 1]], st.kw, st.nelec, sm)
                occ.append(np.asarray(o))
                mus.append(mu)
            cf = c * b.asarray(np.concatenate(occ))[:, None, :]
            rho = cf @ c.mH
            if need_w:
                (cf * e[:, None, :]) @ c.mH
            q = b.to_host(w_dev @ b.widen(rho * s_work.mT).sum(dim=2).real)
            _sync(device)
            stacked["res"] = (ev, mus, q)

        prepare_one()
        prepare_stack()
        hp_one = [sv.linv @ st.h0 @ sv.linv_h for sv, st in zip(solvers, sts)]
        hp_all = stack.linv @ h @ stack.linv_h

        def eigh_one():
            for hp in hp_one:
                _eigh(b, hp)
            _sync(device)

        def eigh_stack():
            _eigh(b, hp_all)
            _sync(device)
        out = {"kpoints": int(cuts[-1])}
        for tag, f_one, f_stack in (("prepare", prepare_one, prepare_stack),
                                    ("iteration", lambda: solve_one(False),
                                     lambda: solve_stack(False)),
                                    ("final", lambda: solve_one(True), lambda: solve_stack(True)),
                                    ("eigh", eigh_one, eigh_stack)):
            f_one()                                     # warm-up
            f_stack()
            t_one, t_stack = _best(f_one, repeat), _best(f_stack, repeat)
            out[tag] = {"one_by_one_s": t_one, "stacked_s": t_stack,
                        "gain": t_one / max(t_stack, 1e-12)}
        solve_one(False)
        solve_stack(False)
        evs = [r[0] for r in one["res"]]
        out["fill_s"] = _best(lambda: [fill(ev, st.kw, st.nelec, sm)
                                       for ev, st in zip(evs, sts)], repeat)
        ev_s, mu_s, q_s = stacked["res"]
        out["max_deig"] = float(np.abs(np.concatenate(evs) - ev_s).max())
        out["max_dmu"] = float(max(abs(r[1] - m) for r, m in zip(one["res"], mu_s)))
        out["max_dq"] = float(np.abs(np.stack([r[2] for r in one["res"]]) - q_s).max())
    return out


def time_stacked_solve(device: str, dtypes=("float64",), quick: bool = False, log=None,
                       repeat: int = 3, groups=None, max_bytes=None) -> list[dict]:
    """The whole solve of an SCC iteration for groups of structures with equal N, stacked
    against one structure at a time (:func:`_stack_group`), on the real H0 and S of PTBP's
    Cu stage (Gamma N = 18 real, beta-Sn N = 18 complex, bcc and fcc N = 9 complex) and of
    STACK_COPIES EOS copies of every Cu cell (N = 9-288).  Per group and matrix type: the times
    of prepare (Cholesky factor), one iteration, the final solve with the energy-weighted
    density and eigh alone (one by one, stacked, gain), of ``fill`` (the same in both), and
    the largest |d eps|, |d mu| and |d q| between the two.  A group larger than half the free
    device memory (4 GiB on the CPU) is skipped."""
    from ..backend import devices
    from .ptbp_workload import structures
    log = log or _silent
    cap = devices.capabilities(device)
    if groups is None:
        structs = structures(quick=quick)
        groups = [(name, [s for s in structs if s.phase in phases])
                  for name, phases in STACK_GROUPS] + _cu_scan(quick)
    budget = _memory_budget(device) if max_bytes is None else max_bytes
    rows = []
    for dtype in dtypes:
        for name, members in groups:
            if not members:
                continue
            complex_ = any(not s.gamma for s in members)
            row = {"group": name, "dtype": KINDS[dtype][1 if complex_ else 0],
                   "structures": len(members), "n": members[0].norb, "complex": complex_}
            need = _stack_bytes(members, dtype)
            if complex_ and not cap.complex_:
                row["skipped"] = "no complex arithmetic on the device"
            elif need > budget:
                row["skipped"] = (f"needs up to {need / 2 ** 30:.2f} GiB, "
                                  f"{budget / 2 ** 30:.2f} GiB allowed")
            else:
                try:
                    row.update(_stack_group(_setups(members, device, dtype), device, repeat))
                except Exception as err:        # noqa: BLE001
                    row["error"] = f"{type(err).__name__}: {err}"
                    if _is_oom(err):
                        _empty_cache(device)
            rows.append(row)
            log("  " + _stack_line(row))
    return rows


def _stack_line(row: dict) -> str:
    head = f"stacked {row['group']:22s} [{row['dtype']}] {row['structures']:2d} structures"
    if "iteration" not in row:
        return f"{head}  {row.get('skipped') or row.get('error')}"
    return (f"{head}, {row['kpoints']} k  prepare x{row['prepare']['gain']:.2f}  "
            f"iteration x{row['iteration']['gain']:.2f}  final x{row['final']['gain']:.2f}  "
            f"eigh x{row['eigh']['gain']:.2f}  |dq| {row['max_dq']:.1e}")


# ---- verdicts -------------------------------------------------------------------------------
def verdicts(report: dict) -> list[str]:
    """The lines that answer M7e-2's questions: the device per structure against the CPU
    pool, the work left on the CPU, and the gain of stacking the solve."""
    out = []
    t = report.get("timings") or {}
    stage = (t.get("ptbp") or {}).get("summary") or {}
    pool = stage.get("numpy_procs", {})
    for mode, s in stage.items():
        if not str(mode).startswith("device_") or not s.get("structures"):
            continue
        line = (f"PTBP stage on the device [{mode[len('device_'):]}]: {s['sum_s']:.1f} s for "
                f"{s['structures']} structures ({s['mean_s']:.3f} s each)")
        if "vs_pool" in s:
            line += (f", x{s['vs_pool']:.2f} the time of {stage['workers']} CPU processes "
                     f"({pool['makespan_s']:.1f} s)")
        out.append(line)
        out.append(f"  left on the CPU: {s['cpu_work_s']:.1f} s ({100 * s['cpu_work_share']:.0f}%); "
                   f"pipeline bound max(CPU work / {stage['workers']}, device solve "
                   f"{s['device_solve_s']:.1f} s) = {s['pipeline_bound_s']:.1f} s")
    for row in t.get("stacked") or []:
        if "iteration" in row:
            out.append(f"stacking the solve, {row['group']} [{row['dtype']}]: iteration "
                       f"x{row['iteration']['gain']:.2f}, eigh alone x{row['eigh']['gain']:.2f}")
    return out


# ---- driver ---------------------------------------------------------------------------------
def _row_line(row: dict) -> str:
    return "  " + ", ".join(f"{k} {_fmt(v)}" for k, v in row.items())


def _eigh_line(row: dict) -> str:
    head = f"  eigh {row['dtype']:10s} N={row['n']:3d} batch={row['batch']:4d}  "
    if "speedup" not in row:
        return head + row["error"]
    return (head + f"device {row['device_s']:.2e} s  numpy {row['numpy_s']:.2e} s  "
            f"x{row['speedup']:.2f}")


def _log_rows(log, rows, line) -> None:
    """The rows of a timing, each by ``line`` or, of another shape, as it is."""
    for row in rows if isinstance(rows, list) else []:
        try:
            log(line(row))
        except Exception:                       # noqa: BLE001  never lose the report to a log line
            log(f"  {row}")


def run(device: str = "auto", dtypes=("float64",), quick: bool = False, timing: bool = True,
        cases=None, log=print, json_path=None, timings=None, workers=None, argv=None) -> dict:
    """All checks and timings; returns the report (``report["ok"]``: every check passed;
    ``report["errors"]``: the sections that could not run).  With ``json_path`` the report is
    rewritten after every section; on KeyboardInterrupt it is written with
    ``"complete": false`` and the interrupt goes on."""
    import torch

    from ..backend import devices
    dev = devices.resolve(device)
    devices.prepare(dev)                        # before CUDA starts (CUBLAS_WORKSPACE_CONFIG)
    cap = devices.capabilities(dev)
    requested = tuple(dtypes)
    if not cap.float64:
        dtypes = tuple(d for d in dtypes if d != "float64") or ("float32",)
    sections = (list(TIMING_SECTIONS) if timings is None else list(timings)) if timing else []
    unknown = [s for s in sections if s not in TIMING_SECTIONS]
    if unknown:
        raise ValueError(f"unknown timings {unknown}: choose from {', '.join(TIMING_SECTIONS)}")
    kinds, skipped = timing_kinds(cap, dtypes)
    report = {"pydftb_gpu_check": SCHEMA, "complete": False,
              "date": time.strftime("%Y-%m-%d %H:%M:%S"), "device": dev,
              "dtypes": list(dtypes),
              "dtypes_skipped": [{"dtype": d, "reason": "no float64 on the device"}
                                 for d in requested if d not in dtypes] + skipped,
              "quick": quick, "timing_sections": sections, "environment": {}, "sections": [],
              "checks": [], "timings": {}, "ok": None, "errors": [], "verdicts": []}
    checks = []
    current = [None]

    def save():
        write_report(report, json_path)

    def section(name, fn):
        current[0] = name
        out, rec = _guard(name, fn, dev, log)
        report["sections"].append(rec)
        current[0] = None
        return out, rec

    def add(items):
        for c in items if isinstance(items, list) else [items]:
            checks.append(c)
            log(c.line())
        report["checks"] = [dataclasses.asdict(c) for c in checks]

    def check(name, fn):
        out, rec = section(name, fn)
        add(out if rec["ok"] else Check(name, False, {"error": rec["error"]}))
        save()

    def finish():
        report["ok"] = all(c.ok is not False for c in checks)
        report["errors"] = [r["section"] for r in report["sections"] if not r["ok"]]
        report["verdicts"] = verdicts(report)

    log(f"[gpu_check] device {dev} ({cap.name}); torch {torch.__version__}; "
        f"dtypes {', '.join(dtypes)}")
    for note in cap.notes:
        log(f"[gpu_check] note: {note}")
    try:
        env, _ = section("environment", lambda: environment(dev, argv))
        report["environment"] = env or {}
        save()
        names = cases or (QUICK_CASES if quick else None)
        for dtype in dtypes:
            check(f"references [{dtype}]", functools.partial(check_references, dev, dtype, names))
            check(f"determinism [{dtype}]", functools.partial(check_determinism, dev, dtype))
        if "float64" in dtypes:
            check("autodiff", lambda: check_autodiff(dev, AUTODIFF_CASES[:2] if quick
                                                     else AUTODIFF_CASES))
        check("fork", lambda: check_fork(dev))
        for name in sections:
            log(f"[gpu_check] timing {name} ...")
            if name == "eigh":
                sizes = (9, 18, 72) if quick else (9, 18, 36, 72, 144, 288)
                batches = (1, 128, 2048) if quick else (1, 16, 128, 1024, 2048)
                out, _ = section("eigh", functools.partial(time_eigh, dev, sizes, batches, kinds))
                _log_rows(log, out, _eigh_line)
            elif name == "cu":
                out, _ = section("cu", lambda: time_cu(dev, dtypes, quick))
                _log_rows(log, out, _row_line)
            elif name == "ptbp":
                out, _ = section("ptbp", lambda: time_ptbp_stage(dev, dtypes, quick, workers,
                                                                 log))
            else:
                out, _ = section("stacked", lambda: time_stacked_solve(dev, dtypes, quick, log))
            if out is not None:
                report["timings"][name] = out
            save()
    except KeyboardInterrupt:
        report["interrupted"] = True
        report["interrupted_in"] = current[0]
        finish()
        save()
        log("[gpu_check] interrupted")
        raise
    finish()
    report["complete"] = True
    save()
    for line in report["verdicts"]:
        log(f"[gpu_check] {line}")
    failed = [c.name for c in checks if c.ok is False]
    log(f"[gpu_check] {len(checks)} checks, {len(failed)} failed"
        + (f": {', '.join(failed)}" if failed else "")
        + (f"; sections that could not run: {', '.join(report['errors'])}"
           if report["errors"] else ""))
    return report


def main(argv=None, prog=None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help="auto (default), cuda, cuda:1, mps, xpu, cpu")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32", "both"])
    ap.add_argument("--quick", action="store_true", help="a subset of cases and timings")
    ap.add_argument("--skip-timing", action="store_true", help="no timings")
    ap.add_argument("--timings", default=",".join(TIMING_SECTIONS),
                    help="the timings to run, comma-separated (default: all of "
                         f"{','.join(TIMING_SECTIONS)})")
    ap.add_argument("--workers", type=int, default=None,
                    help="processes of the CPU pool in the ptbp timing (default: the usable "
                         "cores)")
    ap.add_argument("--cases", nargs="*", default=None, help="reference cases (default: all)")
    ap.add_argument("--json", default=None,
                    help="write the report here (rewritten after every section)")
    a = ap.parse_args(argv)
    sections = [s.strip() for s in a.timings.split(",") if s.strip()]
    bad = [s for s in sections if s not in TIMING_SECTIONS]
    if bad:
        ap.error(f"unknown timings {', '.join(bad)} (choose from {', '.join(TIMING_SECTIONS)})")
    if a.workers is not None and a.workers < 1:
        ap.error("--workers must be at least 1")
    dtypes = ("float64", "float32") if a.dtype == "both" else (a.dtype,)
    command = [prog or "gpu_check", *(sys.argv[1:] if argv is None else argv)]
    try:
        report = run(a.device, dtypes, a.quick, not a.skip_timing, a.cases, json_path=a.json,
                     timings=sections, workers=a.workers, argv=command)
    except KeyboardInterrupt:
        print("[gpu_check] interrupted" + (f"; what was measured is in {a.json}" if a.json
                                           else ""), file=sys.stderr)
        return 2
    if a.json:
        print(f"[gpu_check] report written to {a.json}")
    return 0 if report["ok"] and not report["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
