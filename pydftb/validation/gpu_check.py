"""GPU self-check of pydftb (milestone M4d): correctness and timings on your device.

    python -m pydftb.validation.gpu_check                          # the first GPU, everything
    python -m pydftb.validation.gpu_check --device cuda:1 --json gpu_check.json
    python -m pydftb.validation.gpu_check --quick                  # a subset, a few minutes
    python -m pydftb.validation.gpu_check --device cpu --quick     # the script itself, no GPU

Correctness (exit status 1 if any check fails):

* references: frozen DFTB+ cases on the device, with the tolerances of ``refgen --check``;
* cpu: the same cases with torch on the CPU, float64: |dE|, |dF|, |dq|, |deps| <= 1e-10;
* determinism: one calculation twice, bit for bit;
* autodiff: dF/dtheta on the device in modes "hf" and "implicit" and on the CPU, <= 1e-9;
* fork (CUDA): a process that used CUDA forks a CPU-only NumPy calculation, which must end.

Timings (reported only): ``eigh`` over N = 9-288 orbitals and batches of 1-2048 matrices,
the device against NumPy; Cu cells computed as PTBP computes them, NumPy on one thread and
with k-point threads against the device; and k-point matrices of several structures in one
batch against one structure at a time (whether batching across structures would pay).
``--dtype float32`` (or ``both``) adds float32 runs, reported without a pass/fail.

Send the JSON (``--json``) back: it holds the environment, every number and the verdicts.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
import textwrap
import time

import numpy as np

CPU_TOL = 1e-10              # device vs CPU, float64 (Ha, Ha/Bohr, e)
F32_SCC_TOL = 1e-5           # float32 (M4a); at 1e-6 the mixer of some DFTB3 cases diverges
GRAD_TOL = 1e-9              # relative, gradients
QUICK_CASES = ["h2o", "ch3oh_dftb3", "cu_fcc_k6_T0", "cu8_distorted_k355_mp2_2000K",
               "cuo_chain_spin_k422", "gdo_rocksalt_k444", "water_box_d3bj_3body"]
AUTODIFF_CASES = ["h2o", "cu4co_shell_resolved", "cuo_chain_spin_k422"]
DETERMINISM_CASES = ["ch3oh_dftb3", "cu8_distorted_k355_mp2_2000K"]
SKIP = None                  # a check that does not apply (e.g. fork on the CPU)


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


def _sync(device: str) -> None:
    import torch
    kind = device.split(":")[0]
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()
    elif kind == "xpu":
        torch.xpu.synchronize()


# ---- environment ----------------------------------------------------------------------------
def environment(device: str) -> dict:
    import torch

    from ..backend import devices
    env = {"python": platform.python_version(), "numpy": np.__version__,
           "torch": torch.__version__, "cuda_version": torch.version.cuda,
           "hip_version": getattr(torch.version, "hip", None), "platform": platform.platform(),
           "cpu": devices._cpu_name(), "cpu_count": os.cpu_count(), "device": device,
           "capabilities": devices.capabilities(device).to_json()}
    if device.startswith("cuda"):
        dev = torch.device(device)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        env.update(gpu=prop.name, gpu_memory_gb=round(prop.total_memory / 2 ** 30, 2),
                   compute_capability=f"{prop.major}.{prop.minor}",
                   multiprocessors=prop.multi_processor_count,
                   cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=os.path.dirname(__file__), timeout=10)
        env["pydftb_commit"] = rev.stdout.strip() or None
    except Exception:                           # noqa: BLE001  not a git checkout
        env["pydftb_commit"] = None
    return env


# ---- correctness ----------------------------------------------------------------------------
def _cases(names):
    from . import refgen
    from .cases import CASES
    out = []
    for case in CASES:
        if names and case.name not in names:
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


def check_determinism(device: str, dtype: str, names=DETERMINISM_CASES) -> list[Check]:
    from . import refgen
    out = []
    for doc in _cases(names):
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


# ---- timings --------------------------------------------------------------------------------
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
    """``eigh`` of random Hermitian stacks on the device and with NumPy (all BLAS threads)."""
    import torch
    rng = np.random.default_rng(0)
    tdt = {"float64": torch.float64, "complex128": torch.complex128,
           "float32": torch.float32, "complex64": torch.complex64}
    rows = []
    for name in dtypes:
        for n in sizes:
            for batch in batches:
                item = 16 if name == "complex128" else 8 if name in ("float64", "complex64") else 4
                if batch * n * n * item * 4 > max_bytes:
                    continue
                h = _hermitian(rng, batch, n, name.startswith("complex"))
                h = h.astype({"float64": np.float64, "complex128": np.complex128,
                              "float32": np.float32, "complex64": np.complex64}[name])
                t = torch.as_tensor(h).to(device=device, dtype=tdt[name])

                def dev():
                    torch.linalg.eigh(t)
                    _sync(device)
                dev()                                   # warm-up (kernels, handles)
                rows.append({"dtype": name, "n": n, "batch": batch,
                             "device_s": _best(dev, repeat),
                             "numpy_s": _best(lambda: np.linalg.eigh(h), repeat)})
                rows[-1]["speedup"] = rows[-1]["numpy_s"] / max(rows[-1]["device_s"], 1e-12)
    return rows


CU_CELLS = [("Cu1 k14", "prim", (1, 1, 1), (14, 14, 14)),
            ("Cu2 k12x12x6", "prim", (1, 1, 2), (12, 12, 6)),
            ("Cu4 k8", "cubic", (1, 1, 1), (8, 8, 8)),
            ("Cu8 k6", "prim", (2, 2, 2), (6, 6, 6)),
            ("Cu32 k3", "cubic", (2, 2, 2), (3, 3, 3))]
TIMING_MAX_SCC = 100         # enough to time; larger cells at T = 0 may not converge in 1000


def time_cu(device: str, dtype: str, quick: bool, repeat: int = 1) -> list[dict]:
    """Cu cells with PTBP's settings (PyDFTB's batched solver, T = 0, shell-resolved SCC to
    1e-5, the 0.01 Bohr finite-difference forces, polynomial repulsive) but at most
    TIMING_MAX_SCC iterations: NumPy on one thread, NumPy with k-point threads over all
    cores, and the device.  The SCC iteration counts may differ between them (metals at
    T = 0), hence also the time per iteration."""
    from ase.build import bulk
    from threadpoolctl import threadpool_limits

    from .. import DFTB, ParamSet
    from ..constants import BOHR
    from ..kpoints import monkhorst_pack, reduce_time_reversal
    from .cases import SKF_SETS
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
        for tag, opts in (("numpy_1", {}), ("numpy_kthreads", {"k_threads": os.cpu_count()}),
                          ("device", {"backend": "torch", "device": device, "dtype": dtype})):
            calc = DFTB(sk, **common, **opts)

            def run():
                if tag == "numpy_1":
                    with threadpool_limits(1):
                        res[tag] = calc.calculate(*args)
                else:
                    res[tag] = calc.calculate(*args)
                if tag == "device":
                    _sync(device)
            try:
                row[f"{tag}_s"] = _best(run, repeat)
                row[f"{tag}_n_scc"] = res[tag].n_scc
                row[f"{tag}_converged"] = bool(res[tag].converged)
                row[f"{tag}_per_scc_s"] = row[f"{tag}_s"] / max(res[tag].n_scc, 1)
            except Exception as err:             # noqa: BLE001
                row[f"{tag}_error"] = f"{type(err).__name__}: {err}"
        if "numpy_1" in res and "device" in res:
            row["energy_diff"] = abs(res["numpy_1"].energy - res["device"].energy)
        rows.append(row)
    return rows


def time_stacked(device: str, sizes=(9, 18), structures: int = 32, nk: int = 200,
                 repeat: int = 3) -> list[dict]:
    """The k-point matrices of ``structures`` structures: one ``eigh`` on the whole stack
    against one per structure (complex128)."""
    import torch
    rng = np.random.default_rng(1)
    rows = []
    for n in sizes:
        t = torch.as_tensor(_hermitian(rng, structures * nk, n, True)).to(device)
        parts = list(t.split(nk))

        def stacked():
            torch.linalg.eigh(t)
            _sync(device)

        def one_by_one():
            for p in parts:
                torch.linalg.eigh(p)
            _sync(device)
        stacked()
        one_by_one()
        rows.append({"n": n, "structures": structures, "kpoints": nk,
                     "stacked_s": _best(stacked, repeat), "separate_s": _best(one_by_one, repeat)})
        rows[-1]["gain"] = rows[-1]["separate_s"] / max(rows[-1]["stacked_s"], 1e-12)
    return rows


# ---- driver ---------------------------------------------------------------------------------
def run(device: str = "auto", dtypes=("float64",), quick: bool = False, timing: bool = True,
        cases=None, log=print) -> dict:
    """All checks and timings; returns the report (``report["ok"]``: every check passed)."""
    import torch

    from ..backend import devices
    dev = devices.resolve(device)
    cap = devices.capabilities(dev)
    if not cap.float64:
        dtypes = tuple(d for d in dtypes if d != "float64") or ("float32",)
    report = {"pydftb_gpu_check": 1, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
              "environment": environment(dev), "dtypes": list(dtypes), "quick": quick,
              "checks": [], "timings": {}}
    log(f"[gpu_check] device {dev} ({cap.name}); torch {torch.__version__}; "
        f"dtypes {', '.join(dtypes)}")
    for note in cap.notes:
        log(f"[gpu_check] note: {note}")
    names = cases or (QUICK_CASES if quick else None)
    checks = []

    def add(items):
        for c in items if isinstance(items, list) else [items]:
            checks.append(c)
            log(c.line())
    for dtype in dtypes:
        add(check_references(dev, dtype, names))
        add(check_determinism(dev, dtype))
    if "float64" in dtypes:
        add(check_autodiff(dev, AUTODIFF_CASES[:2] if quick else AUTODIFF_CASES))
    add(check_fork(dev))
    report["checks"] = [dataclasses.asdict(c) for c in checks]
    report["ok"] = all(c.ok is not False for c in checks)
    if timing:
        sizes = (9, 18, 72) if quick else (9, 18, 36, 72, 144, 288)
        batches = (1, 128, 2048) if quick else (1, 16, 128, 1024, 2048)
        kinds = (["float64", "complex128"] if cap.float64 else []) + ["complex64"]
        log("[gpu_check] timing eigh ...")
        report["timings"]["eigh"] = time_eigh(dev, sizes, batches, kinds)
        for row in report["timings"]["eigh"]:
            log(f"  eigh {row['dtype']:10s} N={row['n']:3d} batch={row['batch']:4d}  "
                f"device {row['device_s']:.2e} s  numpy {row['numpy_s']:.2e} s  "
                f"x{row['speedup']:.2f}")
        log("[gpu_check] timing Cu cells ...")
        report["timings"]["cu"] = time_cu(dev, dtypes[0], quick)
        for row in report["timings"]["cu"]:
            log("  " + ", ".join(f"{k} {_fmt(v)}" for k, v in row.items()))
        report["timings"]["stacked"] = time_stacked(dev, sizes=(9, 18) if quick else (9, 18, 36))
        for row in report["timings"]["stacked"]:
            log("  " + ", ".join(f"{k} {_fmt(v)}" for k, v in row.items()))
    failed = [c.name for c in checks if c.ok is False]
    log(f"[gpu_check] {len(checks)} checks, {len(failed)} failed"
        + (f": {', '.join(failed)}" if failed else ""))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help="auto (default), cuda, cuda:1, mps, xpu, cpu")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32", "both"])
    ap.add_argument("--quick", action="store_true", help="a subset of cases and timings")
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--cases", nargs="*", default=None, help="reference cases (default: all)")
    ap.add_argument("--json", default=None, help="write the report here")
    a = ap.parse_args(argv)
    dtypes = ("float64", "float32") if a.dtype == "both" else (a.dtype,)
    report = run(a.device, dtypes, a.quick, not a.skip_timing, a.cases)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(report, fh, indent=1, default=str)
        print(f"[gpu_check] report written to {a.json}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
