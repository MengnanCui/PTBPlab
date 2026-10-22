"""Torch devices of the pydftb backend (milestone M4d).

``device`` is ``"cpu"``, ``"cuda"``/``"cuda:i"`` (NVIDIA, or AMD with a ROCm build of torch),
``"mps"`` (Apple), ``"xpu"`` (Intel) or ``"auto"`` (the first available of these, else the
CPU).  What a device can do decides where each step runs (see :class:`Capabilities`):

* CUDA/ROCm: everything on the device;
* no float64 on the device (MPS; some XPUs): only ``dtype="float32"``, and the float64 work
  (the Cholesky factor of S, the sums of populations and energies) stays on the CPU;
* no Cholesky/eigh on the device: these run on the CPU, the rest on the device.

MPS and XPU are probed once with 2 x 2 matrices instead of relying on lists of supported
operations.  On CUDA a calculation runs with torch's deterministic algorithms
(:func:`deterministic`).  :func:`probe_accelerators` looks for GPUs in a subprocess, so the
calling process does not initialise CUDA and can still fork safely.
"""
from __future__ import annotations

import contextlib
import functools
import json
import os
import platform
import subprocess
import sys
import warnings
from dataclasses import dataclass

KINDS = ("cpu", "cuda", "mps", "xpu")
CUBLAS_WORKSPACE = ":4096:8"         # deterministic cuBLAS (NVIDIA's documentation)


@dataclass(frozen=True)
class Capabilities:
    """What a torch device can do for pydftb."""
    device: str
    kind: str                        # cpu | cuda | mps | xpu
    name: str = ""
    rocm: bool = False               # an AMD GPU through a ROCm build ("cuda" in torch)
    float64: bool = True             # float64/complex128 arrays on the device
    linalg: bool = True              # Cholesky and eigh on the device
    complex_: bool = True            # complex arithmetic (k-points) on the device
    notes: tuple = ()

    def to_json(self) -> dict:
        return {"device": self.device, "kind": self.kind, "name": self.name, "rocm": self.rocm,
                "float64": self.float64, "linalg": self.linalg, "complex": self.complex_,
                "notes": list(self.notes)}


def kind_of(device) -> str:
    """``"cuda"`` for ``"cuda:1"``; an error for unknown device types."""
    kind = str(device).split(":", 1)[0]
    if kind not in KINDS:
        raise ValueError(f"unknown device {device!r}: one of cpu, cuda[:i], mps, xpu or auto")
    return kind


def resolve(device=None) -> str:
    """The device a ``device`` option means: ``"auto"`` becomes the first available of
    cuda, mps and xpu, else ``"cpu"``; None is ``"cpu"``."""
    import torch
    name = "cpu" if device is None else str(device)
    if name != "auto":
        kind_of(name)
        return str(torch.device(name))
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    return "cpu"


def _works(fn) -> bool:
    """Whether ``fn`` runs (an unsupported dtype or operation raises)."""
    try:
        fn()
        return True
    except Exception:                       # noqa: BLE001  TypeError, RuntimeError, NotImplementedError
        return False


def _probe(device: str):
    """(float64, complex, linalg) of a device without a known capability list."""
    import torch
    f64 = _works(lambda: torch.ones(2, dtype=torch.float64, device=device).sum().item())
    real, cplx = (torch.float64, torch.complex128) if f64 else (torch.float32, torch.complex64)
    cpx = _works(lambda: (torch.eye(2, dtype=cplx, device=device)
                          @ torch.eye(2, dtype=cplx, device=device)).sum().item())

    def linalg():
        for dt in (real, cplx) if cpx else (real,):
            a = torch.eye(2, dtype=dt, device=device) * 2
            torch.linalg.cholesky_ex(a)[0].sum().item()
            torch.linalg.solve_triangular(a, a, upper=False).sum().item()
            torch.linalg.eigh(a)[0].sum().item()
    return f64, cpx, _works(linalg)


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "cpu"


@functools.lru_cache(maxsize=None)
def capabilities(device: str) -> Capabilities:
    """The :class:`Capabilities` of ``device`` (resolved, e.g. ``"cuda:1"``); cached."""
    import torch
    kind = kind_of(device)
    if kind == "cpu":
        return Capabilities(device, "cpu", _cpu_name())
    if kind == "cuda":
        dev = torch.device(device)
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        rocm = getattr(torch.version, "hip", None) is not None
        notes = (("ROCm build: how fast batched eigh is depends on torch's solver library "
                  "(hipSOLVER or MAGMA); see the gpu_check timings",) if rocm else ())
        return Capabilities(device, "cuda", torch.cuda.get_device_name(index), rocm,
                            notes=notes)
    f64, cpx, linalg = _probe(device)
    notes = []
    if not f64:
        notes.append("no float64: dtype='float32' only; the float64 work runs on the CPU")
    if not linalg:
        notes.append("no Cholesky/eigh on the device: they run on the CPU")
    if not cpx:
        notes.append("no complex arithmetic: k-point calculations cannot run on this device")
    return Capabilities(device, kind, kind.upper(), False, f64, linalg, cpx, tuple(notes))


def prepare(device: str) -> None:
    """Before the first use of a CUDA device: the cuBLAS workspace setting that makes its
    matrix products deterministic (it must be set before cuBLAS starts)."""
    if kind_of(device) != "cuda" or "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        return
    import torch
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
    if torch.cuda.is_initialized():
        warnings.warn("CUDA was initialised before pydftb set CUBLAS_WORKSPACE_CONFIG="
                      f"{CUBLAS_WORKSPACE}; set it before the first CUDA call for "
                      "deterministic cuBLAS results", RuntimeWarning, stacklevel=3)


@contextlib.contextmanager
def deterministic(backend):
    """On CUDA: torch's deterministic algorithms during a calculation (operations without
    one only warn), the user's setting restored afterwards.  Elsewhere: nothing."""
    if not backend.is_torch or kind_of(backend.device) != "cuda":
        yield
        return
    import torch
    if torch.are_deterministic_algorithms_enabled():
        yield                                  # already on, as the user wants it
        return
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False)


_PROBE = r"""
import json
out = {"torch": None}
try:
    import torch
except ImportError:
    print(json.dumps(out)); raise SystemExit
out["torch"] = torch.__version__
out["cuda_version"] = torch.version.cuda
out["hip_version"] = getattr(torch.version, "hip", None)
out["cuda"] = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
               if torch.cuda.is_available() else [])
mps = getattr(torch.backends, "mps", None)
out["mps"] = bool(mps is not None and mps.is_available())
xpu = getattr(torch, "xpu", None)
out["xpu"] = ([torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())]
              if xpu is not None and xpu.is_available() else [])
print(json.dumps(out))
"""


def probe_accelerators(timeout: float = 120.0) -> dict:
    """The torch version and the GPUs torch sees, found in a subprocess: the caller does
    not initialise CUDA, so it may still fork workers.  Keys: torch (None: not installed),
    cuda (device names), hip_version, cuda_version, mps, xpu, and error if the probe
    failed.  ``auto`` in that subprocess would pick :func:`best_device`."""
    try:
        run = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True,
                             timeout=timeout)
        return json.loads(run.stdout.strip().splitlines()[-1])
    except Exception as err:                    # noqa: BLE001  no torch, a timeout, a crash
        return {"torch": None, "cuda": [], "mps": False, "xpu": [],
                "error": f"{type(err).__name__}: {err}"}


def best_device(probe: dict) -> str:
    """What ``device="auto"`` resolves to for a :func:`probe_accelerators` result."""
    if probe.get("cuda"):
        return "cuda"
    if probe.get("mps"):
        return "mps"
    if probe.get("xpu"):
        return "xpu"
    return "cpu"
