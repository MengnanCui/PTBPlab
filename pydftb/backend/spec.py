"""Array backends of the calculator (milestone M4): NumPy (default) or PyTorch.

With ``Backend("numpy")`` every conversion below returns its argument unchanged, so the
NumPy path makes exactly the NumPy calls it made before M4.  ``Backend("torch", device,
dtype)`` keeps the heavy linear algebra of the SCC loop on a torch device (folding to
k-points, Cholesky, eigensolver, density matrices, Mulliken populations, the traces of
the energy and the real-space densities of the forces); geometry, parameters and the
control flow (potentials, Fermi level, mixer) stay NumPy float64 on the host.

``dtype="float32"`` runs H, the eigensolver and the densities in single precision (complex64
at k-points); the Cholesky factor of S is still computed in double precision, and
everything on the host stays float64.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np

BACKENDS = ("numpy", "torch")
DTYPES = ("float64", "float32")


@dataclass(frozen=True)
class Backend:
    name: str = "numpy"
    device: str = "cpu"
    dtype: str = "float64"

    def __post_init__(self):
        if self.name not in BACKENDS:
            raise ValueError(f"unknown backend {self.name!r} (one of {BACKENDS})")
        if self.dtype not in DTYPES:
            raise ValueError(f"unknown dtype {self.dtype!r} (one of {DTYPES})")
        if self.name == "numpy" and (self.device != "cpu" or self.dtype != "float64"):
            raise ValueError("the NumPy backend runs on the CPU in float64; "
                             "use backend='torch' for other devices or float32")

    @property
    def is_torch(self) -> bool:
        return self.name == "torch"

    @property
    def xp(self):
        """The array namespace: the numpy module, or NumPy names over torch."""
        if not self.is_torch:
            return np
        from . import torch_xp
        return torch_xp

    # ---- torch dtypes ------------------------------------------------------------------
    def _dtypes(self, precise: bool):
        import torch
        if precise or self.dtype == "float64":
            return torch.float64, torch.complex128
        return torch.float32, torch.complex64

    # ---- conversions (identities for NumPy) ----------------------------------------------
    def asarray(self, a, precise: bool = False, complex_: bool = False):
        """Host data on the device in the working precision (``precise``: double;
        ``complex_``: as a complex array, for products with complex ones)."""
        if not self.is_torch:
            return a
        import torch
        real, cplx = self._dtypes(precise)
        if isinstance(a, torch.Tensor):
            t = a
        else:
            t = torch.from_numpy(np.ascontiguousarray(a))
        dt = cplx if (complex_ or t.is_complex()) else real
        return t.to(device=self.device, dtype=dt)

    def cast(self, t):
        """A device array in the working precision (identity for float64)."""
        if not self.is_torch or self.dtype == "float64":
            return t
        real, cplx = self._dtypes(False)
        return t.to(cplx if t.is_complex() else real)

    def widen(self, t):
        """A device array in double precision (identity for NumPy and for float64)."""
        if not self.is_torch or self.dtype == "float64":
            return t
        import torch
        return t.to(torch.complex128 if t.is_complex() else torch.float64)

    def index(self, idx):
        """Integer index array for device arrays."""
        if not self.is_torch:
            return idx
        import torch
        return torch.as_tensor(np.asarray(idx), dtype=torch.long, device=self.device)

    def to_host(self, x):
        """Device array -> NumPy float64/complex128 (NumPy arrays pass through)."""
        if not self.is_torch:
            return x
        from .ops import to_host
        return to_host(x)

    def threads(self, n: int | None):
        """Context: torch's intra-op threads during a calculation (restored after)."""
        if not self.is_torch or n is None:
            return contextlib.nullcontext()
        return _torch_threads(int(n))


@contextlib.contextmanager
def _torch_threads(n: int):
    """Always calls ``torch.set_num_threads``: it also sets the thread count of torch's
    statically linked MKL, which threadpoolctl cannot see.  Changing only OpenMP's count
    (threadpoolctl) leaves MKL asking for its old number of threads, and in a process
    forked after the parent ran parallel regions that request hangs (M4a report)."""
    import torch
    old = torch.get_num_threads()
    torch.set_num_threads(max(1, n))
    try:
        yield
    finally:
        torch.set_num_threads(old)


NUMPY = Backend()


def make_backend(backend=None, device: str | None = None, dtype: str | None = None) -> Backend:
    """``Backend`` from a name (``"numpy"``/``"torch"``) or an existing ``Backend``."""
    if isinstance(backend, Backend):
        if device is not None or dtype is not None:
            raise ValueError("pass device/dtype inside the Backend, not separately")
        return backend
    name = "numpy" if backend is None else str(backend).lower()
    if name == "numpy":
        if device not in (None, "cpu") or dtype not in (None, "float64"):
            raise ValueError("the NumPy backend runs on the CPU in float64; "
                             "use backend='torch' for other devices or float32")
        return NUMPY
    if name != "torch":
        raise ValueError(f"unknown backend {backend!r} (one of {BACKENDS})")
    try:
        import torch
    except ImportError as err:               # pragma: no cover - depends on the environment
        raise ImportError("backend='torch' needs PyTorch: pip install 'torch>=2.14' "
                          "(CPU wheels: --index-url https://download.pytorch.org/whl/cpu)") from err
    dev = str(torch.device(device or "cpu"))
    return Backend("torch", dev, dtype or "float64")
