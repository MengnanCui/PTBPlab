"""Devices of the torch backend (milestone M4d).  There is no GPU here: the device
logic runs on simulated capabilities, and the branches that move work to the CPU (MPS,
XPU) run on CPU tensors, where they must give the same bits as the normal path."""
import os
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pydftb.backend import devices  # noqa: E402
from pydftb.backend.devices import Capabilities  # noqa: E402
from pydftb.backend.spec import Backend, make_backend  # noqa: E402
from pydftb.validation import refgen  # noqa: E402

FIELDS = ("energy", "free_energy", "forces", "charges", "eigenvalues", "stress")


def _run(name, backend):
    doc = refgen.load(name)
    why = refgen.missing_requirement(doc)
    if why:
        pytest.skip(why)
    tol = {"scc_tol": 1e-5} if backend.dtype == "float32" else {}  # float32 (M4a)
    return refgen.run_pydftb(refgen.CASES_BY_NAME[doc["case"]], doc["input"], backend=backend,
                             **tol)


def _same(a, b):
    for f in FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        if x is None:
            assert y is None, f
            continue
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes(), f


# float32 cases at a finite temperature: at T = 0 a metal's Fermi level search may fail in
# single precision (M4a report)
@pytest.mark.parametrize("name, dtype", [("h2o", "float64"), ("cu_fcc_k6_T0", "float64"),
                                         ("cuo_chain_spin_k422", "float64"), ("h2o", "float32"),
                                         ("cu8_distorted_k355_mp2_2000K", "float32")])
def test_linalg_elsewhere_gives_the_same_bits(name, dtype):
    """Cholesky/eigh on ``linalg_device`` (the CPU for MPS/XPU): here a no-op move."""
    plain = _run(name, Backend("torch", "cpu", dtype))
    moved = _run(name, Backend("torch", "cpu", dtype, linalg_device="cpu"))
    _same(plain, moved)


@pytest.mark.parametrize("name", ["h2o", "cu8_distorted_k355_mp2_2000K", "cuo_chain_spin_k422"])
def test_float64_elsewhere_gives_the_same_bits(name):
    """The MPS layout (float32 device work, float64 arrays and linear algebra on the CPU)."""
    plain = _run(name, Backend("torch", "cpu", "float32"))
    mps_like = _run(name, Backend("torch", "cpu", "float32", linalg_device="cpu",
                                  precise_device="cpu"))
    _same(plain, mps_like)


def test_float64_without_float64_is_refused():
    with pytest.raises(ValueError, match="no float64"):
        Backend("torch", "mps", "float64", precise_device="cpu")


def _fake(monkeypatch, cap: Capabilities):
    monkeypatch.setattr(devices, "resolve", lambda device=None: cap.device)
    monkeypatch.setattr(devices, "capabilities", lambda device: cap)


def test_make_backend_follows_the_capabilities(monkeypatch):
    _fake(monkeypatch, Capabilities("mps", "mps", "MPS", float64=False, linalg=False))
    with pytest.raises(ValueError, match="no float64"):
        make_backend("torch", "mps")
    with pytest.warns(RuntimeWarning, match="run on the CPU"):
        b = make_backend("torch", "mps", "float32")
    assert (b.device, b.precise, b.linalg) == ("mps", "cpu", "cpu")
    _fake(monkeypatch, Capabilities("xpu", "xpu", "XPU", float64=True, linalg=True,
                                    complex_=False))
    with pytest.raises(ValueError, match="complex"):
        make_backend("torch", "xpu")
    _fake(monkeypatch, Capabilities("cuda:1", "cuda", "GPU"))
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = make_backend("torch", "cuda:1")
    assert (b.device, b.precise, b.linalg) == ("cuda:1", "cuda:1", "cuda:1")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == devices.CUBLAS_WORKSPACE


def test_resolve(monkeypatch):
    assert devices.resolve(None) == devices.resolve("cpu") == "cpu"
    assert devices.resolve("cuda:1") == "cuda:1"
    assert devices.kind_of("cuda:1") == "cuda"
    with pytest.raises(ValueError, match="unknown device"):
        devices.resolve("tpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert devices.resolve("auto") in ("cpu", "mps", "xpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert devices.resolve("auto") == "cuda"


def test_probed_capabilities(monkeypatch):
    monkeypatch.setattr(devices, "_probe", lambda device: (False, True, False))
    cap = devices.capabilities.__wrapped__("mps")
    assert (cap.kind, cap.float64, cap.linalg, cap.complex_) == ("mps", False, False, True)
    assert any("float32" in n for n in cap.notes) and any("CPU" in n for n in cap.notes)
    cpu = devices.capabilities("cpu")
    assert cpu.kind == "cpu" and cpu.float64 and cpu.linalg and cpu.complex_
    assert set(cpu.to_json()) >= {"device", "float64", "linalg", "complex", "notes"}


def test_rocm_is_cuda_with_a_note(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.2", raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "AMD Instinct MI250X")
    cap = devices.capabilities.__wrapped__("cuda")
    assert (cap.kind, cap.rocm, cap.float64, cap.linalg) == ("cuda", True, True, True)
    assert cap.name == "AMD Instinct MI250X" and any("ROCm" in n for n in cap.notes)


def test_probe_on_the_cpu_finds_everything():
    assert devices._probe("cpu") == (True, True, True)


def test_deterministic_context_restores_the_setting():
    assert not torch.are_deterministic_algorithms_enabled()
    with devices.deterministic(Backend("torch", "cuda:0")):
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert not torch.are_deterministic_algorithms_enabled()
    with devices.deterministic(Backend("torch", "cpu")):      # only CUDA
        assert not torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)                   # the user's own setting stays
    try:
        with devices.deterministic(Backend("torch", "cuda:0")):
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(False)


def test_prepare_sets_the_cublas_workspace_once(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    devices.prepare("cpu")
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    devices.prepare("cuda")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == devices.CUBLAS_WORKSPACE
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    devices.prepare("cuda:1")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_probe_accelerators_in_a_subprocess():
    probe = devices.probe_accelerators()
    assert probe["torch"] == torch.__version__ and "error" not in probe
    assert isinstance(probe["cuda"], list) and isinstance(probe["mps"], bool)
    assert devices.best_device(probe) == ("cuda" if probe["cuda"] else
                                          "mps" if probe["mps"] else
                                          "xpu" if probe["xpu"] else "cpu")
    assert devices.best_device({"cuda": ["A100"], "mps": True}) == "cuda"
    assert devices.best_device({"cuda": [], "mps": False, "xpu": ["Arc"]}) == "xpu"
    assert devices.best_device({"torch": None, "error": "x"}) == "cpu"


# ---- nothing is created off the device (M4d) -------------------------------------------------
# Under a "meta" default device, a tensor made without an explicit device lands on "meta" and
# meeting a real (CPU) tensor raises: the CPU stands in for a GPU that is not here.
@pytest.mark.parametrize("name", ["h2o", "cu_fcc_k6_T0", "cuo_chain_spin_k422", "ch3oh_dftb3"])
def test_forward_backend_keeps_to_its_device(name):
    plain = _run(name, Backend("torch", "cpu"))
    with torch.device("meta"):
        res = _run(name, Backend("torch", "cpu"))
    _same(plain, res)


@pytest.mark.parametrize("name", ["h2o", "cu_fcc_k6_T0", "cuo_chain_spin_k422", "ch3oh_dftb3",
                                  "o2_triplet_wmatrix", "water_box_d3bj_3body"])
@pytest.mark.parametrize("mode", ["hf", "implicit"])
def test_autodiff_keeps_to_its_device(name, mode):
    from pydftb.autodiff import Params, evaluate
    from pydftb.validation import gradcheck
    why = refgen.missing_requirement(refgen.load(name))
    if why:
        pytest.skip(why)
    calc, (species, pos, cell, pbc) = gradcheck.reference_engine(name, variational=False)

    def grads():
        params = Params.from_skset(calc.sk, calc.repulsive, calc.hubbard_derivs,
                                   calc.spin_constants, requires_grad=("tables", "onsite"))
        res = evaluate(calc, species, pos, cell, pbc, params, mode=mode)
        target = res.free_energy + (res.forces.sum() if mode == "implicit" else 0.0)
        g = torch.autograd.grad(target, params.leaves(), allow_unused=True)
        return [None if x is None else x.numpy().tobytes() for x in g]
    plain = grads()
    with torch.device("meta"):
        assert grads() == plain

