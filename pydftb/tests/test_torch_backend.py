"""The torch backend (milestone M4a) against the NumPy backend.

Both converge the SCC to 1e-12; measured over all 74 reference cases the largest
differences are 4e-14 Ha (energy), 4e-14 Ha/Bohr (forces), 1.4e-13 e (charges) and
2.4e-14 Ha (eigenvalues), with identical SCC iteration counts.  The tolerances below
leave two orders of magnitude for other CPUs and BLAS builds.
"""
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ase.build import bulk  # noqa: E402

from pydftb import DFTB, ParamSet  # noqa: E402
from pydftb.ase_calc import PyDFTB  # noqa: E402
from pydftb.backend.spec import NUMPY, Backend, make_backend  # noqa: E402
from pydftb.validation import refgen  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASES = ["h2o", "ch3oh_dftb3", "water_box_gamma", "cu4co_shell_resolved", "cu_fcc_k6_T0",
         "cu8_distorted_k355_mp2_2000K", "cuo_chain_spin_k422", "o2_relaxed_spin",
         "gdo_rocksalt_k444", "gdo2_dftb3_shell", "water_dimer_uff", "ch3oh_d3bj"]


def _run(name, **kw):
    doc = refgen.load(name)
    why = refgen.missing_requirement(doc)
    if why:
        pytest.skip(why)
    return refgen.run_pydftb(refgen.CASES_BY_NAME[doc["case"]], doc["input"], **kw)


@pytest.mark.parametrize("name", CASES)
def test_float64_matches_numpy(name):
    ref = _run(name, solver="batched", scc_tol=1e-12)
    res = _run(name, backend="torch", scc_tol=1e-12)
    assert res.n_scc == ref.n_scc
    for key in ("energy", "free_energy", "e_band", "e_scc", "e_rep", "ts", "e_3rd", "e_disp"):
        assert abs(getattr(res, key) - getattr(ref, key)) < 1e-11, key
    assert np.abs(res.forces - ref.forces).max() < 1e-10
    assert np.abs(res.charges - ref.charges).max() < 1e-10
    if ref.stress is not None:
        assert np.abs(res.stress - ref.stress).max() < 1e-11
    ev, ev_ref = np.array(res.eigenvalues), np.array(ref.eigenvalues)
    assert ev.shape == ev_ref.shape and np.abs(ev - ev_ref).max() < 1e-11
    assert abs(res.fermi - ref.fermi) < 1e-9
    if ref.magmoms is not None:
        assert np.abs(res.magmoms - ref.magmoms).max() < 1e-10


@pytest.mark.parametrize("name", ["h2o", "cu8_distorted_k355_mp2_2000K", "cuo_chain_spin_k422",
                                  "gdo_rocksalt_k444"])
def test_float32_is_close(name):
    """Single precision (largest differences over the reference set: 4e-6 Ha, 2e-6 Ha/Bohr,
    1e-5 e); reported, not promised beyond these bounds."""
    ref = _run(name, solver="batched", scc_tol=1e-12)
    res = _run(name, backend="torch", dtype="float32", scc_tol=1e-5)
    assert res.converged
    assert abs(res.energy - ref.energy) < 2e-5
    assert np.abs(res.forces - ref.forces).max() < 1e-5
    assert np.abs(res.charges - ref.charges).max() < 5e-5


def test_results_are_host_arrays():
    res = _run("cuo_chain_spin_k422", backend="torch")
    assert isinstance(res.energy, float) and isinstance(res.fermi, float)
    for arr in (res.forces, res.stress, res.charges, res.q_in, res.q_shell, res.magmoms):
        assert isinstance(arr, np.ndarray) and arr.dtype == np.float64
    assert all(isinstance(e, np.ndarray) and e.dtype == np.float64 for e in res.eigenvalues)
    assert all(isinstance(o, np.ndarray) for o in res.occupations)


def test_backend_objects_and_options():
    assert make_backend() is NUMPY and make_backend("numpy") is NUMPY
    b = make_backend("torch")
    assert b == Backend("torch", "cpu", "float64") and b.is_torch
    assert make_backend(b) is b
    with pytest.raises(ValueError):
        make_backend("numpy", device="cuda")
    with pytest.raises(ValueError):
        make_backend("jax")
    sk = ParamSet(refgen.SKF_SETS["ptbp_hcocu"], max_l={"O": 1, "H": 0}).skset(["O", "H"])
    with pytest.raises(ValueError, match="own eigensolver"):
        DFTB(sk, backend="torch", solver="sp2")
    with pytest.warns(RuntimeWarning, match="k-point threads"):
        calc = DFTB(sk, backend="torch", k_threads=2)
    assert calc.k_threads.n == 1
    with pytest.warns(RuntimeWarning, match="float32"):
        DFTB(sk, backend="torch", dtype="float32", scc_tol=1e-8)


def test_conversions_are_identities_for_numpy():
    a = np.arange(6.0).reshape(2, 3)
    for f in (NUMPY.asarray, NUMPY.cast, NUMPY.widen, NUMPY.index, NUMPY.to_host):
        assert f(a) is a
    assert NUMPY.xp is np


def test_ase_calculator_threads_restored():
    atoms = bulk("Cu", "fcc", a=3.61)
    skdir = refgen.SKF_SETS["ptbp_hcocu"]
    opts = dict(max_l={"Cu": 2}, kpts=(4, 4, 4), scc_tol=1e-10, shell_resolved=True)
    a_np = atoms.copy()
    a_np.calc = PyDFTB(skdir, **opts)
    old = torch.get_num_threads()
    torch.set_num_threads(3)
    try:
        a_t = atoms.copy()
        a_t.calc = PyDFTB(skdir, backend="torch", threads=1, **opts)
        e_t, f_t = a_t.get_potential_energy(), a_t.get_forces()
        assert torch.get_num_threads() == 3            # restored after the calculation
    finally:
        torch.set_num_threads(old)
    assert abs(e_t - a_np.get_potential_energy()) < 1e-9
    assert np.abs(f_t - a_np.get_forces()).max() < 1e-9


def test_reused_charges_follow_the_numpy_path():
    atoms = bulk("Cu", "fcc", a=3.61).repeat((2, 1, 1))
    atoms.positions[0] += [0.05, 0.02, 0.0]
    skdir = refgen.SKF_SETS["ptbp_hcocu"]
    runs = {}
    for backend in ("numpy", "torch"):
        a = atoms.copy()
        a.calc = PyDFTB(skdir, max_l={"Cu": 2}, kpts=(2, 3, 3), scc_tol=1e-10,
                        shell_resolved=True, reuse_charges=True, backend=backend)
        e1 = a.get_potential_energy()
        n1 = a.calc.raw.n_scc
        a.positions[1] += [0.0, 0.03, -0.02]
        e2 = a.get_potential_energy()
        runs[backend] = (e1, e2, n1, a.calc.raw.n_scc)
    (e1, e2, n1, n2), (t1, t2, m1, m2) = runs["numpy"], runs["torch"]
    assert (n1, n2) == (m1, m2) and n2 < n1
    assert abs(e1 - t1) < 1e-9 and abs(e2 - t2) < 1e-9


FORK_SCRIPT = r"""
import os, signal, sys, time
import torch
from ase.build import bulk
from pydftb.ase_calc import PyDFTB
from pydftb.validation import refgen
skdir = refgen.SKF_SETS["ptbp_hcocu"]
def energy(threads):
    a = bulk("Cu", "fcc", a=3.61)
    a.calc = PyDFTB(skdir, max_l={"Cu": 2}, kpts=(6, 6, 6), shell_resolved=True,
                    backend="torch", threads=threads)
    return a.get_potential_energy()
torch.set_num_threads(4)
x = torch.randn(800, 800, dtype=torch.float64)
(x @ x).sum()                                   # an OpenMP region with 4 threads
e0 = energy(4)
pids = []
for _ in range(3):
    pid = os.fork()
    if pid == 0:
        signal.alarm(90)                        # a hung child must not outlive the test
        os._exit(0 if abs(energy(1) - e0) < 1e-9 else 3)
    pids.append(pid)
print(all(os.waitstatus_to_exitcode(os.waitpid(p, 0)[1]) == 0 for p in pids))
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork")
def test_forked_children_with_one_thread_do_not_hang():
    """After the parent ran torch with several OpenMP threads, a forked child that sets
    its thread count first (PyDFTB(threads=1)) runs normally (one that does not hangs:
    M4a design, section 4)."""
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, "-c", FORK_SCRIPT], cwd=ROOT, env=env, timeout=120,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[-1] == "True"


def test_namespace_follows_the_inputs():
    from pydftb.backend import ops, torch_xp
    a = np.arange(4.0)
    t = torch.arange(4.0, dtype=torch.float64)
    assert ops.namespace(a, 1.0) is np
    assert ops.namespace(a, t) is torch_xp
    assert ops.to_host(a) is a
    h = ops.to_host(torch.conj(torch.tensor([1 + 2j], dtype=torch.complex64)))
    assert h.dtype == np.complex128 and h[0] == 1 - 2j
    x = torch.randn(3, 2, 4, dtype=torch.complex128)
    xn = x.numpy()
    assert np.array_equal(torch_xp.conj(torch_xp.swapaxes(x, 1, 2)).resolve_conj().numpy(),
                          np.conj(np.swapaxes(xn, 1, 2)))
    assert np.array_equal(torch_xp.concatenate([x, x], axis=1).numpy(),
                          np.concatenate([xn, xn], axis=1))
