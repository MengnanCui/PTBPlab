"""Threads over k-points (pydftb.parallel): bitwise identical results for any number.

``min_work=1`` splits even the smallest k stacks, so every case below really runs in
several chunks (also with fewer k-points than threads).
"""
import dataclasses
import os

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from pydftb.ase_calc import PyDFTB
from pydftb.parallel import KThreads, as_kthreads
from pydftb.validation import refgen

from .conftest import SKF_HCOCU

CASES = ["cu_fcc_k6", "cu8_distorted_k355_mp2_2000K", "cuo_chain_spin_k422",
         "cuo_chain_dftb3_k422", "gdo_rocksalt_k444", "h2o"]


def _bits(x):
    if isinstance(x, (list, tuple)):
        return [_bits(v) for v in x]
    return None if x is None else (np.asarray(x).dtype.str, np.shape(x), np.asarray(x).tobytes())


def _assert_identical(ref, res):
    for f in dataclasses.fields(ref):
        if f.name != "timings":
            assert _bits(getattr(res, f.name)) == _bits(getattr(ref, f.name)), f.name


@pytest.mark.parametrize("solver", ["batched", "eigh"])
@pytest.mark.parametrize("name", CASES)
def test_k_threads_are_bitwise_identical(name, solver):
    case = refgen.CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    with threadpool_limits(1, user_api="blas"):
        ref = refgen.run_pydftb(case, inp, solver=solver)
    for n in (2, 3, 4):
        res = refgen.run_pydftb(case, inp, solver=solver, k_threads=KThreads(n, min_work=1))
        _assert_identical(ref, res)


def test_fewer_k_points_than_threads():
    case = refgen.CASES_BY_NAME["cuo_chain_spin_k422"]
    inp = dict(refgen._inputs(case))
    inp["kpoints"], inp["kweights"] = inp["kpoints"][:3], [1 / 3] * 3
    with threadpool_limits(1, user_api="blas"):
        ref = refgen.run_pydftb(case, inp, solver="batched")
    kt = KThreads(8, min_work=1)
    assert len(kt.chunks(3, 36)) == 3
    _assert_identical(ref, refgen.run_pydftb(case, inp, solver="batched", k_threads=kt))


def test_ase_calculator_with_k_threads():
    from ase.build import bulk
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
    atoms.rattle(0.05, seed=1)
    out = []
    for k_threads in (1, 4):
        calc = PyDFTB(SKF_HCOCU, kpts=(4, 4, 4), smearing=("fermi", 1000.0), threads=1,
                      k_threads=KThreads(k_threads, min_work=1))
        a = atoms.copy()
        a.calc = calc
        out.append((a.get_potential_energy(), a.get_forces(), a.get_stress()))
    for x, y in zip(*out):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


def test_chunks():
    assert KThreads(1).chunks(108, 9) == [slice(0, 108)]
    # 9 orbitals: at least 16 k-points per chunk
    assert KThreads(4).chunks(108, 9) == [slice(0, 27), slice(27, 54), slice(54, 81), slice(81, 108)]
    assert KThreads(4).chunks(8, 9) == [slice(0, 8)]
    assert KThreads(4).chunks(40, 9) == [slice(0, 20), slice(20, 40)]
    # large matrices: one k-point is worth a thread
    assert KThreads(4).chunks(3, 400) == [slice(0, 1), slice(1, 2), slice(2, 3)]
    for nk in range(1, 30):
        parts = KThreads(4, min_work=1).chunks(nk, 9)
        assert parts[0].start == 0 and parts[-1].stop == nk
        assert all(a.stop == b.start for a, b in zip(parts, parts[1:]))
        assert all(p.stop > p.start for p in parts)


def test_as_kthreads():
    assert as_kthreads(None).n == 1
    assert as_kthreads(3).n == 3
    assert as_kthreads("auto").n == len(os.sched_getaffinity(0))
    kt = KThreads(2)
    assert as_kthreads(kt) is kt
    with pytest.raises(ValueError):
        as_kthreads(0)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_pool_is_closed_before_fork_and_rebuilt_in_the_child():
    kt = KThreads(2, min_work=1)
    size = lambda sl: sl.stop - sl.start                       # noqa: E731
    assert kt.map(size, 10, 9) == [5, 5]
    assert kt._pool is not None
    pid = os.fork()
    if pid == 0:                                               # child: no threads inherited
        try:
            os._exit(0 if kt.map(size, 10, 9) == [5, 5] else 1)
        except BaseException:
            os._exit(2)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert kt._pool is None                                    # closed by the at-fork hook
    assert kt.map(size, 10, 9) == [5, 5]                       # and rebuilt on use
    kt.close()
