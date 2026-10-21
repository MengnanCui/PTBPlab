"""Fork backend (utils.parallel.backends): order, errors, crashes, cleanup, nesting, threads."""
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from utils.parallel import ForkBackend, SerialBackend, WorkerDied, default_backend, map_tasks

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork backend: Linux")


class SkfMissing(FileNotFoundError):
    """Importable subclass: must come back with its own type."""


def _square(x):
    return x * x


def test_results_in_task_order():
    res = map_tasks(_square, range(20), workers=4, order=list(reversed(range(20))))
    assert [r.index for r in res] == list(range(20))
    assert [r.value for r in res] == [x * x for x in range(20)]
    assert all(r.ok and r.wall >= 0 for r in res)
    assert len({r.pid for r in res}) > 1 and os.getpid() not in {r.pid for r in res}


def test_tasks_are_handed_out_in_the_given_order():
    def start_time(_):
        t = time.monotonic()
        time.sleep(0.01)
        return t
    order = [3, 0, 4, 1, 2]
    res = ForkBackend(1).map(start_time, list(range(5)), order=order)
    assert sorted(range(5), key=lambda i: res[i].value) == order


def test_largest_first_balances_the_load():
    """One long task first and the short ones around it: about the long task's time."""
    durations = [0.6] + [0.1] * 6
    t0 = time.monotonic()
    map_tasks(time.sleep, durations, workers=2, order=list(range(7)))
    assert time.monotonic() - t0 < 1.1          # badly ordered: 0.3 + 0.6 + ... > 1.2


def _failing(x):
    if x == 1:
        raise FileNotFoundError(2, "No such file or directory", "Cu-Cu.skf")
    if x == 2:
        raise SkfMissing(2, "missing", "O-O.skf")
    if x == 3:
        class Local(RuntimeError):
            pass
        raise Local("defined inside the task")
    if x == 4:
        return lambda: None                     # a result that cannot be pickled
    return x


def test_exceptions_come_back_with_their_type():
    res = map_tasks(_failing, range(5), workers=3)
    assert res[0].ok and res[0].value == 0
    assert type(res[1].error) is FileNotFoundError and res[1].error.filename == "Cu-Cu.skf"
    assert type(res[2].error) is SkfMissing and res[2].error.filename == "O-O.skf"
    # not picklable: the nearest built-in type, the original class name in the message
    assert type(res[3].error) is RuntimeError and "Local: defined inside" in str(res[3].error)
    assert not res[4].ok and isinstance(res[4].error, Exception)
    assert "_failing" in res[1].error.remote_traceback
    with pytest.raises(FileNotFoundError):
        res[1].get()


def test_serial_backend_has_the_same_semantics():
    ser = map_tasks(_failing, range(4), workers=1)
    par = map_tasks(_failing, range(4), workers=2)
    assert [type(r.error) for r in ser[:3]] == [type(r.error) for r in par[:3]]
    # a class that cannot travel between processes stays itself in the serial backend
    assert isinstance(ser[3].error, RuntimeError) and isinstance(par[3].error, RuntimeError)
    assert [r.value for r in ser] == [r.value for r in par]
    assert {r.pid for r in ser} == {os.getpid()}


def _crash(x):
    if x == 2:
        os._exit(3)
    if x == 5:
        os.kill(os.getpid(), signal.SIGKILL)
    return x


def test_a_dead_worker_does_not_hang_the_parent():
    res = map_tasks(_crash, range(8), workers=3)
    assert isinstance(res[2].error, WorkerDied) and "exit code 3" in str(res[2].error)
    assert isinstance(res[5].error, WorkerDied)
    assert [r.value for r in res if r.ok] == [0, 1, 3, 4, 6, 7]


def test_all_workers_dead():
    res = map_tasks(os._exit, [1, 1, 1, 1], workers=2)
    assert all(isinstance(r.error, WorkerDied) for r in res)


def _nested(x):
    return sum(r.value for r in map_tasks(_square, range(x), workers=2))


def test_workers_may_start_workers():
    """Not daemonic: map_tasks inside a task (DFTB stage inside a parallel_bo particle)."""
    res = map_tasks(_nested, [3, 4], workers=2)
    assert [r.value for r in res] == [5, 14]


def _thread_settings(_):
    from threadpoolctl import threadpool_info
    blas = {i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"}
    return {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                           "OMP_PROC_BIND")}, blas


def test_workers_get_the_planned_threads(monkeypatch):
    import numpy  # noqa: F401  (load BLAS)
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    for r in map_tasks(_thread_settings, range(2), workers=2, threads=3):
        env, blas = r.value
        assert env == {"OMP_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3", "OMP_PROC_BIND": None}
        assert blas == {1}
    assert os.environ["OMP_NUM_THREADS"] == "64"            # the parent is untouched
    env, _ = map_tasks(_thread_settings, [0], workers=1, threads=2)[0].value
    assert env["OMP_NUM_THREADS"] == "2"                    # serial: for the stage only
    assert os.environ["OMP_NUM_THREADS"] == "64"


def test_backend_choice(monkeypatch):
    assert isinstance(default_backend(1), SerialBackend)
    assert isinstance(default_backend(4), ForkBackend)
    monkeypatch.setenv("PTBP_BACKEND", "serial")
    assert isinstance(default_backend(4), SerialBackend)
    monkeypatch.setenv("PTBP_BACKEND", "mpi")
    with pytest.raises(ValueError):
        default_backend(4)
    with pytest.raises(ValueError):
        ForkBackend(2).map(_square, [1, 2], order=[0, 0])


DRIVER = textwrap.dedent("""
    import os, subprocess, sys, time
    sys.path.insert(0, {root!r})
    from utils.parallel import map_tasks
    out = {out!r}

    def task(i):
        p = subprocess.Popen(["sleep", "60"])            # stands for a DFTB+ run
        with open(os.path.join(out, f"{{i}}.pids"), "w") as fh:
            fh.write(f"{{os.getpid()}} {{p.pid}}")
        p.wait()

    map_tasks(task, range(3), workers=3)
""")


def _gone(pid):
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] == "Z"
    except FileNotFoundError:
        return True


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGKILL])
def test_no_processes_survive_the_parent(tmp_path, sig):
    """Ctrl-C (SIGINT), SIGTERM or SIGKILL of PTBP: the workers and their DFTB+ end too."""
    driver = subprocess.Popen([sys.executable, "-c", DRIVER.format(root=str(ROOT),
                                                                   out=str(tmp_path))])
    deadline = time.monotonic() + 30
    while len(list(tmp_path.glob("*.pids"))) < 3:
        assert time.monotonic() < deadline and driver.poll() is None
        time.sleep(0.05)
    pids = [int(p) for f in tmp_path.glob("*.pids") for p in f.read_text().split()]
    driver.send_signal(sig)
    driver.wait(timeout=30)
    deadline = time.monotonic() + 15
    while not all(_gone(p) for p in pids):
        assert time.monotonic() < deadline, [p for p in pids if not _gone(p)]
        time.sleep(0.05)
