"""CPU budget (utils.parallel.budget) and the M3a plan rule (utils.parallel.plan)."""
import json
import random

import pytest

from utils.parallel import budget as B
from utils.parallel.plan import core_shares, dftb_stage, simple_plan


def _tree(tmp_path, proc_cgroup: str, files: dict):
    root = tmp_path / "cgroup"
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    root.mkdir(exist_ok=True)
    proc = tmp_path / "proc_cgroup"
    proc.write_text(proc_cgroup)
    return root, proc


def test_parse_ncores():
    assert B.parse_ncores(None) is None
    assert B.parse_ncores("auto") is None and B.parse_ncores(" AUTO ") is None
    assert B.parse_ncores("") is None
    assert B.parse_ncores("4") == 4 and B.parse_ncores(2) == 2
    for bad in ("0", -1, "two", 1.5j):
        with pytest.raises(ValueError):
            B.parse_ncores(bad)


def test_cgroup_v2_quota_tightest_along_the_hierarchy(tmp_path):
    root, proc = _tree(tmp_path, "0::/user.slice/job\n", {
        "cpu.max": "max 100000",
        "user.slice/cpu.max": "150000 100000",
        "user.slice/job/cpu.max": "250000 100000",
    })
    assert B.cgroup_cpu_limit(root, proc) == 1.5


def test_cgroup_v1_quota(tmp_path):
    root, proc = _tree(tmp_path, "4:cpu,cpuacct:/docker/abc\n3:memory:/docker/abc\n", {
        "cpu,cpuacct/docker/abc/cpu.cfs_quota_us": "300000",
        "cpu,cpuacct/docker/abc/cpu.cfs_period_us": "100000",
        "cpu,cpuacct/cpu.cfs_quota_us": "-1",
        "cpu,cpuacct/cpu.cfs_period_us": "100000",
    })
    assert B.cgroup_cpu_limit(root, proc) == 3.0
    unlimited, proc2 = _tree(tmp_path / "u", "4:cpu:/\n", {"cpu/cpu.cfs_quota_us": "-1",
                                                            "cpu/cpu.cfs_period_us": "100000"})
    assert B.cgroup_cpu_limit(unlimited, proc2) is None


def test_cgroup_namespaced_path_falls_back_to_the_mount(tmp_path):
    """Inside a container /proc/self/cgroup names a host path the container cannot see."""
    root, proc = _tree(tmp_path, "0::/system.slice/docker-1234.scope\n",
                       {"cpu.max": "200000 100000"})
    assert B.cgroup_cpu_limit(root, proc) == 2.0


def test_no_cgroup_information(tmp_path):
    root, proc = _tree(tmp_path, "", {})
    assert B.cgroup_cpu_limit(root, proc) is None
    assert B.cgroup_memory_limit(root, proc) is None


def test_detect_cpu_takes_the_smallest_limit(tmp_path):
    root, proc = _tree(tmp_path, "0::/\n", {"cpu.max": "250000 100000"})
    kw = dict(affinity=8, cgroup_root=root, proc_cgroup=proc)
    b = B.detect_cpu(None, {}, **kw)
    assert b.ncores == 3 and b.limited_by == ["cgroup quota"]        # 2.5 cores -> 3
    assert B.detect_cpu("auto", {"SLURM_CPUS_PER_TASK": "2"}, **kw).ncores == 2
    assert B.detect_cpu(None, {"PTBP_NCORES": "1", "SLURM_CPUS_PER_TASK": "2"}, **kw).ncores == 1
    b = B.detect_cpu(16, {}, affinity=4, cgroup_root=tmp_path / "none", proc_cgroup=proc)
    assert b.ncores == 4 and b.sources["--ncores"] == 16 and "CPU affinity 4" in b.describe()
    assert B.detect_cpu("2", {}, affinity=4, cgroup_root=tmp_path / "none",
                        proc_cgroup=proc).limited_by == ["--ncores"]


def test_available_memory(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 64000000 kB\nMemAvailable:   8000000 kB\n")
    root, proc = _tree(tmp_path, "0::/job\n", {"job/memory.max": str(6 * 2**30),
                                               "job/memory.current": str(2**30)})
    assert B.available_memory(meminfo, root=root, proc_cgroup=proc) == 5 * 2**30
    root2, proc2 = _tree(tmp_path / "x", "0::/job\n", {"job/memory.max": "max"})
    assert B.available_memory(meminfo, root=root2, proc_cgroup=proc2) == 8000000 * 1024


def test_thread_env(monkeypatch):
    env = B.thread_env(3)
    assert env["OMP_NUM_THREADS"] == "3" and env["OPENBLAS_NUM_THREADS"] == "3"
    assert env["OMP_PROC_BIND"] is None
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    import os
    with B.applied_env(env):
        assert os.environ["OMP_NUM_THREADS"] == "3" and "OMP_PROC_BIND" not in os.environ
    assert os.environ["OMP_NUM_THREADS"] == "16" and os.environ["OMP_PROC_BIND"] == "close"


def test_core_shares():
    assert core_shares(4, 1) == [4]
    assert core_shares(4, 4) == [1, 1, 1, 1]
    assert core_shares(5, 2) == [3, 2]
    assert core_shares(4, 8) == [1] * 8            # oversubscribed: at least one each


@pytest.mark.parametrize("ncores, n, workers, threads", [
    (4, 66, 4, 1), (4, 1, 1, 4), (4, 2, 2, 2), (4, 3, 3, 1), (1, 66, 1, 1), (8, 3, 3, 2),
])
def test_dftb_stage_rule(ncores, n, workers, threads):
    st = dftb_stage(ncores, n, "pydftb")
    assert (st.workers, st.threads) == (workers, threads)
    assert st.order == ("largest-first" if workers > 1 else "index")
    if workers * threads < ncores:
        assert any("idle" in r for r in st.reasons)


def _budget(n):
    return B.CpuBudget(n, {"CPU affinity": n})


def test_processes_times_threads_never_exceed_the_cores():
    rng = random.Random(7)
    for _ in range(500):
        c, s, p = rng.randint(1, 64), rng.randint(0, 300), rng.randint(1, 9)
        plan = simple_plan(_budget(c), s, rng.choice(["pydftb", "dftbplus"]), "parallel_bo", p)
        shares = core_shares(c, p)
        if p <= c:
            assert sum(shares) == c
        for i in range(p):
            sub = plan.for_particle(i)
            assert sub.stage("dftb").cores <= max(1, shares[i])
            assert sub.particles == 1


def test_parallel_bo_splits_the_cores():
    plan = simple_plan(_budget(4), 66, "dftbplus", "parallel_bo", 4)
    assert plan.particles == 4 and plan.ncores == 1
    assert all(plan.for_particle(i).stage("dftb").cores == 1 for i in range(4))
    # bo and pso evaluate one point at a time: all cores for it
    assert simple_plan(_budget(4), 66, "dftbplus", "bo", 4).stage("dftb").workers == 4
    over = simple_plan(_budget(4), 66, "dftbplus", "parallel_bo", 8)
    assert any("oversubscribed" in n for n in over.notes)


def test_render_and_json(tmp_path):
    plan = simple_plan(_budget(4), 2, "pydftb")
    text = plan.render()
    assert "2 processes x 2 k-point threads" in text and "largest first" in text
    assert "hotcent: 4 processes x 1 thread" in text
    assert "hotcent: serial" in simple_plan(_budget(1), 2, "pydftb").render()
    plan.save(tmp_path / "plan.json")
    doc = json.loads((tmp_path / "plan.json").read_text())
    assert doc["budget"]["ncores"] == 4 and doc["stages"]["dftb"]["threads"] == 2
    assert "1 process x 4 OpenMP threads" in simple_plan(_budget(4), 1, "dftbplus").render()
    assert "DFTB: serial" in simple_plan(_budget(1), 66, "dftbplus").render()


def test_cli_forwards_ncores(tmp_path, monkeypatch):
    import sys
    from unittest.mock import patch
    dataset = tmp_path / "my_data.xyz"
    dataset.write_text("0\n\n")
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_run_path(path, run_name=None):
        captured["argv"] = list(sys.argv)

    with patch("cli.cmd_optimize.runpy.run_path", side_effect=fake_run_path):
        from cli.main import dispatch
        assert dispatch(["optimize", str(dataset), "--output", str(tmp_path / "run"),
                         "--mode", "dataset", "--ncores", "2"]) == 0
    argv = captured["argv"]
    assert argv[argv.index("--ncores") + 1] == "2"
    from cli.arg_parser import build_default_arg_parser
    monkeypatch.setattr(sys, "argv", argv)
    assert build_default_arg_parser().ncores == "2"
    monkeypatch.setattr(sys, "argv", ["run.py"])
    assert build_default_arg_parser().ncores == "auto"           # parallel by default
    from cli import config
    assert config.YAML_KEY_TO_ARG["ncores"] == "ncores"
    assert "ncores: '2'" in (tmp_path / "run" / "ptbp_run.yaml").read_text()
