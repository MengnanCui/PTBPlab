"""The GPU self-check (milestones M4d, M7e) runs on the CPU here: every check passes, a
mismatch makes it fail with exit status 1, the JSON is written whatever happens (a timing
error, a check that raises, Ctrl-C), the timings follow ``--dtype`` and the device's
capabilities, and the PTBP stage and the stacked solve produce their tables."""
import json
import os
import shutil
import subprocess

import pytest

pytest.importorskip("torch")

from pydftb.backend import devices  # noqa: E402
from pydftb.validation import gpu_check, ptbp_workload  # noqa: E402

Capabilities, Structure, structures = (devices.Capabilities, ptbp_workload.Structure,
                                       ptbp_workload.structures)


def _quiet(*args, **kw):
    pass


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def test_checks_pass_on_the_cpu(tmp_path):
    report = gpu_check.run("cpu", quick=True, timing=False, cases=["h2o"], log=_quiet)
    assert report["ok"] and report["complete"] and report["errors"] == []
    names = [c["name"] for c in report["checks"]]
    assert "references h2o [float64]" in names and "cpu h2o [float64]" in names
    assert any(n.startswith("determinism") for n in names)
    assert any(n.startswith("autodiff") for n in names)
    fork = next(c for c in report["checks"] if c["name"] == "fork after CUDA")
    assert fork["ok"] is None                                   # CUDA only
    env = report["environment"]
    assert env["capabilities"]["kind"] == "cpu" and env["usable_cpus"] >= 1
    assert env["pydftb"] and os.path.isdir(env["pydftb_path"]) and env["threadpools"]
    assert [s["section"] for s in report["sections"]][:2] == ["environment", "references [float64]"]


def test_a_mismatch_fails(tmp_path, monkeypatch):
    real = gpu_check.compare_case

    def off(doc, device, dtype):
        ref, cpu, res = real(doc, device, dtype)
        return ref, dict(cpu, energy=1e-6), res
    monkeypatch.setattr(gpu_check, "compare_case", off)
    monkeypatch.setattr(gpu_check, "AUTODIFF_CASES", [])
    monkeypatch.setattr(gpu_check, "DETERMINISM_CASES", [])
    out = tmp_path / "report.json"
    code = gpu_check.main(["--device", "cpu", "--quick", "--skip-timing", "--cases", "h2o",
                           "--json", str(out)])
    assert code == 1
    report = json.loads(out.read_text())
    assert not report["ok"] and report["complete"] and report["pydftb_gpu_check"] == 2
    assert [c["name"] for c in report["checks"] if c["ok"] is False] == ["cpu h2o [float64]"]


# ---- the JSON whatever happens, exit codes, dtypes (checks and timings replaced) ------------
def _fast(monkeypatch) -> list:
    """Instant checks and timings; returns the calls of the timings, (name, args)."""
    calls = []

    def ok(name):
        return [gpu_check.Check(name, True, {})]
    monkeypatch.setattr(gpu_check, "check_references",
                        lambda dev, dtype, names: ok(f"references [{dtype}]"))
    monkeypatch.setattr(gpu_check, "check_determinism",
                        lambda dev, dtype: ok(f"determinism [{dtype}]"))
    monkeypatch.setattr(gpu_check, "check_autodiff", lambda dev, names: ok("autodiff"))
    monkeypatch.setattr(gpu_check, "check_fork",
                        lambda dev: gpu_check.Check("fork after CUDA", None, {}))

    def timing(name, result):
        def fake(*args, **kw):
            calls.append((name, args))
            return result
        return fake
    for name in ("time_eigh", "time_cu", "time_stacked_solve"):
        monkeypatch.setattr(gpu_check, name, timing(name, [{"fake": name}]))
    monkeypatch.setattr(gpu_check, "time_ptbp_stage", timing("time_ptbp_stage", {"summary": {}}))
    return calls


def test_the_json_is_written_after_a_timing_error(tmp_path, monkeypatch):
    _fast(monkeypatch)

    def broken(*args, **kw):
        raise RuntimeError("no kernel image is available for execution on the device")
    monkeypatch.setattr(gpu_check, "time_eigh", broken)
    out = tmp_path / "report.json"
    code = gpu_check.main(["--device", "cpu", "--quick", "--timings", "eigh,cu",
                           "--json", str(out)])
    assert code == 1                                            # a section could not run
    report = json.loads(out.read_text())
    assert report["complete"] and report["ok"] and report["errors"] == ["eigh"]
    eigh = next(s for s in report["sections"] if s["section"] == "eigh")
    assert "no kernel image" in eigh["error"] and "Traceback" in eigh["traceback"]
    assert eigh["wall_s"] >= 0 and "max_rss_gb" in eigh
    assert report["timings"] == {"cu": [{"fake": "time_cu"}]}   # the next section ran
    assert report["timing_sections"] == ["eigh", "cu"]


def test_a_check_that_raises_fails_and_the_rest_runs(tmp_path, monkeypatch):
    _fast(monkeypatch)

    def broken(dev, dtype, names):
        raise ValueError("a broken case")
    monkeypatch.setattr(gpu_check, "check_references", broken)
    out = tmp_path / "report.json"
    code = gpu_check.main(["--device", "cpu", "--quick", "--skip-timing", "--json", str(out)])
    assert code == 1
    report = json.loads(out.read_text())
    failed = [c for c in report["checks"] if c["ok"] is False]
    assert [c["name"] for c in failed] == ["references [float64]"]
    assert "a broken case" in failed[0]["details"]["error"]
    assert [c["name"] for c in report["checks"]][1:] == ["determinism [float64]", "autodiff",
                                                         "fork after CUDA"]


def test_an_interrupt_leaves_the_json(tmp_path, monkeypatch):
    _fast(monkeypatch)

    def stop(*args, **kw):
        raise KeyboardInterrupt
    monkeypatch.setattr(gpu_check, "time_cu", stop)
    out = tmp_path / "report.json"
    code = gpu_check.main(["--device", "cpu", "--quick", "--timings", "eigh,cu,ptbp",
                           "--json", str(out)])
    assert code == 2
    report = json.loads(out.read_text())
    assert report["complete"] is False and report["interrupted"]
    assert report["interrupted_in"] == "cu" and report["ok"]
    assert list(report["timings"]) == ["eigh"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]   # no temporary file
    assert out.stat().st_mode & 0o777 == 0o666 & ~_umask()            # as a new file


def test_both_dtypes_reach_every_timing(monkeypatch):
    calls = _fast(monkeypatch)
    report = gpu_check.run("cpu", ("float64", "float32"), quick=True, log=_quiet)
    args = dict(calls)
    assert args["time_eigh"][3] == ["float64", "complex128", "float32", "complex64"]
    for name in ("time_cu", "time_ptbp_stage", "time_stacked_solve"):
        assert tuple(args[name][1]) == ("float64", "float32")
    assert report["dtypes"] == ["float64", "float32"] and report["dtypes_skipped"] == []
    assert [c["name"] for c in report["checks"]][:4] == [
        "references [float64]", "determinism [float64]", "references [float32]",
        "determinism [float32]"]


def test_a_device_without_float64_times_float32_only(monkeypatch):
    calls = _fast(monkeypatch)
    mps = Capabilities("cpu", "cpu", "MPS-like", float64=False,
                       notes=("no float64: dtype='float32' only",))
    monkeypatch.setattr(devices, "capabilities", lambda dev: mps)
    report = gpu_check.run("cpu", ("float64", "float32"), quick=True, log=_quiet)
    args = dict(calls)
    assert args["time_eigh"][3] == ["float32", "complex64"]
    for name in ("time_cu", "time_ptbp_stage", "time_stacked_solve"):
        assert tuple(args[name][1]) == ("float32",)
    assert report["dtypes"] == ["float32"]
    assert report["dtypes_skipped"] == [{"dtype": "float64", "reason": "no float64 on the device"}]
    assert not any("float64" in c["name"] for c in report["checks"])
    no_complex = Capabilities("cpu", "cpu", "x", float64=False, complex_=False)
    assert gpu_check.timing_kinds(no_complex, ("float32",)) == (
        ["float32"], [{"dtype": "complex64", "reason": "no complex arithmetic on the device"}])


def test_unknown_timings_and_workers_are_refused():
    for argv in (["--timings", "eigh,gemm"], ["--workers", "0"]):
        with pytest.raises(SystemExit):
            gpu_check.main(argv)


# ---- environment ---------------------------------------------------------------------------
def test_the_environment_without_git(monkeypatch):
    real = subprocess.run

    def no_git(cmd, *args, **kw):
        if cmd[0] == "git":
            raise FileNotFoundError("git")
        return real(cmd, *args, **kw)
    monkeypatch.setattr(gpu_check.subprocess, "run", no_git)
    env = gpu_check.environment("cpu", argv=["gpu-check", "--quick"])
    assert env["pydftb_commit"] is None and env["pydftb_dirty"] is None
    assert env["command_line"] == ["gpu-check", "--quick"] and env["numpy"]


@pytest.mark.skipif(shutil.which("git") is None, reason="no git")
def test_an_installed_copy_takes_no_commit_from_a_surrounding_repository(monkeypatch, tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    copy = tmp_path / "lib" / "site-packages" / "pydftb" / "validation"
    copy.mkdir(parents=True)
    (copy / "gpu_check.py").write_text("")
    monkeypatch.setattr(gpu_check, "__file__", str(copy / "gpu_check.py"))
    assert gpu_check._git_commit() == (None, None)


# ---- timings on the CPU --------------------------------------------------------------------
def test_cu_timings_on_the_cpu(monkeypatch):
    monkeypatch.setattr(gpu_check, "CU_CELLS", [("Cu1 k3", "prim", (1, 1, 1), (3, 3, 3))])
    row, = gpu_check.time_cu("cpu", ("float64", "float32"), quick=True)
    for tag in ("numpy_1", "numpy_kthreads", "device_float64", "device_float32"):
        assert row[f"{tag}_s"] > 0 and row[f"{tag}_converged"]
        assert row[f"{tag}_per_scc_s"] == row[f"{tag}_s"] / row[f"{tag}_n_scc"]
    assert row["numpy_1_n_scc"] == row["numpy_kthreads_n_scc"]    # k threads: the same bits
    assert row["energy_diff_float64"] < 1e-8                       # SCC to 1e-5 only
    assert row["energy_diff_float32"] < 1e-4


def test_eigh_timings_on_the_cpu():
    rows = gpu_check.time_eigh("cpu", (9,), (1, 4), ["float64", "complex64"], repeat=1)
    assert len(rows) == 4 and all(r["device_s"] > 0 and r["numpy_s"] > 0 for r in rows)


def test_list_scheduling():
    assert gpu_check._lpt([3, 3, 2, 2, 2], 2) == 7               # largest first, not optimal
    assert gpu_check._lpt([1.0] * 4, 4) == 1.0 and gpu_check._lpt([], 3) == 0.0


def _small_stage():
    """A Gamma-point dimer and a bcc cell on a 4 x 4 x 4 grid, from PTBP's Cu stage."""
    dimer = structures(phases=["dimer"])[0]
    bcc = structures(phases=["bcc"])[10]
    return [dimer, Structure("bcc 10 k4", "bcc", 10, bcc.atoms, (4, 4, 4), bcc.norb)]


def test_a_small_ptbp_stage_on_the_cpu():
    out = gpu_check.time_ptbp_stage("cpu", ("float64", "float32"), structs=_small_stage(),
                                    workers=1)
    modes = ["numpy_1", "numpy_kthreads", "numpy_procs", "device_float64", "device_float32"]
    assert out["modes"] == modes and out["workers"] == 1
    for row in out["structures"]:
        for mode in modes:
            r = row["modes"][mode]
            assert r["wall_s"] > 0 and r["converged"] and r["n_scc"] > 1 and r["fill_s"] > 0
            assert {"build", "diag", "forces", "total"} <= set(r["timings"])
        for mode in ("numpy_kthreads", "numpy_procs"):             # the same bits as numpy_1
            assert row["modes"][mode]["dE"] == 0.0 and row["modes"][mode]["dF"] == 0.0
        assert row["modes"]["numpy_procs"]["pid"] != os.getpid()
        dev = row["modes"]["device_float64"]
        assert max(dev["dE"], dev["dF"], dev.get("dS", 0.0)) <= 1e-8
    assert out["structures"][1]["modes"]["numpy_1"]["nk"] == 32   # 4^3, time reversal
    s = out["summary"]
    assert s["numpy_procs"]["structures"] == 2 and s["numpy_procs"]["makespan_s"] > 0
    assert out["pool"]["order"] == ["bcc 10 k4", "dimer 00"]       # largest first
    dev = s["device_float64"]
    assert dev["cpu_work_s"] > 0 and dev["device_solve_s"] > 0 and "cold_start_s" in dev
    assert dev["pipeline_bound_s"] == max(dev["cpu_work_s"], dev["device_solve_s"])
    assert dev["vs_pool"] > 0 and dev["vs_numpy_1_over_workers"] > 0
    json.dumps(out, default=gpu_check._jsonable)                  # no arrays left behind
    lines = gpu_check.verdicts({"timings": {"ptbp": out}})
    assert lines[0].startswith("PTBP stage on the device [float64]")


def test_stacking_the_solve_gives_the_one_by_one_results():
    dimers = structures(quick=True, phases=["dimer"])             # Gamma: real
    bcc = [Structure(s.name, s.phase, s.eos, s.atoms, (3, 3, 3), s.norb)
           for s in structures(quick=True, phases=["bcc"])]       # k-points: complex
    groups = [("dimer", dimers), ("bcc k3", bcc)]
    rows = gpu_check.time_stacked_solve("cpu", ("float64", "float32"), groups=groups, repeat=1)
    assert [(r["group"], r["dtype"]) for r in rows] == [
        ("dimer", "float64"), ("bcc k3", "complex128"), ("dimer", "float32"),
        ("bcc k3", "complex64")]
    for r in rows:
        assert r["kpoints"] == (3 if r["group"] == "dimer" else 3 * 14)
        for tag in ("prepare", "iteration", "final", "eigh"):
            assert r[tag]["one_by_one_s"] > 0 and r[tag]["stacked_s"] > 0 and r[tag]["gain"] > 0
        assert r["fill_s"] > 0
        if r["dtype"] in ("float64", "complex128"):
            assert max(r["max_deig"], r["max_dmu"], r["max_dq"]) <= gpu_check.STACK_TOL
        else:
            assert max(r["max_deig"], r["max_dq"]) <= 1e-4
    lines = gpu_check.verdicts({"timings": {"stacked": rows}})
    assert len(lines) == 4 and lines[1].startswith("stacking the solve, bcc k3 [complex128]")
    skipped, = gpu_check.time_stacked_solve("cpu", ("float64",), groups=groups[:1], max_bytes=1)
    assert skipped["skipped"].startswith("needs up to")
