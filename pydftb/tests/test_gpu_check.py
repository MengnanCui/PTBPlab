"""The GPU self-check (milestone M4d) runs on the CPU here: every check passes, a mismatch
makes it fail with exit status 1, and the timings produce their tables."""
import json

import pytest

pytest.importorskip("torch")

from pydftb.validation import gpu_check  # noqa: E402


def _quiet(*args, **kw):
    pass


def test_checks_pass_on_the_cpu(tmp_path):
    report = gpu_check.run("cpu", quick=True, timing=False, cases=["h2o"], log=_quiet)
    assert report["ok"]
    names = [c["name"] for c in report["checks"]]
    assert "references h2o [float64]" in names and "cpu h2o [float64]" in names
    assert any(n.startswith("determinism") for n in names)
    assert any(n.startswith("autodiff") for n in names)
    fork = next(c for c in report["checks"] if c["name"] == "fork after CUDA")
    assert fork["ok"] is None                                   # CUDA only
    assert report["environment"]["capabilities"]["kind"] == "cpu"


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
    assert not report["ok"]
    assert [c["name"] for c in report["checks"] if c["ok"] is False] == ["cpu h2o [float64]"]


def test_cu_timings_on_the_cpu(monkeypatch):
    monkeypatch.setattr(gpu_check, "CU_CELLS", [("Cu1 k3", "prim", (1, 1, 1), (3, 3, 3))])
    row, = gpu_check.time_cu("cpu", "float64", quick=True)
    for tag in ("numpy_1", "numpy_kthreads", "device"):
        assert row[f"{tag}_s"] > 0 and row[f"{tag}_converged"]
        assert row[f"{tag}_per_scc_s"] == row[f"{tag}_s"] / row[f"{tag}_n_scc"]
    assert row["numpy_1_n_scc"] == row["numpy_kthreads_n_scc"]    # k threads: the same bits
    assert row["energy_diff"] < 1e-8                               # SCC to 1e-5 only


def test_timings_on_the_cpu():
    rows = gpu_check.time_eigh("cpu", (9,), (1, 4), ["float64", "complex64"], repeat=1)
    assert len(rows) == 4 and all(r["device_s"] > 0 and r["numpy_s"] > 0 for r in rows)
    rows = gpu_check.time_stacked("cpu", sizes=(9,), structures=2, nk=3, repeat=1)
    assert rows[0]["structures"] == 2 and rows[0]["gain"] > 0
