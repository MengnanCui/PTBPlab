"""The DFTB+ configuration advisor inside PTBP runs (utils.advisor_hook, M3d)."""
import gzip
import json
import shutil
import sys
from pathlib import Path

import ase.io
import pytest

import utils.data as data
import utils.engines as eng
from utils import advisor_hook, dftb_stage
from utils.parallel import CpuBudget, set_plan, simple_plan

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "pydftb" / "tests" / "data" / "advisor" / "ptbp_legacy"
CU_SKF_GZ = ROOT / "pydftb" / "tests" / "data" / "skf" / "ptbp_hcocu" / "Cu-Cu.skf.gz"
HAVE_DFTBPLUS = shutil.which("dftb+") is not None


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    monkeypatch.delenv("PTBP_ADVISOR", raising=False)
    saved = eng.settings()
    yield
    advisor_hook.reset()
    eng._SETTINGS.clear()
    eng._SETTINGS.update(saved)
    set_plan(None)


def _checks(run_dir):
    return json.loads((run_dir / "advisor.json").read_text())["checks"]


def test_checks_once_and_marks_what_ptbp_expects(tmp_path, capsys):
    advisor_hook.configure(tmp_path, legacy=True)
    findings = advisor_hook.check_once(FIXTURE)
    assert {f.rule for f in findings} >= {"P01", "P03"}
    assert all(f.expected for f in findings if f.rule in ("P01", "P03"))
    out = capsys.readouterr().out
    assert out.startswith("[advisor] DFTB+ advisor:") and "[advisor] [P03] warning" in out
    assert advisor_hook.check_once(FIXTURE) is None              # once per run
    checks = _checks(tmp_path)
    assert len(checks) == 1 and checks[0]["when"] == "first DFTB+ run"
    assert {f["rule"] for f in checks[0]["findings"]} >= {"P01", "P03"}


def test_not_legacy_input_is_not_expected(tmp_path):
    advisor_hook.configure(tmp_path, legacy=False)               # --smearing fermi etc.
    assert not any(f.expected for f in advisor_hook.check_once(FIXTURE))


def test_off_and_unconfigured(tmp_path, monkeypatch):
    assert advisor_hook.check_once(FIXTURE) is None              # library use: nothing
    monkeypatch.setenv("PTBP_ADVISOR", "0")
    advisor_hook.configure(tmp_path)
    assert not advisor_hook.wants_check() and advisor_hook.check_once(FIXTURE) is None
    assert not (tmp_path / "advisor.json").exists()


def test_never_raises(tmp_path, capsys):
    broken = tmp_path / "run"
    shutil.copytree(FIXTURE, broken)
    (broken / "dftb_pin.hsd").write_text("Hamiltonian = DFTB {\n")   # truncated pin
    (broken / "dftb.out").write_bytes(b"\xff\xfe garbage")
    advisor_hook.configure(tmp_path)
    findings = advisor_hook.check_once(broken)
    assert findings is not None and any(f.rule == "P01" for f in findings)
    advisor_hook.reset()
    advisor_hook.configure(tmp_path)
    (broken / "dftb_in.hsd").write_text("Hamiltonian = {\n")         # unreadable input
    assert advisor_hook.check_once(broken) is None
    assert "[advisor] not checked" in capsys.readouterr().out
    err = RuntimeError("dftb+ failed")
    advisor_hook.explain_failure(broken, err)                        # nothing to add, no raise


def test_failure_report_is_attached_to_the_error(tmp_path):
    advisor_hook.configure(tmp_path)
    err = RuntimeError("dftb+ failed")
    advisor_hook.explain_failure(FIXTURE, err)
    assert any("[advisor] [P03]" in note for note in err.__notes__)
    assert _checks(tmp_path)[-1]["when"] == "failed DFTB+ run"


# ---------------------------------------------------------------------------------------
# with DFTB+: the DFTB stage shows the advisor its first run

@pytest.fixture(scope="module")
def dataset():
    images = ase.io.read(ROOT / "dft" / "dft.xyz", ":")
    _, bolt = data.get_bolt_factor(images[5::11], "boltzmann", 3000)
    pick = [0, 1, 2, 60, 63, 65]
    return [images[i] for i in pick], [bolt[i] for i in pick]


def _stage(workdir, dataset, ncores, monkeypatch):
    images, bolt = dataset
    workdir.mkdir(parents=True, exist_ok=True)
    with gzip.open(CU_SKF_GZ, "rb") as fi, open(workdir / "Cu-Cu.skf", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    eng.configure(engine="dftbplus")
    set_plan(simple_plan(CpuBudget(ncores, {"--ncores": ncores}), len(images), "dftbplus"))
    advisor_hook.configure(workdir, legacy=True)
    monkeypatch.chdir(workdir)
    return dftb_stage.run_dataset(images, bolt, 3, 5.0)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DFTBPLUS or not sys.platform.startswith("linux"),
                    reason="dftb+ binary and fork backend needed")
@pytest.mark.parametrize("ncores", [1, 4])
def test_dftb_stage_runs_the_advisor_once(tmp_path, dataset, monkeypatch, ncores):
    import hashlib

    def digests(d):
        return {f: hashlib.sha256((d / f).read_bytes()).hexdigest()
                for f in ("dft_final.xyz", "dftb_band.xyz", "search_index.txt")}

    dftb_stage.write_outputs(*_stage(tmp_path, dataset, ncores, monkeypatch))
    monkeypatch.setenv("PTBP_ADVISOR", "0")                       # the same stage, no advisor
    dftb_stage.write_outputs(*_stage(tmp_path / "off", dataset, ncores, monkeypatch))
    assert digests(tmp_path) == digests(tmp_path / "off")        # the advisor only reads
    assert not (tmp_path / "off" / "advisor.json").exists()
    checks = _checks(tmp_path)
    assert len(checks) == 1
    rules = {f["rule"]: f for f in checks[0]["findings"]}
    assert rules["P01"]["expected"] and rules["P03"]["expected"]
    assert "T01" not in rules and "T03" not in rules            # the plan sets 1 thread
    assert not (tmp_path / dftb_stage.TASK_DIR).exists()        # kept directory removed


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DFTBPLUS or not sys.platform.startswith("linux"),
                    reason="dftb+ binary and fork backend needed")
def test_failed_dftbplus_run_carries_the_advice(tmp_path, dataset, monkeypatch):
    from ase.calculators.calculator import CalculationFailed
    images, bolt = dataset
    bad = [a.copy() for a in images]
    bad[1].positions[1] = bad[1].positions[0]                   # two atoms on one site
    with pytest.raises(CalculationFailed) as info:
        _stage(tmp_path, (bad, bolt), 4, monkeypatch)
    assert any(note.startswith("[advisor] DFTB+ advisor:") for note in info.value.__notes__)
    assert [c["when"] for c in _checks(tmp_path)][-1] == "failed DFTB+ run"
