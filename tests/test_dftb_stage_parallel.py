"""DFTB stage of a BO step (utils.dftb_stage): identical files for any core budget,
the historical group semantics, DFTB+ task directories."""
import gzip
import hashlib
import shutil
import sys
from pathlib import Path

import ase.io
import pytest

import utils.data as data
import utils.engines as eng
from utils import dftb_stage
from utils.parallel import CpuBudget, set_plan, simple_plan

ROOT = Path(__file__).resolve().parents[1]
CU_SKF_GZ = ROOT / "pydftb" / "tests" / "data" / "skf" / "ptbp_hcocu" / "Cu-Cu.skf.gz"
HAVE_DFTBPLUS = shutil.which("dftb+") is not None
FILES = ("dft_final.xyz", "dftb_band.xyz", "search_index.txt")
KDENS = 5.0                     # PTBP's default (coarser grids at T = 0 may not converge)
# three EOS "groups" of three: Gamma-only structures (Boltzmann weight < 0.1) and the
# cheapest structures with a k-point grid (14 x 14 x 14)
PICK = [0, 1, 2, 5, 6, 7, 60, 63, 65]
EOS = 3
pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork backend: Linux")


@pytest.fixture(autouse=True)
def _restore():
    saved = eng.settings()
    yield
    eng._SETTINGS.clear()
    eng._SETTINGS.update(saved)
    set_plan(None)


@pytest.fixture(scope="module")
def dataset():
    images = ase.io.read(ROOT / "dft" / "dft.xyz", ":")
    _, bolt = data.get_bolt_factor(images[5::11], "boltzmann", 3000)   # over all phases, as PTBP
    return [images[i] for i in PICK], [bolt[i] for i in PICK]


def _skf_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    with gzip.open(CU_SKF_GZ, "rb") as fi, open(path / "Cu-Cu.skf", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    return path


def _stage(workdir, dataset, ncores, engine, monkeypatch, eos=EOS):
    """Run the stage in ``workdir`` with a plan for ``ncores`` (None: no plan)."""
    images, bolt = dataset
    eng.configure(engine=engine)
    set_plan(None if ncores is None else
             simple_plan(CpuBudget(ncores, {"--ncores": ncores}), len(images), engine))
    monkeypatch.chdir(_skf_dir(workdir))
    out = dftb_stage.run_dataset(images, bolt, eos, KDENS)
    dftb_stage.write_outputs(*out)
    set_plan(None)
    return out, {f: hashlib.sha256((workdir / f).read_bytes()).hexdigest() for f in FILES}


def test_irreducible_kpoint_count():
    from pydftb.kpoints import monkhorst_pack, reduce_time_reversal
    for grid in [(1, 1, 1), (2, 2, 2), (3, 3, 3), (2, 3, 4), (15, 15, 15), (14, 14, 14), (1, 2, 2),
                 (5, 4, 1), (7, 1, 1)]:
        assert dftb_stage.irreducible_kpoints(grid) == len(reduce_time_reversal(*monkhorst_pack(grid))[0])
    assert dftb_stage.irreducible_kpoints([]) == 1         # no periodic direction


def test_files_are_identical_for_any_core_budget(tmp_path, dataset, monkeypatch):
    digests = [_stage(tmp_path / str(n), dataset, n, "pydftb", monkeypatch)[1]
               for n in (None, 1, 2, 4)]
    assert all(d == digests[0] for d in digests[1:])
    (_, _, kept), _ = _stage(tmp_path / "again", dataset, 3, "pydftb", monkeypatch)
    assert kept == list(range(9))


def test_torch_backend_files_are_identical_for_any_core_budget(tmp_path, dataset, monkeypatch):
    """M4d: pydftb on torch (CPU): serial, 4 processes (torch on one thread each, set before
    the fork) and the GPU path (this process, one structure after the other; simulated)
    write the same files."""
    pytest.importorskip("torch")
    eng.configure(engine="pydftb", pydftb_backend="torch")
    serial = _stage(tmp_path / "serial", dataset, None, "pydftb", monkeypatch)[1]
    assert _stage(tmp_path / "n4", dataset, 4, "pydftb", monkeypatch)[1] == serial
    monkeypatch.setattr(eng, "is_gpu", lambda: True)
    assert _stage(tmp_path / "gpu", dataset, 4, "pydftb", monkeypatch)[1] == serial


def test_pydftb_k_threads_when_cores_outnumber_structures(tmp_path, dataset, monkeypatch):
    images, bolt = dataset
    one = (images[6:7], bolt[6:7])                              # 1372 k-points
    ref = _stage(tmp_path / "serial", one, None, "pydftb", monkeypatch, eos=1)[1]
    assert _stage(tmp_path / "threads", one, 4, "pydftb", monkeypatch, eos=1)[1] == ref


def _inject(monkeypatch, failures):
    """Make run_structure raise for some structure indices (inherited by forked workers)."""
    real = dftb_stage.run_structure

    def run(task):
        if task.index in failures:
            raise failures[task.index]
        return real(task)
    monkeypatch.setattr(dftb_stage, "run_structure", run)


@pytest.mark.parametrize("ncores", [None, 4])
def test_missing_skf_drops_its_group(tmp_path, dataset, monkeypatch, capsys, ncores):
    _inject(monkeypatch, {4: FileNotFoundError(2, "No such file or directory", "./Cu-Cu.skf")})
    (dftb, dft, kept), _ = _stage(tmp_path, dataset, ncores, "pydftb", monkeypatch)
    assert kept == [0, 1, 2, 6, 7, 8]
    assert len(dftb) == len(dft) == 6
    assert dft[3] is dataset[0][6]
    out = capsys.readouterr().out
    assert out.count("DFTB+ Warning: [Errno 2] No such file or directory: './Cu-Cu.skf'") == 1
    assert dftb_stage.SKF_WARNING in out


@pytest.mark.parametrize("ncores", [None, 4])
def test_other_errors_are_raised(tmp_path, dataset, monkeypatch, ncores):
    from ase.calculators.calculator import CalculationFailed
    # the first failure of the first failing group decides, as in the serial loop
    _inject(monkeypatch, {1: FileNotFoundError(2, "missing", "x.skf"),
                          5: CalculationFailed("dftb+ crashed"), 7: ValueError("later group")})
    with pytest.raises(CalculationFailed, match="dftb\\+ crashed"):
        _stage(tmp_path, dataset, ncores, "pydftb", monkeypatch)


@pytest.mark.parametrize("gradient", [False, True])
def test_a_failed_pydftb_run_is_a_calculation_failure(tmp_path, dataset, monkeypatch, gradient):
    """pydftb fails as DFTB+ does through ASE (CalculationFailed, a RuntimeError: the loss
    scores the step as failed), also in pass 1 of the gradient."""
    import numpy as np
    from ase.calculators.calculator import CalculationFailed

    from pydftb.backend import ops
    grad = None
    if gradient:
        pytest.importorskip("torch")
        from utils.hotcent_tangent import Tangent
        grad = dftb_stage.GradPass({"r0_d": Tangent("r0_d", 0.0, zero=True)})

    def not_positive_definite(a):
        raise np.linalg.LinAlgError("Matrix is not positive definite")
    monkeypatch.setattr(ops, "cholesky", not_positive_definite)
    images, bolt = dataset
    eng.configure(engine="pydftb")
    monkeypatch.chdir(_skf_dir(tmp_path))
    with pytest.raises(CalculationFailed, match="not positive definite"):
        dftb_stage.run_dataset(images[:EOS], bolt[:EOS], EOS, KDENS, grad=grad)


def test_remainder_that_does_not_fill_a_group_is_not_computed(tmp_path, dataset, monkeypatch):
    (dftb, _, kept), _ = _stage(tmp_path, dataset, 4, "pydftb", monkeypatch, eos=4)
    assert kept == list(range(8)) and len(dftb) == 8


def test_loss_writes_the_stage_files(tmp_path, dataset, monkeypatch):
    import argparse

    from utils.loss import Loss
    images, bolt = dataset
    eng.configure(engine="pydftb")
    monkeypatch.chdir(_skf_dir(tmp_path))
    loss = Loss(args_input=argparse.Namespace(eos_points=EOS, kpt_density=KDENS),
                dft_input=images, bolt_f11=bolt)
    dftb, dft = loss.run_calculator_seperately(images)
    assert len(dftb) == len(dft) == 9
    assert (tmp_path / "search_index.txt").read_text() == ",".join(map(str, range(9)))
    assert len(ase.io.read(tmp_path / "dftb_band.xyz", ":")) == 9


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DFTBPLUS, reason="dftb+ binary not available")
def test_dftbplus_task_directories(tmp_path, dataset, monkeypatch):
    serial = _stage(tmp_path / "n1", dataset, 1, "dftbplus", monkeypatch)[1]
    assert (tmp_path / "n1" / "dftb_in.hsd").exists()           # serial: current directory
    parallel = _stage(tmp_path / "n4", dataset, 4, "dftbplus", monkeypatch)[1]
    assert parallel == serial                                   # one OpenMP thread in both
    assert not (tmp_path / "n4" / dftb_stage.TASK_DIR).exists()  # removed after success
    assert not (tmp_path / "n4" / "dftb_in.hsd").exists()


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DFTBPLUS, reason="dftb+ binary not available")
def test_failed_dftbplus_task_directory_is_kept(tmp_path, dataset, monkeypatch):
    from ase.calculators.calculator import CalculationFailed
    images, bolt = dataset
    bad = [a.copy() for a in images]
    bad[4].positions[1] = bad[4].positions[0]                   # two atoms on top of each other
    with pytest.raises(CalculationFailed):
        _stage(tmp_path, (bad, bolt), 4, "dftbplus", monkeypatch)
    kept = sorted(p.name for p in (tmp_path / dftb_stage.TASK_DIR).iterdir())
    assert kept == ["s0004"] and (tmp_path / dftb_stage.TASK_DIR / "s0004" / "dftb_in.hsd").exists()


def test_gradient_passes_change_no_file_and_no_bit(tmp_path, dataset, monkeypatch):
    """M4c: the stage with pass 1 of the gradient writes the same files as without it;
    Edot and the pass-2 force contractions are bitwise the same for any core budget."""
    pytest.importorskip("torch")
    import numpy as np

    from pydftb.skf import SKFile
    from utils.hotcent_tangent import Tangent
    images, bolt = dataset
    shape = SKFile.read(str(_skf_dir(tmp_path / "shape") / "Cu-Cu.skf"), homo=True).table.shape
    tan = {"r0_w": Tangent("r0_w", 0.04, tables={
        ("Cu", "Cu"): np.random.default_rng(0).normal(size=shape) * 1e-3}),
        "r0_d": Tangent("r0_d", 0.0, zero=True)}
    eng.configure(engine="pydftb")
    plain = _stage(tmp_path / "plain", dataset, 2, "pydftb", monkeypatch)[1]
    out = []
    for n in (None, 1, 4):
        wd = _skf_dir(tmp_path / f"g{n}")
        monkeypatch.chdir(wd)
        set_plan(None if n is None else simple_plan(CpuBudget(n, {"--ncores": n}), len(images),
                                                    "pydftb"))
        grad = dftb_stage.GradPass(tan)
        res = dftb_stage.run_dataset(images, bolt, EOS, KDENS, grad=grad)
        dftb_stage.write_outputs(*res)
        files = {f: hashlib.sha256((wd / f).read_bytes()).hexdigest() for f in FILES}
        grad.weights = {i: np.random.default_rng(i).choice([-1.0, 1.0], size=(len(images[i]), 3))
                        for i in res[2]}
        grad.q_init = grad.q_in
        fdot = dftb_stage.run_force_gradient(images, bolt, KDENS, grad, str(wd))
        set_plan(None)
        assert files == plain
        out.append((dict(grad.out), fdot))
    assert all(o == out[0] for o in out[1:])
    assert all(v["r0_d"] == 0.0 and np.isfinite(v["r0_w"]) for v in out[0][0].values())


def test_gradient_pass_without_convergence_keeps_the_results(tmp_path, dataset, monkeypatch):
    """An SCC that does not converge has no gradient: the stage keeps the results of that
    same run (no second SCC), writes the files of a plain stage and records why."""
    pytest.importorskip("torch")
    import numpy as np

    import pydftb.calculator as pc
    from pydftb.skf import SKFile
    from utils.hotcent_tangent import Tangent
    images, bolt = dataset
    orig = eng._pydftb_calculator

    def few_iterations(mole, args):
        calc = orig(mole, args)
        calc.engine_opts["max_scc"] = 2
        return calc
    monkeypatch.setattr(eng, "_pydftb_calculator", few_iterations)
    eng.configure(engine="pydftb")
    plain = _stage(tmp_path / "plain", dataset, None, "pydftb", monkeypatch)[1]
    runs = []
    orig_run = pc.DFTB._run

    def counted(self, *a, **k):
        runs.append(1)
        return orig_run(self, *a, **k)
    monkeypatch.setattr(pc.DFTB, "_run", counted)
    shape = SKFile.read(str(_skf_dir(tmp_path / "shape") / "Cu-Cu.skf"), homo=True).table.shape
    grad = dftb_stage.GradPass({"r0_w": Tangent("r0_w", 0.04, tables={("Cu", "Cu"): np.ones(shape)})})
    wd = _skf_dir(tmp_path / "grad")
    monkeypatch.chdir(wd)
    out = dftb_stage.run_dataset(images, bolt, EOS, KDENS, grad=grad)
    dftb_stage.write_outputs(*out)
    assert {f: hashlib.sha256((wd / f).read_bytes()).hexdigest() for f in FILES} == plain
    assert len(runs) == len(images)                      # one SCC per structure
    assert grad.errors and all("SCC did not converge" in e for e in grad.errors.values())
    assert all(grad.out[i] is None for i in grad.errors)
