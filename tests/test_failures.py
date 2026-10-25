"""Follow-ups of M5b (milestone M5b-2): failed evaluations count twice the worst successful
loss (structures without DFTB results included), their record in the checkpoint's meta file and
its use on resume, the optimisers that see failures (parallel BO, the gradient optimisers), the
warning for several types with the energygeometry loss, the default --n-calls by the number of
optimised dimensions, and the documentation of the per-type penalty and --multi_element."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

import utils.engines as eng
import utils.loss as L
from tests.test_types import SKF, SWITCH, _box, _copy_skf, _ho_args
from utils import gradient_opt as GO
from utils import typemodel as TM

ROOT = Path(__file__).resolve().parents[1]
linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork backend")


@pytest.fixture(autouse=True)
def _restore_engine():
    saved = eng.settings()
    switches = eng.type_switching()
    yield
    eng._SETTINGS.clear()
    eng._SETTINGS.update(saved)
    eng.set_type_switching(switches)


def _loss(mode="energygeometry", n=3, eos=1):
    args = argparse.Namespace(optimization_option=mode, eos_points=eos)
    return L.Loss(args_input=args, dft_input=[_box(["Cu"], [[0, 0, 0]]) for _ in range(n)],
                  cov_radius=2.49)


# ---- the failure value -------------------------------------------------------------------
def test_failure_value_is_twice_the_worst_success(capsys):
    loss = _loss("energygeometry")
    assert loss.failure_value() == L.NO_SUCCESS_FAILURE == 1000.0       # no success yet
    assert _loss("reaction", n=22).failure_value() == 22000.0           # 1000 per structure
    assert _loss("dataset", n=23, eos=11).failure_value() == 22000.0    # the structures computed
    for mu in (3.0, 7.5, 2.0):
        loss._record_success(mu)
    assert loss.worst_success == 7.5 and loss.last_failed is False
    v = loss._record_failure([4.0, 9.0], RuntimeError("hotcent failed"))
    assert v == 15.0 == loss.failure_value() and loss.last_failed is True
    assert loss.is_failure([4.0, 9.0]) and not loss.is_failure([4.0, 9.5])
    assert loss.worst_success == 7.5                                     # failures do not count
    assert "2 x the worst successful loss so far (7.5)" in capsys.readouterr().out


def test_structures_without_results_are_a_failure(tmp_path, monkeypatch):
    """The DFTB stage drops the EOS groups whose SKF files are missing; any structure without a
    result makes the point a failure (LossFailure, a RuntimeError: the failure branches catch it).
    The structures past the last whole EOS group are never computed: not missing."""
    monkeypatch.chdir(tmp_path)
    loss = _loss("energygeometry", n=23, eos=11)
    index = tmp_path / "search_index.txt"
    index.write_text(",".join(map(str, range(22))))
    loss._check_complete_results()
    index.write_text(",".join(map(str, range(11, 22))))
    with pytest.raises(L.LossFailure, match="11 of 22 structures have no DFTB results.*Cu x11"):
        loss._check_complete_results()
    index.write_text("")
    with pytest.raises(L.LossFailure, match="no structure has DFTB results"):
        loss._check_complete_results()
    assert issubclass(L.LossFailure, RuntimeError)


def _hco_dataset():
    """H2O, OH, CH4 and CH in boxes with made-up energies and forces: every structure has two
    elements (before M5b-3 the code of before M5b, Loss.get_elements_from_dataset, needed element
    lists of one length: NumPy refused the ragged array; tests/test_shared_parameters.py)."""
    rng = np.random.default_rng(7)
    base = [(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
            (["O", "H"], [[0, 0, 0], [0.98, 0, 0]]),
            (["C", "H", "H", "H", "H"], [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                                         [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]]),
            (["C", "H"], [[0, 0, 0], [1.12, 0, 0]])]
    out = []
    for sym, pos in base:
        pos = np.array(pos, float) + rng.normal(scale=0.03, size=(len(sym), 3)) + 4.0
        a = _box(sym, pos, a=8.0)
        a.info["energy"] = (-7.0 * sym.count("H") - 430.0 * sym.count("O") - 150.0 * sym.count("C")
                            + rng.normal(0.5))
        a.set_array("forces", rng.normal(scale=0.5, size=(len(sym), 3)))
        out.append(a)
    return out


@linux
def test_shared_layout_failures_count_twice_the_worst_success(tmp_path, monkeypatch):
    """The code of before M5b (one parameter set for H, C and O, dataset mode, pydftb, hotcent
    replaced by stored files): a step with structures missing (no C tables: CH4 and CH fail)
    fails with the fixed value (no success yet, 1000 per structure), a full step succeeds, the
    next partial step and a step whose loss is not finite count twice that success; par.out
    keeps its format (sigma_rep 0.4321 on a failure)."""
    eng.configure(engine="pydftb")
    monkeypatch.chdir(tmp_path)
    tables = {"names": {"H", "O"}}
    monkeypatch.setattr(L.hotpy, "full_hotcent",
                        lambda self, superposition, opt_elem=(), label="ptbp":
                        _copy_skf(Path("."), SKF / "ptbp_hcocu", tables["names"]))
    dft = _hco_dataset()
    args = _ho_args(tmp_path)
    args.E0s = {1: -13.6, 6: -150.0, 8: -430.0}
    loss = L.Loss(args_input=args, dft_input=dft, bolt_f11=[0.05] * len(dft), opt_elem=[],
                  path=tmp_path)
    assert not loss.typed
    mu1 = float(loss.Loss_EnergyGeometry([2.2, 4.0]))
    assert mu1 == 4000.0 and loss.is_failure([2.2, 4.0])
    assert (tmp_path / "results" / "opt_2.200_4.000" / "search_index.txt").read_text() == "0,1"
    tables["names"] = {"H", "C", "O"}
    mu2 = float(loss.Loss_EnergyGeometry([2.5, 4.5]))
    assert loss.last_failed is False and loss.worst_success == mu2 < 4000.0
    tables["names"] = {"H", "O"}
    mu3 = float(loss.Loss_EnergyGeometry([2.3, 4.1]))
    assert mu3 == 2 * mu2 and loss.worst_success == mu2
    tables["names"] = {"H", "C", "O"}
    monkeypatch.setattr(L.Loss, "Repulsion_optimization",
                        lambda self, conf, c0, c1: (0.5, float("nan")))
    mu4 = float(loss.Loss_EnergyGeometry([2.4, 4.2]))
    assert mu4 == 2 * mu2 and loss.last_failed is True
    rows = [[float(v) for v in line.split(",")]
            for line in (tmp_path / "par.out").read_text().splitlines()]
    assert [r[2] == 0.4321 for r in rows] == [True, False, True, True]
    assert [r[-1] for r in rows] == pytest.approx([4000.0, mu2, 2 * mu2, 2 * mu2], rel=1e-10)
    assert len(loss.failed_points) == 3


# ---- the record of the failures and resuming ---------------------------------------------
class Synthetic(L.Loss):
    """Loss_EnergyGeometry in closed form (one Cu parameter set), recording its outcomes as
    EnergyGeometry does, and like it changing into the evaluation's directory: the PTBP seed is
    the worst success (50), r0_w > 9 fails."""

    def Loss_EnergyGeometry(self, x):
        x = [float(v) for v in x]
        step = Path(self.path) / "results" / ("opt_" + "_".join(f"{v:.3f}" for v in x))
        step.mkdir(parents=True, exist_ok=True)
        os.chdir(step)
        if x[0] > 9.0:
            return self._record_failure(x, RuntimeError("no SKF files"))
        mu = 50.0 if abs(x[0] - 2.749) < 1e-9 else 1.0 + 0.01 * (x[0] ** 2 + x[1] ** 2)
        self._record_success(mu)
        return mu


def _bo_loss(tmp_path, cls=Synthetic, checkpoint="checkpoint.pkl", **extra):
    """The Loss as cli/run.py builds it: --checkpoint relative to the run folder (the cwd)."""
    args = argparse.Namespace(optimization_option="energygeometry", eos_points=11,
                              checkpoint=checkpoint, **extra)
    return cls(args_input=args, dft_input=[_box(["Cu"], [[0, 0, 0]], a=3.0)], path=tmp_path)


def test_bo_records_failures_and_resumes_the_worst_success(tmp_path, monkeypatch):
    """BO with a checkpoint: every failure counts 2 x the worst success before it, the meta file
    in the run folder lists the failed evaluations (the evaluations change into their step
    directories, --checkpoint is relative); a resumed run starts from the loaded successes (a
    new failure counts 2 x the seed's 50, not 1000 or twice a new success) and extends the record."""
    from skopt import load
    from skopt.callbacks import CheckpointSaver
    monkeypatch.chdir(tmp_path)
    ckpt, meta_path = tmp_path / "checkpoint.pkl", tmp_path / "checkpoint.pkl.meta.json"
    saver = CheckpointSaver(str(ckpt), compress=9, store_objective=False)    # as cli/run.py
    loss = _bo_loss(tmp_path)
    res = loss.Bayesian_optimization(loss.cov_radius, saver, 11)        # seed + 10 random points
    vals = [float(v) for v in res.func_vals]
    failed = [i for i, x in enumerate(res.x_iters) if x[0] > 9.0]
    assert failed and 0 not in failed and loss.failed_indices == failed
    assert all(vals[i] == 100.0 for i in failed)                        # 2 x the seed's 50
    assert all(v < 50.0 for i, v in enumerate(vals) if i and i not in failed)
    assert json.loads(meta_path.read_text()) == {
        "ptbp_loss_version": L.SHARED_LOSS_VERSION, "penalty_radius_bohr": loss.cov_radius,
        "mode": "energygeometry", "failed_evaluations": failed}
    assert not list(tmp_path.glob("results/*/*.meta.json"))           # not in a step directory
    assert [float(v) for v in load(str(ckpt)).func_vals] == vals

    monkeypatch.chdir(tmp_path)                       # a new process starts in the run folder
    resumed = _bo_loss(tmp_path)
    res2 = resumed.Bayesian_optimization(resumed.cov_radius, saver, 10)
    vals2 = [float(v) for v in res2.func_vals]
    assert vals2[:11] == vals and len(vals2) == 21
    new = [i for i, x in enumerate(res2.x_iters) if i >= 11 and x[0] > 9.0]
    assert new and all(vals2[i] == 100.0 for i in new)
    assert resumed.failed_indices == failed + new
    assert json.loads(meta_path.read_text())["failed_evaluations"] == failed + new
    assert resumed.worst_success == 50.0


def test_old_checkpoint_failures_are_recognised_by_their_values(tmp_path, monkeypatch):
    """A checkpoint written before M5b-2 has no record: its values 97, 98, 99 are the failures,
    they count 2 x the worst loaded success in the resumed BO, and the meta file records them; a
    checkpoint with a record keeps its values.  Without failures the meta file is as before."""
    from scipy.optimize import OptimizeResult
    from skopt import dump
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_gp_minimize(**kw):
        seen.update(kw)
        return SimpleNamespace(x_iters=kw["x0"], func_vals=kw["y0"], x=kw["x0"][0],
                               fun=min(kw["y0"]))

    monkeypatch.setattr(L, "gp_minimize", fake_gp_minimize)
    xs = [[2.749, 12.036], [12.0, 5.0], [4.0, 9.0], [15.0, 20.0]]
    meta_path = tmp_path / "checkpoint.pkl.meta.json"

    def resume(ys, meta_extra):
        dump(OptimizeResult(x_iters=xs, func_vals=np.array(ys), x=xs[0], fun=min(ys),
                            specs={"args": {}}), str(tmp_path / "checkpoint.pkl"))
        loss = _bo_loss(tmp_path)
        meta_path.write_text(json.dumps(dict({"ptbp_loss_version": L.SHARED_LOSS_VERSION,
                                              "penalty_radius_bohr": loss.cov_radius,
                                              "mode": "energygeometry"}, **meta_extra)))
        loss.Bayesian_optimization(loss.cov_radius, lambda res: None, 10)
        return loss, [float(v) for v in seen["y0"]], json.loads(meta_path.read_text())

    loss, y0, meta = resume([0.92, 97.0, 3.5, 99.0], {})                 # before M5b-2
    assert y0 == [0.92, 7.0, 3.5, 7.0] and loss.worst_success == 3.5
    assert meta["failed_evaluations"] == [1, 3] == loss.failed_indices
    assert loss.is_failure(xs[1]) and loss.is_failure(xs[3]) and not loss.is_failure(xs[2])
    loss, y0, meta = resume([0.92, 1.84, 3.5, 7.0], {"failed_evaluations": [1, 3]})
    assert y0 == [0.92, 1.84, 3.5, 7.0] and loss.worst_success == 3.5   # values kept
    assert meta["failed_evaluations"] == [1, 3]
    loss, y0, meta = resume([0.92, 2.4, 3.5, 1.7], {})                  # no failure
    assert y0 == [0.92, 2.4, 3.5, 1.7] and "failed_evaluations" not in meta
    assert loss.worst_success == 3.5 and loss.failed_indices == []


@linux
def test_parallel_bo_records_the_failures_of_its_workers(tmp_path, monkeypatch):
    """parallel_bo evaluates in forked workers: a worker's failure counts 2 x the worst success
    of the rounds before (the state it forked with); the parent records every outcome."""
    monkeypatch.chdir(tmp_path)
    loss = _bo_loss(tmp_path, n_particles=2)
    res = loss._optimize_parallel_bo(loss.cov_radius, None, 8)          # 4 rounds of 2 points
    worst, expected = None, []
    for r in range(0, len(res.x_iters), 2):                           # round by round
        oks = []
        for x in res.x_iters[r:r + 2]:
            if x[0] > 9.0:
                expected.append(1000.0 if worst is None else 2 * worst)
            else:
                expected.append(1.0 + 0.01 * (x[0] ** 2 + x[1] ** 2))
                oks.append(expected[-1])
        if oks:
            worst = max(oks + ([] if worst is None else [worst]))
    assert res.func_vals == pytest.approx(expected, rel=1e-12)
    assert sorted(loss.failed_points) == sorted(tuple(x) for x in res.x_iters if x[0] > 9.0)
    assert 0 < len(loss.failed_points) < 8 and loss.worst_success == worst


class GradStub:
    """Loss_EnergyGeometry with a gradient, recording failures as Loss does: r0_w > 4 fails
    with twice the worst success (a value the old markers 97-99 cannot tell)."""

    def __init__(self, base=400.0):
        self.grad_request, self.calls, self.dft_input = None, [], []
        self.base, self.last_failed, self.worst = base, None, 0.0

    def _bo_dimensions_and_cost(self, cov_radius):
        return None, [2.1, 3.4], [7.0 * cov_radius, 9.0 * cov_radius], None

    def mu(self, x):
        return self.base + 80.0 * (x[0] - 3.1) ** 2 + 0.4 * (x[1] - 9.98) ** 2

    def Loss_EnergyGeometry(self, x):
        from utils.loss_grad import GradResult
        self.calls.append(list(x))
        if x[0] > 4.0:
            self.last_failed = True
            return 2 * self.worst
        self.last_failed, mu = False, self.mu(x)
        self.worst = max(self.worst, mu)
        if self.grad_request is not None:
            self.grad_request.result = GradResult(
                x[0], x[1], mu, {"r0_w": 160.0 * (x[0] - 3.1), "r0_d": 0.8 * (x[1] - 9.98)},
                case="smooth")
        return mu


def test_gradient_optimisers_take_the_failures_from_the_loss(tmp_path, monkeypatch):
    """The Evaluator takes Loss.last_failed (a failure is not a valid loss whatever its value);
    bo_grad takes Loss.failed_indices: when BO's minimum is a failure (the fixed value before
    the first success, below the later successes) it starts from the lowest real loss."""
    monkeypatch.chdir(tmp_path)
    stub = GradStub()
    ev = GO.Evaluator(stub, log_path=str(tmp_path / "grad.out"))
    good, bad = ev([3.2, 10.0]), ev([5.0, 10.0])
    assert good.valid and good.failed is False
    assert bad.f == 2 * good.f and bad.failed is True and not bad.valid
    assert GO.lowest([bad]) is None and GO.lowest([bad, good]) is good
    assert GO.loss_failed(stub, bad.f) and not GO.loss_failed(SimpleNamespace(), 2 * good.f)

    stub = GradStub(base=1000.0)
    stub.failed_indices = [0]
    f1, f2 = stub.mu([3.3, 10.5]), stub.mu([3.5, 11.0])
    bo = SimpleNamespace(x=[5.5, 12.0], fun=1000.0, x_iters=[[5.5, 12.0], [3.3, 10.5], [3.5, 11.0]],
                         func_vals=[1000.0, f1, f2])               # BO's minimum is the failure
    stub.Bayesian_optimization = lambda r, saver, n: bo
    res = GO.optimize_bo_grad(stub, 2.4944, None, 3, maxiter=4)
    assert stub.calls[0] == [3.3, 10.5] and 1000.0 < res.fun < f1


# ---- several types with energygeometry -----------------------------------------------------
def test_energygeometry_warning_for_several_types():
    cu = [_box(["Cu", "Cu"], [[0, 0, 0], [2.5, 0, 0]])]
    model = TM.TypeModel.from_dataset(cu, declarations=[SWITCH])
    text = TM.energygeometry_warning(model, "EnergyGeometry")
    assert text.startswith("WARNING") and "(Cu: Cu2, Cu)" in text
    assert "within each EOS group" in text and "--mode reaction" in text and "Continuing" in text
    assert TM.energygeometry_warning(model, "reaction") is None
    assert TM.energygeometry_warning(model, "dataset") is None
    assert TM.energygeometry_warning(TM.TypeModel.from_dataset(cu), "energygeometry") is None


def test_optimize_warns_and_sets_the_default_n_calls(tmp_path):
    """ptbp optimize with two Cu types and energygeometry: the warning, then 4 dimensions and
    the default 30 evaluations (plan.json); an explicit --n-calls wins.  The runs stop at
    --postprocess-only (no result.pkl), after the plan."""
    def run(name, *extra):
        out = subprocess.run(
            [sys.executable, str(ROOT / "parameterize.py"), "optimize", str(ROOT / "dft" / "dft.xyz"),
             "--mode", "energygeometry", "--engine", "pydftb", "--ncores", "1",
             "--switch", "Cu", "Cu2,Cu", "11.0", "1.0", "--postprocess-only",
             "--output", str(tmp_path / name), *extra],
            cwd=tmp_path, capture_output=True, text=True, timeout=900)
        assert "no " in out.stderr and "result.pkl" in out.stderr, out.stderr[-2000:]
        plan = json.loads((tmp_path / name / "plan.json").read_text())
        return out.stdout, plan["estimates"]["evaluations"]

    stdout, evaluations = run("default")
    assert "WARNING: several types of one element (Cu: Cu2, Cu)" in stdout
    assert "[n_calls] 30 evaluations: the default for 4 optimised dimensions" in stdout
    assert evaluations == 30
    stdout, evaluations = run("explicit", "--n-calls", "12")
    assert "WARNING: several types" in stdout and "[n_calls]" not in stdout and evaluations == 12


# ---- the default --n-calls -----------------------------------------------------------------
def test_default_n_calls_follows_the_dimensions(tmp_path, capsys):
    """10 + 5 x dims above 2 dimensions, 11 up to 2; ptbp plan uses the same rule (per element
    by default, --shared-parameters: 2 dimensions) unless --n-calls is given."""
    assert [TM.default_n_calls(d) for d in (1, 2, 4, 6)] == [11, 11, 30, 40]
    from cli.main import dispatch
    dataset = tmp_path / "bnna.xyz"
    import ase.io
    ase.io.write(dataset, [_box(["Na", "N", "N", "N"], [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]]),
                           _box(["B", "B", "N", "N"], [[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0],
                                                       [1.5, 1.5, 0]])])

    def evaluations(*extra):
        assert dispatch(["plan", str(dataset), "--mode", "reaction", "--ncores", "1", "--json",
                         "--eos-points", "1", *extra]) == 0
        return json.loads(capsys.readouterr().out)["estimates"]["evaluations"]

    assert evaluations() == 40                                    # B, N, Na: 6 dimensions
    assert evaluations("--shared-parameters") == 11
    assert evaluations("--multi-element", "N") == 11              # one fitted set
    assert evaluations("--n-calls", "12") == 12


def test_n_calls_from_yaml_or_the_command_line_wins(tmp_path, monkeypatch):
    """ptbp optimize passes --n_calls to run.py only when one is given (command line or YAML),
    so that run.py applies the default otherwise; ptbp_run.yaml does not record a default; the
    TUI writes n_calls only when one is typed."""
    from unittest.mock import patch

    from cli.main import dispatch
    from cli.tui.state import SetupState, assemble_optimize_yaml
    dataset = tmp_path / "d.xyz"
    dataset.write_text("0\n\n")
    config = tmp_path / "c.yaml"
    config.write_text("n_calls: 17\n")
    monkeypatch.chdir(tmp_path)

    def argv(*extra):
        captured = {}
        with patch("cli.cmd_optimize.runpy.run_path",
                   side_effect=lambda p, run_name=None: captured.setdefault("argv", list(sys.argv))):
            assert dispatch(["optimize", str(dataset), "--output", str(tmp_path / "run"),
                             "--mode", "dataset", *extra]) == 0
        return captured["argv"]

    import yaml
    assert "--n_calls" not in argv()
    assert "n_calls" not in yaml.safe_load((tmp_path / "run" / "ptbp_run.yaml").read_text())
    a = argv("--config", str(config))
    assert a[a.index("--n_calls") + 1] == "17"
    a = argv("--config", str(config), "--n-calls", "5")
    assert a[a.index("--n_calls") + 1] == "5"
    state = SetupState()
    assert "n_calls" not in assemble_optimize_yaml(state)
    state.n_calls = 7
    assert assemble_optimize_yaml(state)["n_calls"] == 7


# ---- documentation of the M5b designs ------------------------------------------------------
def test_help_documents_the_penalty_and_multi_element(monkeypatch, capsys):
    """The geometric-mean penalty of the parameter sets and --multi_element ("fit only these
    elements, the others keep PTBP's published parameters") in the help and the YAML docs."""
    from cli import cmd_optimize, config
    from cli.arg_parser import build_default_arg_parser
    p = cmd_optimize.add_parser(argparse.ArgumentParser().add_subparsers())
    helps = {a.dest: " ".join((a.help or "").split()) for a in p._actions}
    assert "geometric mean" in helps["shared_parameters"] and "geometric mean" in p.description
    assert "Fit only these elements" in helps["multi_element"]
    assert "PTBP published" in helps["multi_element"]
    assert "10 + 5 x dims" in helps["n_calls"] and "energygeometry" in helps["switch"]
    doc = " ".join(config.__doc__.replace("#", " ").split())          # YAML comment lines
    assert "geometric mean" in doc and "PTBP published" in doc and "10 + 5 x dims" in doc
    monkeypatch.setattr(sys, "argv", ["run.py", "--help"])
    with pytest.raises(SystemExit):
        build_default_arg_parser()
    text = " ".join(capsys.readouterr().out.split())
    assert "Fit only these elements" in text and "geometric mean" in text
    assert "10 + 5 x dims" in text
