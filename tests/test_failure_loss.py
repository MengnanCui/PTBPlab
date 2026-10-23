"""Failed steps (utils.loss): in the dataset and reaction modes a failed step's loss is the
next float above the largest loss of a successful step so far (1e6 before the first one);
energygeometry keeps 97-99.  The loss marks failures (Loss.failed); BO never takes one for
its optimum, a resumed checkpoint converts the 97-99 written before, batched BO ranks
failures after their round, and tools/bo_vs_grad.py counts the marked ones (M4c report,
section 7: with 97-99, BO took failed steps for the optimum in the dataset mode)."""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms

import utils.data as data
import utils.engines as eng
import utils.loss as L
from utils.parallel import set_plan

ROOT = Path(__file__).resolve().parents[1]
CU_SKF_GZ = ROOT / "pydftb" / "tests" / "data" / "skf" / "ptbp_hcocu" / "Cu-Cu.skf.gz"
R_CU = 2.4944384861047455                     # covalent radius of Cu, Bohr
linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork backend")


@pytest.fixture(autouse=True)
def _clean_state():
    saved = eng.settings()
    set_plan(None)
    yield
    set_plan(None)
    eng._SETTINGS.clear()
    eng._SETTINGS.update(saved)


def _args(tmp_path, mode, **kw):
    return argparse.Namespace(
        optimization_option=mode, eos_points=11 if mode == "energygeometry" else 1,
        kpt_density=5.0, multi_element=None, xc="GGA_X_PBE+GGA_C_PBE",
        superposition="density", results_dir="results", para_file=str(tmp_path / "par.out"),
        log_file=str(tmp_path / "log.out"), checkpoint=str(tmp_path / "checkpoint.pkl"), **kw)


def _cu():
    return Atoms("Cu", cell=np.eye(3) * 3.0, pbc=True)


def test_failure_values_per_mode():
    loss = L.Loss(args_input=argparse.Namespace(optimization_option="reaction"),
                  dft_input=[_cu()])
    assert not loss.failed and loss.worst_loss is None
    assert loss.failure_loss() == L.FAILURE_FALLBACK              # no successful step yet
    for mu in (446.2, 211.3, float("nan")):
        loss.note_loss(mu)
    assert loss.worst_loss == 446.2 and not loss.failed           # NaN does not count
    assert loss.failure_loss() == np.nextafter(446.2, np.inf) > 446.2
    loss.args_input.optimization_option = "energygeometry"
    assert loss.failure_loss() in (97, 98, 99)


def _hotcent(monkeypatch, fail):
    """hotcent replaced by a stored Cu SKF, or failing (no SKF) while fail['now']."""
    def full_hotcent(self, superposition, opt_elem=(), label="ptbp"):
        if fail["now"]:
            raise RuntimeError("hotcent wrote no SKF")
        with gzip.open(CU_SKF_GZ, "rb") as fi, open("Cu-Cu.skf", "wb") as fo:
            shutil.copyfileobj(fi, fo)
    monkeypatch.setattr(L.hotpy, "full_hotcent", full_hotcent)


def _last(path):
    return [float(v) for v in Path(path).read_text().splitlines()[-1].split(",")]


def test_dataset_steps_rank_failures_last(tmp_path, monkeypatch):
    """PTBP's dataset-mode step (pydftb, two Cu dimers; a stored SKF instead of hotcent):
    a failure before any success gets 1e6, a later one the next float above the largest
    successful loss, and the loss marks each."""
    monkeypatch.chdir(tmp_path)
    eng.configure(engine="pydftb")
    fail = {"now": True}
    _hotcent(monkeypatch, fail)
    dimers = ase.io.read(ROOT / "dft" / "dft.xyz", ":2")
    loss = L.Loss(args_input=_args(tmp_path, "dataset",
                                   E0s=data.load_E0s_cache(str(ROOT / "dft" / "E0s_dft.json"))),
                  dft_input=dimers, bolt_f11=[1.0, 1.0], opt_elem=[], cov_radius=R_CU,
                  path=tmp_path)
    mus = []
    for x, fails in (([2.749, 12.036], True), ([2.749, 12.036], False),
                     ([2.8, 12.0], True), ([3.0, 11.0], False), ([3.2, 11.0], True)):
        fail["now"] = fails
        mus.append(float(loss.Loss_EnergyGeometry(x)))
        assert loss.failed == fails
        assert _last(tmp_path / "par.out")[-1] == pytest.approx(mus[-1], rel=1e-11)
        if fails:                           # (a successful step logs every sigma it tried)
            assert _last(tmp_path / "log.out")[-1] == pytest.approx(mus[-1], rel=1e-11)
    assert mus[0] == L.FAILURE_FALLBACK
    assert np.isfinite(mus[1]) and mus[1] != L.FAILURE_FALLBACK
    assert mus[2] == np.nextafter(mus[1], np.inf)
    assert mus[4] == np.nextafter(max(mus[1], mus[3]), np.inf)
    assert loss.worst_loss == max(mus[1], mus[3])


def test_energygeometry_keeps_97_to_99(tmp_path, monkeypatch):
    """The same single random.randint(97, 99) draw and the same par.out/log.out lines."""
    monkeypatch.chdir(tmp_path)
    _hotcent(monkeypatch, {"now": True})
    draws = []
    monkeypatch.setattr(L.random, "randint", lambda a, b: draws.append((a, b)) or 98)
    loss = L.Loss(args_input=_args(tmp_path, "energygeometry"), dft_input=[_cu()],
                  bolt_f11=[1.0], opt_elem=[], path=tmp_path)
    loss.note_loss(1.9)                           # successes do not change the failure value
    assert loss.Loss_EnergyGeometry([2.749, 12.036]) == 98 and loss.failed
    assert draws == [(97, 99)]
    w, d, s, k, r, mu = 2.749, 12.036, 0.4321, 0.0, loss.cov_radius, 98
    assert (tmp_path / "par.out").read_text() == (
        f'{w:15.8f}, {d:15.8f}, {s:15.8f}, {k:20.8e}, {r:15.7f}, {mu:25.11f}\n')
    assert (tmp_path / "log.out").read_text() == (
        f'{w:15.8f}, {d:15.8f}, {s:15.8f}, {k:20.8e}, {r:15.8f}, {10:15.8f}, {10:20.8f}, '
        f'{10: 20.8f}, {10:15.5f}, {mu:25.11f}\n')


def bowl(x):
    """Dataset-mode losses of 400 and more (Cu's are about 210-7000: above 97-99)."""
    return 400.0 + 80.0 * (x[0] - 3.1) ** 2 + 0.4 * (x[1] - 9.98) ** 2


class FailingStep(L.Loss):
    """Loss_EnergyGeometry with EnergyGeometry's bookkeeping on bowl(), failing beyond
    r0_w = 12 Bohr (Cu's overlap matrix is not positive definite beyond about 15 Bohr)."""

    def Loss_EnergyGeometry(self, parameters):
        self.failed = True
        if parameters[0] > 12.0:
            return self.failure_loss()
        mu = bowl(parameters)
        self.note_loss(mu)
        return mu


def _stand_in(tmp_path, mode="dataset", **kw):
    return FailingStep(args_input=_args(tmp_path, mode, **kw), dft_input=[_cu()],
                       bolt_f11=[1.0], opt_elem=[], cov_radius=R_CU, path=tmp_path)


def _saver(tmp_path):
    from skopt.callbacks import CheckpointSaver
    return CheckpointSaver(str(tmp_path / "checkpoint.pkl"), compress=9, store_objective=False)


def _ranked_after_successes(func_vals, failed):
    """Every failure is the next float above the successes before it (1e6 with none)."""
    for i, bad in enumerate(failed):
        real = [y for y, b in zip(func_vals[:i], failed[:i]) if not b]
        if bad:
            assert func_vals[i] == (np.nextafter(max(real), np.inf) if real
                                    else L.FAILURE_FALLBACK), i


def test_bo_never_takes_a_failure_for_its_optimum(tmp_path):
    """Loss.Bayesian_optimization in the dataset mode, from the PTBP seed of Cu."""
    from skopt import load
    loss = _stand_in(tmp_path)
    res = loss.Bayesian_optimization(R_CU, _saver(tmp_path), 14)
    failed = [x[0] > 12.0 for x in res.x_iters]
    assert res.failed == failed and any(failed) and not all(failed)
    real = [y for y, bad in zip(res.func_vals, failed) if not bad]
    assert min(real) > 99.0               # with 97-99 every failure would rank first
    assert not failed[int(np.argmin(res.func_vals))] and res.fun == min(real)
    _ranked_after_successes(res.func_vals, failed)
    saved = load(tmp_path / "checkpoint.pkl")          # the checkpoint format is unchanged
    assert list(saved.func_vals) == list(res.func_vals) and "failed" not in saved


def test_resume_converts_failures_stored_as_97_to_99(tmp_path, capsys):
    """A dataset-mode checkpoint written before (failures at 98, BO's minimum one of them):
    resumed, they are marked and ranked after its successful steps."""
    from skopt import gp_minimize
    loss = _stand_in(tmp_path)
    dims = loss._bo_dimensions_and_cost(R_CU)[0]
    old = gp_minimize(lambda x: 98.0 if x[0] > 12.0 else bowl(x), dims, n_calls=11,
                      x0=[2.749, 12.036], random_state=7, callback=[_saver(tmp_path)])
    stored = [y == 98.0 for y in old.func_vals]
    assert stored[int(np.argmin(old.func_vals))]          # the old optimum: a failure
    Path(f"{tmp_path / 'checkpoint.pkl'}.meta.json").write_text(json.dumps(
        {"ptbp_loss_version": L.LOSS_VERSION, "penalty_radius_bohr": R_CU, "mode": "dataset"}))
    res = loss.Bayesian_optimization(R_CU, _saver(tmp_path), 10)
    n0 = len(old.func_vals)
    assert res.failed[:n0] == stored and "now count as" in capsys.readouterr().out
    worst = max(y for y, bad in zip(old.func_vals, stored) if not bad)
    assert all(res.func_vals[i] == np.nextafter(worst, np.inf) for i in range(n0) if stored[i])
    assert res.failed[n0:] == [x[0] > 12.0 for x in res.x_iters[n0:]]
    assert not res.failed[int(np.argmin(res.func_vals))]


def test_resumed_losses():
    loss = _stand_in(Path("."))
    y0 = np.array([400.0, 98.0, L.FAILURE_FALLBACK, 500.0])
    same, failed = loss._resumed_losses(y0, "energygeometry")   # still its failure value
    assert same is y0 and failed == [False, True, False, False] and loss.worst_loss == 500.0
    y1, failed = loss._resumed_losses(y0, "reaction")
    assert y1 == [400.0, np.nextafter(500.0, np.inf), L.FAILURE_FALLBACK, 500.0]
    assert failed == [False, True, False, False]


@linux
def test_parallel_bo_ranks_failures_after_their_round(tmp_path):
    """Forked workers know the earlier rounds only; the GP gets each failure ranked after
    the successes of its own round too."""
    loss = _stand_in(tmp_path, optimizer="parallel_bo", n_particles=4)
    res = loss._optimize_parallel_bo(R_CU, None, 12)
    failed = [x[0] > 12.0 for x in res.x_iters]
    assert res.failed == failed and any(failed) and not all(failed)
    for r in range(3):
        real = [y for y, bad in zip(res.func_vals[:4 * r + 4], failed) if not bad]
        for i in range(4 * r, 4 * r + 4):
            if failed[i]:
                assert res.func_vals[i] == (np.nextafter(max(real), np.inf) if real
                                            else L.FAILURE_FALLBACK)
    assert res.fun == min(y for y, bad in zip(res.func_vals, failed) if not bad)


def test_bo_vs_grad_counts_marked_failures():
    from tools.bo_vs_grad import best_so_far, failures, summary
    f = float(np.nextafter(446.2, np.inf))                # a failure, the worst so far
    trace = [[1.0, [2.749, 12.036], 446.2], [2.0, [17.1, 12.0], f], [3.0, [4.2, 11.0], 211.3]]
    run = {"trace": trace, "failed": [1]}
    assert summary(trace, failures(run)) == (211.3, [4.2, 11.0], 1)
    assert [b for _, b in best_so_far(trace, failures(run))] == [446.2, 446.2, 211.3]
    # a JSON file written before failures were marked: the losses 97-99 are the failures
    old = {"trace": [[1.0, [2.749, 12.036], 446.2], [2.0, [17.1, 12.0], 97], trace[2]]}
    assert failures(old) == {1}
    assert summary(old["trace"], failures(old)) == (211.3, [4.2, 11.0], 1)
