"""The computation strategy of a PTBP run (utils.recommender, milestone M3c)."""
import json
import random
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

import utils.engines as eng
from pydftb.perfmodel import CostModel, shape
from utils import dftb_stage
from utils import recommender as R
from utils.parallel import CpuBudget, TaskResult, set_plan
from utils.parallel.plan import Plan, StagePlan

ROOT = Path(__file__).resolve().parents[1]
MODEL = CostModel()                                   # calibration constants, no probe
GB = 2 ** 30
CU = R.ptbp_species("Cu")


def budget(n):
    return CpuBudget(n, {"--ncores": n})


def cu_cell(grid, reps=(1, 1, 1)):
    """Shape of a Cu cell (1-atom fcc primitive cell x reps) with a k-point grid."""
    a = bulk("Cu", "fcc", a=3.61) * reps
    return shape(a.get_chemical_symbols(), {"Cu": CU}, cell=a.cell, pbc=a.pbc, grid=grid)


def workload(shapes, engine="pydftb", mode="energygeometry"):
    return R.Workload(engine, mode, ["Cu"], list(shapes), [None] * len(shapes), [("Cu", "Cu")],
                      len(shapes), {"Cu": CU})


def plan_of(shapes, ncores, engine="pydftb", memory=64 * GB, **kw):
    return R.make_plan(budget(ncores), workload(shapes, engine), model=MODEL, memory=memory, **kw)


@pytest.fixture(scope="module")
def cu_dataset():
    structures = ase.io.read(ROOT / "dft" / "dft.xyz", ":")
    bolt = R.bolt_for(structures, "energygeometry", 11)
    return structures, bolt


@pytest.fixture(autouse=True)
def no_plan():
    yield
    set_plan(None)
    dftb_stage._MEASURED.clear()


# ---------------------------------------------------------------------------------------
# workload

def test_ptbp_species():
    assert CU.norb == 9 and sorted(l for l, _ in CU.shells) == [0, 1, 2]
    assert min(u for _, u in CU.shells) == pytest.approx(0.0799802838)
    assert R.ptbp_species("H").norb == 1
    assert R.ptbp_species("Mg").norb == 9                 # 3s 3p 3d in PTBP's table
    assert R.ptbp_species("Gd").norb == 16                # not in the table: f block


def test_workload_of_the_cu_dataset(cu_dataset):
    structures, bolt = cu_dataset
    w = R.workload_from_dataset(structures, bolt, engine="pydftb")
    s = w.summary()
    assert s["structures"] == 66 and s["gamma_only"] == sum(b < 0.1 for b in bolt) == 33
    assert s["orbitals"] == [9, 18] and s["kpoints"] == [1, 2048]
    assert w.tables == [("Cu", "Cu")]
    # whole EOS groups only, as the DFTB stage computes them
    assert R.workload_from_dataset(structures[:30], bolt, engine="pydftb").n == 22


def test_simulate_lpt():
    assert R.simulate_lpt([5, 4, 3, 3, 2, 2, 1], 2)[0] == 10
    assert R.simulate_lpt([1, 1, 1, 9], 2, order=[0, 1, 2, 3])[0] == 10    # given order
    assert R.simulate_lpt([], 3)[0] == 0


# ---------------------------------------------------------------------------------------
# the DFTB stage

def test_cu_dataset_on_4_cores_one_process_per_core(cu_dataset):
    structures, bolt = cu_dataset
    for engine in ("pydftb", "dftbplus"):
        w = R.workload_from_dataset(structures, bolt, engine=engine)
        st = R.make_plan(budget(4), w, model=MODEL, memory=64 * GB).stage("dftb")
        assert (st.workers, st.threads, st.order) == (4, 1, "largest-first")


def test_one_structure_with_many_kpoints_gets_the_cores_as_threads():
    st = plan_of([cu_cell((16, 16, 16))], 4).stage("dftb")       # 2048 k-points
    assert (st.workers, st.threads) == (1, 4)
    assert "k-point threads" in st.reasons[0]


def test_two_structures_two_by_two():
    st = plan_of([cu_cell((16, 16, 16)), cu_cell((14, 14, 14))], 4).stage("dftb")
    assert (st.workers, st.threads) == (2, 2)


def test_gamma_only_structures_get_no_kpoint_threads():
    st = plan_of([cu_cell((1, 1, 1), (2, 2, 2))] * 2, 4).stage("dftb")
    assert (st.workers, st.threads) == (2, 1)
    assert any("Gamma point only" in r for r in st.reasons)
    assert any("2 core(s) idle" in r for r in st.reasons)


def test_dftbplus_openmp_threads_only_for_large_matrices():
    small = plan_of([cu_cell((8, 8, 8))] * 2, 4, engine="dftbplus").stage("dftb")
    assert (small.workers, small.threads) == (2, 1)
    assert any("no OpenMP threads" in r for r in small.reasons)
    large = plan_of([cu_cell((2, 2, 2), (4, 4, 4))] * 2, 4, engine="dftbplus").stage("dftb")
    assert (large.workers, large.threads) == (2, 2)                # 576 orbitals


def test_memory_limits_the_processes(cu_dataset):
    structures, bolt = cu_dataset
    w = R.workload_from_dataset(structures, bolt, engine="pydftb")
    per = max(MODEL.memory("pydftb", s) for s in w.shapes)
    st = R.make_plan(budget(4), w, model=MODEL, memory=2.5 * per * R.MEMORY_SAFETY).stage("dftb")
    assert st.workers == 2
    assert any("memory" in r for r in st.reasons)
    tight = R.make_plan(budget(4), w, model=MODEL, memory=0.5 * per)
    assert tight.stage("dftb").workers == 1
    assert any("memory" in s for s in tight.suggestions)


def test_parallel_bo_splits_the_cores_between_points():
    shapes = [cu_cell((8, 8, 8))] * 20
    plan = plan_of(shapes, 4, optimizer="parallel_bo", n_particles=4)
    assert plan.particles == 4 and plan.ncores == 1
    for i in range(4):
        st = plan.for_particle(i).stage("dftb")
        assert (st.workers, st.threads) == (1, 1)
    six = plan_of(shapes, 6, optimizer="parallel_bo", n_particles=4)
    assert sorted(six.share_stages) == [1, 2]
    assert [six.for_particle(i).ncores for i in range(4)] == [2, 2, 1, 1]
    assert six.for_particle(0).stage("dftb").workers == 2


def test_random_workloads_never_oversubscribe():
    rng = random.Random(7)
    for _ in range(60):
        n = rng.randint(1, 30)
        shapes = [cu_cell(rng.choice([(1, 1, 1), (4, 4, 4), (12, 12, 12)]),
                          rng.choice([(1, 1, 1), (2, 1, 1), (2, 2, 2)])) for _ in range(n)]
        ncores = rng.randint(1, 16)
        engine = rng.choice(["pydftb", "dftbplus"])
        optimizer = rng.choice(["bo", "parallel_bo", "pso"])
        particles = rng.randint(1, 6)
        plan = plan_of(shapes, ncores, engine=engine, optimizer=optimizer, n_particles=particles,
                       memory=rng.choice([None, 64 * GB, 0.3 * GB]))
        for cores, stages in plan.share_stages.items():
            for st in stages.values():
                assert st.workers * st.threads <= max(1, cores)
        if plan.particles <= ncores:
            from utils.parallel import core_shares
            assert sum(core_shares(ncores, plan.particles)) == ncores


# ---------------------------------------------------------------------------------------
# estimates and suggestions

def test_estimates_of_the_cu_dataset_close_to_the_measured_bo_step(cu_dataset):
    """M3b measured 27.1 s (pydftb) and 39.7 s (DFTB+) on 4 cores, 65.5 s on 1 core."""
    structures, bolt = cu_dataset
    for engine, cores, measured in (("pydftb", 4, 27.1), ("dftbplus", 4, 39.7), ("pydftb", 1, 65.5)):
        w = R.workload_from_dataset(structures, bolt, engine=engine)
        est = R.make_plan(budget(cores), w, model=MODEL, memory=64 * GB).estimates
        assert est["step"] == pytest.approx(measured, rel=0.25)
        assert est["hotcent"] + est["dftb"] + est["physrep"] + est["other"] == pytest.approx(est["step"])


def test_run_estimate_counts_the_optimizer_evaluations():
    shapes = [cu_cell((8, 8, 8))] * 8
    assert plan_of(shapes, 4, n_calls=11).estimates["evaluations"] == 11
    pbo = plan_of(shapes, 4, optimizer="parallel_bo", n_particles=4, n_calls=11).estimates
    assert pbo["evaluations"] == 8 and pbo["run"] == pytest.approx(2 * pbo["step"])
    assert plan_of(shapes, 4, optimizer="pso", n_particles=4, n_calls=11).estimates["evaluations"] == 8


def test_suggestions():
    few = plan_of([cu_cell((8, 8, 8))] * 2, 8)
    assert any("parallel_bo" in s for s in few.suggestions)
    big_gamma = plan_of([cu_cell((1, 1, 1), (4, 4, 4))], 4)        # 576 orbitals, Gamma
    assert any("PTBP_BLAS_THREADS" in s for s in big_gamma.suggestions)
    dftbplus = plan_of([cu_cell((14, 14, 14))] * 8, 4, engine="dftbplus")
    assert any("--engine pydftb" in s for s in dftbplus.suggestions)


def test_pydftb_on_a_gpu_is_serial_and_not_estimated():
    """M4d: with a GPU the DFTB stage runs in this process, one structure after the
    other; the cost model knows CPUs only, so there is no estimate; plan.json records it."""
    shapes = [cu_cell((8, 8, 8))] * 8
    gpu = {"pydftb_backend": "torch", "device": "cuda", "dtype": "float64", "torch": "2.14.1"}
    plan = plan_of(shapes, 4, n_calls=11, backend=gpu)
    st = plan.stage("dftb")
    assert (st.workers, st.threads) == (1, 1) and "GPU cuda" in st.reasons[0]
    assert "step" not in plan.estimates and "dftb" not in plan.estimates
    assert any("no time estimate" in n for n in plan.notes)
    text = plan.render()
    assert "pydftb on torch (cuda, float64; torch 2.14.1)" in text
    assert "estimated BO step" not in text
    data = json.loads(json.dumps(plan.to_json()))
    assert data["backend"] == gpu and Plan.from_json(data).backend == gpu
    cpu = plan_of(shapes, 4, backend=dict(gpu, device="cpu"))
    assert cpu.stage("dftb").workers == 4 and "step" in cpu.estimates
    plain = plan_of(shapes, 4)
    assert "backend" not in plain.to_json()          # plan.json of NumPy runs as before


def test_gpu_suggestion_only_for_pydftb_on_numpy():
    shapes = [cu_cell((8, 8, 8))] * 8
    found = {"torch": "2.14.1", "cuda": ["NVIDIA A100"], "mps": False, "xpu": []}
    plan = plan_of(shapes, 4, accelerators=found)
    assert any("GPU available (NVIDIA A100)" in s and "gpu_check" in s for s in plan.suggestions)
    torch_gpu = {"pydftb_backend": "torch", "device": "cuda", "dtype": "float64"}
    assert not any("GPU" in s for s in plan_of(shapes, 4, accelerators=found,
                                                backend=torch_gpu).suggestions)
    assert not any("GPU" in s for s in plan_of(shapes, 4, engine="dftbplus",
                                                accelerators=found).suggestions)
    none = {"torch": "2.14.1", "cuda": [], "mps": False, "xpu": []}
    assert not any("GPU" in s for s in plan_of(shapes, 4, accelerators=none).suggestions)


def test_blas_threads_only_on_request(monkeypatch):
    assert plan_of([cu_cell((1, 1, 1), (4, 4, 4))], 4).stage("dftb").blas_threads == 1
    monkeypatch.setenv("PTBP_BLAS_THREADS", "3")
    assert plan_of([cu_cell((1, 1, 1), (4, 4, 4))], 4).stage("dftb").blas_threads == 3
    assert plan_of([cu_cell((1, 1, 1), (4, 4, 4))], 1).stage("dftb").blas_threads == 3


def test_render_and_json_round_trip(cu_dataset):
    structures, bolt = cu_dataset
    w = R.workload_from_dataset(structures, bolt, engine="pydftb")
    plan = R.make_plan(budget(4), w, model=MODEL, memory=8 * GB, optimizer="parallel_bo",
                       n_particles=2, n_calls=11)
    text = R.render_report(plan)
    for part in ("66 structures", "estimated BO step per BO point", "estimated run", "memory: up to"):
        assert part in text
    data = json.loads(json.dumps(plan.to_json()))
    again = Plan.from_json(data)
    assert again.to_json() == data
    assert again.for_particle(1).stage("dftb").workers == plan.for_particle(1).stage("dftb").workers
    line = plan.step_line({"hotcent": 4.0, "dftb": 9.0, "physrep": 11.0, "total": 25.0})
    assert line.startswith("[plan] BO step: hotcent 4.0 s (estimated") and "total 25.0 s" in line


# ---------------------------------------------------------------------------------------
# the DFTB stage uses the plan's estimates, then the measured times

def test_task_order_measured_then_estimated_then_size():
    tasks = [dftb_stage.StructureTask(i, ["Cu"], np.zeros((1, 3)), np.eye(3), np.ones(3, bool),
                                      cost=c) for i, c in enumerate([1.0, 3.0, 2.0])]
    assert dftb_stage.task_order(tasks) == [1, 2, 0]                     # k-points x N^3
    plan = Plan(budget(4), costs=[5.0, 1.0, 9.0])
    set_plan(plan)
    assert dftb_stage.task_order(tasks) == [2, 0, 1]                     # cost model
    dftb_stage._record(3, [0.1, 0.3, 0.2])
    assert dftb_stage.task_order(tasks) == [1, 2, 0]                     # measured


def _fake_structures(n):
    return [Atoms("Cu2", positions=[[0, 0, 0], [0, 0, 2.2 + 0.01 * i]], cell=[8, 8, 8], pbc=False)
            for i in range(n)]


def test_serial_stage_records_measured_times(monkeypatch):
    monkeypatch.setattr(dftb_stage, "run_structure", lambda task: Atoms("Cu"))
    out = dftb_stage.run_dataset(_fake_structures(4), [1.0] * 4, 2, 5.0)
    assert len(out[0]) == 4
    walls = dftb_stage.measured_times(4)
    assert walls is not None and len(walls) == 4 and all(w >= 0 for w in walls)


def test_parallel_stage_passes_blas_threads_and_records_times(monkeypatch):
    saved = eng.settings()
    eng.configure(engine="pydftb")
    seen = {}

    def fake_map(fn, tasks, workers, order=None, threads=1, blas_threads=1, backend=None):
        seen.update(workers=workers, order=order, threads=threads, blas=blas_threads,
                    args=tasks[0].dftb_args)
        return [TaskResult(i, True, Atoms("Cu"), wall=0.5 + i) for i in range(len(tasks))]

    monkeypatch.setattr(dftb_stage, "map_tasks", fake_map)
    set_plan(Plan(budget(2), engine="pydftb",
                  stages={"dftb": StagePlan(2, 1, 3, "largest-first")}, costs=[1, 4, 2, 3]))
    try:
        dftb_stage.run_dataset(_fake_structures(4), [1.0] * 4, 2, 5.0)
    finally:
        eng._SETTINGS.update(saved)
    assert seen["blas"] == 3 and seen["args"]["blas_threads"] == 3
    assert seen["order"] == [1, 3, 2, 0]
    assert dftb_stage.measured_times(4) == [0.5, 1.5, 2.5, 3.5]


# ---------------------------------------------------------------------------------------
# ptbp plan

def _ptbp(*args):
    return subprocess.run([sys.executable, str(ROOT / "parameterize.py"), "plan", *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)


def test_ptbp_plan_text_and_json():
    out = _ptbp("dft/dft.xyz", "--mode", "energygeometry", "--engine", "pydftb", "--ncores", "4")
    assert out.returncode == 0, out.stderr
    assert "[plan] dataset: 66 structures" in out.stdout
    assert "DFTB: 4 processes x 1 k-point thread" in out.stdout
    assert "estimated BO step" in out.stdout
    js = _ptbp("dft/dft.xyz", "--mode", "dataset", "--ncores", "2", "--json")
    assert js.returncode == 0, js.stderr
    data = json.loads(js.stdout)
    assert data["stages"]["dftb"]["workers"] == 2 and data["workload"]["structures"] == 66
    assert _ptbp("missing.xyz").returncode == 2
