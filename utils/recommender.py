"""Computation strategy of a PTBP run (milestone M3c): the plan from a cost model.

M3a planned the DFTB stage with a rule (a process per structure up to the core budget,
spare cores as threads).  Here every structure gets an estimated time and memory from
``pydftb.perfmodel``, and

1. the **workload** is what one BO step computes: the structures (whole EOS groups),
   each with the k-point grid PTBP will use (Gamma only for Boltzmann weights < 0.1),
   its orbitals and its pair counts, from the valence shells and Hubbard U of PTBP's
   parameter table; plus the hotcent tables;
2. for every split of the cores into processes x threads the **largest-first schedule**
   of the estimated times is simulated, as ``utils.parallel`` hands out tasks; splits
   whose processes would not fit in the available memory are dropped; the shortest
   stage wins, and among splits within 2 % the one with more processes;
3. DFTB+ gets OpenMP threads only when every structure has >= 500 orbitals (below,
   threads make DFTB+ slower: M3a, M3c);
4. hotcent: a process per core (M3b); physrep: in this process.

Only settings that leave every result unchanged are chosen.  The rest -- the
optimizer, BLAS threads for large Gamma-only cells, the engine -- are suggestions.
"""
from __future__ import annotations

import heapq
import os
from dataclasses import dataclass, field
from math import floor
from pathlib import Path

from pydftb.perfmodel import OMP_MIN_ORBITALS, CostModel, Species, load_probe, shape
from utils.parallel.budget import available_memory
from utils.parallel.plan import Plan, StagePlan, THREAD_KIND, core_shares, hotcent_stage

MEMORY_SAFETY = 1.5          # the memory model under-estimates by up to 29 % (M3c report)
TIE = 0.02                   # makespans within 2 %: take more processes, fewer threads
TASK_OVERHEAD = 0.004        # s per task: pipe, result pickling (M3a)
STAGE_OVERHEAD = (0.2, 0.015)  # s, s per structure: tasks, fork, writing the xyz files
CONTENTION = 0.01            # per extra concurrent process: memory bandwidth (M3a: 3 % at 4)
# hotcent, one process (M3c): an atom 0.1-0.3 s (Au 0.9 s); a table of 700 rows
# 1.7 s + 1.56 s per pair of shells (both orders for two elements): H-H 3.0, C-C 7.8,
# Cu-Cu 15.5, Cu-O 19.6 s
HOTCENT_ATOM = 0.25
HOTCENT_TABLE = (1.7, 1.56)
# sigma_rep search of physrep: 74 evaluations per BO step (M3b), per structure each
PHYSREP_EVALS = 74
PHYSREP_PER_STRUCTURE = {"energygeometry": 0.163 / 66, "dataset": 0.042 / 66, "reaction": 0.042 / 66}
OTHER_PER_STEP = 1.0         # data processing and plots (M3b)
BLAS_SUGGEST_ORBITALS = 500  # Gamma-only cells from which BLAS threads pay off


def probe_cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "ptbp" / "probe.json"


# ---------------------------------------------------------------------------------------
# workload

def ptbp_species(symbol: str) -> Species:
    """Valence shells and Hubbard U of PTBP's parameter table (what hotcent writes)."""
    from ase.data import atomic_numbers
    from utils.parameters_set import get_PTBP
    entry = get_PTBP().get(symbol)
    if entry is None:
        return Species.from_block(atomic_numbers[symbol])
    shells = []
    for tok in str(entry["valence_shell"]).split():
        letter = tok[1]                                   # '3d10' -> 'd'
        u = entry.get("U" + letter) or 0.0
        shells.append(("spdf".index(letter), float(u) if u > 0 else 0.08))
    return Species(tuple(shells))


def hotcent_tables(symbols, opt_elem=()) -> list:
    """The tables hotcent computes per BO step (each once, M3b); [] without hotcent."""
    try:
        from utils.hotcent_parallel import job_tables, skf_jobs
    except ImportError:
        return []
    jobs = skf_jobs(list(symbols), list(opt_elem))
    return list(dict.fromkeys(t for job in jobs for t in job_tables(job)))


@dataclass
class Workload:
    """What one BO step computes."""
    engine: str
    mode: str
    symbols: list
    shapes: list                    # perfmodel.Shape per structure of the DFTB stage
    grids: list                     # k-point grid per structure
    tables: list = field(default_factory=list)    # hotcent tables
    n_dataset: int = 0              # structures in the file
    species: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.shapes)

    def summary(self) -> dict:
        def rng(vals):
            return [min(vals), max(vals)] if vals else [0, 0]
        return {"structures": self.n, "in_file": self.n_dataset,
                "gamma_only": sum(1 for s in self.shapes if s.gamma),
                "atoms": rng([s.natoms for s in self.shapes]),
                "orbitals": rng([s.norb for s in self.shapes]),
                "kpoints": rng([s.nk for s in self.shapes]),
                "elements": list(self.symbols), "hotcent_tables": len(self.tables)}


def workload_from_dataset(structures, bolt, *, kpts_density: float = 5.0, eos_points: int = 11,
                          engine: str = "dftbplus", mode: str = "energygeometry",
                          opt_elem=(), species: dict | None = None) -> Workload:
    """The workload of a BO step over ``structures`` (``bolt``: weight per structure)."""
    from utils.dftb_stage import kpoint_grid
    eos = max(1, int(eos_points))
    used = list(structures[:int(len(structures) / eos) * eos])
    symbols = list(dict.fromkeys(s for a in structures for s in a.get_chemical_symbols()))
    species = dict(species or {})
    for sym in symbols:
        species.setdefault(sym, ptbp_species(sym))
    shapes, grids = [], []
    for atoms, b in zip(used, bolt):
        periodic = bool(atoms.pbc.all()) and atoms.cell.rank == 3
        grid = list(map(int, kpoint_grid(atoms, kpts_density, b))) if periodic else [1, 1, 1]
        grids.append(grid)
        shapes.append(shape(atoms.get_chemical_symbols(), species,
                            cell=atoms.cell if periodic else None, pbc=atoms.pbc,
                            positions=atoms.positions, grid=grid))
    return Workload(engine, mode, symbols, shapes, grids, hotcent_tables(symbols, opt_elem),
                    len(structures), species)


def bolt_for(structures, mode: str, eos_points: int, multi_element=None) -> list:
    """Per-structure weights as cli/run.py computes them (bolt_f11)."""
    import utils.data as data
    eos = max(1, int(eos_points))
    center = structures[floor(eos / 2)::eos]
    kind = "boltzmann" if mode == "energygeometry" and not multi_element else "identical"
    return list(data.get_bolt_factor(center, kind, 3000)[1])


# ---------------------------------------------------------------------------------------
# scheduling

def simulate_lpt(times, workers: int, order=None) -> tuple[float, list]:
    """Makespan of handing ``times`` out in ``order`` (default: largest first) to
    ``workers`` processes, each taking the next task when it is free."""
    times = list(times)
    if order is None:
        order = sorted(range(len(times)), key=lambda i: (-times[i], i))
    workers = max(1, int(workers))
    heap = [(0.0, w) for w in range(workers)]
    loads = [0.0] * workers
    for i in order:
        t, w = heapq.heappop(heap)
        loads[w] = t + times[i]
        heapq.heappush(heap, (loads[w], w))
    return max(loads) if loads else 0.0, loads


@dataclass
class Choice:
    workers: int
    threads: int
    makespan: float
    memory: float          # bytes per process (largest calculation)
    busy: float            # sum of task times (core-seconds of work)


def _task_times(model: CostModel, engine: str, shapes, threads: int, workers: int) -> list:
    slow = 1.0 + CONTENTION * (workers - 1)
    return [model.time(engine, s, threads) * slow + TASK_OVERHEAD for s in shapes]


def useful_threads(engine: str, shapes, threads: int) -> int:
    """Threads per process worth giving: pydftb's k-point threads need k-points to split
    (none for Gamma-only structures); DFTB+'s OpenMP threads need large matrices."""
    if threads <= 1 or not shapes:
        return 1
    if engine == "pydftb":
        from pydftb.perfmodel import effective_kthreads
        return max(effective_kthreads(s, threads) for s in shapes)
    return threads if min(s.norb for s in shapes) >= OMP_MIN_ORBITALS else 1


def dftb_choices(model: CostModel, engine: str, shapes, ncores: int) -> list[Choice]:
    """Every processes x threads split of ``ncores`` for the DFTB stage, simulated."""
    n = len(shapes)
    if n == 0:
        return [Choice(1, 1, 0.0, 0.0, 0.0)]
    mem = max(model.memory(engine, s) for s in shapes)
    out = []
    for w in range(1, min(ncores, n) + 1):
        t = useful_threads(engine, shapes, max(1, ncores // w))
        times = _task_times(model, engine, shapes, t, w)
        makespan, _ = simulate_lpt(times, w)
        out.append(Choice(w, t, makespan, mem, sum(times)))
    return out


def pick(choices: list[Choice], memory_avail: float | None) -> tuple[Choice, list[str]]:
    """The split to use and the reasons; memory first, then makespan, then processes."""
    notes = []
    fits = choices
    if memory_avail is not None:
        fits = [c for c in choices if c.workers * c.memory * MEMORY_SAFETY <= memory_avail]
        if len(fits) < len(choices):
            notes.append(f"memory: {len(choices) - len(fits)} split(s) need more than the "
                         f"{memory_avail / 2 ** 30:.1f} GB available")
        if not fits:
            fits = [min(choices, key=lambda c: (c.workers, -c.threads))]
            notes.append(f"memory: even one process may need {fits[0].memory * MEMORY_SAFETY / 2 ** 30:.1f} GB")
    best = min(c.makespan for c in fits)
    near = [c for c in fits if c.makespan <= best * (1 + TIE) + 1e-9]
    return max(near, key=lambda c: (c.workers, -c.threads)), notes


def _dftb_stage_plan(choice: Choice, ncores: int, shapes, engine: str, memory_notes,
                     memory_avail) -> StagePlan:
    kind = THREAD_KIND.get(engine, "thread")
    n = len(shapes)
    w, t = choice.workers, choice.threads
    reasons = []
    limited = bool(memory_notes) and w < min(ncores, n)
    if ncores <= 1:
        reasons.append("1 core")
    elif limited:
        reasons.append(f"memory: {w} process{'es' * (w > 1)} of up to {choice.memory / 2 ** 30:.1f} GB "
                       f"(x{MEMORY_SAFETY} for safety) fit in the {memory_avail / 2 ** 30:.1f} GB available")
        memory_notes = [m for m in memory_notes if "even one process" in m]
    elif w == ncores:
        reasons.append(f"{n} structures >= {ncores} cores: a process per core, "
                       f"the structures handed out largest first")
    elif w == n:
        reasons.append(f"{n} structure{'s' * (n != 1)} < {ncores} cores: a process per structure"
                       + (f", the spare cores as {kind}s" if t > 1 else ""))
    else:
        reasons.append(f"cost model: {w} process{'es' * (w > 1)} x {t} {kind}{'s' * (t > 1)} "
                       f"finish first (estimated {choice.makespan:.1f} s)")
    if t == 1 and ncores > w:
        if engine == "dftbplus":
            reasons.append(f"no OpenMP threads below {OMP_MIN_ORBITALS} orbitals "
                           f"(slower than one thread)")
        elif all(s.nk == 1 for s in shapes):
            reasons.append("Gamma point only: no k-points to split over threads")
    idle = ncores - w * t
    if idle > 0 and ncores > 1:
        reasons.append(f"{idle} core(s) idle")
    reasons.extend(memory_notes)
    blas = _blas_opt_in()
    order = "largest-first" if choice.workers > 1 else "index"
    return StagePlan(choice.workers, choice.threads, blas, order, reasons)


def _blas_opt_in() -> int:
    """PTBP_BLAS_THREADS: BLAS threads of every pydftb calculation (results then differ in
    the last bits with the number of threads); DFTB+ runs keep their OpenMP setting."""
    raw = os.environ.get("PTBP_BLAS_THREADS", "").strip()
    try:
        return max(1, int(raw)) if raw else 1
    except ValueError:
        return 1


# ---------------------------------------------------------------------------------------
# stage estimates

def hotcent_estimate(tables, species: dict, workers: int) -> float:
    """Seconds of the hotcent stage: atoms in this process, table rows sharded."""
    if not tables:
        return 0.0
    elements = {e for t in tables for e in t}
    atoms = HOTCENT_ATOM * len(elements)
    costs = [HOTCENT_TABLE[0] + HOTCENT_TABLE[1] * len(species[a].shells) * len(species[b].shells)
             * (1 if a == b else 2) for a, b in tables]
    nshard = max(1, -(-workers // len(tables)))
    shards = [c / nshard for c in costs for _ in range(nshard)]
    makespan, _ = simulate_lpt(shards, workers, order=list(range(len(shards))))
    return atoms + makespan + 0.05 * len(tables)


def physrep_estimate(mode: str, n: int) -> float:
    per = PHYSREP_PER_STRUCTURE.get(mode, PHYSREP_PER_STRUCTURE["energygeometry"])
    return PHYSREP_EVALS * per * n


def step_estimate(model: CostModel, workload: Workload, stages: dict, ncores: int) -> dict:
    st = stages["dftb"]
    times = _task_times(model, workload.engine, workload.shapes, st.threads, st.workers)
    makespan, _ = simulate_lpt(times, st.workers)
    dftb = makespan + STAGE_OVERHEAD[0] + STAGE_OVERHEAD[1] * workload.n
    est = {"hotcent": hotcent_estimate(workload.tables, workload.species, stages["hotcent"].workers),
           "dftb": dftb, "physrep": physrep_estimate(workload.mode, workload.n),
           "other": OTHER_PER_STEP}
    est["step"] = sum(est.values())
    est["dftb_core_seconds"] = sum(times)
    est["dftb_efficiency"] = sum(times) / max(1e-9, ncores * makespan) if makespan else 1.0
    est["memory_per_process"] = max((model.memory(workload.engine, s) for s in workload.shapes),
                                    default=0.0)
    return est


# ---------------------------------------------------------------------------------------
# the plan

def _stages_for(model, workload, ncores, memory_avail) -> dict:
    shapes = workload.shapes
    stages = {"hotcent": hotcent_stage(ncores),
              "physrep": StagePlan(reasons=["vectorised, in this process"])}
    if ncores <= 1 or not shapes:
        stages["dftb"] = StagePlan(1, 1, _blas_opt_in(), "index",
                                   ["1 core" if shapes else "no structures to compute"])
        return stages
    best, notes = pick(dftb_choices(model, workload.engine, shapes, ncores), memory_avail)
    stages["dftb"] = _dftb_stage_plan(best, ncores, shapes, workload.engine, notes, memory_avail)
    return stages


def make_plan(budget, workload: Workload, *, optimizer: str = "bo", n_particles: int = 4,
              n_calls: int | None = None, model: CostModel | None = None,
              memory: float | None = None, probe: bool = False) -> Plan:
    """The plan of a run over ``workload`` on ``budget`` (utils.parallel.CpuBudget).

    ``n_particles``: BO points per batch (parallel_bo, evaluated at the same time) or
    swarm size (pso, evaluated one after the other), as ``Loss.optimize`` uses them.
    ``memory``: bytes the run may still allocate (default: MemAvailable / cgroup).
    ``model``: cost model (default: calibration constants, scaled by a cached probe of
    this machine if there is one; ``probe=True`` runs the probe when none is cached).
    """
    if model is None:
        speed = load_probe(probe_cache_path(), run=probe)
        model = CostModel(speed=speed or 1.0)
    if memory is None:
        memory = available_memory()
    batch = max(2, int(n_particles))                       # Loss._optimize_parallel_bo/_pso
    particles = batch if optimizer == "parallel_bo" else 1
    shares = core_shares(budget.ncores, particles)
    share_stages = {}
    for share in sorted(set(shares)):
        # concurrent BO points share the memory as they share the cores
        mem = None if memory is None else memory * share / budget.ncores
        share_stages[share] = _stages_for(model, workload, share, mem)
    share = shares[0]
    plan = Plan(budget, workload.engine, optimizer, particles, share, workload.n,
                dict(share_stages[share]))
    plan.share_stages = share_stages
    plan.costs = [round(model.time(workload.engine, s, 1), 6) for s in workload.shapes]
    plan.workload = workload.summary()
    plan.estimates = step_estimate(model, workload, plan.stages, share)
    plan.estimates["speed"] = model.speed
    plan.estimates["memory_available"] = memory
    if n_calls is not None:
        n_calls = int(n_calls)
        if optimizer == "parallel_bo":                     # rounds of `batch` points at once
            rounds = max(1, n_calls // batch)
            plan.estimates["run"] = rounds * plan.estimates["step"]
            plan.estimates["evaluations"] = rounds * batch
        else:
            evals = batch * max(2, n_calls // batch) if optimizer == "pso" else n_calls
            plan.estimates["run"] = evals * plan.estimates["step"]
            plan.estimates["evaluations"] = evals
    if particles > budget.ncores:
        plan.notes.append(f"{particles} BO points at a time on {budget.ncores} cores: "
                          f"the machine is oversubscribed {particles / budget.ncores:.1f}x; "
                          f"use --n-particles {budget.ncores} or fewer")
    plan.suggestions = suggestions(plan, workload, model, memory)
    return plan


def suggestions(plan: Plan, workload: Workload, model: CostModel, memory) -> list:
    """Advice that would change results or the run's set-up: printed, never applied."""
    out = []
    C = plan.budget.ncores
    est = plan.estimates
    # throughput of the optimizer: BO points side by side, each on one core
    points = C
    per_process = est.get("memory_per_process", 0.0) * MEMORY_SAFETY
    if memory is not None and per_process > 0:
        points = min(points, int(memory // per_process))
    if plan.optimizer == "bo" and points > 1:
        serial_stages = _stages_for(model, workload, 1, None)
        one = step_estimate(model, workload, serial_stages, 1)["step"]
        per_point_bo = est["step"]
        per_point_pbo = one / points
        if per_point_pbo < 0.8 * per_point_bo:
            out.append(f"throughput: {points} BO points at a time on 1 core each "
                       f"(--optimizer parallel_bo --n-particles {points}) would take about "
                       f"{per_point_pbo:.0f} s per point instead of {per_point_bo:.0f} s; "
                       f"batched BO proposes points differently, so the optimisation path changes")
    # BLAS threads for large Gamma-only cells (pydftb; DFTB+ gets its threads by the plan)
    big = [s for s in workload.shapes if s.gamma and s.norb >= BLAS_SUGGEST_ORBITALS]
    st = plan.stages.get("dftb")
    if (workload.engine == "pydftb" and big and st is not None
            and st.workers * st.threads < C and st.blas_threads == 1):
        out.append(f"{len(big)} Gamma-only structure(s) with >= {BLAS_SUGGEST_ORBITALS} orbitals "
                   f"and idle cores: PTBP_BLAS_THREADS={max(2, C // max(1, st.workers))} lets "
                   f"BLAS use them (results then differ in the last bits)")
    # the other engine
    other = "dftbplus" if workload.engine == "pydftb" else "pydftb"
    if workload.shapes and st is not None:
        mine = sum(model.time(workload.engine, s, 1) for s in workload.shapes)
        theirs = sum(model.time(other, s, 1) for s in workload.shapes)
        if theirs < mine / 1.5:
            out.append(f"engine: the DFTB stage would take about {mine / theirs:.1f}x less "
                       f"CPU time with --engine {other} (results differ at the 1e-8 level)")
    # memory
    if memory is not None and est.get("memory_per_process", 0) * MEMORY_SAFETY > memory:
        out.append(f"memory: one calculation may need {est['memory_per_process'] / 2 ** 30:.1f} GB, "
                   f"more than the {memory / 2 ** 30:.1f} GB available")
    return out


def render_report(plan: Plan) -> str:
    """``ptbp plan``: dataset statistics, the plan, estimates and suggestions."""
    w = plan.workload
    lines = [f"[plan] dataset: {w.get('in_file', w.get('structures'))} structures, "
             f"{w.get('structures')} per BO step ({w.get('gamma_only')} Gamma only); "
             f"atoms {w['atoms'][0]}-{w['atoms'][1]}, orbitals {w['orbitals'][0]}-{w['orbitals'][1]}, "
             f"k-points {w['kpoints'][0]}-{w['kpoints'][1]}; elements {' '.join(w.get('elements', []))}; "
             f"{w.get('hotcent_tables', 0)} hotcent table(s)"]
    lines.append(plan.render())
    return "\n".join(lines)
