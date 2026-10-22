"""The parallel plan of a PTBP run: processes and threads for each stage of a BO step.

``cli/run.py`` builds the plan once, prints it, saves it as ``plan.json`` and makes it
the process-wide plan (:func:`set_plan`); forked workers inherit it.  Library use
without a plan (``current_plan() is None``) keeps PTBP's historical serial behaviour,
untouched thread settings included.

M3a planned the DFTB stage with a simple rule (:func:`simple_plan`): one process per
structure up to the core budget, spare cores as threads of each process (pydftb k-point
threads or DFTB+ OpenMP threads), structures handed out largest first (k-points x
orbitals^3).  Since M3c ``cli/run.py`` uses the cost-model recommender
(``utils.recommender.make_plan``), which fills in the same plan plus estimates, the
stages of every core share of a parallel_bo batch and suggestions.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .budget import CpuBudget

THREAD_KIND = {"pydftb": "k-point thread", "dftbplus": "OpenMP thread"}


@dataclass
class StagePlan:
    """Processes and threads of one stage."""
    workers: int = 1
    threads: int = 1          # per process: pydftb k-point threads / DFTB+ OpenMP threads
    blas_threads: int = 1     # BLAS inside the worker processes (reproducible results)
    order: str = "index"      # "largest-first": tasks handed out by estimated cost
    reasons: list = field(default_factory=list)

    @property
    def cores(self) -> int:
        return self.workers * self.threads


@dataclass
class Plan:
    budget: CpuBudget
    engine: str = "dftbplus"
    optimizer: str = "bo"
    particles: int = 1          # BO points evaluated at the same time (parallel_bo)
    ncores: int = 1             # cores of one evaluation (the budget, or a particle's share)
    n_structures: int = 0
    stages: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    # utils.recommender (M3c): estimates (seconds, bytes), the stages of every core share
    # of a parallel_bo batch, estimated seconds per structure (task order), suggestions
    estimates: dict = field(default_factory=dict)
    share_stages: dict = field(default_factory=dict)
    costs: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    workload: dict = field(default_factory=dict)
    # M4d: pydftb on torch (backend, device, dtype, torch version); empty for NumPy
    backend: dict = field(default_factory=dict)
    # M4e: hotcent's confinement scheme and version (set by cli/run.py)
    hotcent: dict = field(default_factory=dict)

    def stage(self, name: str) -> StagePlan:
        return self.stages.get(name) or StagePlan(reasons=["serial"])

    def for_particle(self, i: int) -> Plan:
        """The plan inside particle ``i`` of a parallel_bo batch: its share of the cores."""
        share = core_shares(self.budget.ncores, self.particles)[i % self.particles]
        stages = self.share_stages.get(share)
        if stages is None:
            return _plan(self.budget, self.engine, self.optimizer, self.n_structures, 1, share)
        return Plan(self.budget, self.engine, self.optimizer, 1, share, self.n_structures,
                    dict(stages), costs=self.costs,
                    estimates=dict(self.estimates) if share == self.ncores else {})

    def step_line(self, measured: dict) -> str:
        """The measured stage times of a BO step next to the plan's estimates."""
        est = self.estimates
        parts = []
        for key in ("hotcent", "dftb", "physrep"):
            if key in measured:
                guess = f" (estimated {format_seconds(est[key])})" if key in est else ""
                parts.append(f"{key} {format_seconds(measured[key])}{guess}")
        if "total" in measured:
            guess = f" (estimated {format_seconds(est['step'])})" if "step" in est else ""
            parts.append(f"total {format_seconds(measured['total'])}{guess}")
        return "[plan] BO step: " + ", ".join(parts)

    def render(self) -> str:
        head = f"[plan] {self.budget.describe()}; engine {self.engine}; optimizer {self.optimizer}"
        lines = [head]
        if self.backend:
            b = self.backend
            lines.append(f"[plan]   pydftb on {b.get('pydftb_backend')} ({b.get('device')}, "
                         f"{b.get('dtype')}; torch {b.get('torch')})")
        if self.hotcent:
            lines.append(f"[plan]   hotcent {self.hotcent.get('version')}: "
                         f"{self.hotcent.get('confinement')} confinement")
        if self.particles > 1:
            shares = sorted(set(core_shares(self.budget.ncores, self.particles)))
            lines.append(f"[plan]   {self.particles} BO points evaluated at a time, "
                         f"{'/'.join(map(str, shares))} core(s) each")
        per = " (per BO point)" if self.particles > 1 else ""
        for name, title in (("dftb", "DFTB"), ("hotcent", "hotcent"), ("physrep", "physrep")):
            st = self.stage(name)
            if st.workers == 1 and st.threads == 1:
                what = "serial"
            else:
                kind = THREAD_KIND.get(self.engine, "thread") if name == "dftb" else "thread"
                what = (f"{st.workers} process{'es' * (st.workers > 1)} x {st.threads} "
                        f"{kind}{'s' * (st.threads > 1)}")
                if st.order == "largest-first":
                    what += ", largest first"
            why = f" -- {'; '.join(st.reasons)}" if st.reasons else ""
            lines.append(f"[plan]   {title}{per}: {what}{why}")
        lines += [f"[plan]   note: {n}" for n in self.notes]
        lines += self._render_estimates()
        lines += [f"[plan]   suggestion: {n}" for n in self.suggestions]
        return "\n".join(lines)

    def _render_estimates(self) -> list:
        est = self.estimates
        if not est or "step" not in est:
            return []
        per = " per BO point" if self.particles > 1 else ""
        parts = " + ".join(f"{k} {format_seconds(est[k])}"
                           for k in ("hotcent", "dftb", "physrep", "other") if k in est)
        lines = [f"[plan]   estimated BO step{per}: {parts} = {format_seconds(est['step'])}"
                 f"{_speed_note(est.get('speed'))}"]
        if "run" in est:
            n = int(est["evaluations"])
            lines.append(f"[plan]   estimated run: {n} BO point{'s' * (n != 1)}, "
                         f"about {format_seconds(est['run'])}")
        if est.get("memory_per_process"):
            avail = est.get("memory_available")
            lines.append(f"[plan]   memory: up to {format_bytes(est['memory_per_process'])} per "
                         f"DFTB process" + (f" ({format_bytes(avail)} available)" if avail else ""))
        return lines

    def to_json(self) -> dict:
        return {"budget": self.budget.to_json(), "engine": self.engine, "optimizer": self.optimizer,
                "particles": self.particles, "ncores": self.ncores,
                "n_structures": self.n_structures,
                "stages": {k: asdict(v) for k, v in self.stages.items()}, "notes": list(self.notes),
                "estimates": dict(self.estimates), "suggestions": list(self.suggestions),
                "workload": dict(self.workload),
                "share_stages": {str(c): {k: asdict(v) for k, v in st.items()}
                                 for c, st in self.share_stages.items()},
                "costs": list(self.costs),
                **({"backend": dict(self.backend)} if self.backend else {}),
                **({"hotcent": dict(self.hotcent)} if self.hotcent else {})}

    @classmethod
    def from_json(cls, data: dict) -> Plan:
        budget = CpuBudget(int(data["budget"]["ncores"]), dict(data["budget"].get("sources", {})))
        plan = cls(budget, data.get("engine", "dftbplus"), data.get("optimizer", "bo"),
                   int(data.get("particles", 1)), int(data.get("ncores", 1)),
                   int(data.get("n_structures", 0)),
                   {k: StagePlan(**v) for k, v in data.get("stages", {}).items()},
                   list(data.get("notes", [])))
        plan.estimates = dict(data.get("estimates", {}))
        plan.suggestions = list(data.get("suggestions", []))
        plan.workload = dict(data.get("workload", {}))
        plan.share_stages = {int(c): {k: StagePlan(**v) for k, v in st.items()}
                             for c, st in data.get("share_stages", {}).items()}
        plan.costs = list(data.get("costs", []))
        plan.backend = dict(data.get("backend", {}))
        plan.hotcent = dict(data.get("hotcent", {}))
        return plan

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n")


def format_seconds(t: float) -> str:
    if t < 120:
        return f"{t:.1f} s"
    if t < 7200:
        return f"{t / 60:.1f} min"
    return f"{t / 3600:.1f} h"


def format_bytes(b: float) -> str:
    return f"{b / 2 ** 30:.1f} GB" if b >= 2 ** 30 else f"{b / 2 ** 20:.0f} MB"


def _speed_note(speed) -> str:
    if speed is None or abs(speed - 1.0) < 0.05:
        return ""
    return f" (this machine {speed:.2f}x the calibration machine)"


def core_shares(ncores: int, parts: int) -> list[int]:
    """Cores of each of ``parts`` concurrent evaluations (at least 1 each)."""
    parts = max(1, int(parts))
    return [max(1, ncores // parts + (i < ncores % parts)) for i in range(parts)]


def dftb_stage(ncores: int, n_structures: int, engine: str) -> StagePlan:
    """M3a rule: a process per structure up to ``ncores``, spare cores as threads."""
    if ncores <= 1:
        return StagePlan(reasons=["1 core"])
    workers = max(1, min(ncores, n_structures))
    threads = max(1, ncores // workers)
    kind = THREAD_KIND.get(engine, "thread")
    if n_structures >= ncores:
        why = f"{n_structures} structures >= {ncores} cores: one structure per process"
    else:
        why = (f"{n_structures} structure{'s' * (n_structures != 1)} < {ncores} cores: "
               f"the spare cores as {kind}s")
    reasons = [why]
    if workers * threads < ncores:
        reasons.append(f"{ncores - workers * threads} core(s) idle")
    return StagePlan(workers, threads, 1, "largest-first" if workers > 1 else "index", reasons)


def hotcent_stage(ncores: int) -> StagePlan:
    """M3b: every table once, its distance rows interleaved over the processes."""
    if ncores <= 1:
        return StagePlan(reasons=["1 core; every table computed once"])
    return StagePlan(ncores, 1, 1, "index", ["each table once, its rows shared by the processes"])


def _plan(budget, engine, optimizer, n_structures, particles, ncores) -> Plan:
    plan = Plan(budget, engine, optimizer, particles, ncores, n_structures)
    plan.stages["dftb"] = dftb_stage(ncores, n_structures, engine)
    plan.stages["hotcent"] = hotcent_stage(ncores)
    plan.stages["physrep"] = StagePlan(reasons=["vectorised, in this process"])
    return plan


def simple_plan(budget: CpuBudget, n_structures: int, engine: str = "dftbplus",
                optimizer: str = "bo", particles: int = 1) -> Plan:
    """The plan of a run; ``particles`` counts only for ``optimizer='parallel_bo'``."""
    particles = max(1, int(particles)) if optimizer == "parallel_bo" else 1
    share = core_shares(budget.ncores, particles)[0]
    plan = _plan(budget, engine, optimizer, n_structures, particles, share)
    if particles > budget.ncores:
        plan.notes.append(f"{particles} BO points at a time on {budget.ncores} cores: "
                          f"the machine is oversubscribed {particles / budget.ncores:.1f}x; "
                          f"use --n-particles {budget.ncores} or fewer")
    return plan


_CURRENT: Plan | None = None


def set_plan(plan: Plan | None) -> None:
    """Make ``plan`` the plan of this process (and of the processes it forks)."""
    global _CURRENT
    _CURRENT = plan


def current_plan() -> Plan | None:
    return _CURRENT
