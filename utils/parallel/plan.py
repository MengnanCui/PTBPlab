"""The parallel plan of a PTBP run: processes and threads for each stage of a BO step.

``cli/run.py`` builds the plan once (:func:`simple_plan`), prints it, saves it as
``plan.json`` and makes it the process-wide plan (:func:`set_plan`); forked workers
inherit it.  Library use without a plan (``current_plan() is None``) keeps PTBP's
historical serial behaviour, untouched thread settings included.

M3a plans the DFTB stage with a simple rule: one process per structure up to the
core budget, spare cores as threads of each process (pydftb k-point threads or
DFTB+ OpenMP threads), structures handed out largest first (k-points x orbitals^3).
The cost-model recommender of M3c replaces this rule.
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

    def stage(self, name: str) -> StagePlan:
        return self.stages.get(name) or StagePlan(reasons=["serial"])

    def for_particle(self, i: int) -> Plan:
        """The plan inside particle ``i`` of a parallel_bo batch: its share of the cores."""
        share = core_shares(self.budget.ncores, self.particles)[i % self.particles]
        return _plan(self.budget, self.engine, self.optimizer, self.n_structures, 1, share)

    def render(self) -> str:
        head = f"[plan] {self.budget.describe()}; engine {self.engine}; optimizer {self.optimizer}"
        lines = [head]
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
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {"budget": self.budget.to_json(), "engine": self.engine, "optimizer": self.optimizer,
                "particles": self.particles, "ncores": self.ncores,
                "n_structures": self.n_structures,
                "stages": {k: asdict(v) for k, v in self.stages.items()}, "notes": list(self.notes)}

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n")


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
