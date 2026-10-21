"""Parallel execution of PTBP's BO steps on one machine (milestone M3).

* :mod:`.budget`   how many cores a run may use; thread environment of workers
* :mod:`.plan`     processes and threads per stage; the process-wide plan
* :mod:`.backends` ``map_tasks``: forked workers, largest task first

See ``pydftb/docs/design/M3a-dispatch.html``.
"""
from .backends import ForkBackend, SerialBackend, TaskResult, WorkerDied, default_backend, map_tasks
from .budget import CpuBudget, available_memory, detect_cpu, parse_ncores, thread_env
from .plan import Plan, StagePlan, core_shares, current_plan, set_plan, simple_plan

__all__ = ["ForkBackend", "SerialBackend", "TaskResult", "WorkerDied", "default_backend",
           "map_tasks", "CpuBudget", "available_memory", "detect_cpu", "parse_ncores",
           "thread_env", "Plan", "StagePlan", "core_shares", "current_plan", "set_plan",
           "simple_plan"]
