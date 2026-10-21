"""The DFTB stage of a BO step: one DFTB calculation per structure of the dataset.

Serial or in parallel (:mod:`utils.parallel`), with the same results and files:

* the structures go in EOS groups of ``eos_points`` (a remainder that does not fill a
  group is not computed, as before); a ``FileNotFoundError`` (an SKF that hotcent did
  not write) drops its whole group with the historical warning, any other exception
  is raised as it is;
* every calculation returns its Atoms with the calculator's results moved into a
  SinglePointCalculator, so ``dftb_band.xyz`` is byte-identical to the historical file;
* in a parallel stage each DFTB+ run gets a directory ``dftb_tasks/sNNNN/`` with the SKF
  directory as an absolute path; the directory is removed after a successful run and
  kept after a failure.  Serial DFTB+ runs stay in the current directory, as before.

Without a plan (:func:`utils.parallel.current_plan` is None, e.g. library use) the stage
runs serially with the thread settings untouched, i.e. exactly as before M3.

Tasks are handed out largest first (M3a): by the measured time of every structure in
the previous BO step when there is one (M3c), else by the plan's cost-model estimate,
else by k-points x orbitals^3.  The order only affects which process computes what.
"""
from __future__ import annotations

import os
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import ase.io
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from utils.parallel import current_plan, map_tasks
from utils.parallel.budget import applied_env, thread_env

TASK_DIR = "dftb_tasks"
SKF_WARNING = ("The problem caused by failed generation of SKF with this parameter. We will "
               "overlook these failed structures in this round of parameterization.")

# wall time of every structure in the last parallel or serial stage: (n tasks, walls)
_MEASURED: dict = {}


def orbitals_estimate(numbers) -> int:
    """Basis size N for the cost estimate: s (H, He), sp, spd (d block), spdf (f block)."""
    from pydftb.perfmodel import basis_size, block_shells
    return int(sum(basis_size(block_shells(z)) for z in numbers))


def kpoint_grid(atoms: Atoms, kpts_dens: float, bolt: float) -> list:
    """PTBP's k-point rule: Gamma only for structures of Boltzmann weight < 0.1."""
    from utils.calculator import monkhorstpack2kptdensity
    if bolt < 0.1:
        return [1, 1, 1]
    _, grid = monkhorstpack2kptdensity(atoms, kpts_dens)
    return grid


def irreducible_kpoints(grid) -> int:
    """k-points of a Monkhorst-Pack grid after the time-reversal reduction (perfmodel)."""
    from pydftb.perfmodel import irreducible_kpoints as _irreducible
    return _irreducible(grid)


@dataclass
class StructureTask:
    """One DFTB calculation; plain data, so it could also travel to another machine."""
    index: int
    symbols: list
    positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    dftb_args: dict = field(default_factory=dict)   # PTBP schema (utils.engines)
    workdir: str | None = None                      # DFTB+ task directory (parallel stage)
    cost: float = 1.0                               # nk * N^3, for the largest-first order


def make_tasks(structures, bolt, kpts_dens, dftb_args: dict, task_root: str | None = None):
    """Tasks for ``structures`` (k-point grid per structure as PTBP has always chosen it).

    ``task_root``: run each DFTB+ calculation in ``<task_root>/sNNNN`` (parallel stage).
    """
    tasks = []
    for i, (mole, b) in enumerate(zip(structures, bolt)):
        args = dict(dftb_args, kpts=kpoint_grid(mole, kpts_dens, b))
        cost = irreducible_kpoints(args["kpts"]) * orbitals_estimate(mole.numbers) ** 3
        tasks.append(StructureTask(i, mole.get_chemical_symbols(), mole.get_positions(),
                                   np.array(mole.get_cell()), mole.get_pbc().copy(), args,
                                   None if task_root is None else os.path.join(task_root, f"s{i:04d}"),
                                   float(cost)))
    return tasks


def run_structure(task: StructureTask) -> Atoms:
    """Energy and forces of one structure: the Atoms, results in a SinglePointCalculator."""
    from utils.calculator import dftbplus_calculator
    atoms = Atoms(task.symbols, positions=task.positions)
    atoms.set_cell(task.cell)
    atoms.set_pbc(task.pbc)
    args = task.dftb_args if task.workdir is None else dict(task.dftb_args, directory=task.workdir)
    calc = dftbplus_calculator(atoms, args)
    atoms.calc = calc
    atoms.get_forces()
    spc = SinglePointCalculator(atoms)
    spc.results = dict(calc.results)          # as computed: the extxyz output does not change
    atoms.calc = spc
    if task.workdir is not None:
        shutil.rmtree(task.workdir, ignore_errors=True)
    return atoms


def _stage_settings():
    """(workers, threads, BLAS threads, apply thread settings?) of the DFTB stage."""
    plan = current_plan()
    if plan is None:
        return 1, 1, 1, False
    st = plan.stage("dftb")
    return st.workers, st.threads, max(1, int(st.blas_threads)), True


def task_order(tasks) -> list:
    """Largest first: measured times of the last stage, else the plan's estimates, else
    the k-points x orbitals^3 of the task."""
    n = len(tasks)
    walls = _MEASURED.get(n)
    if walls is None:
        plan = current_plan()
        costs = list(getattr(plan, "costs", None) or [])
        walls = costs if len(costs) >= n else None
    key = (lambda i: (-walls[i], i)) if walls is not None else (lambda i: (-tasks[i].cost, i))
    return sorted(range(n), key=key)


def _record(n: int, walls) -> None:
    _MEASURED.clear()
    _MEASURED[n] = list(walls)


def measured_times(n: int):
    """Wall times of the ``n`` structures of the last DFTB stage (None: not measured)."""
    return _MEASURED.get(n)


def run_dataset(structures, bolt, eos_points: int, kpts_dens: float, poly_rep: str = "Yes",
                SCC: str = "Yes"):
    """DFTB for every structure, in EOS groups -> (dftb Atoms, kept DFT structures,
    kept indices); groups with a missing SKF are dropped as before."""
    from utils.engines import settings
    eos = int(eos_points)
    ngroups = int(len(structures) / eos)
    used = list(structures[:ngroups * eos])
    workers, threads, blas, planned = _stage_settings()
    engine = settings()["engine"]
    base = {"calc_type": "static", "poly_rep": poly_rep, "SCC": SCC, "slater_p": "./"}
    if planned and engine == "pydftb":
        base["k_threads"] = threads
        if blas > 1:                               # PTBP_BLAS_THREADS (opt-in)
            base["blas_threads"] = blas
    workers = min(workers, len(used)) if used else 1
    parallel = workers > 1
    if parallel and engine == "dftbplus":
        base["slater_p"] = os.path.abspath(".") + "/"
    tasks = make_tasks(used, bolt, kpts_dens, base,
                       task_root=os.path.abspath(TASK_DIR) if parallel and engine == "dftbplus"
                       else None)

    dftb_out, dft_out, kept = [], [], []
    if not parallel:
        walls = [0.0] * len(tasks)

        def timed(i):
            t0 = time.perf_counter()
            try:
                return run_structure(tasks[i])
            finally:
                walls[i] = time.perf_counter() - t0

        with applied_env(thread_env(threads)) if planned else nullcontext():
            for n in range(ngroups):
                group = range(n * eos, (n + 1) * eos)
                try:
                    atoms = [timed(i) for i in group]
                except FileNotFoundError as err:
                    print("DFTB+ Warning:", err)
                    print(SKF_WARNING)
                    continue
                dft_out.extend(used[i] for i in group)
                dftb_out.extend(atoms)
                kept.extend(group)
        if len(kept) == len(tasks):
            _record(len(tasks), walls)
        return dftb_out, dft_out, kept

    results = map_tasks(run_structure, tasks, workers, order=task_order(tasks), threads=threads,
                        blas_threads=blas)
    if all(r.ok for r in results):
        _record(len(tasks), [r.wall for r in results])
    for n in range(ngroups):
        group = range(n * eos, (n + 1) * eos)
        failed = next((results[i] for i in group if not results[i].ok), None)
        if failed is not None:
            if isinstance(failed.error, FileNotFoundError):
                print("DFTB+ Warning:", failed.error)
                print(SKF_WARNING)
                continue
            raise failed.error
        dft_out.extend(used[i] for i in group)
        dftb_out.extend(results[i].value for i in group)
        kept.extend(group)
    _remove_if_empty(TASK_DIR)
    return dftb_out, dft_out, kept


def write_outputs(dftb_atoms, dft_atoms, kept) -> None:
    """dft_final.xyz, dftb_band.xyz and search_index.txt in the current directory."""
    ase.io.write("dft_final.xyz", dft_atoms)
    ase.io.write("dftb_band.xyz", dftb_atoms)
    with open("search_index.txt", "w") as fh:
        fh.write(",".join(str(i) for i in kept))


def _remove_if_empty(path) -> None:
    try:
        Path(path).rmdir()
    except OSError:
        pass
