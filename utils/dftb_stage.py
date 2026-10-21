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
"""
from __future__ import annotations

import os
import shutil
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

_D_BLOCK = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81)) | set(range(104, 113))
_F_BLOCK = set(range(57, 72)) | set(range(89, 104))


def orbitals_estimate(numbers) -> int:
    """Basis size N for the cost estimate: s (H, He), sp, spd (d block), spdf (f block)."""
    return int(sum(1 if z <= 2 else 16 if z in _F_BLOCK else 9 if z in _D_BLOCK else 4
                   for z in numbers))


def kpoint_grid(atoms: Atoms, kpts_dens: float, bolt: float) -> list:
    """PTBP's k-point rule: Gamma only for structures of Boltzmann weight < 0.1."""
    from utils.calculator import monkhorstpack2kptdensity
    if bolt < 0.1:
        return [1, 1, 1]
    _, grid = monkhorstpack2kptdensity(atoms, kpts_dens)
    return grid


def irreducible_kpoints(grid) -> int:
    """k-points of a Monkhorst-Pack grid after the time-reversal reduction: k and -k
    pair up; only Gamma is its own partner, and it is on the grid when every n is odd."""
    if len(grid) != 3:
        return 1
    n = [int(k) for k in grid]
    return (n[0] * n[1] * n[2] + all(k % 2 for k in n)) // 2


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
    """(workers, threads, apply thread settings?) of the DFTB stage from the current plan."""
    plan = current_plan()
    if plan is None:
        return 1, 1, False
    st = plan.stage("dftb")
    return st.workers, st.threads, True


def run_dataset(structures, bolt, eos_points: int, kpts_dens: float, poly_rep: str = "Yes",
                SCC: str = "Yes"):
    """DFTB for every structure, in EOS groups -> (dftb Atoms, kept DFT structures,
    kept indices); groups with a missing SKF are dropped as before."""
    from utils.engines import settings
    eos = int(eos_points)
    ngroups = int(len(structures) / eos)
    used = list(structures[:ngroups * eos])
    workers, threads, planned = _stage_settings()
    engine = settings()["engine"]
    base = {"calc_type": "static", "poly_rep": poly_rep, "SCC": SCC, "slater_p": "./"}
    if planned and engine == "pydftb":
        base["k_threads"] = threads
    workers = min(workers, len(used)) if used else 1
    parallel = workers > 1
    if parallel and engine == "dftbplus":
        base["slater_p"] = os.path.abspath(".") + "/"
    tasks = make_tasks(used, bolt, kpts_dens, base,
                       task_root=os.path.abspath(TASK_DIR) if parallel and engine == "dftbplus"
                       else None)

    dftb_out, dft_out, kept = [], [], []
    if not parallel:
        with applied_env(thread_env(threads)) if planned else nullcontext():
            for n in range(ngroups):
                group = range(n * eos, (n + 1) * eos)
                try:
                    atoms = [run_structure(tasks[i]) for i in group]
                except FileNotFoundError as err:
                    print("DFTB+ Warning:", err)
                    print(SKF_WARNING)
                    continue
                dft_out.extend(used[i] for i in group)
                dftb_out.extend(atoms)
                kept.extend(group)
        return dftb_out, dft_out, kept

    order = sorted(range(len(tasks)), key=lambda i: (-tasks[i].cost, i))
    results = map_tasks(run_structure, tasks, workers, order=order, threads=threads)
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
