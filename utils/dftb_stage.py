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
  kept after a failure.  Serial DFTB+ runs stay in the current directory, as before;
* in PTBP runs with DFTB+ the configuration advisor (``utils.advisor_hook``, M3d) checks
  the first DFTB+ run once, and every failed run.

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

from utils import advisor_hook
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
    keep_dir: bool = False                          # keep workdir after success (advisor)
    grad: "GradPass | None" = None                  # gradient pass (M4c), pydftb only


@dataclass
class GradPass:
    """Gradient work done with the DFTB stage (milestone M4c, design section 6).

    Pass 1 (``weights`` None) is the stage itself: every structure is computed with
    ``PyDFTB.evaluate`` -- the engine's results, so the files do not change -- and also
    returns, for every tangent, Edot = <dE/dtheta, t> (eV per Bohr) and its converged
    input charges.  Pass 2 (``weights``: task index -> dmu/dF in 1/(eV/A) per component)
    returns <d(w . F)/dtheta, t> with the implicit mode, starting from ``q_init``.
    ``mode``: the mode of pass 1 ("hf" at T = 0; "implicit" with smearing, where mode hf
    has no dE).  torch runs on one thread, so the numbers do not depend on the cores.
    """
    tangents: dict                                   # name -> utils.hotcent_tangent.Tangent
    mode: str = "hf"
    weights: dict | None = None
    q_init: dict | None = None
    out: dict = field(default_factory=dict)          # task index -> {name: value}
    q_in: dict = field(default_factory=dict)         # task index -> q_in (pass 1)
    errors: dict = field(default_factory=dict)       # task index -> why it has no gradient

    def kinds(self) -> tuple:
        return tuple(sorted({k for t in self.tangents.values() for k in t.kinds()}))

    def active(self) -> dict:
        return {n: t for n, t in self.tangents.items() if t.kinds()}


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


def run_structure(task: StructureTask):
    """Energy and forces of one structure: the Atoms, results in a SinglePointCalculator
    (with a gradient pass: (Atoms, {"edot": {name: eV/Bohr}, "q_in": charges}))."""
    from utils.calculator import dftbplus_calculator
    atoms = Atoms(task.symbols, positions=task.positions)
    atoms.set_cell(task.cell)
    atoms.set_pbc(task.pbc)
    args = task.dftb_args if task.workdir is None else dict(task.dftb_args, directory=task.workdir)
    calc = dftbplus_calculator(atoms, args)
    atoms.calc = calc
    extra = None
    if task.grad is None:
        atoms.get_forces()
    else:
        extra = _energy_gradient(calc, atoms, task.grad)
    spc = SinglePointCalculator(atoms)
    spc.results = dict(calc.results)          # as computed: the extxyz output does not change
    atoms.calc = spc
    if task.workdir is not None and not task.keep_dir:
        shutil.rmtree(task.workdir, ignore_errors=True)
    return atoms if extra is None else (atoms, extra)


def _grad_dict(params, grads) -> dict:
    """{kind: {key: gradient array}} for the leaves of ``params`` (None: no dependence)."""
    out, it = {}, iter(grads)
    for kind, tensors in (("tables", params.tables), ("onsite", params.onsite),
                          ("hubbard", params.hubbard)):
        for key, x in tensors.items():
            if x.requires_grad:
                g = next(it)
                out.setdefault(kind, {})[key] = None if g is None else g.detach().numpy()
    return out


def _leaves(params) -> list:
    return [x for d in (params.tables, params.onsite, params.hubbard) for x in d.values()
            if x.requires_grad]


def _energy_gradient(calc, atoms, grad: GradPass) -> dict:
    """Pass 1 for one structure: the calculator's results as ``atoms.get_forces()`` sets
    them, and <dE/dtheta, t> for every active tangent (eV/Bohr)."""
    import torch
    from ase.calculators.calculator import CalculationFailed
    from ase.units import Hartree
    from utils.hotcent_tangent import contract
    torch_one_thread()                    # before any torch operation in this process
    active = grad.active()
    try:
        res, params = calc.evaluate(atoms, requires_grad=grad.kinds(), mode=grad.mode)
    except CalculationFailed:             # the engine failed: so would a run without gradient
        raise
    except Exception as err:              # noqa: BLE001  no gradient (an SCC that did not
        if getattr(err, "raw", None) is None:   # converge ...): the results as without one,
            atoms.get_forces()                  # kept from the same run when there was one
        return {"edot": None, "q_in": None, "error": f"{type(err).__name__}: {err}"}
    edot = {name: 0.0 for name in grad.tangents}
    if active:
        if not res.energy.requires_grad:              # e.g. no parameter enters the energy
            gd = {}
        else:
            gd = _grad_dict(params, torch.autograd.grad(res.energy, _leaves(params),
                                                        allow_unused=True))
        for name, t in active.items():
            edot[name] = contract(gd, t) * Hartree
    return {"edot": edot, "q_in": np.array(res.raw.q_in)}


def _force_gradient(task: StructureTask):
    """Pass 2 for one structure: <d(w . F)/dtheta, t> with F in eV/A (the units of w
    times eV/A, per Bohr), by the implicit mode from the charges of pass 1."""
    import torch
    from ase.units import Bohr, Hartree
    from utils.calculator import dftbplus_calculator
    from utils.hotcent_tangent import contract
    torch_one_thread()
    grad = task.grad
    atoms = Atoms(task.symbols, positions=task.positions)
    atoms.set_cell(task.cell)
    atoms.set_pbc(task.pbc)
    calc = dftbplus_calculator(atoms, task.dftb_args)
    active = grad.active()
    fdot = {name: 0.0 for name in grad.tangents}
    w = np.asarray(grad.weights[task.index], dtype=float)
    if not active or not np.any(w):
        return fdot
    res, params = calc.evaluate(atoms, requires_grad=grad.kinds(), mode="implicit",
                                forces=True, q_init=(grad.q_init or {}).get(task.index))
    # forces in eV/A = Hartree/Bohr x the engine's (Ha/Bohr)
    target = (torch.as_tensor(w * (Hartree / Bohr)) * res.forces).sum()
    if not target.requires_grad:
        return fdot
    gd = _grad_dict(params, torch.autograd.grad(target, _leaves(params), allow_unused=True))
    for name, t in active.items():
        fdot[name] = contract(gd, t)
    return fdot


def torch_one_thread() -> None:
    """torch on one thread in this process (before the DFTB stage forks; M4a fork rule)."""
    import torch
    torch.set_num_threads(1)


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
                SCC: str = "Yes", grad: GradPass | None = None):
    """DFTB for every structure, in EOS groups -> (dftb Atoms, kept DFT structures,
    kept indices); groups with a missing SKF are dropped as before.

    ``grad`` (M4c, pydftb only): also pass 1 of the gradient, collected in ``grad.out`` and
    ``grad.q_in`` for the kept structures; the results and files are the same."""
    from utils.engines import settings
    eos = int(eos_points)
    ngroups = int(len(structures) / eos)
    used = list(structures[:ngroups * eos])
    workers, threads, blas, planned = _stage_settings()
    engine = settings()["engine"]
    if grad is not None and engine != "pydftb":
        raise ValueError("the gradient of the DFTB stage needs engine='pydftb'")
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
    if grad is not None:
        torch_one_thread()                         # before the workers fork (M4a)
        for t in tasks:
            t.grad = grad

    def keep(i, value):
        """The Atoms of task i; pass 1 of the gradient goes to ``grad``."""
        if grad is None:
            return value
        atoms, extra = value
        grad.out[i] = extra["edot"]
        grad.q_in[i] = extra["q_in"]
        if extra.get("error"):
            grad.errors[i] = extra["error"]
        return atoms

    advise = engine == "dftbplus" and advisor_hook.wants_check()
    dftb_out, dft_out, kept = [], [], []
    if not parallel:
        walls = [0.0] * len(tasks)

        def timed(i):
            t0 = time.perf_counter()
            try:
                atoms = run_structure(tasks[i])
            except FileNotFoundError:
                raise
            except Exception as err:            # a DFTB+ failure: ask the advisor why
                if engine == "dftbplus":
                    advisor_hook.explain_failure(os.getcwd(), err)
                raise
            finally:
                walls[i] = time.perf_counter() - t0
            if advise:
                advisor_hook.check_once(os.getcwd())
            return atoms

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
                dftb_out.extend(keep(i, a) for i, a in zip(group, atoms))
                kept.extend(group)
        if len(kept) == len(tasks):
            _record(len(tasks), walls)
        return dftb_out, dft_out, kept

    order = task_order(tasks)
    if advise:                                   # keep one run directory for the advisor
        tasks[order[0]].keep_dir = True
    results = map_tasks(run_structure, tasks, workers, order=order, threads=threads,
                        blas_threads=blas)
    if all(r.ok for r in results):
        _record(len(tasks), [r.wall for r in results])
    if advise and results[order[0]].ok:
        advisor_hook.check_once(tasks[order[0]].workdir, concurrent=workers)
        shutil.rmtree(tasks[order[0]].workdir, ignore_errors=True)
    for n in range(ngroups):
        group = range(n * eos, (n + 1) * eos)
        failed = next((results[i] for i in group if not results[i].ok), None)
        if failed is not None:
            if isinstance(failed.error, FileNotFoundError):
                print("DFTB+ Warning:", failed.error)
                print(SKF_WARNING)
                continue
            if engine == "dftbplus":
                advisor_hook.explain_failure(tasks[failed.index].workdir, failed.error)
            raise failed.error
        dft_out.extend(used[i] for i in group)
        dftb_out.extend(keep(i, results[i].value) for i in group)
        kept.extend(group)
    _remove_if_empty(TASK_DIR)
    return dftb_out, dft_out, kept


def run_force_gradient(structures, bolt, kpts_dens: float, grad: GradPass, skf_dir: str,
                       poly_rep: str = "Yes", SCC: str = "Yes") -> dict:
    """Pass 2 of the gradient (M4c): {index: {name: <d(w . F)/dtheta, t>}} for the
    structures with weights in ``grad.weights`` (indices of ``structures``, as in pass 1),
    with the k-points of the DFTB stage and the SKF files in ``skf_dir`` (those of pass 1),
    distributed like the stage (largest first); a failure is raised."""
    workers, threads, blas, planned = _stage_settings()
    idx = sorted(grad.weights)
    base = {"calc_type": "static", "poly_rep": poly_rep, "SCC": SCC,
            "slater_p": os.path.join(os.path.abspath(skf_dir), "")}
    if planned:
        base["k_threads"] = threads
        if blas > 1:
            base["blas_threads"] = blas
    tasks = make_tasks([structures[i] for i in idx], [bolt[i] for i in idx], kpts_dens, base)
    for t, i in zip(tasks, idx):
        t.index, t.grad = i, grad
    if not tasks:
        return {}
    torch_one_thread()
    workers = min(workers, len(tasks))
    order = sorted(range(len(tasks)), key=lambda k: (-tasks[k].cost, k))
    results = map_tasks(_force_gradient, tasks, workers, order=order, threads=threads,
                        blas_threads=blas)
    for r in results:
        if not r.ok:
            raise r.error
    return {t.index: r.value for t, r in zip(tasks, results)}


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
