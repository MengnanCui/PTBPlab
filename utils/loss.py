
import copy
import json
import math
import os
import re

import ase
import numpy as np
from ase import Atoms

np.random.seed(66)
import random
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from ase.data import atomic_numbers, chemical_symbols, covalent_radii
from ase.units import *
from scipy.optimize import minimize
from skopt import gp_minimize, load
from skopt.space import Categorical, Integer, Real
from scipy.optimize import Bounds, basinhopping, brute, fmin


import multiprocessing as mp
import argparse
from argparse import RawTextHelpFormatter

import copy
import json
import math
import os
import pickle
import pprint as pp
import re
import time
from collections import OrderedDict
from contextlib import nullcontext
from math import ceil
from pathlib import Path
from typing import Tuple

import ase.io
from ase.data import atomic_numbers, chemical_symbols, covalent_radii
from ase.eos import EquationOfState as EOS
from ase.io import Trajectory
from ase.io.jsonio import read_json
from ase.units import Bohr, kB, kJ
from ase.utils.deltacodesdft import delta
from typing import List, Dict, Tuple
import multiprocessing as mp
from itertools import combinations



from math import pi, sqrt
from pathlib import Path
from utils.hotpy import hotpy  
from utils.calculator import dftbplus_calculator, calc_band_binary, delta_band_binary
import utils.dftb_stage as dftb_stage
from utils.parallel import current_plan, set_plan
from utils.parallel.budget import applied_env, thread_env
from math import pi, sqrt, floor
from tools.physrep import physrep
from utils.parameters_set import get_PTBP, get_QNplusRep, get_PRIOR
import utils.data as data
import utils.plot as plot



# Version of the loss definition stored next to skopt checkpoints (checkpoint.pkl.meta.json).
# 1: the penalty always used a covalent radius of 0.5 Bohr (the radius was never passed to
#    Loss); 2: the radius of the element being parametrised, as for the published PTBP losses;
# 3 (M5b): parameters per type (utils.typemodel): (r0_w, r0_d) of every parameter set for the
#    optimisers, a sigma_rep per set, the penalty the geometric mean of the sets' penalties; the
#    meta file records the types.  One parameter set for every type (one element, or
#    --shared-parameters) keeps the loss and the meta file of version 2, bit for bit.
# M5b-2: the meta file of either version may also hold "failed_evaluations", the indices of the
# failed BO points in the checkpoint's history (written only when there are some).  The loss
# version stays: successful losses are the same, only the failed points' values changed.
LOSS_VERSION = 3
SHARED_LOSS_VERSION = 2


# ---------------------------------------------------------------------------
# Failed evaluations (M5b-2).  A BO point whose loss cannot be computed on the whole dataset
# (hotcent or a DFTB run failed, some structures have no DFTB result, the loss is not finite)
# gets twice the worst successful loss this run has seen, so that the optimisers see it as
# worse than every real point.  Before M5b-2 it got a random integer 97-99, which is below the
# real losses of the reaction mode (150-5000 for the B2N2 + N3Na example), and a step with some
# structures missing was scored on the others only, which biases its loss low.
FAILURE_FACTOR = 2.0
# Before any success: energygeometry losses are Boltzmann-weighted averages over the EOS groups
# (M5b's Cu runs: 0.80-166 over 40 points), 1000 is about 6x the worst; dataset/reaction losses
# sum |dE| over the structures (the B2N2 + N3Na example, 22 structures: 147-5205, 6.7-237 per
# structure), 1000 per structure is about 4x the worst.
NO_SUCCESS_FAILURE = 1000.0
# What a failed point got before M5b-2: resumed checkpoints without a record of their failures
# (written before M5b-2) mark them with exactly these values (a real loss is never an integer).
OLD_FAILURE_VALUES = (97.0, 98.0, 99.0)


class LossFailure(RuntimeError):
    """A BO point without a loss of the whole dataset (M5b-2): no structure, or only some of
    them, have DFTB results, or the loss is not finite.  Loss.EnergyGeometry and
    EnergyGeometryTypes turn it, as the other failures, into the failure value."""


# ---------------------------------------------------------------------------
# Repeated work of the sigma_rep search (milestone M3b).  A BO step evaluates the
# repulsion for ~74 values of sigma_rep on the same DFTB results: the step's files
# are parsed once (keyed by their content) and the equation-of-state fits are
# cached by their input (the DFT side is the same for every sigma_rep).
_STEP_FILES: "OrderedDict[tuple, list]" = OrderedDict()
_EOS_FITS: "OrderedDict[tuple, tuple]" = OrderedDict()


def _read_step_file(path) -> list:
    """``ase.io.read(path, ':')``, parsed once per file content.  The Atoms are shared
    between calls: read them, do not modify them (copy first)."""
    import hashlib
    key = (os.path.realpath(path), hashlib.sha1(Path(path).read_bytes()).hexdigest())
    hit = _STEP_FILES.get(key)
    if hit is None:
        hit = ase.io.read(path, ':')
        _STEP_FILES[key] = hit
        while len(_STEP_FILES) > 8:
            _STEP_FILES.popitem(last=False)
    else:
        _STEP_FILES.move_to_end(key)
    return list(hit)


_PHYSREP: "OrderedDict[tuple, physrep]" = OrderedDict()


def _physrep(elements, scale_cov, k_xc, numbers=None) -> "physrep":
    """physrep(elements, rmin=0.5, rmax=20, npoints=700, ...), built once per parameters
    (a pure function of them; evaluated and written, never modified).  ``numbers``: the atomic
    number of every key when the keys are type labels (M5b)."""
    key = (repr(list(elements)), repr(list(scale_cov)), repr(list(k_xc)),
           None if numbers is None else repr(sorted(numbers.items())))
    pr = _PHYSREP.get(key)
    if pr is None:
        pr = physrep(elements=elements, rmin=0.5, rmax=20.0, npoints=700,
                     scale_cov=scale_cov, k_xc=k_xc, numbers=numbers)
        _PHYSREP[key] = pr
        while len(_PHYSREP) > 64:
            _PHYSREP.popitem(last=False)
    return pr


def _eos_fit(atoms_list) -> tuple:
    """(e0, v0, B, Bp) of a Birch-Murnaghan fit to energy and volume per atom, as
    ``Loss.fit`` computes them after the trajectory round trip (bit for bit: the
    trajectory stores the same doubles); RuntimeError when the fit fails."""
    vols = np.array([atoms.get_volume() / len(atoms) for atoms in atoms_list])
    ens = np.array([atoms.get_potential_energy() / len(atoms) for atoms in atoms_list])
    key = (vols.tobytes(), ens.tobytes())
    hit = _EOS_FITS.get(key)
    if hit is None:
        try:
            eos = EOS(list(vols), list(ens), 'birchmurnaghan')
            eos.fit(warn=False)
            e0, B, Bp, v0 = eos.eos_parameters
            hit = (e0, v0, B, Bp)
        except RuntimeError as err:
            hit = str(err)                    # a failed fit: raised anew each time
        _EOS_FITS[key] = hit
        while len(_EOS_FITS) > 512:
            _EOS_FITS.popitem(last=False)
    if isinstance(hit, str):
        raise RuntimeError(hit)
    return hit


def penalty_radius(dataset, opt_elem=()) -> float:
    """Covalent radius (Bohr) of the element whose confinement radii are optimised.

    ``opt_elem[0]`` (symbol or atomic number) in multi-element mode, otherwise the heaviest
    element of the dataset; the BO bounds and seed use the same radius.  The published PTBP
    losses (utils/parameters_set.py) equal (mu_V + mu_E)/2 * r0_penalty(r0_w, r0_d, radius)
    with this radius for every element.
    """
    if len(opt_elem) > 0:
        el = opt_elem[0]
        z = atomic_numbers[el] if isinstance(el, str) else int(el)
    else:
        zs = sorted({int(z) for atoms in dataset for z in atoms.numbers})
        if not zs:
            raise ValueError("cannot choose a penalty radius for an empty dataset")
        z = zs[-1]
    return covalent_radii[z] / Bohr


def r0_penalty(r0_w: float, r0_d: float, cov_radius: float) -> float:
    """Regularisation factor of the loss; its minimum, 4, is at r0_w = 2 r_cov, r0_d = 4 r_cov."""
    return ((r0_w / (2 * cov_radius) - 1)**2 + 1)**2 * ((r0_d / (4 * cov_radius) - 1)**2 + 2)**2


class Loss():
    """
    For storing multiple loss functions and related functions
    """

    def __init__(self,
                 args_input: argparse.Namespace=None,
                 dft_input: list=[],
                 bolt_f11:list=[],
                 opt_elem: list=[],
                 superposition: str='density',
                 cov_radius: float | None=None,
                 path:Path='./',
                 types=None,
                 ):
        self.args_input = args_input
        self.dft_input = dft_input
        self.bolt_f11 = bolt_f11
        self.superposition = superposition
        if cov_radius is None and len(dft_input) > 0:
            cov_radius = penalty_radius(dft_input, opt_elem)
        self.cov_radius = cov_radius
        self.path = path
        self.hp = hotpy()
        self.opt_elem = opt_elem
        # utils.loss_grad.GradRequest: the next evaluation also computes the gradient
        # (gradient optimisers, M4c); None: the loss exactly as before
        self.grad_request = None
        # utils.typemodel.TypeModel (M5b); None or a layout with one parameter set for every
        # type (TypeModel.legacy): the loss of before M5b, bit for bit
        self.types = types
        # failed evaluations (M5b-2): the worst successful loss of this run (None: no success
        # yet), the points that failed (x as the optimisers pass it, also those of a resumed
        # checkpoint), whether the last evaluation failed, and the failed indices of the last BO
        self.worst_success = None
        self.failed_points = []
        self.last_failed = None
        self.failed_indices = []
        # during Bayesian_optimization: the checkpoint's meta file and the index of the next
        # evaluation in its history, to record the failed ones there (None: no checkpoint)
        self._bo_log = None

    @property
    def typed(self) -> bool:
        """Parameters per type (M5b): several parameter sets, fixed types or type switching."""
        model = getattr(self, 'types', None)
        return model is not None and not model.legacy

    # ---- failed evaluations (M5b-2) ------------------------------------------------------
    def no_success_failure_value(self) -> float:
        """The failure value before the first success: NO_SUCCESS_FAILURE in energygeometry,
        NO_SUCCESS_FAILURE per structure of the loss in dataset/reaction (mu_E is a sum over
        the structures)."""
        mode = str(getattr(self.args_input, 'optimization_option', '') or '').lower()
        if mode in ('dataset', 'reaction'):
            eos = max(1, int(getattr(self.args_input, 'eos_points', 1) or 1))
            return NO_SUCCESS_FAILURE * max(1, int(len(self.dft_input) / eos) * eos)
        return NO_SUCCESS_FAILURE

    def failure_value(self) -> float:
        """The loss of a failed evaluation: FAILURE_FACTOR (2) x the worst successful loss this
        run has seen; before the first success, :meth:`no_success_failure_value`."""
        worst = getattr(self, 'worst_success', None)
        return self.no_success_failure_value() if worst is None else FAILURE_FACTOR * worst

    def _record_success(self, mu) -> None:
        mu = float(mu)
        worst = getattr(self, 'worst_success', None)
        self.worst_success = mu if worst is None else max(worst, mu)
        self.last_failed = False
        self._log_evaluation(False)

    def _record_failure(self, x, error) -> float:
        """Record the failed point ``x`` and return its loss (printed with the reason)."""
        worst = getattr(self, 'worst_success', None)
        mu = self.failure_value()
        if not hasattr(self, 'failed_points'):
            self.failed_points = []
        self.failed_points.append(tuple(float(v) for v in x))
        self.last_failed = True
        how = (f"{FAILURE_FACTOR:g} x the worst successful loss so far ({worst:.10g})"
               if worst is not None else "no successful evaluation yet: the fixed value of the "
               f"{getattr(self.args_input, 'optimization_option', '')} mode")
        print(f"[failure] {type(error).__name__}: the point's loss is {mu:.10g} ({how})")
        self._log_evaluation(True)
        return mu

    def _log_evaluation(self, failed: bool) -> None:
        """During Bayesian_optimization: the evaluations come in the order of the checkpoint's
        history (the seed, then one per BO step; a resumed run continues after the loaded
        points).  A failed one is written to the meta file's ``failed_evaluations`` at once,
        before skopt's CheckpointSaver stores the point (the failure values do not tell).
        Recorded here, not by an skopt callback: skopt pickles its callbacks into the checkpoint."""
        log = getattr(self, '_bo_log', None)
        if log is None:
            return
        index = log['next']
        log['next'] += 1
        if failed:
            log['failed'].append(index)
            with open(log['meta_path'], 'w') as fh:
                json.dump(dict(log['meta'], failed_evaluations=list(log['failed'])), fh)

    def _end_bo_log(self) -> None:
        """After Bayesian_optimization: ``failed_indices`` = the failed evaluations of its
        history (loaded ones included); stop recording."""
        log = getattr(self, '_bo_log', None)
        if log is not None:
            self.failed_indices = sorted(log['failed'])
        self._bo_log = None

    def _note_outcome(self, x, mu, failed: bool) -> None:
        """Record an evaluation made in another process (parallel_bo's forked workers)."""
        if failed:
            if not hasattr(self, 'failed_points'):
                self.failed_points = []
            self.failed_points.append(tuple(float(v) for v in x))
        else:
            worst = getattr(self, 'worst_success', None)
            self.worst_success = float(mu) if worst is None else max(worst, float(mu))

    def is_failure(self, x) -> bool:
        """Whether the point ``x`` failed in this run (or in the checkpoint it resumed)."""
        key = tuple(float(v) for v in x)
        return key in set(getattr(self, 'failed_points', ()))

    def _check_complete_results(self) -> None:
        """After the DFTB stage (cwd: the step directory): every structure the stage computes
        must have a result, else :class:`LossFailure` (M5b-2).  The stage drops the EOS groups
        whose SKF files hotcent did not write (search_index.txt lists the others); the loss of
        the remaining structures would be biased low (dataset/reaction: a sum over fewer
        structures; energygeometry: the Boltzmann factors of the dropped groups stay in the
        normalisation).  The structures past the last whole EOS group are never computed (as
        before): they are not missing."""
        eos = max(1, int(getattr(self.args_input, 'eos_points', 1) or 1))
        expected = int(len(self.dft_input) / eos) * eos
        text = Path("search_index.txt").read_text().strip()
        kept = {int(i) for i in text.split(',')} if text else set()
        missing = [i for i in range(expected) if i not in kept]
        if not missing:
            return
        from collections import Counter
        formulas = Counter(self.dft_input[i].get_chemical_formula() for i in missing)
        which = ", ".join(f"{f} x{n}" for f, n in sorted(formulas.items()))
        if len(missing) == expected:
            raise LossFailure(f"no structure has DFTB results at these parameters ({which})")
        raise LossFailure(f"{len(missing)} of {expected} structures have no DFTB results "
                          f"(missing SKF files: {which}); a loss of the others would be biased "
                          f"low, so the point counts as failed")

    def _resume_failures(self, x_iters, func_vals, meta):
        """The failed evaluations of a resumed checkpoint: (func_vals for the BO, their indices).

        ``failed_evaluations`` of the meta file when it has one (M5b-2); otherwise (a checkpoint
        written before M5b-2) the evaluations whose value is exactly 97, 98 or 99, the failure
        values of before (a real loss is never an integer), which get the failure value of this
        run so that the BO sees them as bad points.  The worst successful loss starts from the
        loaded successes."""
        recorded = meta.get("failed_evaluations") if isinstance(meta, dict) else None
        ys = [float(v) for v in func_vals]
        if recorded is not None:
            failed = sorted({int(i) for i in recorded if 0 <= int(i) < len(ys)})
            how = "recorded in the meta file"
        else:
            failed = [i for i, v in enumerate(ys) if v in OLD_FAILURE_VALUES]
            how = "values 97-99: a checkpoint written before M5b-2"
        bad = set(failed)
        ok = [v for i, v in enumerate(ys) if i not in bad]
        self.worst_success = max(ok) if ok else None
        if not hasattr(self, 'failed_points'):
            self.failed_points = []
        self.failed_points.extend(tuple(float(v) for v in x_iters[i]) for i in failed)
        if failed:
            print(f"[checkpoint] {len(failed)} failed evaluation(s) ({how}): {failed}")
        if recorded is None and failed:
            value = self.failure_value()
            for i in failed:
                ys[i] = value
            func_vals = ys
            print(f"[checkpoint] their loss is now {value:.10g} ("
                  + (f"{FAILURE_FACTOR:g} x the worst loaded success" if ok else
                     "no loaded success: the fixed value") + ")")
        if ok:
            print(f"[checkpoint] worst successful loss so far: {self.worst_success:.10g}")
        return func_vals, failed


    def penalty(self, r0_w: float, r0_d: float) -> float:
        """Regularisation factor for these confinement radii (see :func:`r0_penalty`)."""
        if self.cov_radius is None:
            raise ValueError("Loss has no penalty radius: pass cov_radius or a dataset")
        return r0_penalty(r0_w, r0_d, self.cov_radius)

    def get_forces(self, atoms:list, prefix:str):

            if prefix == '_':
                prefix = ''

            forces_label = f"{prefix}forces"
            forces = []
            for a in atoms:
                forces.extend(a.get_array(forces_label).flatten())

            return forces
    def get_elements_from_dataset(self, dataset, opt_elem):
        """utils.data.get_elements_from_dataset (this was an identical copy; M5b-3 fixed both:
        structures with different numbers of elements no longer make NumPy refuse a ragged
        array)."""
        return data.get_elements_from_dataset(dataset, opt_elem)


    def get_energy_per_atom(self, dft:list,dftb:list, bolt_fxx:list=None):
        """
        Calculate the energy per atom

        Note: bolt_fxx is the Boltzmann factor for each structure, it's length is the same as dftb
        provide it while you want to calculate specific structures
        None while you want to calculate all structures, then it will automatically find from files "search_index.txt"

        """
        dftb_energy_per_atom = [dftb_mole.info['energy']/ dftb_mole.get_global_number_of_atoms() for dftb_mole in dftb]
        dft_energy_per_atom  = [dft_mole.get_total_energy()/ dft_mole.get_global_number_of_atoms() for dft_mole in dft]
        min_num = np.argmin(dft_energy_per_atom); print('min_num:', min_num)

        dftb_energy_per_atom_relative = [dftb_energy_per_atom[i] - dftb_energy_per_atom[min_num] for i in range(len(dftb_energy_per_atom))]
        dft_energy_per_atom_relative  = [dft_energy_per_atom[i]  - dft_energy_per_atom[min_num] for i in range(len(dft_energy_per_atom))]

        # delta_energy_per_atom = np.mean(np.abs(np.array(dftb_energy_per_atom_relative) - np.array(dft_energy_per_atom_relative)) )
        if bolt_fxx == None: # which means we want to calculate all structures all at once
            with open ("search_index.txt", 'r') as search_index_file:
                search_index = search_index_file.read().strip().split(',')
            bolt_fxx = [self.bolt_f11[int(i)] for i in search_index]

        delta_energy_per_atom = np.sum(np.abs([dftb_energy_per_atom_relative[m] - dft_energy_per_atom_relative[m] for m in range(len(bolt_fxx)) if bolt_fxx[m] >= 0.1]) / np.sum([bolt_fxx[m] for m in range(len(bolt_fxx)) if bolt_fxx[m] >= 0.1]))
        # This is designed for normalizing structures in this subgroup

        # return mu_E
        return dftb_energy_per_atom_relative, dft_energy_per_atom_relative, delta_energy_per_atom
    

    def get_formation_energy(self, atoms:list, E0:dict, prefix:str):

            if prefix == '_':
                prefix = ''

            energy_label = f"{prefix}energy"

            def _energy_of(mole):
                # Robust to whether energy lives in atoms.info (legacy ASE
                # convention, mirrored back by utils._compat) or only in
                # atoms.calc.results (modern ASE singlepoint). Either ase.io
                # write+read or the inner DFTB+ call may strip one side.
                if energy_label in mole.info:
                    return mole.info[energy_label]
                if mole.calc is not None and energy_label in getattr(mole.calc, 'results', {}):
                    return mole.calc.results[energy_label]
                if not prefix:
                    try:
                        return mole.get_potential_energy()
                    except Exception:
                        pass
                raise KeyError(f"{energy_label!r} not found on Atoms.info, "
                               f".calc.results, nor as get_potential_energy()")

            if isinstance(E0, dict):
                print("Calculating formation energy by E0s...")
                elemental_numbers = [int(i) for i in E0.keys()]
                formation_energy = [
                    _energy_of(mole) - sum(E0[elem] * mole.get_atomic_numbers().tolist().count(elem)
                                           for elem in elemental_numbers)
                    for mole in atoms
                ]
                formation_energy_per_atom = [energy / mole.get_global_number_of_atoms()
                                             for energy, mole in zip(formation_energy, atoms)]

            elif E0 == 'ref.xyz':
                print("Calculating formation energy by reference structure...")
                ref = ase.io.read('ref.xyz')
                E0_energy = _energy_of(ref)
                formation_energy = [_energy_of(mole) - E0_energy for mole in atoms]
                formation_energy_per_atom = [energy / mole.get_global_number_of_atoms()
                                             for energy, mole in zip(formation_energy, atoms)]

            return formation_energy, formation_energy_per_atom
    
    

    def get_atomic_energy(self, numbers, ref_dataset=None):
        """DFTB+ atomic reference energies for each unique element.

        For an isolated atom in a 20 Å cubic box every neighbour is past
        physrep's cutoff, so the result depends on the band-structure SKFs
        (r0_w / r0_d) but not on sigma_rep — meaning we can cache it across
        the inner brute / L-BFGS-B sigma_rep scan within a single BO outer
        iteration and avoid 11+ redundant DFTB+ runs per element.

        Parameters
        ----------
        numbers : iterable[int] | iterable[ase.Atoms]
            Either a list of atomic numbers, or the dataset itself (Atoms
            list) — in the latter case the unique elements are extracted.
        ref_dataset : optional
            Kept for backwards compatibility; unused.
        """
        # Normalise `numbers` to a sorted list of unique atomic numbers.
        if numbers is None:
            unique_Z = []
        elif hasattr(numbers, '__iter__'):
            try:
                unique_Z = sorted({int(z) for z in numbers})
            except (TypeError, ValueError):
                # `numbers` was a dataset (Atoms list) — extract elements
                unique_Z = sorted({int(z) for atoms in numbers
                                   for z in atoms.numbers})
        else:
            raise TypeError(f"Cannot derive elements from {type(numbers)}")

        cache = getattr(self, '_E0_dftb_cache', None)
        if cache is not None and frozenset(unique_Z) in cache:
            return cache[frozenset(unique_Z)]

        dftb_args = {'kpts': [1, 1, 1],
                     'poly_rep': 'No',
                     'SCC': 'Yes',
                     'slater_p': './',
                     'calc_type': 'static'}

        E0s = {}
        # one-atom runs: serial, on one thread whenever the run has a parallel plan
        # (the same E0 for every --ncores)
        with applied_env(thread_env(1)) if current_plan() is not None else nullcontext():
            for Z in unique_Z:
                atoms = ase.Atoms(chemical_symbols[int(Z)], positions=[[0, 0, 0]])
                atoms.set_cell(np.eye(3) * 20.0)  # 20 Å cubic vacuum
                atoms.set_pbc([True, True, True])
                calc = dftbplus_calculator(atoms, dftb_args)
                atoms.calc = calc
                E0s[int(Z)] = atoms.get_potential_energy()

        if not hasattr(self, '_E0_dftb_cache'):
            self._E0_dftb_cache = {}
        self._E0_dftb_cache[frozenset(unique_Z)] = E0s

        print(f"[E0_dftb] Computed isolated-atom energies: {E0s}")
        return E0s


    def run_calculator_seperately(self, dft_incopy:Atoms):
        """
        DFTB energies and forces of the dataset (``utils.dftb_stage``), serial or in
        parallel according to the run's plan.  The dataset goes in EOS groups so that a
        few failed DFTB runs (missing SKF) do not ruin the whole BO step: such a group is
        left out of this round.  Writes dft_final.xyz, dftb_band.xyz, search_index.txt.
        """
        # poly_rep = 'Yes': switch polynomial repulsion (default 0.0 for hotcent skf generator)
        # SCC = 'Yes': switch the self charge consistent calculation(DFTB2)
        grad = getattr(self, 'grad_request', None)
        dftb_bandstructure_final, dft_subincopy_final, kept_indices = dftb_stage.run_dataset(
            dft_incopy, self.bolt_f11, self.args_input.eos_points, self.args_input.kpt_density,
            poly_rep='Yes', SCC='Yes', grad=None if grad is None else grad.grad_pass)
        dftb_stage.write_outputs(dftb_bandstructure_final, dft_subincopy_final, kept_indices)
        return dftb_bandstructure_final, dft_subincopy_final

    def set_parameters(self, conf_parameters:list):
        """
        Set parameters as the input for Hotcent and Physrep

        With parameters per type (M5b, ``self.typed``) ``conf_parameters`` is the vector of the
        optimisers, (r0_w, r0_d) of every parameter set, and the result is
        ({label: [r0_w, r0_d]}, {label: p}) for every type (utils.typemodel); otherwise
        [r0_w, r0_d, p] for every element, as before.
        """
        if self.typed:
            return self.types.confinement(conf_parameters)

        kxc = 0.0
        r0, r0_dens, p = conf_parameters

        # opt_elem may hold symbols (the CLI's --multi_element) or atomic numbers
        opt_numbers = [atomic_numbers[e] if isinstance(e, str) else int(e) for e in self.opt_elem]
        atomic_numbers_, atomic_numbers_exclusive, atomic_symbols, atomic_symbols_exclusive = self.get_elements_from_dataset(self.dft_input, opt_numbers)

        if self.args_input.multi_element is None:
            #! For Single element system
            confinement_r0 = {s: [r0, r0_dens] for s in atomic_symbols}
            confinement_sigma = {s: p for s in atomic_symbols}

        else:
            #! For Binary system
            # The other elements keep PTBP's published radii (the table's keys are r0_w, r0_d;
            # this used to read 'r_wave', 'r_dens', which do not exist)
            confinement_r0 = {s: [get_PTBP()[s]['r0_w'], get_PTBP()[s]['r0_d']] for s in atomic_symbols_exclusive}
            confinement_r0[chemical_symbols[opt_numbers[0]]] = [r0, r0_dens]
            confinement_sigma = {s: p for s in atomic_symbols}

        return confinement_r0, confinement_sigma


    def collectAndcalculate_muEF(self, dft_incopy:list, dftb_full:list, sigma_rep:float, r0_w:float, r0_d:float):

        # Compute DFTB+ atomic reference energies for every unique element in
        # the dataset (cached across the inner brute scan within this BO iter).
        E0_dftb = self.get_atomic_energy(dft_incopy)

        _, dftb_energy_per_atom_relative = self.get_formation_energy(dftb_full, E0_dftb, '_')
        _, dft_energy_per_atom_relative = self.get_formation_energy(dft_incopy, self.args_input.E0s, '_')

        diffs = np.abs(np.array(dftb_energy_per_atom_relative) - np.array(dft_energy_per_atom_relative))
        delta_energy_per_atom = np.sum(diffs)

        mu_E = delta_energy_per_atom

        dft_forces = self.get_forces(dft_incopy, '_')
        dftb_forces = self.get_forces(dftb_full, '_')

        # RMSE & MAE for forces
        forces_rmse = np.sqrt(np.mean(np.square(np.array(dft_forces) - np.array(dftb_forces))))
        forces_mae = np.mean(np.abs(np.array(dft_forces) - np.array(dftb_forces)))

        mu_F = forces_mae
        # Penalty for overlarge parameters (regularization)
        penalty = self.penalty(r0_w, r0_d)

        c = 0.1

        mu = (mu_E * (1-c) + mu_F * c) * penalty

        with  open(self.args_input.log_file, 'a') as output:
            output.write(f"{r0_w:15.8f}, {r0_d:15.8f}, {sigma_rep[0]:15.8f}, {0.0:20.8e}, {self.cov_radius:15.8f}, {mu_E:15.8f}, {mu_F:20.8f}, {0.0: 20.8f}, {penalty:15.5f}, {mu:25.11f}\n")

        # Reaction mode: report per-formula |E_dft - E_dftb| breakdown alongside
        # the aggregate loss so the user can spot which compositions are well-fit
        # vs which are dragging the score.
        if self.args_input.optimization_option.lower() == 'reaction':
            from collections import defaultdict as _dd
            buckets = _dd(list)
            for atoms, d in zip(dft_incopy, diffs):
                buckets[atoms.get_chemical_formula()].append(float(d))
            with open('reaction_per_formula.csv', 'a') as fh:
                fh.write(f"# r0_w={r0_w:.6f} r0_d={r0_d:.6f} sigma_rep={sigma_rep[0]:.6f} mu={mu:.6e}\n")
                fh.write("formula,n_structures,mean_abs_dE_per_atom,max_abs_dE_per_atom\n")
                for formula in sorted(buckets):
                    vals = np.array(buckets[formula])
                    fh.write(f"{formula},{len(vals)},{vals.mean():.6e},{vals.max():.6e}\n")

        return mu



    # ---------------------------------------------------------------
    # EOS helpers (ported from tools/hotpy.py + utils/calculator.py)
    # ---------------------------------------------------------------
    def mk_traj(self, atoms: Atoms, traj_label: str, default_eos_points: int) -> None:
        """Write one .traj per phase (atoms are flat-concatenated, eos_points each)."""
        num_phases = int(len(atoms) / default_eos_points)
        for n in range(num_phases):
            traj = Trajectory('{}_{}.traj'.format(traj_label, n), 'w')
            sub_atoms = atoms[n * default_eos_points: (n + 1) * default_eos_points]
            for a in sub_atoms:
                traj.write(a)

    def fit(self, na: str):
        """Fit a Birch-Murnaghan EOS to the configurations in `<na>.traj`."""
        V, E = [], []
        for atoms in ase.io.read('{}.traj@:'.format(na)):
            V.append(atoms.get_volume() / len(atoms))
            E.append(atoms.get_potential_energy() / len(atoms))
        eos = EOS(V, E, 'birchmurnaghan')
        eos.fit(warn=False)
        e0, B, Bp, v0 = eos.eos_parameters
        return e0, v0, B, Bp

    def eos_atoms(self, atoms_input, default_eos_points: int, calc_label: str, files=None):
        """Run EOS fit on each phase, return dict of {label_n: {energy,volume,B,Bp}}.

        Writes ``<label>_<n>.traj`` per phase and ``<calc_label>.json``.  With ``files``
        (a dict) the fits use the structures in memory and the files are only recorded
        there, for :meth:`write_eos_files` (bit for bit the same numbers and files).
        """
        if isinstance(atoms_input, str):
            atoms_list = ase.io.read(atoms_input, ':')
        else:
            atoms_list = atoms_input

        num_phases = int(len(atoms_list) / default_eos_points)
        atomic_nums = []
        for a in atoms_list:
            atomic_nums.extend(a.numbers)
        atomic_nums = np.unique(atomic_nums)
        atomic_syms = [chemical_symbols[i] for i in atomic_nums]
        traj_label = f"eos_{'_'.join(atomic_syms)}"

        if files is None:
            self.mk_traj(atoms_list, traj_label, default_eos_points)

        out = {}
        for n in range(num_phases):
            na = traj_label + '_' + str(n)
            phase = atoms_list[n * default_eos_points: (n + 1) * default_eos_points]
            try:
                if files is None:
                    e0, v0, B, Bp = self.fit(na)
                else:
                    files[f'{na}.traj'] = list(phase)
                    e0, v0, B, Bp = _eos_fit(phase)
            except RuntimeError:
                e0, v0, B, Bp = 1000, 1000, 1000, 1000
            out[na] = {f'{calc_label}_energy': e0,
                       f'{calc_label}_volume': v0,
                       f'{calc_label}_B': B,
                       f'{calc_label}_Bp': Bp}
            print(f"\n{na}\n{out[na]}")
        if files is None:
            Path(f'{calc_label}.json').write_text(json.dumps(out))
        else:
            files[f'{calc_label}.json'] = json.dumps(out)
        return out

    def write_eos_files(self, files: dict) -> None:
        """Write the files recorded by ``eos_atoms(..., files=...)``: trajectories as
        ``mk_traj`` writes them, JSON as text."""
        for name, content in files.items():
            if name.endswith('.traj'):
                traj = Trajectory(name, 'w')
                for a in content:
                    traj.write(a)
                traj.close()
            else:
                Path(name).write_text(content)

    def eos_cal(self, dataset, bolt_factor: list):
        """Compute EOS-based loss components: (delta_tot, mu_v_tot, mu_logb_tot).

        delta_tot = Σ_phase  Boltzmann(phase) * (Δv * Δlog10B),
        with Δv  = |v0_dftb / v0_dft - 1|,
             ΔlogB = |log10(b_dftb) / log10(b_dft) - 1| + 1.
        Phases with bolt_factor < 0.1 are skipped.
        """
        for name in ['volume', 'B', 'Bp']:
            with open(name + '.csv', 'w') as f:
                print('#symbol, dft, dftb', file=f)
                for symbol, dct in dataset.items():
                    values = [dct[code + '_' + name] for code in ['dft', 'dftb']]
                    if name == 'B':
                        values = [val * 1e24 / kJ for val in values]
                    print('{},'.format(symbol),
                          ', '.join('{:.2f}'.format(v) for v in values), file=f)

        delta_tot, mu_v_tot, b_tot, n_i = 0.0, 0.0, 0.0, 0
        bolt_left = []
        with open('delta.csv', 'w') as f:
            print('#symbol,delta(dft-dftb),bolt_factor,delta*bolt', file=f)
            for symbol, dct in dataset.items():
                v0_dftb = dct['dftb_volume']
                b_dftb = dct['dftb_B'] * 1e24 / kJ
                v0_dft = dct['dft_volume']
                b_dft = dct['dft_B'] * 1e24 / kJ

                mu_v = abs(v0_dftb / v0_dft - 1)
                if math.isnan(mu_v):
                    mu_v = 100

                try:
                    mu_logb = abs(np.log10(b_dftb) / np.log10(b_dft) - 1) + 1
                except ZeroDivisionError:
                    mu_logb = 100
                if math.isnan(mu_logb):
                    mu_logb = 100

                delta_eos = mu_v * mu_logb
                delta_eos_bolt = delta_eos * bolt_factor[n_i]

                if bolt_factor[n_i] < 0.1:
                    n_i += 1
                    continue
                mu_v_tot += mu_v * bolt_factor[n_i]
                b_tot += mu_logb * bolt_factor[n_i]
                delta_tot += delta_eos_bolt
                bolt_left.append(bolt_factor[n_i])
                print('{}, {:5.2f}, {:5.2f}, {:5.2f}'.format(
                    symbol, delta_eos, bolt_factor[n_i], delta_eos_bolt), file=f)
                n_i += 1

        return delta_tot, mu_v_tot, b_tot

    # ---------------------------------------------------------------

    def collectAndcalculate_muEV(self, dft_incopy:Atoms, sigma_rep, r0_w, r0_d, bolt_f) -> float:

        """
        Collected energy forces and calculate mu_E and mu_V

        Parameters
        ------
        dft_incopy: Atoms
            Atoms object of the whole dataset
        
        sigma_rep: list
            Parameter for repulsive potential
            eg: sigma_rep = array([0.4321])
        
        r0_w: float
            Parameter for wavefunction cutoff

        r0_d: float
            Parameter for density function cutoff
        
        """

        # Read dataset
        num_parall = int(len(dft_incopy) / self.args_input.eos_points)

        # The EOS fits use the structures in memory; the files eos_atoms writes on the
        # way (trajectories, dft.json, dftb.json) are written once at the end with the
        # contents the repeated writes used to leave (M3b).
        eos_files = {}
        dftb_full, mu_Es, mu_Vs, mu_logbs = [], [], [], []
        for num in range(num_parall):
            dft_sub = dft_incopy[num*self.args_input.eos_points:(num+1)*self.args_input.eos_points]
            dftb_sub = ase.io.read(f'dftb_sub_{num}.xyz', ':')
            dftb_full.extend(dftb_sub)

            eos_dftb = self.eos_atoms(dftb_sub, self.args_input.eos_points, 'dftb', files=eos_files)
            eos_dft  = self.eos_atoms(dft_sub, self.args_input.eos_points, 'dft', files=eos_files)
            print(f"\neos_dftb: {eos_dftb}\neos_dft: {eos_dft}")


            # Calculate the mu_V
            [eos_dft[conf].update(eos_dftb[conf]) for conf in eos_dftb] # input eos of dft
            mu_cal, mu_V, mu_logb = self.eos_cal(eos_dft, [bolt_f[num]]) 
            mu_Vs.append(mu_V); mu_logbs.append(mu_logb)
            
            mu_Es.append(dftb_sub[0].info['mu_E'] * bolt_f[num]) 

        bolt_left = np.sum([bolt_f[i] for i in range(len(bolt_f)) if bolt_f[i] >= 0.1])
        mu_E, mu_V, mu_logb = np.sum(mu_Es)/bolt_left, np.sum(mu_Vs)/bolt_left, np.sum(mu_logbs)/bolt_left

        self.write_eos_files(eos_files)
        ase.io.write('dftb_full.xyz', dftb_full)
        os.system("rm dftb_sub_*.xyz")

        # Penalty term(regularization)
        penalty = self.penalty(r0_w, r0_d)


        # Combine mu_E and mu_V
        c = 0.5
        mu = (c * mu_V + (1 - c) * mu_E) * penalty 


        # Log the results
        sigma_rep, kxc = sigma_rep[0], 0.0
        with open(self.args_input.log_file, 'a') as output:
            output.write(f'{r0_w:15.8f}, {r0_d:15.8f}, {sigma_rep:15.8f}, {kxc:20.8e}, {self.cov_radius:15.8f}, {mu_E:15.8f}, {mu_V:20.8f}, {mu_logb: 20.8f}, {penalty:15.5f}, {mu:25.11f}\n')


        return mu

    def generate_repulsion(self, input_parameters) -> list:

        """
        Generate repulsive potential for each structure, for parallelization

        Parameters
        ------
        input_parameters: list
            List of parameters for repulsive potential
            eg: input_parameters = [n, dftb_bandstructure, dft_sub, atomic_subnumbers, para_rep_tot, kxc_tot, write_rep]
        
        """

        # Log
        n, dftb_bandstructure, dft_sub, atomic_subnumbers_tot, para_rep_tot, kxc_tot, write_rep = input_parameters
        print(f"\n> Subparallel Physrep Info: {n}\n  atomic_subnumbers_tot:{atomic_subnumbers_tot}\n  para_rep_tot:{para_rep_tot}\n  kxc_tot:{kxc_tot}\n len(dftb_band):{len(dftb_bandstructure)}\n len(dft_sub):{len(dft_sub)}\n write_rep:{write_rep}")


        pr = _physrep(atomic_subnumbers_tot, para_rep_tot, kxc_tot)
        
        if write_rep:
            if self.args_input.multi_element is not None:
                pr.write_specific_skf(self.opt_elem, 'ptbp')
                print("Write specific SKF for Binarys")
            else:
                pr.write_skf('ptbp')
                print("Write SKF for Elementary")

        
        atoms_total = data.get_full_dftb(dftb_bandstructure, dft_sub, pr, True)


        # Incase some structures faild in band energy calculation part
        with open ("search_index.txt", 'r') as search_index_file:
            search_index = search_index_file.read().strip().split(',')
            bolt_f11_final = [self.bolt_f11[int(i)] for i in search_index]


        dftb_energy_per_atom_relative, dft_energy_per_atom_relative, delta_energy_per_atom = self.get_energy_per_atom(dft_sub, atoms_total, bolt_f11_final[n*self.args_input.eos_points:(n+1)*self.args_input.eos_points])
        for atoms in atoms_total:
            atoms.info['mu_E'] = delta_energy_per_atom
    

        ase.io.write(f'dftb_sub_{n}.xyz', atoms_total)
        print(f"Write dftb_sub_{n}.xyz")


        return atoms_total, delta_energy_per_atom



    def parallel_repulsion(self, sigma_rep:list, r0_w:float, r0_d:float, write_rep:bool=False):
        """
        Calculator of repulsion potential, which is parallelized respect to number of structures

        Parameters
        ------
        sigma_rep: list
            Parameter for repulsive potential
            eg: sigma_rep = array([0.4321])
        
        r0_w: float
            Parameter for wavefunction cutoff
        
        r0_d: float
            Parameter for density function cutoff
        
        write_rep: bool
            Whether write the repulsive potential to SKF file

        """

        sigma_rep = sigma_rep.tolist()

        # Prepare data (the step's files, parsed once per BO step; nothing below modifies
        # these structures, so they are not copied)
        dftb_bandstructure_final = _read_step_file('dftb_band.xyz')
        dft_final = _read_step_file('dft_final.xyz')
        dft_incopy = dft_final


        num_parall = int(len(dft_final) / self.args_input.eos_points)
        group_elements = [self.get_elements_from_dataset(
            dft_incopy[num*self.args_input.eos_points:(num+1)*self.args_input.eos_points],
            self.opt_elem)[:2] for num in range(num_parall)]
        # Every group used to write the SKF of its elements: with the same elements in all
        # groups the files are the same each time, and writing them once leaves the same
        # files (groups with different elements write different files: keep every write)
        same_files = all(repr(g) == repr(group_elements[0]) for g in group_elements)
        for num in range(num_parall):
            dft_sub = dft_incopy[num*self.args_input.eos_points:(num+1)*self.args_input.eos_points]
            dftb_bandstructure = dftb_bandstructure_final[num*self.args_input.eos_points:(num+1)*self.args_input.eos_points]
            atomic_subnumbers, atomic_subnumbers_exclusive = group_elements[num]


            if self.args_input.multi_element is None:
                # one parameter set for every element (one element, or --shared-parameters):
                # every element of the group gets the sigma being searched, as
                # unparallel_repulsion broadcasts it (M5b-3; a group of several elements, e.g. an
                # EOS scan of B2N2, used to give physrep one sigma for two elements: IndexError)
                para_rep_tot = [float(np.atleast_1d(sigma_rep)[0])] * len(atomic_subnumbers)
            else:
                para_rep_tot = [get_PTBP()[chemical_symbols[n]]['sigma_rep'] for n in atomic_subnumbers_exclusive] + sigma_rep

            kxc_tot = [0.0] * len(atomic_subnumbers)

            
            # One group at a time, in this process (M3b: the process pool used to get a
            # single task per group and cost more than the work)
            write_now = write_rep and (not same_files or num == num_parall - 1)
            self.generate_repulsion((num, dftb_bandstructure, dft_sub, atomic_subnumbers,
                                     para_rep_tot, kxc_tot, write_now))

        # Judge which loss function to use

        bolt_f = self.bolt_f11[floor(self.args_input.eos_points/2)::self.args_input.eos_points]
        mu = self.collectAndcalculate_muEV(dft_incopy, sigma_rep, r0_w, r0_d, bolt_f)

        return mu



    def unparallel_repulsion(self, sigma_rep:list, r0_w:float, r0_d:float, write_rep:bool=True):

        """
        Calculator of repulsion potential

        Parameters
        ------
        sigma_rep: list
            Parameter for repulsive potential
            eg: sigma_rep = array([0.4321])
        
        r0_w: float
            Parameter for wavefunction cutoff
        
        r0_d: float
            Parameter for density function cutoff
        
        write_rep: bool
            Whether write the repulsive potential to SKF file

        """
        
        # Read dataset from band energy calculation (parsed once per BO step)
        dftb_bandstructure_final = _read_step_file('dftb_band.xyz')
        dft_final = _read_step_file('dft_final.xyz')
        dft_incopy = copy.deepcopy(dft_final)


        atomic_numbers_tot, atomic_subnumbers_exclusive , _, _ = self.get_elements_from_dataset(dft_incopy, self.opt_elem)
        # The outer brute / L-BFGS-B in Repulsion_optimization gives a single
        # scalar sigma_rep. For datasets that span multiple elements (i.e.
        # dataset / reaction mode), broadcast that scalar to all elements so
        # physrep can build one repulsive spline per element.
        scale_cov_per_elem = [float(np.atleast_1d(sigma_rep)[0])] * len(atomic_numbers_tot)
        kxc_tot = [0.0] * len(atomic_numbers_tot)


        # Repuslive potential Calculator
        pr = _physrep(atomic_numbers_tot, scale_cov_per_elem, kxc_tot)
        

        # implement repulsive potential to dftb_bandstructure part 
        dftb_full = data.get_full_dftb(dftb_bandstructure_final, dft_incopy, pr, True)

        ase.io.write('dftb_full.xyz', dftb_full)

        if write_rep:
            if self.args_input.multi_element is not None:
                pr.write_specific_skf(self.opt_elem, 'ptbp')
                print("Write specific SKF for Binarys")
            else:
                pr.write_skf('ptbp')
                print("Write SKF for Elementary")

        mu = self.collectAndcalculate_muEF(dft_incopy, dftb_full, sigma_rep, r0_w, r0_d)

        return mu  
    



    def Repulsion_optimization(self, conf_parameters:list, c0:float, c1:float) -> list:
        """
        Ready for the optimization of repulsion parameters sigma_rep
        
        Parameters
        -------
        conf_parameters: list
            List of parameters to be optimized
            eg: conf_parameters = [r0_w, r0_d, p]
            V_conf = (r_cov/r0)^p

        c0: float
            Lower boundary for sigma
        
        c1: float
            Upper boundary for sigma

        Note that please set a bigger upper boundary for Hydrogen eg. c1 = 1.5

        """
        # Set search boundary
        r0_w, r0_d, p = conf_parameters
        srange = (c0, c1)
        bounds = (Bounds(srange[0], srange[1], keep_feasible=True))

        
        # # Search time 1: Gride search
        conf_args = (r0_w, r0_d, True)
        if self.args_input.optimization_option.lower() in ('dataset', 'reaction'):
            res_brute = brute(self.unparallel_repulsion, (srange,), Ns=11, full_output=True,  disp=True, workers=1, args=conf_args)

        elif self.args_input.optimization_option.lower() == 'energygeometry':
            res_brute = brute(self.parallel_repulsion, (srange,), Ns=11, full_output=True,  disp=True, workers=1, args=conf_args)

        print(f"Search time 1: sigma_rep = {res_brute[0]}, mu = {res_brute[1]}")


        # Search time 2: L-BFGS-B
        # scipy >=1.11 makes brute return res_brute[0] as a 0-d ndarray; wrap to a
        # plain 1-d list so minimize's x0-shape check accepts it.
        x0_bfgs = [float(np.atleast_1d(res_brute[0])[0])]
        if self.args_input.optimization_option.lower() in ('dataset', 'reaction'):
            res_bfgs = minimize(self.unparallel_repulsion, x0_bfgs, method='L-BFGS-B',bounds=bounds, options={'disp':True,'maxfun':20,'ftol':1e-3}, args=conf_args)

        elif self.args_input.optimization_option.lower() == 'energygeometry':
            res_bfgs = minimize(self.parallel_repulsion, x0_bfgs, method='L-BFGS-B',bounds=bounds, options={'disp':True,'maxfun':20,'ftol':1e-3}, args=conf_args)


        sigma_rep = res_bfgs.x[0]
        mu = res_bfgs.fun
        print(f"Search time 2: sigma_rep = {sigma_rep}, mu = {mu}")


        return sigma_rep, mu






    def EnergyGeometry(self, conf_parameters:list,
                       xc:str,
                       superposition:str,
                    ) -> float:
        """
        Calculate the loss function for Energy and Geometry

        Parameters:
        -------
        conf_parameters: list
            List of parameters to be optimized
            eg: conf_parameters = [r0_w, r0_d, p]
            V_rep = (r_cov/r0)^p
        
        xc: str
            Exchange and Correlation functional
            eg: xc = 'GGA_X_PBE+GGA_C_PBE'

        
        superposition: str
            Density or Potential superposition
            default: 'Density' for PTBP, 'Potential' for QN(QUASINANO2013)+rep
        """

        kxc = 0.0
        r0_w, r0_d, p = conf_parameters

        # SKF parameters changed → invalidate the per-iter E0_dftb cache so
        # the inner brute scan recomputes against the new band structure.
        self._E0_dftb_cache = {}

        grad = getattr(self, 'grad_request', None)
        opt_directory = self.path / self.args_input.results_dir / (
            f"opt_{r0_w:.3f}_{r0_d:.3f}" if grad is None else grad.directory_name(r0_w, r0_d))
        opt_directory.mkdir(exist_ok=True, parents = True)

        os.chdir(opt_directory)


        #1. Generate SKF 
        t_ini = perf_counter()
        confinement_r0, confinement_p = self.set_parameters(conf_parameters)

        print(f"Confinement_r0:{confinement_r0}, Confinement_p:{confinement_p}")

        self.hp.set_variables(confinement_r0,
                        confinement_p, 
                        xc=xc,
                        superposition=self.args_input.superposition,
                        )

        try:
            # Write  SKF file
            self.hp.full_hotcent(superposition=superposition,
                            opt_elem=self.opt_elem,
                            )        

            t_hotcent = perf_counter()
            if grad is not None:
                grad.after_hotcent(self, conf_parameters)     # tangents (own timings)
                t_ini += perf_counter() - t_hotcent           # stage times: the BO step's
                t_hotcent = perf_counter()

        
        #2. DFTB Calculation for Band Structure Energy term
            dft_incopy = copy.deepcopy(self.dft_input)

            # Calculate energy/forces for the band energy part  
            self.run_calculator_seperately(dft_incopy)
            # M5b-2: structures without DFTB results (hotcent wrote no SKF for them) make the
            # point a failure, as when none has results
            self._check_complete_results()

            # dft_subincopy_final = ase.io.read('dft_final.xyz', ':'); dftb_bandstructure_final = ase.io.read('dftb_band.xyz', ':')
            t_dftb = perf_counter()


        #3. Parameterization for the Repulsion term, which is parallelized respect to
            sigma_rep_best, mu = self.Repulsion_optimization(conf_parameters, 0.1, 0.8)
            if not math.isfinite(float(mu)):                  # M5b-2: a failure, not a loss
                raise LossFailure(f"the loss is {mu}")
            # Downstream print/plot/par.out blocks read the bare `sigma_rep`
            # name; alias it to the optimised scalar so they don't NameError.
            sigma_rep = float(np.atleast_1d(sigma_rep_best)[0])


            # Write REP into SKF file
            # Calculate repulsive potential based on the optimized parameter
            # tot_para_rep = para_rep + [sigma_rep]
            atomic_numbers, atomic_numbers_exclusive, atomic_symbols, atomic_symbols_exclusive = \
                data.get_elements_from_dataset(self.dft_input, self.opt_elem)
            if self.args_input.multi_element is None:
                # Elementary / dataset / reaction modes: broadcast the brute-
                # optimised scalar sigma_rep to every element in the dataset
                # so physrep can build one repulsive spline per element.
                tot_para_rep = [float(np.atleast_1d(sigma_rep_best)[0])] * len(atomic_numbers)
            else:
                tot_para_rep = [get_PTBP()[chemical_symbols[n]]['sigma_rep']
                                for n in atomic_numbers_exclusive] + [float(np.atleast_1d(sigma_rep_best)[0])]

            pr = physrep(atomic_numbers,
                        rmin=0.5,
                        rmax=20.0,
                        npoints=700,
                        scale_cov=tot_para_rep,
                        k_xc=[0.0] * len(atomic_numbers),
                        )

            if self.args_input.multi_element is not None:
                pr.write_specific_skf(self.opt_elem, 'ptbp')
            else:
                pr.write_skf('ptbp')


            dft_subincopy_final = ase.io.read('dft_final.xyz', ':'); dftb_bandstructure_final = ase.io.read('dftb_band.xyz', ':')
            dftb_full = data.get_full_dftb(dftb_bandstructure_final, dft_subincopy_final, pr, True)
            ase.io.write('dftb_full.xyz', dftb_full)
            
            
            t_physrep = perf_counter()

    # ============================================================================ #
    #  4. Data processing and output

            # Calculate the energy per atom and plot
            with open ("search_index.txt", 'r') as search_index_file:
                search_index = search_index_file.read().strip().split(',')
            bolt_f11_final = [self.bolt_f11[int(i)] for i in search_index]

            if self.args_input.optimization_option.lower() == 'energygeometry':
                dftb_energy_per_atom_relative, dft_energy_per_atom_relative, delta_energy_per_atom = self.get_energy_per_atom(dft_subincopy_final, dftb_full, bolt_f11_final)

            elif self.args_input.optimization_option.lower() in ('dataset', 'reaction'):
                E0_dftb = self.get_atomic_energy(dft_incopy)
                _, dftb_energy_per_atom_relative = self.get_formation_energy(dftb_full, E0_dftb, '_')
                _, dft_energy_per_atom_relative = self.get_formation_energy(dft_incopy, self.args_input.E0s, '_')



            # Plot: figures of one element's phases (labelled dimer ... fcc, at most six, the
            # element in the legend, energies relative to the lowest energy per atom of the
            # dataset).  EOS groups of several elements (--shared-parameters, M5b-3) get none, as
            # with parameters per type; groups of one element each are plotted as before
            one_element_groups = all(len(np.unique(a.numbers)) == 1 for a in dft_subincopy_final)
            if (self.args_input.optimization_option.lower() == 'energygeometry'
                    and self.args_input.multi_element is None and one_element_groups):

                bolt_f = self.bolt_f11[floor(self.args_input.eos_points/2)::self.args_input.eos_points]

                plot.plot_vs(atomic_symbols[0], [r0_w, r0_d, sigma_rep, mu], dft_incopy, dft_energy_per_atom_relative, dftb_energy_per_atom_relative, bolt_f)

                plot.plot_tradition(atomic_symbols[0], [r0_w, r0_d, sigma_rep, 2.0],  mu, dft_energy_per_atom_relative, dftb_energy_per_atom_relative, bolt_f)
            

            with open(self.args_input.para_file, 'a') as para:
                para.write(f'{r0_w:15.8f}, {r0_d:15.8f}, {sigma_rep:15.8f}, {kxc:20.8e}, {self.cov_radius:15.7f}, {mu:25.11f}\n')


            t_end = perf_counter()
            self.step_timings = {'hotcent': t_hotcent - t_ini, 'dftb': t_dftb - t_hotcent,
                                 'physrep': t_physrep - t_dftb, 'data': t_end - t_physrep,
                                 'total': t_end - t_ini}

            # os.chdir(p / DEFAULT_OUT_DIR)        
            print(f"""\nTime:
                \nHotcent: {t_hotcent - t_ini} 
                \nDFTB: {t_dftb - t_hotcent}
                \nPhysrep: {t_physrep - t_dftb}
                \nDataProcessing: {t_end - t_physrep}
                \nTotal: {t_end - t_ini}""")              
            if current_plan() is not None and current_plan().estimates:
                print(current_plan().step_line(self.step_timings))
            if grad is not None:
                grad.after_step(self, conf_parameters, sigma_rep, mu)
            self._record_success(mu)
            return mu
            

        # Except Errors caused by unphysical parameters
        except(RuntimeError, AssertionError, UnboundLocalError, ase.io.formats.UnknownFileTypeError) as error:
            print("Error:", error)
            sigma_rep, kxc = [0.4321, 0.0]
            mu_E, mu_V, mu_logb, penalty = [10, 10, 10, 10]
            # M5b-2: twice the worst successful loss so far (was a random integer 97-99)
            mu = self._record_failure([r0_w, r0_d], error)

            with open(self.args_input.para_file, 'a') as para:
                para.write(f'{r0_w:15.8f}, {r0_d:15.8f}, {sigma_rep:15.8f}, {kxc:20.8e}, {self.cov_radius:15.7f}, {mu:25.11f}\n')
            
            with open(self.args_input.log_file, 'a') as output:
                output.write(f'{r0_w:15.8f}, {r0_d:15.8f}, {sigma_rep:15.8f}, {kxc:20.8e}, {self.cov_radius:15.8f}, {mu_E:15.8f}, {mu_V:20.8f}, {mu_logb: 20.8f}, {penalty:15.5f}, {mu:25.11f}\n')
            if grad is not None:
                grad.fail(conf_parameters, mu, f"{type(error).__name__}: {error}")
            return mu
        

    def BandStructure(self, conf_parameters:list,
                        xc:str,
                        superposition:str,
                        ) -> float:
        
        """
        Calculate the loss function for Band Structure

        Parameters:
        -------
        conf_parameters: list
            List of parameters to be optimized
            eg: conf_parameters = [r0_w, r0_d, p]
        In band structure optimization, r0_d == r0_w

        xc: str
            Exchange and Correlation functional
            eg: xc = 'GGA_X_PBE+GGA_C_PBE'

        
        superposition: str
            density or potential

        """

        # Data preparation
        r0_w, r0_d, p = conf_parameters
        opt_directory = self.path / self.args_input.results_dir / f"opt_{r0_w:.3f}_{p:.3f}"
        opt_directory.mkdir(exist_ok=True, parents = True)
        os.chdir(opt_directory)


        # Generate SKF
        t_ini = perf_counter()
        confinement_r0, confinement_p = self.set_parameters(conf_parameters)
        print(f"Confinement_r0:{confinement_r0}, Confinement_p:{confinement_p}")

        self.hp.set_variables(confinement_r0,
                            confinement_p, 
                            xc=xc,
                            superposition=superposition,
                            )
        try:
            # Write  SKF file
            self.hp.full_hotcent(superposition=superposition,
                            opt_elem=self.opt_elem,
                            )        

            t_hotcent = perf_counter()

    #       DFTB BAND STRUCTURE CALCULATION
            dft_incopy = copy.deepcopy(self.dft_input)

            # Weight of each structure
            bolt_f = [1.0]

            # The dftb_bandstructure is refer to DFTB1 not real band structure
            calc_band_binary(conf_parameters, dft_incopy, self.args_input.kpt_density, bolt_f)

    #       DELAT BAND STRUCTURE CALCULATION
            dft_incopy = copy.deepcopy(self.dft_input)
            mu_homo, mu_lumo = delta_band_binary(conf_parameters, self.args_input.dft_band_file, dft_incopy, bolt_f)

            mu = mu_homo + mu_lumo

            with open(self.args_input.para_file, 'a') as para:
                para.write(f"{r0_w:15.8f}, {r0_d:15.8f}, {p:15.8f}, {self.cov_radius:15.7}, {mu_homo:20.11f}, {mu_lumo:20.11f}, {mu:25.11f}\n")

            return mu
        

        except(RuntimeError, AssertionError, UnboundLocalError, ase.io.formats.UnknownFileTypeError) as error:
            print(f"Error in the bandstruture calculation:{error}")
            mu_homo, mu_lumo, mu = 10, 10, 20
            with open(self.args_input.para_file, 'a') as para:
                para.write(f"{r0_w:15.8f}, {r0_d:15.8f}, {p:15.8f}, {self.cov_radius:15.7}, {mu_homo:20.11f}, {mu_lumo:20.11f}, {mu:25.11f}\n")
            return mu
        

    def Loss_EnergyGeometry(self, parameters:list) -> float:
        """
        Calculate Loss function based on Energy and Geometry

        Parameters:
        -------
        parameters : list
            List of parameters to be optimized
            eg: parameters = [r0_w, r0_d]
        """

        superposition = self.args_input.superposition
        xc = self.args_input.xc

        if self.typed:                       # parameters per type (M5b): (r0_w, r0_d) per set
            return self.EnergyGeometryTypes(list(parameters), xc=xc, superposition=superposition)

        default_p = 2.0
        r0_w, r0_d = parameters
        conf_parameters = [r0_w, r0_d, default_p]

        mu = self.EnergyGeometry(conf_parameters,
                            xc=xc,
                            superposition=superposition,
                            )

        return mu


    def Loss_BandStructure(self, parameters:list) -> float:
        """
        Calculate Loss function based on Band Structure

        Parameters:
        -------
        parameters : list
            List of parameters to be optimized
            eg: conf_parameters = [r0, p]
        """
        
        superposition = self.args_input.superposition
        xc = self.args_input.xc

        r0, p = parameters

        # In band structure optimization, r0_d == r0_w
        conf_parameters = [r0, r0, p]

        mu = self.BandStructure(conf_parameters,
                            xc=xc,
                            superposition=superposition,
                            )

        return mu



    # =========================================================
    # Parameters per type (milestone M5b)
    # =========================================================
    def Repulsion_optimization_types(self, x, c0: float = 0.1, c1: float = 0.8):
        """sigma_rep of every parameter set at the radii ``x``: (sigma vector, mu).

        The loss at sigma is utils.loss_grad.mu on the step's files (in memory, nothing is
        written; every evaluation adds a line to the log, and in reaction mode a block to
        reaction_per_formula.csv).  One sigma (one fitted set): the search of before M5b, an
        11-point grid polished by fmin, then L-BFGS-B (ftol 1e-3, at most 20 evaluations).
        n sigma: the same 1-D search for one sigma common to all sets, then every set on its
        own on the 11-point grid with the others at their current values (kept only if
        better), then L-BFGS-B over all n together (ftol 1e-3, at most 20 n evaluations); the
        grids and L-BFGS-B again while the grids improve the joint minimum (at most 3 rounds).
        A grid of 11^n points would not be affordable.
        """
        from utils import loss_grad
        model = self.types
        step = loss_grad.StepData.from_files(self, x=x)
        self._typed_step = step
        n = model.n_sigma
        srange = (c0, c1)
        if self.args_input.optimization_option.lower() in ('dataset', 'reaction'):
            # The isolated atoms' energies (Loss.get_atomic_energy, once per step) run the DFTB
            # engine on this step's SKF files, and DFTB+ refuses files without a repulsive
            # spline: give them one first.  An atom alone in its 20 A box has no repulsion, so
            # any sigma does; the files get sigma*'s at the end (before M5b they were rewritten
            # at every sigma tried).
            step.physrep_all([0.5 * (c0 + c1)] * n).write_skf('ptbp')

        def f(svec):
            svec = [float(v) for v in np.atleast_1d(np.asarray(svec, dtype=float))]
            parts = {}
            mu = float(loss_grad.mu(step, svec, parts))
            self._log_typed(x, svec, parts, mu)
            return mu

        def first(s):
            return float(np.atleast_1d(s)[0])

        if n == 1:
            res_brute = brute(f, (srange,), Ns=11, full_output=True, disp=True, workers=1)
            print(f"Search time 1: sigma_rep = {res_brute[0]}, mu = {res_brute[1]}")
            res = minimize(f, [first(res_brute[0])], method='L-BFGS-B',
                           bounds=Bounds(c0, c1, keep_feasible=True),
                           options={'maxfun': 20, 'ftol': 1e-3})
            print(f"Search time 2: sigma_rep = {res.x[0]}, mu = {res.fun}")
            return [float(res.x[0])], float(res.fun)

        # 1. one sigma for every set
        res_brute = brute(lambda s: f([first(s)] * n), (srange,), Ns=11, full_output=True,
                          disp=True, workers=1)
        cur = [min(max(first(res_brute[0]), c0), c1)] * n
        f_cur = f(cur)
        print(f"Search time 1 (common sigma_rep): {cur[0]}, mu = {f_cur}")
        # 2. every set on its own on the grid, the others at their current values; 3. all
        # together.  Again from the joint minimum while a grid improves it (at most 3 rounds):
        # L-BFGS-B stops at kinks, and a sigma that hardly matters (a plateau) can leave a
        # better point on the grid once the other sigma have moved
        for rnd in range(3):
            moved = False
            for k in range(n):
                def fk(s, k=k):
                    trial = list(cur)
                    trial[k] = first(s)
                    return f(trial)
                res_k = brute(fk, (srange,), Ns=11, full_output=True, finish=None, disp=False,
                              workers=1)
                if float(res_k[1]) < f_cur:
                    cur[k], f_cur = first(res_k[0]), float(res_k[1])
                    moved = True
            print(f"Search time 2.{rnd + 1} (each set on the grid): sigma_rep = {cur}, mu = {f_cur}")
            if rnd > 0 and not moved:
                break
            res = minimize(f, cur, method='L-BFGS-B', bounds=Bounds([c0] * n, [c1] * n,
                                                                      keep_feasible=True),
                           options={'maxfun': 20 * n, 'ftol': 1e-3})
            if float(res.fun) < f_cur:             # L-BFGS-B ends no higher than it starts
                cur, f_cur = [float(v) for v in res.x], float(res.fun)
            print(f"Search time 3.{rnd + 1} (L-BFGS-B, all sets): sigma_rep = {cur}, mu = {f_cur}")
        return list(cur), f_cur

    def _log_typed(self, x, svec, parts, mu) -> None:
        """One line of the log (and the reaction-mode breakdown) for one sigma evaluation."""
        model = self.types
        mode = self.args_input.optimization_option.lower()
        if mode == 'energygeometry':
            cols = (parts['mu_E'], parts['mu_V'], parts['mu_logb'])
        else:
            cols = (parts['mu_E'], parts['mu_F'], 0.0)
        with open(self.args_input.log_file, 'a') as output:
            output.write(model.log_line(x, svec, *cols, parts['penalty'], mu))
        if mode == 'reaction':
            from collections import defaultdict as _dd
            buckets = _dd(list)
            for atoms, d in zip(self._typed_step.dft, parts['diffs']):
                buckets[atoms.get_chemical_formula()].append(float(d))
            with open('reaction_per_formula.csv', 'a') as fh:
                fh.write(f"# {model.parameter_text(x, svec)} mu={mu:.6e}\n")
                fh.write("formula,n_structures,mean_abs_dE_per_atom,max_abs_dE_per_atom\n")
                for formula in sorted(buckets):
                    vals = np.array(buckets[formula])
                    fh.write(f"{formula},{len(vals)},{vals.mean():.6e},{vals.max():.6e}\n")

    def _write_typed_step(self, x, svec) -> list:
        """The step's files at sigma*: the repulsive in the SKF files, dftb_full.xyz, and in
        energygeometry mode the EOS files (trajectories, dft/dftb.json, the csv files) as the
        search of before M5b leaves them.  Returns dftb_full (the Atoms)."""
        import io

        from utils import loss_grad
        step = self._typed_step
        mode = self.args_input.optimization_option.lower()
        if mode == 'energygeometry':
            eos = step.eos
            eos_files = {}
            bolt_f = step.bolt_f
            for num in range(int(len(step.dft) / eos)):
                sl = slice(num * eos, (num + 1) * eos)
                pr_g, _ = step.physrep(svec, num)
                total = data.get_full_dftb(step.band[sl], step.dft[sl], pr_g, True)
                # as the search of before M5b, through dftb_sub_<num>.xyz: mu_E in the info,
                # the energy (which the EOS fit reads) and the forces in a calculator
                _, _, delta = loss_grad._group_rel_energies(self, step.dft[sl], total,
                                                            step.bolt11[sl])
                for atoms in total:
                    atoms.info['mu_E'] = delta
                buf = io.StringIO()
                ase.io.write(buf, total, format='extxyz')
                buf.seek(0)
                total = ase.io.read(buf, ':', format='extxyz')
                eos_dftb = self.eos_atoms(total, eos, 'dftb', files=eos_files)
                eos_dft = self.eos_atoms(step.dft[sl], eos, 'dft', files=eos_files)
                [eos_dft[conf].update(eos_dftb[conf]) for conf in eos_dftb]
                self.eos_cal(eos_dft, [bolt_f[num]])
            self.write_eos_files(eos_files)
        pr = step.physrep_all(svec)
        pr.write_skf('ptbp')
        dftb_full = data.get_full_dftb(step.band, step.dft, pr, True)
        ase.io.write('dftb_full.xyz', dftb_full)
        return dftb_full

    def EnergyGeometryTypes(self, x, xc: str, superposition: str) -> float:
        """Loss_EnergyGeometry with parameters per type (M5b): hotcent with every type's radii
        (files named after the type labels), the DFTB stage (pydftb mixes switched types), the
        sigma_rep search of every parameter set, the files of the step, par.out and log.out."""
        model = self.types
        x = [float(v) for v in x]
        mode = self.args_input.optimization_option.lower()
        kxc = 0.0
        self._E0_dftb_cache = {}
        grad = getattr(self, 'grad_request', None)
        opt_directory = self.path / self.args_input.results_dir / model.directory_name(
            x, 3 if grad is None else 6)
        opt_directory.mkdir(exist_ok=True, parents=True)
        os.chdir(opt_directory)

        t_ini = perf_counter()
        confinement_r0, confinement_p = self.set_parameters(x)
        print(f"Confinement_r0:{confinement_r0}, Confinement_p:{confinement_p}")
        self.hp.set_variables(confinement_r0, confinement_p, xc=xc,
                              superposition=self.args_input.superposition,
                              elements=model.element_of)
        penalty = model.penalty(x)
        try:
            self.hp.full_hotcent(superposition=superposition, opt_elem=[])
            t_hotcent = perf_counter()
            if grad is not None:
                grad.after_hotcent(self, x)                   # tangents (own timings)
                t_ini += perf_counter() - t_hotcent
                t_hotcent = perf_counter()

            dft_incopy = copy.deepcopy(self.dft_input)
            self.run_calculator_seperately(dft_incopy)
            t_dftb = perf_counter()
            # every structure needs DFTB results (hotcent may write no tables for a type at
            # extreme radii): none or only some is a failure (M5b-2; M5b failed on none only)
            self._check_complete_results()

            svec, mu = self.Repulsion_optimization_types(x)
            if not math.isfinite(float(mu)):                  # M5b-2: a failure, not a loss
                raise LossFailure(f"the loss is {mu}")
            self.last_sigma = list(svec)                  # sigma_rep of every set (mu's point)
            dftb_full = self._write_typed_step(x, svec)
            t_physrep = perf_counter()

            dft_final = self._typed_step.dft
            if mode == 'energygeometry':
                with open("search_index.txt", 'r') as search_index_file:
                    search_index = search_index_file.read().strip().split(',')
                bolt_f11_final = [self.bolt_f11[int(i)] for i in search_index]
                dftb_rel, dft_rel, _ = self.get_energy_per_atom(dft_final, dftb_full, bolt_f11_final)
                elements = sorted(set(model.element_of.values()))
                if len(elements) == 1 and self.args_input.multi_element is None:
                    bolt_f = self.bolt_f11[floor(self.args_input.eos_points/2)::self.args_input.eos_points]
                    title = model.parameter_text(x, svec) + f" mu={mu:.3e}"
                    first = model.sets[0].labels[0]
                    r0, sig = model.r0(x)[first], model.sigma(svec)[first]
                    plot.plot_vs(elements[0], [r0[0], r0[1], sig, mu], dft_incopy, dft_rel,
                                 dftb_rel, bolt_f, title=title)
                    plot.plot_tradition(elements[0], [r0[0], r0[1], sig, 2.0], mu, dft_rel,
                                        dftb_rel, bolt_f, title=title)

            with open(self.args_input.para_file, 'a') as para:
                para.write(model.par_line(x, svec, penalty, mu))

            t_end = perf_counter()
            self.step_timings = {'hotcent': t_hotcent - t_ini, 'dftb': t_dftb - t_hotcent,
                                 'physrep': t_physrep - t_dftb, 'data': t_end - t_physrep,
                                 'total': t_end - t_ini}
            print(f"\nTime:\nHotcent: {t_hotcent - t_ini}\nDFTB: {t_dftb - t_hotcent}\n"
                  f"Physrep: {t_physrep - t_dftb}\nDataProcessing: {t_end - t_physrep}\n"
                  f"Total: {t_end - t_ini}")
            if current_plan() is not None and current_plan().estimates:
                print(current_plan().step_line(self.step_timings))
            if grad is not None:
                grad.after_step(self, x, svec, mu)
            self._record_success(mu)
            return mu

        except(RuntimeError, AssertionError, UnboundLocalError, ase.io.formats.UnknownFileTypeError) as error:
            print("Error:", error)
            svec = [0.4321] * model.n_sigma
            mu_E, mu_V, mu_logb = 10, 10, 10
            mu = self._record_failure(x, error)        # M5b-2 (was a random integer 97-99)
            with open(self.args_input.para_file, 'a') as para:
                para.write(model.par_line(x, svec, penalty, mu))
            with open(self.args_input.log_file, 'a') as output:
                output.write(model.log_line(x, svec, mu_E, mu_V, mu_logb, penalty, mu))
            if grad is not None:
                grad.fail(x, mu, f"{type(error).__name__}: {error}")
            return mu

    # =========================================================
    # Outer-loop optimisers
    # =========================================================
    def optimize(self, cov_radius=None, checkpoint_saver=None, n_calls=11):
        """Dispatch to the optimiser selected by --optimizer.

        Returns an object with at minimum `.x` (best point as list) and
        `.fun` (best loss). The skopt path also exposes `.x_iters` /
        `.func_vals` for plotting; the PSO path provides equivalents.
        ``cov_radius`` (Bohr) sets the search bounds; by default the penalty radius.
        """
        if cov_radius is None:
            cov_radius = self.cov_radius
        elif self.cov_radius is not None and abs(cov_radius - self.cov_radius) > 1e-12:
            print(f"Warning: search bounds use r_cov = {cov_radius} Bohr, "
                  f"the loss penalty {self.cov_radius} Bohr")
        opt = getattr(self.args_input, 'optimizer', 'bo')
        if opt == 'bo':
            return self.Bayesian_optimization(cov_radius, checkpoint_saver, n_calls)
        if opt == 'parallel_bo':
            return self._optimize_parallel_bo(cov_radius, checkpoint_saver, n_calls)
        if opt == 'pso':
            return self._optimize_pso(cov_radius, n_calls)
        if opt in ('gradient', 'bo_grad'):                  # M4c
            from utils import gradient_opt
            maxiter = getattr(self.args_input, 'grad_maxiter', None)
            steps = getattr(self.args_input, 'hotcent_fd_step', None)
            if steps is not None and len(steps) > 2:
                raise SystemExit("--hotcent_fd_step takes one or two steps: r0_w [r0_d]")
            if steps is not None and len(steps) == 1:
                from utils.hotcent_tangent import DEFAULT_STEPS
                steps = [steps[0], DEFAULT_STEPS[1]]
            if opt == 'gradient':
                return gradient_opt.optimize_gradient(self, cov_radius, maxiter=maxiter or 20,
                                                      steps=steps)
            return gradient_opt.optimize_bo_grad(self, cov_radius, checkpoint_saver, n_calls,
                                                 maxiter=maxiter or 5, steps=steps)
        raise ValueError(f"Unknown optimizer: {opt!r}")

    def _bo_dimensions_and_cost(self, cov_radius):
        """Shared helper: returns (dimensions, bounds_lo, bounds_hi, cost_fn).

        Both BO and PSO use the same parameter ranges; the data shape is the
        only thing that differs (skopt wants Real() dimensions, pyswarms
        wants raw lower/upper arrays).
        """
        opt_opt = self.args_input.optimization_option.lower()
        if self.typed and opt_opt in ('energygeometry', 'dataset', 'reaction'):
            # M5b: (r0_w, r0_d) of every parameter set, bounds from the set's covalent radius
            lo, hi = self.types.bounds()
            dims = [Real(name=n, low=a, high=b, prior='uniform')
                    for n, a, b in zip(self.types.dimension_names(), lo, hi)]
            return dims, lo, hi, self.Loss_EnergyGeometry
        if opt_opt in ('energygeometry', 'dataset', 'reaction'):
            r0_w_lower, r0_w_upper = 2.1, 7.0 * cov_radius
            r0_d_lower, r0_d_upper = 3.4, 9.0 * cov_radius
            dims = [Real(name='r0_w', low=r0_w_lower, high=r0_w_upper, prior='uniform'),
                    Real(name='r0_d', low=r0_d_lower, high=r0_d_upper, prior='uniform')]
            lo = [r0_w_lower, r0_d_lower]
            hi = [r0_w_upper, r0_d_upper]
            cost_fn = self.Loss_EnergyGeometry
        else:  # bandstructure
            r0_lower, r0_upper = 2.1, 7.0 * cov_radius
            p_lower, p_upper = 1.01, 11.0
            dims = [Real(name='r0', low=r0_lower, high=r0_upper, prior='uniform'),
                    Real(name='p', low=p_lower, high=p_upper, prior='uniform')]
            lo = [r0_lower, p_lower]
            hi = [r0_upper, p_upper]
            cost_fn = self.Loss_BandStructure
        return dims, lo, hi, cost_fn

    def _optimize_pso(self, cov_radius, n_calls):
        """Particle Swarm Optimization via pyswarms.

        Population × iterations gives the total evaluation budget. We aim
        for `n_particles * iters ≈ n_calls`, with `n_particles` from
        --n_particles (default 4) and `iters` derived (≥2).
        """
        import pyswarms as ps

        n_particles = max(2, int(getattr(self.args_input, 'n_particles', 4)))
        iters = max(2, n_calls // n_particles)
        total = n_particles * iters

        _dims, lo, hi, cost_fn = self._bo_dimensions_and_cost(cov_radius)
        bounds = (np.array(lo, dtype=float), np.array(hi, dtype=float))

        # Standard PSO hyperparameters; tunable later via CLI if needed.
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        def batch_cost(swarm):
            return np.array([float(cost_fn(p.tolist())) for p in swarm])

        print(f"[PSO] particles={n_particles}, iters={iters}, "
              f"total evaluations={total} (bounds lo={lo}, hi={hi})")
        opt = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(lo),
            options=options,
            bounds=bounds,
        )
        cost, pos = opt.optimize(batch_cost, iters=iters)

        # Wrap into a skopt-result-like object so cli/run.py downstream stays
        # the same. Flatten pos_history (per-iter swarm arrays) into a single
        # x_iters list.
        x_iters_flat = [list(map(float, p)) for swarm in opt.pos_history for p in swarm]
        # cost_history is best-so-far per iter; expand to per-evaluation by
        # repeating each iteration's snapshot n_particles times so plotting
        # over `func_vals` still makes sense.
        func_vals_flat = []
        for c in opt.cost_history:
            func_vals_flat.extend([float(c)] * n_particles)

        # SimpleNamespace is picklable (unlike a class defined inside a
        # method), so cli/run.py can dump the result to result.pkl for
        # `--postprocess_only` reuse.
        from types import SimpleNamespace
        return SimpleNamespace(
            x=list(map(float, pos)),
            fun=float(cost),
            x_iters=x_iters_flat,
            func_vals=func_vals_flat,
            optimizer='pso',
        )

    def _optimize_parallel_bo(self, cov_radius, checkpoint_saver, n_calls):
        """Batched BO via skopt.Optimizer.ask(n_points, strategy='cl_min')
        with TRUE parallel evaluation via a forked Process per particle.

        Each round:
          1. opt.ask(n_points=n_particles) proposes a batch using the
             constant-liar strategy.
          2. n_particles forked subprocesses each evaluate Loss_EnergyGeometry
             on one batch member concurrently. Forked workers inherit the
             Loss instance state (dft_input, hp, E0 cache, etc.) without
             needing to pickle bound methods.
          3. Results are collected via Queue and fed back via opt.tell.

        Notes:
        - We use `multiprocessing.Process` (not `Pool`) because Pool workers
          are daemonic and would forbid the nested mp.Pool that
          `hotpy.full_hotcent` and `parallel_repulsion` create internally,
          and the forked DFTB-stage workers (utils.parallel).
        - With a parallel plan (cli/run.py) every particle uses its share of the
          cores (Plan.for_particle), so particles x workers x threads <= cores.
        - par.out and log.out at the parent cwd are appended-to by every
          worker concurrently — line interleaving is possible but writes
          are atomic per-line on POSIX. The per-evaluation
          `results/opt_<r0_w>_<r0_d>/` directories are unique so no
          collision there.
        - Failed points (M5b-2): a worker's failure value is twice the worst
          successful loss of the rounds before its own (the state it forked
          with); the parent records every worker's outcome after the round.
        """
        from skopt import Optimizer
        import multiprocessing as mp

        n_particles = max(2, int(getattr(self.args_input, 'n_particles', 4)))
        rounds = max(1, n_calls // n_particles)

        dims, _lo, _hi, cost_fn = self._bo_dimensions_and_cost(cov_radius)
        opt = Optimizer(dimensions=dims, base_estimator='GP', acq_func='EI',
                        random_state=123)

        ctx = mp.get_context('fork')

        plan = current_plan()

        def _worker(x_arg, q, idx):
            # Closure over `cost_fn` works under fork because the child
            # inherits the parent's full Python state; no pickling involved.
            if plan is not None:
                set_plan(plan.for_particle(idx))     # this BO point's share of the cores
            try:
                val = float(cost_fn(x_arg))
                q.put((idx, val, None, bool(getattr(self, 'last_failed', False))))
            except Exception:
                import traceback as _tb
                q.put((idx, None, _tb.format_exc(), False))

        print(f"[parallel_BO] batch_width={n_particles} (forked workers), "
              f"rounds={rounds}, total evaluations={n_particles*rounds}")
        x_iters_flat, func_vals_flat = [], []
        best_x, best_fun = None, float('inf')
        for r in range(rounds):
            xs = opt.ask(n_points=n_particles, strategy='cl_min')
            queue = ctx.Queue()
            procs = [ctx.Process(target=_worker, args=(xs[i], queue, i))
                     for i in range(len(xs))]
            for p in procs:
                p.start()
            results = [None] * len(xs)
            failures = [False] * len(xs)
            for _ in xs:
                idx, val, err, bad = queue.get()
                if err is not None:
                    for p in procs:
                        if p.is_alive():
                            p.terminate()
                    raise RuntimeError(
                        f"parallel_bo worker {idx} crashed:\n{err}")
                results[idx] = val
                failures[idx] = bad
            for p in procs:
                p.join()
            for x, y, bad in zip(xs, results, failures):    # M5b-2: the workers' outcomes
                self._note_outcome(x, y, bad)
            opt.tell(xs, results)
            for x, y in zip(xs, results):
                x_iters_flat.append(list(map(float, x)))
                func_vals_flat.append(y)
                if y < best_fun:
                    best_fun, best_x = y, list(map(float, x))
            print(f"[parallel_BO] round {r+1}/{rounds}: best so far "
                  f"x={best_x}, fun={best_fun:.6e}")

        from types import SimpleNamespace
        return SimpleNamespace(
            x=best_x,
            fun=best_fun,
            x_iters=x_iters_flat,
            func_vals=func_vals_flat,
            optimizer='parallel_bo',
        )

    def _check_checkpoint_meta(self, meta_path):
        """Refuse to resume a checkpoint whose losses were computed on another scale; returns
        the meta file's record (its failed evaluations, M5b-2)."""
        meta = None
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                meta = json.load(fh)
        if self.typed:                                       # M5b: the types must be the same
            if self.types.matches_meta(meta):
                return meta
            raise RuntimeError(
                f"{self.args_input.checkpoint} has "
                + ("no record of its loss definition" if meta is None else f"loss record {meta}")
                + f"; this run uses loss version {LOSS_VERSION} with the parameter sets "
                f"{[s.name for s in self.types.sets]} ({self.types.layout}). Its points have "
                f"another number or meaning of parameters: delete it (or use another --checkpoint) "
                f"to start afresh.")
        if (meta is not None and meta.get("ptbp_loss_version") == SHARED_LOSS_VERSION
                and abs(meta.get("penalty_radius_bohr", -1.0) - self.cov_radius) < 1e-5):
            return meta
        found = ("no record of its loss definition (written before PTBP fixed the penalty "
                 "radius, which was always 0.5 Bohr)" if meta is None else f"loss record {meta}")
        raise RuntimeError(
            f"{self.args_input.checkpoint} has {found}; this run uses loss version "
            f"{SHARED_LOSS_VERSION} with a penalty radius of {self.cov_radius:.6f} Bohr. Mixing the two "
            f"scales would corrupt the Bayesian model. Convert the checkpoint with "
            f"`python tools/rescale_checkpoint.py {self.args_input.checkpoint} "
            f"--radius {self.cov_radius!r}` or delete it to start afresh.")

    def Bayesian_optimization(self,
                              cov_radius,
                              checkpoint_saver,
                              n_calls):
        """
        Initialize the Bayesian optimization
        """

        # if self.args_input.checkpoint.exists():
        mode = self.args_input.optimization_option.lower()
        # absolute (M5b-2): the failures are recorded during the evaluations, which change into
        # their step directories; here the cwd is the run folder, as before
        meta_path = os.path.abspath(f"{self.args_input.checkpoint}.meta.json")
        failed_before = []                   # failed evaluations of a resumed checkpoint (M5b-2)
        if os.path.exists(self.args_input.checkpoint):
            loaded_meta = None
            if mode in ('energygeometry', 'dataset', 'reaction'):
                loaded_meta = self._check_checkpoint_meta(meta_path)
            res = load(self.args_input.checkpoint)
            x0 = res.x_iters
            y0 = res.func_vals
            done = len(x0)
            if mode in ('energygeometry', 'dataset', 'reaction'):
                # the worst successful loss starts from the loaded successes; the failures are
                # those the meta file records, or the values 97-99 of an older checkpoint
                y0, failed_before = self._resume_failures(x0, y0, loaded_meta)
            # Note: when x0/y0 are passed, gp_minimize treats `n_calls` as the
            # number of *additional* fresh evaluations on top of the prior
            # ones. So `--n_calls 11` means "do 11 more BO steps", giving a
            # total stored history of `done + n_calls`.
            print(f"[checkpoint] Resuming from {self.args_input.checkpoint}: "
                  f"{done} prior iteration(s) loaded; will run {n_calls} more "
                  f"(total stored: {done + n_calls}).")
        elif self.typed and mode in ('energygeometry', 'dataset', 'reaction'):
            # M5b: every parameter set starts at its declared start, PTBP's table, or
            # r_cov x (3, 5), inside the box (utils.typemodel)
            x0 = self.types.seed()
            print(f"BO seed per parameter set: "
                  + ", ".join(f"{n}={v:.4f}" for n, v in zip(self.types.dimension_names(), x0)))
            y0 = None
        else:
            # Smart seed: for an elementary system whose element already has a
            # PTBP entry, start the BO at those published params instead of a
            # generic cov_radius * (3, 5) heuristic. Saves several BO iterations
            # of cold-start exploration.
            x0 = None
            try:
                symbols = sorted({chemical_symbols[int(z)]
                                  for atoms in self.dft_input
                                  for z in atoms.numbers})
                if (len(symbols) == 1
                        and symbols[0] in get_PTBP()
                        and self.args_input.optimization_option.lower() in
                            ('energygeometry', 'dataset', 'reaction')):
                    p = get_PTBP()[symbols[0]]
                    x0 = [p['r0_w'], p['r0_d']]
                    print(f"BO seed from PTBP[{symbols[0]}]: "
                          f"r0_w={x0[0]}, r0_d={x0[1]}")
            except Exception as _e:
                print(f"PTBP-seed lookup failed ({_e}); falling back to heuristic")
            if x0 is None:
                x0 = [cov_radius * 3, cov_radius * 5]  # DEFAULT_START_PARA
                print(f"BO seed from heuristic: r0_w={x0[0]}, r0_d={x0[1]}")
            y0 = None
            n_calls = n_calls

        if self.typed:
            meta = self.types.meta(mode)
        else:
            meta = {"ptbp_loss_version": SHARED_LOSS_VERSION,
                    "penalty_radius_bohr": self.cov_radius, "mode": mode}
        with open(meta_path, 'w') as fh:
            # M5b-2: the failed evaluations of a resumed checkpoint, only when it has some (a
            # run without failures writes the meta file of before)
            json.dump(dict(meta, failed_evaluations=list(failed_before)) if failed_before
                      else meta, fh)
        if mode in ('energygeometry', 'dataset', 'reaction'):
            self._bo_log = {'meta_path': meta_path, 'meta': meta, 'failed': list(failed_before),
                            'next': len(x0) if y0 is not None else 0}

        if self.typed and mode in ('energygeometry', 'dataset', 'reaction'):
            dimensions, lo, hi, cost_fn = self._bo_dimensions_and_cost(cov_radius)
            print("Bounds: " + ", ".join(f"{d.name}: {a} ~ {b:.6g}"
                                         for d, a, b in zip(dimensions, lo, hi)))
            try:
                res = gp_minimize(func=cost_fn,
                                  dimensions=dimensions,
                                  n_calls=n_calls,
                                  acq_func="EI",
                                  x0=x0,
                                  y0=y0,
                                  callback=[checkpoint_saver],
                                  random_state=123)
            finally:
                self._end_bo_log()
        elif self.args_input.optimization_option.lower() in ('energygeometry', 'dataset', 'reaction'):
            r0_w_lower, r0_w_upper = 2.1, 7.0 * cov_radius            # Wavefunction cutoff
            r0_d_lower, r0_d_upper = 3.4, 9.0 * cov_radius            # Density function cutoff
            print(f"Bounds: r0_w: {r0_w_lower} ~ {r0_w_upper}, r0_d: {r0_d_lower} ~ {r0_d_upper}")

            dr0_w   = Real(name='r0_w',  low=r0_w_lower,  high=r0_w_upper,   prior='uniform')
            dr0_d   = Real(name='r0_d',  low=r0_d_lower,  high=r0_d_upper,   prior='uniform')
            dimensions = [dr0_w, dr0_d]

            try:
                res = gp_minimize(func=self.Loss_EnergyGeometry,
                                dimensions=dimensions,
                                n_calls=n_calls,
                                acq_func="EI",
                                x0=x0,
                                y0=y0,
                                callback=[checkpoint_saver],
                                random_state=123)
            finally:
                self._end_bo_log()

        elif self.args_input.optimization_option.lower() == 'bandstructure':
            r0_lower, r0_upper = 2.1, 7.0 * cov_radius
            p_lower, p_upper = 1.01, 11.0

            # V_conf = (r0/r)^p

            print(f"Bounds: r0: {r0_lower} ~ {r0_upper}, p: {p_lower} ~ {p_upper}")
            print(f"Default r0: {x0[0]}, p: {x0[1]}")

            dr0   = Real(name='r0',  low=r0_lower,  high=r0_upper,   prior='uniform')
            dp   = Real(name='p',  low=p_lower,  high=p_upper,   prior='uniform')
            dimensions = [dr0, dp]

            res = gp_minimize(func=self.Loss_BandStructure,
                            dimensions=dimensions,
                            n_calls=n_calls,
                            acq_func="EI",
                            x0=x0,
                            y0=y0,
                            callback=[checkpoint_saver],
                            random_state=self.args_input.random_state)

        
        return res
