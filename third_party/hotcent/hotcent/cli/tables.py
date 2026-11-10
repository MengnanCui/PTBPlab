#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import os
import numpy as np
import sys
import traceback
import yaml
from argparse import ArgumentParser
from itertools import combinations_with_replacement, product
from multiprocessing import Pool
from time import time
from ase.data import atomic_numbers, covalent_radii
from ase.units import Bohr
from hotcent import __version__
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.offsite_chargetransfer import Offsite2cMTable, Offsite2cUTable
from hotcent.offsite_magnetization import Offsite2cWTable
from hotcent.offsite_threecenter import Offsite3cTable
from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.onsite_chargetransfer import (Onsite1cMTable, Onsite1cUTable,
                                           Onsite2cUTable)
from hotcent.onsite_magnetization import Onsite1cWTable, Onsite2cWTable
from hotcent.onsite_threecenter import Onsite3cTable
from hotcent.onsite_twocenter import Onsite2cTable
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT
from hotcent.repulsion_twocenter import Repulsion2cTable
from hotcent.repulsion_threecenter import Repulsion3cTable
from hotcent.utils import get_file_checksum, verify_chemical_symbols


def parse_arguments():
    description = """
    Generate integral tables for all selected element combinations.

    Note: existing output files will be overwritten.
    """
    parser = ArgumentParser(description=description)
    parser.add_argument('include', nargs='*', help='Element combinations '
                        'to consider. To e.g. include all combinations that '
                        'involve H and/or Si, as well as all combinations '
                        'involving only O, write "H,Si O".')
    parser.add_argument('--aux-mappings', default='mulliken,giese_york',
                        help='Comma-separated procedures to consider for '
                        'mapping the atomic orbital products to the auxiliary '
                        'basis. This determines the kind of kernel integral '
                        'and mapping integral tables that will be produced '
                        'by the chg*, mag* and map* tasks. Default: "mulliken,'
                        'giese_york".')
    parser.add_argument('--dry-run', action='store_true', help='Exit after '
                        'printing the task overview, without executing them.')
    parser.add_argument('--exclude', help='Element combinations to exclude. '
                        'To e.g. skip those involving only H or just H and '
                        'Si, write "H,H-Si". By default no combinations get '
                        'excluded.')
    parser.add_argument('--giese-york-constraint-method', default='original',
                        choices=['original', 'reduced'], help='Method for '
                        'selecting the multipole moments entering the '
                        'electrostatic fitting procedure used by the "map2c" '
                        'task for "giese_york" mappings. In the "reduced" '
                        'method, all orbital momenta are included for l <= '
                        'max(lmax_a, lmax_b) = lmax_ab. In the "original" '
                        'method, which is the approach described by Giese and '
                        'York (2011, doi:10.1063/1.3587052), these are '
                        'supplemented by those orbital momenta belonging to l '
                        '== lmax_ab+1 for which |m| <= min(lmax_a, lmax_b).')
    parser.add_argument('--grid-opt-int', help='Path to a YAML file for '
                        'overriding the default integration grid options.')
    parser.add_argument('--grid-opt-tab', help='Path to a YAML file for '
                        'overriding the default tabulation grid options.')
    parser.add_argument('--label', help='Label to use when searching for the '
                        'input YAML files. The expected file names correspond '
                        'to "<Symbol>[.<label>].yaml".')
    parser.add_argument('--processes', type=int, default=1, help='Number of '
                        'processes to use for multiprocessing (default: 1).')
    parser.add_argument('--pseudo-path', default='.', help='Path to the '
                        'directory where the pseudopotential files are stored '
                        '(default: the current working directory).')
    parser.add_argument('--tasks', default='all', help='Comma-separated task '
                        'types to perform. The following types can be chosen: '
                        + ', '.join(TaskGenerator.all_task_types) + '. The '
                        'default "all" selects every type. Types can be '
                        'deselected by prepending a caret (^). For example, '
                        '--tasks=all,^rep3c selects all available task types '
                        'except 3-center repulsion.')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__}')
    args = parser.parse_args()
    return args


class Task:
    def __init__(self, task_type, elements, workdir, task_kwargs):
        self.task_type = task_type
        self.elements = elements
        self.workdir = workdir
        self.task_kwargs = task_kwargs

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class TaskGenerator:
    all_task_types = [
        'chgoff2c', 'chgon1c', 'chgon2c',
        'magoff2c', 'magon1c', 'magon2c',
        'map2c', 'map1c',
        'off2c', 'off3c',
        'on2c', 'on3c',
        'rep2c', 'rep3c',
    ]

    def __init__(self, included, excluded, task_types, task_kwargs):
        self.included = included
        self.excluded = excluded
        self.set_task_types(task_types)
        self.task_kwargs = task_kwargs
        self.generate_tasks()

    def get_task_types(self):
        task_types = [task_type for task_type in self.task_types]
        return task_types

    def set_task_types(self, task_types):
        negation = '^'

        for task_type in task_types:
            if task_type == 'all':
                self.task_types = [task for task in self.all_task_types]
                break
        else:
            self.task_types = []

        msg = 'Unknown task type: {0}'

        for task_type in task_types:
            if task_type != 'all' and not task_type.startswith(negation):
                assert task_type in self.all_task_types, msg.format(task_type)
                if task_type not in self.task_types:
                    self.task_types.append(task_type)

        for task_type in task_types:
            if task_type.startswith(negation):
                assert task_type[1:] in self.all_task_types, \
                       msg.format(task_type)
                if task_type[1:] in self.task_types:
                    index = self.task_types.index(task_type[1:])
                    self.task_types.pop(index)
        return

    def need_combination(self, elements):
        needed = True

        for combination in self.excluded:
            cond1 = all([el in combination for el in elements])
            cond2 = all([el in elements for el in combination])
            if cond1 and cond2:
                needed = False
                break

        return needed

    def need_task_type(self, task_type):
        needed = task_type in self.task_types
        return needed

    def need_task(self, task):
        needed = True

        if not self.need_task_type(task.task_type):
            needed = False

        if not self.need_combination(task.elements):
            needed = False

        if task in self.tasks:
            needed = False

        return needed

    def get_workdir(self, *elements, rootdir='.', prefix='tables_'):
        subdir = prefix + '-'.join(sorted(set(elements)))
        workdir = os.path.join(rootdir, subdir)
        return workdir

    def generate_tasks(self):
        self.tasks = []

        for task in self.yield_all_tasks():
            if self.need_task(task):
                self.tasks.append(task)

        return

    def get_tasks(self):
        priority = [
            'off3c', 'rep3c', 'on3c', 'map2c', 'chgoff2c', 'chgon2c',
            'magoff2c', 'magon2c', 'off2c', 'rep2c', 'on2c', 'map1c',
            'chgon1c', 'magon1c',
        ]
        tasks = sorted([task for task in self.tasks],
                       key=lambda x: priority.index(x.task_type))
        return tasks

    def yield_all_tasks(self):
        for elements in self.included:
            for el1 in elements:
                workdir = self.get_workdir(el1)
                for task_type in ['chgon1c', 'magon1c', 'map1c']:
                    yield Task(task_type, [el1], workdir, self.task_kwargs)

            for el1, el2 in combinations_with_replacement(elements, r=2):
                workdir = self.get_workdir(el1, el2)
                for task_type in ['chgoff2c', 'magoff2c', 'map2c', 'off2c',
                                  'rep2c']:
                    yield Task(task_type, [el1, el2], workdir,
                               self.task_kwargs)

            for el1, el2 in product(elements, repeat=2):
                workdir = self.get_workdir(el1, el2)
                for task_type in ['chgon2c', 'magon2c', 'on2c']:
                    yield Task(task_type, [el1, el2], workdir,
                               self.task_kwargs)

            for el1, el2 in combinations_with_replacement(elements, r=2):
                for el3 in elements:
                    workdir = self.get_workdir(el1, el2, el3)
                    for task_type in ['off3c', 'rep3c']:
                        yield Task(task_type, [el1, el2, el3], workdir,
                                   self.task_kwargs)

            for el1, el2, el3 in product(elements, repeat=3):
                workdir = self.get_workdir(el1, el2, el3)
                for task_type in ['on3c']:
                    yield Task(task_type, [el1, el2, el3], workdir,
                               self.task_kwargs)
        return

    def print_task_overview(self):
        ntasks = len(self.tasks)
        print('Number of tasks: {0}'.format(ntasks))
        if ntasks > 0:
            print('Generated tasks (in order of execution):')

            fmt = '{0:<12} {1:<12} {2}'
            print(fmt.format('Task', 'Elements', 'Working directory'))
            print(fmt.format('='*12, '='*12, '='*17))

            for task in self.get_tasks():
                element_str = ' '.join(['{:<2}'.format(el)
                                        for el in task.elements])
                print(fmt.format(task.task_type, element_str, task.workdir))
        return

    def create_working_directories(self):
        for task in self.tasks:
            if not os.path.exists(task.workdir):
                os.mkdir(task.workdir)
        return


def main():
    args = parse_arguments()

    included = []
    for item in args.include:
        elements = tuple(sorted(set(item.split(','))))
        verify_chemical_symbols(*elements)
        included.append(elements)

    excluded = []
    if args.exclude is not None:
        for item in args.exclude.split(','):
            elements = tuple(sorted(set(item.split('-'))))
            verify_chemical_symbols(*elements)
            excluded.append(elements)

    aux_mappings = args.aux_mappings.split(',')

    need_aux_basis = False
    for mapping in aux_mappings:
        assert mapping in ['mulliken', 'giese_york'], \
               'Unknown mapping: "{0}"'.format(mapping)
        if mapping != 'mulliken':
            need_aux_basis = True

    grid_opt_int = GridOptionsIntegrate.from_yaml(filename=args.grid_opt_int)
    grid_opt_tab = GridOptionsTabulate.from_yaml(filename=args.grid_opt_tab)

    task_types = list(sorted(set(args.tasks.split(','))))

    if not need_aux_basis:
        for task_type in ['map1c', 'map2c']:
            if task_type in task_types:
                msg = 'Warning: the "{0}" task will be skipped because no ' + \
                      'non-Mulliken mapping procedure got selected (see ' + \
                      'the --aux-mappings option)'
                print(msg.format(task_type))
                task_types.remove(task_type)

    atom_kwargs = dict(
        label=args.label,
        pseudo_path=os.path.abspath(args.pseudo_path),
        yaml_path=os.getcwd(),
    )

    task_kwargs = dict(
        atom_kwargs=atom_kwargs,
        aux_mappings=aux_mappings,
        giese_york_constraint_method=args.giese_york_constraint_method,
        grid_opt_int=grid_opt_int,
        grid_opt_tab=grid_opt_tab,
        shift=True,
    )

    generator = TaskGenerator(included, excluded, task_types, task_kwargs)
    generator.print_task_overview()

    if args.dry_run:
        return

    generator.create_working_directories()
    tasks = generator.get_tasks()

    procstr = 'process' if args.processes == 1 else 'processes'
    print('\nStarting pool with %d %s' % (args.processes, procstr), flush=True)
    pool = Pool(processes=args.processes)
    pool.map_async(wrapper, tasks, chunksize=1, callback=callback)
    pool.close()
    pool.join()
    print('All done')
    return


def callback(messages):
    for message in messages:
        print(message, flush=True)
    return


def wrapper(task):
    element_str = '-'.join(task.elements)

    suffix = '{0}_{1}'.format(element_str, task.task_type)
    stdout = os.path.join(task.workdir, 'out_{0}.txt'.format(suffix))
    sys.stdout = open(stdout, 'w')
    stderr = os.path.join(task.workdir, 'err_{0}.txt'.format(suffix))
    sys.stderr = open(stderr, 'w')

    cwd = os.getcwd()
    os.chdir(task.workdir)
    message = 'Task {0} for elements {1}: '.format(task.task_type, element_str)

    try:
        function = globals()[task.task_type]

        t_start = time()
        function(*task.elements, **task.task_kwargs)
        t_end = time()

        t_delta = int(np.ceil(t_end - t_start))
        message += 'completed in {0} seconds'.format(t_delta)
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        message += 'failed (check {0})'.format(stderr)

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr.close()
    sys.stderr = sys.__stderr__

    os.chdir(cwd)
    return message


def read_yaml(filename):
    with open(filename, 'r') as f:
        setup = yaml.safe_load(f)

    assert isinstance(setup, dict), \
           'Loading {0} did not return a dictionary'.format(filename)
    return setup


def get_atoms(*elements, label=None, only_1c=False, pseudo_path=None, txt='-',
              yaml_path=None):
    atoms = {}
    eigenvalues, hubbardvalues, occupations = {}, {}, {}
    offdiagonal_H, offdiagonal_S = {}, {}

    xc = None

    for el in set(elements):
        if label is None:
            filename = '{0}.yaml'.format(el)
        else:
            filename = '{0}.{1}.yaml'.format(el, label)
        filename = os.path.join(yaml_path, filename)

        setup = read_yaml(filename)

        if xc is None:
            xc = setup['atom']['xc']
        else:
            assert xc == setup['atom']['xc'], \
                   'The XC functional must be the same in all YAML files.'

        wf_conf = {}
        for nl, rc in setup['confinement']['rcuts'].items():
            wf_conf[nl] = SoftConfinement(amp=setup['confinement']['amp'],
                                          x_ri=setup['confinement']['x_ri'],
                                          rc=rc)

        if setup['pseudopotential'] is None:
            atom = AtomicDFT(el, txt=txt, wf_confinement=wf_conf,
                             **setup['atom'])
        else:
            pp_path = setup['pseudopotential']['filename']
            pp_path = os.path.join(pseudo_path, pp_path)
            setup['pseudopotential'].update(filename=pp_path)

            if 'sha256sum' in setup['pseudopotential']:
                sha256sum = get_file_checksum(pp_path, algorithm='sha256')
                msg = 'The checksum for {0} does not match the one in {1}'
                assert sha256sum == setup['pseudopotential']['sha256sum'], \
                       msg.format(pp_path, filename)
                del setup['pseudopotential']['sha256sum']

            pp = KleinmanBylanderPP(txt=txt, **setup['pseudopotential'])
            atom = PseudoAtomicDFT(el, pp, txt=txt, wf_confinement=wf_conf,
                                   **setup['atom'])

        atom.run()
        atom.generate_nonminimal_basis(**setup['basis'])
        atom.pp.build_projectors(atom)
        if setup['auxiliary_basis'] is not None:
            atom.generate_auxiliary_basis(**setup['auxiliary_basis'])

        eigenvalues[el], hubbardvalues[el], occupations[el] = {}, {}, {}
        offdiagonal_H[el], offdiagonal_S[el] = {}, {}

        for i, valence1 in enumerate(atom.basis_sets):
            for nl1 in valence1:
                if i == 0 and nl1 in atom.valence:
                    occupations[el][nl1] = atom.configuration[nl1]
                else:
                    occupations[el][nl1] = 0.

                H1c, S1c = atom.get_onecenter_integrals(nl1, nl1)
                eigenvalues[el][nl1] = H1c

                for j, valence2 in enumerate(atom.basis_sets):
                    if i != j:
                        for nl2 in valence2:
                            H1c, S1c = atom.get_onecenter_integrals(nl1, nl2)
                            offdiagonal_H[el][(nl1, nl2)] = H1c
                            offdiagonal_S[el][(nl1, nl2)] = S1c

        atoms[el] = atom

        for i, valence in enumerate(atom.basis_sets):
            for nl in valence:
                hubbardvalues[el][nl] = atom.get_analytical_hubbard_value(nl)

    properties = dict(
        occupations=occupations,
        hubbardvalues=hubbardvalues,
        eigenvalues=eigenvalues,
        offdiagonal_H=offdiagonal_H,
        offdiagonal_S=offdiagonal_S,
    )

    wf_ranges = dict()

    if not only_1c:
        for el1 in set(elements):
            wf_range = max([atoms[el1].wf_confinement[nl].rc
                            for nl in atoms[el1].valence])
            wf_ranges[el1] = wf_range

        for el1 in set(elements):
            for el2 in set(elements):
                rmax = wf_ranges[el1] + wf_ranges[el2] + 1.
                atoms[el1].pp.build_overlaps(atoms[el2], atoms[el1], rmin=1e-2,
                                             rmax=rmax, N=300)

    return (atoms, xc, properties, wf_ranges)


class GridOptions:
    def __init__(self, default_options=None, custom_options=None):
        def update(dictionary, task, **kwargs):
            for key, val in kwargs.items():
                msg = '"{0}" is not a known option for task "{1}"'
                assert key in dictionary, msg.format(key, task)
            dictionary.update(**kwargs)

        self.default_options = self.get_default_options()

        if default_options is not None:
            for task, options in default_options.items():
                assert task in self.default_options, \
                       'Unknown task "{0}"'.format(task)
                update(self.default_options[task], task, **options)

        self.custom_options = dict()

        if custom_options is not None:
            for elements_str, task_options in custom_options.items():
                elements = elements_str.split('-', maxsplit=1)
                elements = elements[:1] + elements[1].split('_', maxsplit=1)
                elements = tuple(elements)
                verify_chemical_symbols(*elements)

                self.custom_options[elements] = self.default_options.copy()

                for task, options in task_options.items():
                    assert task in self.default_options, \
                           'Unknown task "{0}"'.format(task)
                    update(self.custom_options[elements][task], task,
                           **options)

    @classmethod
    def from_yaml(cls, filename=None):
        custom_options = {} if filename is None else read_yaml(filename)
        default_options = custom_options.pop('default', None)
        return cls(default_options, custom_options)

    def get_parameters(self, el1, el2, task, wf_ranges, el3=None,
                       verbose=True):
        elements = tuple([el1, el2] if el3 is None else [el1, el2, el3])

        try:
            parameters = self.custom_options[elements][task].copy()
        except KeyError:
            try:
                parameters = self.custom_options[(el1, el2)][task].copy()
            except KeyError:
                parameters = self.default_options[task].copy()

        for param in parameters:
            if parameters[param] is None:
                func_name = 'generate_{0}'.format(param)
                func = getattr(self, func_name)
                value = func(*elements, task=task, wf_ranges=wf_ranges,
                             **parameters)
                parameters[param] = value

        if verbose:
            print('{0}: {1}'.format(self.__class__.__name__, parameters))

        return parameters


class GridOptionsIntegrate(GridOptions):
    def get_default_options(self):
        default_options = dict()

        for task in ['chgoff2c', 'chgon2c', 'magon2c', 'magoff2c',
                     'off2c', 'on2c', 'rep2c']:
            default_options[task] = dict(nr=200, ntheta=600)

        for task in ['map2c']:
            default_options[task] = dict(nr=100, ntheta=300)

        for task in ['off3c', 'on3c', 'rep3c']:
            default_options[task] = dict(nr=50, ntheta=150, nphi=13)

        return default_options


class GridOptionsTabulate(GridOptions):
    def get_default_options(self):
        default_options = dict()

        for task in ['chgoff2c', 'chgon2c', 'magon2c', 'magoff2c',
                     'map2c', 'off2c', 'on2c', 'rep2c']:
            default_options[task] = dict(dr=0.02, N=None, rmin=None)

        for task in ['off3c', 'on3c', 'rep3c']:
            default_options[task] = \
                dict(min_rAB=None, max_rAB=None, num_rAB=None,
                     min_rCM=None, max_rCM=None, num_rCM=None,
                     num_theta=24)

        return default_options

    def generate_rmin_half(self, el, **kwargs):
        rcov = covalent_radii[atomic_numbers[el]] / Bohr
        rmin_cov_scaling = 2. / 3
        rmin_half = rmin_cov_scaling * rcov
        rmin_half = kwargs['dr'] * np.floor(rmin_half / kwargs['dr'])
        return rmin_half

    def generate_rmin(self, el1, el2, **kwargs):
        rmin = sum([self.generate_rmin_half(e, **kwargs) for e in [el1, el2]])
        return rmin

    def generate_N_half(self, el, **kwargs):
        N_half = int(np.floor(kwargs['wf_ranges'][el] / kwargs['dr']))
        return N_half

    def generate_N(self, el1, el2, **kwargs):
        task = kwargs['task']
        if task.endswith('on2c'):
            N = 2 * self.generate_N_half(el1, **kwargs)
        elif task.endswith('off2c') or task in ['map2c', 'rep2c']:
            N = sum([self.generate_N_half(e, **kwargs) for e in [el1, el2]])
        else:
            raise NotImplementedError(task)

        rmin = kwargs['rmin']
        if rmin is None:
            rmin = self.generate_rmin(el1, el2, **kwargs)

        N -= int(np.round(rmin / kwargs['dr']))
        return N

    def generate_min_rAB(self, el1, el2, el3, **kwargs):
        min_rAB = self.generate_rmin(el1, el2, dr=0.02, **kwargs)
        return min_rAB

    def generate_max_rAB(self, el1, el2, el3, **kwargs):
        max_rAB = 0.98 * sum([kwargs['wf_ranges'][e] for e in [el1, el2]])
        return max_rAB

    def generate_num_rAB(self, el1, el2, el3, **kwargs):
        max_rAB = kwargs['max_rAB']
        if max_rAB is None:
            max_rAB = self.generate_max_rAB(el1, el2, el3, **kwargs)

        min_rAB = kwargs['min_rAB']
        if min_rAB is None:
            min_rAB = self.generate_min_rAB(el1, el2, el3, **kwargs)

        num_rAB = 6 + int(np.ceil((max_rAB - min_rAB) / 0.4))
        return num_rAB

    def generate_min_rCM(self, el1, el2, el3, **kwargs):
        min_rCM = 0.2
        return min_rCM

    def generate_max_rCM(self, el1, el2, el3, **kwargs):
        max_rCM = kwargs['max_rAB']
        if max_rCM is None:
            max_rCM = self.generate_max_rAB(el1, el2, el3, **kwargs)
        return max_rCM

    def generate_num_rCM(self, el1, el2, el3, **kwargs):
        max_rCM = kwargs['max_rCM']
        if max_rCM is None:
            max_rCM = self.generate_max_rCM(el1, el2, el3, **kwargs)

        min_rCM = kwargs['min_rCM']
        if min_rCM is None:
            min_rCM = self.generate_min_rCM(el1, el2, el3, **kwargs)

        num_rCM = 6 + int(np.ceil((max_rCM - min_rCM) / 0.4))
        return num_rCM

    def generate_num_theta(self, el1, el2, el3, **kwargs):
        num_theta = 24
        return num_theta


def get_3c_grids(min_rAB=None, max_rAB=None, num_rAB=None, min_rCM=None,
                 max_rCM=None, num_rCM=None, num_theta=None, verbose=True):
    Rgrid = np.exp(np.linspace(np.log(min_rAB), np.log(max_rAB), num=num_rAB,
                   endpoint=True))
    Sgrid = np.exp(np.linspace(np.log(min_rCM), np.log(max_rCM), num=num_rCM,
                   endpoint=True))
    Tgrid = np.linspace(0., np.pi, num=num_theta)

    if verbose:
        print('numRST:', num_rAB, num_rCM, num_theta)
        print('Rgrid:', Rgrid)
        print('Sgrid:', Sgrid)

    return (Rgrid, Sgrid, Tgrid)


def chgoff2c(el1, el2, **kwargs):
    atoms, xc, properties, wf_ranges = get_atoms(el1, el2, only_1c=False,
                                                 **kwargs['atom_kwargs'])
    grid_args = (el1, el2, 'chgoff2c', wf_ranges)
    grid_opt_int = kwargs['grid_opt_int'].get_parameters(*grid_args)
    grid_opt_tab = kwargs['grid_opt_tab'].get_parameters(*grid_args)

    for mapping in kwargs['aux_mappings']:
        basis = 'main' if mapping == 'mulliken' else 'auxiliary'
        calc = Offsite2cUTable(atoms[el1], atoms[el2], basis=basis,
                               timing=False)

        run_kwargs = dict(shift=kwargs['shift'], xc=xc, **grid_opt_int,
                          **grid_opt_tab)
        calc.run(**run_kwargs)
        calc.write()
    return


def chgon1c(el1, **kwargs):
    atoms, xc, properties, wf_ranges = get_atoms(el1, only_1c=True,
                                                 **kwargs['atom_kwargs'])

    for mapping in kwargs['aux_mappings']:
        basis = 'main' if mapping == 'mulliken' else 'auxiliary'
        calc = Onsite1cUTable(atoms[el1], basis=basis)

        run_kwargs = dict()
        if basis == 'auxiliary':
            run_kwargs.update(xc=xc)
        else:
            run_kwargs.update(method='analytical')

        calc.run(**run_kwargs)
        calc.write()
    return


def chgon2c(el1, el2, **kwargs):
    atoms, xc, properties, wf_ranges = get_atoms(el1, el2, only_1c=False,
                                                 **kwargs['atom_kwargs'])
    grid_args = (el1, el2, 'chgon2c', wf_ranges)
    grid_opt_int = kwargs['grid_opt_int'].get_parameters(*grid_args)
    grid_opt_tab = kwargs['grid_opt_tab'].get_parameters(*grid_args)

    for mapping in kwargs['aux_mappings']:
        basis = 'main' if mapping == 'mulliken' else 'auxiliary'
        calc = Onsite2cUTable(atoms[el1], atoms[el2], basis=basis,
                              timing=False)

        run_kwargs = dict(xc=xc, **grid_opt_int, **grid_opt_tab)
        calc.run(**run_kwargs)
        calc.write()
    return


def magoff2c(el1, el2, **kwargs):
    atoms, xc, properties, wf_ranges = get_atoms(el1, el2, only_1c=False,
                                                 **kwargs['atom_kwargs'])
    grid_args = (el1, el2, 'magoff2c', wf_ranges)
    grid_opt_int = kwargs['grid_opt_int'].get_parameters(*grid_args)
    grid_opt_tab = kwargs['grid_opt_tab'].get_parameters(*grid_args)

    for mapping in kwargs['aux_mappings']:
        basis = 'main' if mapping == 'mulliken' else 'auxiliary'
        calc = Offsite2cWTable(atoms[el1], atoms[el2], basis=basis,
                               timing=False)

        run_kwargs = dict(xc=xc, **grid_opt_int, **grid_opt_tab)
        calc.run(**run_kwargs)
        calc.write()
    return


def magon1c(el1, **kwargs):
    atoms, xc, properties, wf_ranges = get_atoms(el1, only_1c=True,
                                                 **kwargs['atom_kwargs'])

    for mapping in kwargs['aux_mappings']:
        basis = 'main' if mapping == 'mulliken' else 'auxiliary'
        calc = Onsite1cWTable(atoms[el1], basis=basis)

        run_kwargs = dict()
        if basis == 'auxiliary':
            run_kwargs.update(xc=xc)
        else:
            run_kwargs.update(method='analytical')

        calc.run(**run_kwargs)
        calc.write()
    return


def magon2c(el1, el2, **kwargs):
    atoms, xc, properties, wf_ranges = get_atoms(el1, el2, only_1c=False,
                                                 **kwargs['atom_kwargs'])

    grid_args = (el1, el2, 'magon2c', wf_ranges)
    grid_opt_int = kwargs['grid_opt_int'].get_parameters(*grid_args)
    grid_opt_tab = kwargs['grid_opt_tab'].get_parameters(*grid_args)

    for mapping in kwargs['aux_mappings']:
        basis = 'main' if mapping == 'mulliken' else 'auxiliary'
        calc = Onsite2cWTable(atoms[el1], atoms[el2], basis=basis,
                              timing=False)

        run_kwargs = dict(xc=xc, **grid_opt_int, **grid_opt_tab)
        calc.run(**run_kwargs)
        calc.write()
    return


def map1c(el1, **kwargs):
    atoms, xc, properties, wf_ranges = get_atoms(el1, only_1c=True,
                                                 **kwargs['atom_kwargs'])

    for mapping in kwargs['aux_mappings']:
        if mapping == 'giese_york':
            calc = Onsite1cMTable(atoms[el1])
            calc.run()
            calc.write()
    return


def map2c(el1, el2, **kwargs):
    atoms, xc, properties, wf_ranges = get_atoms(el1, el2, only_1c=False,
                                                 **kwargs['atom_kwargs'])

    grid_args = (el1, el2, 'map2c', wf_ranges)
    grid_opt_int = kwargs['grid_opt_int'].get_parameters(*grid_args)
    grid_opt_tab = kwargs['grid_opt_tab'].get_parameters(*grid_args)

    for mapping in kwargs['aux_mappings']:
        if mapping == 'giese_york':
            constraint_method = kwargs['giese_york_constraint_method']
            calc = Offsite2cMTable(atoms[el1], atoms[el2],
                                   constraint_method=constraint_method,
                                   timing=False)

            run_kwargs = dict(**grid_opt_int, **grid_opt_tab)
            calc.run(**run_kwargs)
            calc.write()
    return


def off2c(el1, el2, **kwargs):
    atoms, xc, properties, wf_ranges = get_atoms(el1, el2, only_1c=False,
                                                 **kwargs['atom_kwargs'])

    grid_args = (el1, el2, 'off2c', wf_ranges)
    grid_opt_int = kwargs['grid_opt_int'].get_parameters(*grid_args)
    grid_opt_tab = kwargs['grid_opt_tab'].get_parameters(*grid_args)

    calc = Offsite2cTable(atoms[el1], atoms[el2], timing=False)
    calc.run(superposition='density', xc=xc, **grid_opt_int, **grid_opt_tab)

    write_kwargs = dict()
    if el1 == el2:
        write_kwargs.update(
                eigenvalues=properties['eigenvalues'][el1],
                hubbardvalues=properties['hubbardvalues'][el1],
                occupations=properties['occupations'][el1],
                offdiagonal_H=properties['offdiagonal_H'][el1],
                offdiagonal_S=properties['offdiagonal_S'][el1],
        )
    calc.write(**write_kwargs)
    return


def off3c(el1, el2, el3, **kwargs):
    atoms, xc, properties, wf_ranges = get_atoms(el1, el2, el3, only_1c=False,
                                                 **kwargs['atom_kwargs'])

    grid_args = (el1, el2, 'off3c', wf_ranges)
    grid_opt_int = kwargs['grid_opt_int'].get_parameters(*grid_args, el3=el3)
    grid_opt_tab = kwargs['grid_opt_tab'].get_parameters(*grid_args, el3=el3)
    Rgrid, Sgrid, Tgrid = get_3c_grids(**grid_opt_tab)

    calc = Offsite3cTable(atoms[el1], atoms[el2], timing=False)
    calc.run(atoms[el3], Rgrid, Sgrid, Tgrid, xc=xc, **grid_opt_int)
    return


def on2c(el1, el2, **kwargs):
    atoms, xc, properties, wf_ranges = get_atoms(el1, el2, only_1c=False,
                                                 **kwargs['atom_kwargs'])

    grid_args = (el1, el2, 'on2c', wf_ranges)
    grid_opt_int = kwargs['grid_opt_int'].get_parameters(*grid_args)
    grid_opt_tab = kwargs['grid_opt_tab'].get_parameters(*grid_args)

    calc = Onsite2cTable(atoms[el1], atoms[el2], timing=False)
    calc.run(xc=xc, shift=kwargs['shift'], **grid_opt_int, **grid_opt_tab)
    calc.write()
    return


def on3c(el1, el2, el3, **kwargs):
    atoms, xc, properties, wf_ranges = get_atoms(el1, el2, el3, only_1c=False,
                                                 **kwargs['atom_kwargs'])

    grid_args = (el1, el2, 'on3c', wf_ranges)
    grid_opt_int = kwargs['grid_opt_int'].get_parameters(*grid_args, el3=el3)
    grid_opt_tab = kwargs['grid_opt_tab'].get_parameters(*grid_args, el3=el3)
    Rgrid, Sgrid, Tgrid = get_3c_grids(**grid_opt_tab)

    on3c = Onsite3cTable(atoms[el1], atoms[el2], timing=False)
    on3c.run(atoms[el3], Rgrid, Sgrid, Tgrid, xc=xc, **grid_opt_int)
    return


def rep2c(el1, el2, **kwargs):
    atoms, xc, properties, wf_ranges = get_atoms(el1, el2, only_1c=False,
                                                 **kwargs['atom_kwargs'])

    grid_args = (el1, el2, 'rep2c', wf_ranges)
    grid_opt_int = kwargs['grid_opt_int'].get_parameters(*grid_args)
    grid_opt_tab = kwargs['grid_opt_tab'].get_parameters(*grid_args)

    calc = Repulsion2cTable(atoms[el1], atoms[el2], timing=False)
    calc.run(xc=xc, shift=kwargs['shift'], **grid_opt_int, **grid_opt_tab)
    calc.write()
    return


def rep3c(el1, el2, el3, **kwargs):
    atoms, xc, properties, wf_ranges = get_atoms(el1, el2, el3, only_1c=False,
                                                 **kwargs['atom_kwargs'])

    grid_args = (el1, el2, 'rep3c', wf_ranges)
    grid_opt_int = kwargs['grid_opt_int'].get_parameters(*grid_args, el3=el3)
    grid_opt_tab = kwargs['grid_opt_tab'].get_parameters(*grid_args, el3=el3)
    Rgrid, Sgrid, Tgrid = get_3c_grids(**grid_opt_tab)

    calc = Repulsion3cTable(atoms[el1], atoms[el2], timing=False)
    calc.run(atoms[el3], Rgrid, Sgrid, Tgrid, xc=xc, **grid_opt_int)
    return


if __name__ == '__main__':
    main()
