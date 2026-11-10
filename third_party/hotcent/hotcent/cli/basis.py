#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import os
import yaml
from argparse import ArgumentParser
from datetime import datetime
from hotcent import __version__
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT
from hotcent.siesta_ion import write_ion
from hotcent.utils import get_file_checksum, verify_chemical_symbols


def parse_arguments():
    description = """
    Generate a main and auxiliary basis for the given element.

    Note: existing output files will be overwritten.
    """
    parser = ArgumentParser(description=description)
    parser.add_argument('symbol', type=str, help='Element symbol (e.g. "Si").')
    parser.add_argument('--aux-basis', type=str, help='Size of the auxiliary '
                        'basis set. For e.g. a double-zeta auxiliary basis '
                        'with angular momenta up to d, specify --aux-basis=2D.'
                        ' The default is 2P for H and He and 3D for all other '
                        'elements.')
    parser.add_argument('--aux-cation-charges', type=str,
                        help='Equivalent of the --cation-charges keyword '
                             'for the higher-zeta auxiliary basis functions.')
    parser.add_argument('--aux-cation-potentials', type=str,
                        help='Equivalent of the --cation-potentials keyword '
                             'for the higher-zeta auxiliary basis functions.')
    parser.add_argument('--aux-subshell', type=str, help='Main basis subshell '
                        'from which the auxiliary radial functions will be '
                        'derived. To e.g. select the 3p subshell for Si '
                        'instead of the default choice (the valence subshell '
                        'with the lowest angular momentum, i.e. 3s for Si), '
                        'specify --aux-subshell=3p.')
    parser.add_argument('--aux-tail-degree', type=int, default=None,
                        help='Equivalent of the --tail-degree keyword for the '
                             'higher-zeta auxiliary basis functions. '
                             'The default None means that an appropriate '
                             'degree will be chosen.')
    parser.add_argument('--aux-tail-norms', type=str, default='0.2,0.4',
                        help='Equivalent of the --tail-norms keyword for the '
                             'higher-zeta auxiliary basis functions '
                             '(default: 0.2,0.4).')
    parser.add_argument('--aux-zeta-method', type=str, default='cation',
                        choices=['cation', 'split_valence'],
                        help='Equivalent of the --zeta-method keyword for the '
                             'higher-zeta auxiliary basis functions.')
    parser.add_argument('--basis', '-b', type=str, default='dzp',
                        choices=['sz', 'szp', 'dz', 'dszp', 'dzp', 'dz2p',
                                 'tz', 'tzp', 'tz2p', 'tz3p'],
                        help='Size of the main basis set (default: dzp).')
    parser.add_argument('--cation-charges', type=str,
                        help='Comma-separated charges to use for the '
                             'higher-zeta functions in the "cation" zeta '
                             'scheme (one underscore-separated list for '
                             'each valence subshell). By default charges '
                             'of 2 and 4 are used for s- and p-subshells '
                             'and 3 and 6 for d- and f-subshells.')
    parser.add_argument('--cation-potentials', type=str,
                        help='Comma-separated potential types to use for the '
                             'higher-zeta functions in the "cation" zeta '
                             'scheme (one underscore-separated list for '
                             'each valence subshell). By default the local '
                             'part of the pseudopotential is used for s- and '
                             'p-subshells and point-charge potentials for '
                             'd- and f-subshells.')
    parser.add_argument('--configuration', '-c', type=str, required=True,
                        help='Electronic configuration, e.g. "[Ne],3s2,3p2".')
    parser.add_argument('--energy-shift', '-E', type=str, default='0.2',
                        help='Comma-separated energy shifts in eV for use '
                             'with the "energy_shift_user" and '
                             '"energy_shift_user_sc" rcut approaches (one '
                             'value for each valence subshell). If only a '
                             'single value is given (as in the default value '
                             'of 0.2), it is applied to all subshells.')
    parser.add_argument('-f', '--xcfunctional', type=str, default='LDA',
                        help='Exchange-correlation functional. Default: LDA.')
    parser.add_argument('--label', help='Label to use in the output file '
                        'names, which will be "<Symbol>[.<label>].ion/json" '
                        'and "<Symbol>_Rnl/Anl[.<label>].png".')
    parser.add_argument('--l-pol', type=int, help='Angular momentum for '
                        'polarization functions. By default the lowest value '
                        'is taken which is not present in the valence states.')
    parser.add_argument('--plot', action='store_true', help='Make a plot '
                        'of the generated radial basis functions (requires '
                        'matplotlib).')
    parser.add_argument('--pseudo-format', choices=['psf', 'psml'],
                        help='The file extension for the pseudopotentials. '
                        'The expected file name corresponds to '
                        '"<Symbol>[.<pseudo-label>].<pseudo-format>". '
                        'The default is to try to autodetect the extension.')
    parser.add_argument('--pseudo-label', help='Label to use when searching '
                        'for pseudopotential files (see --pseudo-format).')
    parser.add_argument('--pseudo-local-component', default='siesta',
                        help='How to construct the local part of the '
                        'pseudopotential. Common choices are "siesta" '
                        '(which is the default) and "vanderbilt".')
    parser.add_argument('--pseudo-path', default='.', help='Path to the '
                        'directory where the pseudopotential files are stored '
                        '(default: ".").')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Do not print to standard output.')
    rchar_pol_grp = parser.add_mutually_exclusive_group()
    rchar_pol_grp.add_argument('--rchar-pol', help='Characteristic radius in '
                               'Bohr radii to use for polarization functions.')
    rchar_pol_grp.add_argument('--rchar-pol-rel', type=float, default=None,
                               help='Scaling factor to determine the '
                                    'characteristic radius for polarization '
                                    'functions (default: 0.33 for p-type '
                                    'functions, 0.25 otherwise).')
    parser.add_argument('--rcut', type=str, help='Comma-separated cutoff radii'
                        ' in Bohr radii for every valence subshell (in the '
                        'same order as the -v/--valence keyword). Only used '
                        '(and required) for --rcut-approach=user.')
    parser.add_argument('--rcut-approach', choices=['energy_shift_user',
                        'energy_shift_user_sc', 'user'],
                        default='energy_shift_user',
                        help='Approach for obtaining the cutoff radii for the '
                             '(minimal) basis functions. With the "user" '
                             'approach the cutoff radii need to be supplied '
                             'via the --rcut keyword. With the '
                             '"energy_shift_user" and "energy_shift_user_sc" '
                             'approaches the radii are determined by the '
                             'shifts given via the --energy-shift option. '
                             'Default: "energy_shift_user".')
    parser.add_argument('--rmin', help='Smallest radius in the radial grid, '
                        'to be used in case the default setting does not '
                        'yield well-behaved results.')
    parser.add_argument('--tail-degree', type=int, default=None,
                        help='Degree of the polynomial to use in the '
                             '"split_valence" scheme for generating the '
                             'higher-zeta functions. The default None means '
                             'that an appropriate degree will be chosen.')
    parser.add_argument('--tail-norms', '-T', type=str, default='0.16,0.3',
                        help='Comma-separated tail norms to use in the '
                             '"split_valence" scheme for generating the '
                             'higher-zeta functions (default: 0.16,0.3).')
    parser.add_argument('--valence', '-v', type=str, required=True,
                        help='Comma-separated subshells to include in the '
                             '(minimal) basis (e.g. "3s,3p").')
    parser.add_argument('--vconf-amplitude', type=float, default=12.0,
                        help='Soft confinement amplitude in Hartree (default: '
                             '12.0).')
    parser.add_argument('--vconf-rstart-rel', type=float, default=0.6,
                        help='Soft confinement start radius, relative to the '
                             'cutoff radius (default: 0.6).')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__}')
    parser.add_argument('--zeta-method', type=str, default='cation',
                        choices=['cation', 'split_valence'],
                        help='Method for constructing higher-zeta basis '
                             'functions (default: "cation").')
    args = parser.parse_args()
    return args


def find_cutoff_radii(symbol, pp, rcut_approach, shifts, amp, x_ri, atom_kwargs,
                      verbose=True):
    conf_kwargs = dict(amp=amp, x_ri=x_ri, rcuts=50.)
    atom = get_atom(symbol, pp, atom_kwargs, verbose=verbose, **conf_kwargs)

    conf_kwargs = dict(amp=amp, x_ri=x_ri)
    valence = atom.valence.copy()
    rcuts = {}

    if rcut_approach == 'energy_shift_user':
        for nl, shift in zip(valence, shifts):
            rcuts[nl] = atom.find_cutoff_radius(nl, energy_shift=shift,
                                                neglect_density_change=True,
                                                **conf_kwargs)

    elif rcut_approach == 'energy_shift_user_sc':
        if len(valence) == 1:
            nl = valence[0]
            rcuts[nl] = atom.find_cutoff_radius(nl, energy_shift=shifts[0],
                                                neglect_density_change=False,
                                                **conf_kwargs)
        elif len(valence) == 2:
            rcuts.update(atom.find_cutoff_radii(*valence,
                                                energy_shifts=shifts,
                                                **conf_kwargs))
        else:
            msg = 'The "energy_shift_user_sc" rcut approach is not ' + \
                    'available for more than two valence subshells'
            raise NotImplementedError(msg)
    return rcuts


def find_polarization_radius(symbol, pp, amp, x_ri, rchar_pol_rel, atom_kwargs,
                             l_pol=None, verbose=True):
    conf_kwargs = dict(amp=amp, x_ri=x_ri, rcuts=50.)
    atom = get_atom(symbol, pp, atom_kwargs, verbose=verbose, **conf_kwargs)

    if l_pol is None:
       l_pol = atom.get_polarization_angular_momentum()

    if rchar_pol_rel is None:
        fraction = 0.33 if l_pol == 1 else 0.25
    else:
        fraction = rchar_pol_rel

    r_pol = atom.find_polarization_radius(fraction=fraction, amp=amp,
                                          x_ri=x_ri, l_pol=l_pol)
    return r_pol


def get_zeta_kwargs(valence, arg_zeta_method, arg_tail_degree, arg_tail_norms,
                    arg_cation_chg, arg_cation_pot):
    cation_charges = dict()
    cation_potentials = dict()

    for i, nl in enumerate(valence):
        if arg_cation_chg is None:
            chg = dict(s=[2., 4.], p=[2., 4.], d=[3., 6.], f=[3., 6.])[nl[1]]
        else:
            chg = list(map(float, arg_cation_chg.split(',')[i].split('_')))

        if arg_cation_pot is None:
            pot = dict(s=['pseudo']*2, p=['pseudo']*2,
                       d=['point']*2, f=['point']*2)[nl[1]]
        else:
            pot = arg_cation_pot.split(',')[i].split('_')

        for j in range(2):
            nlz = nl + '+'*(j + 1)
            cation_charges[nlz] = chg[j]
            cation_potentials[nlz] = pot[j]

    tail_degree = None if arg_tail_degree is None else int(arg_tail_degree)
    tail_norms = list(map(float, arg_tail_norms.split(',')))

    zeta_kwargs = dict(
        cation_charges=cation_charges,
        cation_potentials=cation_potentials,
        degree=tail_degree,
        tail_norms=tail_norms,
        zeta_method=arg_zeta_method,
    )
    return zeta_kwargs


def generate_bases(symbol, pp, atom_kwargs, basis_kwargs, conf_kwargs,
                   aux_basis_kwargs, verbose=True):
    atom = get_atom(symbol, pp, atom_kwargs, verbose=verbose, **conf_kwargs)
    atom.run()
    atom.generate_nonminimal_basis(**basis_kwargs)
    atom.pp.build_projectors(atom)
    atom.generate_auxiliary_basis(**aux_basis_kwargs)
    return atom


def get_atom(symbol, pp, atom_kwargs, verbose, amp, x_ri, rcuts):
    wf_conf = {}
    for nl in atom_kwargs['valence']:
        rc = rcuts if isinstance(rcuts, (int, float)) else rcuts[nl]
        wf_conf[nl] = SoftConfinement(amp=amp, x_ri=x_ri, rc=rc)

    atom = PseudoAtomicDFT(symbol, pp, txt='-' if verbose else None,
                           wf_confinement=wf_conf, **atom_kwargs)
    return atom


def sanity_check_atom(symbol, pp, amp, x_ri, atom_kwargs, verbose=True):
    conf_kwargs = dict(amp=amp, x_ri=x_ri, rcuts=50.)
    atom = get_atom(symbol, pp, atom_kwargs, verbose=verbose, **conf_kwargs)
    atom.run()

    for nl in atom_kwargs['valence']:
        H1c, S1c = atom.get_onecenter_integrals(nl, nl)
        msg = 'Suspicious eigenvalue {0:.3f} for {1}_{2}. Consider ' + \
              're-running with a smaller rmin parameter (e.g. 1e-4).'
        assert abs(H1c) < 50., msg.format(H1c, symbol, nl)
    return


def write_yaml(symbol, atom_kwargs, basis_kwargs, conf_kwargs, pp_kwargs,
               aux_basis_kwargs, stem):
    timestamp = datetime.now().replace(microsecond=0).isoformat()
    meta_kwargs = dict(
        symbol=symbol,
        timestamp=timestamp,
        hotcent_version=__version__,
    )

    setup = dict(
        atom=atom_kwargs,
        auxiliary_basis=aux_basis_kwargs,
        basis=basis_kwargs,
        confinement=conf_kwargs,
        meta=meta_kwargs,
        pseudopotential=pp_kwargs,
    )

    filename = '{0}.yaml'.format(stem)

    with open(filename, 'w') as f:
        yaml.dump(setup, f)
    return


def main():
    args = parse_arguments()

    symbol = args.symbol
    verify_chemical_symbols(symbol)

    # Main basis settings

    configuration = args.configuration.replace(',', ' ')
    pseudo_label = '' if args.pseudo_label is None else '.' + args.pseudo_label
    rmin = None if args.rmin is None else float(args.rmin)
    r_pol = None if args.rchar_pol is None else float(args.rchar_pol)
    verbose = not args.quiet
    valence = args.valence.split(',')
    amp = args.vconf_amplitude
    x_ri = args.vconf_rstart_rel

    with_polarization = args.basis.endswith('p')

    if args.rcut_approach == 'user':
        assert args.rcut is not None, 'For --rcut-approach=user, ' + \
               'cutoff radii need to be supplied via --rcut.'

        rcuts = list(map(float, args.rcut.split(',')))
        if len(rcuts) == 1:
            rcuts *= len(valence)
        else:
            assert len(rcuts) == len(valence), 'When supplying more than ' + \
                   'one cutoff radius, the number of radii must be equal ' + \
                   'the number of valence subshells.'
        rcuts = {nl: rcut for nl, rcut in zip(valence, rcuts)}
    else:
        rcuts = None
        shifts = list(map(float, args.energy_shift.split(',')))

        if args.rcut_approach in ['energy_shift_user', 'energy_shift_user_sc']:
            if len(shifts) == 1:
                shifts *= len(valence)
            else:
                assert len(shifts) == len(valence), 'When supplying more ' + \
                       'than one energy shift, the number of shifts must ' + \
                       'be equal to the number of valence subshells.'
        else:
            raise NotImplementedError(args.rcut_approach)

    pp_kwargs = dict(
        local_component=args.pseudo_local_component,
        valence=valence,
        with_polarization=with_polarization,
    )

    if args.pseudo_format is None:
        pp_formats = ['psf', 'psml']
    else:
        pp_formats = [args.pseudo_format]

    pp_path = None
    pp_filename = None

    for pp_format in pp_formats:
        filename = f'{symbol}{pseudo_label}.{pp_format}'
        path = os.path.join(args.pseudo_path, filename)
        if os.path.exists(path):
            assert pp_path is None, \
                   'Use the --pseudo-format option to select either ' + \
                   f'"{pp_path}" or "{path}" (since both files exist)'
            pp_path = path
            pp_filename = filename

    assert pp_path is not None and pp_filename is not None, \
           'No pseudopotential could be found'

    pp = KleinmanBylanderPP(pp_path, txt='-' if verbose else None, **pp_kwargs)

    atom_kwargs = dict(
        configuration=configuration,
        nodegpts=1000,
        perturbative_confinement=True,
        rmin=rmin,
        scalarrel=True,
        valence=valence,
        xc=args.xcfunctional,
    )

    sanity_check_atom(symbol, pp, amp, x_ri, atom_kwargs, verbose=verbose)

    if with_polarization and r_pol is None:
        r_pol = find_polarization_radius(symbol, pp, amp, x_ri,
                                         args.rchar_pol_rel, atom_kwargs,
                                         l_pol=args.l_pol, verbose=verbose)

    basis_kwargs = get_zeta_kwargs(valence, args.zeta_method, args.tail_degree,
                                   args.tail_norms, args.cation_charges,
                                   args.cation_potentials)
    basis_kwargs.update(
        l_pol=args.l_pol,
        r_pol=r_pol,
        size=args.basis,
    )

    if rcuts is None:
        rcuts = find_cutoff_radii(symbol, pp, args.rcut_approach, shifts,
                                  amp, x_ri, atom_kwargs, verbose=verbose)

    conf_kwargs = dict(amp=amp, x_ri=x_ri, rcuts=rcuts)

    # Auxiliary basis settings
    aux_basis_kwargs = get_zeta_kwargs(valence, args.aux_zeta_method,
                                       args.aux_tail_degree,
                                       args.aux_tail_norms,
                                       args.aux_cation_charges,
                                       args.aux_cation_potentials)

    aux_lmax = 1 if symbol in ['H', 'He'] else 2
    aux_nzeta = 2 if symbol in ['H', 'He'] else 3

    aux_basis_kwargs.update(
        degree=None,
        lmax=aux_lmax,
        nzeta=aux_nzeta,
        subshell=None,
    )

    parse_err = 'Cannot parse {0} entry: {1}'

    if args.aux_basis is not None:
        val = args.aux_basis[0]
        assert val.isdigit(), parse_err.format('aux-basis', args.aux_basis)
        aux_basis_kwargs.update(nzeta=int(val))

        val = args.aux_basis[1]
        assert val in 'SPDF', parse_err.format('aux-basis', args.aux_basis)
        aux_basis_kwargs.update(lmax='SPDF'.index(val))

    if args.aux_subshell is not None:
        assert args.aux_subshell in valence, \
               'Auxiliary basis subshell must be chosen from the minimal basis'
        aux_basis_kwargs.update(subshell=args.aux_subshell)

    # Basis generation and plotting
    atom = generate_bases(symbol, pp, atom_kwargs, basis_kwargs, conf_kwargs,
                          aux_basis_kwargs=aux_basis_kwargs, verbose=verbose)

    label = '' if args.label is None else '.'+args.label
    stem = symbol + label

    if args.plot:
        fmt = '{0}_{1}{2}.png'
        atom.plot_Rnl(filename=fmt.format(symbol, 'Rnl', label))
        atom.plot_Anl(filename=fmt.format(symbol, 'Anl', label))

    write_ion(atom, label=stem)

    pp_kwargs.update(
        filename=pp_filename,
        sha256sum=get_file_checksum(pp_path, algorithm='sha256'),
    )
    write_yaml(symbol, atom_kwargs, basis_kwargs, conf_kwargs, pp_kwargs,
               aux_basis_kwargs, stem)
    return


if __name__ == '__main__':
    main()
