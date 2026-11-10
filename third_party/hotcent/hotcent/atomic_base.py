#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
""" Definition of the base class for atomic DFT
calculations.

The code below draws heavily from the Hotbit code
written by Pekka Koskinen (https://github.com/pekkosk/
hotbit/blob/master/hotbit/parametrization/atom.py).
"""
import os
import sys
import collections
import numpy as np
from scipy.optimize import minimize
from ase.data import atomic_numbers, covalent_radii
from ase.units import Bohr
from hotcent.confinement import Confinement, ZeroConfinement
from hotcent.fluctuation_basis import AuxiliaryBasis
from hotcent.interpolation import build_interpolator, CubicSplineFunction
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.phillips_kleinman import PhillipsKleinmanPP
from hotcent.timing import Timer
from hotcent.xc import LibXC, XC_PW92
try:
    import matplotlib.pyplot as plt
except:
    plt = None


NOT_SOLVED_MESSAGE = 'A required attribute is missing. ' \
                     'Please call the run() method first.'


class AtomicBase:
    """
    Base class for atomic DFT calculators.

    Parameters
    ----------
    symbol : str
        Chemical symbol.
    configuration : str
        Electronic configuration, e.g. '[He] 2s2 2p2'.
    valence : list
        Valence subshells, e.g. ['2s','2p'].
    confinement : Confinement-like object, optional
        Confinement potential for the electron density
        (see hotcent.confinement). The default None means
        that means no density confinement will be applied.
    wf_confinement : dict, optional
        Dictionary with confinement potentials for the
        valence subshells. If None, the same confinement will
        be used as for the electron density. If a certain
        hotcent.confinement.Confinement instance is provided,
        this will be applied to all valence states. If a
        dictionary is provided, it is supposed to look like
        this: {nl: <a certain Confinement instance, or None>
        for each nl in your set of valence states}.
        For missing entries, no confinement will be applied.
    scalarrel : bool, optional
        Whether to use scalar relativistic corrections (default:
        False). Setting it to True is strongly recommended for
        all-electron calculations of heavier elements
        (atomic numbers above, say, 24).
    mix : float, optional
        Mixing coefficient for the effective potential during
        the self-consistency cycle.
    maxiter : int, optional
        Maximum number of iterations for self-consistency.
    rmax : float, optional
        Radial cutoff in Bohr.
    nodegpts : int, optional
        Total number of grid points is nodegpts times the max
        number of antinodes for all subshells.
    timing : bool, optional
        Whether to produce a timing summary (default: False).
    verbose : bool, optional
        Whether to increase verbosity during iterations
        (default: False).
    txt : str or None or file handle, optional
        Where output should be printed. Use '-' for stdout
        (default), None for /dev/null, any other string for a
        text file, or a file handle.
    """
    def __init__(self,
                 symbol,
                 configuration='',
                 valence=[],
                 confinement=None,
                 wf_confinement=None,
                 scalarrel=False,
                 mix=0.2,
                 maxiter=200,
                 rmax=100.0,
                 nodegpts=500,
                 timing=False,
                 verbose=False,
                 txt='-'):
        self.symbol = symbol
        self.scalarrel = scalarrel
        self.mix = mix
        self.maxiter = maxiter
        self.rmax = rmax
        self.nodegpts = nodegpts
        self.timing = timing
        self.verbose = verbose

        if txt is None:
            self.txt = open(os.devnull, 'w')
        elif isinstance(txt, str):
            if txt == '-':
                self.txt = sys.stdout
            else:
                self.txt = open(txt, 'a')
        else:
            self.txt = txt

        # Process the choice of labels (ordering by n and then by l)
        def subshell_order(nl):
            n, l = nl2tuple(nl)
            return 100*n + l

        assert len(valence) > 0, 'Please select the valence subshells'
        self.valence = sorted(set(valence), key=subshell_order)
        assert all([nl in valence for nl in self.valence]), \
               f'Problem processing valence "{valence}"'

        self.set_confinement(confinement)
        self.set_wf_confinement(wf_confinement)
        self.rcutnl = {}
        for nl, conf in self.wf_confinement.items():
            if nl in self.valence and hasattr(conf, 'rc'):
                self.rcutnl[nl] = conf.rc

        self.timer = Timer('Atomic', txt=self.txt, enabled=self.timing)

        self.Z = atomic_numbers[self.symbol]

        assert len(configuration) > 0, "Specify the electronic configuration!"
        self.set_configuration(configuration)

        self.maxl = 9
        self.maxn = 9
        self.unlg = {}
        self.Rnlg = {}
        self.unl_fct = {nl: None for nl in self.configuration}
        self.Rnl_fct = {nl: None for nl in self.configuration}
        self.veff_fct = None
        self.dens_fct = None
        self.densval_fct = None
        self.vhar_fct = None
        self.vharval_fct = None
        self.vna_fct = None
        self.energies = {}
        self.solved = False
        self.basis_sets = [self.valence.copy()]
        self.basis_size = 'sz'
        self.aux_basis = AuxiliaryBasis()

        # Set default 'pseudopotential':
        self.pp = PhillipsKleinmanPP(self.symbol)

    def set_configuration(self, configuration):
        """ Set the electron configuration

        configuration: e.g. '[He] 2s2 2p2'
        """
        self.configuration = {}
        noble_conf = {'He':{'1s':2}}
        noble_conf['Ne'] = dict({'2s':2, '2p':6}, **noble_conf['He'])
        noble_conf['Ar'] = dict({'3s':2, '3p':6}, **noble_conf['Ne'])
        noble_conf['Kr'] = dict({'3d':10, '4s':2, '4p':6}, **noble_conf['Ar'])
        noble_conf['Xe'] = dict({'4d':10, '5s':2, '5p':6}, **noble_conf['Kr'])
        noble_conf['Rn'] = dict({'4f':14, '5d':10, '6s':2, '6p':6},
                                **noble_conf['Xe'])
        noble_conf['Og'] = dict({'5f':14, '6d':10, '7s':2, '7p':6},
                                **noble_conf['Rn'])

        for term in configuration.split():
            if term[0] == '[' and term[-1] == ']':
                core = term[1:-1]
                assert core in noble_conf, "[Core] config is not a noble gas!"
                conf = noble_conf[core]
            else:
                conf = {term[:2]: float(term[2:])}
            self.configuration.update(conf)

    def set_confinement(self, confinement):
        if confinement is None:
            self.confinement = ZeroConfinement()
        else:
            self.confinement = confinement

    def set_wf_confinement(self, wf_confinement):
        if wf_confinement is None:
            self.wf_confinement = {}
        elif isinstance(wf_confinement, Confinement):
            self.wf_confinement = {nl: wf_confinement for nl in self.valence}
        elif isinstance(wf_confinement, dict):
            self.wf_confinement = {}
            for nl in self.valence:
                if nl not in wf_confinement or wf_confinement[nl] is None:
                    self.wf_confinement[nl] = ZeroConfinement()
                else:
                    self.wf_confinement[nl] = wf_confinement[nl]
        else:
            msg = "Don't know what to do with the provided wf_confinement:\n"
            msg += str(wf_confinement)
            raise ValueError(msg)

    def run(self, **kwargs):
        """ Child classes must implement a run() method which,
        in turn, is supposed to set the following attributes:

        self.solved: whether the calculations are considered to be done
        self.energies: a dictionary with the total energy and its contributions
        self.rgrid: an array with the radial grid points g
        self.dens: an array with the electron density on the radial grid
        self.vhar: an array with the Hartree potential on the radial grid
        self.veff: an array with the effective potential on the radial grid
                   (note: veff = vnuc + vhar + vxc + vconf)
        self.enl: a {'nl': eigenvalue} dictionary
        self.Rnlg: a {'nl': R_nl(g) array} dictionary
        self.unlg: a {'nl': u_nl(g) array} dictionary (u_nl = R_nl / r)
        """
        raise NotImplementedError('Child class must implement run() method!')

    def __getstate__(self):
        """ Return dictionary of all pickable items. """
        d = self.__dict__.copy()
        for key in self.__dict__:
            if isinstance(d[key], collections.Callable):
                d.pop(key)
        d.pop('out')
        return d

    def get_symbol(self):
        """ Return atom's chemical symbol. """
        return self.symbol

    def get_number_of_electrons(self, only_valence=False):
        if only_valence:
            return sum([self.configuration[nl] for nl in self.valence])
        else:
            return sum(self.configuration.values())

    def list_states(self):
        """ List all potential states {(n,l,'nl')}. """
        states = []
        for l in range(self.maxl + 1):
            for n in range(1, self.maxn + 1):
                nl = tuple2nl(n, l)
                if nl in self.configuration:
                    states.append((n, l, nl))
        return states

    def get_valence_subshells(self):
        """ Get list of valence subshells, e.g. ['2s','2p'] """
        return self.valence

    def get_energy(self):
        assert self.solved, NOT_SOLVED_MESSAGE
        return self.energies['total']

    def get_epsilon(self, nl):
        """ E.g. get_eigenvalue('2p') """
        assert self.solved, NOT_SOLVED_MESSAGE
        return self.enl[nl]

    def get_valence_energies(self):
        """ Return list of valence eigenenergies. """
        assert self.solved, NOT_SOLVED_MESSAGE
        return [(nl, self.enl[nl]) for nl in self.valence]

    def get_eigenvalue(self, nl):
        return self.get_epsilon(nl)

    def get_wf_range(self, nl, fractional_limit=1e-7):
        """ Return the maximum r for which |R(r)| is
        less than fractional_limit * max(|R(r)|) """
        assert self.solved, NOT_SOLVED_MESSAGE
        wfmax = np.nanmax(np.abs(self.Rnlg[nl]))
        for r, wf in zip(self.rgrid[-1::-1], self.Rnlg[nl][-1::-1]):
            if abs(wf) > fractional_limit * wfmax:
                return r

    def Rnl(self, r, nl, der=0):
        """ Rnl(r, '2p') """
        assert self.solved, NOT_SOLVED_MESSAGE
        if self.Rnl_fct[nl] is None:
            rc = self.rcutnl[nl] if nl in self.rcutnl else None
            self.Rnl_fct[nl] = build_interpolator(self.rgrid, self.Rnlg[nl], rc)
        return self.Rnl_fct[nl](r, der=der)

    def unl(self, r, nl, der=0):
        """ unl(r, '2p') = Rnl(r, '2p') / r """
        assert self.solved, NOT_SOLVED_MESSAGE
        if self.unl_fct[nl] is None:
            rc = self.rcutnl[nl] if nl in self.rcutnl else None
            self.unl_fct[nl] = build_interpolator(self.rgrid, self.unlg[nl], rc)
        return self.unl_fct[nl](r, der=der)

    def electron_density(self, r, der=0, only_valence=False):
        """ Return the all-electron density at r. """
        assert self.solved, NOT_SOLVED_MESSAGE

        if self.densval_fct is None or self.dens_fct is None:
            # Find the largest (minimal basis) wave-function cutoff radius
            rcs = [self.rcutnl[nl] for nl in self.valence if nl in self.rcutnl]
            rc = max(rcs) if len(rcs) > 0 else None
            self.dens_fct = build_interpolator(self.rgrid, self.dens, rc)
            self.densval_fct = build_interpolator(self.rgrid, self.densval, rc)

        if only_valence:
            return self.densval_fct(r, der=der)
        else:
            return self.dens_fct(r, der=der)

    def nuclear_potential(self,r):
        return -self.Z / r

    def neutral_atom_potential(self, r):
        """ Return the so-called 'neutral atom potential' at r, defined as
        the sum of the Hartree potential and the local pseudopotential
        (doi:10.1103/PhysRevB.40.3979).
        """
        assert self.solved, NOT_SOLVED_MESSAGE

        if self.vna_fct is None:
            rcs = [self.rcutnl[nl] for nl in self.valence if nl in self.rcutnl]
            rc = max(rcs) if len(rcs) > 0 else None
            vna = self.hartree_potential(self.rgrid, only_valence=False)
            vna += self.pp.local_potential(self.rgrid)
            self.vna_fct = build_interpolator(self.rgrid, vna, rc)

        return self.vna_fct(r)

    def effective_potential(self, r, der=0):
        """ Return effective potential at r or its derivatives. """
        assert self.solved, NOT_SOLVED_MESSAGE
        if self.veff_fct is None:
            self.veff_fct = build_interpolator(self.rgrid, self.veff)
        return self.veff_fct(r, der=der)

    def hartree_potential(self, r, only_valence=False):
        """ Return the Hartree potential at r. """
        assert self.solved, NOT_SOLVED_MESSAGE
        if only_valence:
            if self.vharval_fct is None:
                self.vharval_fct = build_interpolator(self.rgrid, self.vharval)
            return self.vharval_fct(r)
        else:
            if self.vhar_fct is None:
                self.vhar_fct = build_interpolator(self.rgrid, self.vhar)
            return self.vhar_fct(r)

    def plot_Rnl(self, filename=None, only_valence=True):
        """
        Plot the radial parts of the main basis functions with matplotlib.

        Parameters
        ----------
        only_valence : bool, optional
            Whether to only plot the valence subshells or also the core ones.

        Other parameters
        ----------------
        See plot_radial_functions().
        """
        basis = 'valence' if only_valence else 'core+valence'
        self.plot_radial_functions(basis, filename=filename)
        return

    def plot_Anl(self, filename=None):
        """
        Plot the radial parts of the auxiliary basis functions with matplotlib.

        Parameters
        ----------
        See plot_radial_functions().
        """
        self.plot_radial_functions('auxiliary', filename=filename)
        return

    def plot_radial_functions(self, basis, filename=None):
        """
        Plot the radial parts of the chosen basis functions with matplotlib.

        Parameters
        ----------
        basis : str
            The basis from which to gather the radial functions
            ('valence', 'core+valence' or 'auxiliary').
        filename : str, optional
            Output file name. Default: <symbol>_Anl.pdf for the
            auxiliary basis and <symbol>_Rnl.pdf otherwise.
        """
        assert plt is not None, 'Matplotlib could not be imported!'
        assert self.solved, NOT_SOLVED_MESSAGE

        if len(self.rcutnl) > 0:
            rmax = max(self.rcutnl.values())
        else:
            rmax = 3 * covalent_radii[self.Z] / Bohr

        ri = np.where(self.rgrid < rmax)[0][-1]

        labels_core, labels_val, labels_aux = [], [], []

        if basis == 'core+valence':
            prefix = 'R'
            labels_val += self.basis_sets[0]

            for x in self.list_states():
                nl = x[2]
                if nl not in labels_val:
                    labels_core.append(nl)

        elif basis == 'valence':
            prefix = 'R'
            labels_val += self.basis_sets[0]

        elif basis == 'auxiliary':
            prefix = 'A'
            assert self.aux_basis.lmax is not None, \
                   'The auxiliary basis has not been generated yet'
            labels_aux += self.aux_basis.get_angular_momenta()

        labels = labels_core + labels_val + labels_aux

        ncol = 2
        nrow = len(labels)
        fig = plt.figure(figsize=(8, 2*nrow), dpi=400)
        fig.subplots_adjust(hspace=0, wspace=0)

        i = 1
        for label in labels:
            if label in labels_val:
                if label in self.valence:
                    # minimal basis function or higher-zeta variant
                    colors = ['tab:blue', 'tab:orange', 'tab:green']
                else:
                    # polarization function
                    colors = ['tab:purple', 'tab:pink']
                keys = [nl for valence in self.basis_sets
                        for nl in valence if nl[:2] == label]
                yvals = [self.Rnlg[key] for key in keys]
                subscripts = keys
            elif label in labels_aux:
                colors = ['tab:red', 'tab:gray', 'tab:brown']
                keys = [(nl, label)
                        for nl in self.aux_basis.select_radial_functions()]
                yvals = [self.aux_basis.Anlg[key] for key in keys]
                subscripts = ['{0},\\ell={1}'.format(*key) for key in keys]
            elif label in labels_core:
                colors = ['darkslategray']
                keys = [label]
                yvals = [self.Rnlg[key] for key in keys]
                subscripts = keys

            # As a function of grid points
            ax = plt.subplot(nrow, ncol, i)
            for j in range(len(keys)):
                ax.plot(yvals[j], colors[j])
                plt.text(0.95, 0.8-0.2*j,
                         r'$%s_\mathrm{%s}$' % (prefix, subscripts[j]),
                         horizontalalignment='right',
                         transform=ax.transAxes, size=15, color=colors[j])

            plt.xticks(size=5)
            ax.grid(ls='--')
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel('grid point index [-]', size=8)
            i += 1

            # As a function of radius
            ax = plt.subplot(nrow, ncol, i)
            for j in range(len(keys)):
                ax.plot(self.rgrid[:ri], yvals[j][:ri], colors[j])
                plt.text(0.95, 0.8-0.2*j,
                         r'$%s_\mathrm{%s}$' % (prefix, subscripts[j]),
                         horizontalalignment='right',
                         transform=ax.transAxes, size=15, color=colors[j])

            plt.xticks(size=5)
            ax.grid(ls='--')
            ax.yaxis.tick_right()
            ax.yaxis.set_ticks_position('both')
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel('r [Bohr]', size=8)
            i += 1

        plt.tight_layout()
        if filename is None:
            filename = '{0}_{1}nl.pdf'.format(self.symbol, prefix)
        plt.savefig(filename)
        plt.clf()
        return

    def plot_density(self, filename=None):
        """ Plot the electron density and valence subshell densities.

        Note that the plotted electron density (rho_0) generally does
        not correspond to the sum of the valence subshell densities in
        the valence region. For this to be the case, the subshell densities
        would need to be multiplied by their occupation numbers, and
        the same confinement potential would need to be applied throughout.

        Parameters
        ----------
        filename : str or None, optional
            Output file name. If None (the default), '<element>_rho.pdf'
            will be the filename.
        """
        assert plt is not None, 'Matplotlib could not be imported!'
        assert self.solved, NOT_SOLVED_MESSAGE

        rmax = 3 * covalent_radii[self.Z] / Bohr
        ri = np.where(self.rgrid > rmax)[0][0]

        plt.figure(figsize=(6.4, 4.8), dpi=400)
        colors = ['red', 'green', 'blue', 'purple']  # for s/p/d/f

        for n, l, nl in self.list_states():
            if nl not in self.valence:
                continue

            dens = self.Rnlg[nl] ** 2 / (4 * np.pi)
            occupied = self.configuration[nl] > 0
            suffix = '' if occupied else '*'
            ls = '-' if occupied else '--'
            label = r'$|R_\mathrm{%s%s}(r) / \sqrt{4\pi}|^2$' % (nl, suffix)

            plt.semilogy(self.rgrid[:ri], dens[:ri], ls=ls, color=colors[l],
                        label=label)

        dens = self.dens[:ri]
        plt.semilogy(self.rgrid[:ri], dens, 'k-', label=r'$\rho_0(r)$')

        ymax = np.exp(np.ceil(np.log(np.max(dens))))
        plt.ylim([1e-7, ymax])
        plt.xlim([-0.05 * rmax, rmax])
        plt.xlabel('r (Bohr)')
        plt.grid(ls='--')
        plt.legend(loc='upper right', ncol=2)
        plt.title('Electron and subshell densities for %s' % self.symbol)

        if filename is None:
            filename = '%s_rho.pdf' % self.symbol
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()

    def plot_rho(self, *args, **kwargs):
        self.plot_density(*args, **kwargs)

    def write_unl(self, filename, only_valence=True, step=20):
        """
        Append functions unl=Rnl*r into file.

        Parameters
        ----------
        filename : str
            Output file name (e.g. '<element>.elm').
        only_valence : bool, optional
            Whether to write only the valence subshells.
        step : int, optional
            Stride applied to self.rgrid to define the output grid.
        """
        assert self.solved, NOT_SOLVED_MESSAGE
        if only_valence:
            valence = self.valence
        else:
            valence = [nl for n, l, nl in self.list_states()]

        with open(filename, 'a') as f:
            for nl in valence:
                f.write('\n\nu_%s=' % nl)
                for r, u in zip(self.rgrid[::step], self.unlg[nl][::step]):
                    f.write(r, u)
            f.write('\n\n')

    def fit_sto(self, nl, num_exp, num_pow, regularization=1e-6,
                filename=None):
        """
        Fit Slater-type radial functions to the one on the grid.
        See self.write_hsd() for more information.

        Parameters
        ----------
        nl : str
            The (valence) subshell of interest (e.g. '2p').
        num_exp : int
            The number of exponents to use
        num_pow : int
            The number of r-powers for each exponent.
        regularization : float, optional
            Penalty to be used in the L2-regularization
        filename : str or None, optional
            Filename for a figure with the grid-based and STO-fitted
            radial function (to verify that the fit is decent).
            If None (the default), no plotting is performed.

        Returns
        -------
        exponents : np.ndarray
            Fitted Slater function exponents.
        coeff : np.ndarray
            Fitted coefficients for each Slater function.
        values : np.ndarray
            Slater function values on the grid.
        residual : float
            Residual of the fit.
        """
        assert self.solved, NOT_SOLVED_MESSAGE
        print('Fitting Slater-type radial functions for subshell %s' % nl,
              file=self.txt)
        r = self.rgrid
        y = self.Rnlg[nl]
        n, l = nl2tuple(nl)
        num_coeff = num_exp * num_pow
        num_r = len(r)

        def regression(param):
            A = np.zeros((num_r, num_coeff))

            for i in range(num_exp):
                rexp = np.exp(-param[i] * r)[:, None]
                A[:, i * num_pow : (i + 1) * num_pow] = rexp

            for j in range(num_pow):
                rpow = np.power(r, l + j)
                for i in range(num_exp):
                    A[:, num_pow * i + j] *= rpow

            A_loss = np.identity(num_coeff) * regularization
            AA = np.vstack((A, A_loss))

            y_loss = np.zeros(num_coeff)
            yy = np.hstack((y, y_loss))

            coeff, residual, rank, s = np.linalg.lstsq(AA, yy, rcond=None)
            values = np.dot(A, coeff)
            residual = residual[0] if len(residual) > 0 else np.inf
            return coeff, values, residual

        def residual(param):
            coeff, values, residual = regression(param)
            return residual

        if num_exp > 1:
            x0, x1 = self.Z, 0.5
            ratio = (x0 / x1) ** (1. / (num_exp - 1.))
            guess = [x0 / (ratio ** i) for i in range(num_exp)]
        else:
            guess = [1.]

        bounds = [(1e-1, 1e3) for i in range(num_exp)]
        result = minimize(residual, guess, method='COBYQA', tol=1e-8,
                          bounds=bounds, options={'initial_tr_radius': 0.1})
        exponents = result.x
        coeff, values, residual = regression(exponents)

        integral = np.trapezoid((r * y) ** 2, x=r)
        if abs(integral - 1) > 1e-1:
            print('Warning -- significant deviation from unity for integral'
                  ' of grid-based %s radial function: %.5f' % (nl, integral),
                  file=self.txt)

        integral = np.trapezoid((r * values) ** 2, x=r)
        if abs(integral - 1) > 1e-1:
            print('Warning -- significant deviation from unity for integral'
                  ' of STO-based %s radial function: %.5f' % (nl, integral),
                  file=self.txt)

        if filename is not None:
            rmax = 3 * covalent_radii[self.Z] / Bohr
            imax = np.where(r < rmax)[0]
            imax = imax[-1] if len(imax) > 0 else -1
            rmin = 1e-3 * self.Z
            imin = np.where(r < rmin)[0]
            imin = imin[-1] if len(imin) > 0 else 0
            plt.plot(r[imin:imax], y[imin:imax], '-', label='On the grid')
            plt.plot(r[imin:imax], values[imin:imax], '--', label='With STOs')
            plt.xlim([0., rmax])
            plt.grid(ls='--')
            plt.legend(loc='upper right')
            plt.xlabel('r (Bohr radii)')
            plt.ylabel('Psi_%s (a.u.)' % nl)
            plt.savefig(filename)
            plt.clf()

        return exponents, coeff, values, residual

    def write_hsd(self, filename=None, num_exp=None, num_pow=4, wfthr=1e-2):
        r"""
        Writes a HSD-format file with information on the valence
        subshells. This includes a projection of these subshells
        on a set of Slater-type radial functions, for post-processing
        purposes (e.g. using the Waveplot tool part of DFTB+).

        The expansion is the same as in DFTB+.
        For a subshell with angular momentum l:

          R_l(r) = \sum_{i=0}^{num_exp-1} \sum_{j=0}^{num_pow-1}
                     coeff_{i,j} * r ^ (l + j) * \exp(-exponent_i * r)

        Note that also the same normalization is used as in DFTB+.
        This means that \int_{r=0}^{\infty} r^2 * |R_l(r)|^2 dr = 1.

        Starting from a reasonable initial guess, the exponents
        are optimized to reproduce the grid-based radial functions,
        with the coefficient matrix being determined by
        (L2-regularized) linear regression at each iteration.

        Parameters
        ----------
        filename : str or None, optional
            Output file name. If None, the name defaults to
            'wcf.<element>.hsd'.
        num_exp : int or None, optional
            Number of exponents to use. The default (None) implies
            the highest principal quantum number will be used.
        num_pow : int, optional
            Number of powers for each exponent (default: 4).
        wfthr : float, optional
            Parameter determining the 'cutoff' radius which will be
            where the radial function tail goes below wfthr in absolute
            value (default: 1e-2).
        """
        assert self.solved, NOT_SOLVED_MESSAGE
        if filename is None:
            filename = 'wfc.%s.hsd' % self.symbol

        if num_exp is None:
            num_exp = max([nl2tuple(nl)[0] for nl in self.valence])

        with open(filename, 'a') as f:
            f.write('%s = {\n' % self.symbol)
            f.write('  AtomicNumber = %d\n' % self.Z)

            for nl in self.valence:
                n, l = nl2tuple(nl)
                exp, coeff, values, resid = self.fit_sto(nl, num_exp, num_pow)
                icut = len(values) - 1
                while abs(values[icut]) < wfthr:
                    icut -= 1
                rcut = np.round(self.rgrid[icut + 1], 1)

                f.write('  Orbital = {\n')
                f.write('    AngularMomentum = %d\n' % l)
                f.write('    Occupation = %.6f\n' % self.configuration[nl])
                f.write('    Cutoff = %.1f\n' % rcut)
                f.write('    Exponents = {\n    ')
                for e in exp:
                    f.write('  %.8f' % e)
                f.write('\n    }\n    Coefficients = {\n')
                for c in coeff:
                    f.write('      {: .8E}\n'.format(c))
                f.write('    }\n  }\n')
            f.write('}\n')
        return

    def calculate_excited_eigenvalue(self, nl1, nl2, diff, spin):
        """
        Returns <phi_nl1|H|phi_nl1> for an 'excited' electronic
        configuration involving a change in the (possible spin-resolved)
        occupation of a second subshell.

        Parameters
        ----------
        nl1 : str
            First subshell label.
        nl2 : str
            Second subshell label.
        diff : float
            Change in the occupation of the second subshell.
        spin : None or str
            If None (default), the 'up' and 'down' occupations of the
            second subshell are changed by the same amount (0.5 * 'diff'),
            so that the magnetization density remains zero. When
            choosing spin='up' or 'down', only the occupation of the
            selected channel is modified (by a 'diff' amount). The nl1
            eigenvalue is then evaluated for the 'up' spin.
        """
        assert spin in [None, 'up', 'down']

        template = 'Evaluating the {0}{1} eigenvalue for a {2:.3f} change '
        template += 'in the {3}{4} occupation'
        msg = template.format(nl1, '' if spin is None else ' (up)', diff, nl2,
                              '' if spin is None else ' (' + spin + ')')
        print(msg, file=self.txt)

        if spin is not None:
            if isinstance(self.xc, LibXC):
                xcname = self.xc.xcname
                xc = LibXC(xcname, spin_polarized=True)
            elif isinstance(self.xc, XC_PW92):
                xc = LibXC('LDA_X+LDA_C_PW', spin_polarized=True)

        configuration_original = self.configuration.copy()

        if self.perturbative_confinement:
            veff_original = np.copy(self.veff)
            has_pseudo = not isinstance(self.pp, PhillipsKleinmanPP)

            # Obtain dens & veff for the default electronic configuration
            dens = self.calculate_density(self.unlg, only_valence=has_pseudo)

            # Update density and number of electrons
            self.configuration[self.valence[0]] += diff
            dens_nl2 = diff * self.unlg[nl2]**2 / (4 * np.pi * self.rgrid**2)
            dens += dens_nl2

            self.veff = self.calculate_veff(dens)

            if spin is not None:
                # Recalculate the XC part of the effective potential
                dens_xc = np.copy(dens)
                if has_pseudo:
                    dens_xc = self.add_core_electron_density(dens_xc)

                exc, vxc = self.xc.evaluate(dens_xc, self.grid)
                self.veff -= vxc

                dens_up = (dens_xc - dens_nl2) / 2.
                dens_down = np.copy(dens_up)
                if spin == 'up':
                    dens_up += dens_nl2
                else:
                    dens_down += dens_nl2

                vxc = xc.evaluate_polarized(dens_up, dens_down, self.grid)
                self.veff += vxc

            e = self.get_onecenter_integrals(nl1, nl1)[0]
            self.veff = veff_original
        else:
            assert nl1 in self.valence, (nl1, self.valence)
            assert nl2 in self.valence, (nl2, self.valence)
            self.configuration[nl2] += diff
            self.run()
            e = self.get_eigenvalue(nl1)
            self.solved = False

        self.configuration = configuration_original
        return e

    def calculate_eigenvalue_derivative(self, nl1, nl2, maxstep=1, scheme=None,
                                        spin=None):
        """
        Calculates the derivative of <phi_nl1|H|phi_nl1> with respect to
        the occupation of nl2 using (second order) finite differences.

        Note: when using perturbative confinement, the run() method
        has to be called first, because the radial functions are needed
        (and are kept fixed). This is not the case with the non-
        perturbative scheme, where a self-consistent solution is sought
        for every subshell occupancy.

        Parameters
        ----------
        nl1: str
            First subshell label.
        nl2: str
            Second subshell label.
        maxstep : float, optional
            The maximal step size in the subshell occupancy.
            The default default value of 1 means not going further
            than the monovalent ions.
        scheme : None or str, optional
            The finite difference scheme, either 'central', 'forward'
            or 'backward' or None. In the last case the appropriate
            scheme will be chosen based on the subshell occupation.

        Other Parameters
        ----------------
        spin : see calculate_excited_eigenvalue().
        """
        assert scheme in [None, 'central', 'forward', 'backward']
        is_minimal = nl2 in self.valence

        if self.perturbative_confinement:
            assert self.solved, NOT_SOLVED_MESSAGE
        else:
            msg = 'Eigenvalue derivatives for non-minimal basis functions ' + \
                  'are only available for the perturbative confinement scheme.'
            assert is_minimal, msg
            msg = 'Spin constants are only available for the perturbative ' + \
                  'confinement scheme.'
            assert spin is None, msg

        if scheme is None:
            n, l = nl2tuple(nl2[:2])
            max_occup = 2 * (2*l + 1)
            occup = self.configuration[nl2] if is_minimal else 0
            if occup == 0:
                scheme = 'forward'
            elif occup == max_occup:
                scheme = 'backward'
            else:
                scheme = 'central'
        elif not is_minimal:
            assert scheme == 'forward'

        directions, weights = {
            'forward': ([0, 1, 2], [-1.5, 2, -0.5]),
            'central': ([-1, 0, 1], [-0.5, 0, 0.5]),
            'backward': ([-2, -1, 0], [0.5, -2, 1.5]),
            }[scheme]

        delta = maxstep if scheme == 'central' else 0.5 * maxstep
        configuration = self.configuration.copy()
        energies = []

        for direction in directions:
            diff = direction * delta
            e = self.calculate_excited_eigenvalue(nl1, nl2, diff, spin)
            energies.append(e)

        # Check that the original electronic configuration has not changed
        assert self.configuration == configuration

        dedf = sum([c*e for c, e in zip(weights, energies)]) / delta
        return dedf

    def get_hubbard_value(self, nl, nl2=None, maxstep=1., scheme=None):
        """
        Calculates the Hubbard value of the given subshell as the
        derivative of its eigenvalue with respect to the occupation
        of a second subshell.

        Parameters
        ----------
        nl : str
            First subshell label (e.g. '2p').
        nl2 : str, optional
            Second subshell label. If None (the default) it will be
            taken equal to the first subshell label.

        Other Parameters
        ----------------
        maxstep, scheme: see calculate_eigenvalue_derivative().
        """
        nl2 = nl if nl2 is None else nl2
        U = self.calculate_eigenvalue_derivative(nl, nl2, maxstep=maxstep,
                                                 scheme=scheme)
        return U

    def get_analytical_hubbard_value(self, nl1, nl2=None):
        """
        Returns the (on-site, one-center) Hubbard value U associated
        with the given subshell pair, calculated as the corresponding
        matrix element of the Hartree-XC kernel.

        Parameters
        ----------
        nl1 : str
            First subshell label.
        nl2 : str
            Second subshell label. If None (the default) it will be
            taken equal to the first subshell label.
        """
        assert self.perturbative_confinement
        assert self.solved, NOT_SOLVED_MESSAGE

        nl2 = nl1 if nl2 is None else nl2

        if isinstance(self.xc, LibXC):
            xc = self.xc
        elif isinstance(self.xc, XC_PW92):
            xc = LibXC('LDA_X+LDA_C_PW')

        dens = self.electron_density(self.rgrid)
        dens_nl1 = self.Rnlg[nl1]**2 / (4 * np.pi)
        dens_nl2 = self.Rnlg[nl2]**2 / (4 * np.pi)

        U = xc.evaluate_fxc(dens, self.grid, dens_nl1, dens_nl2)
        vhar2 = self.calculate_hartree_potential(dens_nl2, nel=1.)
        U += self.grid.integrate(dens_nl1 * vhar2, use_dV=True)
        return U

    def get_spin_constant(self, nl1, nl2=None, maxstep=0.5, scheme=None):
        """
        Calculates the spin constant of the given subshell based on
        derivatives of its up-eigenvalue with respect to the up/down-
        occupation of a second subshell.

        Parameters
        ----------
        nl1 : str
            First subshell label (e.g. '2p').
        nl2 : str
            Second subshell label. If None (the default) it will be
            taken equal to the first subshell label.

        Other Parameters
        ----------------
        maxstep, scheme: see calculate_eigenvalue_derivative().
        """
        nl2 = nl1 if nl2 is None else nl2

        dedf_up = self.calculate_eigenvalue_derivative(nl1, nl2, maxstep=maxstep,
                                                       scheme=scheme, spin='up')
        dedf_down = self.calculate_eigenvalue_derivative(nl1, nl2,
                                    maxstep=maxstep, scheme=scheme, spin='down')
        W = 0.5 * (dedf_up - dedf_down)
        return W

    def get_analytical_spin_constant(self, nl1, nl2=None):
        """
        Returns the (on-site, one-center) spin constant W associated
        with the given subshell pair, calculated as the corresponding
        matrix element of the spin-polarized XC kernel.

        Parameters
        ----------
        nl1 : str
            First subshell label.
        nl2 : str
            Second subshell label. If None (the default) it will be
            taken equal to the first subshell label.
        """
        assert self.perturbative_confinement
        assert self.solved, NOT_SOLVED_MESSAGE

        nl2 = nl1 if nl2 is None else nl2

        if isinstance(self.xc, LibXC):
            xcname = self.xc.xcname
            xc = LibXC(xcname, spin_polarized=True)
        elif isinstance(self.xc, XC_PW92):
            xc = LibXC('LDA_X+LDA_C_PW', spin_polarized=True)

        dens = self.electron_density(self.rgrid)
        dens_nl1 = self.Rnlg[nl1]**2 / (4 * np.pi)
        dens_nl2 = self.Rnlg[nl2]**2 / (4 * np.pi)

        dens_up = dens / 2.
        dens_down = np.copy(dens_up)
        W = xc.evaluate_fxc_polarized(dens_up, dens_down, self.grid, dens_nl1,
                                      dens_nl2)
        return W

    def generate_nonminimal_basis(self, size, zeta_method='split_valence',
                                  tail_norms=[0.16, 0.3], cation_charges=[2, 4],
                                  cation_potentials=['pseudo', 'pseudo'],
                                  l_pol=None, r_pol=None, **split_kwargs):
        """
        Adds more basis functions to the default minimal basis.

        Polarization functions correspond to so-called "quasi-Gaussians"
        (see Larsen et al., Phys. Rev. B 80, 105112 (2009)).

        Parameters
        ----------
        size : str
            Size of the non-minimal basis set to be generated.
            The currently allowed choices are:
            * 'sz' (single-zeta; this corresponds to a quick return),
            * 'szp' (single-zeta with polarization),
            * 'dz' (double-zeta),
            * 'dzsp' (double-zeta-for-s-functions with polarization),
            * 'dzp' (double-zeta with polarization),
            * 'dz2p' (double-zeta with double-polarization),
            * 'tz' (triple-zeta),
            * 'tzp' (triple-zeta with polarization),
            * 'tz2p' (triple-zeta with double-polarization),
            * 'tz3p' (triple-zeta with triple-polarization).
        zeta_method : str, optional
            Method for constructing higher-zeta basis functions:
            * 'split_valence' (see self.get_split_valence_unl()
              as well as the 'tail_norms' option),
            * 'cation' (see self.get_charge_confined_unl() as well as
              the 'cation_charges' and 'cation_potentials' options).
        tail_norms : dict or list of float, optional
            Parameters determining the radii at which higher-zeta
            functions are 'split off' in the split-valence scheme.
            The split radius is chosen such that tail norm
            equals the given target.
        cation_charges : dict or list of float, optional
            Charges for scaling the electrostatic potentials used for
            generating higher-zeta functions in the 'cation' zeta scheme.
        cation_potentials : dict or list of str, optional
            Type of electrostatic potentials to apply in the
            'cation' zeta scheme.
        l_pol : None or float, optional
            Angular momentum of the polarizing quasi-Gaussian.
            If None (the default), the first angular momentum is
            used which does not appear in the minimal basis.
            Note that higher polarization functions (for example
            double-polarization) will be generated for the same angular
            momentum but with a shorter characteristic radius r_pol.
        r_pol : None or float, optional
            Characteristic radius for the polarizing quasi-Gaussian.

        Other Parameters
        ----------------
        split_kwargs : optional
            Additional keyword arguments to get_split_valence_unl().
        """
        assert self.solved, NOT_SOLVED_MESSAGE
        assert size in ['sz', 'szp', 'dz', 'dszp', 'dzp', 'dz2p',
                        'tz', 'tzp', 'tz2p', 'tz3p'], \
               'Unknown basis size: {0}'.format(size)
        assert zeta_method in ['cation', 'split_valence'], \
               'Unknown zeta method: {0}'.format(zeta_method)

        print('Generating {0} basis for {1}'.format(size, self.symbol),
              file=self.txt)
        self.basis_size = size
        nzeta = {'s': 1, 'd': 2, 't': 3}[size[0]]

        if nzeta > 1:
            for izeta in range(1, nzeta):
                if len(self.basis_sets) == izeta:
                    self.basis_sets.append([])

                for nl in self.valence:
                    nlz = nl + '+'*izeta

                    if size == 'dszp' and not nl.endswith('s'):
                        continue

                    if zeta_method == 'split_valence':
                        if isinstance(tail_norms, dict):
                            tail_norm = tail_norms[nlz]
                        else:
                            tail_norm = tail_norms[izeta-1]

                        msg = 'Tail norm ({0}): {1:.3f}'
                        print(msg.format(nlz, tail_norm), file=self.txt)

                        self.unlg[nlz], r_cut = \
                            self.get_split_valence_unl(nl, tail_norm,
                                                       **split_kwargs)
                        msg = 'Split radius ({0}): {1:.3f}'
                        print(msg.format(nlz, r_cut), file=self.txt)

                    elif zeta_method == 'cation':
                        if isinstance(cation_charges, dict):
                            chg = cation_charges[nlz]
                        else:
                            chg = cation_charges[izeta-1]

                        if isinstance(cation_potentials, dict):
                            pot = cation_potentials[nlz]
                        else:
                            pot = cation_potentials[izeta-1]

                        msg = 'Cation charge ({0}): {1:.3f}, potential: {2}'
                        print(msg.format(nlz, chg, pot), file=self.txt)

                        self.unlg[nlz] = self.get_charge_confined_unl(
                                                        nl, chg, potential=pot)
                        r_cut = self.rcutnl[nl]

                    self.unl_fct[nlz] = None
                    self.Rnlg[nlz] = self.unlg[nlz] / self.rgrid
                    self.Rnl_fct[nlz] = None
                    self.rcutnl[nlz] = r_cut
                    self.basis_sets[izeta].append(nlz)

        if size.endswith('p'):
            assert r_pol is not None, 'r_pol value is needed for polarization'
            assert r_pol > 0
            print('Characteristic radius: {0:.3f}'.format(r_pol), file=self.txt)

            if l_pol is None:
                l_pol = self.get_polarization_angular_momentum()
                l_val = [ANGULAR_MOMENTUM[nl[1]] for nl in self.valence]
                assert l_pol not in l_val
            assert 0 <= l_pol <= 3
            print('Polarization l: {0}'.format(l_pol), file=self.txt)

            try:
                nzeta_pol = int(size[-2])
            except ValueError:
                nzeta_pol = 1
            assert nzeta_pol <= nzeta

            nl = self.get_subshell_to_polarize(l_pol)
            r_cut = self.rcutnl[nl]
            nlp = '0' + 'spdf'[l_pol]
            self.unlg[nlp] = self.get_quasi_gaussian_unl(nlp, l_pol, r_pol,
                                                         r_cut)
            self.unl_fct[nlp] = None
            self.Rnlg[nlp] = self.unlg[nlp] / self.rgrid
            self.Rnl_fct[nlp] = None
            self.rcutnl[nlp] = r_cut
            self.basis_sets[0].append(nlp)

            for izeta_pol in range(1, nzeta_pol):
                nlpz = nlp + '+'*izeta_pol
                r_polz = r_pol * 0.7**izeta_pol
                r_cutz = r_cut * 0.7**izeta_pol
                self.unlg[nlpz] = \
                   self.get_quasi_gaussian_unl(nlp, l_pol, r_polz, r_cutz)

                self.unl_fct[nlpz] = None
                self.Rnlg[nlpz] = self.unlg[nlpz] / self.rgrid
                self.Rnl_fct[nlpz] = None
                self.rcutnl[nlpz] = r_cutz
                self.basis_sets[izeta_pol].append(nlpz)

        print(file=self.txt)
        return

    def get_basis_set_index(self, nl):
        if nl.startswith('proj_'):
            assert '+' not in nl, nl
            return 0
        elif nl.startswith('0'):
            assert nl[0] == '0', nl
            return nl.count('+')
        else:
            assert nl[:2] in self.valence, nl
            return nl.count('+')

    def get_split_valence_unl(self, nl, tail_norm, degree=None, intercept=None):
        """
        Generates a new radial function based on the split-valence approach
        as used in Siesta (see e.g. Artacho et al., Phys. stat. sol. (b) 215,
        809 (1999)).

        Parameters
        ----------
        nl : str
            Subshell label (e.g. '2p') of the parent radial function.
        tail_norm : float
            Norm of the tail for determining the split radius.
        degree : int or None, optional
            Degree of the splitting polynomial. A degree of 2 selects
            the original Siesta form (a - b*r^2). A degree of 3 adds a
            cubic term (a - b*r^2 - c*r^3). The default (None) is to
            choose the second-degree form, except for s-type functions
            for which the parent radial function maximum does not lie
            at the origin (then the third-degree form is used).
        intercept : float or None, optional
            Function value that the splitting polynomial should adopt
            at the origin when a third-degree polynomial is used.
            The default (None) is to set it to 3/4 of the value
            of the parent reduced radial function at the origin.

        Returns
        -------
        u : np.ndarray
            The reduced radial function split off from the parent
            radial function.
        r_split : float
            The split radius.
        """
        # Find split radius based on the tail norm
        r_split, index = self.find_split_radius(nl, tail_norm)

        # Fit the polynomial
        l = ANGULAR_MOMENTUM[nl[1]]
        f0 = self.Rnl(r_split, nl, der=0)
        f1 = self.Rnl(r_split, nl, der=1)

        if degree is None:
            has_peak = self.rgrid[np.argmax(self.Rnlg[nl])] > 0.25
            degree = 3 if (l == 0 and has_peak) else 2
        else:
            assert degree in [2, 3], 'Degree must be 2, 3 or None'

        if degree == 2:
            b = (f1 - l*f0/r_split) / (-2. * r_split**(l+1))
            a = f0/r_split**l + b*r_split**2
            fpoly = self.rgrid**(l+1) * (a - b*self.rgrid**2)

        elif degree == 3:
            if intercept is None:
                a = 0.75 * self.Rnl(self.rgrid[0], nl, der=0)
            else:
                a = intercept

            c = 2*f0/r_split - 2*a*r_split**(l-1) - f1 + l*f0/r_split
            c /= r_split**(l+2)
            b = (f0 - a*r_split**l + c*r_split**(l+3)) / -r_split**(l+2)
            fpoly = self.rgrid**(l+1) * (a - b*self.rgrid**2 - c*self.rgrid**3)

        # Build the new radial function
        u = self.unlg[nl] - fpoly
        u[index:] = 0.
        self.smoothen_tail(u, index)

        norm2 = self.grid.integrate(u**2)
        u /= np.sqrt(norm2)

        if np.any(u < 0):
            msg = 'Warning: a split-valence radial function for nl={0} (tail' \
                  ' norm: {1:.3f}) contains negative values (min(u): {2:.6f})'
            print(msg.format(nl, tail_norm, np.min(u)), file=self.txt)

        return u, r_split

    def find_split_radius(self, nl, tail_norm):
        """
        Returns the radius and the corresponding grid index
        beyond which the radial function's tail has a norm
        equal to the given value.
        """
        u = np.copy(self.unlg[nl])
        norm2 = 1.
        index = len(self.rgrid)
        while norm2 > (1. - tail_norm**2):
            index -= 1
            u[index] = 0.
            norm2 = self.grid.integrate(u**2)
        r_split = self.rgrid[index]
        return r_split, index

    def get_charge_confined_unl(self, nl, charge, potential='point'):
        """
        Generates a new (higher-zeta) radial function by introducing
        an attractive electrostatic potential (as e.g. mentioned in
        Delley, J. Chem. Phys. 92, 1 (1990)).

        Parameters
        ----------
        nl : str
            Subshell label (e.g. '2p') of the parent radial function.
        charge : float
            The total (positive) charge associated with the added
            electrostatic potential.
        potential : str, optional
            The type of electrostatic potential to apply.
            With 'point' a point charge potential is used,
            whereas 'pseudo' selects the local part of the
            pseudopotential.

        Returns
        -------
        u : np.ndarray
            The reduced radial function.
        """
        assert nl in self.valence and nl in self.configuration

        vconf = self.wf_confinement[nl](self.rgrid)
        l = ANGULAR_MOMENTUM[nl[1]]

        if potential == 'point':
            vchg = -charge / self.rgrid
        elif potential == 'pseudo':
            nel = self.get_number_of_electrons(only_valence=True)
            vchg = self.pp.local_potential(self.rgrid)
            vchg *= charge * 1. / nel
        else:
            raise NotImplementedError('Unknown potential type: %s' % potential)

        veff = self.veff_free + vconf + vchg

        if not isinstance(self.pp, PhillipsKleinmanPP):
            veff += self.pp.semilocal_potential(self.rgrid, l)

        enl = {nl: self.enl_free[nl]}
        itmax, enl, d_enl, unlg, Rnlg = self.inner_scf(
               1, veff, enl, {}, solve=[nl], nodes=self.nodes)
        u = unlg[nl]
        return u

    def get_polarization_angular_momentum(self):
        """Returns the angular momentum to use for polarization functions."""
        l_val = [ANGULAR_MOMENTUM[nl[1]] for nl in self.valence]
        for l_pol in sorted(ANGULAR_MOMENTUM.values()):
            if l_pol not in l_val:
                break
        else:
            raise ValueError('Only polarization function functions '
                             'up to l={0} are available'.format(l_pol))
        return l_pol

    def get_subshell_to_polarize(self, l):
        """Returns the highest valence subshell with angular momentum l-1."""
        for nl in sorted(self.valence, reverse=True):
            if ANGULAR_MOMENTUM[nl[1]] == l-1:
                break
        else:
            raise RuntimeError(f'Could not find a subshell with l={l}')
        return nl

    def get_quasi_gaussian_unl(self, nl, l_pol, r_pol, r_cut):
        """Generates a quasi-Gaussian radial function."""
        alpha = 1. / r_pol**2
        alpha_rc2 = (r_cut / r_pol)**2
        a = (1 + alpha_rc2) * np.exp(-alpha_rc2)
        b = alpha * np.exp(-alpha_rc2)

        u = self.rgrid**(l_pol+1) * (np.exp(-alpha * self.rgrid**2) \
                                        - (a - b*self.rgrid**2))
        index = np.argmax(self.rgrid > r_cut)
        u[index:] = 0.
        self.smoothen_tail(u, index)

        norm2 = self.grid.integrate(u**2)
        u /= np.sqrt(norm2)
        return u

    def smoothen_tail(self, u, N):
        """
        Smoothens any derivative kink near a cutoff radius
        by replacing the tail by a polynomial of degree 6.

        Parameters
        ----------
        u : np.ndarray
            Radial function to be smoothened.
        N : int
            Index of the grid point corresponding to the
            cutoff radius.
        """
        rgrid = self.rgrid
        M = N - 4
        tail_length = 0.1 * rgrid[N]
        while (rgrid[N] - rgrid[M]) < tail_length:
            M -= 1

        spl = CubicSplineFunction(rgrid[M-4:M+3], u[M-4:M+3],
                                  bc_type=('not-a-knot', 'not-a-knot'))
        dr = rgrid[N-1] - rgrid[M-1]
        fdr = np.array([spl(rgrid[M-1], der=der) * dr**der
                        for der in range(4)])
        c3 = np.dot([120., 60., 12, 1.], fdr) / (6*dr**3)
        c4 = -np.dot([90., 50., 11, 1.], fdr) / (2*dr**4)
        c5 = np.dot([72., 42., 10., 1.], fdr) / (2*dr**5)
        c6 = -np.dot([60., 36., 9., 1.], fdr) / (6*dr**6)

        for i in range(M, N):
            dr = rgrid[N-1] - rgrid[i]
            u[i] = c3*dr**3 + c4*dr**4 + c5*dr**5 + c6*dr**6

        norm = self.grid.integrate(u**2)
        u /= np.sqrt(norm)

    def generate_auxiliary_basis(self, *args, **kwargs):
        """
        Sets up the auxiliary basis functions (handled via self.aux_basis).

        Parameters
        ----------
        See AuxiliaryBasis.build().
        """
        self.aux_basis.build(self, *args, **kwargs)
        return


SUBSHELLS = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l']


def nl2tuple(nl):
    """ Transforms e.g. '2p' into (2, 1) """
    return (int(nl[0]), SUBSHELLS.index(nl[1]))


def tuple2nl(n, l):
    """ Transforms e.g. (2, 1) into '2p' """
    return '%i%s' % (n, SUBSHELLS[l])
