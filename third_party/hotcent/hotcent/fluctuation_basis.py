#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.interpolation import build_interpolator
from hotcent.orbital_hartree import OrbitalHartreePotential
from hotcent.orbitals import ANGULAR_MOMENTUM, ORBITALS


class AuxiliaryBasis:
    """
    Class for handling sets of auxiliary basis functions.
    """
    def __init__(self):
        self.Anlg = {}
        self.Anl_fct = {}
        self.function_labels = []
        self.lmax = None
        self.nzeta = None
        self.ohp_dict = {}
        self.build_info = {}

    def get_angular_momenta(self):
        """Returns the angular momenta used in the auxiliary basis set."""
        ls = list(range(self.lmax+1))
        return ls

    def get_angular_momentum(self, iaux):
        """Returns the angular momentum for the given basis function index."""
        l = self.function_labels[iaux][1]
        return l

    def get_basis_set_label(self):
        """Returns the label describing the auxiliary basis set."""
        label = '{0}{1}'.format(self.nzeta, 'SPDF'[self.lmax])
        return label

    def get_lmax(self):
        """Returns the highest angular momentum in the auxiliary basis set."""
        lmax = self.lmax
        return lmax

    def get_nzeta(self):
        """Returns the number of radial functions in the auxiliary basis set."""
        nzeta = self.nzeta
        return nzeta

    def get_orbital_label(self, iaux):
        """Returns the orbital label for the given basis function index."""
        label = self.function_labels[iaux][2]
        return label

    def get_radial_label(self, izeta):
        """Returns the subshell label for the given zeta index."""
        label = self.subshell + '+'*izeta
        return label

    def get_subshell_label(self, iaux):
        """Returns the subshell label for the given basis function index."""
        label = self.function_labels[iaux][0]
        return label

    def get_size(self):
        """Returns the total number of auxiliary basis functions."""
        size = self.nzeta * (self.lmax + 1)**2
        return size

    def get_zeta_index(self, label):
        """Returns the zeta index of the given basis function label."""
        izeta = label.count('+')
        return izeta

    def select_radial_functions(self):
        """Returns a list of subshell labels of the auxiliary basis set."""
        selected = [self.get_radial_label(izeta) for izeta in range(self.nzeta)]
        return selected

    def build(self, el, subshell=None, nzeta=2, lmax=2,
              zeta_method='split_valence', tail_norms=[0.2, 0.4],
              cation_charges=[2, 4], cation_potentials=['pseudo', 'pseudo'],
              **split_kwargs):
        """
        Builds a set of auxiliary radial functions and associated
        Hartree potentials.

        Parameters
        ----------
        el : AtomicBase instance
            Provides the main basis set and an integration grid.
        subshell : str, optional
            Subshell label selecting the radial function of the
            main basis from which the auxiliary radial functions are
            to be derived. By default (subshell=None) the minimal
            valence radial function for the lowest angular momentum
            is used.
        nzeta : int, optional
            Number of auxiliary radial functions to generate.
        lmax : int, optional
            Maximum angular momentum of the auxiliary basis functions
            (default: 2, i.e. up to d).
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

        Other Parameters
        ----------------
        split_kwargs : optional
            Additional keyword arguments to AtomicBase.get_split_valence_unl().
        """
        if subshell is None:
            key_fct = lambda nl: ANGULAR_MOMENTUM[nl[1]]
            self.subshell = sorted(el.valence, key=key_fct)[0]
        else:
            assert subshell in el.valence
            self.subshell = subshell

        assert 0 <= lmax <= 2, 'lmax must lie between 0 and 2'
        self.lmax = lmax

        assert nzeta >= 1, 'nzeta must be at least 1'
        assert len(tail_norms) >= nzeta-1, 'additional tail norms are needed'
        self.nzeta = nzeta

        assert zeta_method in ['cation', 'split_valence'], \
               'Unknown zeta method: {0}'.format(zeta_method)
        self.build_info.update(zeta_method=zeta_method)

        for izeta in range(self.nzeta):
            nl = self.get_radial_label(izeta)

            for l in range(self.lmax+1):
                for lm in ORBITALS[l]:
                    self.function_labels.append((nl, l, lm))

            rc = None

            if izeta == 0:
                Rnl = np.copy(el.Rnlg[self.subshell])
                rc = el.rcutnl[self.subshell]
            else:
                if zeta_method == 'split_valence':
                    if isinstance(tail_norms, dict):
                        tail_norm = tail_norms[nl]
                    else:
                        tail_norm = tail_norms[izeta-1]
                    u, rc = el.get_split_valence_unl(self.subshell, tail_norm,
                                                     **split_kwargs)
                elif zeta_method == 'cation':
                    if isinstance(cation_charges, dict):
                        chg = cation_charges[nl]
                    else:
                        chg = cation_charges[izeta-1]

                    if isinstance(cation_potentials, dict):
                        pot = cation_potentials[nl]
                    else:
                        pot = cation_potentials[izeta-1]

                    u = el.get_charge_confined_unl(self.subshell, chg,
                                                   potential=pot)
                    rc = el.rcutnl[self.subshell]

                Rnl = u / el.rgrid

            for l in range(self.lmax+1):
                A = el.rgrid**l * Rnl**2
                norm = el.grid.integrate(A * el.rgrid**2, use_dV=False)
                A /= norm
                self.Anlg[(nl, l)] = A
                self.Anl_fct[(nl, l)] = build_interpolator(el.rgrid, A, rc)
                self.ohp_dict[(nl, l)] = OrbitalHartreePotential(el.rgrid, A,
                                                                 self.lmax)
        return

    def eval(self, r, iaux, der=0):
        """
        Evaluates the selected auxiliary basis function or its derivatives.

        Similar to self.__call__(), but more convenient for callers
        who only need to know the basis function index.

        Parameters
        ----------
        r : float or np.ndarray
            Distance from the basis function origin.
        iaux : int
            Basis function index.
        der : int, optional
            Order of the derivative (default: 0).

        Returns
        -------
        A : float or np.ndarray
            Function value(s) or derivative(s).
        """
        nl = self.get_subshell_label(iaux)
        l = self.get_angular_momentum(iaux)
        A = self.__call__(r, nl, l, der=der)
        return A

    def __call__(self, r, nl, l, der=0):
        """
        Evaluates the selected auxiliary basis function or its derivatives.

        Parameters
        ----------
        r : float or np.ndarray
            Distance from the basis function origin.
        nl : str
            Subshell label defining the radial function.
        l : int
            Angular momentum (also defining the radial function).
        der : int, optional
            Order of the derivative (default: 0).

        Returns
        -------
        A : float or np.ndarray
            Function value(s) or derivative(s).
        """
        A = self.Anl_fct[(nl, l)](r, der=der)
        return A

    def vhar(self, r, iaux):
        """
        Evaluates the radial part of the Hartree potential associated
        with the given auxiliary basis function.

        Parameters
        ----------
        r : float or np.ndarray
            Distance from the basis function origin.
        iaux : int
            Basis function index.

        Returns
        -------
        v : float or np.ndarray
            Hartree potential value(s).
        """
        nl = self.get_subshell_label(iaux)
        l = self.get_angular_momentum(iaux)
        v = self.ohp_dict[(nl, l)].vhar_fct[l](r)
        return v
