#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
import os
import sys
from hotcent.interpolation import CubicSplineFunction
from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.orbitals import (ANGULAR_MOMENTUM, calculate_slako_coeff,
                              ORBITALS, ORBITAL_LABELS)


class SeparablePP:
    """
    Abstract class representing a separable pseudopotential.

    Parameters
    ----------
    symbol : str
        The chemical symbol.
    txt : str or None or file handle, optional
        Where output should be printed. Use '-' for stdout
        (default), None for /dev/null, any other string for a
        text file, or a file handle.
    """
    def __init__(self, txt='-'):
        self.symbol = None
        self.projectors = {}
        self.energies = {}
        self.overlap_fct = {}  # dict with core-valence overlap functions
        self.overlap_onsite = {}
        self.initialized_elements = []
        self.has_nonzero_rho_core = True

        if txt is None:
            self.txt = open(os.devnull, 'w')
        elif isinstance(txt, str):
            if txt == '-':
                self.txt = sys.stdout
            else:
                self.txt = open(txt, 'a')
        else:
            self.txt = txt

    def all_zero_onsite_overlaps(self):
        """ Returns whether all on-site overlaps of the (valence)
        states with the PP projectors are zero.
        """
        raise NotImplementedError

    def get_occupations(self):
        """ Returns the (valence) electronic configuration for which
        the pseudopotential has been generated.

        Returns
        -------
        occupations : dict
            Dictionary with occupation for every (valence) subshell.
        """
        raise NotImplementedError

    def build_overlaps(self, e1, e3, rmin=1e-3, rmax=None, N=500,
                       wflimit=1e-7):
        """ Builds the projector-valence overlap integral interpolators.

        Assumes that the projectors have already been constructed
        through a previous call to build_projectors().

        Parameters
        ----------
        e1 : AtomicBase-like object
            Object with atomic properties (for the valence subshells).
        e3 : AtomicBase-like object
            Object with atomic properties (for the pseudopotential
            projectors).
        rmin : float, optional
            Minimal radius to be used in the overlap tabulation.
        rmax : None or float, optional
            Maximal radius to be used in the overlap tabulation.
            If None, a sufficiently large radius is determined from
            the wf_limit parameter.
        N : int, optional
            Number of grid points to be used in the overlap tabulation.
        wflimit : float, optional
            Wave function cutoff for setting the range of the 2D
            integration grids if the rmax parameter is None.
        """
        sym1, sym3 = e1.get_symbol(), e3.get_symbol()

        assert sym3 == self.symbol
        assert len(self.energies) > 0
        assert len(self.projectors) > 0

        off2c = Offsite2cTable(e1, e3, txt=None, timing=False)
        wf_range = off2c.get_range(wflimit)

        if rmax is None:
            rmax = 2. * off2c.wf_range

        rval = rmin * np.exp(np.linspace(0., np.log(rmax / rmin), N))

        for bas1, valence in enumerate(e1.basis_sets):
            for nl1 in valence:

                l1 = ANGULAR_MOMENTUM[nl1[1]]

                for nl3, projector in self.projectors.items():
                    l3 = ANGULAR_MOMENTUM[nl3[1]]
                    nl3 = 'proj_' + nl3
                    e3.Rnl_fct[nl3] = projector  # temporary

                    for tau in range(min(l1, l3) + 1):
                        key = (sym1, sym3, nl1, nl3, tau)
                        if key in self.overlap_fct:
                            continue

                        print('PP: calculating overlaps for {0}'.format(key),
                              file=self.txt)

                        if l1 < l3:
                            sk_integral = nl1[1] + nl3[-1] + 'spdf'[tau]
                            sk_selected = [(sk_integral, nl1, nl3)]
                        else:
                            sk_integral = nl3[-1] + nl1[1] + 'spdf'[tau]
                            sk_selected = [(sk_integral, nl3, nl1)]

                        sval = []
                        for r13 in rval:
                            grid, area = off2c.make_grid(r13, wf_range, nt=150,
                                                         nr=50)
                            if l1 < l3:
                                s = off2c.calculate(sk_selected, e1, e3, r13,
                                                grid, area, only_overlap=True)
                            else:
                                s = off2c.calculate(sk_selected, e3, e1, r13,
                                                grid, area, only_overlap=True)
                            s = s[sk_selected[0]]

                            if len(grid) == 0:
                                assert abs(s) < 1e-24

                            sval.append(s)

                        self.overlap_fct[key] = CubicSplineFunction(rval, sval)

                    del e3.Rnl_fct[nl3]

        self.initialized_elements.append(sym1)

    def get_overlap(self, sym1, sym2, nl1, nl2, tau, r):
        """ Returns the overlap integral, evaluated by interpolation. """
        key = (sym1, sym2, nl1, nl2, tau)
        s = self.overlap_fct[key](r, der=0)
        return s

    def assert_initialized(self, sym):
        """ Checks whether the PP has been initialized
        for the given element.
        """
        msg = 'Pseudopotential for {0} has not been initialized for use ' \
              'together with element {1}. Please apply the build_overlaps() ' \
              'and build_projectors() functions.'
        assert sym in self.initialized_elements, msg.format(self.symbol, sym)

    def get_nonlocal_integral(self, sym1, sym2, sym3, x0, z0, R, nl1, nl2,
                              lm1, lm2):
        """
        Returns the nonlocal pseudopotential integral involving
        orbitals on the first 2 atoms and a pseudopotential on the 3rd atom
        (sum_proj3 sum_m <phi_1|chi_proj3> e_proj3 <chi_proj3|phi_2>).

        Assumes that the projector-valence overlap interpolators have
        already been constructed through a previous call to build_overlaps().
        This is not needed if the pseudopotential is located on one of the
        first 2 atoms and all on-site overlaps with the projectors are zero
        (as is the case in the Phillips-Kleinman scheme).

        Arguments
        ---------
        sym1, sym2, sym3 : str
            Chemical symbols of the three atoms.
        x0, z0, R : float
            Define the atomic positions (all in the xz-plane).
        nl1, nl2 : str
            Subshell labels ('1s', '2p', ...).
        lm1, lm2 : str
            Orbital labels ('s', 'px', ...)

        Returns
        -------
        result : float
            The nonlocal pseudopotential integral.
        """
        assert sym3 == self.symbol

        tol = 1e-6
        x3, y3, z3 = x0, 0., z0

        v13 = np.array([x3, y3, z3])
        r13 = np.linalg.norm(v13)
        coincide13 = r13 < tol
        if coincide13:
            v13[:] = 0.
            r13 = 0.
        else:
            v13 /= r13

        v23 = np.array([x3, y3, z3-R])
        r23 = np.linalg.norm(v23)
        coincide23 = r23 < tol
        if coincide23:
            v23[:] = 0.
            r23 = 0.
        else:
            v23 /= r23

        # Check for quick return
        if (coincide13 and sym1 != sym3) or (coincide23 and sym2 != sym3):
            return 0.
        elif (coincide13 or coincide23) and self.all_zero_onsite_overlaps():
            return 0.

        self.assert_initialized(sym1)
        self.assert_initialized(sym2)

        l1 = ANGULAR_MOMENTUM[nl1[1]]
        l2 = ANGULAR_MOMENTUM[nl2[1]]

        result = 0.
        for nl3, e3 in self.energies.items():
            l3 = ANGULAR_MOMENTUM[nl3[1]]
            prj = 'proj_' + nl3

            for lm3 in ORBITALS[l3]:
                term = e3
                ilm3 = ORBITAL_LABELS.index(lm3)

                # Atom1
                if coincide13:
                    S3 = self.overlap_onsite[(nl3, nl1)] if lm1 == lm3 else 0.
                else:
                    S3 = 0.
                    x, y, z = v13
                    ilm = ORBITAL_LABELS.index(lm1)
                    minl = min(l1, l3)
                    for tau in range(minl+1):
                        skint = self.get_overlap(sym1, sym3, nl1, prj, tau, r13)
                        if ilm3 >= ilm:
                            coef = calculate_slako_coeff(x, y, z, ilm+1,
                                                         ilm3+1, tau+1)
                        else:
                            coef = calculate_slako_coeff(x, y, z, ilm3+1,
                                                         ilm+1, tau+1)
                            coef *= (-1)**(l1 + l3)
                        S3 += coef * skint
                term *= S3

                # Atom2
                if coincide23:
                    S3 = self.overlap_onsite[(nl3, nl2)] if lm2 == lm3 else 0.
                else:
                    S3 = 0.
                    x, y, z = v23
                    ilm = ORBITAL_LABELS.index(lm2)
                    minl = min(l2, l3)
                    for tau in range(minl+1):
                        skint = self.get_overlap(sym2, sym3, nl2, prj, tau, r23)
                        if ilm3 >= ilm:
                            coef = calculate_slako_coeff(x, y, z, ilm+1,
                                                         ilm3+1, tau+1)
                        else:
                            coef = calculate_slako_coeff(x, y, z, ilm3+1,
                                                         ilm+1, tau+1)
                            coef *= (-1)**(l2 + l3)
                        S3 += coef * skint
                term *= S3

                result += term

        return result
