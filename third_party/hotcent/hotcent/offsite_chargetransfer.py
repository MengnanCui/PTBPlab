#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.fluctuation_twocenter import (
                INTEGRALS_2CK, NUMINT_2CL, NUML_2CK, NUML_2CM, NUMLM_2CM,
                NUMSK_2CK, select_orbitals, select_subshells,
                write_2cl, write_2ck, write_2cm)
from hotcent.gaunt import get_gaunt_coefficient
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.interpolation import CubicSplineFunction
from hotcent.orbital_hartree import OrbitalHartreePotential
from hotcent.orbitals import ANGULAR_MOMENTUM, ORBITAL_LABELS, ORBITALS
from hotcent.slako import (get_integral_pair, get_twocenter_phi_integral,
                           get_twocenter_phi_integrals_derivatives,
                           tail_smoothening)
from hotcent.solid_harmonics import sph_solid_radial
from hotcent.xc import LibXC


class Offsite2cMTable(MultiAtomIntegrator):
    """
    Class for calculations involving off-site mapping coefficients
    (see Giese and York (2011), doi:10.1063/1.3587052).

    Parameters
    ----------
    constraint_method : str, optional
        Method for selecting the multipole moments entering the electrostatic
        fitting procedure. In the 'reduced' method, all orbital momenta are
        included for l <= max(lmax_a, lmax_b) = lmax_ab. In the 'original'
        method, which is the approach described by Giese and York, these are
        supplemented by those orbital momenta belonging to l == lmax_ab+1
        for which |m| <= min(lmax_a, lmax_b).

    Other parameters
    ----------------
    See MultiAtomIntegrator.
    """
    def __init__(self, *args, constraint_method='original', **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)
        assert self.ela.aux_basis.get_lmax() < NUML_2CM
        assert self.elb.aux_basis.get_lmax() < NUML_2CM
        assert constraint_method in ['original', 'reduced'], \
               'The constraint method must be either "original" or "reduced"'
        self.constraint_method = constraint_method

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            smoothen_tails=True):
        """
        Calculates the required mapping coefficients.

        Parameters
        ----------
        See Offsite2cTable.run() and Onsite2cTable.run().
        """
        self.print_header()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_offsiteM')
        wf_range = self.get_range(wflimit)
        self.Rgrid = rmin + dr * np.arange(N)
        self.tables = {}

        for p, (e1, e2) in enumerate(self.pairs):
            Naux1 = e1.aux_basis.get_size()
            for bas1 in range(len(e1.basis_sets)):
                for bas2 in range(len(e2.basis_sets)):
                    shape = (NUMLM_2CM, NUMLM_2CM, N, Naux1)
                    self.tables[(p, bas1, bas2)] = np.zeros(shape)

        for i, R in enumerate(self.Rgrid):
            if R > 2 * wf_range:
                break

            selected = select_orbitals(self.ela, self.elb)

            self.grid_type = 'monopolar'
            grid, area = self.make_grid(wf_range, nt=ntheta, nr=nr)
            inveta = self.calculate_inveta_matrix(self.ela, self.elb, R, grid,
                                                  area)
            D = self.calculate_D_matrix(self.ela, self.elb, R, grid, area)

            self.grid_type = 'bipolar'
            grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)

            if (i == N-1 or N//10 == 0 or i % (N//10) == 0):
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            if len(grid) > 0:
                M = self.calculate(selected, self.ela, self.elb, R, grid, area,
                                   D, inveta)

                for key in selected:
                    (nl1, lm1), (nl2, lm2) = key
                    bas1 = self.ela.get_basis_set_index(nl1)
                    bas2 = self.elb.get_basis_set_index(nl2)
                    ilm = ORBITAL_LABELS.index(lm1)
                    jlm = ORBITAL_LABELS.index(lm2)

                    Naux1 = self.ela.aux_basis.get_size()
                    self.tables[(0, bas1, bas2)][ilm, jlm, i, :] = \
                                M[key][:Naux1]

                    if len(self.pairs) == 2:
                        l1 = ANGULAR_MOMENTUM[lm1[0]]
                        l2 = ANGULAR_MOMENTUM[lm2[0]]
                        signs = self.get_parity_sign_changes(l1, l2)
                        self.tables[(1, bas2, bas1)][jlm, ilm, i, :] = \
                                    signs * M[key][Naux1:]

        if smoothen_tails:
            for key in self.tables:
                p = key[0]
                Naux1 = self.pairs[p][0].aux_basis.get_size()

                for i in range(NUMLM_2CM):
                    for j in range(NUMLM_2CM):
                        for k in range(Naux1):
                            self.tables[key][i, j, :, k] = \
                                tail_smoothening(self.Rgrid,
                                                 self.tables[key][i, j, :, k])

        self.timer.stop('run_offsiteM')
        return

    def get_multipole_moments(self, lmax1, lmax2):
        """
        Returns the multipole moments to consider in the fitting procedure
        (depends on self.constraint_method).

        Parameters
        ----------
        lmax1 : int
            Maximum angular momentum in the auxiliary basis
            of the first element.
        lmax2 : int
            Maximum angular momentum in the auxiliary basis
            of the second element.

        Returns
        -------
        moments : list of (int, str) tuples
            The (angular momentum, lm_label) multipole moments to include.
        """
        if self.constraint_method in ['original', 'reduced']:
            lmax12 = max(lmax1, lmax2)
            moments = []

            for lM in range(lmax12+1):
                for lmM in ORBITALS[lM]:
                    moments.append((lM, lmM))
            assert len(moments) == (lmax12 + 1)**2

            if self.constraint_method == 'original':
                lmin12 = min(lmax1, lmax2)
                lM = lmax12 + 1

                m_dict = {
                    's': 0, 'px': 1, 'pz': 0, 'py': -1, 'dxy': -2, 'dyz': -1,
                    'dz2': 0, 'dxz': 1, 'dx2-y2': 2, 'fx(x2-3y2)': 3,
                    'fy(3x2-y2)': -3, 'fz(x2-y2)': 2, 'fxyz': -2, 'fyz2': -1,
                    'fxz2': 1, 'fz3': 0,
                }

                for lmM in ORBITALS[lM]:
                    if abs(m_dict[lmM]) <= lmin12:
                        moments.append((lM, lmM))

                assert len(moments) == (lmax12 + 1)**2 + 2*lmin12 + 1
        else:
            raise NotImplementedError(self.constraint_method)

        return moments

    def get_parity_sign_changes(self, l1, l2):
        """
        Returns the sign changes that are needed to transform
        a set of mapping coefficients for the second element
        when swapping the positions of the two elements.

        Parameters
        ----------
        l1, l2 : int
            Angular momenta of the main basis orbitals.
            Note that only the sum of l1 and l2 is influential.

        Returns
        -------
        signs : np.ndarray
            Values for every auxiliary basis function of the second element
            (1: no sign change, -1: sign change needed).
        """
        Naux2 = self.elb.aux_basis.get_size()
        signs = []

        for iaux in range(Naux2):
            l3 = self.elb.aux_basis.get_angular_momentum(iaux)
            signs.append((-1)**(l1+l2+l3))

        return np.array(signs)

    def calculate_D_matrix(self, e1, e2, R, grid, area):
        """
        Calculates the 'D' matrix with the multipole moments of the
        atomic orbital products.

        Parameters
        ----------
        See Offsite2cTable.calculate().

        Returns
        -------
        D : np.ndarray
            D matrix.
        """
        self.timer.start('calculate_D')

        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y + R)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y + R) / r2  # cosine of theta_2
        s1 = x / r1  # sine of theta_1
        s2 = x / r2  # sine of theta_2
        aux = area * x

        Naux1 = e1.aux_basis.get_size()
        Naux2 = e2.aux_basis.get_size()
        Naux12 = Naux1 + Naux2

        lmax1 = e1.aux_basis.get_lmax()
        lmax2 = e2.aux_basis.get_lmax()
        moments = self.get_multipole_moments(lmax1, lmax2)
        Nmom = len(moments)

        D = np.zeros((Naux12, Nmom))

        for imom, (l, lm) in enumerate(moments):
            Clm = sph_solid_radial(e1.rgrid, l)
            for iaux in range(Naux1):
                ilm = e1.aux_basis.get_orbital_label(iaux)

                if lm == ilm:
                    integrand = e1.aux_basis.eval(e1.rgrid, iaux) * Clm \
                                * e1.rgrid**2
                    D[iaux, imom] = e1.grid.integrate(integrand, use_dV=False)

            Clm = sph_solid_radial(r2, l)
            for iaux in range(Naux2):
                ilm = e2.aux_basis.get_orbital_label(iaux)
                Anl = e2.aux_basis.eval(r1, iaux)
                gphi = get_twocenter_phi_integral(ilm, lm, c1, c2, s1, s2)
                D[Naux1+iaux, imom] = np.sum(Clm * Anl * aux * gphi)

        self.timer.stop('calculate_D')
        return D

    def calculate_inveta_matrix(self, e1, e2, R, grid, area):
        """
        Calculates the inverse of the 'eta' matrix with the
        Hartree kernel integrals.

        Parameters
        ----------
        See Offsite2cTable.calculate().

        Returns
        -------
        inveta : np.ndarray
            Inverse of the 'eta' matrix.
        """
        self.timer.start('calculate_inveta')

        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y - R) / r2  # cosine of theta_2
        s1 = x / r1  # sine of theta_1
        s2 = x / r2  # sine of theta_2
        aux = area * x

        Naux1 = e1.aux_basis.get_size()
        Naux2 = e2.aux_basis.get_size()
        Naux12 = Naux1 + Naux2
        eta = np.zeros((Naux12, Naux12))

        for iaux in range(Naux1):
            ilm = e1.aux_basis.get_orbital_label(iaux)
            vhar = e1.aux_basis.vhar(e1.rgrid, iaux)

            for jaux in range(Naux1):
                jlm = e1.aux_basis.get_orbital_label(jaux)

                if ilm == jlm:
                    Anl = e1.aux_basis.eval(e1.rgrid, jaux)
                    integrand = vhar * Anl * e1.rgrid**2
                    eta[iaux, jaux] = e1.grid.integrate(integrand, use_dV=False)

        for iaux in range(Naux2):
            ilm = e2.aux_basis.get_orbital_label(iaux)
            vhar = e2.aux_basis.vhar(e2.rgrid, iaux)

            for jaux in range(Naux2):
                jlm = e2.aux_basis.get_orbital_label(jaux)

                if ilm == jlm:
                    Anl = e2.aux_basis.eval(e2.rgrid, jaux)
                    integrand = vhar * Anl * e2.rgrid**2
                    eta[Naux1+iaux, Naux1+jaux] = \
                                      e2.grid.integrate(integrand, use_dV=False)

        for iaux in range(Naux1):
            ilm = e1.aux_basis.get_orbital_label(iaux)
            Anl = e1.aux_basis.eval(r1, iaux)

            for jaux in range(Naux2):
                jlm = e2.aux_basis.get_orbital_label(jaux)
                vhar = e2.aux_basis.vhar(r2, jaux)
                gphi = get_twocenter_phi_integral(ilm, jlm, c1, c2, s1, s2)

                eta[iaux, Naux1+jaux] = np.sum(vhar * Anl * aux * gphi)
                eta[Naux1+jaux, iaux] = eta[iaux, Naux1 + jaux]

        inveta = np.linalg.inv(eta)

        self.timer.stop('calculate_inveta')
        return inveta

    def calculate(self, selected, e1, e2, R, grid, area, D, inveta):
        """
        Calculates the selected mapping coefficients.

        Parameters
        ----------
        D : np.ndarray
            Matrix with multipole moments of the atomic orbital products.
        inveta : np.ndarray
            Inverse of the matrix with the Hartree kernel integrals.

        Other Parameters
        ----------------
        See Offsite2cTable.calculate().

        Returns
        -------
        M: dict of np.ndarray
            Dictionary containing the mapping coefficients
            for each selected orbital pair.
        """
        self.timer.start('calculate_offsiteM')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y - R) / r2  # cosine of theta_2
        s1 = x / r1  # sine of theta_1
        s2 = x / r2  # sine of theta_2
        aux = area * x
        self.timer.stop('prelude')

        Naux1 = e1.aux_basis.get_size()
        Naux2 = e2.aux_basis.get_size()
        Naux12 = Naux1 + Naux2
        g = np.zeros(Naux12)

        lmax1 = e1.aux_basis.get_lmax()
        lmax2 = e2.aux_basis.get_lmax()
        moments = self.get_multipole_moments(lmax1, lmax2)
        lmax_mom = max([moment[0] for moment in moments])
        Nmom = len(moments)
        d = np.zeros(Nmom)

        # Precalculate the Hartree potentials on the grid
        self.timer.start('calculate_vhar')
        vhar1 = [e1.aux_basis.vhar(r1, iaux) for iaux in range(Naux1)]
        vhar2 = [e2.aux_basis.vhar(r2, iaux) for iaux in range(Naux2)]
        self.timer.stop('calculate_vhar')

        M = {}
        for key in selected:
            (nl1, lm1), (nl2, lm2) = key
            l1 = ANGULAR_MOMENTUM[lm1[0]]
            l2 = ANGULAR_MOMENTUM[lm2[0]]
            product = e1.Rnl(r1, nl1) * e2.Rnl(r2, nl2)

            # g vector
            self.timer.start('calculate_g')

            for iaux in range(Naux1):
                ilm = e1.aux_basis.get_orbital_label(iaux)

                gphi = np.zeros_like(r1)
                lmax_gaunt = l1 + lmax1
                for ll in range(lmax_gaunt+1):
                    for llm in ORBITALS[ll]:
                        gaunt = get_gaunt_coefficient(llm, lm1, ilm)
                        if abs(gaunt) > 0:
                            gphi += gaunt * get_twocenter_phi_integral(
                                                llm, lm2, c1, c2, s1, s2)
                g[iaux] = np.sum(product * vhar1[iaux] * aux * gphi)

            for iaux in range(Naux2):
                ilm = e2.aux_basis.get_orbital_label(iaux)

                gphi = np.zeros_like(r2)
                lmax_gaunt = l2 + lmax2
                for ll in range(lmax_gaunt+1):
                    for llm in ORBITALS[ll]:
                        gaunt = get_gaunt_coefficient(llm, lm2, ilm)
                        if abs(gaunt) > 0:
                            gphi += gaunt * get_twocenter_phi_integral(
                                                lm1, llm, c1, c2, s1, s2)
                g[Naux1 + iaux] = np.sum(product * vhar2[iaux] * aux * gphi)

            self.timer.stop('calculate_g')

            # d vector
            self.timer.start('calculate_d')

            for imom, (l, lm) in enumerate(moments):
                Clm = sph_solid_radial(r1, l)

                gphi = np.zeros_like(Clm)
                lmax_gaunt = l1 + lmax_mom
                for ll in range(lmax_gaunt+1):
                    for llm in ORBITALS[ll]:
                        gaunt = get_gaunt_coefficient(llm, lm1, lm)
                        if abs(gaunt) > 0:
                            gphi += gaunt * get_twocenter_phi_integral(
                                                llm, lm2, c1, c2, s1, s2)
                d[imom] = np.sum(product * Clm * aux * gphi)

            self.timer.stop('calculate_d')

            # u vector
            self.timer.start('calculate_u')
            u1 = np.linalg.inv(np.matmul(D.T, np.matmul(inveta, D)))
            u2 = np.matmul(D.T, np.matmul(inveta, g)) - d
            u = np.matmul(u1, u2)
            self.timer.stop('calculate_u')

            # M vector
            M[key] = np.matmul(inveta, (g - np.matmul(D, u)))

        self.timer.stop('calculate_offsiteM')
        return M

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to
        '<el1>-<el2>_offsiteM_<label1>.2cm'.
        """
        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()
            label1 = e1.aux_basis.get_basis_set_label()

            aux_orbitals = [e1.aux_basis.get_orbital_label(iaux)
                            for iaux in range(e1.aux_basis.get_size())]

            for bas1, valence1 in enumerate(e1.basis_sets):
                for bas2, valence2 in enumerate(e2.basis_sets):
                    template = '%s-%s_offsiteM_%s.2cm'
                    filename = template % (sym1 + '+'*bas1, sym2  + '+'*bas2,
                                           label1)
                    print('Writing to %s' % filename, file=self.txt, flush=True)

                    with open(filename, 'w') as f:
                        write_2cm(f, self.Rgrid, self.tables[(p, bas1, bas2)],
                                  aux_orbitals)
        return


class Offsite2cUTable:
    """
    Convenience wrapper around the Offsite2cUMainTable
    and Offsite2cUAuxiliaryTable classes.

    Parameters
    ----------
    basis : str
        Whether to derive parameters from the main basis set in
        the monopole approximation (basis='main') or from the
        (possibly multipolar) auxiliary basis set (basis='auxiliary').
    """
    def __init__(self, *args, basis=None, **kwargs):
        if basis == 'main':
            self.calc = Offsite2cUMainTable(*args, **kwargs)
        elif basis == 'auxiliary':
            self.calc = Offsite2cUAuxiliaryTable(*args, **kwargs)
        else:
            raise ValueError('Unknown basis: {0}'.format(basis))

    def __getattr__(self, attr):
        return getattr(self.calc, attr)

    def run(self, *args, **kwargs):
        self.calc.run(*args, **kwargs)
        return

    def write(self, *args, **kwargs):
        self.calc.write(*args, **kwargs)
        return


class Offsite2cUMainTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA', smoothen_tails=True, shift=False):
        """
        Calculates off-site, distance dependent U (or "Gamma") values
        as matrix elements of the two-center-expanded Hartree-XC kernel
        and the main basis set, in the monopole approximation.

        Parameters
        ----------
        See Offsite2cTable.run() and Onsite2cTable.run().
        """
        self.print_header()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_offsiteU')
        wf_range = self.get_range(wflimit)
        self.Rgrid = rmin + dr * np.arange(N)
        self.tables = {}

        for p, (e1, e2) in enumerate(self.pairs):
            for bas1 in range(len(e1.basis_sets)):
                for bas2 in range(len(e2.basis_sets)):
                    self.tables[(p, bas1, bas2)] = np.zeros((N, NUMINT_2CL))

        for i, R in enumerate(self.Rgrid):
            if R > 2 * wf_range:
                break

            for term in ['hartree', 'xc']:
                if term == 'hartree':
                    self.grid_type = 'monopolar'
                    grid, area = self.make_grid(wf_range, nt=ntheta, nr=nr)
                elif term == 'xc':
                    self.grid_type = 'bipolar'
                    grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)

                if i == N - 1 or N // 10 == 0 or i % (N // 10) == 0:
                    print('R=%8.2f, %s, %i grid points ...' % \
                          (R, term, len(grid)), file=self.txt, flush=True)

                if len(grid) > 0:
                    for p, (e1, e2) in enumerate(self.pairs):
                        selected = select_subshells(e1, e2)

                        if term == 'hartree':
                            U = self.calculate_hartree(selected, e1, e2, R,
                                                       grid, area)
                        else:
                            U = self.calculate_xc(selected, e1, e2, R, grid,
                                                  area, xc=xc)

                        for key in selected:
                            nl1, nl2 = key
                            bas1 = e1.get_basis_set_index(nl1)
                            bas2 = e2.get_basis_set_index(nl2)
                            index = ANGULAR_MOMENTUM[nl1[1]] * 4
                            index += ANGULAR_MOMENTUM[nl2[1]]
                            self.tables[(p, bas1, bas2)][i, index] += U[key]

        for key in self.tables:
            for i in range(NUMINT_2CL):
                if shift and not np.allclose(self.tables[key][:, i], 0):
                    for j in range(N-1, 1, -1):
                        if abs(self.tables[key][j, i]) > 0:
                            self.tables[key][:j+1, i] -= self.tables[key][j, i]
                            break

                if smoothen_tails:
                    self.tables[key][:, i] = \
                        tail_smoothening(self.Rgrid, self.tables[key][:, i])

        self.timer.stop('run_offsiteU')

    def calculate_xc(self, selected, e1, e2, R, grid, area, xc='LDA'):
        """
        Calculates the selected integrals involving the XC kernel.

        Parameters
        ----------
        See Offsite2cTable.calculate().

        Returns
        -------
        U : dict
            Dictionary containing the integral for each selected
            subshell pair.
        """
        self.timer.start('calculate_offsiteU_xc')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        aux = 2 * np.pi * area * x

        xc = LibXC('LDA_X+LDA_C_PW' if xc == 'LDA' else xc)

        rho = e1.electron_density(r1) + e2.electron_density(r2)

        if xc.add_gradient_corrections:
            drho1 = e1.electron_density(r1, der=1)
            drho2 = e2.electron_density(r2, der=1)
            c1 = y / r1  # cosine of theta_1
            c2 = (y - R) / r2  # cosine of theta_2
            s1 = x / r1  # sine of theta_1
            s2 = x / r2  # sine of theta_2

            grad_rho_x = drho1 * s1 + drho2 * s2
            grad_rho_y = drho1 * c1 + drho2 * c2
            sigma = grad_rho_x**2 + grad_rho_y**2
        else:
            sigma = None
        self.timer.stop('prelude')

        self.timer.start('fxc')
        out = xc.compute_vxc(rho, sigma=sigma, fxc=True)
        self.timer.stop('fxc')

        U = {}
        for key in selected:
            nl1, nl2 = key
            dens_nl1 = e1.Rnl(r1, nl1)**2 / (4 * np.pi)
            dens_nl2 = e2.Rnl(r2, nl2)**2 / (4 * np.pi)
            integrand = out['v2rho2'] * dens_nl1 * dens_nl2

            if xc.add_gradient_corrections:
                dnl1 = e1.Rnl(r1, nl1) / (2 * np.pi) * e1.Rnl(r1, nl1, der=1)
                dnl2 = e2.Rnl(r2, nl2) / (2 * np.pi) * e2.Rnl(r2, nl2, der=1)
                grad_nl1_grad_nl2 = (dnl1*s1 * dnl2*s2) + (dnl1*c1 * dnl2*c2)

                grad_nl1_grad_rho = (dnl1*s1 * grad_rho_x) \
                                    + (dnl1*c1 * grad_rho_y)
                grad_nl2_grad_rho = (dnl2*s2 * grad_rho_x) \
                                    + (dnl2*c2 * grad_rho_y)

                integrand += 2. * out['v2rhosigma'] \
                             * (grad_nl1_grad_rho * dens_nl2 \
                                + grad_nl2_grad_rho * dens_nl1)
                integrand += 4. * out['v2sigma2'] \
                             * grad_nl1_grad_rho * grad_nl2_grad_rho
                integrand += 2. * out['vsigma'] * grad_nl1_grad_nl2

            U[key] = np.sum(integrand * aux)

        self.timer.stop('calculate_offsiteU_xc')
        return U

    def calculate_hartree(self, selected, e1, e2, R, grid, area):
        """
        Calculates the selected integrals involving the Hartree kernel.

        Parameters
        ----------
        See Offsite2cTable.calculate().

        Returns
        -------
        U : dict
            Dictionary containing the integral for each selected
            subshell pair.
        """
        self.timer.start('calculate_offsiteU_hartree')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        aux = 2 * np.pi * area * x
        self.timer.stop('prelude')

        U = {}
        for key in selected:
            nl1, nl2 = key
            dens_nl1 = e1.Rnl(r1, nl1)**2 / (4 * np.pi)

            dens_nl2 = e2.Rnlg[nl2]**2 / (4 * np.pi)
            vhar2 = e2.calculate_hartree_potential(dens_nl2, nel=1.)
            spl = CubicSplineFunction(e2.rgrid, vhar2)

            U[key] = np.sum(dens_nl1 * spl(r2) * aux) - 1./R

        self.timer.stop('calculate_offsiteU_hartree')
        return U

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el2>_offsiteU.2cl'.
        """
        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()

            for bas1, valence1 in enumerate(e1.basis_sets):
                angmom1 = [ANGULAR_MOMENTUM[nl[1]] for nl in valence1]

                for bas2, valence2 in enumerate(e2.basis_sets):
                    angmom2 = [ANGULAR_MOMENTUM[nl[1]] for nl in valence2]

                    template = '%s-%s_offsiteU.2cl'
                    filename = template % (sym1 + '+'*bas1, sym2  + '+'*bas2)
                    print('Writing to %s' % filename, file=self.txt, flush=True)

                    table = self.tables[(p, bas1, bas2)]
                    with open(filename, 'w') as f:
                        write_2cl(f, self.Rgrid, table, angmom1, angmom2)
        return


class Offsite2cUAuxiliaryTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)
        assert self.ela.aux_basis.get_lmax() < NUML_2CK
        assert self.elb.aux_basis.get_lmax() < NUML_2CK

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA', smoothen_tails=True, shift=False, subtract_delta=True):
        """
        Calculates off-site and distance-dependent "U" values
        as matrix elements of the two-center-expanded Hartree-XC kernel
        and the auxiliary basis set, in a multipole expansion.

        Parameters
        ----------
        subtract_delta : bool, optional
            Whether to subtract the point multipole contributions from
            the kernel integrals (default: True). Setting it to False
            is only useful for debugging purposes.

        Other parameters
        ----------------
        See Offsite2cTable.run() and Onsite2cTable.run().
        """
        self.print_header()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_offsiteU')
        wf_range = self.get_range(wflimit)
        self.Rgrid = rmin + dr * np.arange(N)

        selected = {}
        for el in [self.ela, self.elb]:
            sym = el.get_symbol()
            selected[sym] = el.aux_basis.select_radial_functions()
        print('Selected subshells:', selected, file=self.txt)

        self.build_ohp(selected)
        self.build_int1c(selected)
        self.build_point_kernels(selected)

        self.tables = {}
        for p, (e1, e2) in enumerate(self.pairs):
            for bas1 in range(e1.aux_basis.get_nzeta()):
                for bas2 in range(e2.aux_basis.get_nzeta()):
                    self.tables[(p, bas1, bas2)] = np.zeros((N, NUMSK_2CK))

        for i, R in enumerate(self.Rgrid):
            for term in ['hartree', 'xc']:
                if term == 'hartree':
                    self.grid_type = 'monopolar'
                    grid, area = self.make_grid(wf_range, nt=ntheta, nr=nr)
                elif term == 'xc':
                    self.grid_type = 'bipolar'
                    grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)

                if i == N - 1 or N // 10 == 0 or i % (N // 10) == 0:
                    print('R=%8.2f, %s, %i grid points ...' % \
                          (R, term, len(grid)), file=self.txt, flush=True)

                if len(grid) > 0:
                    for p, (e1, e2) in enumerate(self.pairs):
                        if term == 'hartree':
                            U = self.calculate_hartree(e1, e2, R, grid, area,
                                                       selected, subtract_delta)
                        else:
                            U = self.calculate_xc(e1, e2, R, grid, area,
                                                  selected, xc=xc)

                        for key in U:
                            nl1, nl2 = key
                            bas1 = e1.aux_basis.get_zeta_index(nl1)
                            bas2 = e2.aux_basis.get_zeta_index(nl2)
                            self.tables[(p, bas1, bas2)][i, :] += U[key][:]

        for key in self.tables:
            for i in range(NUMSK_2CK):
                all0 = np.allclose(self.tables[key][:, i], 0)
                if shift and not all0:
                    for j in range(N-1, 1, -1):
                        if abs(self.tables[key][j, i]) > 0:
                            self.tables[key][:j+1, i] -= self.tables[key][j, i]
                            break

                if smoothen_tails:
                    self.tables[key][:, i] = \
                            tail_smoothening(self.Rgrid, self.tables[key][:, i])

        self.timer.stop('run_offsiteU')
        return

    def build_ohp(self, selected):
        """
        Populates the self.ohp_dict dictionary with the needed
        Hartree potential interpolators.
        """
        self.timer.start('build_ohp')
        print('Building the Hartree potential interpolators', flush=True,
              file=self.txt)

        self.ohp_dict = {}
        for p, (e1, e2) in enumerate(self.pairs):
            sym1 = e1.get_symbol()
            sym2 = e2.get_symbol()

            for nl1 in selected[sym1]:
                for l in e1.aux_basis.get_angular_momenta():
                    if (sym1, nl1, l) not in self.ohp_dict:
                        lmax = NUML_2CK
                        Anl = np.copy(e1.aux_basis.Anlg[(nl1, l)])
                        self.ohp_dict[(sym1, nl1, l)] = \
                            OrbitalHartreePotential(e1.rgrid, Anl, lmax)

            for nl2 in selected[sym2]:
                for l in e2.aux_basis.get_angular_momenta():
                    if (sym2, nl2, l) not in self.ohp_dict:
                        lmax = NUML_2CK
                        Anl = np.copy(e2.aux_basis.Anlg[(nl2, l)])
                        self.ohp_dict[(sym2, nl2, l)] = \
                            OrbitalHartreePotential(e2.rgrid, Anl, lmax)

        self.timer.stop('build_ohp')
        return

    def evaluate_ohp(self, sym, nl, l, r):
        """
        Evaluates one component of the Hartree potential
        for the given orbital and atom.

        Parameters
        ----------
        sym : str
            Chemical symbol for the atomic orbital.
        nl : str
            Subshell label for the atomic orbital.
        l : str
            Angular momentum label for the Hartree potential component.
        r : np.ndarray
            Distances from the atomic center.

        Returns
        -------
        v : np.ndarray
            Hartree potential at the given distances.
        """
        v = self.ohp_dict[(sym, nl, l)].vhar_fct[l](r)
        return v

    def build_int1c(self, selected):
        """
        Populates the self.int1c_dict dictionary with the needed
        one-center integrals for the point multipole contributions.
        """
        self.int1c_dict = {}
        sym1 = self.ela.get_symbol()
        sym2 = self.elb.get_symbol()

        for nl1 in selected[sym1]:
            self.int1c_dict[(sym1, nl1)] = []

            for l in self.ela.aux_basis.get_angular_momenta():
                Anl = np.copy(self.ela.aux_basis.Anlg[(nl1, l)])
                int1c = self.ela.grid.integrate(
                                Anl * self.ela.rgrid**(l+2), use_dV=False)
                self.int1c_dict[(sym1, nl1)].append(int1c)

        if sym2 != sym1:
            for nl2 in selected[sym2]:
                self.int1c_dict[(sym2, nl2)] = []

                for l in self.elb.aux_basis.get_angular_momenta():
                    Anl = np.copy(self.elb.aux_basis.Anlg[(nl2, l)])
                    int1c = self.elb.grid.integrate(
                                    Anl * self.elb.rgrid**(l+2), use_dV=False)
                    self.int1c_dict[(sym2, nl2)].append(int1c)
        return

    def build_point_kernels(self, selected):
        """
        Populates the self.point_kernels dictionary with the needed
        point multipole contributions.
        """
        self.point_kernels = {}

        for p, (e1, e2) in enumerate(self.pairs):
            sym1 = e1.get_symbol()
            sym2 = e2.get_symbol()

            for bas1 in range(e1.aux_basis.get_nzeta()):
                nl1 = selected[sym1][bas1]

                for bas2 in range(e2.aux_basis.get_nzeta()):
                    nl2 = selected[sym2][bas2]
                    self.point_kernels[(p, bas1, bas2)] = np.zeros(NUMSK_2CK)

                    for i, integral in enumerate(INTEGRALS_2CK):
                        lm1, lm2 = get_integral_pair(integral)
                        l1 = ANGULAR_MOMENTUM[lm1[0]]
                        l2 = ANGULAR_MOMENTUM[lm2[0]]
                        if l1 > e1.aux_basis.get_lmax() or \
                           l2 > e2.aux_basis.get_lmax():
                            continue

                        self.point_kernels[(p, bas1, bas2)][i] = \
                            self.evaluate_point_multipole_hartree(
                                    sym1, sym2, nl1, nl2, integral, 1)
        return

    def calculate_xc(self, e1, e2, R, grid, area, selected, xc='LDA'):
        """
        Calculates the selected integrals involving the XC kernel.

        Parameters
        ----------
        selected : dict
            List of subshells to use as radial functions for every element.

        Other parameters
        ----------------
        See Offsite2cTable.calculate().

        Returns
        -------
        U: dict of np.ndarray
            Dictionary with the needed Slater-Koster integrals.
        """
        self.timer.start('calculate_offsiteU_xc')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y - R) / r2  # cosine of theta_2
        s1 = x / r1  # sine of theta_1
        s2 = x / r2  # sine of theta_2
        aux = area * x
        sym1 = e1.get_symbol()
        sym2 = e2.get_symbol()

        xc = LibXC('LDA_X+LDA_C_PW' if xc == 'LDA' else xc)

        rho1 = e1.electron_density(r1)
        rho2 = e2.electron_density(r2)
        rho12 = rho1 + rho2
        Anl1 = {(nl1, l): e1.aux_basis(r1, nl1, l) for nl1 in selected[sym1]
                for l in e1.aux_basis.get_angular_momenta()}
        Anl2 = {(nl2, l): e2.aux_basis(r2, nl2, l) for nl2 in selected[sym2]
                for l in e2.aux_basis.get_angular_momenta()}

        if xc.add_gradient_corrections:
            drho1 = e1.electron_density(r1, der=1)
            drho2 = e2.electron_density(r2, der=1)
            c2 = (y - R) / r2  # cosine of theta_2
            s2 = x / r2  # sine of theta_2

            drho12dx = drho1 * s1 + drho2 * s2
            drho12dy = drho1 * c1 + drho2 * c2
            sigma12 = drho12dx**2 + drho12dy**2

            dr1dx = x/r1
            ds1dx = (r1 - x*dr1dx) / r1**2
            dr1dy = y/r1
            ds1dy = -x*dr1dy / r1**2
            dtheta1dx = ds1dx / c1
            dtheta1dy = ds1dy / c1

            grad_r1_grad_rho12 = dr1dx * drho12dx + dr1dy * drho12dy
            grad_theta1_grad_rho12 = dtheta1dx * drho12dx + dtheta1dy * drho12dy
            dAnl1dr1 = {(nl1, l): e1.aux_basis(r1, nl1, l, der=1)
                        for nl1 in selected[sym1]
                        for l in e1.aux_basis.get_angular_momenta()}

            dr2dx = x/r2
            ds2dx = (r2 - x*dr2dx) / r2**2
            dr2dy = (y - R)/r2
            ds2dy = -x*dr2dy / r2**2
            dtheta2dx = ds2dx / c2
            dtheta2dy = ds2dy / c2

            grad_r2_grad_rho12 = dr2dx * drho12dx + dr2dy * drho12dy
            grad_theta2_grad_rho12 = dtheta2dx * drho12dx + dtheta2dy * drho12dy
            dAnl2dr2 = {(nl2, l): e2.aux_basis(r2, nl2, l, der=1)
                        for nl2 in selected[sym2]
                        for l in e2.aux_basis.get_angular_momenta()}

            grad_r1_grad_r2 = dr1dx*dr2dx + dr1dy*dr2dy
            grad_r1_grad_theta2 = dr1dx * dtheta2dx + dr1dy * dtheta2dy
            grad_r2_grad_theta1 = dr2dx * dtheta1dx + dr2dy * dtheta1dy
            grad_theta1_grad_theta2 = dtheta1dx*dtheta2dx + dtheta1dy*dtheta2dy
        else:
            sigma12 = None
        self.timer.stop('prelude')

        self.timer.start('fxc')
        out12 = xc.compute_vxc(rho12, sigma=sigma12, fxc=True)
        self.timer.stop('fxc')

        keys = [(nl1, nl2) for nl1 in selected[sym1] for nl2 in selected[sym2]]
        U = {key: np.zeros(NUMSK_2CK) for key in keys}

        for index, integral in enumerate(INTEGRALS_2CK):
            lm1, lm2 = get_integral_pair(integral)
            gphi = get_twocenter_phi_integral(lm1, lm2, c1, c2, s1, s2)

            if xc.add_gradient_corrections:
                dgphi = get_twocenter_phi_integrals_derivatives(lm1, lm2,
                                                            c1, c2, s1, s2)

            l1 = ANGULAR_MOMENTUM[lm1[0]]
            l2 = ANGULAR_MOMENTUM[lm2[0]]
            if l1 > e1.aux_basis.get_lmax() or l2 > e2.aux_basis.get_lmax():
                continue

            for key in keys:
                nl1, nl2 = key
                integrand = Anl1[(nl1, l1)] * Anl2[(nl2, l2)] * gphi \
                            * out12['v2rho2']

                if xc.add_gradient_corrections:
                    products = dAnl1dr1[(nl1, l1)] * grad_r1_grad_rho12 \
                               * Anl2[(nl2, l2)] * gphi
                    products += dAnl2dr2[(nl2, l2)] * grad_r2_grad_rho12 \
                                * Anl1[(nl1, l1)] * gphi
                    products += Anl1[(nl1, l1)] * Anl2[(nl2, l2)] \
                                * grad_theta1_grad_rho12 * dgphi[2]
                    products += Anl1[(nl1, l1)] * Anl2[(nl2, l2)] \
                                * grad_theta2_grad_rho12 * dgphi[3]
                    integrand += 2 * products * out12['v2rhosigma']

                    products = dAnl1dr1[(nl1, l1)] * dAnl2dr2[(nl2, l2)] \
                               * grad_r1_grad_rho12 * grad_r2_grad_rho12 * gphi
                    products += dAnl2dr2[(nl2, l2)] * grad_r2_grad_rho12 \
                                * Anl1[(nl1, l1)] * grad_theta1_grad_rho12 \
                                * dgphi[2]
                    products += dAnl1dr1[(nl1, l1)] * grad_r1_grad_rho12 \
                                * Anl2[(nl2, l2)] * grad_theta2_grad_rho12 \
                                * dgphi[3]
                    products += Anl1[(nl1, l1)] * Anl2[(nl2, l2)] \
                                * grad_theta1_grad_rho12 \
                                * grad_theta2_grad_rho12 * dgphi[0]
                    integrand += 4 * products * out12['v2sigma2']

                    products = dAnl1dr1[(nl1, l1)] * dAnl2dr2[(nl2, l2)] \
                               * grad_r1_grad_r2 * gphi
                    products += Anl1[(nl1, l1)] * grad_r2_grad_theta1 \
                                * dAnl2dr2[(nl2, l2)] * dgphi[2]
                    products += Anl2[(nl2, l2)] * grad_r1_grad_theta2 \
                                * dAnl1dr1[(nl1, l1)] * dgphi[3]
                    products += Anl1[(nl1, l1)] * Anl2[(nl2, l2)] * \
                                (grad_theta1_grad_theta2 * dgphi[0] \
                                 + dgphi[1] / x**2)
                    integrand += 2 * products * out12['vsigma']

                U[key][index] = np.sum(integrand * aux)

        self.timer.stop('calculate_offsiteU_xc')
        return U

    def evaluate_point_multipole_hartree(self, sym1, sym2, nl1, nl2, integral,
                                         R):
        """
        Evaluates the point multipole contribution for the given Hartree
        kernel integral (which is also the value that it should approach
        at large distance R).

        Parameters
        ----------
        sym1, sym2 : str
            Chemical symbols of the first and second element.
        nl1, nl2 : str
            Subshell labels.
        integral : str
            Integral label.
        R : float
            Interatomic distance

        Returns
        -------
        U_delta : float
            The point multipole contribution.
        """
        if integral == 'sss':
            U_delta = 4 * np.pi
        elif integral == 'sps':
            U_delta = -4 * np.pi / np.sqrt(3)
        elif integral == 'sds':
            U_delta = 4 * np.pi / np.sqrt(5)
        elif integral == 'pss':
            U_delta = 4 * np.pi / np.sqrt(3)
        elif integral == 'pps':
            U_delta = -8 * np.pi / 3.
        elif integral == 'ppp':
            U_delta = 4 * np.pi / 3.
        elif integral == 'pds':
            U_delta = 12 * np.pi / np.sqrt(15)
        elif integral == 'pdp':
            U_delta = -4 * np.pi / np.sqrt(5)
        elif integral == 'dss':
            U_delta = 4 * np.pi / np.sqrt(5)
        elif integral == 'dps':
            U_delta = -12 * np.pi / np.sqrt(15)
        elif integral == 'dpp':
            U_delta = 4 * np.pi / np.sqrt(5)
        elif integral == 'dds':
            U_delta = 24 * np.pi / 5.
        elif integral == 'ddp':
            U_delta = -16 * np.pi / 5.
        elif integral == 'ddd':
            U_delta = 4 * np.pi / 5
        else:
            raise NotImplementedError(integral)

        lm1, lm2 = get_integral_pair(integral)
        l1 = ANGULAR_MOMENTUM[lm1[0]]
        l2 = ANGULAR_MOMENTUM[lm2[0]]
        U_delta *= self.int1c_dict[(sym1, nl1)][l1]
        U_delta *= self.int1c_dict[(sym2, nl2)][l2]
        U_delta /= R**(1 + l1 + l2)
        return U_delta

    def calculate_hartree(self, e1, e2, R, grid, area, selected,
                          subtract_delta):
        """
        Calculates the selected integrals involving the Hartree kernel.

        Parameters
        ----------
        selected : dict
            List of subshells to use as radial functions for every element.
        subtract_delta : bool
            Whether to subtract the point multipole contributions from
            the kernel integrals.

        Other parameters
        ----------------
        See Offsite2cTable.calculate().

        Returns
        -------
        U: np.ndarray
            Array with the needed Slater-Koster integrals.
        """
        self.timer.start('calculate_offsiteU_hartree')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y - R) / r2  # cosine of theta_2
        s1 = x / r1  # sine of theta_1
        s2 = x / r2  # sine of theta_2
        aux = area * x
        self.timer.stop('prelude')

        sym1 = e1.get_symbol()
        sym2 = e2.get_symbol()
        Anl1 = {(nl1, l): e1.aux_basis(r1, nl1, l) for nl1 in selected[sym1]
                for l in e1.aux_basis.get_angular_momenta()}

        keys = [(nl1, nl2) for nl1 in selected[sym1] for nl2 in selected[sym2]]
        U = {key: np.zeros(NUMSK_2CK) for key in keys}

        for index, integral in enumerate(INTEGRALS_2CK):
            lm1, lm2 = get_integral_pair(integral)
            gphi = get_twocenter_phi_integral(lm1, lm2, c1, c2, s1, s2)

            l1 = ANGULAR_MOMENTUM[lm1[0]]
            l2 = ANGULAR_MOMENTUM[lm2[0]]
            if l1 > e1.aux_basis.get_lmax() or l2 > e2.aux_basis.get_lmax():
                continue

            for key in keys:
                nl1, nl2 = key
                vhar = self.evaluate_ohp(sym2, nl2, l2, r2)
                U[key][index] = np.sum(vhar * Anl1[(nl1, l1)] * aux * gphi)

                if subtract_delta:
                    U_delta = self.evaluate_point_multipole_hartree(sym1, sym2,
                                                        nl1, nl2, integral, R)
                    U[key][index] -= U_delta

        self.timer.stop('calculate_offsiteU_hartree')
        return U

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el2>_offsiteU.2ck'.
        """
        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()

            for bas1 in range(e1.aux_basis.get_nzeta()):
                for bas2 in range(e2.aux_basis.get_nzeta()):
                    template = '%s-%s_offsiteU.2ck'
                    filename = template % (sym1 + '+'*bas1, sym2  + '+'*bas2)
                    print('Writing to %s' % filename, file=self.txt, flush=True)

                    with open(filename, 'w') as f:
                        write_2ck(f, self.Rgrid, self.tables[(p, bas1, bas2)],
                            point_kernels=self.point_kernels[(p, bas1, bas2)])
        return
