#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.fluctuation_twocenter import (
                INTEGRALS_2CK, NUMINT_2CL, NUML_2CK, NUMSK_2CK,
                select_subshells, write_2cl, write_2ck)
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.slako import (get_integral_pair, get_twocenter_phi_integral,
                           get_twocenter_phi_integrals_derivatives,
                           tail_smoothening)
from hotcent.xc import LibXC


class Offsite2cWTable:
    """
    Convenience wrapper around the Offsite2cWMainTable
    and Offsite2cWAuxiliaryTable classes.

    Parameters
    ----------
    basis : str
        Whether to derive parameters from the main basis set in
        the monopole approximation (basis='main') or from the
        (possibly multipolar) auxiliary basis set (basis='auxiliary').
    """
    def __init__(self, *args, basis=None, **kwargs):
        if basis == 'main':
            self.calc = Offsite2cWMainTable(*args, **kwargs)
        elif basis == 'auxiliary':
            self.calc = Offsite2cWAuxiliaryTable(*args, **kwargs)
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


class Offsite2cWMainTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA', smoothen_tails=True):
        """
        Calculates offsite, distance dependent "W" values as matrix
        elements of the two-center-expanded spin-polarized XC kernel
        and the main basis set, in the monopole approximation.

        Parameters
        ----------
        See Offsite2cTable.run().
        """
        self.print_header()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_offsiteW')
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

            grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)

            if i == N - 1 or N // 10 == 0 or i % (N // 10) == 0:
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            if len(grid) > 0:
                for p, (e1, e2) in enumerate(self.pairs):
                    selected = select_subshells(e1, e2)

                    W = self.calculate(selected, e1, e2, R, grid, area, xc=xc)

                    for key in selected:
                        nl1, nl2 = key
                        bas1 = e1.get_basis_set_index(nl1)
                        bas2 = e2.get_basis_set_index(nl2)
                        index = ANGULAR_MOMENTUM[nl1[1]] * 4
                        index += ANGULAR_MOMENTUM[nl2[1]]
                        self.tables[(p, bas1, bas2)][i, index] = W[key]

        if smoothen_tails:
            # Smooth the curves near the cutoff
            for key in self.tables:
                for i in range(NUMINT_2CL):
                    self.tables[key][:, i] = \
                            tail_smoothening(self.Rgrid, self.tables[key][:, i])

        self.timer.stop('run_offsiteW')

    def calculate(self, selected, e1, e2, R, grid, area, xc='LDA'):
        """
        Calculates the selected integrals involving the magnetization kernel.

        Parameters
        ----------
        See Offsite2cTable.calculate().

        Returns
        -------
        W: dict
            Dictionary containing the integral for each selected
            subshell pair.
        """
        self.timer.start('calculate_offsiteW')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        aux = 2 * np.pi * area * x

        xc = LibXC('LDA_X+LDA_C_PW' if xc == 'LDA' else xc,
                   spin_polarized=True)

        rho_up = (e1.electron_density(r1) + e2.electron_density(r2)) / 2.
        rho_down = np.copy(rho_up)

        if xc.add_gradient_corrections:
            drho1_up = e1.electron_density(r1, der=1) / 2.
            drho2_up = e2.electron_density(r2, der=1) / 2.
            c1 = y / r1  # cosine of theta_1
            c2 = (y - R) / r2  # cosine of theta_2
            s1 = x / r1  # sine of theta_1
            s2 = x / r2  # sine of theta_2

            grad_rho_x_up = drho1_up * s1 + drho2_up * s2
            grad_rho_y_up = drho1_up * c1 + drho2_up * c2
            grad_rho_x_down = np.copy(grad_rho_x_up)
            grad_rho_y_down = np.copy(grad_rho_y_up)
            sigma_up = grad_rho_x_up**2 + grad_rho_y_up**2
            sigma_updown = grad_rho_x_up * grad_rho_x_down \
                           + grad_rho_y_up * grad_rho_y_down
            sigma_down = grad_rho_x_down**2 + grad_rho_y_down**2
        else:
            sigma_up, sigma_updown, sigma_down = None, None, None
        self.timer.stop('prelude')

        self.timer.start('fxc')
        out = xc.compute_vxc_polarized(rho_up, rho_down, sigma_up=sigma_up,
                                       sigma_updown=sigma_updown,
                                       sigma_down=sigma_down, fxc=True)
        self.timer.stop('fxc')

        W = {}
        for key in selected:
            nl1, nl2 = key
            dens_nl1 = e1.Rnl(r1, nl1)**2 / (4 * np.pi)
            dens_nl2 = e2.Rnl(r2, nl2)**2 / (4 * np.pi)

            integrand = (out['v2rho2_up'] - out['v2rho2_updown']) \
                        * dens_nl1 * dens_nl2

            if xc.add_gradient_corrections:
                dnl1 = e1.Rnl(r1, nl1) / (2 * np.pi) * e1.Rnl(r1, nl1, der=1)
                dnl2 = e2.Rnl(r2, nl2) / (2 * np.pi) * e2.Rnl(r2, nl2, der=1)
                grad_nl1_grad_nl2 = (dnl1*s1 * dnl2*s2) + (dnl1*c1 * dnl2*c2)

                grad_nl1_grad_rho_up = (dnl1*s1 * grad_rho_x_up) \
                                       + (dnl1*c1 * grad_rho_y_up)
                grad_nl2_grad_rho_up = (dnl2*s2 * grad_rho_x_up) \
                                       + (dnl2*c2 * grad_rho_y_up)
                grad_nl1_grad_rho_down = (dnl1*s1 * grad_rho_x_down) \
                                         + (dnl1*c1 * grad_rho_y_down)
                grad_nl2_grad_rho_down = (dnl2*s2 * grad_rho_x_down) \
                                         + (dnl2*c2 * grad_rho_y_down)

                # up-up contribution
                integrand += 2. * out['v2rhosigma_up_up'] \
                             * (grad_nl1_grad_rho_up * dens_nl2 \
                                + dens_nl1 * grad_nl2_grad_rho_up)

                integrand += 4. * out['v2sigma2_up_up'] \
                             * grad_nl1_grad_rho_up * grad_nl2_grad_rho_up
                integrand += 2. * out['v2sigma2_up_updown'] \
                             * grad_nl1_grad_rho_down * grad_nl2_grad_rho_up
                integrand += 2. * out['v2sigma2_up_updown'] \
                             * grad_nl1_grad_rho_up * grad_nl2_grad_rho_down
                integrand += 1. * out['v2sigma2_updown_updown'] \
                             * grad_nl1_grad_rho_down * grad_nl2_grad_rho_down
                integrand += 2. * out['vsigma_up'] * grad_nl1_grad_nl2

                # up-down contrbution
                integrand -= 2. * out['v2rhosigma_up_down'] \
                             * (grad_nl1_grad_rho_down * dens_nl2 \
                                + dens_nl1 * grad_nl2_grad_rho_down)

                integrand -= 4. * out['v2sigma2_up_down'] \
                             * grad_nl1_grad_rho_down * grad_nl2_grad_rho_up
                integrand -= 2. * out['v2sigma2_up_updown'] \
                             * grad_nl1_grad_rho_down * grad_nl2_grad_rho_up
                integrand -= 2. * out['v2sigma2_updown_down'] \
                             * grad_nl1_grad_rho_down * grad_nl2_grad_rho_down
                integrand -= 1. * out['v2sigma2_updown_updown'] \
                             * grad_nl1_grad_rho_up * grad_nl2_grad_rho_down
                integrand -= 1. * out['vsigma_updown'] * grad_nl1_grad_nl2

            W[key] = np.sum(integrand * aux) / 2.

        self.timer.stop('calculate_offsiteW')
        return W

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el2>_offsiteW.2cl'.
        """
        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()

            for bas1, valence1 in enumerate(e1.basis_sets):
                angmom1 = [ANGULAR_MOMENTUM[nl[1]] for nl in valence1]

                for bas2, valence2 in enumerate(e2.basis_sets):
                    angmom2 = [ANGULAR_MOMENTUM[nl[1]] for nl in valence2]

                    template = '%s-%s_offsiteW.2cl'
                    filename = template % (sym1 + '+'*bas1, sym2  + '+'*bas2)
                    print('Writing to %s' % filename, file=self.txt, flush=True)

                    table = self.tables[(p, bas1, bas2)]
                    with open(filename, 'w') as f:
                        write_2cl(f, self.Rgrid, table, angmom1, angmom2)
        return


class Offsite2cWAuxiliaryTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)
        assert self.ela.aux_basis.get_lmax() < NUML_2CK
        assert self.elb.aux_basis.get_lmax() < NUML_2CK

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA', smoothen_tails=True):
        """
        Calculates off-site and distance-dependent "W" values
        as matrix elements of the two-center-expanded spin-polarized
        XC kernel and the auxiliary basis set.

        Parameters
        ----------
        See Offsite2cTable.run() and Onsite2cTable.run().
        """
        self.print_header()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_offsiteW')
        wf_range = self.get_range(wflimit)
        self.Rgrid = rmin + dr * np.arange(N)

        selected = {}
        for el in [self.ela, self.elb]:
            sym = el.get_symbol()
            selected[sym] = el.aux_basis.select_radial_functions()
        print('Selected subshells:', selected, file=self.txt)

        self.tables = {}
        for p, (e1, e2) in enumerate(self.pairs):
            for bas1 in range(e1.aux_basis.get_nzeta()):
                for bas2 in range(e2.aux_basis.get_nzeta()):
                    self.tables[(p, bas1, bas2)] = np.zeros((N, NUMSK_2CK))

        for i, R in enumerate(self.Rgrid):
            grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)

            if i == N - 1 or N // 10 == 0 or i % (N // 10) == 0:
                print('R=%8.2f, %i grid points ...' % \
                      (R, len(grid)), file=self.txt, flush=True)

            if len(grid) > 0:
                for p, (e1, e2) in enumerate(self.pairs):
                    W = self.calculate(e1, e2, R, grid, area, selected, xc=xc)

                    for key in W:
                        nl1, nl2 = key
                        bas1 = e1.aux_basis.get_zeta_index(nl1)
                        bas2 = e2.aux_basis.get_zeta_index(nl2)
                        self.tables[(p, bas1, bas2)][i, :] = W[key][:]

        if smoothen_tails:
            for key in self.tables:
                for i in range(NUMSK_2CK):
                    self.tables[key][:, i] = \
                        tail_smoothening(self.Rgrid, self.tables[key][:, i])

        self.timer.stop('run_offsiteW')
        return

    def calculate(self, e1, e2, R, grid, area, selected, xc='LDA'):
        """
        Calculates the selected integrals involving the spin-polarized
        XC kernel.

        Parameters
        ----------
        selected : dict
            List of subshells to use as radial functions for every element.

        Other parameters
        ----------------
        See Offsite2cTable.calculate().

        Returns
        -------
        W : dict of np.ndarray
            Dictionary with the needed Slater-Koster integrals.
        """
        self.timer.start('calculate_offsiteW')

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

        xc = LibXC('LDA_X+LDA_C_PW' if xc == 'LDA' else xc,
                   spin_polarized=True)

        rho1 = e1.electron_density(r1) / 2
        rho2 = e2.electron_density(r2) / 2
        rho12 = rho1 + rho2
        Anl1 = {(nl1, l): e1.aux_basis(r1, nl1, l) for nl1 in selected[sym1]
                for l in e1.aux_basis.get_angular_momenta()}
        Anl2 = {(nl2, l): e2.aux_basis(r2, nl2, l) for nl2 in selected[sym2]
                for l in e2.aux_basis.get_angular_momenta()}

        if xc.add_gradient_corrections:
            drho1 = e1.electron_density(r1, der=1) / 2
            drho2 = e2.electron_density(r2, der=1) / 2
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
        out12 = xc.compute_vxc_polarized(rho12, rho12, sigma_up=sigma12,
                                         sigma_updown=sigma12,
                                         sigma_down=sigma12, fxc=True)
        self.timer.stop('fxc')

        keys = [(nl1, nl2) for nl1 in selected[sym1] for nl2 in selected[sym2]]
        W = {key: np.zeros(NUMSK_2CK) for key in keys}

        for index, integral in enumerate(INTEGRALS_2CK):
            lm1, lm2 = get_integral_pair(integral)
            gphi = get_twocenter_phi_integral(lm1, lm2, c1, c2, s1, s2)

            if xc.add_gradient_corrections:
                dgphi = get_twocenter_phi_integrals_derivatives(lm1, lm2, c1,
                                                                c2, s1, s2)

            l1 = ANGULAR_MOMENTUM[lm1[0]]
            l2 = ANGULAR_MOMENTUM[lm2[0]]
            if l1 > e1.aux_basis.get_lmax() or l2 > e2.aux_basis.get_lmax():
                continue

            for key in keys:
                nl1, nl2 = key
                integrand = Anl1[(nl1, l1)] * Anl2[(nl2, l2)] * gphi \
                            * (out12['v2rho2_up'] - out12['v2rho2_updown'])

                if xc.add_gradient_corrections:
                    products = dAnl1dr1[(nl1, l1)] * grad_r1_grad_rho12 \
                               * Anl2[(nl2, l2)] * gphi
                    products += dAnl2dr2[(nl2, l2)] * grad_r2_grad_rho12 \
                                * Anl1[(nl1, l1)] * gphi
                    products += Anl1[(nl1, l1)] * Anl2[(nl2, l2)] \
                                * grad_theta1_grad_rho12 * dgphi[2]
                    products += Anl1[(nl1, l1)] * Anl2[(nl2, l2)] \
                                * grad_theta2_grad_rho12 * dgphi[3]
                    integrand += 2 * products * (out12['v2rhosigma_up_up'] \
                                                 - out12['v2rhosigma_up_down'])

                    products = dAnl1dr1[(nl1, l1)] * dAnl2dr2[(nl2, l2)] \
                               * grad_r1_grad_rho12 * grad_r2_grad_rho12 * gphi
                    products += dAnl2dr2[(nl2, l2)] * grad_r2_grad_rho12 \
                                * Anl1[(nl1, l1)] \
                                * grad_theta1_grad_rho12 * dgphi[2]
                    products += dAnl1dr1[(nl1, l1)] * grad_r1_grad_rho12 \
                                * Anl2[(nl2, l2)] * grad_theta2_grad_rho12 \
                                * dgphi[3]
                    products += Anl1[(nl1, l1)] * Anl2[(nl2, l2)] \
                                * grad_theta1_grad_rho12 \
                                * grad_theta2_grad_rho12 * dgphi[0]
                    integrand += 4 * products * (out12['v2sigma2_up_up'] \
                                                 - out12['v2sigma2_up_down'])
                    integrand += 2 * products * (out12['v2sigma2_up_updown'] \
                                            - out12['v2sigma2_updown_down'])

                    products = dAnl1dr1[(nl1, l1)] * dAnl2dr2[(nl2, l2)] \
                               * grad_r1_grad_r2 * gphi
                    products += Anl1[(nl1, l1)] * grad_r2_grad_theta1 \
                                * dAnl2dr2[(nl2, l2)] * dgphi[2]
                    products += Anl2[(nl2, l2)] * grad_r1_grad_theta2 \
                                * dAnl1dr1[(nl1, l1)] * dgphi[3]
                    products += Anl1[(nl1, l1)] * Anl2[(nl2, l2)] \
                                * (grad_theta1_grad_theta2 * dgphi[0] \
                                   + dgphi[1] / x**2)
                    integrand += products * (2 * out12['vsigma_up'] \
                                             - out12['vsigma_updown'])

                W[key][index] = np.sum(integrand * aux) / 2

        self.timer.stop('calculate_offsiteW')
        return W

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el2>_offsiteW.2ck'.
        """
        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()

            for bas1 in range(e1.aux_basis.get_nzeta()):
                for bas2 in range(e2.aux_basis.get_nzeta()):
                    template = '%s-%s_offsiteW.2ck'
                    filename = template % (sym1 + '+'*bas1, sym2  + '+'*bas2)
                    print('Writing to %s' % filename, file=self.txt, flush=True)

                    with open(filename, 'w') as f:
                        write_2ck(f, self.Rgrid, self.tables[(p, bas1, bas2)])
        return
