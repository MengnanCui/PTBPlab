#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from scipy.interpolate import CubicSpline
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.slako import tail_smoothening
from hotcent.xc import XC_PW92, LibXC


class Repulsion2cTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=600, nr=100, wflimit=1e-7,
            shift=False, smoothen_tails=True, xc='LDA'):
        """
        Calculates the 'repulsive' contributions to the total energy
        (i.e. the double-counting and ion-ion interaction terms),
        which are stored in self.erep.

        Parameters
        ----------
        shift: bool, optional
            Whether to apply rigid shifts such that the integrals
            at the table ends are zero (default: False).

        Other parameters
        ----------------
        See Offsite2cTable.run().
        """
        self.print_header()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'

        self.timer.start('run_repulsion2c')
        wf_range = self.get_range(wflimit)
        self.Rgrid = rmin + dr * np.arange(N)
        self.erep = np.zeros(N)

        for i, R in enumerate(self.Rgrid):
            grid, area = self.make_grid(R, wf_range + R, nt=ntheta, nr=nr)

            if  i == N - 1 or N // 10 == 0 or i % (N // 10) == 0:
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            self.erep[i] = self.calculate(self.ela, self.elb, R, grid, area,
                                          xc=xc)

        if shift and not np.allclose(self.erep, 0):
            for i in range(N-1, 1, -1):
                if abs(self.erep[i]) > 0:
                    self.erep[:i+1] -= self.erep[i]
                    break

        if smoothen_tails:
            # Smooth the curves near the cutoff
            self.erep = tail_smoothening(self.Rgrid, self.erep)

        self.timer.stop('run_repulsion2c')

    def write(self):
        """
        Writes all two-center repulsive energies to file
        in the 'Spline block' style of the SKF format.

        The filename template corresponds to
        '<el1>-<el2>_repulsion2c.spl'.
        """
        lines = self.get_spline_block()

        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()
            template = '%s-%s_repulsion2c.spl'
            filename = template % (sym1, sym2)
            print('Writing to %s' % filename, file=self.txt, flush=True)

            with open(filename, 'w') as f:
                f.write(lines)

    def get_spline_block(self):
        """
        Returns lines with a 'Spline block' representation of
        the repulsive energies (see the SKF format).
        """
        lines = 'Spline\n'

        dr = self.Rgrid[1] - self.Rgrid[0]
        n = len(self.Rgrid) - 1
        lines += '%d %.6f\n' % (n, self.Rgrid[-1])

        # Fit the exponential function for radii below self.Rgrid[0]
        r0 = self.Rgrid[0]
        f0 = self.erep[0]
        f1 = (self.erep[1] - self.erep[0]) / dr
        f2 = (self.erep[2] - 2.*self.erep[1] + self.erep[0]) / dr**2
        assert f1 < 0, 'Cannot fit exponential repulsion when derivative > 0'
        assert f2 > 0, 'Cannot fit exponential repulsion when curvature < 0'

        a1 = -f2 / f1
        a2 = np.log(-f1) - np.log(a1) + a1 * r0
        a3 = f0 - np.exp(-a1*r0 + a2)
        assert np.abs(-a1 * np.exp(-a1*r0 + a2) - f1) < 1e-8
        assert np.abs(a1**2 * np.exp(-a1*r0 + a2) - f2) < 1e-8
        lines += '%.6f %.6f %.6f\n' % (a1, a2, a3)

        # Set up the cubic spline function spanning self.Rgrid
        # and matching the exponential function on its left
        spl = CubicSpline(self.Rgrid, self.erep, bc_type=((1, f1), 'natural'))
        assert np.abs(spl(r0, nu=0) - f0) < 1e-8
        assert np.abs(spl(r0, nu=1) - f1) < 1e-8

        # Fit the additional coefficients for the 5th-order spline at the end
        r0 = self.Rgrid[-1]
        f0 = spl(r0, nu=0)
        f1 = spl(r0, nu=1)
        v = -f0 / dr**4
        w = -f1 / dr**3
        c4 = 5.*v - w
        c5 = (v - c4) / dr
        assert np.abs(spl.c[3][-1] + spl.c[2][-1]*dr + spl.c[1][-1]*dr**2 \
                      + spl.c[0][-1]*dr**3 + c4*dr**4 + c5*dr**5) < 1e-8
        assert np.abs(spl.c[2][-1] + 2.*spl.c[1][-1]*dr + 3.*spl.c[0][-1]*dr**2\
                      + 4.*c4*dr**3 + 5.*c5*dr**4) < 1e-8

        # Now add all the spline lines
        for i in range(n):
            items = [self.Rgrid[i], self.Rgrid[i]+dr, spl.c[3][i],
                     spl.c[2][i], spl.c[1][i], spl.c[0][i]]
            if i == n-1:
                items += [c4, c5]
            lines += ' '.join(map(lambda x: '%.9f' % x, items)) + '\n'

        return lines

    def calculate(self, e1, e2, R, grid, area, xc='LDA'):
        """
        Calculates the 'repulsive' contribution to the total energy for the
        given interatomic distance, with a two-center approximation for the
        exchange-correlation terms.

        NOTE: one-center terms are substracted (as these only shift the
        atom energies). The repulsion should hence decay to 0 for large R.

        Parameters
        ----------
        See Offsite2cTable.calculate().

        Returns
        -------
        Erep : float
            The two-center repulsive energy contribution.
        """
        self.timer.start('calculate_repulsion2c')

        # TODO: boilerplate
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)

        aux = 2 * np.pi * area * x
        rho1 = e1.electron_density(r1)
        rho2 = e2.electron_density(r2)
        rho12 = rho1 + rho2

        if e1.pp.has_nonzero_rho_core:
            rho1_val = e1.electron_density(r1, only_valence=True)
        else:
            rho1_val = rho1

        if e2.pp.has_nonzero_rho_core:
            rho2_val = e2.electron_density(r2, only_valence=True)
        else:
            rho2_val = rho2

        if e1.pp.has_nonzero_rho_core or e2.pp.has_nonzero_rho_core:
            rho12_val = rho1_val + rho2_val
        else:
            rho12_val = rho12

        self.timer.stop('prelude')

        self.timer.start('xc')
        if xc in ['LDA', 'PW92']:
            xc = XC_PW92()
            Exc = -np.sum((rho1 * xc.exc(rho1) + rho2 * xc.exc(rho2)) * aux)
            Evxc = -np.sum((rho1_val * xc.vxc(rho1) \
                            + rho2_val * xc.vxc(rho2)) * aux)
            exc12 = xc.exc(rho12)
            vxc12 = xc.vxc(rho12)
        else:
            xc = LibXC(xc)

            sigma = e1.electron_density(r1, der=1)**2
            out = xc.compute_all(rho1, sigma)
            Exc = -np.sum(rho1 * out['zk'] * aux)
            Evxc = -np.sum(rho1_val * out['vrho'] * aux)
            if xc.add_gradient_corrections:
                if e1.pp.has_nonzero_rho_core:
                    sigma = e1.electron_density(r1, der=1) \
                            * e1.electron_density(r1, der=1, only_valence=True)
                Evxc -= 2. * np.sum(out['vsigma'] * sigma * aux)

            sigma = e2.electron_density(r2, der=1)**2
            out = xc.compute_all(rho2, sigma)
            Exc -= np.sum(rho2 * out['zk'] * aux)
            Evxc -= np.sum(rho2_val * out['vrho'] * aux)
            if xc.add_gradient_corrections:
                if e2.pp.has_nonzero_rho_core:
                    sigma = e2.electron_density(r2, der=1) \
                            * e2.electron_density(r2, der=1, only_valence=True)
                Evxc -= 2. * np.sum(out['vsigma'] * sigma * aux)

            c1 = y / r1  # cosine of theta_1
            c2 = (y - R) / r2  # cosine of theta_2
            s1 = x / r1  # sine of theta_1
            s2 = x / r2  # sine of theta_2
            drho1 = e1.electron_density(r1, der=1)
            drho2 = e2.electron_density(r2, der=1)
            sigma = (drho1*s1 + drho2*s2)**2 + (drho1*c1 + drho2*c2)**2
            out = xc.compute_all(rho12, sigma)
            exc12 = out['zk']
            vxc12 = out['vrho']
            if xc.add_gradient_corrections:
                if e1.pp.has_nonzero_rho_core or e2.pp.has_nonzero_rho_core:
                    if e1.pp.has_nonzero_rho_core:
                        drho1_val = e1.electron_density(r1, der=1,
                                                        only_valence=True)
                    else:
                        drho1_val = drho1

                    if e2.pp.has_nonzero_rho_core:
                        drho2_val = e2.electron_density(r2, der=1,
                                                        only_valence=True)
                    else:
                        drho2_val = drho2

                    sigma = (drho1*s1 + drho2*s2) \
                            * (drho1_val*s1 + drho2_val*s2) \
                            + (drho1*c1 + drho2*c2) \
                            * (drho1_val*c1 + drho2_val*c2)

                Evxc += 2. * np.sum(out['vsigma'] * sigma * aux)

        Exc += np.sum(exc12 * rho12 * aux)
        Evxc += np.sum(vxc12 * rho12_val * aux)
        self.timer.stop('xc')

        vhar1 = e1.hartree_potential(r1, only_valence=True)
        vhar2 = e2.hartree_potential(r2, only_valence=True)
        Ehar = np.sum(vhar1 * rho2_val * aux)
        Ehar += np.sum(vhar2 * rho1_val * aux)

        Z1 = e1.get_number_of_electrons(only_valence=True)
        Z2 = e2.get_number_of_electrons(only_valence=True)
        Enuc = Z1 * Z2 / R

        Erep = Enuc - 0.5*Ehar + Exc - Evxc
        self.timer.stop('calculate_repulsion2c')
        return Erep
