#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from scipy.integrate import quad_vec
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.slako import print_integral_overview
from hotcent.spherical_harmonics import (sph_nophi, sph_nophi_der,
                                         sph_phi, sph_phi_der)
from hotcent.threecenter import select_integrals, write_3cf
from hotcent.xc import LibXC, VXC_PW92_Spline


class Onsite3cTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='monopolar',
                                     **kwargs)

    def run(self, e3, Rgrid, Sgrid, Tgrid, ntheta=150, nr=50, nphi='adaptive',
            wflimit=1e-7, xc='LDA', write=True):
        """
        Calculates on-site three-center Hamiltonian integrals.

        Parameters
        ----------
        e3 : AtomicBase-like object
            Object with atomic properties for the third atom.
        Rgrid, Sgrid, Tgrid : list or array
            Lists with distances defining the three-atom geometries.
        nphi : 'adaptive' or int
            Defines the procedure used for integrating over the 'phi'
            angle. For the default nphi='adaptive', the adaptive method
            in scipy.integrate.quad_vec is used. While it is reliable,
            the number of needed function calls can be high. Setting
            nphi to an integer value selects a simple trapezoidal method
            which is well suited for periodic integrands (see Krylov
            (2006), "Approximate Calculation of Integrals", pp 73-74),
            using nphi equally spaced phi angles in the [0, pi] interval.
            Comparatively low nphi values (e.g. 13) are often sufficient.
        write : bool, optional
            Whether to write the integrals to file (default: True). The
            filename template is '<el1a>-<el1b>_onsite3c_<el2>-<el3>.3cf'.

        Other Parameters
        ----------------
        See Offsite2cTable.run().

        Returns
        -------
        tables : dict of dict of np.ndarray
            Dictionary of the onsite threecenter integrals for each
            basis set pair (of the first element) and integral type.
        """
        self.print_header(suffix='-'+e3.get_symbol())
        self.timer.start('run_onsite3c')

        assert nphi == 'adaptive' or (isinstance(nphi, int) and nphi > 0), nphi

        wf_range = self.get_range(wflimit)
        grid, area = self.make_grid(wf_range, nt=ntheta, nr=nr)
        numST = len(Sgrid) * len(Tgrid)
        tables = {}

        selected = select_integrals(self.ela, self.ela)
        print_integral_overview(self.ela, self.ela, selected, file=self.txt)

        for (integral, nl1a, nl1b) in selected:
            bas1a = self.ela.get_basis_set_index(nl1a)
            bas1b = self.ela.get_basis_set_index(nl1b)
            key = (bas1a, bas1b)
            if key not in tables:
                tables[key] = {}
            tables[key][integral] = []

        for i, R in enumerate(Rgrid):
            print('Starting for R=%.3f' % R, file=self.txt, flush=True)

            if R < 2 * wf_range:
                d = self.calculate(selected, self.ela, self.elb, e3, R, grid,
                                   area, Sgrid=Sgrid, Tgrid=Tgrid,
                                   nphi=nphi, xc=xc)
            else:
                d = {key: np.zeros(1 + numST) for key in selected}

            for key in selected:
                integral, nl1a, nl1b = key
                bas1a = self.ela.get_basis_set_index(nl1a)
                bas1b = self.ela.get_basis_set_index(nl1b)
                tables[(bas1a, bas1b)][integral].append(d[key])

        if write:
            syma, symb = self.ela.get_symbol(), self.elb.get_symbol()
            symc = e3.get_symbol()
            template =  '%s-%s_onsite3c_%s-%s.3cf'

            for bas1a in range(len(self.ela.basis_sets)):
                for bas1b in range(len(self.ela.basis_sets)):
                    items = (syma + '+'*bas1a, syma + '+'*bas1b, symb, symc)
                    filename = template % items
                    print('Writing to %s' % filename, file=self.txt,
                          flush=True)
                    write_3cf(filename, Rgrid, Sgrid, Tgrid,
                              tables[(bas1a, bas1b)])

        self.timer.stop('run_onsite3c')
        return tables

    def calculate(self, selected, e1, e2, e3, R, grid, area, Sgrid, Tgrid,
                  nphi='adaptive', xc='LDA'):
        self.timer.start('calculate_onsite3c')

        # TODO: boilerplate
        # common for all integrals (not wf-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y -R)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y - R) / r2  # cosine of theta_2
        s1 = x / r1  # sine of theta_1
        s2 = x / r2  # sine of theta_2
        aux = area * x

        rho1 = e1.electron_density(r1)
        rho12 = rho1 + e2.electron_density(r2)

        self.timer.start('vxc')
        if xc in ['LDA', 'PW92']:
            xc = VXC_PW92_Spline()
            vxc1 = xc(rho1)
            vxc12 = xc(rho12)
        else:
            xc = LibXC(xc)

            drho1 = e1.electron_density(r1, der=1)
            grad_rho1_x = drho1 * s1
            grad_rho1_y = drho1 * c1
            sigma1 = drho1**2
            out = xc.compute_vxc(rho1, sigma1)
            vxc1 = out['vrho']
            if xc.add_gradient_corrections:
                vsigma1 = out['vsigma']

            drho2 = e2.electron_density(r2, der=1)
            grad_rho12_x = grad_rho1_x + drho2 * s2
            grad_rho12_y = grad_rho1_y + drho2 * c2
            sigma12 = grad_rho12_x**2 + grad_rho12_y**2
            out = xc.compute_vxc(rho12, sigma12)
            vxc12 = out['vrho']
            if xc.add_gradient_corrections:
                vsigma12 = out['vsigma']
        self.timer.stop('vxc')
        self.timer.stop('prelude')


        def integrands(phi):
            rA = np.sqrt((x - x0*np.cos(phi))**2 + (y - y0)**2 \
                         + (x0*np.sin(phi))**2)

            V = vxc1 - vxc12

            rho3 = e3.electron_density(rA)
            rho13 = rho1 + rho3
            rho123 = rho12 + rho3

            if isinstance(xc, VXC_PW92_Spline):
                V += xc(rho123) - xc(rho13)
            else:
                drdx = (x - x0*np.cos(phi)) / rA
                drdy = (y - y0) / rA
                drdphi = ((x - x0*np.cos(phi))*x0*np.sin(phi) \
                          + x0*np.sin(phi)*x0*np.cos(phi)) / rA

                drho3 = e3.electron_density(rA, der=1)
                grad_rho3_phi = drho3 * drdphi / x

                grad_rho13_x = grad_rho1_x + drho3 * drdx
                grad_rho13_y = grad_rho1_y + drho3 * drdy
                sigma13 = grad_rho13_x**2 + grad_rho13_y**2 \
                           + grad_rho3_phi**2
                out = xc.compute_vxc(rho13, sigma13)
                V -= out['vrho']
                if xc.add_gradient_corrections:
                    vsigma13 = out['vsigma']

                grad_rho123_x = grad_rho12_x + drho3 * drdx
                grad_rho123_y = grad_rho12_y + drho3 * drdy
                sigma123 = grad_rho123_x**2 + grad_rho123_y**2 \
                           + grad_rho3_phi**2
                out = xc.compute_vxc(rho123, sigma123)
                V += out['vrho']
                if xc.add_gradient_corrections:
                    vsigma123 = out['vsigma']

            V *= aux

            vals = np.zeros(len(selected))
            for i, key in enumerate(selected):
                integral, nl1, nl2 = key
                lm1, lm2 = integral.split('_')
                Phi1 = sph_phi(lm1, phi)
                Phi2 = sph_phi(lm2, phi)
                vals[i] = np.dot(Rnl12[key], V) * Phi1 * Phi2

                if xc.add_gradient_corrections:
                    dPhi1 = sph_phi_der(lm1, phi)
                    dPhi2 = sph_phi_der(lm2, phi)
                    grad_phi_x = grad_phi_x_nophi[key] * Phi1 * Phi2
                    grad_phi_y = grad_phi_y_nophi[key] * Phi1 * Phi2
                    grad_phi_phi = Rnl12[key] \
                                   * (dPhi1 * Phi2 + Phi1 * dPhi2) / x
                    grad_rho123_grad_phi = grad_rho123_x * grad_phi_x \
                                           + grad_rho123_y * grad_phi_y \
                                           + grad_rho3_phi * grad_phi_phi
                    grad_rho13_grad_phi = grad_rho13_x * grad_phi_x \
                                           + grad_rho13_y * grad_phi_y \
                                           + grad_rho3_phi * grad_phi_phi
                    grad_rho12_grad_phi = grad_rho12_x * grad_phi_x \
                                           + grad_rho12_y * grad_phi_y
                    grad_rho1_grad_phi = grad_rho1_x * grad_phi_x \
                                           + grad_rho1_y * grad_phi_y
                    vals[i] += 2. * np.dot(vsigma123 * grad_rho123_grad_phi \
                                           - vsigma13 * grad_rho13_grad_phi \
                                           - vsigma12 * grad_rho12_grad_phi \
                                           + vsigma1 * grad_rho1_grad_phi, aux)
            return vals

        # Pre-calculate the phi-indedendent wave function parts/products
        Rnl12 = {}
        if xc.add_gradient_corrections:
            grad_phi_x_nophi, grad_phi_y_nophi = {}, {}
            dr1dx = x/r1
            dc1dx = -y*dr1dx / r1**2
            ds1dx = (r1 - x*dr1dx) / r1**2
            dr1dy = y/r1
            dc1dy = (r1 - y*dr1dy) / r1**2
            ds1dy = -x*dr1dy / r1**2

        for key in selected:
            integral, nl1, nl2 = key
            Rnl1 = e1.Rnl(r1, nl1)
            Rnl2 = e1.Rnl(r1, nl2)
            lm1, lm2 = integral.split('_')
            Theta1 = sph_nophi(lm1, c1, s1)
            Theta2 = sph_nophi(lm2, c1, s1)
            Rnl12[key] = Rnl1 * Rnl2 * Theta1 * Theta2

            if xc.add_gradient_corrections:
                dRnl1 = e1.Rnl(r1, nl1, der=1)
                dRnl2 = e1.Rnl(r1, nl2, der=1)
                dTheta1 = sph_nophi_der(lm1, c1, s1)
                dTheta2 = sph_nophi_der(lm2, c1, s1)
                grad_phi_x_nophi[key] = (dRnl1*s1*Rnl2 + Rnl1*dRnl2*s1) \
                                        * Theta1 * Theta2
                grad_phi_x_nophi[key] += Rnl1 * Rnl2 \
                         * ((dTheta1[0]*dc1dx + dTheta1[1]*ds1dx) * Theta2 \
                            + Theta1 * (dTheta2[0]*dc1dx + dTheta2[1]*ds1dx))
                grad_phi_y_nophi[key] = (dRnl1*c1*Rnl2 + Rnl1*dRnl2*c1) \
                                        * Theta1 * Theta2
                grad_phi_y_nophi[key] += Rnl1 * Rnl2 \
                         * ((dTheta1[0]*dc1dy + dTheta1[1]*ds1dy) * Theta2 \
                            + Theta1 * (dTheta2[0]*dc1dy + dTheta2[1]*ds1dy))


        def add_integrals(results):
            rmin = 1e-2
            if ((x0**2 + y0**2) < rmin or (x0**2 + (y0 - R)**2) < rmin):
                # Third atom too close to one of the first two atoms
                vals = np.zeros(len(selected))
            elif nphi == 'adaptive':
                # Breakpoints and precision thresholds for the integration
                break_points = np.pi * np.linspace(0., 1, num=4, endpoint=False)
                epsrel, epsabs = 1e-2, 1e-5
                vals, err = quad_vec(integrands, 0., np.pi,
                                     epsrel=epsrel, epsabs=epsabs,
                                     points=break_points)
            else:
                phis = np.linspace(0., np.pi, num=nphi, endpoint=True)
                vals = np.zeros(len(selected))
                dphi = 2. * np.pi / (2 * (nphi - 1))
                for i, phi in enumerate(phis):
                    parts = integrands(phi)
                    if i == 0 or i == nphi-1:
                        parts *= 0.5
                    vals += parts * dphi

            for i, key in enumerate(selected):
                results[key].append(2. * vals[i])


        results = {key: [] for key in selected}

        # First the values for rCM = 0
        x0 = 0.
        y0 = 0.5 * R
        add_integrals(results)

        # Now the main grid
        for r in Sgrid:
            for a in Tgrid:
                x0 = r * np.sin(a)
                y0 = 0.5 * R + r * np.cos(a)
                add_integrals(results)

        self.timer.stop('calculate_onsite3c')
        return results
