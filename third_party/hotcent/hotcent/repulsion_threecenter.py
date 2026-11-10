#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from scipy.integrate import quad_vec
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.threecenter import write_3cf
from hotcent.xc import EXC_PW92_Spline, LibXC, VXC_PW92_Spline


class Repulsion3cTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, e3, Rgrid, Sgrid, Tgrid, ntheta=150, nr=50, nphi='adaptive',
            wflimit=1e-7, xc='LDA', write=True, filename=None):
        """
        Calculates the three-center repulsive energy contribution.

        Parameters
        ----------
        e3 : AtomicBase-like object
            Object with atomic properties for the third atom.
        Rgrid, Sgrid, Tgrid : list or array
            Lists with distances defining the three-atom geometries.
        nphi : 'adaptive' or int
            Defines the procedure used for integrating over the 'phi'
            angle. For the default nphi='adaptive', the adaptive method
            in scipy.integrate.quad is used. While it is reliable,
            the number of needed function calls can be high. Setting
            nphi to an integer value selects a simple trapezoidal method
            which is well suited for periodic integrands (see Krylov
            (2006), "Approximate Calculation of Integrals", pp 73-74),
            using nphi equally spaced phi angles in the [0, pi] interval.
            Comparatively low nphi values (e.g. 13) are often sufficient.
        write : bool, optional
            Whether to write the integrals to file (the default)
            or return them as a dictionary instead.
        filename : str, optional
            File name to use in case write=True. The default (None)
            implies that a '<el1>-<el2>_repulsion3c_<el3>.3cf'
            template is used.

        Other Parameters
        ----------------
        See Offsite2cTable.run().

        Returns
        -------
        output : dict of dict of np.ndarray, optional
            Dictionary with the values for each el1-el2 pair
            and integral type (only 's_s' in this case).
            Only returned if write=False.
        """
        self.print_header(suffix='-'+e3.get_symbol())

        assert nphi == 'adaptive' or (isinstance(nphi, int) and nphi > 0), nphi

        wf_range = self.get_range(wflimit)
        numST = len(Sgrid) * len(Tgrid)
        output = {}

        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2, sym3 = e1.get_symbol(), e2.get_symbol(), e3.get_symbol()

            output[(sym1, sym2)] = {'s_s': []}

            for i, R in enumerate(Rgrid):
                print('Starting for R=%.3f' % R, file=self.txt, flush=True)

                d = None
                if R < 2 * wf_range:
                    grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)
                    if len(grid) > 0:
                        d = self.calculate(e1, e2, e3, R, grid, area,
                                           Sgrid=Sgrid, Tgrid=Tgrid, nphi=nphi,
                                           xc=xc)

                if d is None:
                    d = np.zeros(1 + numST)

                output[(sym1, sym2)]['s_s'].append(d)

            if write:
                if filename is None:
                    fname = '%s-%s_repulsion3c_%s.3cf' % (sym1, sym2, sym3)
                else:
                    fname = filename
                print('Writing to %s' % fname, file=self.txt, flush=True)
                write_3cf(fname, Rgrid, Sgrid, Tgrid, output[(sym1, sym2)])

        self.timer.stop('run_repulsion3c')
        if not write:
            return output

    def calculate(self, e1, e2, e3, R, grid, area, Sgrid, Tgrid,
                  nphi='adaptive', xc='LDA'):
        self.timer.start('calculate_repulsion3c')

        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (R - y)**2)
        aux = area * x

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

        self.timer.start('vxc')
        if xc in ['LDA', 'PW92']:
            exc_spl = EXC_PW92_Spline()
            exc1 = np.sum(rho1 * exc_spl(rho1) * aux)
            exc2 = np.sum(rho2 * exc_spl(rho2) * aux)
            exc12 = np.sum(rho12 * exc_spl(rho12) * aux)

            vxc_spl = VXC_PW92_Spline()
            evxc1 = np.sum(rho1_val * vxc_spl(rho1) * aux)
            evxc2 = np.sum(rho2_val * vxc_spl(rho2) * aux)
            evxc12 = np.sum(rho12_val * vxc_spl(rho12) * aux)
        else:
            xc = LibXC(xc)

            sigma = e1.electron_density(r1, der=1)**2
            out = xc.compute_all(rho1, sigma)
            exc1 = np.sum(rho1 * out['zk'] * aux)
            evxc1 = np.sum(rho1_val * out['vrho'] * aux)
            if xc.add_gradient_corrections:
                if e1.pp.has_nonzero_rho_core:
                    sigma = e1.electron_density(r1, der=1) \
                            * e1.electron_density(r1, der=1, only_valence=True)
                evxc1 += 2. * np.sum(out['vsigma'] * sigma * aux)

            sigma = e2.electron_density(r2, der=1)**2
            out = xc.compute_all(rho2, sigma)
            exc2 = np.sum(rho2 * out['zk'] * aux)
            evxc2 = np.sum(rho2_val * out['vrho'] * aux)
            if xc.add_gradient_corrections:
                if e2.pp.has_nonzero_rho_core:
                    sigma = e2.electron_density(r2, der=1) \
                            * e2.electron_density(r2, der=1, only_valence=True)
                evxc2 += 2. * np.sum(out['vsigma'] * sigma * aux)

            c1 = y / r1  # cosine of theta_1
            c2 = (y - R) / r2  # cosine of theta_2
            s1 = x / r1  # sine of theta_1
            s2 = x / r2  # sine of theta_2
            drho1 = e1.electron_density(r1, der=1)
            drho2 = e2.electron_density(r2, der=1)
            grad_rho1_x, grad_rho1_y = drho1 * s1, drho1 * c1
            grad_rho2_x, grad_rho2_y = drho2 * s2, drho2 * c2
            grad_rho12_x = grad_rho1_x + grad_rho2_x
            grad_rho12_y = grad_rho1_y + grad_rho2_y
            sigma = grad_rho12_x**2 + grad_rho12_y**2
            out = xc.compute_all(rho12, sigma)
            exc12 = np.sum(rho12 * out['zk'] * aux)
            evxc12 = np.sum(rho12_val * out['vrho'] * aux)

            if xc.add_gradient_corrections:
                if e1.pp.has_nonzero_rho_core:
                    drho1_val = e1.electron_density(r1, der=1,
                                                    only_valence=True)
                    grad_rho1_val_x = drho1_val * s1
                    grad_rho1_val_y = drho1_val * c1
                else:
                    grad_rho1_val_x = grad_rho1_x
                    grad_rho1_val_y = grad_rho1_y

                if e2.pp.has_nonzero_rho_core:
                    drho2_val = e2.electron_density(r2, der=1,
                                                    only_valence=True)
                    grad_rho2_val_x = drho2_val * s2
                    grad_rho2_val_y = drho2_val * c2
                else:
                    grad_rho2_val_x = grad_rho2_x
                    grad_rho2_val_y = grad_rho2_y

                if e1.pp.has_nonzero_rho_core or e2.pp.has_nonzero_rho_core:
                    grad_rho12_val_x = grad_rho1_val_x + grad_rho2_val_x
                    grad_rho12_val_y = grad_rho1_val_y + grad_rho2_val_y
                    sigma = grad_rho12_x * grad_rho12_val_x \
                            + grad_rho12_y * grad_rho12_val_y
                else:
                    grad_rho12_val_x = grad_rho12_x
                    grad_rho12_val_y = grad_rho12_y

                evxc12 += 2. * np.sum(out['vsigma'] * sigma * aux)

        self.timer.stop('vxc')
        self.timer.stop('prelude')

        def integrands(phi):
            rA = np.sqrt((x - x0*np.cos(phi))**2 + (y - y0)**2 \
                         + (x0*np.sin(phi))**2)

            rho3 = e3.electron_density(rA)
            rho13 = rho1 + rho3
            rho23 = rho2 + rho3
            rho123 = rho12 + rho3

            if e3.pp.has_nonzero_rho_core:
                rho3_val = e3.electron_density(rA, only_valence=True)
            else:
                rho3_val = rho3

            if e1.pp.has_nonzero_rho_core or \
               e3.pp.has_nonzero_rho_core:
                rho13_val = rho1_val + rho3_val
            else:
                rho13_val = rho13

            if e2.pp.has_nonzero_rho_core or \
               e3.pp.has_nonzero_rho_core:
                rho23_val = rho2_val + rho3_val
            else:
                rho23_val = rho23

            if e1.pp.has_nonzero_rho_core or \
               e2.pp.has_nonzero_rho_core or \
               e3.pp.has_nonzero_rho_core:
                rho123_val = rho12_val + rho3_val
            else:
                rho123_val = rho123

            if xc in ['LDA', 'PW92']:
                exc3 = np.sum(rho3 * exc_spl(rho3) * aux)
                exc13 = np.sum(rho13 * exc_spl(rho13) * aux)
                exc23 = np.sum(rho23 * exc_spl(rho23) * aux)
                exc123 = np.sum(rho123 * exc_spl(rho123) * aux)
                evxc3 = np.sum(rho3_val * vxc_spl(rho3) * aux)
                evxc13 = np.sum(rho13_val * vxc_spl(rho13) * aux)
                evxc23 = np.sum(rho23_val * vxc_spl(rho23) * aux)
                evxc123 = np.sum(rho123_val * vxc_spl(rho123) * aux)
            else:
                drdx = (x - x0*np.cos(phi)) / rA
                drdy = (y - y0) / rA
                drdphi = ((x - x0*np.cos(phi))*x0*np.sin(phi) \
                          + x0*np.sin(phi)*x0*np.cos(phi)) / rA

                drho3 = e3.electron_density(rA, der=1)
                grad_rho3_x = drho3 * drdx
                grad_rho3_y = drho3 * drdy
                grad_rho3_phi = drho3 * drdphi / x

                if e3.pp.has_nonzero_rho_core:
                    drho3_val = e3.electron_density(rA, der=1,
                                                    only_valence=True)
                else:
                    drho3_val = drho3

                if xc.add_gradient_corrections:
                    if e3.pp.has_nonzero_rho_core:
                        grad_rho3_val_x = drho3_val * drdx
                        grad_rho3_val_y = drho3_val * drdy
                        grad_rho3_val_phi = drho3_val * drdphi / x
                    else:
                        grad_rho3_val_x = grad_rho3_x
                        grad_rho3_val_y = grad_rho3_y
                        grad_rho3_val_phi = grad_rho3_phi

                sigma = drho3**2 * (drdx**2 + drdy**2 + (drdphi/x)**2)
                out = xc.compute_all(rho3, sigma)
                exc3 = np.sum(rho3 * out['zk'] * aux)
                evxc3 = np.sum(rho3_val * out['vrho'] * aux)
                if xc.add_gradient_corrections:
                    if e3.pp.has_nonzero_rho_core:
                        sigma = drho3 * drho3_val \
                                * (drdx**2 + drdy**2 + (drdphi/x)**2)
                    evxc3 += 2. * np.sum(out['vsigma'] * sigma * aux)

                sigma = (grad_rho1_x + grad_rho3_x)**2 \
                        + (grad_rho1_y + grad_rho3_y)**2 \
                        + grad_rho3_phi**2
                out = xc.compute_all(rho13, sigma)
                exc13 = np.sum(rho13 * out['zk'] * aux)
                evxc13 = np.sum(rho13_val * out['vrho'] * aux)
                if xc.add_gradient_corrections:
                    if e1.pp.has_nonzero_rho_core or \
                       e3.pp.has_nonzero_rho_core:
                        sigma = (grad_rho1_x + grad_rho3_x) \
                                * (grad_rho1_val_x + grad_rho3_val_x) \
                                + (grad_rho1_y + grad_rho3_y) \
                                * (grad_rho1_val_y + grad_rho3_val_y) \
                                + grad_rho3_phi * grad_rho3_val_phi
                    evxc13 += 2. * np.sum(out['vsigma'] * sigma * aux)

                sigma = (grad_rho2_x + grad_rho3_x)**2 \
                        + (grad_rho2_y + grad_rho3_y)**2 \
                        + grad_rho3_phi**2
                out = xc.compute_all(rho23, sigma)
                exc23 = np.sum(rho23 * out['zk'] * aux)
                evxc23 = np.sum(rho23_val * out['vrho'] * aux)
                if xc.add_gradient_corrections:
                    if e2.pp.has_nonzero_rho_core or \
                       e3.pp.has_nonzero_rho_core:
                        sigma = (grad_rho2_x + grad_rho3_x) \
                                * (grad_rho2_val_x + grad_rho3_val_x) \
                                + (grad_rho2_y + grad_rho3_y) \
                                * (grad_rho2_val_y + grad_rho3_val_y) \
                                + grad_rho3_phi * grad_rho3_val_phi
                    evxc23 += 2. * np.sum(out['vsigma'] * sigma * aux)

                sigma = (grad_rho12_x + grad_rho3_x)**2 \
                        + (grad_rho12_y + grad_rho3_y)**2 \
                        + grad_rho3_phi**2
                out = xc.compute_all(rho123, sigma)
                exc123 = np.sum(rho123 * out['zk'] * aux)
                evxc123 = np.sum(rho123_val * out['vrho'] * aux)
                if xc.add_gradient_corrections:
                    if e1.pp.has_nonzero_rho_core or \
                       e2.pp.has_nonzero_rho_core or \
                       e3.pp.has_nonzero_rho_core:
                        sigma = (grad_rho12_x + grad_rho3_x) \
                                * (grad_rho12_val_x + grad_rho3_val_x) \
                                + (grad_rho12_y + grad_rho3_y) \
                                * (grad_rho12_val_y + grad_rho3_val_y) \
                                + grad_rho3_phi * grad_rho3_val_phi
                    evxc123 += 2. * np.sum(out['vsigma'] * sigma * aux)

            Exc = exc123 - exc12 - exc13 - exc23 + exc1 + exc2 + exc3
            Evxc = evxc123 - evxc12 - evxc13 - evxc23 + evxc1 + evxc2 + evxc3
            vals = np.array([Exc, -Evxc])
            return vals


        def add_integral(results):
            rmin = 1e-2
            if ((x0**2 + y0**2) < rmin or (x0**2 + (y0 - R)**2) < rmin):
                # Third atom too close to one of the first two atoms
                vals = np.zeros(2)
            elif nphi == 'adaptive':
                epsrel, epsabs = 1e-2, 1e-5
                vals, err = quad_vec(integrands, 0., np.pi,
                                     epsrel=epsrel, epsabs=epsabs)
            else:
                phis = np.linspace(0., np.pi, num=nphi, endpoint=True)
                vals = np.zeros(2)
                dphi = 2. * np.pi / (2 * (nphi - 1))
                for i, phi in enumerate(phis):
                    parts = integrands(phi)
                    if i == 0 or i == nphi-1:
                        parts *= 0.5
                    vals += parts * dphi

            results.append(2. * sum(vals))


        results = []

        # First the values for rCM = 0
        x0 = 0.
        y0 = 0.5 * R
        add_integral(results)

        # Now the main grid
        for r in Sgrid:
            for a in Tgrid:
                x0 = r * np.sin(a)
                y0 = 0.5 * R + r * np.cos(a)
                add_integral(results)

        self.timer.stop('calculate_repulsion3c')
        return results
