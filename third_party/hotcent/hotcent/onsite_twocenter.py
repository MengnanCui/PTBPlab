#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from ase.data import atomic_numbers, atomic_masses
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.slako import (dg, g, INTEGRAL_PAIRS, INTEGRALS, NUMSK,
                           print_integral_overview, select_integrals,
                           tail_smoothening, write_skf)
from hotcent.xc import XC_PW92, LibXC


class Onsite2cTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='monopolar',
                                     **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            smoothen_tails=True, shift=False, superposition='density',
            xc='LDA'):
        """
        Calculates on-site two-center Hamiltonian integrals.

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
        self.timer.start('run_onsite2c')

        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'
        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert superposition == 'density'

        wf_range = self.get_range(wflimit)
        grid, area = self.make_grid(wf_range, nt=ntheta, nr=nr)

        self.Rgrid = rmin + dr * np.arange(N)
        self.tables = {}

        e1, e2 = self.ela, self.elb
        selected = select_integrals(e1, e1)
        print_integral_overview(e1, e1, selected, file=self.txt)

        for bas1a in range(len(e1.basis_sets)):
            for bas1b in range(len(e1.basis_sets)):
                self.tables[(bas1a, bas1b)] = np.zeros((N, NUMSK))

        for i, R in enumerate(self.Rgrid):
            if R < 2 * wf_range:
                if i % 10 == 0:
                    print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                          file=self.txt, flush=True)

                H = self.calculate(selected, e1, e2, R, grid, area,
                                   superposition=superposition, xc=xc)
                for key in selected:
                    integral, nl1a, nl1b = key
                    bas1a = e1.get_basis_set_index(nl1a)
                    bas1b = e1.get_basis_set_index(nl1b)
                    index = INTEGRALS.index(integral)
                    self.tables[(bas1a, bas1b)][i, index] = H[key]

        for key in self.tables:
            for i in range(NUMSK):
                if shift and not np.allclose(self.tables[key][:, i], 0):
                    for j in range(N-1, 1, -1):
                        if abs(self.tables[key][j, i]) > 0:
                            self.tables[key][:j+1, i] -= self.tables[key][j, i]
                            break

                if smoothen_tails:
                    self.tables[key][:, i] = tail_smoothening(self.Rgrid,
                                                        self.tables[key][:, i])
        self.timer.stop('run_onsite2c')

    def calculate(self, selected, e1, e2, R, grid, area,
                  superposition='density', xc='LDA'):
        self.timer.start('calculate_onsite2c')

        assert superposition == 'density'

        # TODO: boilerplate
        # common for all integrals (not wf-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y - R) / r2  # cosine of theta_2
        s1 = x / r1  # sine of theta_1
        s2 = x / r2  # sine of theta_2

        rho = e1.electron_density(r1) + e2.electron_density(r2)
        veff = e1.neutral_atom_potential(r1)
        veff += e2.neutral_atom_potential(r2)

        self.timer.start('vxc')
        if xc in ['LDA', 'PW92']:
            xc = XC_PW92()
            veff += xc.vxc(rho)
        else:
            xc = LibXC(xc)
            drho1 = e1.electron_density(r1, der=1)
            drho2 = e2.electron_density(r2, der=1)
            # TODO: boilerplate
            grad_rho_x = drho1 * s1 + drho2 * s2
            grad_rho_y = drho1 * c1 + drho2 * c2
            sigma = grad_rho_x**2 + grad_rho_y**2
            out = xc.compute_vxc(rho, sigma)
            veff += out['vrho']
            if xc.add_gradient_corrections:
                vsigma = out['vsigma']
                dr1dx = x/r1
                dc1dx = -y*dr1dx / r1**2
                ds1dx = (r1 - x*dr1dx) / r1**2
                dr1dy = y/r1
                dc1dy = (r1 - y*dr1dy) / r1**2
                ds1dy = -x*dr1dy / r1**2
        self.timer.stop('vxc')

        assert np.shape(veff) == (len(grid),)
        V = veff - e1.effective_potential(r1)
        sym1, sym2 = e1.get_symbol(), e2.get_symbol()
        self.timer.stop('prelude')

        results = {}
        for key in selected:
            integral, nl1a, nl1b = key

            Rnl1a = e1.Rnl(r1, nl1a)
            Rnl1b = e1.Rnl(r1, nl1b)
            gphi = g(c1, c1, s1, s1, integral)
            aux = gphi * area * x
            val = np.sum(Rnl1a * Rnl1b * V * aux)

            lm1a, lm1b = INTEGRAL_PAIRS[integral]
            val += e2.pp.get_nonlocal_integral(sym1, sym1, sym2, 0., R, 0.,
                                               nl1a, nl1b, lm1a, lm1b)

            if xc.add_gradient_corrections:
                self.timer.start('vsigma')
                dRnl1a = e1.Rnl(r1, nl1a, der=1)
                dRnl1b = e1.Rnl(r1, nl1b, der=1)
                dgphi = dg(c1, c1, s1, s1, integral)
                dgphidx = (dgphi[0] + dgphi[1]) * dc1dx \
                          + (dgphi[2] + dgphi[3]) * ds1dx
                dgphidy = (dgphi[0] + dgphi[1]) * dc1dy \
                          + (dgphi[2] + dgphi[3]) * ds1dy
                grad_phi_x = (dRnl1a * Rnl1b + Rnl1a * dRnl1b) * s1 * gphi
                grad_phi_x += Rnl1a * Rnl1b * dgphidx
                grad_phi_y = (dRnl1a * Rnl1b + Rnl1a * dRnl1b) * c1 * gphi
                grad_phi_y += Rnl1a * Rnl1b * dgphidy
                grad_rho_grad_phi = grad_rho_x * grad_phi_x \
                                    + grad_rho_y * grad_phi_y
                val += 2. * np.sum(vsigma * grad_rho_grad_phi * area * x)
                self.timer.stop('vsigma')

            results[key] = val

        self.timer.stop('calculate_onsite2c')
        return results

    def write(self):
        """
        Writes all Slater-Koster integral tables to file.

        The filename template corresponds to
        '<el1>-<el1>_onsite2c_<el2>.skf'.

        By default the 'simple' format is chosen, and the 'extended'
        format is only used when necessary (i.e. when there are f-electrons
        included in the valence).
        """
        sym1, sym2 = self.ela.get_symbol(), self.elb.get_symbol()

        for bas1a, valence1a in enumerate(self.ela.basis_sets):
            for bas1b, valence1b in enumerate(self.ela.basis_sets):
                template = '%s-%s_onsite2c_%s.skf'
                filename = template % (sym1 + '+'*bas1a, sym1 + '+'*bas1b, sym2)
                print('Writing to %s' % filename, file=self.txt, flush=True)

                is_extended = any([nl[1] == 'f' for nl in valence1a+valence1b])
                mass = atomic_masses[atomic_numbers[self.ela.get_symbol()]]
                table = self.tables[(bas1a, bas1b)]

                with open(filename, 'w') as f:
                    write_skf(f, self.Rgrid, table, False, is_extended,
                              {}, {}, {}, 0., mass, False, {}, {})
