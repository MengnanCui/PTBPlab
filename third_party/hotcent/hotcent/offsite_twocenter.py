#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from ase.units import Bohr
from ase.data import atomic_numbers, atomic_masses, covalent_radii
from hotcent.interpolation import CubicSplineFunction
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.slako import (dg, g, INTEGRAL_PAIRS, INTEGRALS, NUMSK,
                           print_integral_overview, select_integrals,
                           tail_smoothening, write_skf)
from hotcent.xc import XC_PW92, LibXC
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class Offsite2cTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            superposition='density', xc='LDA', stride=1, smoothen_tails=True):
        """
        Calculates off-site two-center Hamiltonian and overlap integrals.

        Parameters
        ----------
        rmin : float, optional
            Shortest interatomic separation to consider.
        dr : float, optional
            Grid spacing for the interatomic separations.
        N : int
            Number of grid points for the interatomic separations.
        ntheta : int, optional
            Number of angular divisions in polar grid (more dense towards
            the bonding region).
        nr : int, optional
            Number of radial divisions in polar grid (more dense towards
            the atomic centers). With p=q=2 (default powers in polar grid)
            ntheta ~ 3*nr is optimal (for a given grid size).
            With ntheta=150, nr=50 you get ~1e-4 accuracy for H integrals
            (beyond that, gain is slow with increasing grid size).
        wflimit : float, optional
            Value below which the radial wave functions are considered
            to be negligible. This determines how far the polar grids
            around the atomic centers extend in space.
        superposition : str, optional
            Whether to use the density superposition ('density', default)
            or potential superposition ('potential') scheme for the
            Hamiltonian integrals.
        xc : str, optional
            Name of the exchange-correlation functional to be used
            in calculating the effective potential in the density
            superposition scheme (default: 'LDA').
            If the pylibxc module is available, any LDA or GGA (but not
            hybrid or MGGA) functional available via Libxc can be specified.
            To e.g. use the N12 functional, set 'XC_GGA_X_N12+XC_GGA_C_N12'.
            If pylibxc is not available, only the local density approximation
            xc='LDA' (alias: 'PW92') can be chosen.
        stride : int, optional
            The desired Skater-Koster table typically has quite a large
            number of points (N=500-1000), even though the integrals
            themselves are comparatively smooth. To speed up the
            construction of the SK-table, one can therefore restrict the
            expensive integrations to a subset N' = N // stride and map
            the resulting curves on the N-grid afterwards. The default
            stride (1) means that N' = N (no shortcut).
        smoothen_tails : bool, optional
            Whether to modify the 'tails' of the Slater-Koster integrals
            so that they smoothly decay to zero (default: True).
        """
        self.print_header()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'
        assert superposition in ['density', 'potential']

        self.timer.start('run_offsite2c')
        wf_range = self.get_range(wflimit)
        Nsub = N // stride
        Rgrid = rmin + stride * dr * np.arange(Nsub)
        tables = {}

        for p, (e1, e2) in enumerate(self.pairs):
            selected = select_integrals(e1, e2)
            print_integral_overview(e1, e2, selected, file=self.txt)

            for bas1 in range(len(e1.basis_sets)):
                for bas2 in range(len(e2.basis_sets)):
                    tables[(p, bas1, bas2)] = np.zeros((Nsub, 2*NUMSK))

        for i, R in enumerate(Rgrid):
            if R > 2 * wf_range:
                break

            grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)

            if i == Nsub - 1 or Nsub // 10 == 0 or i % (Nsub // 10) == 0:
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            for p, (e1, e2) in enumerate(self.pairs):
                selected = select_integrals(e1, e2)
                if len(grid) > 0:
                    S, H, H2 = self.calculate(selected, e1, e2, R, grid, area,
                                             xc=xc, superposition=superposition)
                    for key in selected:
                        integral, nl1, nl2 = key
                        bas1 = e1.get_basis_set_index(nl1)
                        bas2 = e2.get_basis_set_index(nl2)
                        index = INTEGRALS.index(integral)
                        tables[(p, bas1, bas2)][i, index] = H[key]
                        tables[(p, bas1, bas2)][i, NUMSK+index] = S[key]

        self.Rgrid = rmin + dr * np.arange(N)

        if stride > 1:
            self.tables = {}
            for key in tables:
                self.tables[key] = np.zeros((N, 2*NUMSK))
                for i in range(2*NUMSK):
                    spl = CubicSplineFunction(Rgrid, tables[key][:, i])
                    self.tables[key][:, i] = spl(self.Rgrid)
        else:
            self.tables = tables

        if smoothen_tails:
            # Smooth the curves near the cutoff
            for key in self.tables:
                for i in range(2*NUMSK):
                    self.tables[key][:, i] = \
                            tail_smoothening(self.Rgrid, self.tables[key][:, i])

        self.timer.stop('run_offsite2c')

    def calculate(self, selected, e1, e2, R, grid, area,
                  superposition='density', xc='LDA', only_overlap=False,
                  symmetrize_kinetic=True):
        """
        Calculates the selected Hamiltonian and overlap integrals.

        Parameters
        ----------
        selected : list of (integral, nl1, nl2) tuples
            Selected integrals to evaluate (for example
            [('dds', '3d', '4d'), (...)]).
        e1 : AtomicBase-like object
            <bra| element.
        e2 : AtomicBase-like object
            |ket> element.
        R : float
            Interatomic distance (e1 is at the origin, e2 at z=R).
        grid : np.ndarray
            2D arrray of grid points in the (x, z)-plane.
        area : np.ndarray
            1D array of areas associated with the grid points.
        superposition : str, optional
            Superposition scheme ('density' or 'potential').
        xc : str, optional
            Exchange-correlation functional (see self.run()).
            Defaults to LDA.
        only_overlap : bool, optional
            Whether to only evaluate the overlap integrals
            (default: False). If True, the 'xc' and 'superposition'
            have no influence.
        symmetrize_kinetic : bool, optional
            Whether to symmetrize the kinetic energy contribution
            (default: True). This helps to ensure that Hamiltonian
            integrals do not change when swapping the bra and ket
            states.

        Returns
        -------
        S: dict
            Overlap integrals (R1 * R2 * angle_part).

        H: dict
            Hamiltonian integrals (<R1 | T + Veff - Vconf1 - Vconf2 | R2>).
            With potential superposition: Veff = Veff1 + Veff2.
            With density superposition: Veff = Vna1 + Vna2 + Vxc(rho1 + rho2).

        H2: dict
            Hamiltonian integrals calculated in an alternative way:
            H2 =  <R1 | (T1 + Veff1) + Veff2 - Vconf1 - Vconf2 | R2>
            = <R1 | H1 + Veff2 - Vconf1 - Vconf2 | R2> (operate with H1 on left)
            = <R1 | e1 + Veff2 - VConf1 - Vconf2 | R2>
            = e1 * S + <R1 | Veff2 - Vconf1 - Vconf2 | R2>
            By comparing H and H2, a numerical error can be estimated.
            This is only performend in the potential superposition scheme.
            In the density superposition scheme H2 only contains zeros.
        """
        self.timer.start('calculate_offsite2c')

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

        self.timer.start('veff')
        if not only_overlap and superposition == 'potential':
            v1 = e1.effective_potential(r1) - e1.confinement(r1)
            v2 = e2.effective_potential(r2) - e2.confinement(r2)
            veff = v1 + v2
        elif not only_overlap and superposition == 'density':
            rho = e1.electron_density(r1) + e2.electron_density(r2)
            veff = e1.neutral_atom_potential(r1)
            veff += e2.neutral_atom_potential(r2)
            if xc in ['LDA', 'PW92']:
                xc = XC_PW92()
                veff += xc.vxc(rho)
                vsigma = None
            else:
                xc = LibXC(xc)
                drho1 = e1.electron_density(r1, der=1)
                drho2 = e2.electron_density(r2, der=1)
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
                    dr2dx = x/r2
                    dc2dx = -(y - R)*dr2dx / r2**2
                    ds2dx = (r2 - x*dr2dx) / r2**2
                    dr1dy = y/r1
                    dc1dy = (r1 - y*dr1dy) / r1**2
                    ds1dy = -x*dr1dy / r1**2
                    dr2dy = (y - R)/r2
                    dc2dy = (r2 - (y - R)*dr2dy) / r2**2
                    ds2dy = -x*dr2dy / r2**2
        self.timer.stop('veff')

        if not only_overlap:
            assert np.shape(veff) == (len(grid),)
        self.timer.stop('prelude')

        # calculate all selected integrals
        Sl, Hl, H2l = {}, {}, {}
        sym1, sym2 = e1.get_symbol(), e2.get_symbol()

        for key in selected:
            integral, nl1, nl2 = key

            gphi = g(c1, c2, s1, s2, integral)
            aux = gphi * area * x

            Rnl1 = e1.Rnl(r1, nl1)
            Rnl2 = e2.Rnl(r2, nl2)
            S = np.sum(Rnl1 * Rnl2 * aux)
            Sl[key] = S

            if not only_overlap:
                l2 = ANGULAR_MOMENTUM[nl2[1]]
                ddunl2 = e2.unl(r2, nl2, der=2)

                H = np.sum(Rnl1 * (-0.5 * ddunl2 / r2 + (veff + \
                           l2 * (l2 + 1) / (2 * r2**2)) * Rnl2) * aux)

                if symmetrize_kinetic:
                    l1 = ANGULAR_MOMENTUM[nl1[1]]
                    ddunl1 = e1.unl(r1, nl1, der=2)
                    H += np.sum(Rnl2 * (-0.5 * ddunl1 / r1 + (veff + \
                                l1 * (l1 + 1) / (2 * r1**2)) * Rnl1) * aux)
                    H /= 2.

                lm1, lm2 = INTEGRAL_PAIRS[integral]
                H += e1.pp.get_nonlocal_integral(sym1, sym2, sym1, 0., 0., R,
                                                 nl1, nl2, lm1, lm2)
                H += e2.pp.get_nonlocal_integral(sym1, sym2, sym2, 0., R, R,
                                                 nl1, nl2, lm1, lm2)

                if superposition == 'potential':
                    H2 = np.sum(Rnl1 * Rnl2 * aux * (v2 - e1.confinement(r1)))
                    H2 += e1.get_epsilon(nl1) * S
                elif superposition == 'density':
                    H2 = 0

                if superposition == 'density' and xc.add_gradient_corrections:
                    self.timer.start('vsigma')
                    dRnl1 = e1.Rnl(r1, nl1, der=1)
                    dRnl2 = e2.Rnl(r2, nl2, der=1)
                    dgphi = dg(c1, c2, s1, s2, integral)
                    dgphidx = dgphi[0]*dc1dx + dgphi[1]*dc2dx \
                              + dgphi[2]*ds1dx + dgphi[3]*ds2dx
                    dgphidy = dgphi[0]*dc1dy + dgphi[1]*dc2dy \
                              + dgphi[2]*ds1dy + dgphi[3]*ds2dy
                    grad_phi_x = (dRnl1 * s1 * Rnl2 + Rnl1 * dRnl2 * s2) * gphi
                    grad_phi_x += Rnl1 * Rnl2 * dgphidx
                    grad_phi_y = (dRnl1 * c1 * Rnl2 + Rnl1 * dRnl2 * c2) * gphi
                    grad_phi_y += Rnl1 * Rnl2 * dgphidy
                    grad_rho_grad_phi = grad_rho_x * grad_phi_x \
                                        + grad_rho_y * grad_phi_y
                    H += 2. * np.sum(vsigma * grad_rho_grad_phi * area * x)
                    self.timer.stop('vsigma')

                Hl[key] = H
                H2l[key] = H2

        self.timer.stop('calculate_offsite2c')
        if only_overlap:
            return Sl
        else:
            return Sl, Hl, H2l

    def write(self, eigenvalues=None, hubbardvalues=None, occupations=None,
              spe=None, offdiagonal_H=None, offdiagonal_S=None,
              filename_template='{el1}-{el2}_offsite2c.skf'):
        """
        Writes all Slater-Koster integral tables to file.
        By default the 'simple' SKF format is chosen, and the 'extended'
        SKF format is only used when necessary (i.e. when a basis set
        includes f-electrons).

        Note that the parameters related to one-center properties
        ('eigenvalues', 'hubbardvalues', ..., 'offdiagonal_S')
        are only used in the homonuclear case.

        Parameters
        ----------
        eigenvalues : None or dict, optional
            {nl: value} dictionary with valence subshell eigenvalues
            (or one-center onsite Hamiltonian integrals, if you will).
        hubbardvalues : None or dict, optional
            {nl: value} dictionary with valence subshell Hubbard values.
        occupations : None or dict, optional
            {nl: value} dictionary with valence subshell occupations.
        spe : None or (list of) float, optional
            Spin-polarization error. Needs to be a list for non-minimal
            basis sets.
        offdiagonal_H : None or dict, optional
            {(nl1, nl2): value} dictionary with the off-diagonal,
            one-center, onsite Hamiltonian integrals. Only needed for
            non-minimal basis sets.
        offdiagonal_S : None or dict, optional
            {(nl1, nl2): value} dictionary with the off-diagonal,
            one-center, onsite overlap integrals. Only needed for
            non-minimal basis sets.
        filename_template : str, optional
            Template for the names of the SKF output file(s).
            Needs to contain '{el1}' and '{el2}' fields, which will be
            filled in with the element symbols (followed, as usual,
            with '+' characters in the case of second-or-higher-zeta
            basis subsets).
        """
        def copy_dict1(dict_src, dict_dest, valence):
            if dict_src is None:
                return
            for nl in valence:
                l = nl[1]
                assert l not in dict_dest
                if dict_src is not None and nl in dict_src:
                    dict_dest[l] = dict_src[nl]

        def copy_dict2(dict_src, dict_dest, valence1, valence2):
            if dict_src is None:
                return
            for nl1 in valence1:
                l1 = nl1[1]
                for nl2 in valence2:
                    l2 = nl2[1]
                    if l1 == l2 and (nl1, nl2) in dict_src:
                        assert l1 not in dict_dest
                        dict_dest[l1] = dict_src[(nl1, nl2)]

        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()

            for bas1, valence1 in enumerate(e1.basis_sets):
                for bas2, valence2 in enumerate(e2.basis_sets):
                    filename = filename_template.format(el1=sym1 + '+'*bas1,
                                                        el2=sym2 + '+'*bas2)
                    print('Writing to %s' % filename, file=self.txt, flush=True)

                    is_extended = any([nl[1]=='f' for nl in valence1+valence2])
                    mass = atomic_masses[atomic_numbers[sym1]]

                    eigval, hubval, occup, SPE = {}, {}, {}, 0.
                    has_diagonal_data = sym1 == sym2 and bas1 == bas2
                    if has_diagonal_data:
                        copy_dict1(eigenvalues, eigval, valence1)
                        copy_dict1(hubbardvalues, hubval, valence1)
                        copy_dict1(occupations, occup, valence1)

                    offdiag_H, offdiag_S = {}, {}
                    has_offdiagonal_data = sym1 == sym2 and bas1 != bas2
                    if has_offdiagonal_data:
                        copy_dict2(offdiagonal_H, offdiag_H, valence1, valence2)
                        copy_dict2(offdiagonal_S, offdiag_S, valence1, valence2)

                    table = self.tables[(p, bas1, bas2)]
                    with open(filename, 'w') as f:
                        write_skf(f, self.Rgrid, table, has_diagonal_data,
                                  is_extended, eigval, hubval, occup, SPE, mass,
                                  has_offdiagonal_data, offdiag_H, offdiag_S)

    def plot(self, filename=None, bas1=0, bas2=0):
        """
        Plot the Slater-Koster tables with matplotlib.

        Parameters
        ----------
        filename : str, optional
            Figure filename (default: '<el1>-<el2>_slako.pdf')
        bas1, bas2 : int, optional
            Basis set indices, when dealing with non-minimal
            basis sets.
        """
        self.timer.start('plotting')
        assert plt is not None, 'Matplotlib could not be imported!'

        fig = plt.figure()
        fig.subplots_adjust(hspace=1e-4, wspace=1e-4)

        el1 = self.ela.get_symbol()
        rmax = 6 * covalent_radii[atomic_numbers[el1]] / Bohr
        ymax = max(1, self.tables[(0, bas1, bas2)].max())
        if self.nel == 2:
            el2 = self.elb.get_symbol()
            rmax = max(rmax, 6 * covalent_radii[atomic_numbers[el2]] / Bohr)
            ymax = max(ymax, self.tables[(1, bas1, bas2)].max())

        for i in range(NUMSK):
            name = INTEGRALS[i]
            ax = plt.subplot(NUMSK//2, 2, i + 1)

            for p, (e1, e2) in enumerate(self.pairs):
                s1, s2 = e1.get_symbol(), e2.get_symbol()
                key = (p, bas1, bas2)

                if p == 0:
                    s = '-'
                    lw = 1
                    alpha = 1.0
                else:
                    s = '--'
                    lw = 4
                    alpha = 0.2

                if np.all(abs(self.tables[key][:, i]) < 1e-10):
                    ax.text(0.03, 0.5 + p * 0.15,
                            'No %s integrals for <%s|%s>' % (name, s1, s2),
                            transform=ax.transAxes, size=10, va='center')

                    if not ax.get_subplotspec().is_last_row():
                        plt.xticks([], [])
                    if not ax.get_subplotspec().is_first_col():
                        plt.yticks([], [])
                else:
                    plt.plot(self.Rgrid, self.tables[key][:, i] , c='r',
                             ls=s, lw=lw, alpha=alpha)
                    plt.plot(self.Rgrid, self.tables[key][:, i+NUMSK], c='b',
                             ls=s, lw=lw, alpha=alpha)
                    plt.axhline(0, c='k', ls='--')
                    ax.text(0.8, 0.1 + p * 0.15, name, size=10,
                            transform=ax.transAxes)

                    if ax.get_subplotspec().is_last_row():
                        plt.xlabel('r (Bohr)')
                    else:
                        plt.xticks([], [])
                    if not ax.get_subplotspec().is_first_col():
                        plt.yticks([],[])

                plt.xlim([0, rmax])
                plt.ylim(-ymax, ymax)

        plt.figtext(0.3, 0.95, 'H', color='r', size=20)
        plt.figtext(0.34, 0.95, 'S', color='b', size=20)
        plt.figtext(0.38, 0.95, ' Slater-Koster tables', size=20)
        sym1, sym2 = self.ela.get_symbol(), self.elb.get_symbol()
        plt.figtext(0.3, 0.92, '(thin solid: <%s|%s>, wide dashed: <%s|%s>)' \
                    % (sym1, sym2, sym2, sym1), size=10)

        if filename is None:
            filename = '%s-%s_slako.pdf' % (sym1 + '+'*bas1, sym2 + '+'*bas2)
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
        self.timer.stop('plotting')
