#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
""" Definition of the AtomicDFT class for atomic
DFT calculations.

The code below draws heavily from the Hotbit code
written by Pekka Koskinen (https://github.com/pekkosk/
hotbit/blob/master/hotbit/parametrization/atom.py).
"""
import pickle
import numpy as np
from ase.units import Ha
from hotcent.interpolation import CubicSplineFunction
from hotcent.atomic_base import AtomicBase
from hotcent.confinement import SoftConfinement, ZeroConfinement
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.radial_grid import RadialGrid
from hotcent.xc import XC_PW92, LibXC
try:
    import _hotcent
except ModuleNotFoundError:
    print('Warning: C-extensions not available')
    from hotcent.shoot import shoot
    _hotcent = None


class AtomicDFT(AtomicBase):
    def __init__(self,
                 symbol,
                 xc='LDA',
                 convergence={'density':1e-7, 'energies':1e-7},
                 perturbative_confinement=True,
                 rmin=None,
                 **kwargs):
        """ Run Kohn-Sham all-electron calculations for a given atom.

        Example:
        ---------
        from hotcent.atomic_dft import AtomicDFT
        from hotcent.confinement import PowerConfinement
        atom = AtomicDFT('C',
                         xc='GGA_C_PBE+GGA_X_PBE',
                         configuration='[He] 2s2 2p2',
                         valence=['2s', '2p'],
                         confinement=PowerConfinement(r0=3.0, s=2))
        atom.run()

        Parameters:
        -----------
        xc: Name of the XC functional. If 'LDA' or 'PW92' are provided,
            then Hotcent's native LDA implementation will be used.
            For all other functionals, the pylibxc module is required,
            which is bundled with Libxc.
            The names of the implemented functionals can be found
            on https://www.tddft.org/programs/libxc/functionals/
            Often one needs to combine different Libxc functionals, e.g.
                xc='GGA_X_PBE+GGA_C_PBE'  # for PBE XC

        convergence: convergence criterion dictionary
                        * density: max change for integrated |n_old-n_new|
                        * energies: max change in single-particle energy (Ha)

        perturbative_confinement: determines which type of self-
            consistent calculation is performed when applying each
            of the subshell- or density-confinement potentials:

            True: add the confinement potential to the effective
                  potential of the free (nonconfined) atom and
                  solve for the subshell(s)* while keeping this
                  potential fixed.

            False: apply the confinement potential in a conventional
                   calculation with self-consistency between
                   the density and the effective potential.

            * i.e. all valence subshells when confining the density and
            only the subshell in question in wave function confinement

            The perturbative scheme is e.g. how basis sets are
            generated in GPAW. This option is also faster than the
            self-consistent one, in particular for heavier atoms.

        rmin: smallest radius in the radial grid (default: 1e-2 / Z).
              For heavier elements, smaller rmin values (e.g. 1e-4 / Z)
              can be needed for high precision.
        """
        AtomicBase.__init__(self, symbol, **kwargs)
        self.timer.start('init')

        self.xcname = xc
        if xc in ['PW92', 'LDA']:
            self.xc = XC_PW92()
        else:
            self.xc = LibXC(xc)

        self.print_header()

        self.convergence = convergence
        self.perturbative_confinement = perturbative_confinement

        maxnodes = max([n - l - 1 for n, l, nl in self.list_states()])
        self.rmin = 1e-2 / self.Z if rmin is None else rmin
        self.N = (maxnodes + 1) * self.nodegpts
        print('max %i nodes, %i grid points' % (maxnodes, self.N),
              file=self.txt)

        self.xgrid = np.linspace(0, np.log(self.rmax / self.rmin), self.N)
        self.rgrid = self.rmin * np.exp(self.xgrid)
        self.grid = RadialGrid(self.rgrid)
        self.timer.stop('init')

    def __del__(self):
        self.timer.summary()

    def print_header(self):
        template = '{0}-relativistic all-electron {1} calculator for {2}'
        header = template.format('Scalar' if self.scalarrel else 'Non',
                                 self.xcname, self.symbol)
        header = '\n'.join(['*' * len(header), header, '*' * len(header)])
        print(header, file=self.txt)

    def calculate_energies(self, enl, dens, dens_xc=None, echo='valence',
                           only_valence=False):
        """ Returns a dictionary with the total energy and its contributions,
        which also get printed out.

        Parameters
        ----------
        enl : dict
            Dictionary with the electronic eigenvalues.
        dens : np.ndarray
            The valence or all-electron electron density on the radial grid.
        dens_xc : np.ndarray, optional
            Electron density to be used instead of 'dens' when evaluating
            the exchange-correlation energies.
        echo : str or None, optional
            Controls the output that gets printed (None, 'valence' or 'all').
        only_valence : bool, optional
            Whether the supplied density is the valence or all-electron
            density. Also determines whether or not the core eigenvalues
            are included in the band energy.

        Returns
        -------
        energies : dict
            Dictionary with the total energy and its contributions.
        """
        self.timer.start('energies')
        assert echo in [None, 'valence', 'all']
        assert not (only_valence and echo == 'all')

        band_energy = 0.0
        for n, l, nl in self.list_states():
            if only_valence and nl not in self.valence:
                continue
            band_energy += self.configuration[nl] * enl[nl]

        vhar = self.calculate_hartree_potential(dens, only_valence=only_valence)
        har_energy = 0.5 * self.grid.integrate(vhar * dens, use_dV=True)

        d = dens if dens_xc is None else dens_xc
        exc, vxc = self.xc.evaluate(d, self.grid)
        vxc_energy = self.grid.integrate(vxc * dens, use_dV=True)
        exc_energy = self.grid.integrate(exc * d, use_dV=True)

        total_energy = band_energy - har_energy - vxc_energy + exc_energy

        if echo is not None:
            line = '%s subshell eigenvalues:' % echo
            print('\n'+line, file=self.txt)
            print('-' * len(line), file=self.txt)
            for n, l, nl in self.list_states():
                if echo == 'all' or nl in self.valence:
                    print('  %s:   %.12f' % (nl, enl[nl]), file=self.txt)

            print('\nenergy contributions:', file=self.txt)
            print('----------------------------------------', file=self.txt)
            print('sum of eigenvalues:     %.12f' % band_energy, file=self.txt)
            print('Hartree energy:         %.12f' % har_energy, file=self.txt)
            print('vxc correction:         %.12f' % vxc_energy, file=self.txt)
            print('exchange-corr. energy:  %.12f' % exc_energy, file=self.txt)
            print('----------------------------------------', file=self.txt)
            print('total energy:           %.12f' % total_energy, file=self.txt)
            print(file=self.txt)

        energies = {'total': total_energy,
                    'band': band_energy,
                    'hartree': har_energy,
                    'vxc': vxc_energy,
                    'exc': exc_energy}

        self.timer.stop('energies')
        return energies

    def calculate_density(self, unlg, only_valence=False):
        """ Calculate the radial electron density:
        sum_nl occ_nl |Rnl(r)|**2 / (4*pi)
        """
        self.timer.start('density')
        dens = np.zeros_like(self.rgrid)
        for n, l, nl in self.list_states():
            if only_valence and nl not in self.valence:
                continue
            dens += self.configuration[nl] * (unlg[nl] ** 2)

        nel1 = self.grid.integrate(dens)
        nel2 = self.get_number_of_electrons(only_valence=only_valence)

        if abs(nel1 - nel2) > 1e-10:
            err = 'Integrated density %.3g' % nel1
            err += ', number of electrons %.3g' % nel2
            raise RuntimeError(err)

        dens = dens / (4 * np.pi * self.rgrid ** 2)

        self.timer.stop('density')
        return dens

    def calculate_hartree_potential(self, dens, only_valence=False, nel=None):
        """ Calculate the Hartree potential. """
        self.timer.start('Hartree')
        dV = self.grid.get_dvolumes()
        r, r0 = self.rgrid, self.grid.get_r0grid()
        N = self.N

        if nel is None:
            nel = self.get_number_of_electrons(only_valence=only_valence)

        if np.isclose(nel, 0.):
            n0 = np.zeros(np.size(dens) - 1)
        else:
            n0 = 0.5 * (dens[1:] + dens[:-1])
            n0 *= nel / np.sum(n0 * dV)

        if _hotcent is not None:
            vhar = _hotcent.hartree(n0, dV, r, r0, N)
        else:
            lo, hi, vhar = np.zeros(N), np.zeros(N), np.zeros(N)
            lo[0] = 0.0
            for i in range(1, N):
                lo[i] = lo[i-1] + dV[i-1] * n0[i-1]

            hi[-1] = 0.0
            for i in range(N - 2, -1, -1):
                hi[i] = hi[i + 1] + n0[i] * dV[i] / r0[i]

            for i in range(N):
                vhar[i] = lo[i] / r[i] + hi[i]

        self.timer.stop('Hartree')
        return vhar

    def calculate_veff(self, dens):
        """ Calculate effective potential. """
        self.timer.start('veff')
        vnuc = self.nuclear_potential(self.rgrid)
        vhar = self.calculate_hartree_potential(dens)
        exc, vxc = self.xc.evaluate(dens, self.grid)
        vconf = self.confinement(self.rgrid)
        self.timer.stop('veff')
        return vnuc + vhar + vxc + vconf

    def guess_density(self):
        """ Guess initial density. """
        r2 = 0.02 * self.Z  # radius at which density has dropped to half;
                            # can this be improved?
        dens = np.exp(-self.rgrid / (r2 / np.log(2)))
        nel = self.get_number_of_electrons()
        dens = dens / self.grid.integrate(dens, use_dV=True) * nel
        return dens

    def run(self, write=None):
        """ Execute the required atomic DFT calculations

        Parameters
        ----------
        write : None or str, optional
            Filename for saving the rgrid, effective
            potential and electron density (if not None).
        """
        def header(*args):
            print('=' * 50, file=self.txt)
            print('\n'.join(args), file=self.txt)
            print('=' * 50, file=self.txt)

        val = self.get_valence_subshells()
        confinement = self.confinement

        assert all([nl in val for nl in self.wf_confinement])
        nl_x = [nl for nl in val if nl not in self.wf_confinement]
        assert len(nl_x) == 0 or len(nl_x) == len(val), nl_x

        self.enl = {}
        self.unlg = {}
        self.Rnlg = {}
        self.unl_fct = {nl: None for nl in self.configuration}
        self.Rnl_fct = {nl: None for nl in self.configuration}

        if self.perturbative_confinement:
            self.confinement = ZeroConfinement()
            header('Initial run without any confinement',
                   'for pre-converging subshells and eigenvalues')
            dens_free, veff_free, enl_free, unlg_free, Rnlg_free = \
                                                               self.outer_scf()

        for nl, wf_confinement in self.wf_confinement.items():
            self.confinement = wf_confinement
            if self.confinement is None:
                self.confinement = ZeroConfinement()
            header('Applying %s' % self.confinement,
                   'to get a confined %s subshell' % nl)

            if self.perturbative_confinement:
                veff = veff_free + self.confinement(self.rgrid)
                enl = {nl: enl_free[nl]}
                itmax, enl, d_enl, unlg, Rnlg = self.inner_scf(0, veff, enl,
                                                               {}, solve=[nl])
                print('Confined %s eigenvalue: %.6f' % (nl, enl[nl]),
                      file=self.txt)
            else:
                dens, veff, enl, unlg, Rnlg = self.outer_scf()

            self.enl[nl] = enl[nl]
            self.unlg[nl] = unlg[nl]
            self.Rnlg[nl] = Rnlg[nl]

        self.confinement = confinement
        if self.confinement is None:
            self.confinement = ZeroConfinement()
        extra = '' if len(nl_x) == 0 else '\nand the confined %s subshell(s)' \
                                           % ' and '.join(nl_x)
        header('Applying %s' % self.confinement,
               'to get the confined electron density%s' % extra)

        if self.perturbative_confinement:
            veff = veff_free + self.confinement(self.rgrid)
            enl = {nl_: enl_free[nl_] for nl_ in val}
            itmax, enl, d_enl, unlg, Rnlg = self.inner_scf(0, veff, enl, {},
                                                           solve=val)
        else:
            self.dens, veff, enl, unlg, Rnlg = self.outer_scf()

        for n, l, nl in self.list_states():
            if nl not in self.wf_confinement:
                assert nl in enl or self.perturbative_confinement
                self.enl[nl] = enl[nl] if nl in enl else enl_free[nl]
                self.unlg[nl] = unlg[nl] if nl in enl else unlg_free[nl]
                self.Rnlg[nl] = Rnlg[nl] if nl in enl else Rnlg_free[nl]

        if self.perturbative_confinement:
            self.dens = self.calculate_density(self.unlg)

        self.veff = self.calculate_veff(self.dens)
        self.vhar = self.calculate_hartree_potential(self.dens)
        self.densval = self.calculate_density(self.unlg, only_valence=True)
        self.vharval = self.calculate_hartree_potential(self.densval,
                                                        only_valence=True)

        if write is not None:
            with open(write, 'w') as f:
                pickle.dump(self.rgrid, f)
                pickle.dump(self.veff, f)
                pickle.dump(self.dens, f)

        self.solved = True
        self.txt.flush()

    def get_onecenter_integrals(self, nl1, nl2):
        """
        Calculates one-center Hamiltonian and overlap integrals.

        Parameters
        ----------
        nl1, nl2 : str
            Subshell labels.

        Returns
        -------
        H, S : float
            The selected H and S integrals (<phi_nl1|H|phi_nl2>
            and <phi_nl1|phi_nl2>, respectively).
        """
        l = ANGULAR_MOMENTUM[nl2[1]]

        if ANGULAR_MOMENTUM[nl1[1]] != l:
            return 0., 0.

        S = self.grid.integrate(self.unlg[nl1] * self.unlg[nl2])

        # Non-scalar-relativistic H
        hpsi = -0.5 * self.unl(self.rgrid, nl2, der=2)
        hpsi += (self.veff + l*(l+1) / (2.*self.rgrid**2)) * self.unlg[nl2]
        H = self.grid.integrate(self.unlg[nl1] * hpsi)

        if self.scalarrel:
            spl = CubicSplineFunction(self.rgrid, self.veff)
            dveff = spl(self.rgrid, der=1)

            c = 137.036
            eps = H  # initial guess

            while True:
                M = 1. - (self.veff - eps) / (2. * c**2)
                hpsi = -0.5 * self.unl(self.rgrid, nl2, der=2)
                hpsi += (l*(l+1) / (2. * self.rgrid**2) \
                         + M * (self.veff - eps) + eps) * self.unlg[nl2]
                hpsi -= 1. / (4. * M * c**2) * dveff \
                        * (self.unl(self.rgrid, nl2, der=1) \
                           - self.unlg[nl2] / self.rgrid)

                H = self.grid.integrate(self.unlg[nl1] * hpsi)
                de = eps - H
                if abs(de) < 1e-8:
                    break
                else:
                    eps = self.mix * H + (1. - self.mix) * eps

        return H, S

    def outer_scf(self):
        """ Solve the self-consistent potential. """
        self.timer.start('outer_scf')
        print('\nStart iteration...', file=self.txt)
        enl = {nl: 0. for n, l, nl in self.list_states()}
        d_enl = {nl: 0. for n, l, nl in self.list_states()}

        dens = self.guess_density()
        veff = self.nuclear_potential(self.rgrid)
        veff += self.confinement(self.rgrid)

        for it in range(self.maxiter):
            veff *= 1. - self.mix
            veff += self.mix * self.calculate_veff(dens)

            dveff = None
            if self.scalarrel:
                spl = CubicSplineFunction(self.rgrid, veff)
                dveff = spl(self.rgrid, der=1)

            itmax, enl, d_enl, unlg, Rnlg = self.inner_scf(it, veff, enl, d_enl,
                                                           dveff=dveff)
            dens0 = dens.copy()
            dens = self.calculate_density(unlg)
            diff = self.grid.integrate(np.abs(dens - dens0), use_dV=True)

            if diff < self.convergence['density'] and it > 5:
                d_enl_max = max(d_enl.values())
                if d_enl_max < self.convergence['energies']:
                    break

            if np.mod(it, 10) == 0:
                line = 'iter %3i, dn=%.1e>%.1e, max %i sp-iter' % \
                       (it, diff, self.convergence['density'], itmax)
                print(line, file=self.txt, flush=True)

            if it == self.maxiter - 1:
                err = 'Density not converged in %i iterations' % (it + 1)
                raise RuntimeError(err)

        self.energies = self.calculate_energies(enl, dens, echo='valence')
        print('converged in %i iterations' % it, file=self.txt)
        nel = self.get_number_of_electrons()
        line = '%9.4f electrons, should be %9.4f' % \
               (self.grid.integrate(dens, use_dV=True), nel)
        print(line, file=self.txt, flush=True)

        self.timer.stop('outer_scf')
        return dens, veff, enl, unlg, Rnlg

    def inner_scf(self, iteration, veff, enl, d_enl, dveff=None, maxiter=100,
                  solve='all', nodes=None):
        """
        Solve the eigenstates for given effective potential.

        u''(r) - 2*(v_eff(r)+l*(l+1)/(2r**2)-e)*u(r)=0
        ( u''(r) + c0(r)*u(r) = 0 )

        r=r0*exp(x) --> (to get equally spaced integration mesh)

        u''(x) - u'(x) + c0(x(r))*u(r) = 0

        Parameters
        ----------
        iteration : int
            Iteration number in the SCF cycle.
        veff : np.ndarray
            Effective potential on the radial grid.
        enl : dict
            Initial guesses for each subshell's eigenvalue.
        d_enl : dict
            Dictionary that will contain the eigenvalue residuals.
        dveff : np.ndarray or None, optional
            Precomputed derivative of the effective potential
            (used in the scalar-relativistic case).
        maxiter : int, optional
            Maximum number of optimization steps per subshell
            (default: 100).
        solve : str or list of str, optional
            Which subshells to solve. The default 'all' means that all
            subshells will be selected. To only solve for specific
            subshells, use [nl1, nl2, ...].
        nodes : dict or None, optional
            Dictionary with the number of nodes to enforce
            for each subshell. By default (None) this is determined
            from the main quantum numbers, as appropriate for
            all-electron calculations.

        Returns
        -------
        itmax : int
            Maximum number of iterations needed for solving a subshell.
        enl : dict of float
            Dictionary with the solved subshell eigenvalues.
        d_enl : dict of float
            Dictionary with the solved subshell eigenvalue residuals.
        unlg : dict of np.ndarray
            Reduced radial functions on the grid for the solved subshells.
        Rnlg : dict of np.ndarray
            Radial functions on the grid for the solved subshells.
        """
        self.timer.start('inner_scf')

        N = np.argmax(veff == np.inf)
        has_finite_range = N > 0
        if has_finite_range:
            while veff[N-1] > 1e4:
                N -= 1
        else:
            N = self.N

        if self.scalarrel and dveff is None:
            spl = CubicSplineFunction(self.rgrid[:N], veff[:N])
            dveff = spl(self.rgrid[:N], der=1)
        elif not self.scalarrel:
            dveff = np.array([])

        rgrid = self.rgrid
        xgrid = self.xgrid
        dx = xgrid[1] - xgrid[0]
        unlg, Rnlg = {}, {}
        itmax = 0

        for n, l, nl in self.list_states():
            if solve != 'all' and nl not in solve:
                continue

            if nodes is None:
                nodes_in = n - l - 1
            else:
                nodes_in = nodes[nl]

            if iteration == 0:
                eps = -1.0 * self.Z ** 2 / n ** 2
            else:
                eps = enl[nl]

            if iteration <= 3:
                delta = 0.5 * self.Z ** 2 / n ** 2  # previous!!!!!!!!!!
            else:
                delta = d_enl[nl]

            direction = 'none'

            if not has_finite_range:
                epsmax = veff[-1] - l * (l + 1) / (2 * self.rgrid[-1] ** 2)

            it = 0
            u = np.zeros(self.N)
            hist = []

            while True:
                eps0 = eps
                self.timer.start('coeff')
                if _hotcent is not None:
                    c0, c1, c2 = _hotcent.construct_coefficients(l, eps,
                                                veff[:N], dveff, self.rgrid[:N])
                else:
                    c0, c1, c2 = self.construct_coefficients(l, eps, veff[:N],
                                                             dveff=dveff)
                assert c0[-2] < 0 and c0[-1] < 0
                self.timer.stop('coeff')

                # boundary conditions for integration from analytic behaviour
                # (unscaled)
                # u(r)~r**(l+1)   r->0
                # u(r)~exp( -sqrt(c0(r)) ) (set u[-1]=1
                # and use expansion to avoid overflows)
                u[0:2] = rgrid[0:2] ** (l + 1)
                u[N-1] = 0.
                self.timer.start('shoot')
                if _hotcent is not None:
                    u[:N], nodes_out, A, ctp = _hotcent.shoot(u[:N], dx,
                                                              c2, c1, c0, N)
                else:
                    u[:N], nodes_out, A, ctp = shoot(u[:N], dx, c2, c1, c0, N)
                self.timer.stop('shoot')

                self.timer.start('norm')
                norm = self.grid.integrate(u ** 2)
                u /= np.sqrt(norm)
                self.timer.stop('norm')

                if nodes_out > nodes_in:
                    # decrease energy
                    if direction == 'up':
                        delta /= 2
                    eps -= delta
                    direction = 'down'
                elif nodes_out < nodes_in:
                    # increase energy
                    if direction == 'down':
                        delta /= 2
                    eps += delta
                    direction = 'up'
                elif nodes_out == nodes_in:
                    shift = -0.5 * A / (rgrid[ctp] * norm)
                    if abs(shift) < 1e-8:  # convergence
                        break
                    if shift > 0:
                        direction = 'up'
                    elif shift < 0:
                        direction = 'down'
                    eps += shift

                if has_finite_range:
                    if abs(eps - eps0) > 0.5*abs(eps0):
                        eps = eps0 + np.sign(eps - eps0) * 0.5*abs(eps0)
                elif eps > epsmax:
                    eps = 0.5 * (epsmax + eps0)
                hist.append(eps)

                it += 1
                if it > maxiter:
                    msg = 'Epsilon history for %s\n' % nl
                    msg += '\n'.join(map(str, hist)) + '\n'
                    msg += 'nl=%s, eps=%f\n' % (nl, eps)
                    if not has_finite_range:
                        msg += 'max epsilon: %f\n' % epsmax
                    msg += 'Eigensolver out of iterations. Atom not stable?'
                    raise RuntimeError(msg)

            itmax = max(it, itmax)

            if has_finite_range:
                self.smoothen_tail(u, N)

            unlg[nl] = u
            Rnlg[nl] = unlg[nl] / self.rgrid
            d_enl[nl] = abs(eps - enl[nl])
            enl[nl] = eps

            if self.verbose:
                line = '-- state %s, %i eigensolver iterations' % (nl, it)
                line += ', e=%9.5f, de=%9.5f' % (enl[nl], d_enl[nl])
                print(line, file=self.txt)

            assert nodes_out == nodes_in
            assert u[1] > 0.0

        self.timer.stop('inner_scf')
        return itmax, enl, d_enl, unlg, Rnlg

    def construct_coefficients(self, l, eps, veff, dveff=None):
        """ Construct the coefficients for Numerov's method; see shoot.py """
        c = 137.036
        ll = l * (l + 1)
        c2 = np.ones(self.N)
        if not self.scalarrel:
            c0 = -ll - 2 * self.rgrid ** 2 * (veff - eps)
            c1 = -1. * np.ones(self.N)
        else:
            assert dveff is not None
            # from Paolo Giannozzi: Notes on pseudopotential generation
            ScR_mass = 1 - 0.5 * (veff - eps) / c ** 2
            c0 = -ll - 2 * ScR_mass * self.rgrid ** 2 * (veff - eps)
            c0 -= dveff * self.rgrid / (2 * ScR_mass * c ** 2)
            c1 = dveff * self.rgrid / (2 * ScR_mass * c ** 2) - 1
        return c0, c1, c2

    def find_cutoff_radius(self, nl, energy_shift=0.2, tolerance=1e-3,
                           rcuts=None, neglect_density_change=True, **kwargs):
        """
        Returns the subshell cutoff radius such that the corresponding
        energy upshift upon soft confinement equals the given value.

        Parameters
        ----------
        nl : str
            Subshell label.
        energy_shift : float, optional
            Energy shift in eV (default: 0.2).
        tolerance : float, optional
            Tolerance (in eV) for termination of the search
            (default: 1e-3).
        rcuts : dict, optional
            Cutoff radii for the other valence subshells.
            If not given, these will be left unconfined.
        neglect_density_change : bool, optional
            By default (True) the shifted energy is calculated in a
            non-self-consistent fashion from an effective potential
            that corresponds to that of the unconfined atom plus the
            the subshell's confinement potential. By choosing False,
            also the change in electron density due to the subshell's
            confinement is taken into account.

        Other parameters
        ----------------
        kwargs : additional parameters to the SoftConfinement
                 potentials

        Returns
        -------
        rc : float
            The cutoff radius.
        """
        assert energy_shift > 0
        assert nl in self.valence
        assert self.perturbative_confinement

        rmax = float(self.rgrid[-1])

        # Set up the initial confinement potentials
        wf_confinement = self.wf_confinement.copy()
        self.wf_confinement = {}
        for nl_ in self.valence:
            if nl_ != nl and rcuts is not None and nl_ in rcuts:
                rc_ = rcuts[nl_]
            else:
                rc_ = rmax
            self.wf_confinement[nl_] = SoftConfinement(rc=rc_, **kwargs)

        # Find the eigenvalue for the unconfined subshell
        self.run()
        enl_free = self.get_eigenvalue(nl)

        # Bisection
        rmin, _ = self.find_split_radius(nl, 0.3)
        rc = 6.  # initial guess
        diff = np.inf

        while abs(diff) > tolerance:
            if np.isfinite(diff):
                if diff < 0:
                    rmax = rc
                    rc = (rc + rmin) / 2.
                else:
                    rmin = rc
                    rc = (rc + rmax) / 2.

            self.wf_confinement[nl].rc = rc
            try:
                self.run()

                if neglect_density_change:
                    e = self.get_eigenvalue(nl)
                else:
                    e, _ = self.get_onecenter_integrals(nl, nl)

                diff = (e - enl_free) * Ha - energy_shift
            except RuntimeError:
                # Convergence problems. This is usually due to
                # too strong confinement (eigenvalues becoming positive)
                diff = 1.

        self.wf_confinement = wf_confinement
        self.solved = False
        rc = float(rc)
        return rc

    def find_cutoff_radii(self, nl1, nl2, energy_shifts, tolerance=1e-3,
                          **kwargs):
        """
        Returns the valence subshell cutoff radii such that the
        corresponding (self-consistent) energy upshifts upon soft
        confinement equal the given values.

        Parameters
        ----------
        nl1 : str
            First subshell label.
        nl2 : str
            Second subshell label.
        energy_shifts : list of float
            The energy shifts in eV for the two subshells.
        tolerance : float, optional
            Tolerance (in eV) for termination of the search
            (default: 1e-3).

        Other parameters
        ----------------
        kwargs : additional parameters to the SoftConfinement
                 potentials

        Returns
        -------
        rcuts : dict
            Cutoff radii for the given subshells.
        """
        assert self.perturbative_confinement

        rmax = float(self.rgrid[-1])

        # Set up the initial confinement potentials
        wf_confinement = self.wf_confinement.copy()
        self.wf_confinement = {nl_: SoftConfinement(rc=rmax, **kwargs)
                               for nl_ in self.valence}

        # Find the eigenvalues in the free atom
        self.run()
        enl_free1 = self.get_eigenvalue(nl1)
        enl_free2 = self.get_eigenvalue(nl2)

        # Bisection
        rmin = 3.
        rc1 = 6.  # initial guess
        diff1 = np.inf
        diff2 = np.inf

        while max(abs(diff1), abs(diff2)) > tolerance:
            if np.isfinite(diff1):
                if diff1 < 0:
                    rmax = rc1
                    rc1 = (rc1 + rmin) / 2.
                else:
                    rmin = rc1
                    rc1 = (rc1 + rmax) / 2.

            self.wf_confinement = wf_confinement.copy()
            rcuts = {nl1: rc1}

            try:
                rc2 = self.find_cutoff_radius(nl2,
                                              energy_shift=energy_shifts[1],
                                              tolerance=tolerance,
                                              neglect_density_change=False,
                                              rcuts=rcuts, **kwargs)
                rcuts[nl2] = rc2
            except RuntimeError:
                # Convergence problems. This is usually due to
                # too strong confinement (eigenvalues becoming positive)
                diff1 = 1.
                continue

            self.wf_confinement = {
                nl1: SoftConfinement(rc=rc1, **kwargs),
                nl2: SoftConfinement(rc=rc2, **kwargs),
            }
            try:
                self.run()
                e1, _ = self.get_onecenter_integrals(nl1, nl1)
                e2, _ = self.get_onecenter_integrals(nl2, nl2)
                diff1 = (e1 - enl_free1) * Ha - energy_shifts[0]
                diff2 = (e2 - enl_free2) * Ha - energy_shifts[1]
            except RuntimeError:
                # Convergence problems. This is usually due to
                # too strong confinement (eigenvalues becoming positive)
                diff1 = 1.

        self.wf_confinement = wf_confinement
        self.solved = False
        return rcuts

    def find_polarization_radius(self, energy_shift=0.3, fraction=0.25,
                                 l_pol=None, **kwargs):
        """
        Returns the characteristic radius used in GPAW's quasi-Gaussian
        method for generating polarization functions.

        Parameters
        ----------
        energy_shift : float, optional
            Energy shift in eV (default: 0.3).
        fraction : float, optional
            The characteristic radius will be this fraction of the
            obtained cutoff radius (default: 0.25).
        l_pol : int, optional
            Angular momentum of the polarization function.
            Default: see get_polarization_angular_momentum().

        Other parameters
        ----------------
        kwargs : additional parameters to the find_cutoff_radius() method

        Returns
        -------
        r_pol : float
            The characteristic radius.
        """
        assert self.perturbative_confinement

        if l_pol is None:
            l = self.get_polarization_angular_momentum()
        else:
            l = l_pol

        nl = self.get_subshell_to_polarize(l)

        rc = self.find_cutoff_radius(nl, energy_shift=energy_shift,
                                     **kwargs)

        r_pol = fraction * rc
        return r_pol
