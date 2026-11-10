#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
import pickle
import warnings
from hotcent.atomic_base import NOT_SOLVED_MESSAGE
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import SoftConfinement, ZeroConfinement
from hotcent.interpolation import build_interpolator
from hotcent.orbitals import ANGULAR_MOMENTUM
try:
    import matplotlib.pyplot as plt
except:
    plt = None


class PseudoAtomicDFT(AtomicDFT):
    def __init__(self, symbol, pp, *args, **kwargs):
        """ Create an atomic DFT calculator with a pseudopotential.

        Parameters:
        -----------
        symbol: chemical symbol
        pp: a pseudopotential instance

        The remaining (keyword) arguments are as in the AtomicDFT class.
        """
        AtomicDFT.__init__(self, symbol, *args, **kwargs)

        self.pp = pp
        self.Rnl_free_fct = {nl: None for nl in self.pp.get_subshells()}
        self.veff_free = None
        self.check_pseudo_valence()

        self.nodes = {}
        for nl in sorted(set(self.valence + self.pp.get_subshells())):
            nodes = 0
            for nl2 in self.nodes:
                if nl2[1] == nl[1]:
                    nodes += 1
            self.nodes[nl] = nodes

    def check_pseudo_valence(self):
        """ Checks whether the present electronic configuration is the
        same as the one for which the pseudopotential has been generated
        and raises a warning if this is not the case.
        """
        pp_occupations = self.pp.get_occupations()

        all_subshells = list(self.configuration.keys())
        all_subshells += list(pp_occupations.keys())
        all_subshells = set(all_subshells)

        for nl in all_subshells:
            occ_self = self.configuration.get(nl, None)
            occ_pp = pp_occupations.get(nl, None)

            if nl in self.configuration:
                if nl in pp_occupations:
                    consistent = np.isclose(occ_self, occ_pp)
                else:
                    consistent = nl not in self.valence
            else:
                if nl in pp_occupations:
                    consistent = np.isclose(occ_pp, 0)
                else:
                    raise ValueError('Internal error')

            if not consistent:
                warnings.warn('The chosen electronic configuration differs '
                              'from the one for which the pseudopotential has '
                              f'been generated (subshell: {nl})')
        return

    def print_header(self):
        template = '{0}-relativistic pseudopotential {1} calculator for {2}'
        header = template.format('Scalar' if self.scalarrel else 'Non',
                                 self.xcname, self.symbol)
        header = '\n'.join(['*' * len(header), header, '*' * len(header)])
        print(header, file=self.txt)

    def add_core_electron_density(self, dens):
        """ Returns the given density plus any (partial)
        core electron density.
        """
        return dens + self.pp.get_core_density(self.rgrid)

    def Rnl_free(self, r, nl, der=0):
        """ Rnl_free(r, '2p') """
        assert self.solved, NOT_SOLVED_MESSAGE
        if self.Rnl_free_fct[nl] is None:
            self.Rnl_free_fct[nl] = build_interpolator(self.rgrid,
                                                       self.Rnlg_free[nl])
        return self.Rnl_free_fct[nl](r, der=der)

    def calculate_veff(self, dens):
        """ Returns the effective potential (without nonlocal
        pseudopotential contributions) from the given density.
        """
        vhar = self.calculate_hartree_potential(dens, only_valence=True)
        dens_tot = self.add_core_electron_density(dens)
        exc, vxc = self.xc.evaluate(dens_tot, self.grid)
        vconf = self.confinement(self.rgrid)
        vloc = self.pp.local_potential(self.rgrid)
        return vhar + vxc + vconf + vloc

    def guess_density(self):
        """ Returns a guess for the initial (valence) electron density. """
        dens = self.pp.get_valence_density(self.rgrid)
        return dens

    def run(self, fixed_veff=None, write=None):
        """ Execute the required atomic DFT calculations.

        Parameters
        ----------
        fixed_veff : None or np.ndarray, optional
            Effective potential to use for obtaining the minimal
            basis functions (without confinement or semilocal
            pseudopotential contributions). The default (None)
            is to determine this potential from the density
            of the unconfined atom.
        write : None or str, optional
            Filename for saving the rgrid, effective
            potential and electron density (if not None).
        """
        def header(*args):
            print('=' * 50, file=self.txt)
            print('\n'.join(args), file=self.txt)
            print('=' * 50, file=self.txt)

        assert self.perturbative_confinement
        assert all([nl in self.valence for nl in self.wf_confinement])
        nl_x = [nl for nl in self.valence if nl not in self.wf_confinement]
        assert len(nl_x) == 0 or len(nl_x) == len(self.valence), nl_x

        self.enl = {}
        self.unlg = {}
        self.unl_fct = {nl: None for nl in self.valence}
        self.Rnlg = {}
        self.Rnl_fct = {nl: None for nl in self.valence}
        self.Rnl_free_fct = {nl: None for nl in self.pp.get_subshells()}

        header('Initial run without any confinement',
               'for pre-converging subshells and eigenvalues')
        self.confinement = ZeroConfinement()
        dens_free, self.veff_free, self.enl_free, unlg_free, self.Rnlg_free = \
                                                                self.outer_scf()
        print('\nEigenvalues in the free atom:', file=self.txt)
        for nl in self.valence:
            print('%s: %.6f' % (nl, self.enl_free[nl]), file=self.txt)

        # Additional inner SCF calls for unoccupied states
        # that are needed for generating pseudopotential projectors
        for nl in self.pp.get_subshells():
            if nl not in self.valence:
                # subshell needs to be in self.configuration
                if nl in self.configuration:
                    conf0 = self.configuration[nl]
                    assert conf0 == 0
                else:
                    conf0 = None
                    self.configuration[nl] = 0

                l = ANGULAR_MOMENTUM[nl[1]]
                veff = self.veff_free \
                       + self.pp.semilocal_potential(self.rgrid, l)
                # Add similar weak confinement as Siesta with the
                # KB.New.Reference.Orbitals=True option.
                veff += SoftConfinement(rc=60.)(self.rgrid)

                for enl in [{nl: -0.1}, {nl: 0.1}]:
                    try:
                        itmax, enl, d_enl, unlg, Rnlg = self.inner_scf(
                            1, veff, enl, {}, solve=[nl], nodes=self.nodes)
                        break
                    except RuntimeError:
                        continue
                else:
                    msg = 'Could not solve for %s (needed for projectors)'
                    raise RuntimeError(msg % nl)

                if conf0 is None:
                    del self.configuration[nl]

                self.enl_free.update(enl)
                unlg_free.update(unlg)
                self.Rnlg_free.update(Rnlg)
                print('%s: %.6f' % (nl, self.enl_free[nl]), file=self.txt)

        # Now generate the (usually confined) minimal basis states
        for nl, wf_confinement in self.wf_confinement.items():
            self.confinement = wf_confinement
            if self.confinement is None:
                self.confinement = ZeroConfinement()
            header('Applying %s' % self.confinement,
                   'to get a confined %s subshell' % nl)

            l = ANGULAR_MOMENTUM[nl[1]]
            if fixed_veff is None:
                veff = np.copy(self.veff_free)
            else:
                veff = np.copy(fixed_veff)
            veff += self.confinement(self.rgrid) \
                    + self.pp.semilocal_potential(self.rgrid, l)

            enl = {nl: self.enl_free[nl]}
            itmax, enl, d_enl, unlg, Rnlg = self.inner_scf(0, veff, enl, {},
                                                           solve=[nl],
                                                           nodes=self.nodes)
            print('Confined %s eigenvalue: %.6f' % (nl, enl[nl]),
                  file=self.txt)

            self.enl.update(enl)
            self.unlg.update(unlg)
            self.Rnlg.update(Rnlg)

        for nl in self.valence:
            if nl not in self.wf_confinement:
                self.enl[nl] = self.enl_free[nl]
                self.unlg[nl] = unlg_free[nl]
                self.Rnlg[nl] = self.Rnlg_free[nl]

        self.densval = self.calculate_density(self.unlg, only_valence=True)
        self.dens = self.add_core_electron_density(self.densval)
        self.vharval = self.calculate_hartree_potential(self.densval,
                                                        only_valence=True)
        self.vhar = self.vharval
        self.confinement = ZeroConfinement()
        self.veff = self.calculate_veff(self.densval)

        if write is not None:
            with open(write, 'w') as f:
                pickle.dump(self.rgrid, f)
                pickle.dump(self.veff, f)
                pickle.dump(self.dens, f)

        self.solved = True

        # Calculate and print the total energy contributions
        # Note: we need to pass the eigenvalues calculated with a V_eff
        # that is consistent with the density as a sum of subshell densities.
        # The eigenvalues in self.enl, by contrast, are 'perturbative' in
        # character, calculated with V_eff = V_eff,free + V_confinement.
        enl_sc = {nl: self.get_onecenter_integrals(nl, nl)[0]
                  for nl in self.valence}
        self.energies = self.calculate_energies(enl_sc, self.densval,
                                                dens_xc=self.dens,
                                                echo='valence',
                                                only_valence=True)
        self.timer.summary()
        self.txt.flush()

    def get_onecenter_integrals(self, nl1, nl2):
        """ Wraps around AtomicDFT.get_onecenter_integrals(). """
        assert self.solved, NOT_SOLVED_MESSAGE
        l = ANGULAR_MOMENTUM[nl2[1]]
        veff = np.copy(self.veff)
        self.veff += self.pp.semilocal_potential(self.rgrid, l)
        H, S = AtomicDFT.get_onecenter_integrals(self, nl1, nl2)
        self.veff = veff
        return H, S

    def outer_scf(self):
        """ Solve the self-consistent potential. """
        self.timer.start('outer_scf')
        print('\nStart iteration...', file=self.txt)
        enl = {nl: 0. for nl in self.valence}
        d_enl = {nl: 0. for nl in self.valence}
        unlg = {nl: None for nl in self.valence}
        Rnlg = {nl: None for nl in self.valence}

        dens = self.guess_density()
        veff = self.calculate_veff(dens)

        vnonloc = {}
        for nl in self.valence:
            l = 'spdf'.index(nl[1])
            vnonloc[nl] = self.pp.semilocal_potential(self.rgrid, l)

        for it in range(self.maxiter):
            veff *= 1. - self.mix
            veff += self.mix * self.calculate_veff(dens)

            for nl in self.valence:
                itmax, e, d_e, ug, Rg = self.inner_scf(it, veff + vnonloc[nl],
                                                       enl, d_enl, solve=[nl],
                                                       nodes=self.nodes)
                enl.update(e)
                d_enl.update(d_e)
                unlg.update(ug)
                Rnlg.update(Rg)

            dens0 = dens.copy()
            dens = self.calculate_density(unlg, only_valence=True)
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
                if self.timing:
                    self.timer.summary()
                err = 'Density not converged in %i iterations' % (it + 1)
                raise RuntimeError(err)

        print('converged in %i iterations' % it, file=self.txt)
        nel = self.get_number_of_electrons(only_valence=True)
        line = '%9.4f electrons, should be %9.4f' % \
               (self.grid.integrate(dens, use_dV=True), nel)
        print(line, file=self.txt, flush=True)

        self.timer.stop('outer_scf')
        return dens, veff, enl, unlg, Rnlg
