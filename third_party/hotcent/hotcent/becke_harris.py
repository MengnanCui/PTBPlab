#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from scipy import linalg
import becke
from ase.io.cube import write_cube
from ase.units import Bohr
from hotcent.fluctuation_twocenter import INTEGRALS_2CK
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.orbital_hartree import OrbitalHartreePotential
from hotcent.orbitals import ANGULAR_MOMENTUM, ORBITAL_LABELS, ORBITALS
from hotcent.phillips_kleinman import PhillipsKleinmanPP
from hotcent.slako import get_integral_pair, INTEGRAL_PAIRS as INTEGRAL_PAIRS_2c
from hotcent.spherical_harmonics import (sph_cartesian, sph_cartesian_der,
                                         sph_cartesian_der2)
from hotcent.threecenter import (INTEGRAL_PAIRS as INTEGRAL_PAIRS_3c,
                                 XZ_ANTISYMMETRIC_ORBITALS,
                                 XZ_SYMMETRIC_ORBITALS)
from hotcent.xc import LibXC


def ase2becke(atoms_ase):
    atoms_becke = [(atom.number, tuple(atom.position / Bohr))
                   for atom in atoms_ase]
    return atoms_becke


def set_becke_settings(settings):
    becke.settings.radial_grid_factor, becke.settings.lebedev_order = settings


def h2gpts(h, cell, idiv=1):
    # Based on gpaw.utilities.h2gpts
    L_c = (np.linalg.inv(cell)**2).sum(0)**-0.5
    return np.maximum(idiv, (L_c / h / idiv + 0.5).astype(int) * idiv)


class BeckeHarris:
    """
    Calculator for Hamiltonian and overlap matrix elements and
    repulsive energies using Becke integration grids. The electron
    density is taken as the sum of the atomic densities, which is
    similar to how the Harris functional is typically applied.

    Note: this calculator is primarily meant for checking Hotcent's
    integration procedures on few-atom, nonperiodic structures.
    As such, it is not optimized for speed.

    Note: this requires the [becke](
    https://github.com/humeniuka/becke_multicenter_integration)
    Python package written by Alexander Humeniuk.

    Parameters
    ----------
    elements : dict of AtomicBase-like
        Atomic calculators for every anticipated chemical element.
    xc : str, optional
        Exchange-correlation functional from Libxc.
        Default: LDA_X+LDA_C_PW, i.e. LDA in the PW92 parametrization.
    settings_T, settings_V : tuple of 2 ints
        Settings for the Becke integration grids for integrals involving
        kinetic operators and potential operators, respectively.
        The first item is the radial grid factor, the second the
        Lebedev order for the angular grid.
    """
    def __init__(self, elements, xc='LDA_X+LDA_C_PW', settings_T=(13, 47),
                 settings_V=(13, 47)):
        self.elements = elements
        self.xc = LibXC(xc)
        self.settings_T = settings_T
        self.settings_V = settings_V
        set_becke_settings(settings_V)

        # Find the pseudopotential type
        all_pk_pp = all([isinstance(el.pp, PhillipsKleinmanPP)
                         for sym, el in self.elements.items()])
        all_kb_pp = all([isinstance(el.pp, KleinmanBylanderPP)
                         for sym, el in self.elements.items()])
        assert all_pk_pp or all_kb_pp, \
               'Cannot handle mixture of PK and KB pseudopotentials'
        self.pp_type = 'PK' if all_pk_pp else 'KB'

    def get_symbol(self, index):
        return self.atoms_ase.symbols[index]

    def get_position(self, index):
        return self.atoms_ase.positions[index, :] / Bohr

    def get_valence_orbitals(self, index):
        sym = self.get_symbol(index)
        nls, lms = [], []
        for valence in self.elements[sym].basis_sets:
            for nl in valence:
                l = ANGULAR_MOMENTUM[nl[1]]
                for lm in ORBITALS[l]:
                    nls.append(nl)
                    lms.append(lm)
        return (nls, lms)

    def run_selected_HS(self, atoms_ase, mode, check_mc=True):
        self.atoms_ase = atoms_ase
        self.atoms_becke = ase2becke(atoms_ase)
        assert mode in ['onsite', 'offsite']

        N = len(atoms_ase)
        assert N in [1, 2, 3]

        self.iA = 0
        is_onsite = mode == 'onsite'
        self.iB = 0 if is_onsite else 1

        symA = self.get_symbol(self.iA)
        self.elA = self.elements[symA]
        self.xA, self.yA, self.zA = self.get_position(self.iA)

        symB = self.get_symbol(self.iB)
        self.elB = self.elements[symB]
        self.xB, self.yB, self.zB = self.get_position(self.iB)

        # Find out which integrals need to be calculated
        if N == 1:
            assert is_onsite
            pairs = [('%s_%s' % (lmA, lmA), (lmA, lmA))
                     for lmA in ORBITAL_LABELS]
            labels = 'on1c,overlap'
        elif N == 2:
            pairs = [(k, v) for k, v in INTEGRAL_PAIRS_2c.items()]
            labels = 'on2c' if is_onsite else 'off2c,overlap'
        elif N == 3:
            pairs = []
            for integral, (lmA, lmB) in INTEGRAL_PAIRS_3c.items():
                if lmA[0] == 'f' or lmB[0] == 'f':
                    continue
                if lmA in XZ_ANTISYMMETRIC_ORBITALS:
                    is_nonzero = lmB in XZ_ANTISYMMETRIC_ORBITALS
                elif lmA in XZ_SYMMETRIC_ORBITALS:
                    is_nonzero = lmB not in XZ_ANTISYMMETRIC_ORBITALS
                else:
                    raise NotImplementedError(lmA)
                if is_nonzero:
                    pairs.append((integral, (lmA, lmB)))
            labels = 'on3c' if is_onsite else 'off3c'

        # Evaluate the integrals
        for ibasisA, valenceA in enumerate(self.elA.basis_sets):
            for ibasisB, valenceB in enumerate(self.elB.basis_sets):

                results = {}
                for integral, (lmA, lmB) in pairs:
                    for nlA in valenceA:
                        if nlA[1] == lmA[0]:
                            break
                    else:
                        continue

                    for nlB in valenceB:
                        if nlB[1] == lmB[0]:
                            break
                    else:
                        continue

                    self.nlA, self.nlB = nlA, nlB
                    self.lmA, self.lmB = lmA, lmB
                    nlmA = nlA[0] + lmA + nlA[2:]
                    nlmB = nlB[0] + lmB + nlB[2:]

                    print('Running %s_%s %s_%s' % (symA, nlmA, symB, nlmB),
                          flush=True)
                    if is_onsite:
                        out = self.calculate_onsite_integral_approximations()
                    else:
                        out = self.calculate_offsite_integral_approximations()

                    if check_mc:
                        out_mc = self.calculate_multicenter_integral()
                        tol = 1e-8
                    print()

                    if N == 1:
                        if check_mc:
                            assert abs(out_mc['H_mc'] - out['H_1c']) < tol
                        results[integral] = (out['H_1c'], out['S'])
                    elif N == 2:
                        if check_mc:
                            assert abs(out_mc['H_mc'] - out['H_2c']) < tol
                        if is_onsite:
                            results[integral] = out['H_2c'] - out['H_1c']
                        else:
                            results[integral] = (out['H_2c'], out['S'])
                    elif N == 3:
                        if check_mc:
                            assert abs(out_mc['H_mc'] - out['H_3c']) < tol
                        results[integral] = out['H_3c'] - out['H_2c']

                # Print summary
                print('=== {0} [{1}, {2}] ==='.format(labels, ibasisA, ibasisB))
                for integral, values in results.items():
                    print("'%s': " % integral, end='')
                    fmt = lambda x: '%.8f' % x
                    if isinstance(values, tuple):
                        print("(%s)," % ', '.join(map(fmt, values)))
                    else:
                        print(fmt(values) + ",")
                print()

        return

    def run_all_HS(self, atoms_ase, indices_A=None, indices_B=None,
                   print_matrices=True, print_eigenvalues=True):
        self.atoms_ase = atoms_ase
        self.atoms_becke = ase2becke(atoms_ase)

        # Collect atom indices and valence orbital labels
        if indices_A is None:
            indices_A = list(range(len(self.atoms_becke)))
        else:
            assert len(indices_A) == len(set(indices_A)), indices_A

        if indices_B is None:
            indices_B = list(range(len(self.atoms_becke)))
        else:
            assert len(indices_B) == len(set(indices_B)), indices_B

        iAs, nlAs, lmAs = [], [], []
        for index in indices_A:
            nls, lms = self.get_valence_orbitals(index)
            iAs.extend([index] * len(lms))
            nlAs.extend(nls)
            lmAs.extend(lms)

        iBs, nlBs, lmBs = [], [], []
        for index in indices_B:
            nls, lms = self.get_valence_orbitals(index)
            iBs.extend([index] * len(lms))
            nlBs.extend(nls)
            lmBs.extend(lms)

        # Calculate (parts of the) Hamiltonian and overlap matrices
        shape = (len(lmAs), len(lmBs))
        S = np.zeros(shape)
        H_22 = np.zeros(shape)
        H_23 = np.zeros(shape)
        H_32 = np.zeros(shape)
        H_33 = np.zeros(shape)
        H_mc = np.zeros(shape)

        for indexA, (iA, nlA, lmA) in enumerate(zip(iAs, nlAs, lmAs)):
            self.iA, self.nlA, self.lmA = iA, nlA, lmA
            nlmA = nlA[0] + lmA + nlA[2:]
            self.xA, self.yA, self.zA = self.get_position(self.iA)
            symA = self.get_symbol(self.iA)
            self.elA = self.elements[symA]

            for indexB, (iB, nlB, lmB) in enumerate(zip(iBs, nlBs, lmBs)):
                self.iB, self.nlB, self.lmB = iB, nlB, lmB
                nlmB = nlB[0] + lmB + nlB[2:]
                self.xB, self.yB, self.zB = self.get_position(self.iB)
                symB = self.get_symbol(self.iB)
                self.elB = self.elements[symB]

                print('\nRunning iA %d iB %d nlmA %s nlmB %s' % \
                      (self.iA, self.iB, nlmA, nlmB))
                is_onsite = self.iA == self.iB

                if is_onsite:
                    out = self.calculate_onsite_integral_approximations()
                else:
                    out = self.calculate_offsite_integral_approximations()
                out.update(self.calculate_multicenter_integral())
                print(out, flush=True)

                H_mc[indexA, indexB] = out['H_mc']
                H_22[indexA, indexB] = out['H_2c']
                if is_onsite:
                    H_23[indexA, indexB] = out['H_2c']
                    H_32[indexA, indexB] = out['H_3c']
                else:
                    H_23[indexA, indexB] = out['H_3c']
                    H_32[indexA, indexB] = out['H_2c']
                H_33[indexA, indexB] = out['H_3c']
                S[indexA, indexB] = out['S']

        if print_matrices:
            def print_matrix(M, nperline=4):
                print('np.array([')
                for row in M:
                    print('[', end='')
                    i = 0
                    while i < len(row):
                        part = row[i:min(i+nperline, len(row))]
                        items = list(map(lambda x: '%.6e' % x, part))
                        print(', '.join(items), end='')
                        i += nperline
                        if i >= len(row):
                            print('],')
                        else:
                            print(',')
                print('])\n')

            print('=== Overlap matrix S ===')
            print_matrix(S)

            for name, H in zip(['H_22', 'H_23', 'H_32', 'H_33', 'H_mc'],
                               [H_22, H_23, H_32, H_33, H_mc]):
                print('=== Hamiltonian matrix %s ===' % name)
                print_matrix(H, nperline=4)

                if print_eigenvalues:
                    eigE, eigV = linalg.eig(H, b=S)
                    print('bands:')
                    items = list(map(lambda x: '%.6e' % x.real, sorted(eigE)))
                    print('[' + ', '.join(items) + '],')
                    print()

        results = {'S': S, 'H_22': H_22, 'H_23': H_23, 'H_32': H_32,
                   'H_33': H_33, 'H_mc': H_mc}
        return results

    def calculate_onsite_integral_approximations(self):
        assert self.iA == self.iB
        assert self.elA.get_symbol() == self.elB.get_symbol()

        results = {}

        atoms_1c = [self.atoms_becke[self.iA]]
        Sab = becke.overlap(atoms_1c, self.phiA, self.phiB)
        print("<a|b> =", Sab)
        results['S'] = Sab

        if self.lmA == self.lmB:
            set_becke_settings(self.settings_T)
            T = lambda x, y, z: self.phiA(x, y, z) * self.lap_phiB(x, y, z)
            Tab = -0.5 * becke.integral(atoms_1c, T)
            set_becke_settings(self.settings_V)

            V1c = lambda x, y, z: self.phiA(x,y,z) * self.Veff_on1c(x, y, z) \
                                  * self.phiB(x, y, z)
            Vab1c = becke.integral(atoms_1c, V1c)
            nlPP1c = self.get_nonlocal_pseudopotential_term('on1c')
        else:
            Tab = 0.
            Vab1c = 0.
            nlPP1c = 0.
        print("<a|T|b> =", Tab)
        print("<a|V1c|b> =", Vab1c)
        print("<a|nlPP1c|b> =", nlPP1c)
        print("<a|T+V1c+nlPP1c|b> =", Tab+Vab1c+nlPP1c)
        results['H_1c'] = Tab + Vab1c + nlPP1c

        Vcf2c = lambda x, y, z: self.phiA(x, y, z) * self.phiB(x,y,z) \
                                * self.Veff_on2c(x, y, z)
        Vabcf2c = becke.integral(self.atoms_becke, Vcf2c)
        print("<a|Vcf2c|b> =", Vabcf2c)
        nlPP2c = self.get_nonlocal_pseudopotential_term('on2c')
        print("<a|nlPP2c|b> =", nlPP2c)
        print("<a|Vcf2c+nlPP2c|b> =", Vabcf2c+nlPP2c)
        print("<a|T+V1c+Vcf2c+nlPP1c+nlPP2c|b> =",
              Tab+Vab1c+Vabcf2c+nlPP1c+nlPP2c)
        results['H_2c'] = Tab + Vab1c + Vabcf2c + nlPP1c + nlPP2c

        Vcf3c = lambda x, y, z: self.phiA(x,y,z) * self.phiB(x,y,z) \
                                * self.Veff_on3c(x, y, z)
        Vabcf3c = becke.integral(self.atoms_becke, Vcf3c)
        print("<a|Vcf3c|b> =", Vabcf3c)
        print("<a|T+V1c+Vcf2c+Vcf3c+nlPP1c+nlPP2c|b> =",
              Tab+Vab1c+Vabcf2c+Vabcf3c+nlPP1c+nlPP2c)
        results['H_3c'] = Tab + Vab1c + Vabcf2c + Vabcf3c + nlPP1c + nlPP2c
        return results

    def calculate_offsite_integral_approximations(self):
        assert self.iA != self.iB

        results = {}

        atoms_2c = [self.atoms_becke[self.iA], self.atoms_becke[self.iB]]
        Sab = becke.overlap(atoms_2c, self.phiA, self.phiB)
        print("<a|b> =", Sab)
        results['S'] = Sab

        set_becke_settings(self.settings_T)
        T = lambda x, y, z: self.phiA(x, y, z) * self.lap_phiB(x, y, z)
        Tab = -0.5 * becke.integral(atoms_2c, T)
        set_becke_settings(self.settings_V)
        print("<a|T|b> =", Tab)

        V2c = lambda x, y, z: self.phiA(x, y, z) * self.phiB(x, y, z) \
                              * self.Veff_off2c(x, y, z)
        Vab2c = becke.integral(self.atoms_becke, V2c)
        print("<a|V2c|b> =", Vab2c)
        nlPP2c = self.get_nonlocal_pseudopotential_term('off2c')
        print("<a|nlPP2c|b> =", nlPP2c)
        print("<a|T+V2c+nlPP2c|b> =", Tab+Vab2c+nlPP2c)
        results['H_2c'] = Tab + Vab2c + nlPP2c

        V3c = lambda x, y, z: self.phiA(x,y,z) * self.phiB(x, y, z) \
                              * self.Veff_off3c(x, y, z)
        Vab3c = becke.integral(self.atoms_becke, V3c)
        nlPP3c = self.get_nonlocal_pseudopotential_term('off3c')
        print("<a|V3c|b> =", Vab3c)
        print("<a|nlPP3c|b> =", nlPP3c)
        print("<a|V3c+nlPP3c|b> =", Vab3c+nlPP3c)
        print("<a|T+V2c+V3c+nlPP2c+nlPP3c|b> =", Tab+Vab2c+Vab3c+nlPP2c+nlPP3c)
        results['H_3c'] = Tab + Vab2c + Vab3c + nlPP2c + nlPP3c
        return results

    def calculate_multicenter_integral(self):
        results = {}

        if self.iA == self.iB:
            if self.lmA == self.lmB:
                atoms_1c = [self.atoms_becke[self.iA]]
                set_becke_settings(self.settings_T)
                T = lambda x, y, z: self.phiA(x, y, z) * self.lap_phiB(x, y, z)
                Tab = -0.5 * becke.integral(atoms_1c, T)
                set_becke_settings(self.settings_V)
            else:
                Tab = 0.
        else:
            atoms_2c = [self.atoms_becke[self.iA], self.atoms_becke[self.iB]]
            set_becke_settings(self.settings_T)
            T = lambda x, y, z: self.phiA(x, y, z) * self.lap_phiB(x, y, z)
            Tab = -0.5 * becke.integral(atoms_2c, T)
            set_becke_settings(self.settings_V)
        print("<a|T|b> =", Tab)

        nlPPmc = self.get_nonlocal_pseudopotential_term('mc')
        Vmc = lambda x, y, z: self.phiA(x, y, z) * self.phiB(x, y, z) \
                              * self.Veff_mc(x, y, z)
        Vabmc = becke.integral(self.atoms_becke, Vmc)
        print("<a|nlPPmc|b> =", nlPPmc)
        print("<a|Vmc|b> =", Vabmc)
        print("<a|T+Vmc|b> =", Tab+Vabmc)
        print("<a|T+Vmc+nlPPmc|b> =", Tab+Vabmc+nlPPmc)
        results['H_mc'] = Tab+Vabmc+nlPPmc
        return results

    def phiA(self, x, y, z):
        dx, dy, dz = x - self.xA, y - self.yA, z - self.zA
        rA = np.sqrt(dx**2 + dy**2 + dz**2)
        return self.elA.Rnl(rA, self.nlA) \
               * sph_cartesian(dx, dy, dz, rA, self.lmA)

    def phiB(self, x, y, z):
        dx, dy, dz = x - self.xB, y - self.yB, z - self.zB
        rB = np.sqrt(dx**2 + dy**2 + dz**2)
        return self.elB.Rnl(rB, self.nlB) \
               * sph_cartesian(dx, dy, dz, rB, self.lmB)

    def lap_phiA(self, x, y, z):
        dx, dy, dz = x - self.xA, y - self.yA, z - self.zA
        rA = np.sqrt(dx**2 + dy**2 + dz**2)

        Rnl = self.elA.Rnl(rA, self.nlA)
        dRnldr = self.elA.Rnl(rA, self.nlA, der=1)
        d2Rnldr2 = self.elA.Rnl(rA, self.nlA, der=2)

        lap = 0.
        for i, d in enumerate([dx, dy, dz]):
            drdx = d / rA
            d2rdx2 = (rA - d * drdx) / rA**2
            dYlmdx = sph_cartesian_der(dx, dy, dz, rA, self.lmA, der='xyz'[i])
            d2Ylmdx2 = sph_cartesian_der2(dx, dy, dz, rA, self.lmA, der='xyz'[i])
            d2Rnldx2 = d2Rnldr2 * drdx**2 + dRnldr * d2rdx2
            lap += d2Rnldx2 * sph_cartesian(dx, dy, dz, rA, self.lmA) \
                   + 2. * dRnldr * drdx * dYlmdx + Rnl * d2Ylmdx2
        return lap

    def lap_phiB(self, x, y, z):
        dx, dy, dz = x - self.xB, y - self.yB, z - self.zB
        rB = np.sqrt(dx**2 + dy**2 + dz**2)

        Rnl = self.elB.Rnl(rB, self.nlB)
        dRnldr = self.elB.Rnl(rB, self.nlB, der=1)
        d2Rnldr2 = self.elB.Rnl(rB, self.nlB, der=2)

        lap = 0.
        for i, d in enumerate([dx, dy, dz]):
            drdx = d / rB
            d2rdx2 = (rB - d * drdx) / rB**2
            dYlmdx = sph_cartesian_der(dx, dy, dz, rB, self.lmB, der='xyz'[i])
            d2Ylmdx2 = sph_cartesian_der2(dx, dy, dz, rB, self.lmB, der='xyz'[i])
            d2Rnldx2 = d2Rnldr2 * drdx**2 + dRnldr * d2rdx2
            lap += d2Rnldx2 * sph_cartesian(dx, dy, dz, rB, self.lmB) \
                   + 2. * dRnldr * drdx * dYlmdx + Rnl * d2Ylmdx2
        return lap

    def get_rho(self, x, y, z, indices, **kwargs):
        N = np.size(x)
        rho = np.zeros(N)
        for iC in range(len(self.atoms_becke)):
            if iC not in indices: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            rho += self.elements[symC].electron_density(rC, **kwargs)
        return rho

    def get_grad_rho(self, x, y, z, indices, **kwargs):
        N = np.size(x)
        grad_rho = np.zeros((3, N))
        for iC in range(len(self.atoms_becke)):
            if iC not in indices: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            drdx, drdy, drdz = (x - xC)/rC, (y - yC)/rC, (z - zC)/rC
            drhodr = self.elements[symC].electron_density(rC, der=1, **kwargs)
            grad_rho += drhodr * np.array([drdx, drdy, drdz])
        return grad_rho

    def get_grad_rho_grad_sigma(self, x, y, z, indices, **kwargs):
        grad_rho = self.get_grad_rho(x, y, z, indices, **kwargs)
        N = np.size(x)
        grad_sigma = np.zeros((3, N))

        for iC in range(len(self.atoms_becke)):
            if iC not in indices: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)

            drdx, drdy, drdz = (x - xC)/rC, (y - yC)/rC, (z - zC)/rC
            drhodr = self.elements[symC].electron_density(rC, der=1, **kwargs)
            d2rdx2, d2rdy2, d2rdz2 = (1. - drdx**2)/rC, (1. - drdy**2)/rC, \
                                     (1. - drdz**2)/rC
            d2rdxy, d2rdxz, d2rdyz = -drdx*drdy/rC, -drdx*drdz/rC, -drdy*drdz/rC
            d2rhodr2 = self.elements[symC].electron_density(rC, der=2, **kwargs)

            grad_sigma[0, :] += 2. * grad_rho[0, :] \
                                * (d2rhodr2 * drdx * drdx + drhodr * d2rdx2)
            grad_sigma[0, :] += 2. * grad_rho[1, :] \
                                * (d2rhodr2 * drdx * drdy + drhodr * d2rdxy)
            grad_sigma[0, :] += 2. * grad_rho[2, :] \
                                * (d2rhodr2 * drdx * drdz + drhodr * d2rdxz)
            grad_sigma[1, :] += 2. * grad_rho[0, :] \
                                * (d2rhodr2 * drdy * drdx + drhodr * d2rdxy)
            grad_sigma[1, :] += 2. * grad_rho[1, :] \
                                * (d2rhodr2 * drdy * drdy + drhodr * d2rdy2)
            grad_sigma[1, :] += 2. * grad_rho[2, :] \
                                * (d2rhodr2 * drdy * drdz + drhodr * d2rdyz)
            grad_sigma[2, :] += 2. * grad_rho[0, :] \
                                * (d2rhodr2 * drdz * drdx + drhodr * d2rdxz)
            grad_sigma[2, :] += 2. * grad_rho[1, :] \
                                * (d2rhodr2 * drdz * drdy + drhodr * d2rdyz)
            grad_sigma[2, :] += 2. * grad_rho[2, :] \
                                * (d2rhodr2 * drdz * drdz + drhodr * d2rdz2)

        grad_rho_grad_sigma = np.sum(grad_rho * grad_sigma, axis=0)
        return grad_rho_grad_sigma

    def get_lap_rho(self, x, y, z, indices, **kwargs):
        N = np.size(x)
        lap_rho = np.zeros(N)

        for iC in range(len(self.atoms_becke)):
            if iC not in indices: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            drdx, drdy, drdz = (x - xC)/rC, (y - yC)/rC, (z - zC)/rC
            drhodr = self.elements[symC].electron_density(rC, der=1, **kwargs)
            d2rdx2, d2rdy2, d2rdz2 = (1. - drdx**2)/rC, (1. - drdy**2)/rC, \
                                     (1. - drdz**2)/rC
            d2rhodr2 = self.elements[symC].electron_density(rC, der=2, **kwargs)
            lap_rho += d2rhodr2 * drdx**2 + drhodr * d2rdx2
            lap_rho += d2rhodr2 * drdy**2 + drhodr * d2rdy2
            lap_rho += d2rhodr2 * drdz**2 + drhodr * d2rdz2
        return lap_rho

    def get_rho_deriv1(self, x, y, z, indices, **kwargs):
        """ Returns a dict with selected first derivatives of the density. """
        grad_rho = self.get_grad_rho(x, y, z, indices, **kwargs)
        results = {
            'sigma': np.sum(grad_rho**2, axis=0),
        }
        return results

    def get_rho_deriv2(self, x, y, z, indices, **kwargs):
        """ Returns a dict with selected second derivatives of the density. """
        results = {
            'lap_rho': self.get_lap_rho(x, y, z, indices, **kwargs),
            'grad_rho_grad_sigma': self.get_grad_rho_grad_sigma(x, y, z,
                                                                indices,
                                                                **kwargs),
        }
        return results

    def get_exc(self, rho, sigma=None):
        return self.xc.compute_exc(rho, sigma=sigma)

    def get_vxc(self, rho, sigma=None, lap_rho=None, grad_rho_grad_sigma=None):
        out = self.xc.compute_vxc(rho, sigma=sigma, fxc=True)
        vxc = out['vrho']
        if self.xc.add_gradient_corrections:
           vxc -= 2. * out['vsigma'] * lap_rho
           vxc -= 2. * out['v2rhosigma'] * sigma
           vxc -= 2. * out['v2sigma2'] * grad_rho_grad_sigma
        return vxc

    def Veff_on1c(self, x, y, z):
        dx, dy, dz = x - self.xA, y - self.yA, z - self.zA
        rA = np.sqrt(dx**2 + dy**2 + dz**2)
        v = self.elA.neutral_atom_potential(rA)
        rho = self.elA.electron_density(rA)
        drho = {}
        if self.xc.add_gradient_corrections:
            drho.update(self.get_rho_deriv1(x, y, z, [self.iA]))
            drho.update(self.get_rho_deriv2(x, y, z, [self.iA]))
        v += self.get_vxc(rho, **drho)
        return v

    def Veff_off2c(self, x, y, z):
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        rB = np.sqrt((x - self.xB)**2 + (y - self.yB)**2 + (z - self.zB)**2)
        v = self.elA.neutral_atom_potential(rA)
        v += self.elB.neutral_atom_potential(rB)
        rho = self.elA.electron_density(rA) + self.elB.electron_density(rB)
        drho = {}
        if self.xc.add_gradient_corrections:
            drho.update(self.get_rho_deriv1(x, y, z, [self.iA, self.iB]))
            drho.update(self.get_rho_deriv2(x, y, z, [self.iA, self.iB]))
        v += self.get_vxc(rho, **drho)
        return v

    def Veff_on2c(self, x, y, z):
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        rhoA = self.elA.electron_density(rA)
        drho = {}
        if self.xc.add_gradient_corrections:
            drho.update(self.get_rho_deriv1(x, y, z, [self.iA]))
            drho.update(self.get_rho_deriv2(x, y, z, [self.iA]))
        vxcA = self.get_vxc(rhoA, **drho)

        v = 0.
        for iC in range(len(self.atoms_becke)):
            if iC in [self.iA, self.iB]: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            v += self.elements[symC].neutral_atom_potential(rC)
            rhoC = self.elements[symC].electron_density(rC)
            drho = {}
            if self.xc.add_gradient_corrections:
                drho.update(self.get_rho_deriv1(x, y, z, [self.iA, iC]))
                drho.update(self.get_rho_deriv2(x, y, z, [self.iA, iC]))
            v += self.get_vxc(rhoA+rhoC, **drho) - vxcA
        return v

    def Veff_on3c(self, x, y, z):
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        rhoA = self.elA.electron_density(rA)
        drho = {}
        if self.xc.add_gradient_corrections:
            drho.update(self.get_rho_deriv1(x, y, z, [self.iA]))
            drho.update(self.get_rho_deriv2(x, y, z, [self.iA]))
        vxcA = self.get_vxc(rhoA, **drho)

        v = 0.
        for iC in range(len(self.atoms_becke)):
            if iC in [self.iA, self.iB]: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            rhoC = self.elements[symC].electron_density(rC)
            drho = {}
            if self.xc.add_gradient_corrections:
                drho.update(self.get_rho_deriv1(x, y, z, [self.iA, iC]))
                drho.update(self.get_rho_deriv2(x, y, z, [self.iA, iC]))
            vxcAC = self.get_vxc(rhoA+rhoC, **drho)

            for iD in range(len(self.atoms_becke)):
                if iD in [self.iA, self.iB, iC]: continue
                symD = self.get_symbol(iD)
                xD, yD, zD = self.get_position(iD)
                rD = np.sqrt((x - xD)**2 + (y - yD)**2 + (z - zD)**2)
                rhoD = self.elements[symD].electron_density(rD)
                drho = {}
                if self.xc.add_gradient_corrections:
                    drho.update(self.get_rho_deriv1(x, y, z, [self.iA, iD]))
                    drho.update(self.get_rho_deriv2(x, y, z, [self.iA, iD]))
                vxcAD = self.get_vxc(rhoA+rhoD, **drho)
                drho = {}
                if self.xc.add_gradient_corrections:
                    drho.update(self.get_rho_deriv1(x, y, z, [self.iA, iC, iD]))
                    drho.update(self.get_rho_deriv2(x, y, z, [self.iA, iC, iD]))
                v += self.get_vxc(rhoA+rhoC+rhoD, **drho) - vxcAC - vxcAD + vxcA
        v *= 0.5
        return v

    def Veff_off3c(self, x, y, z):
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        rB = np.sqrt((x - self.xB)**2 + (y - self.yB)**2 + (z - self.zB)**2)
        rhoAB = self.elA.electron_density(rA) + self.elB.electron_density(rB)
        drho = {}
        if self.xc.add_gradient_corrections:
            drho.update(self.get_rho_deriv1(x, y, z, [self.iA, self.iB]))
            drho.update(self.get_rho_deriv2(x, y, z, [self.iA, self.iB]))
        vxcAB = self.get_vxc(rhoAB, **drho)

        v = 0.
        for iC in range(len(self.atoms_becke)):
            if iC in [self.iA, self.iB]: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            v += self.elements[symC].neutral_atom_potential(rC)
            rhoABC = rhoAB + self.elements[symC].electron_density(rC)
            drho = {}
            if self.xc.add_gradient_corrections:
                drho.update(self.get_rho_deriv1(x, y, z,
                                                [self.iA, self.iB, iC]))
                drho.update(self.get_rho_deriv2(x, y, z,
                                                [self.iA, self.iB, iC]))
            v += self.get_vxc(rhoABC, **drho) - vxcAB
        return v

    def Veff_mc(self, x, y, z):
        rho = 0.
        v = 0.
        for iC in range(len(self.atoms_becke)):
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            v += self.elements[symC].neutral_atom_potential(rC)
            rho += self.elements[symC].electron_density(rC)
        drho = {}
        if self.xc.add_gradient_corrections:
            all_indices = list(range(len(self.atoms_ase)))
            drho.update(self.get_rho_deriv1(x, y, z, all_indices))
            drho.update(self.get_rho_deriv2(x, y, z, all_indices))
        v += self.get_vxc(rho, **drho)
        return v

    def get_nonlocal_pseudopotential_term(self, expansion):
        assert expansion in ['on1c', 'on2c', 'off2c', 'off3c', 'mc']
        if expansion in ['on1c', 'on2c']:
            assert self.iA == self.iB
        elif expansion in ['off2c', 'off3c']:
            assert self.iA != self.iB

        if self.pp_type == 'PK':
            nlpp = self.get_pseudo_phillips_kleinman(expansion)
        elif self.pp_type == 'KB':
            nlpp = self.get_pseudo_kleinman_bylander(expansion)
        return nlpp

    def get_pseudo_phillips_kleinman(self, expansion):
        if expansion == 'on1c':
            indices = []
        elif expansion == 'on2c':
            indices = []
            for index in range(len(self.atoms_becke)):
                if index != self.iA:
                    indices.append(index)
        elif expansion == 'off2c':
            indices = []
        elif expansion in ['off3c', 'mc']:
            indices = []
            for index in range(len(self.atoms_becke)):
                if index not in [self.iA, self.iB]:
                    indices.append(index)

        nlpp = 0.
        for iC in range(len(self.atoms_becke)):
            if iC not in indices: continue

            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            atoms_AC = [self.atoms_becke[self.iA], self.atoms_becke[iC]]
            atoms_CB = [self.atoms_becke[iC], self.atoms_becke[self.iB]]

            for nC, lC, nlC in self.elements[symC].list_states():
                if nlC in self.elements[symC].valence:
                    continue

                for lmC in ORBITALS[lC]:
                    def phiC(x, y, z):
                        rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                        return self.elements[symC].Rnl(rC, nlC) \
                               * sph_cartesian(x - xC, y - yC, z - zC, rC, lmC)

                    eps = -self.elements[symC].get_eigenvalue(nlC)
                    sAC = becke.overlap(atoms_AC, self.phiA, phiC)
                    sCB = becke.overlap(atoms_CB, phiC, self.phiB)
                    nlpp += sAC * eps * sCB
        return nlpp

    def get_pseudo_kleinman_bylander(self, expansion):
        if expansion == 'on1c':
            indices = [self.iA]
        elif expansion == 'on2c':
            indices = []
            for index in range(len(self.atoms_becke)):
                if index != self.iA:
                    indices.append(index)
        elif expansion == 'off2c':
            indices = [self.iA, self.iB]
        elif expansion == 'off3c':
            indices = []
            for index in range(len(self.atoms_becke)):
                if index not in [self.iA, self.iB]:
                    indices.append(index)
        elif expansion == 'mc':
            indices = list(range(len(self.atoms_becke)))

        nlpp = 0.
        for iC in range(len(self.atoms_becke)):
            if iC not in indices: continue

            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            atoms_AC = [self.atoms_becke[self.iA], self.atoms_becke[iC]]
            atoms_CB = [self.atoms_becke[iC], self.atoms_becke[self.iB]]

            rAC = np.linalg.norm(self.get_position(iC) \
                                 - self.get_position(self.iA))
            rCB = np.linalg.norm(self.get_position(self.iB) \
                                 - self.get_position(iC))

            for nlC in self.elements[symC].pp.energies.keys():
                lC = ANGULAR_MOMENTUM[nlC[1]]

                for lmC in ORBITALS[lC]:
                    def phiC(x, y, z):
                        rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                        return self.elements[symC].pp.projectors[nlC](rC) \
                               * sph_cartesian(x - xC, y - yC, z - zC, rC, lmC)

                    eps = self.elements[symC].pp.energies[nlC]

                    if rAC < 1e-3:
                        if self.lmA == lmC:
                            sAC = self.elements[symC].pp.overlap_onsite[ \
                                                                (nlC, self.nlA)]
                        else:
                            sAC = 0.
                    else:
                        sAC = becke.overlap(atoms_AC, self.phiA, phiC)

                    if rCB < 1e-3:
                        if self.lmB == lmC:
                            sCB = self.elements[symC].pp.overlap_onsite[ \
                                                                (nlC, self.nlB)]
                        else:
                            sCB = 0.
                    else:
                        sCB = becke.overlap(atoms_CB, phiC, self.phiB)

                    nlpp += sAC * eps * sCB
        return nlpp

    def run_repulsion(self, atoms_ase, **kwargs):
        self.atoms_ase = atoms_ase
        self.atoms_becke = ase2becke(atoms_ase)
        print('=== Repulsion ===')

        def print_energies(results, prefix):
            print(prefix, 'energies:')
            for key, val in results.items():
                if isinstance(val, list):
                    print(key, '=', ' '.join(map(lambda x: '%.8f' % x, val)))
                else:
                    print(key, '=', '%.8f' % val)

            Etot = np.sum(results['Enuc']) - 0.5 * np.sum(results['Evhar'])
            Etot += np.sum(results['Exc']) - np.sum(results['Evxc'])
            print('Etotal = %.8f' % Etot)
            print()
            return

        results_mc = self.calculate_multicenter_repulsion(**kwargs)
        print_energies(results_mc, 'Multi-center')

        results_1c = self.calculate_onecenter_repulsion(**kwargs)
        print_energies(results_1c, 'One-center')

        for key, values in results_1c.items():
            results_mc[key] -= np.sum(values)
        print_energies(results_mc, 'Multi-center minus one-center')

        results_2c = self.calculate_twocenter_repulsion(**kwargs)
        print_energies(results_2c, 'Two-center')

        for key, values in results_2c.items():
            results_mc[key] -= np.sum(values)
        print_energies(results_mc, 'Multi-center minus one- & two-center')
        return

    def calculate_multicenter_repulsion(self, only_valence=True):
        all_indices = list(range(len(self.atoms_becke)))
        only_val = dict(only_valence=only_valence)

        def get_drho(x, y, z, indices, deriv2=True):
            drho = {}
            if self.xc.add_gradient_corrections:
                drho.update(self.get_rho_deriv1(x, y, z, indices))
                if deriv2:
                    drho.update(self.get_rho_deriv2(x, y, z, indices))
            return drho

        def Vhar_rho_mc(x, y, z):
            rho = self.get_rho(x, y, z, all_indices, **only_val)
            vhar = 0.
            for iC in all_indices:
                symC = self.get_symbol(iC)
                xC, yC, zC = self.get_position(iC)
                rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                vhar += self.elements[symC].hartree_potential(rC, **only_val)
            return vhar * rho

        def Exc_rho_mc(x, y, z):
            rho = self.get_rho(x, y, z, all_indices)
            drho = get_drho(x, y, z, all_indices, deriv2=False)
            exc = self.get_exc(rho, **drho)
            return exc * rho

        def Vxc_rho_mc(x, y, z):
            rho = self.get_rho(x, y, z, all_indices)
            rho_val = self.get_rho(x, y, z, all_indices, **only_val)
            drho = get_drho(x, y, z, all_indices, deriv2=True)
            vxc = self.get_vxc(rho, **drho)
            return vxc * rho_val

        def Enuc_mc():
            Enuc = 0.
            for iC in all_indices:
                symC = self.get_symbol(iC)
                xC, yC, zC = self.get_position(iC)
                ZC = self.elements[symC].get_number_of_electrons(**only_val)

                for iD in all_indices:
                    if iD == iC: continue
                    symD = self.get_symbol(iD)
                    xD, yD, zD = self.get_position(iD)
                    ZD = self.elements[symD].get_number_of_electrons(**only_val)
                    rCD = np.sqrt((xD - xC)**2 + (yD - yC)**2 + (zD - zC)**2)
                    Enuc += 0.5 * ZC * ZD / rCD
            return Enuc

        results = {
            'Enuc': Enuc_mc(),
            'Evhar': becke.integral(self.atoms_becke, Vhar_rho_mc),
            'Evxc': becke.integral(self.atoms_becke, Vxc_rho_mc),
            'Exc': becke.integral(self.atoms_becke, Exc_rho_mc),
        }
        return results

    def calculate_onecenter_repulsion(self, only_valence=True):
        results = {key: [] for key in ['Enuc', 'Evhar', 'Evxc', 'Exc']}
        only_val = dict(only_valence=only_valence)

        def get_drho(x, y, z, indices, deriv2=True):
            drho = {}
            if self.xc.add_gradient_corrections:
                drho.update(self.get_rho_deriv1(x, y, z, indices))
                if deriv2:
                    drho.update(self.get_rho_deriv2(x, y, z, indices))
            return drho

        for iC in range(len(self.atoms_becke)):
            atoms_1c = [self.atoms_becke[iC]]
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)

            def Vhar_rho_1c(x, y, z):
                rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                rho = self.elements[symC].electron_density(rC, **only_val)
                vhar = self.elements[symC].hartree_potential(rC, **only_val)
                return vhar * rho

            def Exc_rho_1c(x, y, z):
                rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                rho = self.elements[symC].electron_density(rC)
                drho = get_drho(x, y, z, [iC], deriv2=False)
                return self.get_exc(rho, **drho) * rho

            def Vxc_rho_1c(x, y, z):
                rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                rho = self.elements[symC].electron_density(rC)
                rho_val = self.elements[symC].electron_density(
                             rC, **only_val)
                drho = get_drho(x, y, z, [iC], deriv2=True)
                return self.get_vxc(rho, **drho) * rho_val

            results['Enuc'].append(0.)
            results['Evhar'].append(becke.integral(atoms_1c, Vhar_rho_1c))
            results['Exc'].append(becke.integral(atoms_1c, Exc_rho_1c))
            results['Evxc'].append(becke.integral(atoms_1c, Vxc_rho_1c))

        return results

    def calculate_twocenter_repulsion(self, only_valence=True):
        all_indices = list(range(len(self.atoms_becke)))
        only_val = dict(only_valence=only_valence)

        def get_drho(x, y, z, indices, deriv2=True):
            drho = {}
            if self.xc.add_gradient_corrections:
                drho.update(self.get_rho_deriv1(x, y, z, indices))
                if deriv2:
                    drho.update(self.get_rho_deriv2(x, y, z, indices))
            return drho

        results = {key: [] for key in ['Enuc', 'Evhar', 'Evxc', 'Exc']}

        for iC in all_indices:
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)

            for iD in all_indices:
                if iD == iC: continue
                symD = self.get_symbol(iD)
                xD, yD, zD = self.get_position(iD)
                atoms_2c = [self.atoms_becke[iC], self.atoms_becke[iD]]

                def Vhar_rho_2c(x, y, z):
                    rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                    rhoC = self.elements[symC].electron_density(rC, **only_val)
                    vharC = self.elements[symC].hartree_potential(rC, **only_val)
                    rD = np.sqrt((x - xD)**2 + (y - yD)**2 + (z - zD)**2)
                    rhoD = self.elements[symD].electron_density(rD, **only_val)
                    vharD = self.elements[symD].hartree_potential(rD, **only_val)
                    vhar_rho = (vharC + vharD) * (rhoC + rhoD)
                    vhar_rho -= rhoC * vharC + rhoD * vharD
                    return 0.5 * vhar_rho

                def Exc_rho_2c(x, y, z):
                    rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                    rhoC = self.elements[symC].electron_density(rC)
                    drho = get_drho(x, y, z, [iC], deriv2=False)
                    exc_rho = -self.get_exc(rhoC, **drho) * rhoC

                    rD = np.sqrt((x - xD)**2 + (y - yD)**2 + (z - zD)**2)
                    rhoD = self.elements[symD].electron_density(rD)
                    drho = get_drho(x, y, z, [iD], deriv2=False)
                    exc_rho -= self.get_exc(rhoD, **drho) * rhoD

                    drho = get_drho(x, y, z, [iC, iD], deriv2=False)
                    exc_rho += self.get_exc(rhoC+rhoD, **drho) * (rhoC+rhoD)
                    return 0.5 * exc_rho

                def Vxc_rho_2c(x, y, z):
                    rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                    rhoC = self.elements[symC].electron_density(rC)
                    rhoC_val = self.elements[symC].electron_density(
                                   rC, **only_val)
                    drho = get_drho(x, y, z, [iC], deriv2=True)
                    vxc_rho = -self.get_vxc(rhoC, **drho) * rhoC_val

                    rD = np.sqrt((x - xD)**2 + (y - yD)**2 + (z - zD)**2)
                    rhoD = self.elements[symD].electron_density(rD)
                    rhoD_val = self.elements[symD].electron_density(
                                   rD, **only_val)
                    drho = get_drho(x, y, z, [iD], deriv2=True)
                    vxc_rho -= self.get_vxc(rhoD, **drho) * rhoD_val

                    drho = get_drho(x, y, z, [iC, iD], deriv2=True)
                    vxc_rho += self.get_vxc(rhoC+rhoD, **drho) \
                               * (rhoC_val + rhoD_val)
                    return 0.5 * vxc_rho

                def Enuc_2c():
                    ZC = self.elements[symC].get_number_of_electrons(**only_val)
                    ZD = self.elements[symD].get_number_of_electrons(**only_val)
                    rCD = np.sqrt((xD - xC)**2 + (yD - yC)**2 + (zD - zC)**2)
                    return 0.5 * ZC * ZD / rCD

                results['Enuc'].append(Enuc_2c())
                results['Evhar'].append(becke.integral(atoms_2c, Vhar_rho_2c))
                results['Exc'].append(becke.integral(atoms_2c, Exc_rho_2c))
                results['Evxc'].append(becke.integral(atoms_2c, Vxc_rho_2c))

        return results


class BeckeHarrisKernels(BeckeHarris):
    """
    Abstract calculator for "kernel" matrix elements involving
    second derivatives of the Hartree and XC energies.

    Parameters
    ----------
    See BeckeHarris.__init__()
    """
    def __init__(self, *args, xc='LDA_X+LDA_C_PW', **kwargs):
        BeckeHarris.__init__(self, *args, xc=xc, **kwargs)
        self.xc_polarized = LibXC(xc, spin_polarized=True)

    def get_fxc(self, rho, spin):
        assert not self.xc.add_gradient_corrections, 'Not yet implemented'

        if spin:
            out = self.xc_polarized.compute_vxc_polarized(rho / 2., rho / 2.,
                                                          fxc=True)
            fxc = (out['v2rho2_up'] - out['v2rho2_updown']) / 2.
        else:
            out = self.xc.compute_vxc(rho, fxc=True)
            fxc = out['v2rho2']
        return fxc

    def fxc_on1c(self, x, y, z, spin):
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        rho = self.elA.electron_density(rA)
        fxc = self.get_fxc(rho, spin)
        return fxc

    def fxc_off2c(self, x, y, z, spin):
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        rB = np.sqrt((x - self.xB)**2 + (y - self.yB)**2 + (z - self.zB)**2)
        rho = self.elA.electron_density(rA) + self.elB.electron_density(rB)
        fxc = self.get_fxc(rho, spin)
        return fxc

    def fxc_on2c(self, x, y, z, spin):
        all_indices = list(range(len(self.atoms_becke)))
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        rhoA = self.elA.electron_density(rA)
        fxcA = self.get_fxc(rhoA, spin)

        fxc = 0.
        for iC in all_indices:
            if iC in [self.iA, self.iB]: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            rhoC = self.elements[symC].electron_density(rC)
            fxc += self.get_fxc(rhoA+rhoC, spin) - fxcA
        return fxc

    def fxc_mc(self, x, y, z, spin):
        all_indices = list(range(len(self.atoms_becke)))
        rho = self.get_rho(x, y, z, all_indices)
        fxc = self.get_fxc(rho, spin)
        return fxc


class BeckeHarrisMainKernels(BeckeHarrisKernels):
    """
    Calculator for "kernel" matrix elements involving the main basis set
    and second derivatives of the Hartree and XC energies.

    Parameters
    ----------
    See BeckeHarris.__init__()
    """
    def get_valence_subshells(self, index):
        sym = self.get_symbol(index)
        nls = []
        for valence in self.elements[sym].basis_sets:
            nls.extend(valence)
        return nls

    def run_all_kernels(self, atoms_ase, indices_A=None, indices_B=None,
                        print_matrices=True, spin=False,
                        subtract_delta=False):
        self.atoms_ase = atoms_ase
        self.atoms_becke = ase2becke(atoms_ase)

        # Collect atom indices and subshell labels
        if indices_A is None:
            indices_A = list(range(len(self.atoms_becke)))
        else:
            assert len(indices_A) == len(set(indices_A)), indices_A

        if indices_B is None:
            indices_B = list(range(len(self.atoms_becke)))
        else:
            assert len(indices_B) == len(set(indices_B)), indices_B

        iAs, nlAs = [], []
        for index in indices_A:
            nls = self.get_valence_subshells(index)
            iAs.extend([index] * len(nls))
            nlAs.extend(nls)

        iBs, nlBs = [], []
        for index in indices_B:
            nls = self.get_valence_subshells(index)
            iBs.extend([index] * len(nls))
            nlBs.extend(nls)

        # Calculate kernels
        shape = (len(nlAs), len(nlBs))
        K_10 = np.zeros(shape)
        K_20 = np.zeros(shape)
        K_12 = np.zeros(shape)
        K_22 = np.zeros(shape)
        K_mc = np.zeros(shape)

        for indexA, (iA, nlA) in enumerate(zip(iAs, nlAs)):
            self.iA, self.nlA = iA, nlA
            self.xA, self.yA, self.zA = self.get_position(self.iA)
            symA = self.get_symbol(self.iA)
            self.elA = self.elements[symA]

            vharA = self.get_subshell_vharA()

            for indexB, (iB, nlB) in enumerate(zip(iBs, nlBs)):
                self.iB, self.nlB = iB, nlB
                self.xB, self.yB, self.zB = self.get_position(self.iB)
                symB = self.get_symbol(self.iB)
                self.elB = self.elements[symB]

                print('\nRunning iA %d iB %d nlA %s nlB %s %s' % \
                      (self.iA, self.iB, nlA, nlB, spin))
                is_onsite = self.iA == self.iB

                # Hartree contribution
                if spin:
                    KharAB = 0.
                else:
                    KharAB = self.evaluate_KharAB(vharA)
                    if subtract_delta and not is_onsite:
                        R = np.linalg.norm(self.get_position(self.iA) \
                                           - self.get_position(self.iB))
                        KharAB -= 1. / R
                print("<a|fhartree|b> =", KharAB)

                # XC contribution
                if is_onsite:
                    out = self.calculate_onsite_kernel_approximations(spin)
                else:
                    out = self.calculate_offsite_kernel_approximations(spin)
                out.update(self.calculate_multicenter_kernel(spin))
                print(out, flush=True)

                K_mc[indexA, indexB] = out['K_mc'] + KharAB
                if is_onsite:
                    K_10[indexA, indexB] = out['K_1c'] + KharAB
                    K_20[indexA, indexB] = out['K_2c'] + KharAB
                    K_12[indexA, indexB] = out['K_1c'] + KharAB
                else:
                    K_12[indexA, indexB] = out['K_2c'] + KharAB
                K_22[indexA, indexB] = out['K_2c'] + KharAB

        if print_matrices:
            def print_matrix(M, nperline=4):
                print('np.array([')
                for row in M:
                    print('[', end='')
                    i = 0
                    while i < len(row):
                        part = row[i:min(i+nperline, len(row))]
                        items = list(map(lambda x: '%.6e' % x, part))
                        print(', '.join(items), end='')
                        i += nperline
                        if i >= len(row):
                            print('],')
                        else:
                            print(',')
                print('])\n')

            for name, K in zip(['K_10', 'K_20', 'K_12', 'K_22', 'K_mc'],
                               [K_10, K_20, K_12, K_22, K_mc]):
                template = '=== Kernel matrix {0} (spin-polarized = {1}) ==='
                print(template.format(name, spin))
                print_matrix(K, nperline=4)

        results = {'K_10': K_10, 'K_20': K_20, 'K_12': K_12, 'K_22': K_22,
                   'K_mc': K_mc}
        return results

    def calculate_onsite_kernel_approximations(self, spin):
        assert self.iA == self.iB
        assert self.elA.get_symbol() == self.elB.get_symbol()

        results = {}
        atoms_1c = [self.atoms_becke[self.iA]]

        Kxc1c = lambda x, y, z: self.subshellA(x, y, z) \
                                * self.fxc_on1c(x, y, z, spin) \
                                * self.subshellB(x, y, z)
        Kxcab1c = becke.integral(atoms_1c, Kxc1c)
        print("<a|fxc1c|b> =", Kxcab1c)
        results['K_1c'] = Kxcab1c

        Kxc2c = lambda x, y, z: self.subshellA(x, y, z) \
                                * self.fxc_on2c(x, y, z, spin) \
                                * self.subshellB(x, y, z)
        Kxcab2c = becke.integral(self.atoms_becke, Kxc2c)
        print("<a|fxc2c|b> =", Kxcab2c)
        results['K_2c'] = Kxcab1c + Kxcab2c
        return results

    def calculate_offsite_kernel_approximations(self, spin):
        assert self.iA != self.iB

        results = {}

        Kxc2c = lambda x, y, z: self.subshellA(x, y, z) \
                                * self.fxc_off2c(x, y, z, spin) \
                                * self.subshellB(x, y, z)
        Kxcab2c = becke.integral(self.atoms_becke, Kxc2c)
        print("<a|fxc2c|b> =", Kxcab2c)

        results['K_2c'] = Kxcab2c
        return results

    def calculate_multicenter_kernel(self, spin):
        results = {}

        Kxcmc = lambda x, y, z: self.subshellA(x, y, z) \
                                * self.fxc_mc(x, y, z, spin) \
                                * self.subshellB(x, y, z)
        Kxcabmc = becke.integral(self.atoms_becke, Kxcmc)
        print("<a|fxcmc|b> =", Kxcabmc)
        results['K_mc'] = Kxcabmc
        return results

    def subshellA(self, x, y, z):
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        return self.elA.Rnl(rA, self.nlA)**2 / (4. * np.pi)

    def subshellB(self, x, y, z):
        rB = np.sqrt((x - self.xB)**2 + (y - self.yB)**2 + (z - self.zB)**2)
        return self.elB.Rnl(rB, self.nlB)**2 / (4. * np.pi)

    def get_subshell_vharA(self):
        vharA = becke.poisson(self.atoms_becke, self.subshellA)
        return vharA

    def evaluate_KharAB(self, vharA):
        atoms_B = [self.atoms_becke[self.iB]]

        def integrand(x, y, z):
            return vharA(x, y, z) * self.subshellB(x, y, z)

        KharAB = becke.integral(atoms_B, integrand)
        return KharAB


class BeckeHarrisAuxiliaryKernels(BeckeHarrisKernels):
    """
    Calculator for kernel integrals and (Giese-York) mapping coefficients
    for auxiliary basis sets.

    Parameters
    ----------
    See BeckeHarris.__init__()
    """
    def __init__(self, *args, **kwargs):
        BeckeHarrisKernels.__init__(self, *args, **kwargs)

    def run_selected_kernels(self, atoms_ase, mode, spin=False, check_mc=True,
                             lmax=2, nzeta=1, subtract_delta=False):
        self.atoms_ase = atoms_ase
        self.atoms_becke = ase2becke(atoms_ase)
        assert mode in ['onsite', 'offsite']
        tol = 1e-3 if check_mc else None

        N = len(atoms_ase)
        assert N in [1, 2]

        self.iK = 0
        is_onsite = mode == 'onsite'
        self.iL = 0 if is_onsite else 1

        symK = self.get_symbol(self.iK)
        self.elK = self.elements[symK]
        self.xK, self.yK, self.zK = self.get_position(self.iK)

        symL = self.get_symbol(self.iL)
        self.elL = self.elements[symL]
        self.xL, self.yL, self.zL = self.get_position(self.iL)

        self.iA = self.iK
        self.elA = self.elK
        self.xA, self.yA, self.zA = self.xK, self.yK, self.zK

        self.iB = self.iL
        self.elB = self.elL
        self.xB, self.yB, self.zB = self.xL, self.yL, self.zL

        # Find out which integrals need to be calculated
        pairs = []
        if N == 1:
            assert is_onsite
            for l1 in range(lmax+1):
                for lm1 in ORBITALS[l1]:
                    pair = ('%s_%s' % (lm1, lm1), (lm1, lm1))
                    pairs.append(pair)
            labels = 'on1c'
        elif N == 2:
            for integral in INTEGRALS_2CK:
                lm1, lm2 = get_integral_pair(integral)
                l1 = ANGULAR_MOMENTUM[lm1[0]]
                l2 = ANGULAR_MOMENTUM[lm2[0]]
                if l1 <= lmax and l2 <= lmax:
                    pair = (integral, (lm1, lm2))
                    pairs.append(pair)
            labels = 'on2c' if is_onsite else 'off2c'

        # Evaluate the integrals
        for self.nlK in self.elK.aux_basis.select_radial_functions():
            izeta = self.elK.aux_basis.get_zeta_index(self.nlK)
            if izeta+1 > nzeta:
                continue

            for self.nlL in self.elL.aux_basis.select_radial_functions():
                jzeta = self.elL.aux_basis.get_zeta_index(self.nlL)
                if jzeta+1 > nzeta:
                    continue

                results = {}
                for integral, (lmK, lmL) in pairs:
                    self.lmK = lmK
                    self.lmL = lmL
                    print('\nRunning iK %d iL %d lmK %s lmL %s %s' % \
                          (self.iK, self.iL, lmK, lmL, spin), flush=True)

                    # Hartree contribution
                    if spin or (is_onsite and N == 2):
                        KharKL = 0.
                    else:
                        vharK = self.get_orbital_vharK()
                        KharKL = self.evaluate_KharKL(vharK)
                        if subtract_delta and not is_onsite:
                            KharKL -= self.calculate_point_multipole_kernel()
                    print("<a|fhartree|b> =", KharKL)

                    # XC contribution
                    if is_onsite:
                        out = self.calculate_onsite_kernel_approximations(spin)
                    else:
                        out = self.calculate_offsite_kernel_approximations(spin)

                    if check_mc:
                        out.update(self.calculate_multicenter_kernel(spin))
                    print(out, flush=True)

                    if N == 1:
                        if check_mc:
                            assert abs(out['K_mc'] - out['K_1c']) < tol
                        results[integral] = out['K_1c'] + KharKL
                    elif N == 2:
                        if check_mc:
                            assert abs(out['K_mc'] - out['K_2c']) < tol
                        if is_onsite:
                            results[integral] = out['K_2c'] - out['K_1c']
                        else:
                            results[integral] = out['K_2c'] + KharKL

                # Print summary
                print('=== {0} [{1}, {2}] ==='.format(labels, izeta, jzeta))
                for integral, values in results.items():
                    print("'%s': " % integral, end='')
                    fmt = lambda x: '%.8f' % x
                    if isinstance(values, tuple):
                        print("(%s)," % ', '.join(map(fmt, values)))
                    else:
                        print(fmt(values) + ",")
                print()

        return

    def calculate_point_multipole_kernel(self):
        vec = self.get_position(self.iK) - self.get_position(self.iL)
        R = np.linalg.norm(vec)

        if self.lmK == self.lmL == 's':
            KharKL = 4 * np.pi / R
        else:
            KharKL = 0  # not (yet) implemented
        return KharKL

    def get_multipoles(self, lmax):
        lms = []
        for l in range(lmax+1):
            for lm in ORBITALS[l]:
                lms.append(lm)
        return lms

    def run_all_kernels(self, atoms_ase, print_matrices=True, spin=False,
                        lmax={}, nzeta={}, subtract_delta=False, check_mc=True):
        self.atoms_ase = atoms_ase
        self.atoms_becke = ase2becke(atoms_ase)
        indices = list(range(len(self.atoms_ase)))

        # Auxiliary basis indices/orbitals/...
        iXs, nlXs, lmXs = [], [], []
        Nchi_dict = {}  # number of aux basis func for each element

        for iA in indices:
            symA = self.get_symbol(iA)
            elA = self.elements[symA]

            if symA not in Nchi_dict:
                Nchi_dict[symA] = len(self.get_multipoles(lmax[symA])) \
                                  * nzeta[symA]

            for izeta in range(nzeta[symA]):
                lms = self.get_multipoles(lmax[symA])
                lmXs.extend(lms)
                iXs.extend([iA] * len(lms))
                nl = elA.aux_basis.get_radial_label(izeta)
                nlXs.extend([nl] * len(lms))

        shape = (len(lmXs), len(lmXs))
        K_10 = np.zeros(shape)
        K_20 = np.zeros(shape)
        K_12 = np.zeros(shape)
        K_22 = np.zeros(shape)
        K_mc = np.zeros(shape)

        for indexK, (iK, nlK, lmK) in enumerate(zip(iXs, nlXs, lmXs)):
            self.iK = iK
            self.nlK = nlK
            self.lmK = lmK
            self.symK = self.get_symbol(self.iK)
            self.xK, self.yK, self.zK = self.get_position(self.iK)
            self.elK = self.elements[self.symK]
            lK = ANGULAR_MOMENTUM[lmK[0]]

            self.iA = self.iK
            self.xA = self.xK
            self.yA = self.yK
            self.zA = self.zK
            self.elA = self.elements[self.symK]

            if not spin:
                vharK = OrbitalHartreePotential(
                                        self.elK.rgrid,
                                        self.elK.aux_basis.Anlg[(nlK, lK)],
                                        lmax=lmax[self.symK]+1)

            for indexL, (iL, nlL, lmL) in enumerate(zip(iXs, nlXs, lmXs)):
                self.iL = iL
                self.nlL = nlL
                self.lmL = lmL
                self.symL = self.get_symbol(self.iL)
                self.xL, self.yL, self.zL = self.get_position(self.iL)
                self.elL = self.elements[self.symL]

                self.iB = self.iL
                self.xB = self.xL
                self.yB = self.yL
                self.zB = self.zL
                self.elB = self.elements[self.symL]

                print('\nRunning iK %d iL %d nlmK %s %s nlmL %s %s' % \
                      (iK, iL, nlK, lmK, nlL, lmL), flush=True)
                is_onsite = iK == iL

                # Hartree contribution
                if spin:
                    KharKL = 0.
                else:
                    def Khar(x, y, z):
                        dx = x - self.xK
                        dy = y - self.yK
                        dz = z - self.zK
                        r = np.sqrt(dx**2 + dy**2 + dz**2)
                        return vharK.vhar_fct[lK](r) \
                               * sph_cartesian(dx, dy, dz, r, self.lmK) \
                               * self.chiL(x, y, z)

                    atoms_har = [self.atoms_becke[self.iK]]
                    if not is_onsite:
                        atoms_har += [self.atoms_becke[self.iL]]
                    KharKL = becke.integral(atoms_har, Khar)

                    if subtract_delta and not is_onsite:
                        KharKL -= self.calculate_point_multipole_kernel()
                print("<a|fhartree|b> =", KharKL, flush=True)

                # XC contribution
                if is_onsite:
                    out = self.calculate_onsite_kernel_approximations(spin)
                else:
                    out = self.calculate_offsite_kernel_approximations(spin)

                if check_mc:
                    out.update(self.calculate_multicenter_kernel(spin))
                    K_mc[indexK, indexL] = out['K_mc'] + KharKL

                print(out, flush=True)

                if is_onsite:
                    K_10[indexK, indexL] = out['K_1c'] + KharKL
                    K_20[indexK, indexL] = out['K_2c'] + KharKL
                    K_12[indexK, indexL] = out['K_1c'] + KharKL
                else:
                    K_12[indexK, indexL] = out['K_2c'] + KharKL
                K_22[indexK, indexL] = out['K_2c'] + KharKL

        if print_matrices:
            def print_matrix(M, nperline=4):
                print('np.array([')
                for row in M:
                    print('[', end='')
                    i = 0
                    while i < len(row):
                        part = row[i:min(i+nperline, len(row))]
                        items = list(map(lambda x: '%.6e' % x, part))
                        print(', '.join(items), end='')
                        i += nperline
                        if i >= len(row):
                            print('],')
                        else:
                            print(',')
                print('])\n')

            for name, K in zip(['K_10', 'K_20', 'K_12', 'K_22', 'K_mc'],
                               [K_10, K_20, K_12, K_22, K_mc]):
                template = '=== Kernel matrix {0} (spin-polarized = {1}) ==='
                print(template.format(name, spin))
                print_matrix(K, nperline=4)

        results = {'K_10': K_10, 'K_20': K_20, 'K_12': K_12, 'K_22': K_22,
                   'K_mc': K_mc}
        return results

    def calculate_onsite_kernel_approximations(self, spin):
        assert self.iL == self.iK
        assert self.elK.get_symbol() == self.elL.get_symbol()

        results = {}

        if self.xc.add_gradient_corrections:
            KxcKL1c, KxcKL2c = self.calculate_onsite_xc_kernel_gga(spin)
        else:
            KxcKL1c, KxcKL2c = self.calculate_onsite_xc_kernel_lda(spin)

        print("<a|fxc1c|b> =", KxcKL1c)
        results['K_1c'] = KxcKL1c

        print("<a|fxc2c|b> =", KxcKL2c)
        results['K_2c'] = KxcKL1c + KxcKL2c
        return results

    def calculate_onsite_xc_kernel_lda(self, spin):
        atoms_1c = [self.atoms_becke[self.iK]]
        Kxc1c = lambda x, y, z: self.chiK(x, y, z) \
                                * self.fxc_on1c(x, y, z, spin) \
                                * self.chiL(x, y, z)
        KxcKL1c = becke.integral(atoms_1c, Kxc1c)

        Kxc2c = lambda x, y, z: self.chiK(x, y, z) \
                                * self.fxc_on2c(x, y, z, spin) \
                                * self.chiL(x, y, z)
        KxcKL2c = becke.integral(self.atoms_becke, Kxc2c)
        return (KxcKL1c, KxcKL2c)

    def calculate_onsite_xc_kernel_gga(self, spin):
        def kxc_on1c(x, y, z):
            dx, dy, dz = x - self.xK, y - self.yK, z - self.zK
            rK = np.sqrt(dx**2 + dy**2 + dz**2)
            rho = self.elK.electron_density(rK)
            drhodr = self.elK.electron_density(rK, der=1)
            drho = self.get_rho_deriv1(x, y, z, [self.iK])

            if spin:
                rho /= 2
                drhodr /= 2
                drho = dict(sigma_up=drho['sigma'] / 4,
                            sigma_updown=drho['sigma'] / 4,
                            sigma_down=drho['sigma'] / 4)
                out = self.xc_polarized.compute_vxc_polarized(rho, rho,
                                                              fxc=True, **drho)
            else:
                out = self.xc.compute_vxc(rho, fxc=True, **drho)

            mK = self.chiK(x, y, z)
            mL = self.chiL(x, y, z)
            dmKdr = self.AnlK(rK, der=1) \
                    * sph_cartesian(dx, dy, dz, rK, self.lmK)
            dmLdr = self.AnlL(rK, der=1) \
                    * sph_cartesian(dx, dy, dz, rK, self.lmL)
            grad_mK_grad_rho = dmKdr * drhodr
            grad_mL_grad_rho = dmLdr * drhodr

            grad_mK_grad_mL = 0.
            AnlK = self.AnlK(rK, der=0)
            AnlL = self.AnlL(rK, der=0)
            drdx = [dx/rK, dy/rK, dz/rK]
            for i in range(3):
                der = 'xyz'[i]
                dYlmK = sph_cartesian_der(dx, dy, dz, rK, self.lmK, der=der)
                dmKdx = dmKdr * drdx[i] + AnlK * dYlmK
                dYlmL = sph_cartesian_der(dx, dy, dz, rK, self.lmL, der=der)
                dmLdx = dmLdr * drdx[i] + AnlL * dYlmL
                grad_mK_grad_mL += dmKdx * dmLdx

            if spin:
                k = (out['v2rho2_up'] - out['v2rho2_updown']) * mK * mL
                k += 2 * (out['v2rhosigma_up_up'] - out['v2rhosigma_up_down']) \
                     * grad_mK_grad_rho * mL
                k += 2 * (out['v2rhosigma_up_up'] - out['v2rhosigma_up_down']) \
                     * mK * grad_mL_grad_rho
                k += 4 * (out['v2sigma2_up_up'] - out['v2sigma2_up_down']) \
                     * grad_mK_grad_rho * grad_mL_grad_rho
                k += 2 * (out['v2sigma2_up_updown'] - out['v2sigma2_updown_down']) \
                     * grad_mK_grad_rho * grad_mL_grad_rho
                k += (2 * out['vsigma_up'] - out['vsigma_updown']) \
                     * grad_mK_grad_mL
                k /= 2
            else:
                k = out['v2rho2'] * mK * mL
                k += 2 * out['v2rhosigma'] * grad_mK_grad_rho * mL
                k += 2 * out['v2rhosigma'] * mK * grad_mL_grad_rho
                k += 4 * out['v2sigma2'] * grad_mK_grad_rho * grad_mL_grad_rho
                k += 2 * out['vsigma'] * grad_mK_grad_mL
            return k

        def kxc_on2c(x, y, z):
            dxK, dyK, dzK = x - self.xK, y - self.yK, z - self.zK
            rK = np.sqrt(dxK**2 + dyK**2 + dzK**2)
            drKdx = [dxK/rK, dyK/rK, dzK/rK]

            rhoK = self.elK.electron_density(rK)
            drhoKdrK = self.elK.electron_density(rK, der=1)
            drho = self.get_rho_deriv1(x, y, z, [self.iK])

            if spin:
                rhoK /= 2
                drhoKdrK /= 2
                drho = dict(sigma_up=drho['sigma'] / 4,
                            sigma_updown=drho['sigma'] / 4,
                            sigma_down=drho['sigma'] / 4)
                outK = self.xc_polarized.compute_vxc_polarized(rhoK, rhoK,
                                                               fxc=True, **drho)
            else:
                outK = self.xc.compute_vxc(rhoK, fxc=True, **drho)

            mK = self.chiK(x, y, z)
            mL = self.chiL(x, y, z)
            dmKdr = self.AnlK(rK, der=1) \
                    * sph_cartesian(dxK, dyK, dzK, rK, self.lmK)
            dmLdr = self.AnlL(rK, der=1) \
                    * sph_cartesian(dxK, dyK, dzK, rK, self.lmL)

            dYlmKs, dYlmLs = [], []
            grad_mK_grad_mL = 0.
            grad_mK_grad_rho = 0.
            grad_mL_grad_rho = 0.
            AnlK = self.AnlK(rK, der=0)
            AnlL = self.AnlL(rK, der=0)
            for i in range(3):
                der = 'xyz'[i]
                dYlmK = sph_cartesian_der(dxK, dyK, dzK, rK, self.lmK, der=der)
                dYlmKs.append(dYlmK)
                dYlmL = sph_cartesian_der(dxK, dyK, dzK, rK, self.lmL, der=der)
                dYlmLs.append(dYlmL)
                dmKdx = dmKdr * drKdx[i] + AnlK * dYlmK
                dmLdx = dmLdr * drKdx[i] + AnlL * dYlmL
                grad_mK_grad_mL += dmKdx * dmLdx
                drhodx = drhoKdrK * drKdx[i]
                grad_mK_grad_rho += dmKdx * drhodx
                grad_mL_grad_rho += dmLdx * drhodx

            if spin:
                kK = (outK['v2rho2_up'] - outK['v2rho2_updown']) * mK * mL
                kK += 2 * (outK['v2rhosigma_up_up'] - outK['v2rhosigma_up_down']) \
                      * grad_mK_grad_rho * mL
                kK += 2 * (outK['v2rhosigma_up_up'] - outK['v2rhosigma_up_down']) \
                      * mK * grad_mL_grad_rho
                kK += 4 * (outK['v2sigma2_up_up'] - outK['v2sigma2_up_down']) \
                      * grad_mK_grad_rho * grad_mL_grad_rho
                kK += 2 * (outK['v2sigma2_up_updown'] - outK['v2sigma2_updown_down']) \
                      * grad_mK_grad_rho * grad_mL_grad_rho
                kK += (2 * outK['vsigma_up'] - outK['vsigma_updown']) \
                      * grad_mK_grad_mL
                kK /= 2
            else:
                kK = outK['v2rho2'] * mK * mL
                kK += 2 * outK['v2rhosigma'] * grad_mK_grad_rho * mL
                kK += 2 * outK['v2rhosigma'] * mK * grad_mL_grad_rho
                kK += 4 * outK['v2sigma2'] * grad_mK_grad_rho * grad_mL_grad_rho
                kK += 2 * outK['vsigma'] * grad_mK_grad_mL

            k = 0.
            for iM in list(range(len(self.atoms_ase))):
                if iM in [self.iK, self.iL]: continue
                symM = self.get_symbol(iM)
                xM, yM, zM = self.get_position(iM)
                dxM, dyM, dzM = x - xM, y - yM, z - zM
                rM = np.sqrt(dxM**2 + dyM**2 + dzM**2)
                drMdx = [dxM/rM, dyM/rM, dzM/rM]

                rhoM = self.elements[symM].electron_density(rM)
                drhoMdrM = self.elements[symM].electron_density(rM, der=1)
                drho = self.get_rho_deriv1(x, y, z, [self.iK, iM])

                if spin:
                    rhoM /= 2
                    drhoMdrM /= 2
                    drho = dict(sigma_up=drho['sigma'] / 4,
                                sigma_updown=drho['sigma'] / 4,
                                sigma_down=drho['sigma'] / 4)
                    outKM = self.xc_polarized.compute_vxc_polarized(rhoK+rhoM,
                                                rhoK+rhoM, fxc=True, **drho)
                else:
                    outKM = self.xc.compute_vxc(rhoK+rhoM, fxc=True, **drho)

                grad_mK_grad_rho = 0.
                grad_mL_grad_rho = 0.
                for i in range(3):
                    drhodx = drhoKdrK * drKdx[i] + drhoMdrM * drMdx[i]
                    dmKdx = dmKdr * drKdx[i] + AnlK * dYlmKs[i]
                    dmLdx = dmLdr * drKdx[i] + AnlL * dYlmLs[i]
                    grad_mK_grad_rho += dmKdx * drhodx
                    grad_mL_grad_rho += dmLdx * drhodx

                if spin:
                    kKM = (outKM['v2rho2_up'] - outKM['v2rho2_updown']) * mK * mL
                    kKM += 2 * (outKM['v2rhosigma_up_up'] - outKM['v2rhosigma_up_down']) \
                           * grad_mK_grad_rho * mL
                    kKM += 2 * (outKM['v2rhosigma_up_up'] - outKM['v2rhosigma_up_down']) \
                           * mK * grad_mL_grad_rho
                    kKM += 4 * (outKM['v2sigma2_up_up'] - outKM['v2sigma2_up_down']) \
                           * grad_mK_grad_rho * grad_mL_grad_rho
                    kKM += 2 * (outKM['v2sigma2_up_updown'] - outKM['v2sigma2_updown_down']) \
                           * grad_mK_grad_rho * grad_mL_grad_rho
                    kKM += (2 * outKM['vsigma_up'] - outKM['vsigma_updown']) \
                           * grad_mK_grad_mL
                    kKM /= 2
                else:
                    kKM = outKM['v2rho2'] * mK * mL
                    kKM += 2 * outKM['v2rhosigma'] * grad_mK_grad_rho * mL
                    kKM += 2 * outKM['v2rhosigma'] * mK * grad_mL_grad_rho
                    kKM += 4 * outKM['v2sigma2'] * grad_mK_grad_rho \
                           * grad_mL_grad_rho
                    kKM += 2 * outKM['vsigma'] * grad_mK_grad_mL

                k += kKM - kK
            return k

        atoms_1c = [self.atoms_becke[self.iA]]
        Kxcab1c = becke.integral(atoms_1c, kxc_on1c)
        Kxcab2c = becke.integral(self.atoms_becke, kxc_on2c)
        return (Kxcab1c, Kxcab2c)

    def calculate_offsite_kernel_approximations(self, spin):
        assert self.iK != self.iL

        results = {}

        if self.xc.add_gradient_corrections:
            KxcKL2c = self.calculate_offsite_xc_kernel_gga(spin)
        else:
            KxcKL2c = self.calculate_offsite_xc_kernel_lda(spin)
        print("<a|fxc2c|b> =", KxcKL2c)

        results['K_2c'] = KxcKL2c
        return results

    def calculate_offsite_xc_kernel_lda(self, spin):
        Kxc2c = lambda x, y, z: self.chiK(x, y, z) \
                                * self.fxc_off2c(x, y, z, spin) \
                                * self.chiL(x, y, z)
        KxcKL2c = becke.integral(self.atoms_becke, Kxc2c)
        return KxcKL2c

    def calculate_offsite_xc_kernel_gga(self, spin):
        def kxc_off2c(x, y, z):
            dxK, dyK, dzK = x - self.xK, y - self.yK, z - self.zK
            dxL, dyL, dzL = x - self.xL, y - self.yL, z - self.zL
            rK = np.sqrt(dxK**2 + dyK**2 + dzK**2)
            rL = np.sqrt(dxL**2 + dyL**2 + dzL**2)
            drKdx = [dxK/rK, dyK/rK, dzK/rK]
            drLdx = [dxL/rL, dyL/rL, dzL/rL]

            rhoK = self.elK.electron_density(rK)
            rhoL = self.elL.electron_density(rL)
            drhoKdrK = self.elK.electron_density(rK, der=1)
            drhoLdrL = self.elL.electron_density(rL, der=1)
            drho = self.get_rho_deriv1(x, y, z, [self.iK, self.iL])

            if spin:
                rhoK /= 2
                rhoL /= 2
                drhoKdrK /= 2
                drhoLdrL /= 2
                drho = dict(sigma_up=drho['sigma'] / 4,
                            sigma_updown=drho['sigma'] / 4,
                            sigma_down=drho['sigma'] / 4)
                outKL = self.xc_polarized.compute_vxc_polarized(rhoK+rhoL,
                                            rhoK+rhoL, fxc=True, **drho)
            else:
                outKL = self.xc.compute_vxc(rhoK+rhoL, fxc=True, **drho)

            AnlK = self.AnlK(rK, der=0)
            AnlL = self.AnlL(rL, der=0)
            mK = self.chiK(x, y, z)
            mL = self.chiL(x, y, z)
            dmKdr = self.AnlK(rK, der=1) \
                    * sph_cartesian(dxK, dyK, dzK, rK, self.lmK)
            dmLdr = self.AnlL(rL, der=1) \
                    * sph_cartesian(dxL, dyL, dzL, rL, self.lmL)

            grad_mK_grad_mL = 0.
            grad_mK_grad_rho = 0.
            grad_mL_grad_rho = 0.
            for i in range(3):
                der = 'xyz'[i]
                dYlmK = sph_cartesian_der(dxK, dyK, dzK, rK, self.lmK, der=der)
                dYlmL = sph_cartesian_der(dxL, dyL, dzL, rL, self.lmL, der=der)
                dmKdx = dmKdr * drKdx[i] + AnlK * dYlmK
                dmLdx = dmLdr * drLdx[i] + AnlL * dYlmL
                grad_mK_grad_mL += dmKdx * dmLdx
                drhodx = drhoKdrK * drKdx[i] + drhoLdrL * drLdx[i]
                grad_mK_grad_rho += dmKdx * drhodx
                grad_mL_grad_rho += dmLdx * drhodx

            if spin:
                k = (outKL['v2rho2_up'] - outKL['v2rho2_updown']) * mK * mL
                k += 2 * (outKL['v2rhosigma_up_up'] - outKL['v2rhosigma_up_down']) \
                     * grad_mK_grad_rho * mL
                k += 2 * (outKL['v2rhosigma_up_up'] - outKL['v2rhosigma_up_down']) \
                     * mK * grad_mL_grad_rho
                k += 4 * (outKL['v2sigma2_up_up'] - outKL['v2sigma2_up_down']) \
                     * grad_mK_grad_rho * grad_mL_grad_rho
                k += 2 * (outKL['v2sigma2_up_updown'] - outKL['v2sigma2_updown_down']) \
                     * grad_mK_grad_rho * grad_mL_grad_rho
                k += (2 * outKL['vsigma_up'] - outKL['vsigma_updown']) \
                     * grad_mK_grad_mL
                k /= 2
            else:
                k = outKL['v2rho2'] * mK * mL
                k += 2 * outKL['v2rhosigma'] * grad_mK_grad_rho * mL
                k += 2 * outKL['v2rhosigma'] * mK * grad_mL_grad_rho
                k += 4 * outKL['v2sigma2'] * grad_mK_grad_rho * grad_mL_grad_rho
                k += 2 * outKL['vsigma'] * grad_mK_grad_mL
            return k

        Kxcab2c = becke.integral(self.atoms_becke, kxc_off2c)
        return Kxcab2c

    def calculate_multicenter_kernel(self, spin):
        results = {}

        Kxcmc = lambda x, y, z: self.chiK(x, y, z) \
                                * self.fxc_mc(x, y, z, spin) \
                                * self.chiL(x, y, z)
        KxcKLmc = becke.integral(self.atoms_becke, Kxcmc)
        print("<a|fxcmc|b> =", KxcKLmc)
        results['K_mc'] = KxcKLmc
        return results

    def solid_harmonic(self, x, y, z, l, lm):
        # Note: located at the origin
        r = np.sqrt(x**2 + y**2 + z**2)
        return np.sqrt(4 * np.pi / (2*l + 1)) * r**l \
               * sph_cartesian(x, y, z, r, lm)

    def chiK(self, x, y, z):
        dx, dy, dz = x - self.xK, y - self.yK, z - self.zK
        rK = np.sqrt(dx**2 + dy**2 + dz**2)
        lK = ANGULAR_MOMENTUM[self.lmK[0]]
        return self.elK.aux_basis(rK, self.nlK, lK) \
               * sph_cartesian(dx, dy, dz, rK, self.lmK)

    def chiL(self, x, y, z):
        dx, dy, dz = x - self.xL, y - self.yL, z - self.zL
        rL = np.sqrt(dx**2 + dy**2 + dz**2)
        lL = ANGULAR_MOMENTUM[self.lmL[0]]
        return self.elL.aux_basis(rL, self.nlL, lL) \
               * sph_cartesian(dx, dy, dz, rL, self.lmL)

    def AnlK(self, r, der=0):
        lK = ANGULAR_MOMENTUM[self.lmK[0]]
        return self.elK.aux_basis(r, self.nlK, lK, der=der)

    def AnlL(self, r, der=0):
        lL = ANGULAR_MOMENTUM[self.lmL[0]]
        return self.elL.aux_basis(r, self.nlL, lL, der=der)

    def get_orbital_vharK(self):
        vharK = becke.poisson(self.atoms_becke, self.chiK)
        return vharK

    def evaluate_KharKL(self, vharK):
        atoms_L = [self.atoms_becke[self.iL]]

        def integrand(x, y, z):
            return vharK(x, y, z) * self.chiL(x, y, z)

        KharKL = becke.integral(atoms_L, integrand)
        return KharKL

    def calculate_mapping_coefficients(self, atoms_ase, lmax={}, nzeta={},
                                       print_matrices=True,
                                       constraint_method='original'):
        self.atoms_ase = atoms_ase
        self.atoms_becke = ase2becke(atoms_ase)
        assert constraint_method in ['original', 'reduced'], constraint_method
        indices = list(range(len(self.atoms_ase)))

        # Main basis indices/orbitals/...
        iPs, nlPs, lmPs = [], [], []
        for index in indices:
            nls, lms = self.get_valence_orbitals(index)
            iPs.extend([index] * len(lms))
            nlPs.extend(nls)
            lmPs.extend(lms)

        def get_moments(lmax1, lmax2=None):
            lmax12 = lmax1 if lmax2 is None else max(lmax1, lmax2)
            lMs, lmMs = [], []

            for lM in range(lmax12+1):
                for lmM in ORBITALS[lM]:
                    lMs.append(lM)
                    lmMs.append(lmM)
            assert len(lMs) == len(lmMs) == (lmax12 + 1)**2

            if (constraint_method == 'original') and (lmax2 is not None):
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
                        lMs.append(lM)
                        lmMs.append(lmM)
                assert len(lMs) == len(lmMs) == (lmax12 + 1)**2 + 2*lmin12 + 1

            return (lMs, lmMs)

        # Auxiliary basis indices/orbitals/...
        iXs, nlXs, lmXs = [], [], []
        Nchi_dict = {}  # number of aux basis func for each element

        for iA in indices:
            symA = self.get_symbol(iA)
            elA = self.elements[symA]

            if symA not in Nchi_dict:
                Nchi_dict[symA] = len(self.get_multipoles(lmax[symA])) \
                                  * nzeta[symA]

            for izeta in range(nzeta[symA]):
                lms = self.get_multipoles(lmax[symA])
                lmXs.extend(lms)
                iXs.extend([iA] * len(lms))
                nl = elA.aux_basis.get_radial_label(izeta)
                nlXs.extend([nl] * len(lms))

        Nchi_max = max(Nchi_dict.values())

        # Calculate mapping coefficients
        # First the 'eta' matrix, which is the same for all AO products;
        # Also storing the needed Hartree potentials and auxiliary basis
        # function moments
        eta_dict = {}
        vhar_dict = {}
        D_dict = {}

        for indexK, (iK, nlK, lmK) in enumerate(zip(iXs, nlXs, lmXs)):
            self.symK = self.get_symbol(iK)
            self.nlK = nlK
            self.lmK = lmK
            self.xK, self.yK, self.zK = self.get_position(iK)
            lK = ANGULAR_MOMENTUM[lmK[0]]
            self.elK = self.elements[self.symK]

            keyK = (iK, nlK, lmK)
            vhar_dict[keyK] = OrbitalHartreePotential(
                                    self.elK.rgrid,
                                    self.elK.aux_basis.Anlg[(nlK, lK)],
                                    lmax=lmax[self.symK]+1)

            lMs, lmMs = get_moments(max(lmax.values()),
                                    lmax2=max(lmax.values()))
            for lM, lmM in zip(lMs, lmMs):
                def moment_chiK(x, y, z):
                    return self.chiK(x, y, z) \
                           * self.solid_harmonic(x, y, z, lM, lmM)
                keyM = (iK, nlK, lmK, lM, lmM)
                D_dict[keyM] = becke.integral([self.atoms_becke[iK]],
                                              moment_chiK)

            for indexL, (iL, nlL, lmL) in enumerate(zip(iXs, nlXs, lmXs)):
                self.symL = self.get_symbol(iL)
                self.nlL = nlL
                self.lmL = lmL
                self.xL, self.yL, self.zL = self.get_position(iL)
                self.elL = self.elements[self.symL]

                def Khar(x, y, z):
                    dx = x - self.xK
                    dy = y - self.yK
                    dz = z - self.zK
                    r = np.sqrt(dx**2 + dy**2 + dz**2)
                    return vhar_dict[keyK].vhar_fct[lK](r) \
                           * sph_cartesian(dx, dy, dz, r, self.lmK) \
                           * self.chiL(x, y, z)

                print('\nRunning iK %d iL %d nlmK %s %s nlmL %s %s' % \
                      (iK, iL, nlK, lmK, nlL, lmL), flush=True)

                atoms_har = [self.atoms_becke[iK]]
                if iK != iL:
                    atoms_har += [self.atoms_becke[iL]]
                keyKL = (iK, nlK, lmK, iL, nlL, lmL)
                eta_dict[keyKL] = becke.integral(atoms_har, Khar)
                print('eta:', eta_dict[keyKL])

        # Now looping over the AO products to get the M1 and M2 matrices
        M1 = np.zeros((len(lmPs), len(lmPs), Nchi_max))
        M2 = np.zeros_like(M1)

        for indexA, (iA, nlA, lmA) in enumerate(zip(iPs, nlPs, lmPs)):
            symA = self.get_symbol(iA)
            self.elA = self.elements[symA]
            self.nlA = nlA
            self.lmA = lmA
            nlmA = nlA[0] + lmA + nlA[2:]
            self.xA, self.yA, self.zA = self.get_position(iA)
            Nchi_A = Nchi_dict[symA]

            for indexB, (iB, nlB, lmB) in enumerate(zip(iPs, nlPs, lmPs)):
                symB = self.get_symbol(iB)
                self.elB = self.elements[symB]
                self.nlB = nlB
                self.lmB = lmB
                nlmB = nlB[0] + lmB + nlB[2:]
                self.xB, self.yB, self.zB = self.get_position(iB)
                Nchi_B = Nchi_dict[symB]

                print('\nRunning iA %d iB %d nlmA %s nlmB %s' % \
                      (iA, iB, nlmA, nlmB), flush=True)

                is_onsite = iA == iB
                if is_onsite:
                    Nchi = Nchi_A
                    lMs, lmMs = get_moments(lmax[symA], lmax2=None)
                else:
                    Nchi = Nchi_A + Nchi_B
                    lMs, lmMs = get_moments(lmax[symA], lmax2=lmax[symB])
                Nmp = len(lmMs)

                # d vector
                d = np.zeros(Nmp)
                for indexM, (lM, lmM) in enumerate(zip(lMs, lmMs)):
                    def moment_AOprod(x, y, z):
                        return self.phiA(x, y, z) * self.phiB(x, y, z) \
                               * self.solid_harmonic(x, y, z, lM, lmM)

                    atoms_2c = [self.atoms_becke[iA]]
                    if not is_onsite:
                        atoms_2c += [self.atoms_becke[iB]]
                    d[indexM] = becke.integral(atoms_2c, moment_AOprod)

                assert indexM == Nmp-1, (indexM, Nmp-1)
                print('d:', d, d.shape, flush=True)

                # g vector
                g = np.zeros(Nchi)
                ichi_dict = {iA: 0} if is_onsite else {iA: 0, iB: Nchi_A}

                for iK, nlK, lmK in zip(iXs, nlXs, lmXs):
                    if iK not in [iA, iB]:
                        continue

                    keyK = (iK, nlK, lmK)
                    xK, yK, zK = self.get_position(iK)
                    lK = ANGULAR_MOMENTUM[lmK[0]]

                    def KharAB(x, y, z):
                        dx = x - xK
                        dy = y - yK
                        dz = z - zK
                        r = np.sqrt(dx**2 + dy**2 + dz**2)
                        return self.phiA(x, y, z) * self.phiB(x, y, z) \
                               * vhar_dict[keyK].vhar_fct[lK](r) \
                               * sph_cartesian(dx, dy, dz, r, lmK)

                    atoms_har = [self.atoms_becke[iA]]
                    if not is_onsite:
                        atoms_har += [self.atoms_becke[iB]]
                    g[ichi_dict[iK]] = becke.integral(atoms_har, KharAB)
                    ichi_dict[iK] += 1

                assert ichi_dict[iA] == Nchi_A, (ichi_dict, Nchi_A)
                if not is_onsite:
                    assert ichi_dict[iB] == Nchi, (ichi_dict, Nchi)
                print('g:', g, g.shape, flush=True)

                # D matrix
                D = np.zeros((Nchi, Nmp))
                for indexM, (lM, lmM) in enumerate(zip(lMs, lmMs)):
                    ichi_dict = {iA: 0} if is_onsite else {iA: 0, iB: Nchi_A}

                    for iK, nlK, lmK in zip(iXs, nlXs, lmXs):
                        if iK not in [iA, iB]:
                            continue

                        keyM = (iK, nlK, lmK, lM, lmM)
                        D[ichi_dict[iK], indexM] = D_dict[keyM]
                        ichi_dict[iK] += 1

                    assert ichi_dict[iA] == Nchi_A, (ichi_dict, Nchi_A)
                    if not is_onsite:
                        assert ichi_dict[iB] == Nchi, (ichi_dict, Nchi)

                assert indexM == Nmp-1, (indexM, Nmp-1)
                print('D:', D, D.shape, flush=True)

                # (inv)eta matrices
                eta = np.zeros((Nchi, Nchi))
                ichi_dict = {iA: 0} if is_onsite else {iA: 0, iB: Nchi_A}

                for iK, nlK, lmK in zip(iXs, nlXs, lmXs):
                    if iK not in [iA, iB]:
                        continue

                    ichi_dict2 = {iA: 0} if is_onsite else {iA: 0, iB: Nchi_A}

                    for iL, nlL, lmL in zip(iXs, nlXs, lmXs):
                        if iL not in [iA, iB]:
                            continue
                        keyKL = (iK, nlK, lmK, iL, nlL, lmL)

                        eta[ichi_dict[iK], ichi_dict2[iL]] = eta_dict[keyKL]
                        ichi_dict2[iL] += 1

                    assert ichi_dict2[iA] == Nchi_A, (ichi_dict2, Nchi_A)
                    if not is_onsite:
                        assert ichi_dict2[iB] == Nchi, (ichi_dict2, Nchi)

                    ichi_dict[iK] += 1

                assert ichi_dict[iA] == Nchi_A, (ichi_dict, Nchi_A)
                if not is_onsite:
                    assert ichi_dict[iB] == Nchi, (ichi_dict, Nchi)
                print('eta:', eta, eta.shape, flush=True)
                inveta = np.linalg.inv(eta)

                # u vector
                u1 = np.linalg.inv(np.matmul(D.T, np.matmul(inveta, D)))
                u2 = np.matmul(D.T, np.matmul(inveta, g)) - d
                u = np.matmul(u1, u2)

                # M vector
                # For a given (ij), M should be a vector, containing
                # first the NauxA elements for Mijk^(1) and then the NauxB
                # elements for Mijk^(2).
                M = np.matmul(inveta, (g - np.matmul(D, u)))
                assert np.shape(M) == (Nchi,), (np.shape(M), Nchi)
                print('M:', M, flush=True)

                if is_onsite:
                    M1[indexA, indexB, :Nchi] = M[:]
                else:
                    M1[indexA, indexB, :Nchi_A] = M[:Nchi_A]
                    M2[indexA, indexB, :Nchi_B] = M[Nchi_A:]

        if print_matrices:
            def print_matrix(M, nperline=4):
                print('np.array([')
                for row in M:
                    print('[', end='')
                    i = 0
                    while i < len(row):
                        part = row[i:min(i+nperline, len(row))]
                        items = list(map(lambda x: '%.6e' % x, part))
                        print(', '.join(items), end='')
                        i += nperline
                        if i >= len(row):
                            print('],')
                        else:
                            print(',')
                print('])\n')

            template = '=== Mapping coefficient matrix {0} (ichi={1}) ==='
            for name, M in zip(['M1', 'M2'], [M1, M2]):
                np.save('{0}.npy'.format(name), M)

                for ichi in range(Nchi_max):
                    print(template.format(name, ichi))
                    print_matrix(M[:, :, ichi], nperline=4)

        return (M1, M2)

    def analyze_density_matrix(self, atoms_ase, density_matrices, M1, M2,
                               lmax={}, nzeta={}, h=0.1, vacuum=4, name=None):
        self.atoms_ase = atoms_ase
        self.atoms_ase.center(vacuum=vacuum)
        self.atoms_becke = ase2becke(atoms_ase)
        indices = list(range(len(self.atoms_ase)))

        nspin = len(density_matrices)
        for ispin in range(nspin):
            assert np.allclose(density_matrices[ispin],
                               density_matrices[ispin].T), \
                   'Density matrices needs to be symmetric'

        assert nspin in [1, 2], nspin
        spin_polarized = nspin == 2

        # Main basis indices/orbitals/...
        iAs, nlAs, lmAs = [], [], []
        for index in indices:
            nls, lms = self.get_valence_orbitals(index)
            iAs.extend([index] * len(lms))
            nlAs.extend(nls)
            lmAs.extend(lms)

        # Auxiliary basis indices/orbitals/...
        iXs, nlXs, lmXs = [], [], []
        Nchi_dict = {}  # number of aux basis func for each element

        for iA in indices:
            symA = self.get_symbol(iA)
            elA = self.elements[symA]

            if symA not in Nchi_dict:
                Nchi_dict[symA] = len(self.get_multipoles(lmax[symA])) \
                                  * nzeta[symA]

            for izeta in range(nzeta[symA]):
                lms = self.get_multipoles(lmax[symA])
                lmXs.extend(lms)
                iXs.extend([iA] * len(lms))
                nl = elA.aux_basis.get_radial_label(izeta)
                nlXs.extend([nl] * len(lms))

        # Grid generation
        # Note: only correct for orthogonal cell vectors
        print('Generating grid ...', flush=True)
        cell = self.atoms_ase.get_cell()
        self.gpts = tuple(h2gpts(h, cell))
        cell /= Bohr
        print('gpts:', self.gpts)

        x = np.linspace(0., 1., num=self.gpts[0], endpoint=False)
        y = np.linspace(0., 1., num=self.gpts[1], endpoint=False)
        z = np.linspace(0., 1., num=self.gpts[2], endpoint=False)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        XYZ = np.array([X, Y, Z])
        xyz = np.matmul(XYZ.T, cell)
        self.x = xyz[:, :, :, 0]
        self.y = xyz[:, :, :, 1]
        self.z = xyz[:, :, :, 2]
        self.Ngpts = np.prod(self.gpts)
        self.dV = abs(np.linalg.det(cell)) / self.Ngpts
        print('dV:', self.dV)

        self.dx = (x[1] - x[0]) * cell[0, 0]
        self.dy = (y[1] - y[0]) * cell[1, 1]
        self.dz = (z[1] - z[0]) * cell[2, 2]
        print('dx dy dz:', self.dx, self.dy, self.dz)
        assert np.isclose(self.dV, self.dx * self.dy * self.dz)

        # Sum of atomic densities
        rho0 = np.zeros_like(self.x)
        rho0_separate = [np.zeros_like(self.x) for iA in indices]

        for iA in indices:
            xA, yA, zA = self.get_position(iA)
            dxA = self.x - xA
            dyA = self.y - yA
            dzA = self.z - zA
            rA = np.sqrt(dxA**2 + dyA**2 + dzA**2)
            symA = self.get_symbol(iA)
            elA = self.elements[symA]
            rho0_separate[iA] = elA.electron_density(rA, only_valence=True)
            rho0 += rho0_separate[iA]

        total0 = self.dV * np.sum(rho0)
        print('Integrated initial density:', total0, flush=True)

        # Building drho from densmat
        rho = [np.zeros_like(self.x) for i in range(nspin)]

        for indexA, (iA, nlA, lmA) in enumerate(zip(iAs, nlAs, lmAs)):
            xA, yA, zA = self.get_position(iA)
            dxA = self.x - xA
            dyA = self.y - yA
            dzA = self.z - zA
            rA = np.sqrt(dxA**2 + dyA**2 + dzA**2)
            symA = self.get_symbol(iA)
            elA = self.elements[symA]

            for indexB, (iB, nlB, lmB) in enumerate(zip(iAs, nlAs, lmAs)):
                xB, yB, zB = self.get_position(iB)
                dxB = self.x - xB
                dyB = self.y - yB
                dzB = self.z - zB
                rB = np.sqrt(dxB**2 + dyB**2 + dzB**2)
                symB = self.get_symbol(iB)
                elB = self.elements[symB]

                overlap = elA.Rnl(rA, nl=nlA) \
                        * sph_cartesian(dxA, dyA, dzA, rA, lmA) \
                        * elB.Rnl(rB, nl=nlB) \
                        * sph_cartesian(dxB, dyB, dzB, rB, lmB)
                assert not np.any(np.isnan(overlap)), \
                       [(iA, nlA, lmA), (iB, nlB, lmB)]

                for ispin in range(nspin):
                    coeff = density_matrices[ispin][indexA, indexB]
                    rho[ispin] += coeff * overlap

        if spin_polarized:
            drho = rho[0] - rho[1]
        else:
            drho = rho[0] - rho0

        totals = [self.dV * np.sum(rho[ispin]) for ispin in range(nspin)]
        print('Integrated densities:', *totals)

        total = self.dV * np.sum(drho)
        print('Integrated difference density:', total)

        if name is not None:
            filename = 'out_{0}_densmat.cube'.format(name)
            with open(filename, 'w') as f:
                write_cube(f, self.atoms_ase, data=drho.T, origin=None,
                           comment=None)

        # Building drho from multipoles

        # Reference density matrix is either the atomic one or
        # the down-channel one
        if spin_polarized:
            densmat_ref = np.copy(density_matrices[1])
        else:
            densmat_ref = np.zeros_like(density_matrices[0])
            for indexA, (iA, nlA, lmA) in enumerate(zip(iAs, nlAs, lmAs)):
                symA = self.get_symbol(iA)
                elA = self.elements[symA]

                for indexB, (iB, nlB, lmB) in enumerate(zip(iAs, nlAs, lmAs)):
                    if (indexA == indexB) and (nlA in elA.valence):
                        occup = elA.configuration[nlA]
                        l = ANGULAR_MOMENTUM[nlA[1]]
                        numlm = (l + 1)**2 - l**2
                        densmat_ref[indexA, indexA] = occup * 1. / numlm

        # Multipole moments m and their drho contributions
        m = np.zeros(len(lmXs))
        drho = 0.

        for indexA, (iA, nlA, lmA) in enumerate(zip(iAs, nlAs, lmAs)):
            for indexB, (iB, nlB, lmB) in enumerate(zip(iAs, nlAs, lmAs)):
                coeff = density_matrices[0][indexA, indexB]
                coeff -= densmat_ref[indexA, indexB]
                term1 = coeff * M1[indexA, indexB, :]
                term2 = coeff * M2[indexA, indexB, :]

                is_onsite = iA == iB
                ichi_dict = {iA: 0} if is_onsite else {iA: 0, iB: 0}

                for indexK, (iK, nlK, lmK) in enumerate(zip(iXs, nlXs, lmXs)):
                    if is_onsite:
                        if iA == iK:
                            m[indexK] += term1[ichi_dict[iK]]
                            ichi_dict[iK] += 1
                    else:
                        if iA == iK:
                            m[indexK] += term1[ichi_dict[iK]]
                            ichi_dict[iK] += 1
                        elif iB == iK:
                            m[indexK] += term2[ichi_dict[iK]]
                            ichi_dict[iK] += 1

        for indexK, (iK, nlK, lmK) in enumerate(zip(iXs, nlXs, lmXs)):
            self.xK, self.yK, self.zK = self.get_position(iK)
            self.symK = self.get_symbol(iK)
            self.nlK = nlK
            self.lmK = lmK
            self.elK = self.elements[self.symK]
            drho += m[indexK] * self.chiK(self.x, self.y, self.z)

        m_str = np.array2string(m, separator=', ', max_line_width=72)
        print('m:', m_str)

        total = self.dV * np.sum(drho)
        print('Integrated difference density:', total)

        if name is not None:
            filename = 'out_{0}_multipole.cube'.format(name)
            with open(filename, 'w') as f:
                write_cube(f, self.atoms_ase, data=drho.T, origin=None,
                           comment=None)
        return
