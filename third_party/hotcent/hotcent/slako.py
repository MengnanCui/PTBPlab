#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
"""Utilities related to Slater-Koster integrals."""
import numpy as np


INTEGRALS = ['ffs', 'ffp', 'ffd', 'fff',
             'dfs', 'dfp', 'dfd', 'dds', 'ddp', 'ddd',
             'pfs', 'pfp', 'pds', 'pdp', 'pps', 'ppp',
             'sfs', 'sds', 'sps', 'sss']

NUMSK = len(INTEGRALS)


INTEGRAL_PAIRS = {
    'sss': ('s', 's'),
    'sps': ('s', 'pz'),
    'sds': ('s', 'dz2'),
    'sfs': ('s', 'fz3'),
    'sgs': ('s', 'g5'),
    'pps': ('pz', 'pz'),
    'ppp': ('px', 'px'),
    'pds': ('pz', 'dz2'),
    'pdp': ('px', 'dxz'),
    'pgs': ('pz', 'g5'),
    'pgp': ('px', 'g6'),
    'pfs': ('pz', 'fz3'),
    'pfp': ('px', 'fxz2'),
    'dds': ('dz2', 'dz2'),
    'ddp': ('dxz', 'dxz'),
    'ddd': ('dxy', 'dxy'),
    'dfs': ('dz2', 'fz3'),
    'dfp': ('dxz', 'fxz2'),
    'dfd': ('dxy', 'fxyz'),
    'dgs': ('dz2', 'g5'),
    'dgp': ('dxz', 'g6'),
    'dgd': ('dxy', 'g3'),
    'ffs': ('fz3', 'fz3'),
    'ffp': ('fxz2', 'fxz2'),
    'ffd': ('fxyz', 'fxyz'),
    'fff': ('fx(x2-3y2)', 'fx(x2-3y2)'),
    'fgs': ('fz3', 'g5'),
    'fgp': ('fxz2', 'g6'),
    'fgd': ('fxyz', 'g3'),
    'fgf': ('fx(x2-3y2)', 'g8'),
    'ggs': ('g5', 'g5'),
    'ggp': ('g6', 'g6'),
    'ggd': ('g3', 'g3'),
    'ggf': ('g8', 'g8'),
    'ggg': ('g1', 'g1'),
}


def get_integral_pair(integral):
    """ Returns the orbital pair used for calculating
    the given Slater-Koster integral. """
    if integral in INTEGRAL_PAIRS:
        lm1, lm2 = INTEGRAL_PAIRS[integral]
    else:
        lm2, lm1 = INTEGRAL_PAIRS[integral[1] + integral[0] + integral[2]]
    return (lm1, lm2)


def search_integrals(lm1, lm2):
    """ Returns the sigma/pi/... integrals to be considered for the
    given pair of orbitals. """
    integrals, ordered = [], []

    for integral, pair in INTEGRAL_PAIRS.items():
        if pair == (lm1, lm2):
            integrals.append(integral)
            ordered.append(True)
        elif pair == (lm2, lm1):
            integrals.append(integral)
            ordered.append(False)

    return integrals, ordered


def select_subshells(val1, val2, integral):
    """
    Select subshells from given valence sets to calculate given
    Slater-Koster integral.

    Parameters
    ----------
    val1, val2 : list of str
        Valence subshell sets (e.g. ['2s', '2p'], ['4s', '3d']).
    integral : str
        Slater-Koster integral label (e.g. 'sds').

    Returns
    -------
    nl1, nl2 : str
        Matching subshell pair (e.g. ('2s', '3d') in this example).
    """
    nl1 = None
    for nl in val1:
        if nl[1] == integral[0]:
            nl1 = nl

    nl2 = None
    for nl in val2:
        if nl[1] == integral[1]:
            nl2 = nl

    return nl1, nl2


def select_integrals(e1, e2):
    """ Return list of integrals (integral, nl1, nl2)
    to be done for element pair e1, e2. """
    selected = []
    for ival1, valence1 in enumerate(e1.basis_sets):
        for ival2, valence2 in enumerate(e2.basis_sets):
            for integral in INTEGRALS:
                nl1, nl2 = select_subshells(valence1, valence2, integral)
                if nl1 is not None and nl2 is not None:
                    selected.append((integral, nl1, nl2))
    return selected


def print_integral_overview(e1, e2, selected, file):
    """ Prints an overview of the selected Slater-Koster integrals. """
    for bas1 in range(len(e1.basis_sets)):
        for bas2 in range(len(e2.basis_sets)):
            sym1 = e1.get_symbol() + '+'*bas1
            sym2 = e2.get_symbol() + '+'*bas2
            print('Integrals for %s-%s pair:' % (sym1, sym2), end=' ',
                  file=file)
            for integral, nl1, nl2 in selected:
                if e1.get_basis_set_index(nl1) == bas1 and \
                   e2.get_basis_set_index(nl2) == bas2:
                    print('_'.join([nl1, nl2, integral]), end=' ', file=file)
            print(file=file, flush=True)
    return


def g(c1, c2, s1, s2, integral):
    """ Returns the angle-dependent part of the given two-center integral,
    with c1 and s1 (c2 and s2) the sine and cosine of theta_1 (theta_2)
    for the atom at origin (atom at z=Rz). These expressions are obtained
    by integrating analytically over phi.
    """
    if integral == 'ffs':
        return 7. / 8 * c1 * (5 * c1**2 - 3) * c2 * (5 * c2**2 - 3)
    elif integral == 'ffp':
        return 21. / 32 * (5 * c1**2 - 1) * s1 * (5 * c2**2 - 1) * s2
    elif integral == 'ffd':
        return 105. / 16 * s1**2 * c1 * s2**2 * c2
    elif integral == 'fff':
        return 35. / 32 * s1**3 * s2**3
    elif integral == 'dfs':
        return np.sqrt(35.) / 8 * (3 * c1**2 - 1) * c2 * (5 * c2**2 - 3)
    elif integral == 'dfp':
        return 3. / 8 * np.sqrt(17.5) * s1 * c1 * s2 * (5 * c2**2 - 1)
    elif integral == 'dfd':
        return 15. / 16 * np.sqrt(7.) * s1**2 * s2**2 * c2
    elif integral == 'dds':
        return 5. / 8 * (3 * c1**2 - 1) * (3 * c2**2 - 1)
    elif integral == 'ddp':
        return 15. / 4 * s1 * c1 * s2 * c2
    elif integral == 'ddd':
        return 15. / 16 * s1**2 * s2**2
    elif integral == 'pfs':
        return np.sqrt(21.) / 4 * c1 * c2 * (5 * c2**2 - 3)
    elif integral == 'pfp':
        return 3. / 8 * np.sqrt(3.5) * s1 * s2 * (5 * c2**2 - 1)
    elif integral == 'pds':
        return np.sqrt(15.) / 4 * c1 * (3 * c2**2 - 1)
    elif integral == 'pdp':
        return np.sqrt(45.) / 4 * s1 * s2 * c2
    elif integral == 'pps':
        return 3. / 2 * c1 * c2
    elif integral == 'ppp':
        return 3. / 4 * s1 * s2
    elif integral == 'sfs':
        return np.sqrt(7.) / 4 * c2 * (5 * c2**2 - 3)
    elif integral == 'sds':
        return np.sqrt(5.) / 4 * (3 * c2**2 - 1)
    elif integral == 'sps':
        return np.sqrt(3.) / 2 * c2
    elif integral == 'sss':
        return 0.5 * np.ones_like(c1)
    elif integral == 'sgs':
        return 105*c2**4/16. - 45*c2**2/8 + 9/16.
    elif integral == 'pgs':
        return 3*np.sqrt(3)*(35*c2**4 - 30*c2**2 + 3)*c1/16.
    elif integral == 'pgp':
        return 3*np.sqrt(30)*(7*c2**2 - 3)*s2*s1*c2/16.
    elif integral == 'dgs':
        return 3*np.sqrt(5)*(3*c1**2 - 1)*(35*c2**4 - 30*c2**2 + 3)/32.
    elif integral == 'dgp':
        return 15*np.sqrt(6)*(28*(-8*s2**3*c2 + 4*s2*c2)*s1*c1 \
                              + 16*s2*s1*c2*c1)/512.
    elif integral == 'dgd':
        return 15*np.sqrt(3)*(7*c2**2 - 1)*s2**2*s1**2/32.
    elif integral == 'fgs':
        return 3*np.sqrt(7)*(5*c1**2 - 3)*(35*c2**4 - 30*c2**2 + 3)*c1/32.
    elif integral == 'fgp':
        return 3*np.sqrt(105)*(7*c2**2 - 3)*(5*c1**2 - 1)*s2*s1*c2/32.
    elif integral == 'fgd':
        return 15*np.sqrt(21)*(7*c2**2 - 1)*s2**2*s1**2*c1/32.
    elif integral == 'fgf':
        return 105*s2**3*s1**3*c2/32.
    elif integral == 'ggs':
        return 9*(35*c2**4 - 30*c2**2 + 3)*(35*c1**4 - 30*c1**2 + 3)/128.
    elif integral == 'ggp':
        return 45*(7*c2**2 - 3)*(7*c1**2 - 3)*s2*s1*c2*c1/32.
    elif integral == 'ggd':
        return 45*(7*c2**2 - 1)*(7*c1**2 - 1)*s2**2*s1**2/64.
    elif integral == 'ggf':
        return 315*s2**3*s1**3*c2*c1/32.
    elif integral == 'ggg':
        return 315*s2**4*s1**4/256.


def dg(c1, c2, s1, s2, integral):
    """ Returns an array with the c1, c2, s1 and s2 derivatives
    of g(c1, c2, s1, s2) for the given integral.
    """
    if integral == 'sss':
        return [0, 0, 0, 0]
    elif integral == 'sps':
        return [0, np.sqrt(3)/2, 0, 0]
    elif integral == 'sds':
        return [0, 3*np.sqrt(5)*c2/2, 0, 0]
    elif integral == 'sfs':
        return [0, 3*np.sqrt(7)*(5*c2**2 - 1)/4, 0, 0]
    elif integral == 'pps':
        return [3*c2/2, 3*c1/2, 0, 0]
    elif integral == 'ppp':
        return [0, 0, 3*s2/4, 3*s1/4]
    elif integral == 'pds':
        return [np.sqrt(15)*(3*c2**2 - 1)/4, 3*np.sqrt(15)*c1*c2/2, 0, 0]
    elif integral == 'pdp':
        return [0, 3*np.sqrt(5)*s1*s2/4, 3*np.sqrt(5)*c2*s2/4,
                3*np.sqrt(5)*c2*s1/4]
    elif integral == 'pfs':
        return [np.sqrt(21)*c2*(5*c2**2 - 3)/4,
                3*np.sqrt(21)*c1*(5*c2**2 - 1)/4, 0, 0]
    elif integral == 'pfp':
        return [0, 15*np.sqrt(14)*c2*s1*s2/8,
                3*np.sqrt(14)*s2*(5*c2**2 - 1)/16,
                3*np.sqrt(14)*s1*(5*c2**2 - 1)/16]
    elif integral == 'dds':
        return [15*c1*(3*c2**2 - 1)/4, 15*c2*(3*c1**2 - 1)/4, 0, 0]
    elif integral == 'ddp':
        return [15*c2*s1*s2/4, 15*c1*s1*s2/4, 15*c1*c2*s2/4, 15*c1*c2*s1/4]
    elif integral == 'ddd':
        return [0, 0, 15*s1*s2**2/8, 15*s1**2*s2/8]
    elif integral == 'dfs':
        return [3*np.sqrt(35)*c1*c2*(5*c2**2 - 3)/4,
                3*np.sqrt(35)*(3*c1**2 - 1)*(5*c2**2 - 1)/8, 0, 0]
    elif integral == 'dfp':
        return [3*np.sqrt(70)*s1*s2*(5*c2**2 - 1)/16,
                15*np.sqrt(70)*c1*c2*s1*s2/8,
                3*np.sqrt(70)*c1*s2*(5*c2**2 - 1)/16,
                3*np.sqrt(70)*c1*s1*(5*c2**2 - 1)/16]
    elif integral == 'dfd':
        return [0, 15*np.sqrt(7)*s1**2*s2**2/16, 15*np.sqrt(7)*c2*s1*s2**2/8,
                15*np.sqrt(7)*c2*s1**2*s2/8]
    elif integral == 'ffs':
        return [21*c2*(5*c1**2 - 1)*(5*c2**2 - 3)/8,
                21*c1*(5*c1**2 - 3)*(5*c2**2 - 1)/8, 0, 0]
    elif integral == 'ffp':
        return [105*c1*s1*s2*(5*c2**2 - 1)/16, 105*c2*s1*s2*(5*c1**2 - 1)/16,
                21*s2*(5*c1**2 - 1)*(5*c2**2 - 1)/32,
                21*s1*(5*c1**2 - 1)*(5*c2**2 - 1)/32]
    elif integral == 'ffd':
        return [105*c2*s1**2*s2**2/16, 105*c1*s1**2*s2**2/16,
                105*c1*c2*s1*s2**2/8, 105*c1*c2*s1**2*s2/8]
    elif integral == 'fff':
        return [0, 0, 105*s1**2*s2**3/32, 105*s1**3*s2**2/32]


def get_twocenter_phi_integral(lm1, lm2, c1, c2, s1, s2):
    r""" Similar to the g() function above, but for every
    (lm1, lm2) combination including s/p/d/f/g/h angular momenta.

    Parameters
    ----------
    lm1, lm2 : str
        Orbital labels.
    c1, c2, s1, s2 : np.ndarray
        Cosine (c1 and c2) and sine (s1 and s2) of the
        theta_1 and theta2 angles, respectively.

    Returns
    -------
    ghi : np.ndarray
        Y_{lm1}(\theta_1) Y_{lm1}(\theta_2)
        \int Y_{lm1}(\phi_1) Y_{lm1}(\phi_2) d\phi
    """
    if lm1 == 's' and lm2 == 's':
        gphi = 1/2.
    elif lm1 == 's' and lm2 == 'pz':
        gphi = np.sqrt(3)*c2/2.
    elif lm1 == 's' and lm2 == 'dz2':
        gphi = np.sqrt(5)*(3*c2**2 - 1)/4.
    elif lm1 == 's' and lm2 == 'fz3':
        gphi = np.sqrt(7)*(5*c2**2 - 3)*c2/4.
    elif lm1 == 's' and lm2 == 'g5':
        gphi = 105*c2**4/16. - 45*c2**2/8 + 9/16.
    elif lm1 == 's' and lm2 == 'h6':
        gphi = np.sqrt(11)*(63*c2**4 - 70*c2**2 + 15)*c2/16.
    elif lm1 == 'px' and lm2 == 'px':
        gphi = 3*s2*s1/4.
    elif lm1 == 'px' and lm2 == 'dxz':
        gphi = 3*np.sqrt(5)*s2*s1*c2/4.
    elif lm1 == 'px' and lm2 == 'fxz2':
        gphi = 3*np.sqrt(14)*(5*c2**2 - 1)*s2*s1/16.
    elif lm1 == 'px' and lm2 == 'g6':
        gphi = 3*np.sqrt(30)*(7*c2**2 - 3)*s2*s1*c2/16.
    elif lm1 == 'px' and lm2 == 'h7':
        gphi = 3*np.sqrt(55)*(21*c2**4 - 14*c2**2 + 1)*s2*s1/32.
    elif lm1 == 'py' and lm2 == 'py':
        gphi = 3*s2*s1/4.
    elif lm1 == 'py' and lm2 == 'dyz':
        gphi = 3*np.sqrt(5)*s2*s1*c2/4.
    elif lm1 == 'py' and lm2 == 'fyz2':
        gphi = 3*np.sqrt(14)*(5*c2**2 - 1)*s2*s1/16.
    elif lm1 == 'py' and lm2 == 'g4':
        gphi = 3*np.sqrt(30)*(7*c2**2 - 3)*s2*s1*c2/16.
    elif lm1 == 'py' and lm2 == 'h5':
        gphi = 3*np.sqrt(55)*(21*c2**4 - 14*c2**2 + 1)*s2*s1/32.
    elif lm1 == 'pz' and lm2 == 's':
        gphi = np.sqrt(3)*c1/2.
    elif lm1 == 'pz' and lm2 == 'pz':
        gphi = 3*c2*c1/2.
    elif lm1 == 'pz' and lm2 == 'dz2':
        gphi = np.sqrt(15)*(3*c2**2 - 1)*c1/4.
    elif lm1 == 'pz' and lm2 == 'fz3':
        gphi = np.sqrt(21)*(5*c2**2 - 3)*c2*c1/4.
    elif lm1 == 'pz' and lm2 == 'g5':
        gphi = 3*np.sqrt(3)*(35*c2**4 - 30*c2**2 + 3)*c1/16.
    elif lm1 == 'pz' and lm2 == 'h6':
        gphi = np.sqrt(33)*(63*c2**4 - 70*c2**2 + 15)*c2*c1/16.
    elif lm1 == 'dxy' and lm2 == 'dxy':
        gphi = 15*s2**2*s1**2/16.
    elif lm1 == 'dxy' and lm2 == 'fxyz':
        gphi = 15*np.sqrt(7)*s2**2*s1**2*c2/16.
    elif lm1 == 'dxy' and lm2 == 'g3':
        gphi = 15*np.sqrt(3)*(6 - 7*s2**2)*s2**2*s1**2/32.
    elif lm1 == 'dxy' and lm2 == 'h4':
        gphi = 15*np.sqrt(77)*(2 - 3*s2**2)*s2**2*s1**2*c2/32.
    elif lm1 == 'dyz' and lm2 == 'py':
        gphi = 3*np.sqrt(5)*s2*s1*c1/4.
    elif lm1 == 'dyz' and lm2 == 'dyz':
        gphi = 15*s2*s1*c2*c1/4.
    elif lm1 == 'dyz' and lm2 == 'fyz2':
        gphi = 3*np.sqrt(70)*(5*c2**2 - 1)*s2*s1*c1/16.
    elif lm1 == 'dyz' and lm2 == 'g4':
        gphi = 15*np.sqrt(6)*(7*c2**2 - 3)*s2*s1*c2*c1/16.
    elif lm1 == 'dyz' and lm2 == 'h5':
        gphi = 15*np.sqrt(11)*(21*c2**4 - 14*c2**2 + 1)*s2*s1*c1/32.
    elif lm1 == 'dxz' and lm2 == 'px':
        gphi = 3*np.sqrt(5)*s2*s1*c1/4.
    elif lm1 == 'dxz' and lm2 == 'dxz':
        gphi = 15*s2*s1*c2*c1/4.
    elif lm1 == 'dxz' and lm2 == 'fxz2':
        gphi = 3*np.sqrt(70)*(5*c2**2 - 1)*s2*s1*c1/16.
    elif lm1 == 'dxz' and lm2 == 'g6':
        gphi = 15*np.sqrt(6)*(7*c2**2 - 3)*s2*s1*c2*c1/16.
    elif lm1 == 'dxz' and lm2 == 'h7':
        gphi = 15*np.sqrt(11)*(21*c2**4 - 14*c2**2 + 1)*s2*s1*c1/32.
    elif lm1 == 'dx2-y2' and lm2 == 'dx2-y2':
        gphi = 15*s2**2*s1**2/16.
    elif lm1 == 'dx2-y2' and lm2 == 'fz(x2-y2)':
        gphi = 15*np.sqrt(7)*s2**2*s1**2*c2/16.
    elif lm1 == 'dx2-y2' and lm2 == 'g7':
        gphi = 15*np.sqrt(3)*(6 - 7*s2**2)*s2**2*s1**2/32.
    elif lm1 == 'dx2-y2' and lm2 == 'h8':
        gphi = 15*np.sqrt(77)*(2 - 3*s2**2)*s2**2*s1**2*c2/32.
    elif lm1 == 'dz2' and lm2 == 's':
        gphi = np.sqrt(5)*(3*c1**2 - 1)/4.
    elif lm1 == 'dz2' and lm2 == 'pz':
        gphi = np.sqrt(15)*(3*c1**2 - 1)*c2/4.
    elif lm1 == 'dz2' and lm2 == 'dz2':
        gphi = 5*(3*c2**2 - 1)*(3*c1**2 - 1)/8.
    elif lm1 == 'dz2' and lm2 == 'fz3':
        gphi = np.sqrt(35)*(5*c2**2 - 3)*(3*c1**2 - 1)*c2/8.
    elif lm1 == 'dz2' and lm2 == 'g5':
        gphi = 3*np.sqrt(5)*(3*c1**2 - 1)*(35*c2**4 - 30*c2**2 + 3)/32.
    elif lm1 == 'dz2' and lm2 == 'h6':
        gphi = np.sqrt(55)*(3*c1**2 - 1)*(63*c2**4 - 70*c2**2 + 15)*c2/32.
    elif lm1 == 'fx(x2-3y2)' and lm2 == 'fx(x2-3y2)':
        gphi = 35*s2**3*s1**3/32.
    elif lm1 == 'fx(x2-3y2)' and lm2 == 'g8':
        gphi = 105*s2**3*s1**3*c2/32.
    elif lm1 == 'fx(x2-3y2)' and lm2 == 'h9':
        gphi = 35*np.sqrt(11)*(9*c2**2 - 1)*s2**3*s1**3/128.
    elif lm1 == 'fy(3x2-y2)' and lm2 == 'fy(3x2-y2)':
        gphi = 35*s2**3*s1**3/32.
    elif lm1 == 'fy(3x2-y2)' and lm2 == 'g2':
        gphi = 105*s2**3*s1**3*c2/32.
    elif lm1 == 'fy(3x2-y2)' and lm2 == 'h3':
        gphi = 35*np.sqrt(11)*(9*c2**2 - 1)*s2**3*s1**3/128.
    elif lm1 == 'fz(x2-y2)' and lm2 == 'dx2-y2':
        gphi = 15*np.sqrt(7)*s2**2*s1**2*c1/16.
    elif lm1 == 'fz(x2-y2)' and lm2 == 'fz(x2-y2)':
        gphi = 105*s2**2*s1**2*c2*c1/16.
    elif lm1 == 'fz(x2-y2)' and lm2 == 'g7':
        gphi = 15*np.sqrt(21)*(6 - 7*s2**2)*s2**2*s1**2*c1/32.
    elif lm1 == 'fz(x2-y2)' and lm2 == 'h8':
        gphi = 105*np.sqrt(11)*(2 - 3*s2**2)*s2**2*s1**2*c2*c1/32.
    elif lm1 == 'fxyz' and lm2 == 'dxy':
        gphi = 15*np.sqrt(7)*s2**2*s1**2*c1/16.
    elif lm1 == 'fxyz' and lm2 == 'fxyz':
        gphi = 105*s2**2*s1**2*c2*c1/16.
    elif lm1 == 'fxyz' and lm2 == 'g3':
        gphi = 15*np.sqrt(21)*(6 - 7*s2**2)*s2**2*s1**2*c1/32.
    elif lm1 == 'fxyz' and lm2 == 'h4':
        gphi = 105*np.sqrt(11)*(2 - 3*s2**2)*s2**2*s1**2*c2*c1/32.
    elif lm1 == 'fyz2' and lm2 == 'py':
        gphi = 3*np.sqrt(14)*(5*c1**2 - 1)*s2*s1/16.
    elif lm1 == 'fyz2' and lm2 == 'dyz':
        gphi = 3*np.sqrt(70)*(5*c1**2 - 1)*s2*s1*c2/16.
    elif lm1 == 'fyz2' and lm2 == 'fyz2':
        gphi = 21*(5*c2**2 - 1)*(5*c1**2 - 1)*s2*s1/32.
    elif lm1 == 'fyz2' and lm2 == 'g4':
        gphi = 3*np.sqrt(105)*(7*c2**2 - 3)*(5*c1**2 - 1)*s2*s1*c2/32.
    elif lm1 == 'fyz2' and lm2 == 'h5':
        gphi = 3*np.sqrt(770)*(5*c1**2 - 1)*(21*c2**4 - 14*c2**2 + 1)*s2*s1/128.
    elif lm1 == 'fxz2' and lm2 == 'px':
        gphi = 3*np.sqrt(14)*(5*c1**2 - 1)*s2*s1/16.
    elif lm1 == 'fxz2' and lm2 == 'dxz':
        gphi = 3*np.sqrt(70)*(5*c1**2 - 1)*s2*s1*c2/16.
    elif lm1 == 'fxz2' and lm2 == 'fxz2':
        gphi = 21*(5*c2**2 - 1)*(5*c1**2 - 1)*s2*s1/32.
    elif lm1 == 'fxz2' and lm2 == 'g6':
        gphi = 3*np.sqrt(105)*(7*c2**2 - 3)*(5*c1**2 - 1)*s2*s1*c2/32.
    elif lm1 == 'fxz2' and lm2 == 'h7':
        gphi = 3*np.sqrt(770)*(5*c1**2 - 1)*(21*c2**4 - 14*c2**2 + 1)*s2*s1/128.
    elif lm1 == 'fz3' and lm2 == 's':
        gphi = np.sqrt(7)*(5*c1**2 - 3)*c1/4.
    elif lm1 == 'fz3' and lm2 == 'pz':
        gphi = np.sqrt(21)*(5*c1**2 - 3)*c2*c1/4.
    elif lm1 == 'fz3' and lm2 == 'dz2':
        gphi = np.sqrt(35)*(3*c2**2 - 1)*(5*c1**2 - 3)*c1/8.
    elif lm1 == 'fz3' and lm2 == 'fz3':
        gphi = 7*(5*c2**2 - 3)*(5*c1**2 - 3)*c2*c1/8.
    elif lm1 == 'fz3' and lm2 == 'g5':
        gphi = 3*np.sqrt(7)*(5*c1**2 - 3)*(35*c2**4 - 30*c2**2 + 3)*c1/32.
    elif lm1 == 'fz3' and lm2 == 'h6':
        gphi = np.sqrt(77)*(5*c1**2 - 3)*(63*c2**4 - 70*c2**2 + 15)*c2*c1/32.
    elif lm1 == 'g1' and lm2 == 'g1':
        gphi = 315*s2**4*s1**4/256.
    elif lm1 == 'g1' and lm2 == 'h2':
        gphi = 315*np.sqrt(11)*s2**4*s1**4*c2/256.
    elif lm1 == 'g2' and lm2 == 'fy(3x2-y2)':
        gphi = 105*s2**3*s1**3*c1/32.
    elif lm1 == 'g2' and lm2 == 'g2':
        gphi = 315*s2**3*s1**3*c2*c1/32.
    elif lm1 == 'g2' and lm2 == 'h3':
        gphi = 105*np.sqrt(11)*(9*c2**2 - 1)*s2**3*s1**3*c1/128.
    elif lm1 == 'g3' and lm2 == 'dxy':
        gphi = 15*np.sqrt(3)*(6 - 7*s1**2)*s2**2*s1**2/32.
    elif lm1 == 'g3' and lm2 == 'fxyz':
        gphi = 15*np.sqrt(21)*(6 - 7*s1**2)*s2**2*s1**2*c2/32.
    elif lm1 == 'g3' and lm2 == 'g3':
        gphi = 45*(7*s2**2 - 6)*(7*s1**2 - 6)*s2**2*s1**2/64.
    elif lm1 == 'g3' and lm2 == 'h4':
        gphi = 15*np.sqrt(231)*(3*s2**2 - 2)*(7*s1**2 - 6)*s2**2*s1**2*c2/64.
    elif lm1 == 'g4' and lm2 == 'py':
        gphi = 3*np.sqrt(30)*(7*c1**2 - 3)*s2*s1*c1/16.
    elif lm1 == 'g4' and lm2 == 'dyz':
        gphi = 15*np.sqrt(6)*(7*c1**2 - 3)*s2*s1*c2*c1/16.
    elif lm1 == 'g4' and lm2 == 'fyz2':
        gphi = 3*np.sqrt(105)*(5*c2**2 - 1)*(7*c1**2 - 3)*s2*s1*c1/32.
    elif lm1 == 'g4' and lm2 == 'g4':
        gphi = 45*(7*c2**2 - 3)*(7*c1**2 - 3)*s2*s1*c2*c1/32.
    elif lm1 == 'g4' and lm2 == 'h5':
        gphi = 15*np.sqrt(66)*(7*c1**2 - 3)*(21*c2**4 - 14*c2**2 + 1)*s2*s1*c1/128.
    elif lm1 == 'g5' and lm2 == 's':
        gphi = 105*c1**4/16. - 45*c1**2/8 + 9/16.
    elif lm1 == 'g5' and lm2 == 'pz':
        gphi = 3*np.sqrt(3)*(35*c1**4 - 30*c1**2 + 3)*c2/16.
    elif lm1 == 'g5' and lm2 == 'dz2':
        gphi = 3*np.sqrt(5)*(3*c2**2 - 1)*(35*c1**4 - 30*c1**2 + 3)/32.
    elif lm1 == 'g5' and lm2 == 'fz3':
        gphi = 3*np.sqrt(7)*(5*c2**2 - 3)*(35*c1**4 - 30*c1**2 + 3)*c2/32.
    elif lm1 == 'g5' and lm2 == 'g5':
        gphi = 9*(35*c2**4 - 30*c2**2 + 3)*(35*c1**4 - 30*c1**2 + 3)/128.
    elif lm1 == 'g5' and lm2 == 'h6':
        gphi = 3*np.sqrt(11)*(63*c2**4 - 70*c2**2 + 15)*(35*c1**4 - 30*c1**2 + 3)*c2/128.
    elif lm1 == 'g6' and lm2 == 'px':
        gphi = 3*np.sqrt(30)*(7*c1**2 - 3)*s2*s1*c1/16.
    elif lm1 == 'g6' and lm2 == 'dxz':
        gphi = 15*np.sqrt(6)*(7*c1**2 - 3)*s2*s1*c2*c1/16.
    elif lm1 == 'g6' and lm2 == 'fxz2':
        gphi = 3*np.sqrt(105)*(5*c2**2 - 1)*(7*c1**2 - 3)*s2*s1*c1/32.
    elif lm1 == 'g6' and lm2 == 'g6':
        gphi = 45*(7*c2**2 - 3)*(7*c1**2 - 3)*s2*s1*c2*c1/32.
    elif lm1 == 'g6' and lm2 == 'h7':
        gphi = 15*np.sqrt(66)*(7*c1**2 - 3)*(21*c2**4 - 14*c2**2 + 1)*s2*s1*c1/128.
    elif lm1 == 'g7' and lm2 == 'dx2-y2':
        gphi = 15*np.sqrt(3)*(6 - 7*s1**2)*s2**2*s1**2/32.
    elif lm1 == 'g7' and lm2 == 'fz(x2-y2)':
        gphi = 15*np.sqrt(21)*(6 - 7*s1**2)*s2**2*s1**2*c2/32.
    elif lm1 == 'g7' and lm2 == 'g7':
        gphi = 45*(7*s2**2 - 6)*(7*s1**2 - 6)*s2**2*s1**2/64.
    elif lm1 == 'g7' and lm2 == 'h8':
        gphi = 15*np.sqrt(231)*(3*s2**2 - 2)*(7*s1**2 - 6)*s2**2*s1**2*c2/64.
    elif lm1 == 'g8' and lm2 == 'fx(x2-3y2)':
        gphi = 105*s2**3*s1**3*c1/32.
    elif lm1 == 'g8' and lm2 == 'g8':
        gphi = 315*s2**3*s1**3*c2*c1/32.
    elif lm1 == 'g8' and lm2 == 'h9':
        gphi = 105*np.sqrt(11)*(9*c2**2 - 1)*s2**3*s1**3*c1/128.
    elif lm1 == 'g9' and lm2 == 'g9':
        gphi = 315*s2**4*s1**4/256.
    elif lm1 == 'g9' and lm2 == 'h10':
        gphi = 315*np.sqrt(11)*s2**4*s1**4*c2/256.
    elif lm1 == 'h1' and lm2 == 'h1':
        gphi = 693*s2**5*s1**5/512.
    elif lm1 == 'h2' and lm2 == 'g1':
        gphi = 315*np.sqrt(11)*s2**4*s1**4*c1/256.
    elif lm1 == 'h2' and lm2 == 'h2':
        gphi = 3465*s2**4*s1**4*c2*c1/256.
    elif lm1 == 'h3' and lm2 == 'fy(3x2-y2)':
        gphi = 35*np.sqrt(11)*(9*c1**2 - 1)*s2**3*s1**3/128.
    elif lm1 == 'h3' and lm2 == 'g2':
        gphi = 105*np.sqrt(11)*(9*c1**2 - 1)*s2**3*s1**3*c2/128.
    elif lm1 == 'h3' and lm2 == 'h3':
        gphi = 385*(9*c2**2 - 1)*(9*c1**2 - 1)*s2**3*s1**3/512.
    elif lm1 == 'h4' and lm2 == 'dxy':
        gphi = 15*np.sqrt(77)*(2 - 3*s1**2)*s2**2*s1**2*c1/32.
    elif lm1 == 'h4' and lm2 == 'fxyz':
        gphi = 105*np.sqrt(11)*(2 - 3*s1**2)*s2**2*s1**2*c2*c1/32.
    elif lm1 == 'h4' and lm2 == 'g3':
        gphi = 15*np.sqrt(231)*(7*s2**2 - 6)*(3*s1**2 - 2)*s2**2*s1**2*c1/64.
    elif lm1 == 'h4' and lm2 == 'h4':
        gphi = 1155*(3*s2**2 - 2)*(3*s1**2 - 2)*s2**2*s1**2*c2*c1/64.
    elif lm1 == 'h5' and lm2 == 'py':
        gphi = 3*np.sqrt(55)*(21*c1**4 - 14*c1**2 + 1)*s2*s1/32.
    elif lm1 == 'h5' and lm2 == 'dyz':
        gphi = 15*np.sqrt(11)*(21*c1**4 - 14*c1**2 + 1)*s2*s1*c2/32.
    elif lm1 == 'h5' and lm2 == 'fyz2':
        gphi = 3*np.sqrt(770)*(5*c2**2 - 1)*(21*c1**4 - 14*c1**2 + 1)*s2*s1/128.
    elif lm1 == 'h5' and lm2 == 'g4':
        gphi = 15*np.sqrt(66)*(7*c2**2 - 3)*(21*c1**4 - 14*c1**2 + 1)*s2*s1*c2/128.
    elif lm1 == 'h5' and lm2 == 'h5':
        gphi = 165*(21*c2**4 - 14*c2**2 + 1)*(21*c1**4 - 14*c1**2 + 1)*s2*s1/256.
    elif lm1 == 'h6' and lm2 == 's':
        gphi = np.sqrt(11)*(63*c1**4 - 70*c1**2 + 15)*c1/16.
    elif lm1 == 'h6' and lm2 == 'pz':
        gphi = np.sqrt(33)*(63*c1**4 - 70*c1**2 + 15)*c2*c1/16.
    elif lm1 == 'h6' and lm2 == 'dz2':
        gphi = np.sqrt(55)*(3*c2**2 - 1)*(63*c1**4 - 70*c1**2 + 15)*c1/32.
    elif lm1 == 'h6' and lm2 == 'fz3':
        gphi = np.sqrt(77)*(5*c2**2 - 3)*(63*c1**4 - 70*c1**2 + 15)*c2*c1/32.
    elif lm1 == 'h6' and lm2 == 'g5':
        gphi = 3*np.sqrt(11)*(35*c2**4 - 30*c2**2 + 3)*(63*c1**4 - 70*c1**2 + 15)*c1/128.
    elif lm1 == 'h6' and lm2 == 'h6':
        gphi = 11*(63*c2**4 - 70*c2**2 + 15)*(63*c1**4 - 70*c1**2 + 15)*c2*c1/128.
    elif lm1 == 'h7' and lm2 == 'px':
        gphi = 3*np.sqrt(55)*(21*c1**4 - 14*c1**2 + 1)*s2*s1/32.
    elif lm1 == 'h7' and lm2 == 'dxz':
        gphi = 15*np.sqrt(11)*(21*c1**4 - 14*c1**2 + 1)*s2*s1*c2/32.
    elif lm1 == 'h7' and lm2 == 'fxz2':
        gphi = 3*np.sqrt(770)*(5*c2**2 - 1)*(21*c1**4 - 14*c1**2 + 1)*s2*s1/128.
    elif lm1 == 'h7' and lm2 == 'g6':
        gphi = 15*np.sqrt(66)*(7*c2**2 - 3)*(21*c1**4 - 14*c1**2 + 1)*s2*s1*c2/128.
    elif lm1 == 'h7' and lm2 == 'h7':
        gphi = 165*(21*c2**4 - 14*c2**2 + 1)*(21*c1**4 - 14*c1**2 + 1)*s2*s1/256.
    elif lm1 == 'h8' and lm2 == 'dx2-y2':
        gphi = 15*np.sqrt(77)*(2 - 3*s1**2)*s2**2*s1**2*c1/32.
    elif lm1 == 'h8' and lm2 == 'fz(x2-y2)':
        gphi = 105*np.sqrt(11)*(2 - 3*s1**2)*s2**2*s1**2*c2*c1/32.
    elif lm1 == 'h8' and lm2 == 'g7':
        gphi = 15*np.sqrt(231)*(7*s2**2 - 6)*(3*s1**2 - 2)*s2**2*s1**2*c1/64.
    elif lm1 == 'h8' and lm2 == 'h8':
        gphi = 1155*(3*s2**2 - 2)*(3*s1**2 - 2)*s2**2*s1**2*c2*c1/64.
    elif lm1 == 'h9' and lm2 == 'fx(x2-3y2)':
        gphi = 35*np.sqrt(11)*(9*c1**2 - 1)*s2**3*s1**3/128.
    elif lm1 == 'h9' and lm2 == 'g8':
        gphi = 105*np.sqrt(11)*(9*c1**2 - 1)*s2**3*s1**3*c2/128.
    elif lm1 == 'h9' and lm2 == 'h9':
        gphi = 385*(9*c2**2 - 1)*(9*c1**2 - 1)*s2**3*s1**3/512.
    elif lm1 == 'h10' and lm2 == 'g9':
        gphi = 315*np.sqrt(11)*s2**4*s1**4*c1/256.
    elif lm1 == 'h10' and lm2 == 'h10':
        gphi = 3465*s2**4*s1**4*c2*c1/256.
    elif lm1 == 'h11' and lm2 == 'h11':
        gphi = 693*s2**5*s1**5/512.
    else:
        gphi = 0
    return gphi


def get_twocenter_phi_integrals_derivatives(lm1, lm2, c1, c2, s1, s2):
    r"""
    Returns the equivalent of get_twocenter_phi_integrals() with
    selected derivatives of the involved spherical harmonics
    for combinations including s/p/d angular momenta.

    Parameters
    ----------
    lm1, lm2 : str
        Orbital labels.
    c1, c2, s1, s2 : np.ndarray
        Cosine (c1 and c2) and sine (s1 and s2) of the
        theta1 and theta2 angles, respectively.

    Returns
    -------
    dghi : list of 4 np.ndarray
        List with the following four integrals:
            dY_{lm1}(\theta_1)/d\theta_1 dY_{lm1}(\theta_2)/d\theta_2
            \int Y_{lm1}(\phi) Y_{lm1}(\phi) d\phi,
            Y_{lm1}(\theta_1) Y_{lm1}(\theta_2)
            \int dY_{lm1}(\phi)/d\phi dY_{lm1}(\phi)/d\phi d\phi,
            dY_{lm1}(\theta_1)/d\theta_1 Y_{lm1}(\theta_2)
            \int Y_{lm1}(\phi) Y_{lm1}(\phi) d\phi,
            Y_{lm1}(\theta_1) dY_{lm1}(\theta_2)/d\theta_2
            \int Y_{lm1}(\phi) Y_{lm1}(\phi) d\phi.
    """
    if lm1 == 's' and lm2 == 'pz':
        dgphi = [
            0,
            0,
            0,
            -np.sqrt(3)*s2/2.,
        ]
    elif lm1 == 's' and lm2 == 'dz2':
        dgphi = [
            0,
            0,
            0,
            -3*np.sqrt(5)*s2*c2/2.,
        ]
    elif lm1 == 'px' and lm2 == 's':
        dgphi = [
            0,
            np.sqrt(3)*s1/4.,
            0,
            0,
        ]
    elif lm1 == 'px' and lm2 == 'px':
        dgphi = [
            3*c2*c1/4.,
            3*s2*s1/4.,
            3*s2*c1/4.,
            3*s1*c2/4.,
        ]
    elif lm1 == 'px' and lm2 == 'py':
        dgphi = [
            0,
            3*s2*s1/4.,
            0,
            0,
        ]
    elif lm1 == 'px' and lm2 == 'pz':
        dgphi = [
            0,
            3*s1*c2/4.,
            0,
            0,
        ]
    elif lm1 == 'px' and lm2 == 'dxy':
        dgphi = [
            0,
            3*np.sqrt(5)*s2**2*s1/8.,
            0,
            0,
        ]
    elif lm1 == 'px' and lm2 == 'dyz':
        dgphi = [
            0,
            3*np.sqrt(5)*s2*s1*c2/4.,
            0,
            0,
        ]
    elif lm1 == 'px' and lm2 == 'dxz':
        dgphi = [
            3*np.sqrt(5)*(2*c2**2 - 1)*c1/4.,
            3*np.sqrt(5)*s2*s1*c2/4.,
            3*np.sqrt(5)*s2*c2*c1/4.,
            3*np.sqrt(5)*(2*c2**2 - 1)*s1/4.,
        ]
    elif lm1 == 'px' and lm2 == 'dx2-y2':
        dgphi = [
            0,
            3*np.sqrt(5)*s2**2*s1/8.,
            0,
            0,
        ]
    elif lm1 == 'px' and lm2 == 'dz2':
        dgphi = [
            0,
            np.sqrt(15)*(3*c2**2 - 1)*s1/8.,
            0,
            0,
        ]
    elif lm1 == 'py' and lm2 == 's':
        dgphi = [
            0,
            np.sqrt(3)*s1/4.,
            0,
            0,
        ]
    elif lm1 == 'py' and lm2 == 'px':
        dgphi = [
            0,
            3*s2*s1/4.,
            0,
            0,
        ]
    elif lm1 == 'py' and lm2 == 'py':
        dgphi = [
            3*c2*c1/4.,
            3*s2*s1/4.,
            3*s2*c1/4.,
            3*s1*c2/4.,
        ]
    elif lm1 == 'py' and lm2 == 'pz':
        dgphi = [
            0,
            3*s1*c2/4.,
            0,
            0,
        ]
    elif lm1 == 'py' and lm2 == 'dxy':
        dgphi = [
            0,
            3*np.sqrt(5)*s2**2*s1/8.,
            0,
            0,
        ]
    elif lm1 == 'py' and lm2 == 'dyz':
        dgphi = [
            3*np.sqrt(5)*(2*c2**2 - 1)*c1/4.,
            3*np.sqrt(5)*s2*s1*c2/4.,
            3*np.sqrt(5)*s2*c2*c1/4.,
            3*np.sqrt(5)*(2*c2**2 - 1)*s1/4.,
        ]
    elif lm1 == 'py' and lm2 == 'dxz':
        dgphi = [
            0,
            3*np.sqrt(5)*s2*s1*c2/4.,
            0,
            0,
        ]
    elif lm1 == 'py' and lm2 == 'dx2-y2':
        dgphi = [
            0,
            3*np.sqrt(5)*s2**2*s1/8.,
            0,
            0,
        ]
    elif lm1 == 'py' and lm2 == 'dz2':
        dgphi = [
            0,
            np.sqrt(15)*(3*c2**2 - 1)*s1/8.,
            0,
            0,
        ]
    elif lm1 == 'pz' and lm2 == 's':
        dgphi = [
            0,
            0,
            -np.sqrt(3)*s1/2.,
            0,
        ]
    elif lm1 == 'pz' and lm2 == 'pz':
        dgphi = [
            3*s2*s1/2.,
            0,
            -3*s1*c2/2.,
            -3*s2*c1/2.,
        ]
    elif lm1 == 'pz' and lm2 == 'dz2':
        dgphi = [
            3*np.sqrt(15)*s2*s1*c2/2.,
            0,
            np.sqrt(15)*(1 - 3*c2**2)*s1/4.,
            -3*np.sqrt(15)*s2*c2*c1/2.,
        ]
    elif lm1 == 'dxy' and lm2 == 's':
        dgphi = [
            0,
            np.sqrt(15)*s1**2/2.,
            0,
            0,
        ]
    elif lm1 == 'dxy' and lm2 == 'px':
        dgphi = [
            0,
            3*np.sqrt(5)*s2*s1**2/2.,
            0,
            0,
        ]
    elif lm1 == 'dxy' and lm2 == 'py':
        dgphi = [
            0,
            3*np.sqrt(5)*s2*s1**2/2.,
            0,
            0,
        ]
    elif lm1 == 'dxy' and lm2 == 'pz':
        dgphi = [
            0,
            3*np.sqrt(5)*s1**2*c2/2.,
            0,
            0,
        ]
    elif lm1 == 'dxy' and lm2 == 'dxy':
        dgphi = [
            15*s2*s1*c2*c1/4.,
            15*s2**2*s1**2/4.,
            15*s2**2*s1*c1/8.,
            15*s2*s1**2*c2/8.,
        ]
    elif lm1 == 'dxy' and lm2 == 'dyz':
        dgphi = [
            0,
            15*s2*s1**2*c2/2.,
            0,
            0,
        ]
    elif lm1 == 'dxy' and lm2 == 'dxz':
        dgphi = [
            0,
            15*s2*s1**2*c2/2.,
            0,
            0,
        ]
    elif lm1 == 'dxy' and lm2 == 'dx2-y2':
        dgphi = [
            0,
            15*s2**2*s1**2/4.,
            0,
            0,
        ]
    elif lm1 == 'dxy' and lm2 == 'dz2':
        dgphi = [
            0,
            5*np.sqrt(3)*(3*c2**2 - 1)*s1**2/4.,
            0,
            0,
        ]
    elif lm1 == 'dyz' and lm2 == 's':
        dgphi = [
            0,
            np.sqrt(15)*s1*c1/4.,
            0,
            0,
        ]
    elif lm1 == 'dyz' and lm2 == 'px':
        dgphi = [
            0,
            3*np.sqrt(5)*s2*s1*c1/4.,
            0,
            0,
        ]
    elif lm1 == 'dyz' and lm2 == 'py':
        dgphi = [
            3*np.sqrt(5)*(2*c1**2 - 1)*c2/4.,
            3*np.sqrt(5)*s2*s1*c1/4.,
            3*np.sqrt(5)*(2*c1**2 - 1)*s2/4.,
            3*np.sqrt(5)*s1*c2*c1/4.,
        ]
    elif lm1 == 'dyz' and lm2 == 'pz':
        dgphi = [
            0,
            3*np.sqrt(5)*s1*c2*c1/4.,
            0,
            0,
        ]
    elif lm1 == 'dyz' and lm2 == 'dxy':
        dgphi = [
            0,
            15*s2**2*s1*c1/8.,
            0,
            0,
        ]
    elif lm1 == 'dyz' and lm2 == 'dyz':
        dgphi = [
            15*(2*c2**2 - 1)*(2*c1**2 - 1)/4.,
            15*s2*s1*c2*c1/4.,
            15*(2*c1**2 - 1)*s2*c2/4.,
            15*(2*c2**2 - 1)*s1*c1/4.,
        ]
    elif lm1 == 'dyz' and lm2 == 'dxz':
        dgphi = [
            0,
            15*s2*s1*c2*c1/4.,
            0,
            0,
        ]
    elif lm1 == 'dyz' and lm2 == 'dx2-y2':
        dgphi = [
            0,
            15*s2**2*s1*c1/8.,
            0,
            0,
        ]
    elif lm1 == 'dyz' and lm2 == 'dz2':
        dgphi = [
            0,
            5*np.sqrt(3)*(3*c2**2 - 1)*s1*c1/8.,
            0,
            0,
        ]
    elif lm1 == 'dxz' and lm2 == 's':
        dgphi = [
            0,
            np.sqrt(15)*s1*c1/4.,
            0,
            0,
        ]
    elif lm1 == 'dxz' and lm2 == 'px':
        dgphi = [
            3*np.sqrt(5)*(2*c1**2 - 1)*c2/4.,
            3*np.sqrt(5)*s2*s1*c1/4.,
            3*np.sqrt(5)*(2*c1**2 - 1)*s2/4.,
            3*np.sqrt(5)*s1*c2*c1/4.,
        ]
    elif lm1 == 'dxz' and lm2 == 'py':
        dgphi = [
            0,
            3*np.sqrt(5)*s2*s1*c1/4.,
            0,
            0,
        ]
    elif lm1 == 'dxz' and lm2 == 'pz':
        dgphi = [
            0,
            3*np.sqrt(5)*s1*c2*c1/4.,
            0,
            0,
        ]
    elif lm1 == 'dxz' and lm2 == 'dxy':
        dgphi = [
            0,
            15*s2**2*s1*c1/8.,
            0,
            0,
        ]
    elif lm1 == 'dxz' and lm2 == 'dyz':
        dgphi = [
            0,
            15*s2*s1*c2*c1/4.,
            0,
            0,
        ]
    elif lm1 == 'dxz' and lm2 == 'dxz':
        dgphi = [
            15*(2*c2**2 - 1)*(2*c1**2 - 1)/4.,
            15*s2*s1*c2*c1/4.,
            15*(2*c1**2 - 1)*s2*c2/4.,
            15*(2*c2**2 - 1)*s1*c1/4.,
        ]
    elif lm1 == 'dxz' and lm2 == 'dx2-y2':
        dgphi = [
            0,
            15*s2**2*s1*c1/8.,
            0,
            0,
        ]
    elif lm1 == 'dxz' and lm2 == 'dz2':
        dgphi = [
            0,
            5*np.sqrt(3)*(3*c2**2 - 1)*s1*c1/8.,
            0,
            0,
        ]
    elif lm1 == 'dx2-y2' and lm2 == 's':
        dgphi = [
            0,
            np.sqrt(15)*s1**2/2.,
            0,
            0,
        ]
    elif lm1 == 'dx2-y2' and lm2 == 'px':
        dgphi = [
            0,
            3*np.sqrt(5)*s2*s1**2/2.,
            0,
            0,
        ]
    elif lm1 == 'dx2-y2' and lm2 == 'py':
        dgphi = [
            0,
            3*np.sqrt(5)*s2*s1**2/2.,
            0,
            0,
        ]
    elif lm1 == 'dx2-y2' and lm2 == 'pz':
        dgphi = [
            0,
            3*np.sqrt(5)*s1**2*c2/2.,
            0,
            0,
        ]
    elif lm1 == 'dx2-y2' and lm2 == 'dxy':
        dgphi = [
            0,
            15*s2**2*s1**2/4.,
            0,
            0,
        ]
    elif lm1 == 'dx2-y2' and lm2 == 'dyz':
        dgphi = [
            0,
            15*s2*s1**2*c2/2.,
            0,
            0,
        ]
    elif lm1 == 'dx2-y2' and lm2 == 'dxz':
        dgphi = [
            0,
            15*s2*s1**2*c2/2.,
            0,
            0,
        ]
    elif lm1 == 'dx2-y2' and lm2 == 'dx2-y2':
        dgphi = [
            15*s2*s1*c2*c1/4.,
            15*s2**2*s1**2/4.,
            15*s2**2*s1*c1/8.,
            15*s2*s1**2*c2/8.,
        ]
    elif lm1 == 'dx2-y2' and lm2 == 'dz2':
        dgphi = [
            0,
            5*np.sqrt(3)*(3*c2**2 - 1)*s1**2/4.,
            0,
            0,
        ]
    elif lm1 == 'dz2' and lm2 == 's':
        dgphi = [
            0,
            0,
            -3*np.sqrt(5)*s1*c1/2.,
            0,
        ]
    elif lm1 == 'dz2' and lm2 == 'pz':
        dgphi = [
            3*np.sqrt(15)*s2*s1*c1/2.,
            0,
            -3*np.sqrt(15)*s1*c2*c1/2.,
            np.sqrt(15)*(1 - 3*c1**2)*s2/4.,
        ]
    elif lm1 == 'dz2' and lm2 == 'dz2':
        dgphi = [
            45*s2*s1*c2*c1/2.,
            0,
            15*(1 - 3*c2**2)*s1*c1/4.,
            15*(1 - 3*c1**2)*s2*c2/4.,
        ]
    else:
        dgphi = [0, 0, 0, 0]

    return dgphi


def tail_smoothening(x, y_in, eps_inner=1e-8, eps_outer=1e-16, window_size=5):
    """ Smoothens the tail for the given function y(x).

    Parameters
    ----------
    x : np.array
        Array with grid points (strictly increasing).
    y_in : np.array
        Array with function values.
    eps_inner : float, optional
        Inner threshold. Tail values with magnitudes between this value and
        the outer threshold are subjected to moving window averaging to
        reduce noise.
    eps_outer : float, optional
        Outer threshold. Tail values with magnitudes below this value
        are set to zero.
    window_size : int, optional
        Moving average window size (odd integers only).

    Returns
    -------
    y_out : np.array
        Array with function values with a smoothed tail.
    """
    assert window_size % 2 == 1, 'Window size needs to be odd.'

    y_out = np.copy(y_in)
    N = len(y_out)

    if np.all(abs(y_in) < eps_outer):
        return y_out

    Nzero = 0
    izero = -1
    for izero in range(N-1, 1, -1):
        if abs(y_out[izero]) < eps_outer:
            Nzero += 1
        else:
            break

    y_out[izero+1:] = 0.

    Nsmall = 0
    for ismall in range(izero, 1, -1):
        if abs(y_out[ismall]) < eps_inner:
            Nsmall += 1
        else:
            break
    else:
        ismall -= 1

    if Nsmall > 0:
        tail = np.empty(Nsmall-1)
        half = (window_size - 1) // 2
        for j, i in enumerate(range(ismall+1, izero)):
            tail[j] = np.mean(y_out[i-half:i+half+1])

        y_out[ismall+1:izero] = tail

    return y_out


def write_skf(handle, Rgrid, table, has_diagonal_data, is_extended, eigval,
              hubval, occup, spe, mass, has_offdiagonal_data, offdiag_H,
              offdiag_S):
    """
    Writes a parameter file in '.skf' format.

    Parameters
    ----------
    handle : file handle
        Handle of an open file.
    Rgrid : list or array
        Lists with interatomic distances.
    table : nd.ndarray
        Two-dimensional array with the Slater-Koster table.

    Other parameters
    ----------------
    See Offsite2cTable.write()
    """
    assert not (has_diagonal_data and has_offdiagonal_data)

    if is_extended:
        print('@', file=handle)

    grid_dist = Rgrid[1] - Rgrid[0]
    grid_npts, numint = np.shape(table)
    assert (numint % NUMSK) == 0
    nzeros = int(np.round(Rgrid[0] / grid_dist)) - 1
    assert nzeros >= 0
    print("%.12f, %d" % (grid_dist, grid_npts + nzeros), file=handle)

    if has_diagonal_data or has_offdiagonal_data:
        if has_diagonal_data:
            prefixes, dicts = ['E', 'U', 'f'], [eigval, hubval, occup]
            fields = ['E_f', 'E_d', 'E_p', 'E_s', 'SPE', 'U_f', 'U_d',
                      'U_p', 'U_s', 'f_f', 'f_d', 'f_p', 'f_s']
            labels = {'SPE': spe}
        elif has_offdiagonal_data:
            prefixes, dicts = ['H', 'S'], [offdiag_H, offdiag_S]
            fields = ['H_f', 'H_d', 'H_p', 'H_s',
                      'S_f', 'S_d', 'S_p', 'S_s']
            labels = {}

        if not is_extended:
            fields = [field for field in fields if field[-1] != 'f']

        for prefix, d in zip(prefixes, dicts):
            for l in ['s', 'p', 'd', 'f']:
                if l in d:
                    key = '%s_%s' % (prefix, l)
                    labels[key] = d[l]

        line = ' '.join(fields)
        for field in fields:
            val = labels[field] if field in labels else 0
            s = '%d' % val if isinstance(val, int) else '%.6f' % val
            line = line.replace(field, s)

        print(line, file=handle)

    print("%.3f, 19*0.0" % mass, file=handle)

    # Table containing the Slater-Koster integrals
    numtab = numint // NUMSK
    assert numtab > 0

    if is_extended:
        indices = list(range(numtab*NUMSK))
    else:
        selected = [INTEGRALS.index(name) for name in INTEGRALS
                    if 'f' not in name[:2]]
        indices = []
        for itab in range(numtab):
            indices.extend([itab*NUMSK+j for j in selected])

    for i in range(nzeros):
        print('%d*0.0,' % len(indices), file=handle)

    for i in range(grid_npts):
        line = ''
        num_zero = 0
        zero_str = ''

        for j in indices:
            if table[i, j] == 0:
                num_zero += 1
                zero_str = str(num_zero) + '*0.0 '
            else:
                num_zero = 0
                line += zero_str
                zero_str = ''
                line += '{0: 1.12e}  '.format(table[i, j])

        if zero_str != '':
            line += zero_str

        print(line, file=handle)
