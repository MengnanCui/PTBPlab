#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
from hotcent.orbitals import ORBITAL_LABELS


INTEGRAL_PAIRS = {'%s_%s' % (lm1, lm2): (lm1, lm2)
                  for lm1 in ORBITAL_LABELS for lm2 in ORBITAL_LABELS}

INTEGRALS = INTEGRAL_PAIRS.keys()

XZ_SYMMETRIC_ORBITALS = ['s', 'px', 'pz', 'dxz', 'dx2-y2', 'dz2']

XZ_ANTISYMMETRIC_ORBITALS = ['py', 'dxy', 'dyz']


def select_integrals(e1, e2):
    """ Return list of non-zero integrals (integral, nl1, nl2)
    to be evaluated for element pair e1, e2. """
    selected = []

    for ival1, valence1 in enumerate(e1.basis_sets):
        for ival2, valence2 in enumerate(e2.basis_sets):
            for integral in INTEGRALS:
                nl1, nl2 = select_subshells(valence1, valence2, integral)
                if nl1 is not None and nl2 is not None:
                    lm1, lm2 = integral.split('_')

                    if lm1 in XZ_ANTISYMMETRIC_ORBITALS:
                        is_nonzero = lm2 in XZ_ANTISYMMETRIC_ORBITALS
                    else:
                        is_nonzero = lm2 not in XZ_ANTISYMMETRIC_ORBITALS

                    if is_nonzero:
                        selected.append((integral, nl1, nl2))
    return selected


def select_subshells(val1, val2, integral):
    """
    Select subshells from given valence sets to evaluate
    the given integral.

    Parameters
    ----------
    val1, val2 : list of str
        Valence subshell sets (e.g. ['2s', '2p'], ['4s', '3d']).
    integral : str
        Integral label (e.g. 's_dxy').

    Returns
    -------
    nl1, nl2 : str
        Matching subshell pair (e.g. ('2s', '3d') in this example).
    """
    nl1 = None
    for nl in val1:
        if nl[1] == integral.split('_')[0][0]:
            nl1 = nl

    nl2 = None
    for nl in val2:
        if nl[1] == integral.split('_')[1][0]:
            nl2 = nl

    return nl1, nl2


def write_3cf(filename, Rgrid, Sgrid, Tgrid, data, fmt='%.8e'):
    """
    Writes a parameter file in '.3cf' format.

    Parameters
    ----------
    filename : str
        File name.
    Rgrid, Sgrid, Tgrid : list or array
        Lists with distances defining the three-atom geometries.
    data : dict
        Dictionary with the tabulated values for each integral type.
    fmt : str, optional
        Formatting string for the integrals.
    """
    numR = len(Rgrid)
    numS = len(Sgrid)
    numT = len(Tgrid)

    with open(filename, 'w') as f:
        # Header
        f.write('%.6f %.6f %d\n' % (Rgrid[0], Rgrid[-1], numR))
        f.write('%.6f %.6f %d\n' % (Sgrid[0], Sgrid[-1], numS))
        f.write('%d\n' % numT)

        keys = list(data.keys())
        f.write(' '.join(keys) + '\n')

        # Body
        for i in range(numR):
            for j in range(1 + numS*numT):
                f.write(' '.join([fmt % data[key][i][j] for key in keys]))
                f.write('\n')
