#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.orbitals import ANGULAR_MOMENTUM, ORBITALS


NUML_1CK = 3  # total number of multipoles considered for .1ck files (up to d)

NUML_1CL = 4  # number of subshells included in .1cl files (4 = up to f)

NUML_1CM = 3  # number of subshells included in .1cm files (3 = up to d)

NUMLM_1CM = NUML_1CM**2  # number of orbitals considered for .1cm files


def select_orbitals(el):
    """ Returns a list of orbitals (nl, lm) to be evaluated
    for the given element. """
    orbitals = []
    for valence in el.basis_sets:
        for nl in valence:
            l = ANGULAR_MOMENTUM[nl[1]]
            for lm in ORBITALS[l]:
                orbitals.append((nl, lm))

    selected = []
    for i, orbital1 in enumerate(orbitals):
        for j, orbital2 in enumerate(orbitals):
            if j <= i:
                selected.append((orbital1, orbital2))
    return selected


def write_1ck(handle, radmom, kernels):
    """
    Writes a parameter file in '1ck' format.

    Parameters
    ----------
    handle : file handle
        Handle of an open file.
    radmom : nd.ndarray
        One-dimensional array of (NUML_1CK,) shape
    kernels : nd.ndarray
        One-dimensional array of (NUML_1CK,) shape
    """
    assert np.shape(radmom) == (NUML_1CK,)
    assert np.shape(kernels) == (NUML_1CK,)

    for row in [radmom, kernels]:
        for item in row:
            handle.write('0 ' if abs(item) < 1e-16 else '%1.12e ' % item)
        handle.write('\n')
    return


def write_1cl(handle, table):
    """
    Writes a parameter file in '1cl' format.

    Parameters
    ----------
    handle : file handle
        Handle of an open file.
    table : nd.ndarray
        Two-dimensional array of (NUML_1CL, NUML_1CL) shape
    """
    assert np.shape(table) == (NUML_1CL, NUML_1CL)

    for row in table:
        for item in row:
            handle.write('0 ' if abs(item) < 1e-16 else '%1.12e ' % item)
        handle.write('\n')
    return


def write_1cm(handle, table):
    """
    Writes a parameter file in '1cm' format.

    Parameters
    ----------
    handle : file handle
        Handle of an open file.
    table : nd.ndarray
        Three-dimensional table.
    """
    assert table.ndim == 3

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            for item in table[i, j, :]:
                handle.write('0 ' if abs(item) < 1e-16 else '%1.12e ' % item)
            handle.write('\n')
    return
