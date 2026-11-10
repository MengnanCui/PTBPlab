#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.orbitals import ANGULAR_MOMENTUM, ORBITAL_LABELS, ORBITALS


NUMINT_2CL = 4*4  # number of subshell-resolved integrals in .2cl files

NUML_2CK = 3  # number of subshells included in .2ck files (3 = up to d)

NUMLM_2CK = NUML_2CK**2  # number of orbitals considered for .2ck files

NUML_2CM = 3  # number of subshells included in .2cm files (3 = up to d)

NUMLM_2CM = NUML_2CM**2  # number of orbitals considered for .2cm files

INTEGRALS_2CK = [
    'sss', 'sps', 'sds', 'pss', 'pps', 'ppp', 'pds', 'pdp',
    'dss', 'dps', 'dpp', 'dds', 'ddp', 'ddd',
]

NUMSK_2CK = len(INTEGRALS_2CK)

NONZERO_2CM = np.array([
       [[ True, False, False,  True, False, False, False, False,  True],
        [False,  True, False, False, False, False,  True, False, False],
        [False, False,  True, False, False,  True, False, False, False],
        [ True, False, False,  True, False, False, False, False,  True],
        [False, False, False, False,  True, False, False, False, False],
        [False, False,  True, False, False,  True, False, False, False],
        [False,  True, False, False, False, False,  True, False, False],
        [False, False, False, False, False, False, False,  True, False],
        [ True, False, False,  True, False, False, False, False,  True]],
       [[False,  True, False, False, False, False,  True, False, False],
        [ True, False, False,  True, False, False, False,  True,  True],
        [False, False, False, False,  True, False, False, False, False],
        [False,  True, False, False, False, False,  True, False, False],
        [False, False,  True, False, False,  True, False, False, False],
        [False, False, False, False,  True, False, False, False, False],
        [ True, False, False,  True, False, False, False,  True,  True],
        [False,  True, False, False, False, False,  True, False, False],
        [False,  True, False, False, False, False,  True, False, False]],
       [[False, False,  True, False, False,  True, False, False, False],
        [False, False, False, False,  True, False, False, False, False],
        [ True, False, False,  True, False, False, False,  True,  True],
        [False, False,  True, False, False,  True, False, False, False],
        [False,  True, False, False, False, False,  True, False, False],
        [ True, False, False,  True, False, False, False,  True,  True],
        [False, False, False, False,  True, False, False, False, False],
        [False, False,  True, False, False,  True, False, False, False],
        [False, False,  True, False, False,  True, False, False, False]],
       [[ True, False, False,  True, False, False, False, False,  True],
        [False,  True, False, False, False, False,  True, False, False],
        [False, False,  True, False, False,  True, False, False, False],
        [ True, False, False,  True, False, False, False, False,  True],
        [False, False, False, False,  True, False, False, False, False],
        [False, False,  True, False, False,  True, False, False, False],
        [False,  True, False, False, False, False,  True, False, False],
        [False, False, False, False, False, False, False,  True, False],
        [ True, False, False,  True, False, False, False, False,  True]],
       [[False, False, False, False,  True, False, False, False, False],
        [False, False,  True, False, False,  True, False, False, False],
        [False,  True, False, False, False, False,  True, False, False],
        [False, False, False, False,  True, False, False, False, False],
        [ True, False, False,  True, False, False, False, False,  True],
        [False,  True, False, False, False, False,  True, False, False],
        [False, False,  True, False, False,  True, False, False, False],
        [False, False, False, False, False, False, False, False, False],
        [False, False, False, False,  True, False, False, False, False]],
       [[False, False,  True, False, False,  True, False, False, False],
        [False, False, False, False,  True, False, False, False, False],
        [ True, False, False,  True, False, False, False,  True,  True],
        [False, False,  True, False, False,  True, False, False, False],
        [False,  True, False, False, False, False,  True, False, False],
        [ True, False, False,  True, False, False, False,  True,  True],
        [False, False, False, False,  True, False, False, False, False],
        [False, False,  True, False, False,  True, False, False, False],
        [False, False,  True, False, False,  True, False, False, False]],
       [[False,  True, False, False, False, False,  True, False, False],
        [ True, False, False,  True, False, False, False,  True,  True],
        [False, False, False, False,  True, False, False, False, False],
        [False,  True, False, False, False, False,  True, False, False],
        [False, False,  True, False, False,  True, False, False, False],
        [False, False, False, False,  True, False, False, False, False],
        [ True, False, False,  True, False, False, False,  True,  True],
        [False,  True, False, False, False, False,  True, False, False],
        [False,  True, False, False, False, False,  True, False, False]],
       [[False, False, False, False, False, False, False,  True, False],
        [False,  True, False, False, False, False,  True, False, False],
        [False, False,  True, False, False,  True, False, False, False],
        [False, False, False, False, False, False, False,  True, False],
        [False, False, False, False, False, False, False, False, False],
        [False, False,  True, False, False,  True, False, False, False],
        [False,  True, False, False, False, False,  True, False, False],
        [ True, False, False,  True, False, False, False, False,  True],
        [False, False, False, False, False, False, False,  True, False]],
       [[ True, False, False,  True, False, False, False, False,  True],
        [False,  True, False, False, False, False,  True, False, False],
        [False, False,  True, False, False,  True, False, False, False],
        [ True, False, False,  True, False, False, False, False,  True],
        [False, False, False, False,  True, False, False, False, False],
        [False, False,  True, False, False,  True, False, False, False],
        [False,  True, False, False, False, False,  True, False, False],
        [False, False, False, False, False, False, False,  True, False],
        [ True, False, False,  True, False, False, False, False,  True]],
     ])  # (NUMLM_2CM, NUMLM_2CM, NUMLM_2CK) array of non-zero .2cm integrals


def select_subshells(e1, e2):
    """ Returns a list of subshell pairs (nl1, nl2)
    to be evaluated for the given element pair. """
    selected = []
    for ival1, valence1 in enumerate(e1.basis_sets):
        for nl1 in valence1:
            for ival2, valence2 in enumerate(e2.basis_sets):
                for nl2 in valence2:
                    selected.append((nl1, nl2))
    return selected


def select_orbitals(e1, e2):
    """ Returns a list of orbital pairs ((nl1, lm1), (nl2, lm2))
    to be evaluated for the given element pair. """
    orbitals1 = []
    for valence in e1.basis_sets:
        for nl in valence:
            l = ANGULAR_MOMENTUM[nl[1]]
            for lm in ORBITALS[l]:
                orbitals1.append((nl, lm))

    orbitals2 = []
    for valence in e2.basis_sets:
        for nl in valence:
            l = ANGULAR_MOMENTUM[nl[1]]
            for lm in ORBITALS[l]:
                orbitals2.append((nl, lm))

    selected = []
    for i, orbital1 in enumerate(orbitals1):
        for j, orbital2 in enumerate(orbitals2):
            selected.append((orbital1, orbital2))
    return selected


def write_2cl(handle, Rgrid, table, angmom1, angmom2):
    """
    Writes a parameter file in '2cl' format.

    Parameters
    ----------
    angmom1: list of int
        Included angular momenta for the first element.
    angmom2: list of int
        Included angular momenta for the second element.

    Other parameters
    ----------------
    See slako.write_skf()
    """
    grid_dist = Rgrid[1] - Rgrid[0]
    grid_npts, numint = np.shape(table)
    assert numint == NUMINT_2CL

    nzeros = int(np.round(Rgrid[0] / grid_dist)) - 1
    assert nzeros >= 0

    print("%.12f, %d" % (grid_dist, grid_npts + nzeros), file=handle)

    for i in range(nzeros):
        print(' '.join(['0.0'] * NUMINT_2CL), file=handle)

    formats = []
    for i in range(NUMINT_2CL):
        l1 = i // 4
        l2 = i % 4
        fmt = '%1.12e' if l1 in angmom1 and l2 in angmom2 else '%.1f'
        formats.append(fmt)

    np.savetxt(handle, table, fmt=formats)
    return


def write_2ck(handle, Rgrid, table, point_kernels=None):
    """
    Writes a parameter file in '2ck' format.

    Parameters
    ----------
    point_kernels : None or np.ndarray, optional
        Point multipole kernel values for every integral,
        evaluated at R = 1 Bohr. If None, a zero-valued array
        will be used.

    Other parameters
    ----------------
    See slako.write_skf()
    """
    grid_dist = Rgrid[1] - Rgrid[0]
    grid_npts, numint = np.shape(table)
    assert numint == NUMSK_2CK

    nzeros = int(np.round(Rgrid[0] / grid_dist)) - 1
    assert nzeros >= 0

    print("%.12f, %d" % (grid_dist, grid_npts + nzeros), file=handle)
    if point_kernels is None:
        np.savetxt(handle, np.zeros((1, NUMSK_2CK)), fmt='%.12f')
    else:
        np.savetxt(handle, [point_kernels], fmt='%.12f')

    for i in range(nzeros):
        print(' '.join(['0.0'] * numint), file=handle)

    np.savetxt(handle, table, fmt='%1.12e')
    return


def write_2cm(handle, Rgrid, table, aux_orbitals):
    """
    Writes a parameter file in '2cm' format.

    Parameters
    ----------
    handle : file handle
        Handle of an open file.
    Rgrid : np.ndarray
        Array of interatomic distances.
    table : nd.ndarray
        Three-dimensional table.
    aux_orbitals : list of str
        Orbital labels for every auxiliary basis function.
    """
    grid_dist = Rgrid[1] - Rgrid[0]
    grid_npts = len(Rgrid)
    nzeros = int(np.round(Rgrid[0] / grid_dist)) - 1
    assert nzeros >= 0
    Naux = len(aux_orbitals)
    assert table.ndim == 4
    assert table.shape[0] == NUMLM_2CM, table.shape[0]
    assert table.shape[1] == NUMLM_2CM, table.shape[1]
    assert table.shape[2] == grid_npts, table.shape[2]
    assert table.shape[3] == Naux, table.shape[3]

    print("%.12f, %d" % (grid_dist, grid_npts + nzeros), file=handle)

    for ilm in range(NUMLM_2CM):
        for jlm in range(NUMLM_2CM):
            header = '# {0}_{1}'.format(ORBITAL_LABELS[ilm],
                                        ORBITAL_LABELS[jlm])
            print(header, file=handle)

            for _ in range(nzeros):
                print(' '.join(['0.0'] * table.shape[3]), file=handle)

            formats = ['%1.12e'] * Naux

            for iaux in range(Naux):
                label = aux_orbitals[iaux]
                klm = ORBITAL_LABELS.index(label)
                allzero = np.allclose(table[ilm, jlm, :, iaux], 0.)

                if not NONZERO_2CM[ilm, jlm, klm]:
                    assert allzero, (ilm, jlm, klm, iaux)

                if allzero:
                    formats[iaux] = '%.1f'

            np.savetxt(handle, table[ilm, jlm, :, :], fmt=formats)
    return
