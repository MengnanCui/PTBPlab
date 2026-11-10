#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.spherical_harmonics import sph_cartesian


def sph_solid_radial(r, l):
    """
    Returns the radial part of the selected (regular) solid harmonic.

    Parameters
    ----------
    r : float or np.ndarray
        Corresponding distances from the origin.
    l : int
        Angular momentum.

    Returns
    -------
    R : float or np.ndarray
        Function value(s).
    """
    R = np.sqrt(4 * np.pi / (2*l + 1)) * r**l
    return R


def sph_solid(x, y, z, r, lm):
    """
    Returns the value of the selected (regular) solid harmonic.

    Parameters
    ----------
    x, y, z : float or np.ndarray
        Cartesian coordinates.
    r : float or np.ndarray
        Corresponding distances from the origin.
    lm : str
        Orbital label (e.g. 'px').

    Returns
    -------
    C : float or np.ndarray
        Function value(s).
    """
    l = ANGULAR_MOMENTUM[lm[0]]
    R = sph_solid_radial(r, l)
    Y = sph_cartesian(x, y, z, r, lm)
    C = R * Y
    return C
