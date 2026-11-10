#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np


def sph_nophi(lm, c, s):
    """ Evaluates the phi-independent part of the chosen spherical harmonic,
    as function of the cosine and sine of the theta angle.
    """
    if lm == 's':
        return 0.5 / np.sqrt(np.pi)
    elif lm == 'px':
        return 0.5 * np.sqrt(3. / np.pi) * s
    elif lm == 'py':
        return 0.5 * np.sqrt(3. / np.pi) * s
    elif lm == 'pz':
        return 0.5 * np.sqrt(3. / np.pi) * c
    elif lm == 'dxy':
        return 0.25 * np.sqrt(15. / np.pi) * s**2
    elif lm == 'dyz':
        return 0.5 * np.sqrt(15. / np.pi) * s * c
    elif lm == 'dxz':
        return 0.5 * np.sqrt(15. / np.pi) * s * c
    elif lm == 'dx2-y2':
        return 0.25 * np.sqrt(15. / np.pi) * s**2
    elif lm == 'dz2':
        return 0.25 * np.sqrt(5. / np.pi) * (3*c**2 - 1.)


def sph_nophi_der(lm, c, s):
    """
    Evaluates the [c, s]-derivatives of the phi-independent part
    of the chosen spherical harmonic, as function of the cosine and
    sine of the theta angle.

    Parameters
    ----------
    lm : str
        Orbital label (e.g. 'px').
    c : float or np.ndarray
        Cosine of theta angle.
    s : float or np.ndarray
        Sine of theta angle.

    Returns
    -------
    derivs : list of float or list of np.ndarray
        List with the derivatives with respect to c and s, respectively.
    """
    if lm == 's':
        derivs = [0., 0.]
    elif lm == 'px':
        derivs = [0., 0.5 * np.sqrt(3. / np.pi)]
    elif lm == 'py':
        derivs = [0., 0.5 * np.sqrt(3. / np.pi)]
    elif lm == 'pz':
        derivs = [0.5 * np.sqrt(3. / np.pi), 0.]
    elif lm == 'dxy':
        derivs = [0., 0.5 * np.sqrt(15. / np.pi) * s]
    elif lm == 'dyz':
        derivs = [0.5 * np.sqrt(15. / np.pi) * s,
                0.5 * np.sqrt(15. / np.pi) * c]
    elif lm == 'dxz':
        derivs = [0.5 * np.sqrt(15. / np.pi) * s,
                0.5 * np.sqrt(15. / np.pi) * c]
    elif lm == 'dx2-y2':
        derivs = [0., 0.5 * np.sqrt(15. / np.pi) * s]
    elif lm == 'dz2':
        derivs = [1.5 * np.sqrt(5. / np.pi) * c, 0.]
    return derivs


def sph_phi(lm, phi):
    """ Evaluates the phi-dependent part of the chosen spherical harmonic. """
    if lm == 's':
        return 1.
    elif lm == 'px':
        return np.cos(phi)
    elif lm == 'py':
        return np.sin(phi)
    elif lm == 'pz':
        return 1.
    elif lm == 'dxy':
        return np.sin(2*phi)
    elif lm == 'dyz':
        return np.sin(phi)
    elif lm == 'dxz':
        return np.cos(phi)
    elif lm == 'dx2-y2':
        return np.cos(2*phi)
    elif lm == 'dz2':
        return 1.


def sph_phi_der(lm, phi):
    """
    Evaluates the phi-derivative of the phi-dependent part
    of the chosen spherical harmonic.

    Parameters
    ----------
    lm : str
        Orbital label (e.g. 'px').
    phi : float or np.ndarray
        Phi angle.

    Returns
    -------
    deriv : float or np.ndarray
        Derivative with respect to phi.
    """
    if lm == 's':
        deriv = 0.
    elif lm == 'px':
        deriv = -np.sin(phi)
    elif lm == 'py':
        deriv = np.cos(phi)
    elif lm == 'pz':
        deriv = 0.
    elif lm == 'dxy':
        deriv = 2.*np.cos(2*phi)
    elif lm == 'dyz':
        deriv = np.cos(phi)
    elif lm == 'dxz':
        deriv = -np.sin(phi)
    elif lm == 'dx2-y2':
        deriv = -2.*np.sin(2*phi)
    elif lm == 'dz2':
        deriv = 0.
    return deriv


def sph(lm, c, s, phi):
    """ Evaluates the chosen spherical harmonic based on cos(theta),
    sin(theta) and phi.
    """
    return sph_nophi(lm, c, s) * sph_phi(lm, phi)


def sph_cartesian(x, y, z, r, lm):
    """
    Returns the value of the chosen spherical harmonic
    in cartesian coordinates.

    Parameters
    ----------
    x, y, z : float or np.ndarray
        Cartesian coordinates.
    r : float or np.ndarray
        Corresponding distances from the origin.
    lm : str
        Orbital label (e.g. 'px').
    """
    if lm == 's':
        return 0.5 / np.sqrt(np.pi)
    elif lm == 'px':
        return np.sqrt(3. / (4.*np.pi)) * x/r
    elif lm == 'py':
        return np.sqrt(3. / (4.*np.pi)) * y/r
    elif lm == 'pz':
        return np.sqrt(3. / (4.*np.pi)) * z/r
    elif lm == 'dxy':
        return np.sqrt(15. / (4.*np.pi)) * x*y/r**2
    elif lm == 'dyz':
        return np.sqrt(15. / (4.*np.pi)) * y*z/r**2
    elif lm == 'dxz':
        return np.sqrt(15. / (4.*np.pi)) * x*z/r**2
    elif lm == 'dx2-y2':
        return np.sqrt(15. / (16.*np.pi)) * (x**2-y**2)/r**2
    elif lm == 'dz2':
        return np.sqrt(5. / (16.*np.pi)) * (2*z**2-x**2-y**2)/r**2
    elif lm == 'fx(x2-3y2)':
        return np.sqrt(35. / (32.*np.pi)) * x*(x**2 - 3*y**2)/r**3
    elif lm == 'fy(3x2-y2)':
        return np.sqrt(35. / (32.*np.pi)) * y*(3*x**2 - y**2)/r**3
    elif lm == 'fz(x2-y2)':
        return np.sqrt(105. / (16.*np.pi)) * z*(x**2 - y**2)/r**3
    elif lm == 'fxyz':
        return np.sqrt(105. / (4.*np.pi)) * x*y*z/r**3
    elif lm == 'fyz2':
        return np.sqrt(21. / (32.*np.pi)) * y*(4*z**2 - x**2 - y**2)/r**3
    elif lm == 'fxz2':
        return np.sqrt(21. / (32.*np.pi)) * x*(4*z**2 - x**2 - y**2)/r**3
    elif lm == 'fz3':
        return np.sqrt(7. / (16.*np.pi)) * z*(2*z**2 - 3*x**2 - 3*y**2)/r**3
    else:
        raise NotImplementedError('Unknown orbital label: ' + lm)



def sph_cartesian_der(x, y, z, r, lm, der):
    """
    Returns the derivative of the chosen spherical harmonic
    in cartesian coordinates.

    Parameters
    ----------
    x, y, z : float or np.ndarray
        Cartesian coordinates.
    r : float or np.ndarray
        Corresponding distances from the origin.
    lm : str
        Orbital label (e.g. 'px').
    der : str
        Derivative to evaluate ('x', 'y' or 'z').
    """
    sqrtpi = np.sqrt(np.pi)

    if lm == "s":
        if der == "x":
            return 0
        elif der == "y":
            return 0
        elif der == "z":
            return 0
    elif lm == "pz":
        if der == "x":
            return -np.sqrt(3)*x*z/(2*sqrtpi*r**3)
        elif der == "y":
            return -np.sqrt(3)*y*z/(2*sqrtpi*r**3)
        elif der == "z":
            return -np.sqrt(3)*z**2/(2*sqrtpi*r**3) \
                   + np.sqrt(3)/(2*sqrtpi*r)
    elif lm == "py":
        if der == "x":
            return -np.sqrt(3)*x*y/(2*sqrtpi*r**3)
        elif der == "y":
            return -np.sqrt(3)*y**2/(2*sqrtpi*r**3) \
                   + np.sqrt(3)/(2*sqrtpi*r)
        elif der == "z":
            return -np.sqrt(3)*y*z/(2*sqrtpi*r**3)
    elif lm == "px":
        if der == "x":
            return -np.sqrt(3)*x**2/(2*sqrtpi*r**3) \
                   + np.sqrt(3)/(2*sqrtpi*r)
        elif der == "y":
            return -np.sqrt(3)*x*y/(2*sqrtpi*r**3)
        elif der == "z":
            return -np.sqrt(3)*x*z/(2*sqrtpi*r**3)
    elif lm == "dz2":
        if der == "x":
            return -np.sqrt(5)*x*(-x**2 - y**2 + 2*z**2)/(2*sqrtpi*r**4) \
                        - np.sqrt(5)*x/(2*sqrtpi*r**2)
        elif der == "y":
            return -np.sqrt(5)*y*(-x**2 - y**2 + 2*z**2)/(2*sqrtpi*r**4) \
                        - np.sqrt(5)*y/(2*sqrtpi*r**2)
        elif der == "z":
            return -np.sqrt(5)*z*(-x**2 - y**2 + 2*z**2)/(2*sqrtpi*r**4) \
                        + np.sqrt(5)*z/(sqrtpi*r**2)
    elif lm == "dyz":
        if der == "x":
            return -np.sqrt(15)*x*y*z/(sqrtpi*r**4)
        elif der == "y":
            return -np.sqrt(15)*y**2*z/(sqrtpi*r**4) \
                        + np.sqrt(15)*z/(2*sqrtpi*r**2)
        elif der == "z":
            return -np.sqrt(15)*y*z**2/(sqrtpi*r**4) \
                        + np.sqrt(15)*y/(2*sqrtpi*r**2)
    elif lm == "dxz":
        if der == "x":
            return -np.sqrt(15)*x**2*z/(sqrtpi*r**4) \
                        + np.sqrt(15)*z/(2*sqrtpi*r**2)
        elif der == "y":
            return -np.sqrt(15)*x*y*z/(sqrtpi*r**4)
        elif der == "z":
            return -np.sqrt(15)*x*z**2/(sqrtpi*r**4) \
                        + np.sqrt(15)*x/(2*sqrtpi*r**2)
    elif lm == "dxy":
        if der == "x":
            return -np.sqrt(15)*x**2*y/(sqrtpi*r**4) \
                        + np.sqrt(15)*y/(2*sqrtpi*r**2)
        elif der == "y":
            return -np.sqrt(15)*x*y**2/(sqrtpi*r**4) \
                        + np.sqrt(15)*x/(2*sqrtpi*r**2)
        elif der == "z":
            return -np.sqrt(15)*x*y*z/(sqrtpi*r**4)
    elif lm == "dx2-y2":
        if der == "x":
            return -np.sqrt(15)*x*(x**2 - y**2)/(2*sqrtpi*r**4) \
                        + np.sqrt(15)*x/(2*sqrtpi*r**2)
        elif der == "y":
            return -np.sqrt(15)*y*(x**2 - y**2)/(2*sqrtpi*r**4) \
                        - np.sqrt(15)*y/(2*sqrtpi*r**2)
        elif der == "z":
            return -np.sqrt(15)*z*(x**2 - y**2)/(2*sqrtpi*r**4)
    elif lm == "fx(x2-3y2)":
        if der == "x":
            return -3*np.sqrt(70)*x**2*(x**2 - 3*y**2)/(8*sqrtpi*r**5) \
                        + np.sqrt(70)*x**2/(4*sqrtpi*r**3) \
                        + np.sqrt(70)*(x**2 - 3*y**2)/(8*sqrtpi*r**3)
        elif der == "y":
            return -3*np.sqrt(70)*x*y*(x**2 - 3*y**2)/(8*sqrtpi*r**5) \
                        - 3*np.sqrt(70)*x*y/(4*sqrtpi*r**3)
        elif der == "z":
            return -3*np.sqrt(70)*x*z*(x**2 - 3*y**2)/(8*sqrtpi*r**5)
    elif lm == "fy(3x2-y2)":
        if der == "x":
            return -3*np.sqrt(70)*x*y*(3*x**2 - y**2)/(8*sqrtpi*r**5) \
                        + 3*np.sqrt(70)*x*y/(4*sqrtpi*r**3)
        elif der == "y":
            return -3*np.sqrt(70)*y**2*(3*x**2 - y**2)/(8*sqrtpi*r**5) \
                        - np.sqrt(70)*y**2/(4*sqrtpi*r**3) \
                        + np.sqrt(70)*(3*x**2 - y**2)/(8*sqrtpi*r**3)
        elif der == "z":
            return -3*np.sqrt(70)*y*z*(3*x**2 - y**2)/(8*sqrtpi*r**5)
    elif lm == "fz(x2-y2)":
        if der == "x":
            return -3*np.sqrt(105)*x*z*(x**2 - y**2)/(4*sqrtpi*r**5) \
                        + np.sqrt(105)*x*z/(2*sqrtpi*r**3)
        elif der == "y":
            return -3*np.sqrt(105)*y*z*(x**2 - y**2)/(4*sqrtpi*r**5) \
                        - np.sqrt(105)*y*z/(2*sqrtpi*r**3)
        elif der == "z":
            return -3*np.sqrt(105)*z**2*(x**2 - y**2)/(4*sqrtpi*r**5) \
                        + np.sqrt(105)*(x**2 - y**2)/(4*sqrtpi*r**3)
    elif lm == "fxyz":
        if der == "x":
            return -3*np.sqrt(105)*x**2*y*z/(2*sqrtpi*r**5) \
                        + np.sqrt(105)*y*z/(2*sqrtpi*r**3)
        elif der == "y":
            return -3*np.sqrt(105)*x*y**2*z/(2*sqrtpi*r**5) \
                        + np.sqrt(105)*x*z/(2*sqrtpi*r**3)
        elif der == "z":
            return -3*np.sqrt(105)*x*y*z**2/(2*sqrtpi*r**5) \
                        + np.sqrt(105)*x*y/(2*sqrtpi*r**3)
    elif lm == "fyz2":
        if der == "x":
            return -3*np.sqrt(42)*x*y*(-x**2 - y**2 + 4*z**2) \
                        / (8*sqrtpi*r**5) - np.sqrt(42)*x*y/(4*sqrtpi*r**3)
        elif der == "y":
            return -3*np.sqrt(42)*y**2*(-x**2 - y**2 + 4*z**2) \
                        / (8*sqrtpi*r**5) - np.sqrt(42)*y**2/(4*sqrtpi*r**3) \
                        + np.sqrt(42)*(-x**2 - y**2 + 4*z**2)/(8*sqrtpi*r**3)
        elif der == "z":
            return -3*np.sqrt(42)*y*z*(-x**2 - y**2 + 4*z**2) \
                        / (8*sqrtpi*r**5) + np.sqrt(42)*y*z/(sqrtpi*r**3)
    elif lm == "fxz2":
        if der == "x":
            return -3*np.sqrt(42)*x**2*(-x**2 - y**2 + 4*z**2) \
                        / (8*sqrtpi*r**5) - np.sqrt(42)*x**2/(4*sqrtpi*r**3) \
                        + np.sqrt(42)*(-x**2 - y**2 + 4*z**2)/(8*sqrtpi*r**3)
        elif der == "y":
            return -3*np.sqrt(42)*x*y*(-x**2 - y**2 + 4*z**2) \
                        / (8*sqrtpi*r**5) - np.sqrt(42)*x*y/(4*sqrtpi*r**3)
        elif der == "z":
            return -3*np.sqrt(42)*x*z*(-x**2 - y**2 + 4*z**2) \
                        / (8*sqrtpi*r**5) + np.sqrt(42)*x*z/(sqrtpi*r**3)
    elif lm == "fz3":
        if der == "x":
            return -3*np.sqrt(7)*x*z*(-3*x**2 - 3*y**2 + 2*z**2) \
                        / (4*sqrtpi*r**5) - 3*np.sqrt(7)*x*z/(2*sqrtpi*r**3)
        elif der == "y":
            return -3*np.sqrt(7)*y*z*(-3*x**2 - 3*y**2 + 2*z**2) \
                        / (4*sqrtpi*r**5) - 3*np.sqrt(7)*y*z/(2*sqrtpi*r**3)
        elif der == "z":
            return -3*np.sqrt(7)*z**2*(-3*x**2 - 3*y**2 + 2*z**2) \
                        / (4*sqrtpi*r**5) + np.sqrt(7)*z**2/(sqrtpi*r**3) \
                        + np.sqrt(7)*(-3*x**2 - 3*y**2 + 2*z**2)/(4*sqrtpi*r**3)
    else:
        raise NotImplementedError('Unknown orbital label: ' + lm)

def sph_cartesian_der2(x, y, z, r, lm, der):
    """
    Returns the second derivative of the chosen spherical harmonic
    in cartesian coordinates.

    Parameters
    ----------
    x, y, z : float or np.ndarray
        Cartesian coordinates.
    r : float or np.ndarray
        Corresponding distances from the origin.
    lm : str
        Orbital label (e.g. 'px').
    der : str
        Derivative to evaluate ('x', 'y' or 'z').
    """
    sqrtpi = np.sqrt(np.pi)

    if lm == "s":
        if der == "x":
            return 0
        elif der == "y":
            return 0
        elif der == "z":
            return 0
    elif lm == "pz":
        if der == "x":
            return np.sqrt(3)*z*(3*x**2/r**2 - 1)/(2*sqrtpi*r**3)
        elif der == "y":
            return np.sqrt(3)*z*(3*y**2/r**2 - 1)/(2*sqrtpi*r**3)
        elif der == "z":
            return 3*np.sqrt(3)*z*(z**2/r**2 - 1)/(2*sqrtpi*r**3)
    elif lm == "py":
        if der == "x":
            return np.sqrt(3)*y*(3*x**2/r**2 - 1)/(2*sqrtpi*r**3)
        elif der == "y":
            return 3*np.sqrt(3)*y*(y**2/r**2 - 1)/(2*sqrtpi*r**3)
        elif der == "z":
            return np.sqrt(3)*y*(3*z**2/r**2 - 1)/(2*sqrtpi*r**3)
    elif lm == "px":
        if der == "x":
            return 3*np.sqrt(3)*x*(x**2/r**2 - 1)/(2*sqrtpi*r**3)
        elif der == "y":
            return np.sqrt(3)*x*(3*y**2/r**2 - 1)/(2*sqrtpi*r**3)
        elif der == "z":
            return np.sqrt(3)*x*(3*z**2/r**2 - 1)/(2*sqrtpi*r**3)
    elif lm == "dz2":
        if der == "x":
            return np.sqrt(5)*(-4*x**2*(x**2 + y**2 - 2*z**2)/r**4 \
                        + 4*x**2/r**2 + (x**2 + y**2 - 2*z**2)/r**2 - 1) \
                        / (2*sqrtpi*r**2)
        elif der == "y":
            return np.sqrt(5)*(-4*y**2*(x**2 + y**2 - 2*z**2)/r**4 \
                        + 4*y**2/r**2 + (x**2 + y**2 - 2*z**2)/r**2 - 1) \
                        / (2*sqrtpi*r**2)
        elif der == "z":
            return np.sqrt(5)*(-4*z**2*(x**2 + y**2 - 2*z**2)/r**4 \
                        - 8*z**2/r**2 + (x**2 + y**2 - 2*z**2)/r**2 + 2) \
                        / (2*sqrtpi*r**2)
    elif lm == "dyz":
        if der == "x":
            return np.sqrt(15)*y*z*(4*x**2/r**2 - 1)/(sqrtpi*r**4)
        elif der == "y":
            return np.sqrt(15)*y*z*(4*y**2/r**2 - 3)/(sqrtpi*r**4)
        elif der == "z":
            return np.sqrt(15)*y*z*(4*z**2/r**2 - 3)/(sqrtpi*r**4)
    elif lm == "dxz":
        if der == "x":
            return np.sqrt(15)*x*z*(4*x**2/r**2 - 3)/(sqrtpi*r**4)
        elif der == "y":
            return np.sqrt(15)*x*z*(4*y**2/r**2 - 1)/(sqrtpi*r**4)
        elif der == "z":
            return np.sqrt(15)*x*z*(4*z**2/r**2 - 3)/(sqrtpi*r**4)
    elif lm == "dxy":
        if der == "x":
            return np.sqrt(15)*x*y*(4*x**2/r**2 - 3)/(sqrtpi*r**4)
        elif der == "y":
            return np.sqrt(15)*x*y*(4*y**2/r**2 - 3)/(sqrtpi*r**4)
        elif der == "z":
            return np.sqrt(15)*x*y*(4*z**2/r**2 - 1)/(sqrtpi*r**4)
    elif lm == "dx2-y2":
        if der == "x":
            return np.sqrt(15)*(4*x**2*(x**2 - y**2)/r**4 - 4*x**2/r**2 \
                        - (x**2 - y**2)/r**2 + 1)/(2*sqrtpi*r**2)
        elif der == "y":
            return np.sqrt(15)*(4*y**2*(x**2 - y**2)/r**4 + 4*y**2/r**2 \
                        - (x**2 - y**2)/r**2 - 1)/(2*sqrtpi*r**2)
        elif der == "z":
            return np.sqrt(15)*(x**2 - y**2)*(4*z**2/r**2 - 1)/(2*sqrtpi*r**4)
    elif lm == "fx(x2-3y2)":
        if der == "x":
            return 3*np.sqrt(70)*x*(-4*x**2/r**2 \
                        + (x**2 - 3*y**2)*(5*x**2/r**2 - 1)/r**2 \
                        - 2*(x**2 - 3*y**2)/r**2 + 2)/(8*sqrtpi*r**3)
        elif der == "y":
            return 3*np.sqrt(70)*x*(12*y**2/r**2 \
                        + (x**2 - 3*y**2)*(5*y**2/r**2 - 1)/r**2 - 2) \
                        / (8*sqrtpi*r**3)
        elif der == "z":
            return 3*np.sqrt(70)*x*(x**2 - 3*y**2)*(5*z**2/r**2 - 1) \
                        / (8*sqrtpi*r**5)
    elif lm == "fy(3x2-y2)":
        if der == "x":
            return 3*np.sqrt(70)*y*(-12*x**2/r**2 + (3*x**2 - y**2) \
                        * (5*x**2/r**2 - 1)/r**2 + 2)/(8*sqrtpi*r**3)
        elif der == "y":
            return 3*np.sqrt(70)*y*(4*y**2/r**2 + (3*x**2 - y**2) \
                        * (5*y**2/r**2 - 1)/r**2 - 2*(3*x**2 - y**2)/r**2 - 2) \
                        / (8*sqrtpi*r**3)
        elif der == "z":
            return 3*np.sqrt(70)*y*(3*x**2 - y**2)*(5*z**2/r**2 - 1) \
                        / (8*sqrtpi*r**5)
    elif lm == "fz(x2-y2)":
        if der == "x":
            return np.sqrt(105)*z*(-3*x**2/r**2 + 3*(x**2 - y**2) \
                        * (5*x**2/r**2 - 1)/(4*r**2) + 1/2)/(sqrtpi*r**3)
        elif der == "y":
            return np.sqrt(105)*z*(3*y**2/r**2 + 3*(x**2 - y**2) \
                        * (5*y**2/r**2 - 1)/(4*r**2) - 1/2)/(sqrtpi*r**3)
        elif der == "z":
            return 3*np.sqrt(105)*z*(x**2 - y**2)*(5*z**2/r**2 - 3) \
                        / (4*sqrtpi*r**5)
    elif lm == "fxyz":
        if der == "x":
            return 3*np.sqrt(105)*x*y*z*(5*x**2/r**2 - 3)/(2*sqrtpi*r**5)
        elif der == "y":
            return 3*np.sqrt(105)*x*y*z*(5*y**2/r**2 - 3)/(2*sqrtpi*r**5)
        elif der == "z":
            return 3*np.sqrt(105)*x*y*z*(5*z**2/r**2 - 3)/(2*sqrtpi*r**5)
    elif lm == "fyz2":
        if der == "x":
            return np.sqrt(42)*y*(12*x**2/r**2 - 3*(5*x**2/r**2 - 1) \
                        * (x**2 + y**2 - 4*z**2)/r**2 - 2)/(8*sqrtpi*r**3)
        elif der == "y":
            return 3*np.sqrt(42)*y*(4*y**2/r**2 - (5*y**2/r**2 - 1) \
                        * (x**2 + y**2 - 4*z**2)/r**2 \
                        + 2*(x**2 + y**2 - 4*z**2)/r**2 - 2)/(8*sqrtpi*r**3)
        elif der == "z":
            return np.sqrt(42)*y*(-6*z**2/r**2 - 3*(5*z**2/r**2 - 1) \
                        * (x**2 + y**2 - 4*z**2)/(8*r**2) + 1)/(sqrtpi*r**3)
    elif lm == "fxz2":
        if der == "x":
            return 3*np.sqrt(42)*x*(4*x**2/r**2 - (5*x**2/r**2 - 1) \
                        * (x**2 + y**2 - 4*z**2)/r**2 \
                        + 2*(x**2 + y**2 - 4*z**2)/r**2 - 2)/(8*sqrtpi*r**3)
        elif der == "y":
            return np.sqrt(42)*x*(12*y**2/r**2 - 3*(5*y**2/r**2 - 1) \
                        * (x**2 + y**2 - 4*z**2)/r**2 - 2)/(8*sqrtpi*r**3)
        elif der == "z":
            return np.sqrt(42)*x*(-6*z**2/r**2 - 3*(5*z**2/r**2 - 1) \
                        * (x**2 + y**2 - 4*z**2)/(8*r**2) + 1)/(sqrtpi*r**3)
    elif lm == "fz3":
        if der == "x":
            return 3*np.sqrt(7)*z*(3*x**2/r**2 - (5*x**2/r**2 - 1) \
                        * (3*x**2 + 3*y**2 - 2*z**2)/(4*r**2) - 1/2) \
                        / (sqrtpi*r**3)
        elif der == "y":
            return 3*np.sqrt(7)*z*(3*y**2/r**2 - (5*y**2/r**2 - 1) \
                        * (3*x**2 + 3*y**2 - 2*z**2)/(4*r**2) - 1/2) \
                        /(sqrtpi*r**3)
        elif der == "z":
            return 3*np.sqrt(7)*z*(-2*z**2/r**2 - (5*z**2/r**2 - 1) \
                        * (3*x**2 + 3*y**2 - 2*z**2)/(4*r**2) + 1 \
                        + (3*x**2 + 3*y**2 - 2*z**2)/(2*r**2))/(sqrtpi*r**3)
    else:
        raise NotImplementedError('Unknown orbital label: ' + lm)
