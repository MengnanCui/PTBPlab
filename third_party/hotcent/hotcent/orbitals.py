#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np


ANGULAR_MOMENTUM = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}

ORBITALS = {
    0: ('s', ),
    1: ('px', 'py', 'pz'),
    2: ('dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2'),
    3: ('fx(x2-3y2)', 'fy(3x2-y2)', 'fz(x2-y2)', 'fxyz', 'fyz2', 'fxz2', 'fz3'),
    4: tuple(['g{0}'.format(i) for i in range(1, 10)]),
    5: tuple(['h{0}'.format(i) for i in range(1, 12)]),
}

ORBITAL_LABELS = [lm for l in range(6) for lm in ORBITALS[l]]


def calculate_slako_coeff(x, y, z, ilm, jlm, tau):
    # @NOTE Only calculates the values for which jlm >= ilm (otherwise c = 0).
    # For jlm < ilm, multiply the result by \( (-1)^{(\ell_i + \ell_j)} \).

    c = 0.

    if (ilm == 1):
        if (jlm == 1):
            if (tau == 1):
                # s s sigma
                c = 1.0
        elif (jlm == 2):
            if (tau == 1):
                # s px sigma
                c = x
        elif (jlm == 3):
            if (tau == 1):
                # s py sigma
                c = y
        elif (jlm == 4):
            if (tau == 1):
                # s pz sigma
                c = z
        elif (jlm == 5):
            if (tau == 1):
                # s dxy sigma
                c = np.sqrt(3) * x * y
        elif (jlm == 6):
            if (tau == 1):
                # s dyz sigma
                c = np.sqrt(3) * y * z
        elif (jlm == 7):
            if (tau == 1):
                # s dxz sigma
                c = np.sqrt(3) * x * z
        elif (jlm == 8):
            if (tau == 1):
                # s dx2-y2 sigma
                c = 0.5 * np.sqrt(3) * (x**2 - y**2)
        elif (jlm == 9):
            if (tau == 1):
                # s dz2 sigma
                c = z**2 - 0.5 * (x**2 + y**2)
        elif (jlm == 10):
            if (tau == 1):
                # s fx(x2-3y2) sigma
                c = np.sqrt(2)*x*(4.0*x**2-3.0+3.0*z**2)*np.sqrt(5)/4.0
        elif (jlm == 11):
            if (tau == 1):
                # s fy(3x2-y2) sigma
                c = np.sqrt(2)*y*(4.0*x**2-1.0+z**2)*np.sqrt(5)/ 4.0
        elif (jlm == 12):
            if (tau == 1):
                # s fz(x2-y2) sigma
                c = (2.0*x**2-1.0+z**2)*np.sqrt(15)*z/ 2.0
        elif (jlm == 13):
            if (tau == 1):
                # s fxyz sigma
                c = x*y*np.sqrt(15)*z
        elif (jlm == 14):
            if (tau == 1):
                # s fyz2 sigma
                c = np.sqrt(2)*y*np.sqrt(3)*(5.0*z**2-1.0)/ 4.0
        elif (jlm == 15):
            if (tau == 1):
                # s fxz2 sigma
                c = np.sqrt(2)*x*np.sqrt(3)*(5.0*z**2-1.0)/4.0
        elif (jlm == 16):
            if (tau == 1):
                # s fz3 sigma
                c = (z*(5.0*z**2-3.0))/ 2.0
    elif (ilm == 2):
        if (jlm == 2):
            if (tau == 1):
                # px px sigma
                c = x**2
            elif (tau == 2):
                # px px pi
                c = 1. - x**2
        elif (jlm == 3):
            if (tau == 1):
                # px py sigma
                c = x * y
            elif (tau == 2):
                # px py pi
                c = -x * y
        elif (jlm == 4):
            if (tau == 1):
                # px pz sigma
                c = x * z
            elif (tau == 2):
                # px pz pi
                c = -x * z
        elif (jlm == 5):
            if (tau == 1):
                # px dxy sigma
                c = np.sqrt(3) * x**2 * y
            elif (tau == 2):
                # px dxy pi
                c = y * (1. - 2. * x ** 2)
        elif (jlm == 6):
            if (tau == 1):
                # px dyz sigma
                c = np.sqrt(3) * x * y * z
            elif (tau == 2):
                # px dyz pi
                c = -2. * x * y * z
        elif (jlm == 7):
            if (tau == 1):
                # px dxz sigma
                c = np.sqrt(3) * x**2 * z
            elif (tau == 2):
                # px dxz pi
                c = z * (1. - 2. * x ** 2)
        elif (jlm == 8):
            if (tau == 1):
                # px dx2-y2 sigma
                c = 0.5 * np.sqrt(3) * x * (x**2 - y**2)
            elif (tau == 2):
                # px dx2-y2 pi
                c = x * (1. - (x**2 - y**2))
        elif (jlm == 9):
            if (tau == 1):
                # px dz2 sigma
                c = x * (z**2 - 0.5 * (x**2 + y**2))
            elif (tau == 2):
                # px dz2 pi
                c = -np.sqrt(3) * x * z**2
        elif (jlm == 10):
            if (tau == 1):
                # px fx(x2-3y2) sigma
                c = (x**2)*(4.0*x**2-3.0+3.0*z**2)*np.sqrt(2)*np.sqrt(5)/4.0
            elif (tau == 2):
                # px fx(x2-3y2) pi
                c = -np.sqrt(15)*(3.0*x**2*z**2-z**2-5.0*x**2+4.0*x**4+1.0)/4.0
        elif (jlm == 11):
            if (tau == 1):
                # px fy(3x2-y2) sigma
                c = x*y*(4.0*x**2-1.0+z**2)*np.sqrt(2)*np.sqrt(5)/4.0
            elif (tau == 2):
                # px fy(3x2-y2) pi
                c = -x*y*np.sqrt(15)*(z**2+4.0*x**2-3.0)/4.0
        elif (jlm == 12):
            if (tau == 1):
                # px fz(x2-y2) sigma
                c = x*(2.0*x**2-1.0+z**2)*np.sqrt(15)*z/2.0
            elif (tau == 2):
                # px fz(x2-y2) pi
                c = -(3.0*z**2+6.0*x**2-5.0)*z*x*np.sqrt(10)/4.0
        elif (jlm == 13):
            if (tau == 1):
                # px fxyz sigma
                c = (x**2)*y*np.sqrt(15)*z
            elif (tau == 2):
                # px fxyz pi
                c = -(3.0*x**2-1.0)*z*y*np.sqrt(10)/2.0
        elif (jlm == 14):
            if (tau == 1):
                # px fyz2 sigma
                c = x*y*np.sqrt(2)*np.sqrt(3)*(5.0*z**2-1.0)/4.0
            elif (tau == 2):
                # px fyz2 pi
                c = -(15.0*z**2-1.0)*x*y/4.0
        elif (jlm == 15):
            if (tau == 1):
                # px fxz2 sigma
                c = x**2*np.sqrt(2)*np.sqrt(3)*(5.0*z**2-1.0)/4.0
            elif (tau == 2):
                # px fxz2 pi
                c = (-15.0/4.0*x**2*(z**2)+5.0/4.0*(z**2)- 1.0/4.0+x**2/4.0)
        elif (jlm == 16):
            if (tau == 1):
                # px fz3 sigma
                c = (x*z*(5.0*z**2-3.0))/2.0
            elif (tau == 2):
                # px fz3 pi
                c = -(5.0*z**2-1.0)*np.sqrt(3)*np.sqrt(2)*z*x/4.0
    elif (ilm == 3):
        if (jlm == 3):
            if (tau == 1):
                # py py sigma
                c = y**2
            elif (tau == 2):
                # py py pi
                c = 1. - y**2
        elif (jlm == 4):
            if (tau == 1):
                # py pz sigma
                c = y * z
            elif (tau == 2):
                # py pz pi
                c = -y * z
        elif (jlm == 5):
            if (tau == 1):
                # py dxy sigma
                c = np.sqrt(3) * x * y**2
            elif (tau == 2):
                # py dxy pi
                c = x * (1. - 2. * y**2)
        elif (jlm == 6):
            if (tau == 1):
                # py dyz sigma
                c = np.sqrt(3) * z * y**2
            elif (tau == 2):
                # py dyz pi
                c = z * (1. - 2. * y**2)
        elif (jlm == 7):
            if (tau == 1):
                # py dxz sigma
                c = np.sqrt(3) * x * y * z
            elif (tau == 2):
                # py dxz pi
                c = -2. * x * y * z
        elif (jlm == 8):
            if (tau == 1):
                # py dx2-y2 sigma
                c = 0.5 * np.sqrt(3) * y * (x**2 - y**2)
            elif (tau == 2):
                # py dx2-y2 pi
                c = -y * (1. + (x**2 - y**2))
        elif (jlm == 9):
            if (tau == 1):
                # py dz2 sigma
                c = y * (z**2 - 0.5 * (x**2 + y**2))
            elif (tau == 2):
                # py dz2 pi
                c = -np.sqrt(3) * y * z**2
        elif (jlm == 10):
            if (tau == 1):
                # py fx(x2-3y2) sigma
                c = y*x*(4.0*x**2-3.0+3.0*z**2)*np.sqrt(2)*np.sqrt(5)/4.0
            elif (tau == 2):
                # py fx(x2-3y2) pi
                c = -x*y*np.sqrt(15)*(3.0*z**2+4.0*x**2-1.0)/4.0
        elif (jlm == 11):
            if (tau == 1):
                # py fy(3x2-y2) sigma
                c = -(-1.0+z**2+x**2)*(4.0*x**2-1.0+z**2)*np.sqrt(2)*np.sqrt(5)/4.0
            elif (tau == 2):
                # py fy(3x2-y2) pi
                c = np.sqrt(15)*(z**4-z**2+5.0*z**2*x**2-3.0*x**2+4.0*x**4)/4.0
        elif (jlm == 12):
            if (tau == 1):
                # py fz(x2-y2) sigma
                c = y*(2.0*x**2-1.0+z**2)*np.sqrt(15)*z/2.0
            elif (tau == 2):
                # py fz(x2-y2) pi
                c = -(3.0*z**2+6.0*x**2-1.0)*z*y*np.sqrt(10)/4.0
        elif (jlm == 13):
            if (tau == 1):
                # py fxyz sigma
                c = -(-1.0+z**2+x**2)*x*np.sqrt(15)*z
            elif (tau == 2):
                # py fxyz pi
                c = (3.0*z**2+3.0*x**2-2.0)*x*z*np.sqrt(10)/2.0
        elif (jlm == 14):
            if (tau == 1):
                # py fyz2 sigma
                c = -(-1.0+z**2+x**2)*np.sqrt(2)*np.sqrt(3)*(5.0*z**2-1.0)/4.0
            elif (tau == 2):
                # py fyz2 pi
                c = (15.0/4.0*(z**4)+ 15.0/ 4.0*(x**2)*(z**2)- 11.0/4.0*(z**2)-(x**2)/ 4.0)
        elif (jlm == 15):
            if (tau == 1):
                # py fxz2 sigma
                c = y*x*np.sqrt(2)*np.sqrt(3)*(5.0*z**2-1.0)/4.0
            elif (tau == 2):
                # py fxz2 pi
                c = -(15.0*z**2-1.0)*x*y/4.0
        elif (jlm == 16):
            if (tau == 1):
                # py fz3 sigma
                c = y*z*(5.0*z**2-3.0)/2.0
            elif (tau == 2):
                # py fz3 pi
                c = -(5.0*z**2-1.0)*np.sqrt(3)*np.sqrt(2)*z*y/4.0
    elif (ilm == 4):
        if (jlm == 4):
            if (tau == 1):
                # pz pz sigma
                c = z ** 2
            elif (tau == 2):
                # pz pz pi
                c = 1. - z ** 2
        elif (jlm == 5):
            if (tau == 1):
                # pz dxy sigma
                c = np.sqrt(3) * x * y * z
            elif (tau == 2):
                # pz dxy pi
                c = -2. * x *y * z
        elif (jlm == 6):
            if (tau == 1):
                # pz dyz sigma
                c = np.sqrt(3) * y * z**2
            elif (tau == 2):
                # pz dyz pi
                c = y * (1. - 2. * z**2)
        elif (jlm == 7):
            if (tau == 1):
                # pz dxz sigma
                c = np.sqrt(3) * x * z**2
            elif (tau == 2):
                # pz dxz pi
                c = x * (1. - 2. * z**2)
        elif (jlm == 8):
            if (tau == 1):
                # pz dx2-y2 sigma
                c = 0.5 * np.sqrt(3) * z * (x**2 - y**2)
            elif (tau == 2):
                # pz dx2-y2 pi
                c = -z * (x**2 - y**2)
        elif (jlm == 9):
            if (tau == 1):
                # pz dz2 sigma
                c = z * (z**2 - 0.5 * (x**2 + y**2))
            elif (tau == 2):
                # pz dz2 pi
                c = np.sqrt(3) * z * (x**2 + y**2)
        elif (jlm == 10):
            if (tau == 1):
                # pz fx(x2-3y2) sigma
                c = np.sqrt(2)*x*(4.0*x**2-3.0+3.0*z**2)*z*np.sqrt(5)/4.0
            elif (tau == 2):
                # pz fx(x2-3y2) pi
                c = -x*(4.0*x**2-3.0+3.0*z**2)*np.sqrt(15)*z/4.0
        elif (jlm == 11):
            if (tau == 1):
                # pz fy(3x2-y2) sigma
                c = np.sqrt(2)*y*(4.0*x**2-1.0+z**2)*z*np.sqrt(5)/4.0
            elif (tau == 2):
                # pz fy(3x2-y2) pi
                c = -y*(4.0*x**2-1.0+z**2)*np.sqrt(15)*z/4.0
        elif (jlm == 12):
            if (tau == 1):
                # pz fz(x2-y2) sigma
                c = (2.0*x**2-1.0+z**2)*(z**2)*np.sqrt(15)/2.0
            elif (tau == 2):
                # pz fz(x2-y2) pi
                c = -(3.0*z**2-1.0)*(2.0*x**2-1.0+z**2)*np.sqrt(10)/4.0
        elif (jlm == 13):
            if (tau == 1):
                # pz fxyz sigma
                c = x*y*(z**2)*np.sqrt(15)
            elif (tau == 2):
                # pz fxyz pi
                c = -(3.0*z**2-1.0)*x*y*np.sqrt(10)/2.0
        elif (jlm == 14):
            if (tau == 1):
                # pz fyz2 sigma
                c = np.sqrt(2)*y*z*np.sqrt(3)*(5.0*z**2-1.0)/4.0
            elif (tau == 2):
                # pz fyz2 pi
                c = -(15.0*z**2-11.0)*z*y/4.0
        elif (jlm == 15):
            if (tau == 1):
                # pz fxz2 sigma
                c = np.sqrt(2)*x*z*np.sqrt(3)*(5.0*z**2-1.0)/4.0
            elif (tau == 2):
                # pz fxz2 pi
                c = -(15.0*z**2-11.0)*z*x/4.0
        elif (jlm == 16):
            if (tau == 1):
                # pz fz3 sigma
                c = (z**2*(5.0*z**2-3.0))/2.0
            elif (tau == 2):
                # pz fz3 pi
                c = -(5.0*z**2-1.0)*np.sqrt(3)*np.sqrt(2)*(-1.0+z**2)/4.0
    elif (ilm == 5):
        if (jlm == 5):
            if (tau == 1):
                # dxy dxy sigma
                c = 3. * x**2 * y**2
            elif (tau == 2):
                # dxy dxy pi
                c = (x**2 + y**2) - 4. * x**2 * y**2
            elif (tau == 3):
                # dxy dxy delta
                c = z**2 + x**2 * y**2
        elif (jlm == 6):
            if (tau == 1):
                # dxy dyz sigma
                c = 3. * x * y**2 * z
            elif (tau == 2):
                # dxy dyz pi
                c = x * z * (1. - 4. * y**2)
            elif (tau == 3):
                # dxy dyz delta
                c = x * z * (y**2 - 1.)
        elif (jlm == 7):
            if (tau == 1):
                # dxy dxz sigma
                c = 3. * x**2 * y * z
            elif (tau == 2):
                # dxy dxz pi
                c = y * z * (1. - 4. * x**2)
            elif (tau == 3):
                # dxy dxz delta
                c = y * z * (x**2 - 1.)
        elif (jlm == 8):
            if (tau == 1):
                # dxy dx2-y2 sigma
                c = 1.5 * x * y * (x**2 - y**2)
            elif (tau == 2):
                # dxy dx2-y2 pi
                c = -2. * x * y * (x**2 - y**2)
            elif (tau == 3):
                # dxy dx2-y2 delta
                c = 0.5 * x * y * (x**2 - y**2)
        elif (jlm == 9):
            if (tau == 1):
                # dxy dz2 sigma
                c = np.sqrt(3) * x * y * (z**2 - 0.5 * (x**2 + y**2))
            elif (tau == 2):
                # dxy dz2 pi
                c = -2. * np.sqrt(3) * x * y * z**2
            elif (tau == 3):
                # dxy dz2 delta
                c = 0.5 * np.sqrt(3) * x * y * (1. + z**2)
        elif (jlm == 10):
            if (tau == 1):
                # dxy fx(x2-3y2) sigma
                c = (x**2)*y*(4.0*x**2-3.0+3.0*z**2)*np.sqrt(6)*np.sqrt(5)/ 4.0
            elif (tau == 2):
                # dxy fx(x2-3y2) pi
                c = -np.sqrt(15)*y*(6.0*x**2*z**2-z**2+1.0+8.0*x**4-6.0*x**2)/4.0
            elif (tau == 3):
                # dxy fx(x2-3y2) delta
                c = np.sqrt(6)*y*(3.0*x**2*z**2-2.0*z**2+4.0*x**4-3.0*x**2)/4.0
        elif (jlm == 11):
            if (tau == 1):
                # dxy fy(3x2-y2) sigma
                c = -x*(-1.0+z**2+x**2)*(4.0*x**2-1.0+z**2)*np.sqrt(6)*np.sqrt(5)/4.0
            elif (tau == 2):
                # dxy fy(3x2-y2) pi
                c = np.sqrt(15)*x*(2.0*z**4-5.0*z**2+10.0*x**2*z**2+3.0-10.0*x**2+8.0*x**4)/4.0
            elif (tau == 3):
                # dxy fy(3x2-y2) delta
                c = -np.sqrt(6)*x*(z**4+5.0*x**2*z**2-4.0*z**2+1.0+4.0*x**4-5.0*x**2)/4.0
        elif (jlm == 12):
            if (tau == 1):
                # dxy fz(x2-y2) sigma
                c = 3.0/2.0*x*y*(2.0*x**2-1.0+z**2)*np.sqrt(5)*z
            elif (tau == 2):
                # dxy fz(x2-y2) pi
                c = -3.0/2.0*z*np.sqrt(10)*y*x*(2.0*x**2-1.0+z**2)
            elif (tau == 3):
                # dxy fz(x2-y2) delta
                c = 3.0/2.0*x*y*(2.0*x**2-1.0+z**2)*z
        elif (jlm == 13):
            if (tau == 1):
                # dxy fxyz sigma
                c = -3.0*(x**2)*(-1.0+z**2+x**2)*np.sqrt(5)*z
            elif (tau == 2):
                # dxy fxyz pi
                c = (6.0*x**2*z**2-z**2+1.0+6.0*x**4-6.0*x**2)*np.sqrt(10)*z/2.0
            elif (tau == 3):
                # dxy fxyz delta
                c = -(z*(3.0*x**2*z**2-2.0*z**2+1.0-3.0*x**2+3.0*x**4))
        elif (jlm == 14):
            if (tau == 1):
                # dxy fyz2 sigma
                c = -3.0/4.0*x*(-1.0+z**2+x**2)*np.sqrt(2)*(5.0*z**2-1.0)
            elif (tau == 2):
                # dxy fyz2 pi
                c = ((30.0*z**4+30.0*x**2*z**2-27.0*z**2-2.0*x**2+1.0)*x)/4.0
            elif (tau == 3):
                # dxy fyz2 delta
                c = -x*np.sqrt(10)*(3.0*z**4+3.0*x**2*z**2+x**2-1.0)/4.0
        elif (jlm == 15):
            if (tau == 1):
                # dxy fxz2 sigma
                c = 3.0/4.0*(x**2)*y*np.sqrt(2)*(5.0*z**2-1.0)
            elif (tau == 2):
                # dxy fxz2 pi
                c = -(30.0*x**2*z**2-5.0*z**2-2.0*x**2+1.0)*y/4.0
            elif (tau == 3):
                # dxy fxz2 delta
                c = y*np.sqrt(10)*(3.0*x**2*z**2-2.0*z**2+x**2)/4.0
        elif (jlm == 16):
            if (tau == 1):
                # dxy fz3 sigma
                c = x*y*np.sqrt(3)*z*(5.0*z**2-3.0)/2.0
            elif (tau == 2):
                # dxy fz3 pi
                c = -(5.0*z**2-1.0)*z*x*y*np.sqrt(6)/2.0
            elif (tau == 3):
                # dxy fz3 delta
                c = x*y*(z**2+1.0)*np.sqrt(15)*z/2.0
    elif (ilm == 6):
        if (jlm == 6):
            if (tau == 1):
                # dyz dyz sigma
                c = 3. * y**2 * z**2
            elif (tau == 2):
                # dyz dyz pi
                c = y**2 + z**2 - 4. * y**2 * z**2
            elif (tau == 3):
                # dyz dyz delta
                c = x**2 + y**2 * z**2
        elif (jlm == 7):
            if (tau == 1):
                # dyz dxz sigma
                c = 3. * x * y * z**2
            elif (tau == 2):
                # dyz dxz pi
                c = x * y * (1. - 4. * z**2)
            elif (tau == 3):
                # dyz dxz delta
                c = x * y * (z**2 - 1.)
        elif (jlm == 8):
            if (tau == 1):
                # dyz dx2-y2 sigma
                c = 1.5 * y * z * (x**2 - y**2)
            elif (tau == 2):
                # dyz dx2-y2 pi
                c = -y * z * (1. + 2. * (x**2 - y**2))
            elif (tau == 3):
                # dyz dx2-y2 delta
                c = y * z * (1. + 0.5 * (x**2 - y**2))
        elif (jlm == 9):
            if (tau == 1):
                # dyz dz2 sigma
                c = np.sqrt(3) * y * z * (z**2 - 0.5 * (x**2 + y**2))
            elif (tau == 2):
                # dyz dz2 pi
                c = np.sqrt(3) * y * z * ((x**2 + y**2) - z**2)
            elif (tau == 3):
                # dyz dz2 delta
                c = -0.5 * np.sqrt(3) * y * z * (x**2 + y**2)
        elif (jlm == 10):
            if (tau == 1):
                # dyz fx(x2-3y2) sigma
                c = y*x*(4.0*x**2-3.0+3.0*z**2)*np.sqrt(6)*z*np.sqrt(5)/4.0
            elif (tau == 2):
                # dyz fx(x2-3y2) pi
                c = -y*x*np.sqrt(15)*z*(3.0*z**2-2.0+4.0*x**2)/2.0
            elif (tau == 3):
                # dyz fx(x2-3y2) delta
                c = np.sqrt(6)*y*x*z*(3.0*z**2+1.0+4.0*x**2)/4.0
        elif (jlm == 11):
            if (tau == 1):
                # dyz fy(3x2-y2) sigma
                c = -(-1.0+z**2+x**2)*(4.0*x**2-1.0+z**2)*np.sqrt(6)*z*np.sqrt(5)/4.0
            elif (tau == 2):
                # dyz fy(3x2-y2) pi
                c = np.sqrt(15)*z*(2.0*z**4-3.0*z**2+10.0*x**2*z**2+1.0+8.0*x**4-8.0*x**2)/4.0
            elif (tau == 3):
                # dyz fy(3x2-y2) delta
                c = -np.sqrt(6)*z*(z**4+5.0*x**2*z**2-1.0+4.0*x**4-x**2)/4.0
        elif (jlm == 12):
            if (tau == 1):
                # dyz fz(x2-y2) sigma
                c = 3.0/2.0*y*(2.0*x**2-1.0+z**2)*(z**2)*np.sqrt(5)
            elif (tau == 2):
                # dyz fz(x2-y2) pi
                c = -(6.0*z**4+12.0*x**2*z**2-5.0*z**2+1.0-2.0*x**2)*y*np.sqrt(10)/4.0
            elif (tau == 3):
                # dyz fz(x2-y2) delta
                c = y*(3.0*z**4-z**2+6.0*x**2*z**2-4.0*x**2)/2.0
        elif (jlm == 13):
            if (tau == 1):
                # dyz fxyz sigma
                c = -3.0*(-1.0+z**2+x**2)*x*(z**2)*np.sqrt(5)
            elif (tau == 2):
                # dyz fxyz pi
                c = (6.0*z**4-6.0*z**2+6.0*x**2*z**2+1.0-x**2)*x*np.sqrt(10)/2.0
            elif (tau == 3):
                # dyz fxyz delta
                c = -(x*(3.0*z**4+3.0*x**2*z**2-3.0*z**2+1.0-2.0*x**2))
        elif (jlm == 14):
            if (tau == 1):
                # dyz fyz2 sigma
                c = -3.0/4.0*(-1.0+z**2+x**2)*np.sqrt(2)*z*(5.0*z**2-1.0)
            elif (tau == 2):
                # dyz fyz2 pi
                c = ((30.0*z**4-37.0*z**2+30.0*x**2*z**2+11.0-12.0*x**2)*z)/4.0
            elif (tau == 3):
                # dyz fyz2 delta
                c = -(-1.0+z)*np.sqrt(10)*(3.0*z**3+3.0*z**2-z+3.0*x**2*z-1.0+3.0*x**2)*z/4.0
        elif (jlm == 15):
            if (tau == 1):
                # dyz fxz2 sigma
                c = 3.0/4.0*y*x*np.sqrt(2)*z*(5.0*z**2-1.0)
            elif (tau == 2):
                # dyz fxz2 pi
                c = -3.0/2.0*(5.0*z**2-2.0)*y*x*z
            elif (tau == 3):
                # dyz fxz2 delta
                c = 3.0/4.0*z*np.sqrt(10)*x*y*(-1.0+z**2)
        elif (jlm == 16):
            if (tau == 1):
                # dyz fz3 sigma
                c = y*np.sqrt(3)*(z**2)*(5.0*z**2-3.0)/2.0
            elif (tau == 2):
                # dyz fz3 pi
                c = -(5.0*z**2-1.0)*np.sqrt(3)*np.sqrt(2)*(2.0*z**2-1.0)*y/4.0
            elif (tau == 3):
                # dyz fz3 delta
                c = (-1.0+z**2)*np.sqrt(15)*(z**2)*y/2.0
    elif (ilm == 7):
        if (jlm == 7):
            if (tau == 1):
                # dxz dxz sigma
                c = 3. * x**2 * z**2
            elif (tau == 2):
                # dxz dxz pi
                c = x**2 + z**2 - 4. * x**2 * z**2
            elif (tau == 3):
                # dxz dxz delta
                c = y**2 + x**2 * z**2
        elif (jlm == 8):
            if (tau == 1):
                # dxz dx2-y2 sigma
                c = 1.5 * x * z * (x**2 - y**2)
            elif (tau == 2):
                # dxz dx2-y2 pi
                c = x * z * (1. - 2. * (x**2 - y**2))
            elif (tau == 3):
                # dxz dx2-y2 delta
                c = -x * z * (1. - 0.5 * (x**2 - y**2))
        elif (jlm == 9):
            if (tau == 1):
                # dxz dz2 sigma
                c = np.sqrt(3) * x * z * (z**2 - 0.5 * (x**2 + y**2))
            elif (tau == 2):
                # dxz dz2 pi
                c = np.sqrt(3) * x * z * ((x**2 + y**2) - z**2)
            elif (tau == 3):
                # dxz dz2 delta
                c = -0.5 * np.sqrt(3) * x * z * (x**2 + y**2)
        elif (jlm == 10):
            if (tau == 1):
                # dxz fx(x2-3y2) sigma
                c = (x**2)*(4.0*x**2-3.0+3.0*z**2)*np.sqrt(6)*z*np.sqrt(5)/4.0
            elif (tau == 2):
                # dxz fx(x2-3y2) pi
                c = -np.sqrt(15)*z*(6.0*x**2.0*z**2-z**2+1.0+8.0*x**4-8.0*x**2)/4.0
            elif (tau == 3):
                # dxz fx(x2-3y2) delta
                c = np.sqrt(6)*z*(3.0*x**2.0*z**2-2.0*z**2-7.0*x**2+2.0+4.0*x**4)/4.0
        elif (jlm == 11):
            if (tau == 1):
                # dxz fy(3x2-y2) sigma
                c = x*y*(4.0*x**2-1.0+z**2)*np.sqrt(6)*z*np.sqrt(5)/4.0
            elif (tau == 2):
                # dxz fy(3x2-y2) pi
                c = -x*y*np.sqrt(15)*z*(z**2-2.0+4.0*x**2)/2.0
            elif (tau == 3):
                # dxz fy(3x2-y2) delta
                c = np.sqrt(6)*y*x*z*(z**2+4.0*x**2-5.0)/4.0
        elif (jlm == 12):
            if (tau == 1):
                # dxz fz(x2-y2) sigma
                c = 3.0/2.0*x*(2.0*x**2-1.0+z**2)*(z**2)*np.sqrt(5)
            elif (tau == 2):
                # dxz fz(x2-y2) pi
                c = -(6.0*z**4+12.0*x**2.0*z**2-9.0*z**2+1.0-2.0*x**2)*x*np.sqrt(10)/4.0
            elif (tau == 3):
                # dxz fz(x2-y2) delta
                c = (x*(3.0*z**4-9.0*z**2+6.0*x**2.0*z**2+4.0-4.0*x**2))/2.0
        elif (jlm == 13):
            if (tau == 1):
                # dxz fxyz sigma
                c = 3.0*(x**2)*y*(z**2)*np.sqrt(5)
            elif (tau == 2):
                # dxz fxyz pi
                c = -(6.0*x**2.0*z**2-z**2-x**2)*y*np.sqrt(10)/2.0
            elif (tau == 3):
                # dxz fxyz delta
                c = y*(-2.0*z**2+3.0*x**2.0*z**2+1.0-2.0*x**2)
        elif (jlm == 14):
            if (tau == 1):
                # dxz fyz2 sigma
                c = 3.0/4.0*x*y*np.sqrt(2)*z*(5.0*z**2-1.0)
            elif (tau == 2):
                # dxz fyz2 pi
                c = -3.0/2.0*(5.0*z**2-2.0)*x*y*z
            elif (tau == 3):
                # dxz fyz2 delta
                c = 3.0/4.0*z*np.sqrt(10)*y*x*(-1.0+z**2)
        elif (jlm == 15):
            if (tau == 1):
                # dxz fxz2 sigma
                c = 3.0/ 4.0*x**2.0*np.sqrt(2)*z*(5.0*z**2-1.0)
            elif (tau == 2):
                # dxz fxz2 pi
                c = -(30.0*x**2.0*(z**2)-(5.0*z**2)-12.0*x**2+1.0)*z/4.0
            elif (tau == 3):
                # dxz fxz2 delta
                c = (-1.0+z)*np.sqrt(10)*(-(2.0*z)+3.0*x**2.0*z-2.0+3.0*x**2)*z/4.0
        elif (jlm == 16):
            if (tau == 1):
                # dxz fz3 sigma
                c = x*np.sqrt(3)*(z**2)*(5.0*z**2-3.0)/2.0
            elif (tau == 2):
                # dxz fz3 pi
                c = -(5.0*z**2-1.0)*np.sqrt(3)*np.sqrt(2)*(2.0*z**2-1.0)*x/4.0
            elif (tau == 3):
                # dxz fz3 delta
                c = (-1.0+z**2)*np.sqrt(15)*(z**2)*x/2.0
    elif (ilm == 8):
        if (jlm == 8):
            if (tau == 1):
                # dx2-y2 dx2-y2 sigma
                c = 0.75 * (x**2 - y**2)**2
            elif (tau == 2):
                # dx2-y2 dx2-y2 pi
                c = (x**2 + y**2) - (x**2 - y**2)**2
            elif (tau == 3):
                # dx2-y2 dx2-y2 delta
                c = z**2 + 0.25 * (x**2 - y**2)**2
        elif (jlm == 9):
            if (tau == 1):
                # dx2-y2 dz2 sigma
                c = 0.5 * np.sqrt(3) * (x**2 - y**2) * (z**2 - 0.5 * (x**2 + y**2))
            elif (tau == 2):
                # dx2-y2 dz2 pi
                c = -np.sqrt(3) * z**2 * (x**2 - y**2)
            elif (tau == 3):
                # dx2-y2 dz2 delta
                c = 0.25 * np.sqrt(3) * (1. + z**2) * (x**2 - y**2)
        elif (jlm == 10):
            if (tau == 1):
                # dx2-y2 fx(x2-3y2) sigma
                c = (2.0*x**2-1.0+z**2)*x*(4.0*x**2-3.0+3.0*z**2)*np.sqrt(6)*np.sqrt(5)/8.0
            elif (tau == 2):
                # dx2-y2 fx(x2-3y2) pi
                c = -np.sqrt(15)*x*(3.0*z**4+10.0*x**2*z**2-5.0*z**2-10.0*x**2+8.0*x**4+2.0)/4.0
            elif (tau == 3):
                # dx2-y2 fx(x2-3y2) delta
                c = np.sqrt(6)*x*(3.0*z**4+10.0*x**2*z**2-2.0*z**2+3.0+8.0*x**4-10.0*x**2)/8.0
        elif (jlm == 11):
            if (tau == 1):
                # dx2-y2 fy(3x2-y2) sigma
                c = (2.0*x**2-1.0+z**2)*y*(4.0*x**2-1.0+z**2)*np.sqrt(6)*np.sqrt(5)/8.0
            elif (tau == 2):
                # dx2-y2 fy(3x2-y2) pi
                c = -np.sqrt(15)*y*(z**4-z**2+6.0*x**2.0*z**2+8.0*x**4-6.0*x**2)/4.0
            elif (tau == 3):
                # dx2-y2 fy(3x2-y2) delta
                c = np.sqrt(6)*y*(z**4+6.0*x**2.0*z**2+2.0*z**2+1.0+8.0*x**4-6.0*x**2)/8.0
        elif (jlm == 12):
            if (tau == 1):
                # dx2-y2 fz(x2-y2) sigma
                c = 3.0/4.0*((2.0*x**2-1.0+z**2)**2)*np.sqrt(5)*z
            elif (tau == 2):
                # dx2-y2 fz(x2-y2) pi
                c = -(3.0*z**4+12.0*x**2*z**2-4.0*z**2+12.0*x**4+1.0-12.0*x**2)*np.sqrt(10)*z/4.0
            elif (tau == 3):
                # dx2-y2 fz(x2-y2) delta
                c = (z*(3.0*z**4+12.0*x**2*z**2+2.0*z**2-12.0*x**2-1.0+12.0*x**4))/4.0
        elif (jlm == 13):
            if (tau == 1):
                # dx2-y2 fxyz sigma
                c = 3.0/2.0*(2.0*x**2-1.0+z**2)*x*y*np.sqrt(5)*z
            elif (tau == 2):
                # dx2-y2 fxyz pi
                c = -3.0/2.0*z*np.sqrt(10)*(2.0*x**2-1.0+z**2)*x*y
            elif (tau == 3):
                # dx2-y2 fxyz delta
                c = 3.0/2.0*(2.0*x**2-1.0+z**2)*x*y*z
        elif (jlm == 14):
            if (tau == 1):
                # dx2-y2 fyz2 sigma
                c = 3.0/8.0*(2.0*x**2-1.0+z**2)*y*np.sqrt(2)*(5.0*z**2-1.0)
            elif (tau == 2):
                # dx2-y2 fyz2 pi
                c = -(15.0*z**4+30.0*x**2*z**2-11.0*z**2-2.0*x**2)*y/4.0
            elif (tau == 3):
                # dx2-y2 fyz2 delta
                c = y*np.sqrt(10)*(3.0*z**4+2.0*z**2+6.0*x**2*z**2-1.0+2.0*x**2)/8.0
        elif (jlm == 15):
            if (tau == 1):
                # dx2-y2 fxz2 sigma
                c = 3.0/8.0*(2.0*x**2-1.0+z**2)*x*np.sqrt(2)*(5.0*z**2-1.0)
            elif (tau == 2):
                # dx2-y2 fxz2 pi
                c = -((15.0*z**4+30.0*x**2*z**2-21.0*z**2+2.0-2.0*x**2)*x)/4.0
            elif (tau == 3):
                # dx2-y2 fxz2 delta
                c = x*np.sqrt(10)*(3.0*z**4+6.0*x**2*z**2-6.0*z**2+2.0*x**2-1.0)/8.0
        elif (jlm == 16):
            if (tau == 1):
                # dx2-y2 fz3 sigma
                c = (2.0*x**2-1.0+z**2)*np.sqrt(3)*z*(5.0*z**2-3.0)/4.0
            elif (tau == 2):
                # dx2-y2 fz3 pi
                c = -(5.0*z**2-1.0)*z*(2.0*x**2-1.0+z**2)*np.sqrt(6)/4.0
            elif (tau == 3):
                # dx2-y2 fz3 delta
                c = (2.0*x**2-1.0+z**2)*(z**2+1.0)*np.sqrt(15)*z/4.0
    elif (ilm == 9):
        if (jlm == 9):
            if (tau == 1):
                # dz2 dz2 sigma
                c = (z**2 - 0.5 * (x**2 + y**2))**2
            elif (tau == 2):
                # dz2 dz2 pi
                c = 3. * z**2 * (x**2 + y**2)
            elif (tau == 3):
                # dz2 dz2 delta
                c = 0.75 * (x**2 + y**2)**2
        elif (jlm == 10):
            if (tau == 1):
                # dz2 fx(x2-3y2) sigma
                c = np.sqrt(2)*x*(4.0*x**2-3.0+3.0*z**2)*(3.0*z**2-1.0)*np.sqrt(5)/8.0
            elif (tau == 2):
                # dz2 fx(x2-3y2) pi
                c = -3.0/4.0*(z**2)*x*(4.0*x**2-3.0+3.0*z**2)*np.sqrt(5)
            elif (tau == 3):
                # dz2 fx(x2-3y2) delta
                c = 3.0/8.0*np.sqrt(2)*x*(4.0*x**2-3.0+3.0*z**2)*(z**2+1.0)
        elif (jlm == 11):
            if (tau == 1):
                # dz2 fy(3x2-y2) sigma
                c = np.sqrt(2)*y*(4.0*x**2-1.0+z**2)*(3.0*z**2-1.0)*np.sqrt(5)/8.0
            elif (tau == 2):
                # dz2 fy(3x2-y2) pi
                c = -3.0/4.0*(z**2)*y*(4.0*x**2-1.0+z**2)*np.sqrt(5)
            elif (tau == 3):
                # dz2 fy(3x2-y2) delta
                c = 3.0/8.0*np.sqrt(2)*y*(4.0*x**2-1.0+z**2)*(z**2+1.0)
        elif (jlm == 12):
            if (tau == 1):
                # dz2 fz(x2-y2) sigma
                c = (2.0*x**2-1.0+z**2)*(3.0*z**2-1.0)*np.sqrt(15)*z/4.0
            elif (tau == 2):
                # dz2 fz(x2-y2) pi
                c = -(3.0*z**2-1.0)*(2.0*x**2-1.0+z**2)*z*np.sqrt(30)/4.0
            elif (tau == 3):
                # dz2 fz(x2-y2) delta
                c = np.sqrt(3)*(2.0*x**2-1.0+z**2)*z*(3.0*z**2-1.0)/4.0
        elif (jlm == 13):
            if (tau == 1):
                # dz2 fxyz sigma
                c = x*y*(3.0*z**2-1.0)*np.sqrt(15)*z/2.0
            elif (tau == 2):
                # dz2 fxyz pi
                c = -(3.0*z**2-1.0)*x*z*y*np.sqrt(30)/2.0
            elif (tau == 3):
                # dz2 fxyz delta
                c = np.sqrt(3)*x*y*z*(3.0*z**2-1.0)/2.0
        elif (jlm == 14):
            if (tau == 1):
                # dz2 fyz2 sigma
                c = np.sqrt(2)*y*(3.0*z**2-1.0)*np.sqrt(3)*(5.0*z**2-1.0)/8.0
            elif (tau == 2):
                # dz2 fyz2 pi
                c = -(15.0*z**2-11.0)*y*(z**2)*np.sqrt(3)/4.0
            elif (tau == 3):
                # dz2 fyz2 delta
                c = (3.0*z**3+3.0*z**2-z-1.0)*(-1.0+z)*np.sqrt(5)*np.sqrt(2)*y*np.sqrt(3)/8.0
        elif (jlm == 15):
            if (tau == 1):
                # dz2 fxz2 sigma
                c = np.sqrt(2)*x*(3.0*z**2-1.0)*np.sqrt(3)*(5.0*z**2-1.0)/8.0
            elif (tau == 2):
                # dz2 fxz2 pi
                c = -(15.0*z**2-11.0)*x*(z**2)*np.sqrt(3)/4.0
            elif (tau == 3):
                # dz2 fxz2 delta
                c = (3.0*z**3+3.0*z**2-z-1.0)*(-1.0+z)*np.sqrt(5)*np.sqrt(2)*x*np.sqrt(3)/8.0
        elif (jlm == 16):
            if (tau == 1):
                # dz2 fz3 sigma
                c = ((3.0*z**2-1.0)*z*(5.0*z**2-3.0))/4.0
            elif (tau == 2):
                # dz2 fz3 pi
                c = -3.0/4.0*(5.0*z**2-1.0)*(-1.0+z**2)*z*np.sqrt(2)
            elif (tau == 3):
                # dz2 fz3 delta
                c = 3.0/4.0*((-1.0+z**2)**2)*np.sqrt(5)*z
    elif (ilm == 10):
        if (jlm == 10):
            if (tau == 1):
                # fx(x2-3y2) fx(x2-3y2) sigma
                c = 5.0/8.0*(x**2)*((4.0*x**2-3.0+3.0*z**2)**2)
            elif (tau == 2):
                # fx(x2-3y2) fx(x2-3y2) pi
                c = (-135.0/16.0*(x**2)*(z**4)+15.0/16.0*(z**4)-45.0/2.0*(x**4)*(z**2)-15.0/8.0*(z**2)+135.0/8.0*(x**2)*(z**2)+45.0/2.0*(x**4)+15.0/16.0-135.0/16.0*(x**2)-(15.0*x**6))
            elif (tau == 3):
                # fx(x2-3y2) fx(x2-3y2) delta
                c = (27.0/8.0*(x**2)*(z**4)-3.0/2.0*(z**4)+3.0/2.0*(z**2)+(9.0*x**4*z**2)-27.0/4.0*(x**2)*(z**2)+27.0/8.0*(x**2)-(9.0*x**4)+(6.0*x**6))
            elif (tau == 4):
                # fx(x2-3y2) fx(x2-3y2) phi
                c = (9.0/16.0*(z**4)-9.0/ 16.0*(x**2)*(z**4)-3.0/2.0*(x**4)*(z**2)+3.0/8.0*(z**2)+9.0/8.0*(x**2)*(z**2)+3.0/2.0*(x**4)+1.0/16.0-9.0/16.0*(x**2)-(x**6))
        elif (jlm == 11):
            if (tau == 1):
                # fx(x2-3y2) fy(3x2-y2) sigma
                c = 5.0/8.0*x*(4.0*x**2-3.0+3.0*z**2)*y*(4.0*x**2-1.0+z**2)
            elif (tau == 2):
                # fx(x2-3y2) fy(3x2-y2) pi
                c = -15.0/16.0*x*(4.0*x**2-3.0+3.0*z**2)*y*(4.0*x**2-1.0+z**2)
            elif (tau == 3):
                # fx(x2-3y2) fy(3x2-y2) delta
                c = 3.0/8.0*x*(4.0*x**2-3.0+3.0*z**2)*y*(4.0*x**2-1.0+z**2)
            elif (tau == 4):
                # fx(x2-3y2) fy(3x2-y2) phi
                c = -x*(4.0*x**2-3.0+3.0*z**2)*y*(4.0*x**2-1.0+z**2)/16.0
        elif (jlm == 12):
            if (tau == 1):
                # fx(x2-3y2) fz(x2-y2) sigma
                c = 5.0/8.0*(2.0*x**2-1.0+z**2)*x*(4.0*x**2-3.0+3.0*z**2)*np.sqrt(6)*z
            elif (tau == 2):
                # fx(x2-3y2) fz(x2-y2) pi
                c = -5.0/ 16.0*np.sqrt(6)*x*z*(9.0*z**4+30.0*x**2*z**2-16.0*z**2-30.0*x**2+24.0*x**4+7.0)
            elif (tau == 3):
                # fx(x2-3y2) fz(x2-y2) delta
                c = np.sqrt(6)*x*z*(9.0*z**4+30.0*x**2*z**2-10.0*z**2+24.0*x**4-30.0*x**2+5.0)/8.0
            elif (tau == 4):
                # fx(x2-3y2) fz(x2-y2) phi
                c = -np.sqrt(6)*x*(3.0*z**4+10.0*x**2*z**2+8.0*x**4+5.0-10.0*x**2)*z/16.0
        elif (jlm == 13):
            if (tau == 1):
                # fx(x2-3y2) fxyz sigma
                c = 5.0/4.0*(x**2)*y*(4.0*x**2-3.0+3.0*z**2)*np.sqrt(6)*z
            elif (tau == 2):
                # fx(x2-3y2) fxyz pi
                c = -5.0/8.0*np.sqrt(6)*y*z*(9.0*z**2*x**2-z**2+1.0+12.0*x**4-9.0*x**2)
            elif (tau == 3):
                # fx(x2-3y2) fxyz delta
                c = np.sqrt(6)*y*z*(9.0*z**2*x**2-4.0*z**2+12.0*x**4-9.0*x**2+2.0)/4.0
            elif (tau == 4):
                # fx(x2-3y2) fxyz phi
                c = -y*np.sqrt(6)*(3.0*z**2*x**2-3.0*z**2-1.0-3.0*x**2+4.0*x**4)*z/8.0
        elif (jlm == 14):
            if (tau == 1):
                # fx(x2-3y2) fyz2 sigma
                c = y*x*(4.0*x**2-3.0+3.0*z**2)*np.sqrt(3)*(5.0*z**2-1.0)*np.sqrt(5)/8.0
            elif (tau == 2):
                # fx(x2-3y2) fyz2 pi
                c = -y*x*np.sqrt(15)*(45.0*z**4+60.0*z**2*x**2-38.0*z**2+1.0-4.0*x**2)/16.0
            elif (tau == 3):
                # fx(x2-3y2) fyz2 delta
                c = y*x*np.sqrt(15)*(9.0*z**4+2.0*z**2+12.0*z**2*x**2+4.0*x**2-3.0)/8.0
            elif (tau == 4):
                # fx(x2-3y2) fyz2 phi
                c = -y*np.sqrt(15)*x*(3.0*z**4+4.0*z**2*x**2+6.0*z**2-1.0+4.0*x**2)/ 16.0
        elif (jlm == 15):
            if (tau == 1):
                # fx(x2-3y2) fxz2 sigma
                c = (x**2)*(4.0*x**2-3.0+3.0*z**2)*np.sqrt(3)*(5.0*z**2-1.0)*np.sqrt(5)/8.0
            elif (tau == 2):
                # fx(x2-3y2) fxz2 pi
                c = -np.sqrt(15)*(-5.0*z**4+45.0*x**2*z**4-58.0*x**2*z**2+6.0*z**2+60.0*x**4*z**2-1.0+5.0*x**2-4.0*x**4)/16.0
            elif (tau == 3):
                # fx(x2-3y2) fxz2 delta
                c = np.sqrt(15)*(-4.0*z**4+9.0*x**2*z**4-14.0*x**2*z**2+12.0*x**4*z**2+4.0*z**2-3.0*x**2+4.0*x**4)/8.0
            elif (tau == 4):
                # fx(x2-3y2) fxz2 phi
                c = -np.sqrt(15)*(3.0*x**2.0*z**4-3.0*z**4+2*z**2+4.0*x**4*z**2-6.0*x**2*z**2+1.0-5.0*x**2+4.0*x**4)/16.0
        elif (jlm == 16):
            if (tau == 1):
                # fx(x2-3y2) fz3 sigma
                c = np.sqrt(2)*x*(4.0*x**2-3.0+3.0*z**2)*z*(5.0*z**2-3.0)*np.sqrt(5)/8.0
            elif (tau == 2):
                # fx(x2-3y2) fz3 pi
                c = -3.0/16.0*np.sqrt(2)*(5.0*z**2-1.0)*x*(4.0*x**2-3.0+3.0*z**2)*np.sqrt(5)*z
            elif (tau == 3):
                # fx(x2-3y2) fz3 delta
                c = 3.0/8.0*np.sqrt(10)*z*x*(4.0*x**2-3.0+3.0*z**2)*(z**2+1.0)
            elif (tau == 4):
                # fx(x2-3y2) fz3 phi
                c = -np.sqrt(2)*np.sqrt(5)*x*(4.0*x**2-3.0+3.0*z**2)*z*(z**2+3.0)/16.0
    elif (ilm == 11):
        if (jlm == 11):
            if (tau == 1):
                # fy(3x2-y2) fy(3x2-y2) sigma
                c = - 5.0/ 8.0*(-1.0+z**2+x**2)*((4.0*x**2-1.0+z**2)**2)
            elif (tau == 2):
                # fy(3x2-y2) fy(3x2-y2) pi
                c = (15.0/16.0*(z**6)- 15.0/8.0*(z**4)+135.0/ 16.0*(z**4)*(x**2)-135.0/8.0*(x**2)*(z**2)+15.0/16.0*(z**2)+45.0/2.0*(z**2)*(x**4)+135.0/16.0*(x**2)-45.0/2.0*(x**4)+(15.0*x**6))
            elif (tau == 3):
                # fy(3x2-y2) fy(3x2-y2) delta
                c = (- 3.0/ 8.0*(z**6)- 27.0/8.0*(z**4)*(x**2)-3.0/8.0*(z**4)+3.0/8.0*(z**2)-(9.0*z**2*x**4)+27.0/4.0*(x**2)*(z**2)+3.0/8.0-(6.0*x**6)-27.0/8.0*(x**2)+(9.0*x**4))
            elif (tau == 4):
                # fy(3x2-y2) fy(3x2-y2) phi
                c = ((z**6)/16.0+3.0/8.0*(z**4)+9.0/16.0*(z**4)*(x**2)-9.0/8.0*(x**2)*(z**2)+9.0/16.0*(z**2)+3.0/2.0*(z**2)*(x**4)+9.0/16.0*(x**2)-3.0/2.0*(x**4)+(x**6))
        elif (jlm == 12):
            if (tau == 1):
                # fy(3x2-y2) fz(x2-y2) sigma
                c = 5.0/8.0*y*(4.0*x**2-1.0+z**2)*(2.0*x**2-1.0+z**2)*np.sqrt(6)*z
            elif (tau == 2):
                # fy(3x2-y2) fz(x2-y2) pi
                c = -5.0/16.0*np.sqrt(6)*y*z*(3.0*z**4+18.0*x**2*z**2-4.0*z**2+24.0*x**4+1.0-18.0*x**2)
            elif (tau == 3):
                # fy(3x2-y2) fz(x2-y2) delta
                c = np.sqrt(6)*y*z*(3.0*z**4+18.0*x**2*z**2+2.0*z**2+24.0*x**4-18.0*x**2-1.0)/8.0
            elif (tau == 4):
                # fy(3x2-y2) fz(x2-y2) phi
                c = -y*np.sqrt(6)*(z**4+6.0*x**2*z**2+4.0*z**2+3.0-6.0*x**2+8.0*x**4)*z/16.0
        elif (jlm == 13):
            if (tau == 1):
                # fy(3x2-y2) fxyz sigma
                c = - 5.0/4.0*(-1.0+z**2+x**2)*(4.0*x**2-1.0+z**2)*x*np.sqrt(6)*z
            elif (tau == 2):
                # fy(3x2-y2) fxyz pi
                c = 5.0/8.0*np.sqrt(6)*z*x*(3.0*z**4+15.0*x**2*z**2-7.0*z**2+4.0-15.0*x**2+12.0*x**4)
            elif (tau == 3):
                # fy(3x2-y2) fxyz delta
                c = -np.sqrt(6)*z*x*(3.0*z**4+15.0*x**2*z**2-10.0*z**2+5.0-15.0*x**2+12.0*x**4)/4.0
            elif (tau == 4):
                # fy(3x2-y2) fxyz phi
                c = x*np.sqrt(6)*(z**4+5.0*x**2*z**2-5.0*z**2+4.0*x**4-5.0*x**2)*z/8.0
        elif (jlm == 14):
            if (tau == 1):
                # fy(3x2-y2) fyz2 sigma
                c = -(-1.0+z**2+x**2)*(4.0*x**2-1.0+z**2)*np.sqrt(5)*np.sqrt(3)*(5.0*z**2-1.0)/8.0
            elif (tau == 2):
                # fy(3x2-y2) fyz2 pi
                c = np.sqrt(15)*(15.0*z**6-26.0*z**4+75.0*x**2*z**4-70.0*x**2*z**2+11.0*z**2+60.0*x**4*z**2-4.0*x**4+3.0*x**2)/16.0
            elif (tau == 3):
                # fy(3x2-y2) fyz2 delta
                c = -np.sqrt(15)*(3.0*z**6-z**4+15.0*x**2*z**4-3.0*z**2-2.0*x**2*z**2+12.0*x**4*z**2+1.0+4.0*x**4-5.0*x**2)/8.0
            elif (tau == 4):
                # fy(3x2-y2) fyz2 phi
                c = np.sqrt(15)*(z**6+2.0*z**4+5.0*x**2*z**4-3.0*z**2+6.0*x**2*z**2+4.0*x**4*z**2-3.0*x**2+4.0*x**4)/16.0
        elif (jlm == 15):
            if (tau == 1):
                # fy(3x2-y2) fxz2 sigma
                c = y*(4.0*x**2-1.0+z**2)*x*np.sqrt(5)*np.sqrt(3)*(5.0*z**2-1.0)/8.0
            elif (tau == 2):
                # fy(3x2-y2) fxz2 pi
                c = -y*np.sqrt(15)*x*(15.0*z**4+60.0*x**2*z**2-26.0*z**2+3.0-4.0*x**2)/ 16.0
            elif (tau == 3):
                # fy(3x2-y2) fxz2 delta
                c = y*np.sqrt(15)*x*(3.0*z**4+12.0*x**2*z**2-10.0*z**2+4.0*x**2-1.0)/8.0
            elif (tau == 4):
                # fy(3x2-y2) fxz2 phi
                c = -y*np.sqrt(15)*x*(z**4+4.0*x**2*z**2-6.0*z**2+4.0*x**2-3.0)/16.0
        elif (jlm == 16):
            if (tau == 1):
                # fy(3x2-y2) fz3 sigma
                c = y*(4.0*x**2-1.0+z**2)*np.sqrt(2)*np.sqrt(5)*z*(5.0*z**2-3.0)/8.0
            elif (tau == 2):
                # fy(3x2-y2) fz3 pi
                c = -3.0/16.0*y*(4.0*x**2-1.0+z**2)*np.sqrt(5)*z*np.sqrt(2)*(5.0*z**2-1.0)
            elif (tau == 3):
                # fy(3x2-y2) fz3 delta
                c = 3.0/8.0*np.sqrt(10)*y*(4.0*x**2-1.0+z**2)*(z**2+1.0)*z
            elif (tau == 4):
                # fy(3x2-y2) fz3 phi
                c = -np.sqrt(5)*np.sqrt(2)*(z**2+3.0)*z*(4.0*x**2-1.0+z**2)*y/16.0
    elif (ilm == 12):
        if (jlm == 12):
            if (tau == 1):
                # fz(x2-y2) fz(x2-y2) sigma
                c = 15.0/4.0*((2.0*x**2-1.0+z**2)**2)*(z**2)
            elif (tau == 2):
                # fz(x2-y2) fz(x2-y2) pi
                c = (- 45.0/8.0*(z**6)-45.0/2.0*(x**2)*(z**4)+75.0/8.0*(z**4)+(25.0*x**2*z**2)-35.0/8.0*(z**2)-45.0/2.0*(x**4)*(z**2)-5.0/2.0*(x**2)+5.0/8.0+5.0/2.0*(x**4))
            elif (tau == 3):
                # fz(x2-y2) fz(x2-y2) delta
                c = (9.0/4.0*(z**6)+(9.0*x**2*z**4)-3.0/2.0*(z**4)-(13.0*x**2*z**2)+(z**2)/4.0+(9.0*x**4*z**2)-(4.0*x**4)+(4.0*x**2))
            elif (tau == 4):
                # fz(x2-y2) fz(x2-y2) phi
                c = (- 3.0/8.0*(z**6)-3.0/2.0*(x**2)*(z**4)-3.0/8.0*(z**4)+ 3.0/8.0*(z**2)+(3.0*x**2*z**2)-3.0/2.0*(x**4)*(z**2)+3.0/2.0*(x**4)-3.0/2.0*(x**2)+3.0/8.0)
        elif (jlm == 13):
            if (tau == 1):
                # fz(x2-y2) fxyz sigma
                c = 15.0/2.0*x*y*(2.0*x**2-1.0+z**2)*(z**2)
            elif (tau == 2):
                # fz(x2-y2) fxyz pi
                c = -5.0/4.0*(9.0*z**2-1.0)*(2.0*x**2-1.0+z**2)*x*y
            elif (tau == 3):
                # fz(x2-y2) fxyz delta
                c = (2.0*x**2-1.0+z**2)*x*y*(9.0*z**2-4.0)/2.0
            elif (tau == 4):
                # fz(x2-y2) fxyz phi
                c = -3.0/4.0*(-1.0+z**2)*(2.0*x**2-1.0+z**2)*x*y
        elif (jlm == 14):
            if (tau == 1):
                # fz(x2-y2) fyz2 sigma
                c = 3.0/8.0*y*(2.0*x**2-1.0+z**2)*(5.0*z**2-1.0)*np.sqrt(10)*z
            elif (tau == 2):
                # fz(x2-y2) fyz2 pi
                c = -(45.0*z**4+90.0*z**2*x**2-48.0*z**2+11.0-26.0*x**2)*z*y*np.sqrt(10)/16.0
            elif (tau == 3):
                # fz(x2-y2) fyz2 delta
                c = np.sqrt(10)*y*(9.0*z**4-6.0*z**2+18.0*z**2*x**2-10.0*x**2+1.0)*z/8.0
            elif (tau == 4):
                # fz(x2-y2) fyz2 phi
                c = -3.0/16.0*y*(-1.0+z)*np.sqrt(5)*np.sqrt(2)*(z**3+z**2+2.0*z*x**2+z+2.0*x**2+1.0)*z
        elif (jlm == 15):
            if (tau == 1):
                # fz(x2-y2) fxz2 sigma
                c = 3.0/ 8.0*x*(2.0*x**2-1.0+z**2)*(5.0*z**2-1.0)*np.sqrt(10)*z
            elif (tau == 2):
                # fz(x2-y2) fxz2 pi
                c = -(45.0*z**4+90.0*x**2*z**2-68.0*z**2+15.0-26.0*x**2)*z*x*np.sqrt(10)/ 16.0
            elif (tau == 3):
                # fz(x2-y2) fxz2 delta
                c = np.sqrt(10)*x*(9.0*z**4-22.0*z**2+18.0*x**2*z**2+9.0-10.0*x**2)*z/ 8.0
            elif (tau == 4):
                # fz(x2-y2) fxz2 phi
                c = -3.0/ 16.0*x*(-1.0+z)*np.sqrt(5)*np.sqrt(2)*(z**3+z**2+2.0*x**2*z-3.0*z+2.0*x**2-3.0)*z
        elif (jlm == 16):
            if (tau == 1):
                # fz(x2-y2) fz3 sigma
                c = (2.0*x**2-1.0+z**2)*(z**2)*(5.0*z**2-3.0)*np.sqrt(15)/4.0
            elif (tau == 2):
                # fz(x2-y2) fz3 pi
                c = -(3.0*z**2-1.0)*(2.0*x**2-1.0+z**2)*(5.0*z**2-1.0)*np.sqrt(15)/8.0
            elif (tau == 3):
                # fz(x2-y2) fz3 delta
                c = np.sqrt(15)*(z**2)*(2.0*x**2-1.0+z**2)*(3.0*z**2-1.0)/4.0
            elif (tau == 4):
                # fz(x2-y2) fz3 phi
                c = -np.sqrt(5)*(2.0*x**2-1.0+z**2)*np.sqrt(3)*(-1.0+z**4)/8.0
    elif (ilm == 13):
        if (jlm == 13):
            if (tau == 1):
                # fxyz fxyz sigma
                c = -(15.0*x**2*(-1.0+z**2+x**2)*z**2)
            elif (tau == 2):
                # fxyz fxyz pi
                c = (45.0/2.0*(x**2)*(z**4)-5.0/2.0*(z**4)-(25.0*x**2*z**2)+45.0/2.0*(x**4)*(z**2)+5.0/2.0*(z**2)+5.0/2.0*(x**2)-5.0/2.0*(x**4))
            elif (tau == 3):
                # fxyz fxyz delta
                c = ((-9.0*x**2*z**4+4.0*z**4+13.0*x**2*z**2-9.0*x**4*z**2-4.0*z**2-4.0*x**2+4.0*x**4+1.0))
            elif (tau == 4):
                # fxyz fxyz phi
                c = (3.0/2.0*(x**2)*(z**4)-3.0/2.0*(z**4)+3.0/2.0*(z**2)+3.0/2.0*(x**4)*(z**2)-(3.0*x**2*z**2)+3.0/2.0*(x**2)-3.0/2.0*(x**4))
        elif (jlm == 14):
            if (tau == 1):
                # fxyz fyz2 sigma
                c = - 3.0/4.0*x*(-1.0+z**2+x**2)*np.sqrt(10)*z*(5.0*z**2-1.0)
            elif (tau == 2):
                # fxyz fyz2 pi
                c = (45.0*z**4-53.0*z**2+45.0*x**2*z**2+12.0-13.0*x**2)*z*x*np.sqrt(10)/8.0
            elif (tau == 3):
                # fxyz fyz2 delta
                c = -np.sqrt(10)*x*(9.0*z**4+9.0*x**2*z**2-10.0*z**2+3.0-5.0*x**2)*z/4.0
            elif (tau == 4):
                # fxyz fyz2 phi
                c = 3.0/8.0*x*np.sqrt(2)*(-1.0+z)*np.sqrt(5)*(z**3+z**2+x**2*z+x**2)*z
        elif (jlm == 15):
            if (tau == 1):
                # fxyz fxz2 sigma
                c = 3.0/4.0*(x**2)*y*np.sqrt(10)*z*(5.0*z**2-1.0)
            elif (tau == 2):
                # fxyz fxz2 pi
                c = -(45.0*x**2*z**2-5.0*z**2-13.0*x**2+1.0)*z*y*np.sqrt(10)/8.0
            elif (tau == 3):
                # fxyz fxz2 delta
                c = np.sqrt(10)*y*(-4.0*z**2+9.0*x**2*z**2+2.0-5.0*x**2)*z/4.0
            elif (tau == 4):
                # fxyz fxz2 phi
                c = -3.0/8.0*y*np.sqrt(2)*(-1.0+z)*np.sqrt(5)*(-z+x**2*z-1.0+x**2)*z
        elif (jlm == 16):
            if (tau == 1):
                # fxyz fz3 sigma
                c = x*y*np.sqrt(15)*(z**2)*(5.0*z**2-3.0)/2.0
            elif (tau == 2):
                # fxyz fz3 pi
                c = -(5.0*z**2-1.0)*(3.0*z**2-1.0)*x*y*np.sqrt(15)/4.0
            elif (tau == 3):
                # fxyz fz3 delta
                c = x*y*(z**2)*(3.0*z**2-1.0)*np.sqrt(15)/2.0
            elif (tau == 4):
                # fxyz fz3 phi
                c = -x*y*np.sqrt(3)*(z**3+z**2+z+1.0)*np.sqrt(5)*(-1.0+z)/4.0
    elif (ilm == 14):
        if (jlm == 14):
            if (tau == 1):
                # fyz2 fyz2 sigma
                c = -3.0/8.0*(-1.0+z**2+x**2)*((5.0*z**2-1.0)**2)
            elif (tau == 2):
                # fyz2 fyz2 pi
                c = (225.0/16.0*(z**6)-165.0/8.0*(z**4)+225.0/16.0*(x**2)*(z**4)+121.0/16.0*(z**2)-65.0/8.0*(z**2)*(x**2)+(x**2)/16.0)
            elif (tau == 3):
                # fyz2 fyz2 delta
                c = -5.0/8.0*(-1.0+z)*(9.0*z**5+9.0*z**4-6.0*z**3+9.0*x**2*z**3-6.0*z**2+9.0*z**2*x**2+z-x**2*z+1.0-x**2)
            elif (tau == 4):
                # fyz2 fyz2 phi
                c = 15.0/16.0*(-z**2+z**4+z**2*x**2-x**2)*(-1.0+z**2)
        elif (jlm == 15):
            if (tau == 1):
                # fyz2 fxz2 sigma
                c = 3.0/8.0*y*x*((5.0*z**2-1.0)**2)
            elif (tau == 2):
                # fyz2 fxz2 pi
                c = -(225.0*z**4-130.0*z**2+1.0)*y*x/16.0
            elif (tau == 3):
                # fyz2 fxz2 delta
                c = 5.0/8.0*(9.0*z**3+9.0*z**2-z-1.0)*x*(-1.0+z)*y
            elif (tau == 4):
                # fyz2 fxz2 phi
                c = - 15.0/16.0*((-1.0+z**2)**2)*y*x
        elif (jlm == 16):
            if (tau == 1):
                # fyz2 fz3 sigma
                c = y*np.sqrt(2)*np.sqrt(3)*(5.0*z**2-1.0)*z*(5.0*z**2-3.0)/8.0
            elif (tau == 2):
                # fyz2 fz3 pi
                c = -(5.0*z**2-1.0)*np.sqrt(3)*np.sqrt(2)*(15.0*z**2-11.0)*z*y/ 16.0
            elif (tau == 3):
                # fyz2 fz3 delta
                c = 5.0/8.0*(-1.0+z**2)*z*np.sqrt(3)*(3.0*z**2-1.0)*np.sqrt(2)*y
            elif (tau == 4):
                # fyz2 fz3 phi
                c = -5.0/16.0*(-2.0*z**2+1.0+z**4)*np.sqrt(2)*z*np.sqrt(3)*y
    elif (ilm == 15):
        if (jlm == 15):
            if (tau == 1):
                # fxz2 fxz2 sigma
                c = 3.0/8.0*(x**2)*((5.0*z**2-1.0)**2)
            elif (tau == 2):
                # fxz2 fxz2 pi
                c = (-225.0/ 16.0*(x**2)*(z**4)+25.0/16.0*(z**4)+65.0/8.0*(x**2)*(z**2)-5.0/8.0*(z**2)+1.0/16.0-(x**2)/16.0)
            elif (tau == 3):
                # fxz2 fxz2 delta
                c = 5.0/8.0*(-1.0+z)*(9.0*x**2*z**3-4.0*z**3+9.0*x**2*z**2-4.0*z**2-x**2*z-x**2)
            elif (tau == 4):
                # fxz2 fxz2 phi
                c = - 15.0/16.0*(x**2*z**2+1.0-z**2-x**2)*(-1.0+z**2)
        elif (jlm == 16):
            if (tau == 1):
                # fxz2 fz3 sigma
                c = np.sqrt(2)*x*z*(5.0*z**2-3.0)*np.sqrt(3)*(5.0*z**2-1.0)/8.0
            elif (tau == 2):
                # fxz2 fz3 pi
                c = -(15.0*z**2-11.0)*z*x*(5.0*z**2-1.0)*np.sqrt(3)*np.sqrt(2)/16.0
            elif (tau == 3):
                # fxz2 fz3 delta
                c = 5.0/8.0*(3.0*z**3+3.0*z**2-z-1.0)*(-1.0+z)*np.sqrt(2)*x*z*np.sqrt(3)
            elif (tau == 4):
                # fxz2 fz3 phi
                c = -5.0/16.0*z*np.sqrt(3)*x*((-1.0+z**2)**2)*np.sqrt(2)
    elif (ilm == 16):
        if (jlm == 16):
            if (tau == 1):
                # fz3 fz3 sigma
                c = (z**2*(5.0*z**2-3.0)**2)/4.0
            elif (tau == 2):
                # fz3 fz3 pi
                c = -3.0/8.0*(-1.0+z**2)*((5.0*z**2-1.0)**2)
            elif (tau == 3):
                # fz3 fz3 delta
                c = 15.0/4.0*(z**2)*((-1.0+z**2)**2)
            elif (tau == 4):
                # fz3 fz3 phi
                c = -5.0/8.0*(-1.0+z**2)*(-2.0*z**2+1.0+z**4)
    return c
