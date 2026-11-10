#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
"""Module for Gaunt coefficients."""
from math import sqrt, pi
from hotcent.orbitals import ANGULAR_MOMENTUM, ORBITAL_LABELS


def get_gaunt_coefficient(lm1, lm2, lm3):
    """
    Wrapper around calculate_gaunt_coefficient(), returning the Gaunt
    coefficient for the chosen spherical harmonic combination.

    Parameters
    ----------
    lm1, lm2, lm3 : str
        Orbital labels (e.g. 'px').

    Returns
    -------
    c : int or float
        The rotation coefficient.
    """
    ilm = sorted([ORBITAL_LABELS.index(lm) for lm in [lm1, lm2, lm3]])
    l = [ANGULAR_MOMENTUM[ORBITAL_LABELS[lm][0]] for lm in ilm]

    assert l[0] <= 2 and l[1] <= 3 and l[2] <= 5, \
           'Gaunt coefficients are only implemented up to d-f-h combinations.'

    c = calculate_gaunt_coefficient(ilm[0]+1, ilm[1]+1, ilm[2]+1)
    return c


def calculate_gaunt_coefficient(ilm, jlm, klm):
    """
    Returns the Gaunt coefficient for the chosen spherical harmonic
    combination.

    These have been generated with the help of [SymPy](
    https://www.sympy.org).

    Parameters
    ----------
    ilm, jlm, klm : int
        Orbital indices, starting from 1. Assumed to be in ascending
        order, i.e. ilm <= jlm <= klm.

    Returns
    -------
    c : float
        The Gaunt coefficient.
    """
    c = 0

    if (ilm == 1):
        if (jlm == 1):
            if (klm == 1):
                # s s s
                c = 1./(2*sqrt(pi))

        elif (jlm == 2):
            if (klm == 2):
                # s px px
                c = 1./(2*sqrt(pi))

        elif (jlm == 3):
            if (klm == 3):
                # s py py
                c = 1./(2*sqrt(pi))

        elif (jlm == 4):
            if (klm == 4):
                # s pz pz
                c = 1./(2*sqrt(pi))

        elif (jlm == 5):
            if (klm == 5):
                # s dxy dxy
                c = 1./(2*sqrt(pi))

        elif (jlm == 6):
            if (klm == 6):
                # s dyz dyz
                c = 1./(2*sqrt(pi))

        elif (jlm == 7):
            if (klm == 7):
                # s dxz dxz
                c = 1./(2*sqrt(pi))

        elif (jlm == 8):
            if (klm == 8):
                # s dx2-y2 dx2-y2
                c = 1./(2*sqrt(pi))

        elif (jlm == 9):
            if (klm == 9):
                # s dz2 dz2
                c = 1./(2*sqrt(pi))

        elif (jlm == 10):
            if (klm == 10):
                # s fx(x2-3y2) fx(x2-3y2)
                c = 1./(2*sqrt(pi))

        elif (jlm == 11):
            if (klm == 11):
                # s fy(3x2-y2) fy(3x2-y2)
                c = 1./(2*sqrt(pi))

        elif (jlm == 12):
            if (klm == 12):
                # s fz(x2-y2) fz(x2-y2)
                c = 1./(2*sqrt(pi))

        elif (jlm == 13):
            if (klm == 13):
                # s fxyz fxyz
                c = 1./(2*sqrt(pi))

        elif (jlm == 14):
            if (klm == 14):
                # s fyz2 fyz2
                c = 1./(2*sqrt(pi))

        elif (jlm == 15):
            if (klm == 15):
                # s fxz2 fxz2
                c = 1./(2*sqrt(pi))

        elif (jlm == 16):
            if (klm == 16):
                # s fz3 fz3
                c = 1./(2*sqrt(pi))

    elif (ilm == 2):
        if (jlm == 2):
            if (klm == 8):
                # px px dx2-y2
                c = sqrt(15)/(10*sqrt(pi))
            elif (klm == 9):
                # px px dz2
                c = -sqrt(5)/(10*sqrt(pi))

        elif (jlm == 3):
            if (klm == 5):
                # px py dxy
                c = sqrt(15)/(10*sqrt(pi))

        elif (jlm == 4):
            if (klm == 7):
                # px pz dxz
                c = sqrt(15)/(10*sqrt(pi))

        elif (jlm == 5):
            if (klm == 11):
                # px dxy fy(3x2-y2)
                c = 3*sqrt(14)/(28*sqrt(pi))
            elif (klm == 14):
                # px dxy fyz2
                c = -sqrt(210)/(140*sqrt(pi))

        elif (jlm == 6):
            if (klm == 13):
                # px dyz fxyz
                c = sqrt(21)/(14*sqrt(pi))

        elif (jlm == 7):
            if (klm == 12):
                # px dxz fz(x2-y2)
                c = sqrt(21)/(14*sqrt(pi))
            elif (klm == 16):
                # px dxz fz3
                c = -3*sqrt(35)/(70*sqrt(pi))

        elif (jlm == 8):
            if (klm == 10):
                # px dx2-y2 fx(x2-3y2)
                c = 3*sqrt(14)/(28*sqrt(pi))
            elif (klm == 15):
                # px dx2-y2 fxz2
                c = -sqrt(210)/(140*sqrt(pi))

        elif (jlm == 9):
            if (klm == 15):
                # px dz2 fxz2
                c = 3*sqrt(70)/(70*sqrt(pi))

        elif (jlm == 10):
            if (klm == 23):
                # px fx(x2-3y2) g7
                c = -sqrt(42)/(84*sqrt(pi))
            elif (klm == 25):
                # px fx(x2-3y2) g9
                c = sqrt(6)/(6*sqrt(pi))

        elif (jlm == 11):
            if (klm == 17):
                # px fy(3x2-y2) g1
                c = sqrt(6)/(6*sqrt(pi))
            elif (klm == 19):
                # px fy(3x2-y2) g3
                c = -sqrt(42)/(84*sqrt(pi))

        elif (jlm == 12):
            if (klm == 22):
                # px fz(x2-y2) g6
                c = -sqrt(14)/(28*sqrt(pi))
            elif (klm == 24):
                # px fz(x2-y2) g8
                c = sqrt(2)/(4*sqrt(pi))

        elif (jlm == 13):
            if (klm == 18):
                # px fxyz g2
                c = sqrt(2)/(4*sqrt(pi))
            elif (klm == 20):
                # px fxyz g4
                c = -sqrt(14)/(28*sqrt(pi))

        elif (jlm == 14):
            if (klm == 19):
                # px fyz2 g3
                c = sqrt(70)/(28*sqrt(pi))

        elif (jlm == 15):
            if (klm == 21):
                # px fxz2 g5
                c = -sqrt(14)/(14*sqrt(pi))
            elif (klm == 23):
                # px fxz2 g7
                c = sqrt(70)/(28*sqrt(pi))

        elif (jlm == 16):
            if (klm == 22):
                # px fz3 g6
                c = sqrt(210)/(42*sqrt(pi))

    elif (ilm == 3):
        if (jlm == 3):
            if (klm == 8):
                # py py dx2-y2
                c = -sqrt(15)/(10*sqrt(pi))
            elif (klm == 9):
                # py py dz2
                c = -sqrt(5)/(10*sqrt(pi))

        elif (jlm == 4):
            if (klm == 6):
                # py pz dyz
                c = sqrt(15)/(10*sqrt(pi))

        elif (jlm == 5):
            if (klm == 10):
                # py dxy fx(x2-3y2)
                c = -3*sqrt(14)/(28*sqrt(pi))
            elif (klm == 15):
                # py dxy fxz2
                c = -sqrt(210)/(140*sqrt(pi))

        elif (jlm == 6):
            if (klm == 12):
                # py dyz fz(x2-y2)
                c = -sqrt(21)/(14*sqrt(pi))
            elif (klm == 16):
                # py dyz fz3
                c = -3*sqrt(35)/(70*sqrt(pi))

        elif (jlm == 7):
            if (klm == 13):
                # py dxz fxyz
                c = sqrt(21)/(14*sqrt(pi))

        elif (jlm == 8):
            if (klm == 11):
                # py dx2-y2 fy(3x2-y2)
                c = 3*sqrt(14)/(28*sqrt(pi))
            elif (klm == 14):
                # py dx2-y2 fyz2
                c = sqrt(210)/(140*sqrt(pi))

        elif (jlm == 9):
            if (klm == 14):
                # py dz2 fyz2
                c = 3*sqrt(70)/(70*sqrt(pi))

        elif (jlm == 10):
            if (klm == 17):
                # py fx(x2-3y2) g1
                c = sqrt(6)/(6*sqrt(pi))
            elif (klm == 19):
                # py fx(x2-3y2) g3
                c = sqrt(42)/(84*sqrt(pi))

        elif (jlm == 11):
            if (klm == 23):
                # py fy(3x2-y2) g7
                c = -sqrt(42)/(84*sqrt(pi))
            elif (klm == 25):
                # py fy(3x2-y2) g9
                c = -sqrt(6)/(6*sqrt(pi))

        elif (jlm == 12):
            if (klm == 18):
                # py fz(x2-y2) g2
                c = sqrt(2)/(4*sqrt(pi))
            elif (klm == 20):
                # py fz(x2-y2) g4
                c = sqrt(14)/(28*sqrt(pi))

        elif (jlm == 13):
            if (klm == 22):
                # py fxyz g6
                c = -sqrt(14)/(28*sqrt(pi))
            elif (klm == 24):
                # py fxyz g8
                c = -sqrt(2)/(4*sqrt(pi))

        elif (jlm == 14):
            if (klm == 21):
                # py fyz2 g5
                c = -sqrt(14)/(14*sqrt(pi))
            elif (klm == 23):
                # py fyz2 g7
                c = -sqrt(70)/(28*sqrt(pi))

        elif (jlm == 15):
            if (klm == 19):
                # py fxz2 g3
                c = sqrt(70)/(28*sqrt(pi))

        elif (jlm == 16):
            if (klm == 20):
                # py fz3 g4
                c = sqrt(210)/(42*sqrt(pi))

    elif (ilm == 4):
        if (jlm == 4):
            if (klm == 9):
                # pz pz dz2
                c = sqrt(5)/(5*sqrt(pi))

        elif (jlm == 5):
            if (klm == 13):
                # pz dxy fxyz
                c = sqrt(21)/(14*sqrt(pi))

        elif (jlm == 6):
            if (klm == 14):
                # pz dyz fyz2
                c = sqrt(210)/(35*sqrt(pi))

        elif (jlm == 7):
            if (klm == 15):
                # pz dxz fxz2
                c = sqrt(210)/(35*sqrt(pi))

        elif (jlm == 8):
            if (klm == 12):
                # pz dx2-y2 fz(x2-y2)
                c = sqrt(21)/(14*sqrt(pi))

        elif (jlm == 9):
            if (klm == 16):
                # pz dz2 fz3
                c = 3*sqrt(105)/(70*sqrt(pi))

        elif (jlm == 10):
            if (klm == 24):
                # pz fx(x2-3y2) g8
                c = sqrt(3)/(6*sqrt(pi))

        elif (jlm == 11):
            if (klm == 18):
                # pz fy(3x2-y2) g2
                c = sqrt(3)/(6*sqrt(pi))

        elif (jlm == 12):
            if (klm == 23):
                # pz fz(x2-y2) g7
                c = sqrt(7)/(7*sqrt(pi))

        elif (jlm == 13):
            if (klm == 19):
                # pz fxyz g3
                c = sqrt(7)/(7*sqrt(pi))

        elif (jlm == 14):
            if (klm == 20):
                # pz fyz2 g4
                c = sqrt(35)/(14*sqrt(pi))

        elif (jlm == 15):
            if (klm == 22):
                # pz fxz2 g6
                c = sqrt(35)/(14*sqrt(pi))

        elif (jlm == 16):
            if (klm == 21):
                # pz fz3 g5
                c = 2*sqrt(21)/(21*sqrt(pi))

    elif (ilm == 5):
        if (jlm == 5):
            if (klm == 9):
                # dxy dxy dz2
                c = -sqrt(5)/(7*sqrt(pi))
            elif (klm == 21):
                # dxy dxy g5
                c = 1./(14*sqrt(pi))
            elif (klm == 25):
                # dxy dxy g9
                c = -sqrt(35)/(14*sqrt(pi))

        elif (jlm == 6):
            if (klm == 7):
                # dxy dyz dxz
                c = sqrt(15)/(14*sqrt(pi))
            elif (klm == 22):
                # dxy dyz g6
                c = -sqrt(10)/(28*sqrt(pi))
            elif (klm == 24):
                # dxy dyz g8
                c = -sqrt(70)/(28*sqrt(pi))

        elif (jlm == 7):
            if (klm == 18):
                # dxy dxz g2
                c = sqrt(70)/(28*sqrt(pi))
            elif (klm == 20):
                # dxy dxz g4
                c = -sqrt(10)/(28*sqrt(pi))

        elif (jlm == 8):
            if (klm == 17):
                # dxy dx2-y2 g1
                c = sqrt(35)/(14*sqrt(pi))

        elif (jlm == 9):
            if (klm == 19):
                # dxy dz2 g3
                c = sqrt(15)/(14*sqrt(pi))

        elif (jlm == 10):
            if (klm == 14):
                # dxy fx(x2-3y2) fyz2
                c = 1./(6*sqrt(pi))
            elif (klm == 26):
                # dxy fx(x2-3y2) h1
                c = 5*sqrt(33)/(66*sqrt(pi))
            elif (klm == 30):
                # dxy fx(x2-3y2) h5
                c = -sqrt(770)/(924*sqrt(pi))

        elif (jlm == 11):
            if (klm == 15):
                # dxy fy(3x2-y2) fxz2
                c = -1./(6*sqrt(pi))
            elif (klm == 32):
                # dxy fy(3x2-y2) h7
                c = sqrt(770)/(924*sqrt(pi))
            elif (klm == 36):
                # dxy fy(3x2-y2) h11
                c = -5*sqrt(33)/(66*sqrt(pi))

        elif (jlm == 12):
            if (klm == 27):
                # dxy fz(x2-y2) h2
                c = sqrt(55)/(22*sqrt(pi))

        elif (jlm == 13):
            if (klm == 16):
                # dxy fxyz fz3
                c = -1./(3*sqrt(pi))
            elif (klm == 31):
                # dxy fxyz h6
                c = 5*sqrt(77)/(462*sqrt(pi))
            elif (klm == 35):
                # dxy fxyz h10
                c = -sqrt(55)/(22*sqrt(pi))

        elif (jlm == 14):
            if (klm == 15):
                # dxy fyz2 fxz2
                c = sqrt(15)/(15*sqrt(pi))
            elif (klm == 32):
                # dxy fyz2 h7
                c = -5*sqrt(462)/(924*sqrt(pi))
            elif (klm == 34):
                # dxy fyz2 h9
                c = -5*sqrt(11)/(66*sqrt(pi))

        elif (jlm == 15):
            if (klm == 28):
                # dxy fxz2 h3
                c = 5*sqrt(11)/(66*sqrt(pi))
            elif (klm == 30):
                # dxy fxz2 h5
                c = -5*sqrt(462)/(924*sqrt(pi))

        elif (jlm == 16):
            if (klm == 29):
                # dxy fz3 h4
                c = 5*sqrt(11)/(66*sqrt(pi))

    elif (ilm == 6):
        if (jlm == 6):
            if (klm == 8):
                # dyz dyz dx2-y2
                c = -sqrt(15)/(14*sqrt(pi))
            elif (klm == 9):
                # dyz dyz dz2
                c = sqrt(5)/(14*sqrt(pi))
            elif (klm == 21):
                # dyz dyz g5
                c = -2./(7*sqrt(pi))
            elif (klm == 23):
                # dyz dyz g7
                c = -sqrt(5)/(7*sqrt(pi))

        elif (jlm == 7):
            if (klm == 19):
                # dyz dxz g3
                c = sqrt(5)/(7*sqrt(pi))

        elif (jlm == 8):
            if (klm == 18):
                # dyz dx2-y2 g2
                c = sqrt(70)/(28*sqrt(pi))
            elif (klm == 20):
                # dyz dx2-y2 g4
                c = sqrt(10)/(28*sqrt(pi))

        elif (jlm == 9):
            if (klm == 20):
                # dyz dz2 g4
                c = sqrt(30)/(14*sqrt(pi))

        elif (jlm == 10):
            if (klm == 13):
                # dyz fx(x2-3y2) fxyz
                c = -sqrt(10)/(12*sqrt(pi))
            elif (klm == 27):
                # dyz fx(x2-3y2) h2
                c = sqrt(330)/(66*sqrt(pi))
            elif (klm == 29):
                # dyz fx(x2-3y2) h4
                c = sqrt(110)/(132*sqrt(pi))

        elif (jlm == 11):
            if (klm == 12):
                # dyz fy(3x2-y2) fz(x2-y2)
                c = sqrt(10)/(12*sqrt(pi))
            elif (klm == 33):
                # dyz fy(3x2-y2) h8
                c = -sqrt(110)/(132*sqrt(pi))
            elif (klm == 35):
                # dyz fy(3x2-y2) h10
                c = -sqrt(330)/(66*sqrt(pi))

        elif (jlm == 12):
            if (klm == 14):
                # dyz fz(x2-y2) fyz2
                c = -sqrt(6)/(12*sqrt(pi))
            elif (klm == 28):
                # dyz fz(x2-y2) h3
                c = sqrt(110)/(33*sqrt(pi))
            elif (klm == 30):
                # dyz fz(x2-y2) h5
                c = sqrt(1155)/(231*sqrt(pi))

        elif (jlm == 13):
            if (klm == 15):
                # dyz fxyz fxz2
                c = sqrt(6)/(12*sqrt(pi))
            elif (klm == 32):
                # dyz fxyz h7
                c = -sqrt(1155)/(231*sqrt(pi))
            elif (klm == 34):
                # dyz fxyz h9
                c = -sqrt(110)/(33*sqrt(pi))

        elif (jlm == 14):
            if (klm == 16):
                # dyz fyz2 fz3
                c = sqrt(10)/(30*sqrt(pi))
            elif (klm == 31):
                # dyz fyz2 h6
                c = -5*sqrt(770)/(462*sqrt(pi))
            elif (klm == 33):
                # dyz fyz2 h8
                c = -5*sqrt(66)/(132*sqrt(pi))

        elif (jlm == 15):
            if (klm == 29):
                # dyz fxz2 h4
                c = 5*sqrt(66)/(132*sqrt(pi))

        elif (jlm == 16):
            if (klm == 30):
                # dyz fz3 h5
                c = 10*sqrt(77)/(231*sqrt(pi))

    elif (ilm == 7):
        if (jlm == 7):
            if (klm == 8):
                # dxz dxz dx2-y2
                c = sqrt(15)/(14*sqrt(pi))
            elif (klm == 9):
                # dxz dxz dz2
                c = sqrt(5)/(14*sqrt(pi))
            elif (klm == 21):
                # dxz dxz g5
                c = -2./(7*sqrt(pi))
            elif (klm == 23):
                # dxz dxz g7
                c = sqrt(5)/(7*sqrt(pi))

        elif (jlm == 8):
            if (klm == 22):
                # dxz dx2-y2 g6
                c = -sqrt(10)/(28*sqrt(pi))
            elif (klm == 24):
                # dxz dx2-y2 g8
                c = sqrt(70)/(28*sqrt(pi))

        elif (jlm == 9):
            if (klm == 22):
                # dxz dz2 g6
                c = sqrt(30)/(14*sqrt(pi))

        elif (jlm == 10):
            if (klm == 12):
                # dxz fx(x2-3y2) fz(x2-y2)
                c = sqrt(10)/(12*sqrt(pi))
            elif (klm == 33):
                # dxz fx(x2-3y2) h8
                c = -sqrt(110)/(132*sqrt(pi))
            elif (klm == 35):
                # dxz fx(x2-3y2) h10
                c = sqrt(330)/(66*sqrt(pi))

        elif (jlm == 11):
            if (klm == 13):
                # dxz fy(3x2-y2) fxyz
                c = sqrt(10)/(12*sqrt(pi))
            elif (klm == 27):
                # dxz fy(3x2-y2) h2
                c = sqrt(330)/(66*sqrt(pi))
            elif (klm == 29):
                # dxz fy(3x2-y2) h4
                c = -sqrt(110)/(132*sqrt(pi))

        elif (jlm == 12):
            if (klm == 15):
                # dxz fz(x2-y2) fxz2
                c = sqrt(6)/(12*sqrt(pi))
            elif (klm == 32):
                # dxz fz(x2-y2) h7
                c = -sqrt(1155)/(231*sqrt(pi))
            elif (klm == 34):
                # dxz fz(x2-y2) h9
                c = sqrt(110)/(33*sqrt(pi))

        elif (jlm == 13):
            if (klm == 14):
                # dxz fxyz fyz2
                c = sqrt(6)/(12*sqrt(pi))
            elif (klm == 28):
                # dxz fxyz h3
                c = sqrt(110)/(33*sqrt(pi))
            elif (klm == 30):
                # dxz fxyz h5
                c = -sqrt(1155)/(231*sqrt(pi))

        elif (jlm == 14):
            if (klm == 29):
                # dxz fyz2 h4
                c = 5*sqrt(66)/(132*sqrt(pi))

        elif (jlm == 15):
            if (klm == 16):
                # dxz fxz2 fz3
                c = sqrt(10)/(30*sqrt(pi))
            elif (klm == 31):
                # dxz fxz2 h6
                c = -5*sqrt(770)/(462*sqrt(pi))
            elif (klm == 33):
                # dxz fxz2 h8
                c = 5*sqrt(66)/(132*sqrt(pi))

        elif (jlm == 16):
            if (klm == 32):
                # dxz fz3 h7
                c = 10*sqrt(77)/(231*sqrt(pi))

    elif (ilm == 8):
        if (jlm == 8):
            if (klm == 9):
                # dx2-y2 dx2-y2 dz2
                c = -sqrt(5)/(7*sqrt(pi))
            elif (klm == 21):
                # dx2-y2 dx2-y2 g5
                c = 1./(14*sqrt(pi))
            elif (klm == 25):
                # dx2-y2 dx2-y2 g9
                c = sqrt(35)/(14*sqrt(pi))

        elif (jlm == 9):
            if (klm == 23):
                # dx2-y2 dz2 g7
                c = sqrt(15)/(14*sqrt(pi))

        elif (jlm == 10):
            if (klm == 15):
                # dx2-y2 fx(x2-3y2) fxz2
                c = -1./(6*sqrt(pi))
            elif (klm == 32):
                # dx2-y2 fx(x2-3y2) h7
                c = sqrt(770)/(924*sqrt(pi))
            elif (klm == 36):
                # dx2-y2 fx(x2-3y2) h11
                c = 5*sqrt(33)/(66*sqrt(pi))

        elif (jlm == 11):
            if (klm == 14):
                # dx2-y2 fy(3x2-y2) fyz2
                c = -1./(6*sqrt(pi))
            elif (klm == 26):
                # dx2-y2 fy(3x2-y2) h1
                c = 5*sqrt(33)/(66*sqrt(pi))
            elif (klm == 30):
                # dx2-y2 fy(3x2-y2) h5
                c = sqrt(770)/(924*sqrt(pi))

        elif (jlm == 12):
            if (klm == 16):
                # dx2-y2 fz(x2-y2) fz3
                c = -1./(3*sqrt(pi))
            elif (klm == 31):
                # dx2-y2 fz(x2-y2) h6
                c = 5*sqrt(77)/(462*sqrt(pi))
            elif (klm == 35):
                # dx2-y2 fz(x2-y2) h10
                c = sqrt(55)/(22*sqrt(pi))

        elif (jlm == 13):
            if (klm == 27):
                # dx2-y2 fxyz h2
                c = sqrt(55)/(22*sqrt(pi))

        elif (jlm == 14):
            if (klm == 14):
                # dx2-y2 fyz2 fyz2
                c = -sqrt(15)/(15*sqrt(pi))
            elif (klm == 28):
                # dx2-y2 fyz2 h3
                c = 5*sqrt(11)/(66*sqrt(pi))
            elif (klm == 30):
                # dx2-y2 fyz2 h5
                c = 5*sqrt(462)/(924*sqrt(pi))

        elif (jlm == 15):
            if (klm == 15):
                # dx2-y2 fxz2 fxz2
                c = sqrt(15)/(15*sqrt(pi))
            elif (klm == 32):
                # dx2-y2 fxz2 h7
                c = -5*sqrt(462)/(924*sqrt(pi))
            elif (klm == 34):
                # dx2-y2 fxz2 h9
                c = 5*sqrt(11)/(66*sqrt(pi))

        elif (jlm == 16):
            if (klm == 33):
                # dx2-y2 fz3 h8
                c = 5*sqrt(11)/(66*sqrt(pi))

    elif (ilm == 9):
        if (jlm == 9):
            if (klm == 9):
                # dz2 dz2 dz2
                c = sqrt(5)/(7*sqrt(pi))
            elif (klm == 21):
                # dz2 dz2 g5
                c = 3./(7*sqrt(pi))

        elif (jlm == 10):
            if (klm == 10):
                # dz2 fx(x2-3y2) fx(x2-3y2)
                c = -sqrt(5)/(6*sqrt(pi))
            elif (klm == 34):
                # dz2 fx(x2-3y2) h9
                c = sqrt(55)/(33*sqrt(pi))

        elif (jlm == 11):
            if (klm == 11):
                # dz2 fy(3x2-y2) fy(3x2-y2)
                c = -sqrt(5)/(6*sqrt(pi))
            elif (klm == 28):
                # dz2 fy(3x2-y2) h3
                c = sqrt(55)/(33*sqrt(pi))

        elif (jlm == 12):
            if (klm == 33):
                # dz2 fz(x2-y2) h8
                c = sqrt(55)/(22*sqrt(pi))

        elif (jlm == 13):
            if (klm == 29):
                # dz2 fxyz h4
                c = sqrt(55)/(22*sqrt(pi))

        elif (jlm == 14):
            if (klm == 14):
                # dz2 fyz2 fyz2
                c = sqrt(5)/(10*sqrt(pi))
            elif (klm == 30):
                # dz2 fyz2 h5
                c = 5*sqrt(154)/(154*sqrt(pi))

        elif (jlm == 15):
            if (klm == 15):
                # dz2 fxz2 fxz2
                c = sqrt(5)/(10*sqrt(pi))
            elif (klm == 32):
                # dz2 fxz2 h7
                c = 5*sqrt(154)/(154*sqrt(pi))

        elif (jlm == 16):
            if (klm == 16):
                # dz2 fz3 fz3
                c = 2*sqrt(5)/(15*sqrt(pi))
            elif (klm == 31):
                # dz2 fz3 h6
                c = 5*sqrt(385)/(231*sqrt(pi))

    return c
