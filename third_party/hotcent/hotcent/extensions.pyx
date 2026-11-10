#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
#cython: language_level=3
import numpy as np
from libc.math cimport sin, cos, tan, acos, sqrt, exp, pi, log
cimport cython
from cython.parallel import prange


DTYPE = np.float64


def shoot(double[:] u, double dx, double[:] c2, double[:] c1, double[:] c0,
          int N):
    """ Cython version of shoot.py for faster atomic DFT calculations """
    cdef Py_ssize_t u_len = u.shape[0]
    assert u_len == N
    cdef int nodes = 0
    cdef int ctp = 0
    cdef double A = 0.
    cdef Py_ssize_t i, j

    u_new = np.zeros(N, dtype=DTYPE)
    cdef double[:] u_view = u_new
    for i in range(N):
        u_view[i] = u[i]

    fp = np.zeros(N, dtype=DTYPE)
    fm = np.zeros(N, dtype=DTYPE)
    f0 = np.zeros(N, dtype=DTYPE)
    cdef double[:] fp_view = fp
    cdef double[:] fm_view = fm
    cdef double[:] f0_view = f0

    cdef int all_negative = 1
    for i in range(N):
        fp_view[i] = c2[i] / dx ** 2 + 0.5 * c1[i] / dx
        fm_view[i] = c2[i] / dx ** 2 - 0.5 * c1[i] / dx
        f0_view[i] = c0[i] - 2 * c2[i] / (dx ** 2)
        if c0[i] > 0:
            all_negative = 0

    # backward integration down to classical turning point ctp
    # (or one point beyond to get derivative)
    # If no ctp, integrate half-way
    u_view[-1] = 1.0
    u_view[-2] = u_view[-1] * f0_view[-1] / fm_view[-1]
    for i in range(N - 2 , 0, -1):
        u_view[i - 1] = -fp_view[i] * u_view[i + 1] - f0_view[i] * u_view[i]
        u_view[i - 1] /= fm_view[i]

        # numerical stability
        if abs(u_view[i - 1]) > 1e10:
            for j in range(N-1, i-2, -1):
                u_view[j] /= u_view[i-1]

        if c0[i] > 0:
            ctp = i
            break

        if all_negative > 0 and i == N // 2:
            ctp = N // 2
            break

    cdef double utp, utp1, dright
    utp = u_view[ctp]
    utp1 = u_view[ctp + 1]
    dright = (u_view[ctp + 1] - u_view[ctp - 1]) / (2 * dx)

    for i in range(1, ctp + 1):
        u_view[i + 1] = -f0_view[i] * u_view[i] - fm_view[i] * u_view[i - 1]
        u_view[i + 1] /= fp_view[i]

    cdef double dleft, scale
    dleft = (u_view[ctp + 1] - u_view[ctp - 1]) / (2 * dx)
    scale = utp / u_view[ctp]
    for i in range(ctp + 1):
        u_view[i] *= scale
    u_view[ctp + 1] = utp1  # above overrode
    dleft *= scale

    if u_view[1] < 0:
        for i in range(N):
            u_view[i] *= -1

    for i in range(ctp - 1):
        if u_view[i] * u_view[i+1] < 0:
            nodes += 1

    A = (dright - dleft) * utp
    return u_new, nodes, A, ctp


def hartree(double[:] rho, double[:] dV, double[:] r, double[:] r0, int N):
    """ Calculate Hartree potential from radial density """
    vhar = np.zeros(N, dtype=DTYPE)
    lo = np.zeros(N, dtype=DTYPE)
    hi = np.zeros(N, dtype=DTYPE)
    cdef double[:] lo_view = lo
    cdef double[:] hi_view = hi
    cdef double[:] vhar_view = vhar
    cdef Py_ssize_t i

    for i in range(1, N):
        lo_view[i] = lo_view[i-1] + dV[i-1] * rho[i-1]

    for i in range(N - 2, -1, -1):
        hi_view[i] = hi_view[i + 1] + rho[i] * dV[i] / r0[i]

    for i in range(N):
        vhar_view[i] = lo_view[i] / r[i] + hi_view[i]

    return vhar


def make_grid(double Rz, double rmin, double rmax, int nt, int nr,
              int p, int q):
    """ Constructs a double-polar grid (see slako.make_grid) """
    cdef Py_ssize_t i, j

    T = np.zeros(nt, dtype=DTYPE)
    cdef double[:] T_view = T
    for i in range(nt):
        T_view[i] = acos(-1.0) * ((i / (nt - 1)) ** p)

    R = np.zeros(nr, dtype=DTYPE)
    cdef double[:] R_view = R
    for i in range(nr):
        R_view[i] = (rmax - rmin) * ((i / (nr - 1)) ** q)

    grid = np.zeros(((nt - 1) * (nr - 1), 2), dtype=DTYPE)
    cdef double[:, :] grid_view = grid

    area = np.zeros((nt - 1) * (nr - 1), dtype=DTYPE)
    cdef double[:] area_view = area

    cdef double h = Rz / 2.
    cdef double d, d1, d2, d3, d4, z, z1, z2, z3, z4, A, A0, r0, t0, Th1, Th2

    # first calculate grid for polar centered on atom 1:
    # the z=h-like starts cutting full elements starting from point (1)
    cdef int counter = 0
    for j in range(nt - 1):
        for i in range(nr - 1):
            # corners of area element
            d1 = R_view[i+1] * sin(T_view[j])
            d2 = R_view[i] * sin(T_view[j])
            d3 = R_view[i] * sin(T_view[j+1])
            d4 = R_view[i+1] * sin(T_view[j+1])
            z1 = R_view[i+1] * cos(T_view[j])
            z2 = R_view[i] * cos(T_view[j])
            z3 = R_view[i] * cos(T_view[j+1])
            z4 = R_view[i+1] * cos(T_view[j+1])
            A0 = R_view[i+1] ** 2 - R_view[i] ** 2
            A0 *= (T_view[j+1] - T_view[j]) / 2

            if z1 <= h:
                # area fully inside region
                r0 = 0.5 * (R_view[i] + R_view[i+1])
                t0 = 0.5 * (T_view[j] + T_view[j+1])
                A = A0
            elif z1 > h and z2 <= h and z4 <= h:
                # corner 1 outside region
                Th2 = acos(h / R_view[i+1])
                r0 = 0.5 * (R_view[i] + R_view[i+1])
                t0 = 0.5 * (Th2 + T_view[j+1])
                A = A0 - 0.5 * R_view[i+1] ** 2 * (Th2 - T_view[j])
                A += 0.5 * h ** 2 * (tan(Th2) - tan(T_view[j]))
            elif z1 > h and z2 > h and z3 <= h and z4 <= h:
                # corners 1 and 2 outside region
                Th1 = acos(h / R_view[i])
                Th2 = acos(h / R_view[i+1])
                r0 = 0.5 * (R_view[i] + R_view[i+1])
                t0 = 0.5 * (Th2 + T_view[j+1])
                A = A0 * (1. - (Th1 - T_view[j]) / (T_view[j+1]- T_view[j]))
                A -= 0.5 * R_view[i+1] ** 2 * (Th2 - Th1)
                A += 0.5 * h ** 2 * (tan(Th2) - tan(Th1))
            elif z1 > h and z2 > h and z4 > h and z3 <= h:
                # only corner 3 inside region
                Th1 = acos(h / R_view[i])
                r0 = 0.5 * (R_view[i] + h / cos(T_view[j+1]))
                t0 = 0.5 * (Th1 + T_view[j+1])
                A = 0.5 * h ** 2 * (tan(T_view[j+1]) - tan(Th1))
                A -= 0.5 * R_view[i] ** 2 * (T_view[j+1] - Th1)
            elif z1 > h and z4 > h and z2 <= h and z3 <=h:
                # corners 1 and 4 outside region
                r0 = 0.5 * (R_view[i] + h / cos(T_view[j+1]))
                t0 = 0.5 * (T_view[j] + T_view[j+1])
                A = 0.5 * h ** 2 * (tan(T_view[j+1]) - tan(T_view[j]))
                A -= 0.5 * R_view[i] ** 2 * (T_view[j+1] - T_view[j])
            elif z3 > h:
                A, r0, t0 = -1, 0, 0
            else:
                raise RuntimeError('Illegal coordinates.')

            if A > 0:
                d = r0 * sin(t0)
                z = r0 * cos(t0)
                if sqrt(d ** 2 + z ** 2) < rmax:
                    if sqrt(d ** 2 + (Rz - z) ** 2) < rmax:
                        grid_view[counter, 0] = d
                        grid_view[counter, 1] = z
                        area_view[counter] = A
                        counter += 1

    grid_new = np.zeros((counter, 2), dtype=DTYPE)
    cdef double[:, :] grid_new_view = grid_new

    area_new = np.zeros(counter, dtype=DTYPE)
    cdef double[:] area_new_view = area_new

    for i in range(counter):
        grid_new_view[i, 0] = grid_view[i, 0]
        grid_new_view[i, 1] = grid_view[i, 1]
        area_new_view[i] = area_view[i]

    return grid_new, area_new


def construct_coefficients(int l, double eps, double[:] veff, double[:] dveff,
                           double[:] rgrid):
    """ Construct the coefficients for Numerov's method; see shoot.py """
    cdef int N = veff.shape[0]
    c0 = np.zeros(N, dtype=DTYPE)
    c1 = np.zeros(N, dtype=DTYPE)
    c2 = np.ones(N, dtype=DTYPE)
    cdef double[:] c0_view = c0
    cdef double[:] c1_view = c1

    cdef Py_ssize_t i
    cdef double ScR_mass
    cdef double c = 137.036
    cdef int ll = l * (l + 1)

    if dveff.shape[0] == 0:
        for i in range(N):
            c0_view[i] = -ll - 2 * rgrid[i] ** 2 * (veff[i] - eps)
            c1_view[i] = -1.
    else:
        for i in range(N):
            ScR_mass = 1 - 0.5 * (veff[i] - eps) / c ** 2
            c0_view[i] = -ll - 2 * ScR_mass * rgrid[i] ** 2 * (veff[i] - eps)
            c0_view[i] -= dveff[i] * rgrid[i] / (2 * ScR_mass * c ** 2)
            c1_view[i] = dveff[i] * rgrid[i] / (2 * ScR_mass * c ** 2) - 1

    return c0, c1, c2


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
def vxc_lda(double[::1] rho, double[::1] vxc):
    """ Calculate LDA exchange-correlation potential from the density """
    cdef int N = rho.shape[0]
    cdef double small = 1e-16
    cdef double a1 = 0.21370
    cdef double c0 = 0.031091
    cdef double c1 = 0.046644
    cdef double b1 = 0.5 / c0 * exp(-c1 / 2.0 / c0)
    cdef double b2 = 2 * c0 * b1**2
    cdef double b3 = 1.6382
    cdef double b4 = 0.49294
    cdef double aux
    cdef double rs
    cdef Py_ssize_t i

    for i in prange(N, nogil=True):
        if rho[i] < small:
            vxc[i] = 0.
        else:
            # Exchange (der=0)
            vxc[i] = -0.75 * (3 * rho[i] / pi)**(1. / 3)

            # Exchange (der=1)
            vxc[i] += rho[i] * -3. / (4 * pi) * (3 * rho[i] / pi)**(-2. / 3)

            rs = (3. / (4 * pi * rho[i]))**(1. / 3)
            aux = 2 * c0 * (b1 * sqrt(rs) + b2 * rs + b3 * rs**1.5 + b4 * rs**2)

            # Correlation (der=0)
            vxc[i] += -2 * c0 * (1 + a1 * rs) * log(1 + 1./aux)

            # Correlation (der=1)
            vxc[i] += 0.5 * c0 / (pi * rho[i] * rs**2) \
                      * (a1 * log(1 + 1./aux) \
                         + (1 + a1 * rs) * (1 + 1./aux)**-1 * (-aux**-2) \
                           * 2 * c0 * (b1 / (2 * sqrt(rs)) + b2 \
                                       + 3 * b3 * sqrt(rs) / 2 + 2 * b4 * rs))
    return vxc
