#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    plt = None


class RadialGrid:
    def __init__(self,grid):
        """
        mode
        ----

        rmin                                                        rmax
        r[0]     r[1]      r[2]            ...                     r[N-1] grid
        I----'----I----'----I----'----I----'----I----'----I----'----I
           r0[0]     r0[1]     r0[2]       ...              r0[N-2]     r0grid
           dV[0]     dV[1]     dV[2]       ...              dV[N-2]         dV

           dV[i] is volume element of shell between r[i] and r[i+1]
        """
        N = len(grid)
        self.N = N
        self.grid = grid
        self.dr = self.grid[1:N] - self.grid[0:N - 1]
        self.r0 = self.grid[0:N - 1] + self.dr / 2
        # first dV is sphere (treat separately), others are shells
        self.dV = 4 * np.pi * self.r0 ** 2 * self.dr
        rmax = grid[-1]
        self.dV *= (4 * np.pi * rmax ** 3 / 3) / sum(self.dV)

    def get_grid(self):
        """ Return the whole radial grid. """
        return self.grid

    def get_N(self):
        """ Return the number of grid points. """
        return self.N

    def get_drgrid(self):
        """ Return the grid spacings (array of length N-1). """
        return self.dr

    def get_r0grid(self):
        """ Return the mid-points between grid spacings
        (array of length N-1).
        """
        return self.r0

    def get_dvolumes(self):
        """ Return dV(r) = 4 * pi * r ** 2 * dr. """
        return self.dV

    def plot(self, screen=True):
        rgrid = self.get_grid()
        plt.scatter(list(range(len(rgrid))), rgrid)
        if screen:
            plt.show()

    def integrate(self, f, use_dV=False):
        """ Integrate function f (given with N grid points).
        int_rmin^rmax f*dr (use_dv=False) or int_rmin^rmax*f dV (use_dV=True)
        """
        if use_dV:
            return ((f[0:self.N - 1] + f[1:self.N]) * self.dV).sum() * 0.5
        else:
            return ((f[0:self.N - 1] + f[1:self.N]) * self.dr).sum() * 0.5

    def gradient(self, f):
        return np.gradient(f, self.grid)

    def divergence(self, f):
        return (1. / self.grid ** 2) * self.gradient(self.grid ** 2 * f)
