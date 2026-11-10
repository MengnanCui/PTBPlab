#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import os
import sys
import numpy as np
from hotcent.timing import Timer
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import _hotcent
except ModuleNotFoundError:
    print('Warning: C-extensions not available')
    _hotcent = None


class MultiAtomIntegrator:
    """
    A base class for integrations involving multiple atoms.

    Parameters
    ----------
    ela, elb : AtomicBase-like objects
        Objects with atomic properties for two atoms.
    grid_type : str
        Type of 2D integration grid in the XZ plane.
        Choose between 'bipolar' and 'monopolar'.
    txt : str, optional
        Where output should be printed.
        Use '-' for stdout (default), None for /dev/null,
        any other string for a text file, or a file handle,
    timing : bool, optional
        Whether to print a timing summary before destruction
        (default: False).
    """
    def __init__(self, ela, elb, grid_type, txt='-', timing=False):
        self.ela = ela
        self.elb = elb

        assert grid_type in ['bipolar', 'monopolar']
        self.grid_type = grid_type

        if ela.get_symbol() != elb.get_symbol():
            self.nel = 2
            self.pairs = [(ela, elb), (elb, ela)]
            self.elements = [ela, elb]
        else:
            self.nel = 1
            self.pairs = [(ela, elb)]
            self.elements = [ela]

        if txt is None:
            self.txt = open(os.devnull, 'w')
        elif isinstance(txt, str):
            if txt == '-':
                self.txt = sys.stdout
            else:
                self.txt = open(txt, 'a')
        else:
            self.txt = txt

        self.timer = Timer('MultiAtomIntegrator', txt=self.txt, enabled=timing)

    def __del__(self):
        self.timer.summary()

    def print_header(self, suffix=''):
        print('\n\n', file=self.txt)
        title = '{0} run for {1}-{2}{3}'.format(self.__class__.__name__,
                                                self.ela.get_symbol(),
                                                self.elb.get_symbol(),
                                                suffix)
        print('*'*len(title), file=self.txt)
        print(title, file=self.txt)
        print('*'*len(title), file=self.txt)
        self.txt.flush()
        return

    def get_range(self, wf_limit):
        """
        Get the maximal radius beyond which all involved valence
        subshell magnitudes are below the given value.

        Parameters
        ----------
        wf_limit : float
            Threshold for the radial components of the valence
            subshells.

        Returns
        -------
        wf_range : float
            The largest radius r such that Rnl(r) < wf_limit.
        """
        wf_range = 0.

        for el in self.elements:
            r = max([el.get_wf_range(nl, wf_limit)
                     for nl in el.get_valence_subshells()])
            print('Wave function range for %s = %.5f a0' % (el.get_symbol(), r),
                  file=self.txt)
            wf_range = max(r, wf_range)

        assert wf_range < 20, 'Wave function range exceeds 20 Bohr radii. ' \
                              'Decrease wflimit?'
        return wf_range

    def make_grid(self, *args, **kwargs):
        """
        Wraps around make_bipolar_grid().
        """
        self.timer.start('make_grid')
        if self.grid_type == 'bipolar':
            grid = make_bipolar_grid(*args, **kwargs)
        elif self.grid_type == 'monopolar':
            grid = make_monopolar_grid(*args, **kwargs)
        self.timer.stop('make_grid')
        return grid


def make_bipolar_grid(Rz, wf_range, nt, nr, p=2, q=2, view=False):
    r"""
    Construct a bipolar grid.

    Parameters
    ----------
    Rz : float
        Distance between the two grid centers (element #1 is at origin,
        element #2 is at z=Rz).
    wf_range : float
        Maximal radial extent of the grid.
    nt : int
        Number of theta grid points.
    nr : int
        Number of radial grid points.
    p : int
        Power describing the angular distribution of grid points
        (larger puts more weight towards theta=0).
    q : int
        Power describing the radial disribution of grid points
        (larger puts more weight towards centers).
    view : bool
        View the distribution of grid points with matplotlib.

    Returns
    -------
    grid : np.ndarray
        2D array with the x- and z-coordinates of the grid points.
    area : np.ndarray
        1D array with the area element associated with each grid point.


    Plane at R/2 divides two polar grids.


     ^ (z-axis)
     |--------_____               phi_j
     |       /     ----__         *
     |      /            \       /  *
     |     /               \    /  X *                X=coordinates of the center of area element(z,d),
     |    /                  \  \-----* phi_(j+1)     area=(r_(i+1)**2-r_i**2)*(phi_(j+1)-phi_j)/2
     |   /                    \  r_i   r_(i+1)
     |  /                      \
     | /                       |
     *2------------------------|           polar centered on atom 2
     | \                       |
     |  \                     /                                                     1
     |   \                   /                                                     /  \
     |-------------------------- z=h -line         ordering of sector slice       /     \
     |   /                   \                                      points:      /        \
     |  /                     \                                                 /          \
     | /                       |                                               /     0       4
     *1------------------------|--->      polar centered on atom 1            2            /
     | \                       |    (r_perpendicular (xy-plane) = 'd-axis')    \        /
     |  \                      /                                                 \   /
     |   \                    /                                                    3
     |    \                  /
     |     \               /
     |      \           /
     |       \ ___ ---
     |---------
    """
    rmin, rmax = 1e-7, wf_range
    h = Rz / 2

    if _hotcent is not None:
        grid, area = _hotcent.make_grid(Rz, rmin, rmax, nt, nr, p, q)
    else:
        max_range = wf_range
        T = np.linspace(0, 1, nt) ** p * np.pi
        R = rmin + np.linspace(0, 1, nr) ** q * (rmax - rmin)

        area = np.array([])
        d = np.array([])
        z = np.array([])

        # first calculate grid for polar centered on atom 1:
        # the z=h-like starts cutting full elements starting from point (1)
        Tj0 = T[:nt - 1]
        Tj1 = T[1: nt]

        for i in range(nr - 1):
            # corners of area element
            z1 = R[i + 1] * np.cos(Tj0)
            z2 = R[i] * np.cos(Tj0)
            z3 = R[i] * np.cos(Tj1)
            z4 = R[i + 1] * np.cos(Tj1)

            cond_list = [z1 <= h,  # area fully inside region
                 (z1 > h) * (z2 <= h) * (z4 <= h),  # corner 1 outside region
                 (z1 > h) * (z2 > h) * (z3 <= h) * (z4 <= h),  # 1 & 2 outside
                 (z1 > h) * (z2 > h) * (z3 <= h) * (z4 > h),  # only 3 inside
                 (z1 > h) * (z2 <= h) * (z3 <= h) * (z4 > h),  # 1 & 4 outside
                 (z1 > h) * (z3 > h) * ~((z2 <= h) * (z4 > h))]

            r0_list = [0.5 * (R[i] + R[i + 1]),
                       0.5 * (R[i] + R[i + 1]),
                       0.5 * (R[i] + R[i + 1]),
                       lambda x: 0.5 * (R[i] + h / np.cos(x)),
                       lambda x: 0.5 * (R[i] + h / np.cos(x)),
                       0,
                       np.nan]
            r0 = np.piecewise(Tj1, cond_list, r0_list)

            Th0 = np.piecewise(h / R[i], [np.abs(h / R[i]) > 1],
                               [np.nan, lambda x: np.arccos(x)])
            Th1 = np.piecewise(h / R[i + 1], [np.abs(h / R[i + 1]) > 1],
                               [np.nan, lambda x: np.arccos(x)])

            t0_list = [lambda x: 0.5 * x,
                       0.5 * Th1,
                       0.5 * Th1,
                       0.5 * Th0,
                       lambda x: 0.5 * x,
                       0,
                       np.nan]
            t0 = 0.5 * Tj1
            t0 += np.piecewise(Tj0, cond_list, t0_list)

            rr = 0.5 * (R[i + 1] ** 2 - R[i] ** 2)
            A_list0 = [lambda x: rr * -x,
                       lambda x: rr * -x - 0.5 * R[i + 1] ** 2 * (Th1 - x) \
                                 + 0.5 * h ** 2 * (np.tan(Th1) - np.tan(x)),
                       lambda x: rr * -x - (rr * -x + 0.5 * R[i + 1] ** 2 \
                                 * (Th1 - Th0)),
                       0.,
                       lambda x: 0.5 * h ** 2 * -np.tan(x) \
                                 - 0.5 * R[i] ** 2 * -x,
                       -1,
                       np.nan]
            A = np.piecewise(Tj0, cond_list, A_list0)

            A_list1 = [lambda x: rr * x,
                       lambda x: rr * x,
                       lambda x: rr * x - (rr * Th0 - 0.5 * h ** 2 \
                                 * (np.tan(Th1) - np.tan(Th0))),
                       lambda x: 0.5 * h ** 2 * (np.tan(x) - np.tan(Th0)) \
                                 - 0.5 * R[i] ** 2 * (x - Th0),
                       lambda x: 0.5 * h ** 2 * np.tan(x) \
                                 - 0.5 * R[i] ** 2 * x,
                       0,
                       np.nan]
            A += np.piecewise(Tj1, cond_list, A_list1)

            dd = r0 * np.sin(t0)
            zz = r0 * np.cos(t0)
            select = np.sqrt(dd ** 2 + zz ** 2) < max_range
            select *= np.sqrt(dd ** 2 + (Rz - zz) ** 2) < max_range
            select *= A > 0
            area = np.hstack((area, A[select]))
            d = np.hstack((d, dd[select]))
            z = np.hstack((z, zz[select]))
        grid = np.array([d, z]).T

    # calculate the polar centered on atom 2 by mirroring the other grid
    grid2 = grid.copy()
    grid2[:, 1] = -grid[:, 1]
    shift = np.zeros_like(grid)
    shift[:, 1] = 2 * h
    grid = np.concatenate((grid, grid2 + shift))
    area = np.concatenate((area, area))

    if view:
        assert plt is not None, 'Matplotlib could not be imported!'
        plt.plot([h, h ,h])
        plt.scatter(grid[:, 0], grid[:, 1], s=10 * area / np.max(area))
        plt.show()
        plt.clf()

    return grid, area


def make_monopolar_grid(wf_range, nt, nr, q=2, view=False):
    """
    Construct a simple monopolar grid.

    Parameters
    ----------
    See make_bipolar_grid().

    Returns
    -------
    See make_bipolar_grid().
    """
    rmin, rmax = 1e-7, wf_range

    R = rmin + np.linspace(0., 1., num=nr, endpoint=True)**q * (rmax - rmin)

    T = np.linspace(0.5/nt, 1.-0.5/nt, num=nt, endpoint=True) * np.pi
    s = np.sin(T)
    c = np.cos(T)

    N = (nr - 1) * nt
    x = np.zeros(N)
    z = np.zeros(N)
    area = np.zeros(N)

    counter = 0
    for i in range(nr - 1):
        r = 0.5 * (R[i] + R[i+1])
        x[counter:counter+nt] = r * s
        z[counter:counter+nt] = r * c
        area[counter:counter+nt] = np.pi * (R[i+1]**2 - R[i]**2) / (2. * nt)
        counter += nt

    grid = np.array([x, z]).T

    if view:
        assert plt is not None, 'Matplotlib could not be imported!'
        plt.scatter(grid[:, 0], grid[:, 1], s=10 * area / np.max(area))
        plt.grid()
        plt.show()
        plt.clf()

    return grid, area
