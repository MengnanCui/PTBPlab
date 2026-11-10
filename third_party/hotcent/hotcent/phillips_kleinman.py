#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
from ase.data import atomic_numbers
from hotcent.interpolation import build_interpolator
from hotcent.separable_pseudopotential import SeparablePP


class PhillipsKleinmanPP(SeparablePP):
    """
    Class representing a Phillips-Kleinman(-like) pseudopotential
    (doi:10.1103/PhysRev.116.287).

    Parameters
    ----------
    symbol : str
        The chemical symbol.

    Other Parameters
    ----------------
    Additional keyword arguments for SeparablePP initialization.
    """
    def __init__(self, symbol, **kwargs):
        SeparablePP.__init__(self, **kwargs)
        self.symbol = symbol
        self.Z = atomic_numbers[self.symbol]

    def all_zero_onsite_overlaps(self):
        """ Returns whether all on-site overlaps of the (valence)
        states with the PP projectors are zero.
        """
        return True

    def local_potential(self, r):
        """ Returns the local part of the pseudopotential at r. """
        return -self.Z / r

    def build_projectors(self, e3):
        """ Build the pseudopotential projectors and the
        associated energies.
        """
        assert e3.get_symbol() == self.symbol
        self.projectors = {}
        self.energies = {}

        for n, l, nl in e3.list_states():
            if nl not in e3.valence:
                self.projectors[nl] = build_interpolator(e3.rgrid, e3.Rnlg[nl])
                self.energies[nl] = -e3.get_eigenvalue(nl)

                for valence in e3.basis_sets:
                    for nl2 in valence:
                        self.overlap_onsite[(nl, nl2)] = 0.
