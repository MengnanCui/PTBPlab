#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import hashlib
from ase.data import chemical_symbols


def get_file_checksum(filename, algorithm='sha256'):
    """
    Computes the chosen checksum for the given file.

    Parameters
    ----------
    filename : str
        Path to the file for which to compute the checksum.

    algorithm : str, optional
        Name of the (hashlib) checksum algorithm to use.

    Returns
    -------
    checksum : str
        The checksum itself.
    """
    h = hashlib.new(algorithm)

    with open(filename, 'rb') as f:
        content = f.read()
        h.update(content)

    checksum = h.hexdigest()
    return checksum


def verify_chemical_symbols(*symbols):
    """
    Checks whether the given chemical symbols correspond to known
    elements.

    Parameters
    ----------
    symbols : str
        Chemical symbol(s).
    """
    for symbol in symbols:
        assert symbol in chemical_symbols and symbol != 'X', \
               'Unknown element: {0}'.format(symbol)
    return
