#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import os
import sys


class SingleAtomIntegrator:
    """
    A base class for integrations involving a single atom.

    Parameters
    ----------
    el : AtomicBase-like object
        Object with atomic properties of an element.
    txt : str, optional
        Where output should be printed.
        Use '-' for stdout (default), None for /dev/null,
        any other string for a text file, or a file handle,
    """
    def __init__(self, el, txt='-'):
        self.el = el

        if txt is None:
            self.txt = open(os.devnull, 'w')
        elif isinstance(txt, str):
            if txt == '-':
                self.txt = sys.stdout
            else:
                self.txt = open(txt, 'a')
        else:
            self.txt = txt

    def print_header(self):
        print('\n\n', file=self.txt)
        title = '{0} run for {1}'.format(self.__class__.__name__,
                                         self.el.get_symbol())
        print('*'*len(title), file=self.txt)
        print(title, file=self.txt)
        print('*'*len(title), file=self.txt)
        self.txt.flush()
        return
