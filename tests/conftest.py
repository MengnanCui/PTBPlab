"""Shared pytest fixtures.

Importing PTBP triggers `utils._compat` which forces the multiprocessing
start method to 'fork' on macOS — make sure that import happens before
any test runs by simply importing utils here.
"""
import utils  # noqa: F401  applies the fork-mp + ASE-3.28 shims
