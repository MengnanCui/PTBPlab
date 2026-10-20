"""Physical constants and numerical tolerances, mirroring DFTB+.

Units: DFTB+ uses these legacy CODATA-like values (``common/constants.F90``);
ASE uses CODATA 2014, which differs at the 1e-7 relative level.
Tolerances: ``common/accuracy.F90`` of DFTB+.
"""
BOHR = 0.529177249           # Angstrom per Bohr
HARTREE = 27.2113845         # eV per Hartree
KB_HARTREE = 3.166811429e-6  # Boltzmann constant, Hartree / K

MIN_TEMP = 1.0e-8            # Ha; T = 0 is treated as Fermi filling at this kT
MIN_HUB_DIFF = 0.3125e-5     # Ha; Hubbard U values closer than this count as equal
MIN_HUB_TOL = 1.0e-6         # Ha; smallest admissible Hubbard U
TOL_SAME_DIST = 1.0e-5       # Bohr; distances below this are "on site"
MIN_SHORT_GAMMA = 1.0e-10    # cut-off value for the short-range gamma
ELEC_TOL_MAX = 1.0e-7        # max. electron-count error after the Fermi search
