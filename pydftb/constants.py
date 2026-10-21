"""Physical constants and numerical tolerances, mirroring DFTB+.

Units: DFTB+ uses these legacy CODATA-like values (``common/constants.F90``);
ASE uses CODATA 2014, which differs at the 1e-7 relative level.
Tolerances: ``common/accuracy.F90`` of DFTB+.
"""
BOHR = 0.529177249           # Angstrom per Bohr
HARTREE = 27.2113845         # eV per Hartree
KB_HARTREE = 3.16681534524639e-6  # Boltzmann constant, Hartree / K (DFTB+ Boltzmann; the
                                  # binary gives identical results for T in K and in Hartree)
KCAL_MOL = 0.0015946683859874898  # Hartree per kcal/mol (DFTB+'s value)
AMU_AU = 1.0 / 0.00054857990945  # electron masses per atomic mass unit (DFTB+ amu__au)
AU_FS = 0.02418884326505     # femtoseconds per atomic time unit (DFTB+ au__fs)

MIN_TEMP = 1.0e-8            # Ha; T = 0 is treated as Fermi filling at this kT
MIN_HUB_DIFF = 0.3125e-5     # Ha; Hubbard U values closer than this count as equal
MIN_HUB_TOL = 1.0e-6         # Ha; smallest admissible Hubbard U
TOL_SAME_DIST = 1.0e-5       # Bohr; distances below this are "on site"
MIN_SHORT_GAMMA = 1.0e-10    # short-range gamma S is dropped beyond where it falls below this
TOL_SHORT_GAMMA = 1.0e-10    # ... and that distance is bisected to this tolerance in S
ELEC_TOL_MAX = 1.0e-7        # max. electron-count error after the Fermi search
TOL_DISPERSION = 1.0e-9      # Ha; truncation tolerance of the dispersion sums
TOL_DISP_DAMP = 1.0e-10      # Slater-Kirkwood: damping counts below this
N_SEARCH_ITER = 30           # max. bisection steps when searching a cutoff
