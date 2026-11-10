=============
Release notes
=============


Development version
===================


Version 2.1.0
=============

08 September 2026

* The Python, NumPy and SciPy versions are now required to be at least
  v3.10, v2.0.0 and v1.14.0, respectively.

* Additional dependencies for e.g. running the testsuite are
  now only included when installing the `dev` variant.

* The STO fitting routine now uses the COBYQA method instead of COBYLA
  (see `Issue #8 <https://gitlab.com/mvdb/hotcent/-/issues/8>`__).

* Added (experimental) support for dszp, dz2p, tz2p and tz3p
  basis sets. These are mainly intended for NAO-DFT applications,
  rather than AITB.

* Added an "--rchar-pol-rel" option to hotcent-basis.
  By default, p-type polarization orbitals in the 'quasi-gaussian'
  scheme will furthermore be generated with a scaling factor of
  0.33 (instead of 0.25).

* Added (experimental) support for pseudopotentials in the
  [PSML](https://siesta-project.github.io/psml-docs/page/01_the_psml_format/)
  format.

* Added (experimental) support for atomic DFT calculations using
  PSML pseudopotentials with semicore states included in the valence.
  The main use case for this is generating ion files for Siesta.

* The quasi-gaussian scheme now more closely follows the GPAW
  implementation where it is based on a cutoff radius of the
  highest subshell with angular momentum `l_pol - 1` (instead
  of looping over all valence subshells and taking the maximum).
  This introduces only limited (if any) backwards incompatibility
  given that this change is normally only noticeable with semicore
  states in the valence.


Version 2.0.1
=============

02 March 2025

* The hotcent.offsite_twocenter.Offsite2cTable.write method now accepts
  a 'filename_template' keyword argument which can be used to similar effect
  as the corresponding 'filename' keyword argument from Hotcent <= v1.

* Bugfix: in all-electron calculations as well as pseudopotential
  calculations with non-linear core corrections, there was an error in
  the Vxc integral contribution to the two- and three-center repulsion.


Version 2.0.0
=============

18 March 2024

* Extended the Slater-Koster tabulation procedure to *f* electrons.
  If one of the two atoms contains *f* electrons in the valence,
  the SKF files are now written in the 'extended' format instead
  of the 'simple' format.

* The 'perturbative' confinement scheme is now the default for
  AtomicDFT calculations (the 'perturbative_confinement' keyword).

* Maintenance related to changes in ASE, Matplotlib and Libxc.

* Bugfix: there was an off-by-one error in the number of zero-valued
  integrals printed in the Slater-Koster files for distances below
  'rmin'.

* The Python version is now required to be >= v3.6.
  The ASE dependency now needs to be >= v22.0 and
  the (optional) Libxc dependency >= v5.1.
  PyYAML and pytest are two new dependencies.

* Bugfix: gradient corrections to the exchange-correlation potential
  for two-center integrals were wrong.

* The hotcent.slako.SlaterKosterTable class has been moved to
  hotcent.offsite_twocenter.Offsite2cTable. The default
  '<el1>-<el2>_no_repulsion.skf' template for the corresponding
  Slater-Koster file name has furthermore been changed to
  '<el1>-<el2>_offsite2c.skf' and the default superposition scheme
  is changed to density superposition.

* Dropped support for writing '.par' Slater-Koster files
  (only the SKF format remains).

* Added functionality for non-minimal basis sets (see
  AtomicBase.generate_nonminimal_basis()). Currently basis sets up to
  triple-zeta with polarization (TZP) can be generated.

* Hubbard parameters (U values) are now calculated as derivatives of
  the corresponding Hamiltonian matrix elements in the isolated atom
  (no longer as second derivatives of the total energy). The present
  version may hence produce U values that are slightly different from
  previous versions.

* Added the possibility to calculate spin constants (W values).
  This requires the otherwise optional Libxc dependency.

* Implemented analytical methods for Hubbard value and spin constant
  calculations based on the (Hartree-)XC kernel (instead of finite
  differentiation of atomic eigenvalues).

* Added the possibility to calculate higher-order corrections to
  Hamiltonian and (Hartree-)XC kernel matrix elements:

  - three-center expansion for off-site H integrals
  - two- and three-center expansions for on-site H integrals
  - two-center expansion for on- and off-site "U" and "W"
    integrals associated with difference and magnetization densities,
    respectively.

* Added the possibility to calculate the needed (Hartree-)XC kernel
  and (Giese-York) mapping integrals when the difference and magnetization
  densities are expanded in a (possibly multipolar) auxiliary basis set.

* Changed the tail smoothening procedure to a moving average for
  simplicity and robustness.

* Added the 'hotcent-basis', 'hotcent-concat' and 'hotcent-tables'
  command-line tools.

* Renamed the `hotcent.tools` submodule to `hotcent.confinement_optimization`.

* Updated the existing two tutorials and added `Tutorial 3` about generating
  3cTB-GY parameter sets.

* Added an upper bound (default: 1e4 Ha) to PowerConfinement potentials
  to avoid problems for higher *s* values (e.g. *s* >= 4), see issue #4.


Version 1.0
===========

26 December 2019

* The GPAW-based atomic DFT calculator has been removed, as it was
  simply no longer needed (the native calculator now runs as
  fast and is easier to extend with more functionality).

* The native atomic DFT calculator has been renamed to 'AtomicDFT'
  and now resides in hotcent.atomic_dft.py. Its 'xcname' keyword
  argument (for the exchange-correlation functional) has furthermore
  been shortened to 'xc'.

* Calculation of Hubbard values (see examples/hubbard.py).

* Optional C-extensions allow for significantly faster atomic
  atomic DFT calculations (8-10x speedup) and construction of
  Slater-Koster tables (2-3x speedup).

* A new `stride` option for SlaterKosterTable.run() with default = 1.
  Setting higher integer values for `stride` means that the
  H and S integrals are only be explicitly calculated every
  `stride` points and will then be mapped on the final, denser grid
  using cubic spline interpolation. Use with care.

* No more backwards compatibility with Python2 (only Python3).

* The SlaterKosterTable.run() method now requires rmin, dr and N
  as arguments to specify the interatomic distances for which the
  Slater-Koster integrals are tabulated (rmin being the minimal
  distance, dr the grid spacing dr, and N the number of grid points).

* Major restructuring of hotcent.tools for fitting the confinement
  parameters, with also a new tutorial on this topic (`tutorial_2.ipynb`).


Version 0.9
===========

03 December 2019

* Start of versioning.
