# Solvation test data: provenance

## `param_gbsa_test.txt`

A small parameter file in the xtb format that DFTB+'s `GeneralisedBorn { ParamFile = ... }`
reads (`pydftb.solvation.GeneralizedBorn.from_param_file` reads the same format), written for
the tests of milestone M6c:

- Header (8 lines): dielectric constant 80.2, molecular mass 18.0, density 1.0, Born scale,
  probe radius (Angstrom), free energy shift (kcal/mol), Born offset (0.1 Angstrom), unused.
- Then one line per element Z = 1..94: surface tension, descreening, hydrogen-bond strength.
- The numbers of the header and of the H, C, N and O lines are those of the GBSA water
  example in the DFTB+ 25.1 manual (section "GeneralisedBorn", GFN2-xTB/GBSA(water)), written
  in this file's units: the probe radius and the Born offset are read as Angstrom and
  0.1 Angstrom here, whereas the manual's HSD example gives the same digits in Bohr. Every
  other element is zero tension, unit descreening and no hydrogen bond; they are
  placeholders, not parameters for those elements.

Used by the reference case `ch3oh_gbsa_paramfile` (`pydftb/validation/cases.py`) and by
`tests/test_solvation.py::test_param_file`.
