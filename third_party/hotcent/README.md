# Hotcent

*Hotcent* is a tool for generating tight-binding parameter files
starting from atomic DFT calculations with spherical confinement.

Hotcent was originally based on parts of the [Hotbit](
https://github.com/pekkosk/hotbit/) code but has much evolved
since then.

Hotcent was initially developed as part of the following study:

M. Van den Bossche, J. Chem. Phys. A. **2019**, 123 (13), 3038-3045
[(doi)](https://dx.doi.org/10.1021/acs.jpca.9b00927).

The code has then been considerably expanded and reworked for:

M. Van den Bossche, J. Chem. Theory Comput. **2024**
[(doi)](https://doi.org/10.1021/acs.jctc.4c00018).

Please consider citing the 2019 paper when using Hotcent version v1.0
(and earlier) and the 2024 paper for later Hotcent versions.


## Features

Hotcent can generate parameters representing:

- [x] up-to-three-center contributions to the zeroth-order Hamiltonian
      matrix elements
- [x] up-to-three-center contributions to the repulsive energy
- [x] up-to-two-center contributions to the U and W kernel matrix elements
- [x] Giese-York mapping to a (multipolar) auxiliary basis (up to quadrupoles)

As such, Hotcent can generate parameter files for semi-empirical
tight-binding as well as ab-initio tight-binding calculations.
For more information please consult the [2024 paper](
https://doi.org/10.1021/acs.jctc.4c00018).

With regards to exchange-correlation functionals, the PW92
(LDA) functional is natively available, and other LDA/GGA functionals
can be applied [through integration with Libxc](#libxc).
Hybrid and meta-GGA functionals cannot currently be used in Hotcent.


## Installation

Clone or download Hotcent and install with e.g.
```shell
pip install .
```

If you want or need to first regenerate the C extensions,
you need to have Cython installed and invoke it as e.g.
```shell
cython --module-name=_hotcent hotcent/extensions.pyx
```

For developing Hotcent it is more convenient to install in editable mode,
with additional dependencies for running the testsuite:
```shell
pip install -e .[dev]
```

Aside from the Python module and the compiled extensions (`_hotcent.so`),
the installation should also provide the `hotcent-basis`, `hotcent-concat`
and `hotcent-setup` command line tools.


### Libxc

If you want to use functionals other than the PW92 LDA, the [pylibxc](
https://www.tddft.org/programs/libxc/installation/#python-library) module
needs to be available, which provides a Python interface to all
Libxc functionals. A recent Libxc version is required (>= v5.1).


## Testing

The test suite makes use of the [pytest](https://docs.pytest.org) framework
and requires that pylibxc is installed.

Example usage:
```shell
cd tests
pytest                      # run all the tests with default options
pytest -s                   # show test stdout/sterr output (disable capturing)
pytest test_S.py            # only run the tests in the test_S.py file
pytest --collect-only       # show all generated test IDs
pytest -k on1c              # only run the tests with names matching '*on1c*'
```
