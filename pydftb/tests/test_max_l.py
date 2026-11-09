"""MaxAngularMomentum (``max_l``): the default is the highest occupied shell (ASE's
``read_max_angular_momentum``, the basis PTBP fits with), with a warning when the SKF has
integrals for a higher shell; "highest" takes every parametrised shell; DFTB+ letters work."""
import gzip
import os
import warnings

import numpy as np
import pytest
from ase.build import molecule

from .conftest import SKF_HCOCU
from pydftb import ParamSet
from pydftb.ase_calc import PyDFTB
from pydftb.dftbplus_calc import _max_l
from pydftb.skf import (UnusedShellsWarning, _expand_fortran, default_max_l, max_l_available,
                        max_l_from_skf, parse_max_l)


def _plain_copy(dst, empty_d_of_cu=False):
    """The H-C-O-Cu set uncompressed; optionally Cu-Cu with f_d = 0 (d parametrised, empty)."""
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(SKF_HCOCU):
        with gzip.open(os.path.join(SKF_HCOCU, name), "rt") as fh:
            text = fh.read()
        if empty_d_of_cu and name == "Cu-Cu.skf.gz":
            lines = text.splitlines(keepends=True)
            vals = _expand_fortran(lines[1])
            vals[7] = 0.0                                   # occupation f_d
            lines[1] = " ".join(repr(float(v)) for v in vals) + "\n"
            text = "".join(lines)
        with open(os.path.join(dst, name[:-3]), "w") as fh:
            fh.write(text)
    return dst


def test_highest_shell_of_the_ptbp_files():
    path = lambda a: os.path.join(SKF_HCOCU, f"{a}-{a}.skf.gz")
    assert [max_l_available(path(a)) for a in ("H", "C", "O", "Cu")] == [0, 1, 1, 2]


def test_highest_and_occupied_rules_differ_for_an_empty_parametrised_shell(tmp_path):
    d = _plain_copy(str(tmp_path / "skf"), empty_d_of_cu=True)
    cu = os.path.join(d, "Cu-Cu.skf")
    assert max_l_from_skf(cu) == 0                          # f_p = f_d = 0: s only
    assert max_l_available(cu) == 2                         # but the d integrals are there
    assert default_max_l(cu, "highest") == 2
    with pytest.warns(UnusedShellsWarning, match="also has integrals up to d"):
        assert default_max_l(cu) == 0
    assert ParamSet(d).skset(["Cu"]).max_l["Cu"] == 0
    assert ParamSet(d, max_l="highest").skset(["Cu"]).max_l["Cu"] == 2
    # the DFTB+ engine writes the same MaxAngularMomentum
    assert _max_l(d, ["Cu", "O"], None, {}) == {"Cu": 0, "O": 1}
    assert _max_l(d, ["Cu", "O"], "highest", {}) == {"Cu": 2, "O": 1}


def test_unused_shells_warn_once_per_file_and_not_when_given(tmp_path):
    d = _plain_copy(str(tmp_path / "skf"), empty_d_of_cu=True)
    cu = os.path.join(d, "Cu-Cu.skf")
    with pytest.warns(UnusedShellsWarning):
        default_max_l(cu)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        default_max_l(cu)                                    # once per file
        ParamSet(d, max_l={"Cu": "s"}).skset(["Cu"])         # explicit: no rule, no warning
        default_max_l(os.path.join(d, "O-O.skf"))            # nothing unused
        ParamSet(d, max_l="occupied").skset(["Cu", "O"])     # explicit rule: no warning
        default_max_l(cu, "occupied", warn=False)


def test_parse_max_l_takes_dftbplus_letters_and_a_default_rule():
    assert parse_max_l(None) == ("occupied", {})
    assert parse_max_l("highest") == ("highest", {})
    assert parse_max_l({"S": "d", "H": 0}) == ("occupied", {"S": 2, "H": 0})
    assert parse_max_l({"default": "highest", "Li": "S"}) == ("highest", {"Li": 0})
    for bad in ("lowest", {"S": "g"}, {"S": 4}, {"default": "x"}):
        with pytest.raises(ValueError):
            parse_max_l(bad)


def test_explicit_max_l_overrides_the_rule(tmp_path):
    d = _plain_copy(str(tmp_path / "skf"), empty_d_of_cu=True)
    sk = ParamSet(d, max_l={"Cu": "p"}).skset(["Cu", "O"])
    assert sk.max_l == {"Cu": 1, "O": 1}
    sk = ParamSet(d, max_l={"default": "highest", "O": "s"}).skset(["Cu", "O"])
    assert sk.max_l == {"Cu": 2, "O": 0}


def test_default_unchanged_where_the_rules_agree():
    """H, C, O, Cu of PTBP: the highest shell is the occupied one, so both rules agree."""
    w = molecule("H2O")
    w.calc = PyDFTB(SKF_HCOCU)
    e_default = w.get_potential_energy()
    w.calc = PyDFTB(SKF_HCOCU, max_l="highest")
    assert w.get_potential_energy() == e_default


def test_the_dropped_shell_changes_the_energy(tmp_path):
    d = _plain_copy(str(tmp_path / "skf"), empty_d_of_cu=True)
    from ase import Atoms
    cu2 = Atoms("Cu2", positions=[[0, 0, 0], [0, 0, 2.3]])
    cu2.calc = PyDFTB(d, max_l="highest", smearing=("fermi", 1000.0))
    e_d = cu2.get_potential_energy()
    cu2.calc = PyDFTB(d, max_l={"Cu": "s"}, smearing=("fermi", 1000.0))
    assert abs(cu2.get_potential_energy() - e_d) > 1e-3
    assert np.isfinite(e_d)
