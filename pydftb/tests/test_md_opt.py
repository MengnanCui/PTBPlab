"""Geometry optimisation and MD against frozen DFTB+ driver references, and charge reuse
between calculations (milestone M2f).

The JSON files are produced by ``python -m pydftb.validation.dynamics --regenerate`` (needs
dftb+); these tests only need pydftb and ASE.  Forces at a loose SCC tolerance are checked
against DFTB+ in test_reference.py (the *_loose_scc cases).
"""
import glob
import os

import numpy as np
import pytest
from ase.build import molecule

from pydftb.ase_calc import PyDFTB
from pydftb.calculator import DFTB
from pydftb.validation import dynamics, refgen
from pydftb.validation.cases import CASES_BY_NAME

from .conftest import SKF_HCOCU

REFS = sorted(glob.glob(os.path.join(dynamics.REF_DIR, "*.json")))


def _dftbplus_onsite_rule(monkeypatch):
    """DFTB+ leaves the onsite corrections out of the first SCC iteration of every geometry
    step (main.F90: iSCCIter > 1 .or. tReadChrg); pydftb counts reused charges as read in."""
    scc, pots, first = DFTB._scc, DFTB._block_potentials, []

    def _scc(self, *args):
        first.append(True)
        return scc(self, *args)

    def _pots(self, st, qvec, onsite=True):
        return pots(self, st, qvec, onsite=onsite and not (first and first.pop()))

    monkeypatch.setattr(DFTB, "_scc", _scc)
    monkeypatch.setattr(DFTB, "_block_potentials", _pots)


@pytest.mark.parametrize("path", REFS, ids=[os.path.basename(p)[:-5] for p in REFS])
def test_against_dftbplus_driver(path, monkeypatch):
    doc = dynamics.load(os.path.basename(path)[:-5])
    why = refgen.missing_requirement(doc)
    if why:
        pytest.skip(why)
    r = dynamics.compare(doc)
    assert r["ok"], f"{doc['case']}: deviations {r['dev']} exceed {doc['tolerances']}"
    if doc["kind"] == "md":
        # charge reuse gives DFTB+'s SCC starting points: the same iterations in every step, give
        # or take one where the last SCC error lands at the tolerance (O2: 3 pairs of steps swap)
        n_py, n_ref = np.array(r["result"]["n_scc"]), np.array(doc["reference"]["n_scc"])
        if CASES_BY_NAME[doc["base"]].onsite is not None:
            # pydftb starts every step with the onsite corrections in H: never more iterations
            # (water: 463 against 557); with DFTB+'s rule the iterations of DFTB+
            assert (n_py <= n_ref).all() and n_py.sum() < n_ref.sum()
            _dftbplus_onsite_rule(monkeypatch)
            n_py = np.array(dynamics.compare(doc)["result"]["n_scc"])
        assert np.abs(n_py - n_ref).max() <= 1 and abs(n_py.sum() - n_ref.sum()) <= 2


def test_reference_set_is_complete():
    names = {os.path.basename(p)[:-5] for p in REFS}
    missing = {c.name for c in dynamics.DYN_CASES} - names
    assert not missing, f"no reference JSON for {missing}; run dynamics --regenerate"


def test_reuse_charges():
    """The second geometry starts from the first one's charges: fewer SCC iterations, the
    same result; another molecule (other type labels) starts from neutral atoms again."""
    a = molecule("H2O")
    a.rattle(0.05, seed=1)
    b = a.copy()
    b.positions[0] += [0.02, -0.01, 0.0]
    out = {}
    for reuse in (False, True):
        calc = PyDFTB(SKF_HCOCU, scc_tol=1e-10, reuse_charges=reuse)
        for atoms in (a.copy(), b.copy()):
            atoms.calc = calc
            forces = atoms.get_forces()
        out[reuse] = (calc.raw.n_scc, forces, calc.raw.free_energy)
    assert out[True][0] < out[False][0]
    assert np.abs(out[True][1] - out[False][1]).max() < 1e-7          # eV/A
    assert abs(out[True][2] - out[False][2]) < 1e-10
    m = molecule("CH3OH")
    m.calc = calc                                                       # reuse_charges=True
    fresh = PyDFTB(SKF_HCOCU, scc_tol=1e-10)
    assert m.get_potential_energy() == pytest.approx(fresh.get_potential_energy(m.copy()), abs=1e-10)
    assert calc.raw.n_scc == fresh.raw.n_scc
