"""pydftb vs frozen DFTB+ references (pydftb/tests/data/ref/*.json).

The JSON files are produced by ``python -m pydftb.validation.refgen
--regenerate`` (needs dftb+); this test only needs pydftb.
"""
import glob
import os

import pytest

from pydftb.validation import refgen

REFS = sorted(glob.glob(os.path.join(refgen.REF_DIR, "*.json")))


@pytest.mark.parametrize("path", REFS, ids=[os.path.basename(p)[:-5] for p in REFS])
def test_against_dftbplus_reference(path):
    doc = refgen.load(os.path.basename(path)[:-5])
    r = refgen.compare(doc)
    assert r["ok"], f"{doc['case']}: deviations {r['dev']} exceed {doc['tolerances']}"


def test_reference_set_is_complete():
    names = {os.path.basename(p)[:-5] for p in REFS}
    missing = {c.name for c in refgen.CASES} - names
    assert not missing, f"no reference JSON for {missing}; run refgen --regenerate"


@pytest.mark.dftbplus
def test_live_dftbplus_water():
    """Fresh DFTB+ run (only when the binary is available)."""
    case = refgen.CASES_BY_NAME["ch3oh"]
    inp = refgen._inputs(case)
    ref = refgen.run_dftbplus(case, inp)
    res = refgen.run_pydftb(case, inp)
    assert abs(res.energy - ref["energy"]) < 1e-6
