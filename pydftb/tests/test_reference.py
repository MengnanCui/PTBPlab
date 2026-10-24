"""pydftb vs frozen DFTB+ references (pydftb/tests/data/ref/*.json).

The JSON files are produced by ``python -m pydftb.validation.refgen
--regenerate`` (needs dftb+); this test only needs pydftb.
"""
import glob
import importlib.util
import os

import numpy as np
import pytest

from pydftb.validation import refgen

REFS = sorted(glob.glob(os.path.join(refgen.REF_DIR, "*.json")))
# milestone M4: the same references with the torch backend (skipped without torch)
BACKENDS = ["numpy", pytest.param("torch", marks=pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"))]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("path", REFS, ids=[os.path.basename(p)[:-5] for p in REFS])
def test_against_dftbplus_reference(path, backend):
    doc = refgen.load(os.path.basename(path)[:-5])
    why = refgen.missing_requirement(doc)      # D3/D4 cases need the matching library
    if why:
        pytest.skip(why)
    if backend == "torch" and not refgen.CASES_BY_NAME[doc["case"]].torch:
        pytest.skip("NumPy-only physics (two-component M6a, block potentials M6b, "
                    "environment M6c, excitations M6d, SCC start and mixers M6f)")
    r = refgen.compare(doc, backend=backend)
    assert r["ok"], f"{doc['case']}: deviations {r['dev']} exceed {doc['tolerances']}"
    # same mixer (DFTB+ modified Broyden) and Fermi search -> same SCC path
    if r["n_scc_ref"] is not None:
        assert abs(r["n_scc"] - r["n_scc_ref"]) <= 1, (r["n_scc"], r["n_scc_ref"])


@pytest.mark.parametrize("name", ["h2o", "h2o_nonscc", "water_dimer_polyrep", "li_bcc_k666_node",
                                  "ch3oh_loose_scc", "water_dimer_dftb3_loose_scc",
                                  "cuo_chain_spin_loose_scc", "gdo2_dftb3_shell", "gdo_spin",
                                  "gdo_rocksalt_k444", "cu_slab_types_k441",
                                  "h2o_selected_s2", "h2o_selected_both"])
def test_dftbplus_finite_difference_emulation(name):
    """With DFTB+'s finite-difference step pydftb reproduces DFTB+ forces and stress exactly:
    the residual (~1e-6 Ha/Bohr for water, 1.2e-8 Ha/Bohr^3 for the Li stress) is DFTB+'s
    numerical derivative, not a pydftb error.  The *_loose_scc cases (SCC tolerance 1e-5)
    hold only because the forces use the output charges, as DFTB+ does."""
    doc = refgen.load(name)
    exact = refgen.compare(doc)["dev"]
    fd = refgen.compare(doc, fd_delta=float(np.finfo(float).eps) ** 0.25)["dev"]
    for key in ("forces", "stress"):
        assert fd[key] < 1e-9
        assert fd[key] <= max(exact[key], 1e-12)


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
