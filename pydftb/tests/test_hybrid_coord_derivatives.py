"""Coordinate derivatives with a hybrid functional (milestone M6h-4d; molecules): DFTB+ refuses
them, so the checks are pydftb's own -- Born charges against the field derivative of the
analytic hybrid forces (Maxwell's relation; no atom moves, so no kinks of the SK tables),
M6g-2's, M6g-3a's and M6g-3b's coordinate derivatives without the exchange, geometry finite
differences where the SK tables are smooth, the two-component collinear limit and the screened
exchange's explicit derivative.
"""
import dataclasses
import os

import numpy as np
import pytest

from pydftb import dftbplus_io as io
from pydftb import hybrid as hy
from pydftb import response as R
from pydftb import response_hybrid as H
from pydftb.constants import BOHR
from pydftb.tests import test_hybrid_response as T
from pydftb.validation import refgen
from pydftb.validation.refgen import VOLT_PER_ANGSTROM_AU

CD = {"coord_derivatives": True}
UNSCREENED = {"hybrid": {"type": "lc", "screening": "threshold", "threshold": 1e-14}}


def _case(name, **over):
    return dataclasses.replace(refgen.CASES_BY_NAME[name], scc_tol=1e-13, **over)


def _run(name, analysis=None, **over):
    case = _case(name, analysis=analysis, **over)
    return refgen.run_pydftb(case, refgen._inputs(case))


def _dfde(name, h=1e-4, **over):
    """dF_K,i / dE_j by four-point finite fields (on top of the case's own field)."""
    case = _case(name, analysis=None, **over)
    inp = refgen._inputs(case)
    base = np.zeros(3)
    if case.efield:
        d = np.asarray(case.efield["direction"], float)
        base = case.efield["strength"] * d / np.linalg.norm(d)            # V/Angstrom
    out = np.zeros((len(case.build()), 3, 3))
    for j in range(3):
        f = []
        for s in (2, 1, -1, -2):
            e = base + s * h / VOLT_PER_ANGSTROM_AU * np.eye(3)[j]
            c = dataclasses.replace(case, efield={"strength": float(np.linalg.norm(e)),
                                                  "direction": (e / np.linalg.norm(e)).tolist()})
            f.append(np.asarray(refgen.run_pydftb(c, inp).forces))
        out[:, :, j] = (8.0 * (f[1] - f[2]) - (f[0] - f[3])) / (12.0 * h)
    return out


def _no_exchange(m, hyb):
    m.setattr(type(hyb), "response_fock", lambda self, ps, antihermitian=False:
              [np.zeros_like(p) for p in ps])
    m.setattr(type(hyb), "exchange_derivative", lambda self, dps, ds, dg:
              [np.zeros(ds.shape[:-2] + p.shape, p.dtype) for p in dps])


# ---- Born charges ------------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
@pytest.mark.parametrize("name, over", [
    ("h2o_lc", {}), ("h2co_cam", {}), ("c2h4_global", {}),          # LC, CAM, global hybrids
    ("ch3_lc_spin", {}), ("cuco_lc_300K", {}),                      # spin, a metal
    ("h2co_lc_plusu", {}), ("h2o_lc_onsite", {}),                   # block potentials
    ("ch3oh_lc_dftb3_pol", {}), ("ch3oh_lc_field_pc", {}),         # DFTB3, field + charges
    ("hco_lc_spin_neighbour", {}), ("h2o_lc_threshold_pol", UNSCREENED),
    ("hco_lc_noncolinear", {}), ("hco_lc_noncolinear_300K", {}),    # two-component spin
    ("hco_lc_noncolinear_field_pc", {})])
def test_born_charges_are_the_field_derivative_of_the_forces(name, over):
    """Z*_K,ij = d mu_j / d x_K,i = d F_K,i / d E_j of pydftb's analytic hybrid forces (checked
    against DFTB+ in M6h-1, M6h-1b): every configuration of the real path and two-component
    spin; the Thresholded screening with an (effectively) zero threshold, whose SCC is then the
    limit the derivatives take."""
    born = _run(name, CD, **over).analysis.coord_derivatives.born
    assert np.abs(born - _dfde(name, **over)).max() <= 1e-9


# ---- the limits ------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["h2o_lc_onsite", "cuco_lc_300K", "hco_lc_plusu_spin",
                                  "ch3oh_lc_field_pc", "hco_lc_noncolinear"])
def test_without_exchange_is_m6g(name, monkeypatch):
    """Without the exchange (its response and its explicit derivative): M6g-2's, M6g-3a's
    (blocks) and M6g-3b's (two-component) coordinate derivatives on the same ground state --
    populations, levels and the metal's Fermi level."""
    calc, st, sc, opts = T._ground(name, monkeypatch, scc_tol=1e-13)
    monkeypatch.undo()
    with pytest.MonkeyPatch.context() as m:
        _no_exchange(m, st.hybrid)
        mine = H.coordinate_derivatives(calc, st, sc, True, opts)
    ref = R.coordinate_derivatives(calc, dataclasses.replace(st, hybrid=None), sc, True, opts)
    assert np.abs(mine.atom - ref.atom).max() <= 1e-13
    assert np.abs(mine.eigenvalue_derivatives - ref.eigenvalue_derivatives).max() <= 1e-13
    assert np.abs(mine.born - ref.born).max() <= 1e-13
    if ref.fermi_derivatives is not None:
        assert np.abs(mine.fermi_derivatives - ref.fermi_derivatives).max() <= 1e-13


def test_two_component_collinear_limit():
    """Moments along z: the collinear hybrid coordinate derivatives (charge, m_z; the transverse
    magnetisations zero)."""
    nc = refgen.CASES_BY_NAME["hco_lc_noncolinear"].noncolinear
    z = {**nc, "initial_spins": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]}
    two = _run("hco_lc_noncolinear", CD, noncolinear=z).analysis.coord_derivatives
    col = _run("hco_lc_spin_neighbour", CD, hybrid="lc").analysis.coord_derivatives
    assert np.abs(two.atom[:, :, 0] - col.atom[:, :, 0]).max() <= 1e-10
    assert np.abs(two.atom[:, :, 3] - col.atom[:, :, 1]).max() <= 1e-10
    assert np.abs(two.atom[:, :, 1:3]).max() <= 1e-12
    assert np.abs(two.born - col.born).max() <= 1e-10


# ---- geometry finite differences -------------------------------------------------------------
@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
@pytest.mark.parametrize("name, atom", [("ch3_lc_spin", 0), ("cuco_lc_300K", 0),
                                        ("hco_lc_noncolinear", 1), ("c6h6_lc_neighbour_10", 0)])
def test_geometry_finite_differences(name, atom):
    """dN/dx, the levels' and the Fermi level's derivatives against four-point differences of
    the SCC (h = 1e-4 Bohr) where the SK tables are smooth (not at H2O's or CH3OH's distances,
    pitfall 171); NeighbourBased with CutoffReduction 10 only so: its forces are not the energy's
    derivative, the coordinate derivatives the screened model's."""
    res = _run(name, {"coord_derivatives": {"atoms": [atom]}})
    cd = res.analysis.coord_derivatives
    e0 = np.concatenate([np.asarray(e, float).ravel() for e in res.eigenvalues])
    single = np.ones(e0.size, bool)                      # degenerate levels split: left out
    close = np.abs(np.diff(e0)) < 1e-6
    single[:-1] &= ~close
    single[1:] &= ~close
    nlev = [np.asarray(e).size for e in res.eigenvalues]
    case = _case(name)
    atoms0 = case.build()
    h = 1e-4
    for c in range(3):
        rs = []
        for s in (2, 1, -1, -2):
            at = atoms0.copy()
            p = at.get_positions()
            p[atom, c] += s * h * BOHR
            at.set_positions(p)
            cc = dataclasses.replace(case, build=lambda a=at: a.copy())
            rs.append(refgen.run_pydftb(cc, refgen._inputs(cc)))

        def fd(x):
            return (8.0 * (x[1] - x[2]) - (x[0] - x[3])) / (12.0 * h)
        assert np.abs(cd.atom[0, c, 0] - fd([np.asarray(r.charges) for r in rs])).max() <= 2e-9
        de = fd([np.concatenate([np.asarray(e, float).ravel() for e in r.eigenvalues])
                 for r in rs])
        ana = np.concatenate([cd.eigenvalue_derivatives[0, c, b, :k] for b, k in enumerate(nlev)])
        assert np.abs(ana - de)[single].max() <= 2e-9
        if cd.fermi_derivatives is not None:
            ef = fd([np.atleast_1d(np.asarray(r.fermi, float)) for r in rs])
            assert np.abs(cd.fermi_derivatives[0, c] - ef).max() <= 1e-9


def test_screened_exchange_derivative(monkeypatch):
    """The explicit derivative of the NeighbourBased exchange at fixed DeltaP (CutoffReduction
    10: S' within the CAM cutoff, on the atom pairs) against differences of the operator at
    displaced geometries."""
    calc, st, sc, opts = T._ground("c6h6_lc_neighbour_10", monkeypatch, scc_tol=1e-10)
    hyb = st.hybrid
    n = hyb.n
    dp = np.random.default_rng(1).normal(size=(n, n))
    dp = dp + dp.T
    pos0 = np.asarray(st.pos, float)
    pb = R._pair_derivatives(calc, st)
    ai = pb.i[pb.pair_of]
    offsets = np.asarray(st.basis.offsets)
    orb_atom = np.asarray(st.basis.orb_atom)
    atom, h = 0, 1e-4
    lo, hi = offsets[atom], offsets[atom] + int(np.count_nonzero(orb_atom == atom))
    rows = np.flatnonzero(ai == atom)
    dsm = np.zeros((3, n, n))
    for c in range(3):
        d_s = np.zeros((hi - lo, n))
        np.add.at(d_s, (pb.rows[rows] - lo, pb.cols[rows]), -pb.ds[rows, c])
        dsm[c, lo:hi, :] += d_s
        dsm[c, :, lo:hi] += d_s.T
    ana = hyb.exchange_derivative([dp], dsm, hyb.gamma_derivative([atom]))[0]
    for c in range(3):
        vals = []
        for s in (2, 1, -1, -2):
            p = pos0.copy()
            p[atom, c] += s * h
            hb = calc._setup(st.species, p, None, np.zeros(3, bool), False, {}).hybrid
            vals.append(-0.125 * hy.four_terms(hb.S_cam, hb.G, dp))
        fd = (8.0 * (vals[1] - vals[2]) - (vals[0] - vals[3])) / (12.0 * h)
        assert np.abs(ana[c] - fd).max() <= 1e-9 * np.abs(fd).max()


# ---- bookkeeping -------------------------------------------------------------------------------
def test_atom_subsets_batches_and_writers(monkeypatch, tmp_path):
    """A subset of the atoms gives the full calculation's rows and one atom per batch the same
    derivatives (the right-hand sides are independent; to the BLAS's rounding); DFTB+'s
    coordinate-derivative block of detailed.out and dBand_dx.out are written."""
    calc, st, sc, opts = T._ground("ch3_lc_spin", monkeypatch, scc_tol=1e-12)
    monkeypatch.undo()
    full = H.coordinate_derivatives(calc, st, sc, True, opts)
    part = H.coordinate_derivatives(calc, st, sc, {"atoms": [2, 0]}, opts)
    assert np.array_equal(part.atoms, [0, 2])
    assert np.abs(part.atom - full.atom[[0, 2]]).max() <= 1e-13
    with pytest.MonkeyPatch.context() as m:
        m.setattr(H, "RHS_ELEMENTS", 1)
        batched = H.coordinate_derivatives(calc, st, sc, True, opts)
    assert np.abs(batched.atom - full.atom).max() <= 1e-13
    assert np.abs(batched.eigenvalue_derivatives - full.eigenvalue_derivatives).max() <= 1e-13
    res = _run("hco_lc_noncolinear", CD)
    assert "Born effective charges" in io.coord_derivative_tables(res)
    io.write_band_derivatives(os.path.join(tmp_path, "dBand_dx.out"), res, "coordinate")
    assert os.path.getsize(os.path.join(tmp_path, "dBand_dx.out")) > 0
