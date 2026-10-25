"""Responses of two-component calculations (milestone M6g-3b).

The DFTB+ references are the M6g-3b cases of :mod:`pydftb.validation.refgen` (the ``*_pol``,
``*_kernel``, ``*_dx`` and ``*_extq`` cases with ``noncolinear``/``spin_orbit``).  Here: the
polarisability and the four Pauli components' charge derivatives against differences in a
field, the coordinate derivatives against differences of the geometry with the sum rules, the
point charges' positions, for non-collinear spin, both spin-orbit models and dual spin-orbit
coupling with +U and onsite corrections (where DFTB+ 25.1 drops the skew block response); the
symmetry of chi0, the rotation zero modes without spin-orbit coupling (longitudinal response,
rotated ground states, a loose SCC tolerance), collinear moments against collinear spin,
Kramers pairs, the compat switch, the text outputs and the refusals.
"""
import dataclasses
import os
import tempfile
import warnings

import numpy as np
import pytest

from pydftb import DFTB, ParamSet
from pydftb import dftbplus_io as io
from pydftb import response_pauli as rp
from pydftb.constants import BOHR
from pydftb.kpoints import monkhorst_pack
from pydftb.validation import cases as C
from pydftb.validation import refgen

from .test_vibrations import EXACT, _differences

SKF = C.SKF_SETS["ptbp_hcocu"]
W = {e: C.W_SPIN[e] for e in ("H", "O")}
NC = {"spin": "noncolinear", "spin_constants": W, "initial_spins": [C.TILT, [0.0, 0.0, 0.0]],
      "smearing": ("fermi", 300.0)}
SOC = {"spin_orbit": {"O": [0.0, 0.01]}}
PC = {"positions": np.array([[3.0, 0.2, 0.1], [-2.5, 1.0, 0.5], [0.3, -2.8, 1.2]]) / BOHR,
      "charges": [0.8, -0.6, 0.4]}
MODELS = {
    "noncolinear": NC,
    "dual": {**NC, **SOC},
    "onsite_soc": {**NC, **SOC, "spin_orbit_dual": False},
    "dual_u": {**NC, **SOC, "plus_u": {"O": [("p", 0.1)]}},
    "dual_onsite_corr": {**NC, **SOC,
                         "onsite_corrections": {k: C.ONS[k] for k in ("H", "O")}},
}


def _calc(atoms, analysis=None, **kw):
    labels = atoms.get_chemical_symbols()
    p = ParamSet(SKF, max_l={e: C.MAXL[e] for e in set(labels)})
    return DFTB(p.skset(labels), scc_tol=kw.pop("scc_tol", 1e-13), max_scc=500,
                analysis=analysis, **kw), labels


def _run(atoms, analysis=None, **kw):
    calc, labels = _calc(atoms, analysis, **kw)
    return calc.calculate(labels, atoms.positions / BOHR)


def _components(r):
    """(4, n_atom): electron populations and the magnetisation vectors of a result."""
    return np.vstack([r.charges, np.asarray(r.magmoms).T])


def _field_differences(atoms, h=5e-4, **kw):
    """Fourth-order differences in a field: d mu / d E (j, i) and d(q, m) / d E_i (i, 4, A)."""
    dmu, dq = np.zeros((3, 3)), np.zeros((3, 4, len(atoms)))
    for i in range(3):
        mu, q = [], []
        for s in (2, 1, -1, -2):
            e = np.zeros(3)
            e[i] = s * h
            r = _run(atoms, electric_field=e, **kw)
            mu.append(r.dipole)
            q.append(_components(r))
        dmu[:, i] = (8 * (mu[1] - mu[2]) - (mu[0] - mu[3])) / (12 * h)
        dq[i] = (8 * (q[1] - q[2]) - (q[0] - q[3])) / (12 * h)
    return dmu, dq


@pytest.mark.parametrize("model", list(MODELS))
def test_polarisability_against_field_differences(model):
    """alpha and the four components' dN/dE against differences in a field; with dual
    spin-orbit coupling and +U or onsite corrections the skew block response is in (DFTB+ 25.1
    leaves it out: 1e-4 off).  The magnetisation of a spin-orbit calculation converges more
    softly (differences: ~1e-11 noise / h)."""
    atoms = C._oh()
    pol = _run(atoms, {"polarisability": True}, **MODELS[model]).analysis.polarisability
    dmu, dq = _field_differences(atoms, **MODELS[model])
    alpha = pol.tensor[0]
    assert np.abs(alpha - dmu).max() < 2e-8 * np.abs(alpha).max()
    assert np.abs(alpha - alpha.T).max() < 1e-10 * np.abs(alpha).max()
    mine = pol.charge_derivatives[0]                     # (3, 4, n_atom)
    assert np.abs(mine[:, 0] - dq[:, 0]).max() < 2e-8 * np.abs(dq[:, 0]).max()
    assert np.abs(mine[:, 1:] - dq[:, 1:]).max() < 1e-7 * max(1.0, np.abs(dq[:, 1:]).max())
    assert np.abs(mine[:, 0].sum(-1)).max() < 1e-10    # the electron count


@pytest.mark.parametrize("model", list(MODELS))
def test_coordinate_derivatives_against_differences(model):
    """Born charges and the four components' dN/dx (DFTB+ refuses two-component coordinate
    derivatives) against differences of the geometry; with dual spin-orbit coupling H_SO moves
    with S.  Electrons conserved, translation invariance, sum_K Z_K = 0."""
    atoms = C._oh()
    kw = MODELS[model]
    cd = _run(atoms, {"coord_derivatives": True}, **kw).analysis.coord_derivatives
    born = _differences(atoms, lambda r: r.dipole, h=2e-4, **kw)
    assert np.abs(cd.born - born).max() < 2e-8 * max(1.0, np.abs(born).max())
    dq = _differences(atoms, _components, h=2e-4, **kw)        # (K, 3, 4, A)
    assert np.abs(cd.atom[:, :, 0] - dq[:, :, 0]).max() < 2e-8 * np.abs(dq[:, :, 0]).max()
    assert np.abs(cd.atom[:, :, 1:] - dq[:, :, 1:]).max() < 1e-7
    assert np.abs(cd.atom[:, :, 0].sum(-1)).max() < 1e-10
    assert np.abs(cd.atom.sum(0)).max() < 1e-9                # every component
    assert np.abs(cd.born.sum(0)).max() < 1e-9


def test_external_charge_derivatives_against_differences():
    """dN/dR of the point charges with dual spin-orbit coupling and +U: all four components
    (the development DFTB+ prints the charge part) against differences moving the charges."""
    atoms, kw = C._oh(), MODELS["dual_u"]
    ex = _run(atoms, {"external_charge_derivatives": True}, point_charges=PC,
              **kw).analysis.external_charge_derivatives
    h = 1e-3
    fd = np.zeros((3, 3, 4, len(atoms)))
    for j in range(3):
        for c in range(3):
            vals = []
            for s in (2, 1, -1, -2):
                pos = PC["positions"].copy()
                pos[j, c] += s * h
                vals.append(_components(_run(atoms, point_charges={**PC, "positions": pos},
                                             **kw)))
            fd[j, c] = (8 * (vals[1] - vals[2]) - (vals[0] - vals[3])) / (12 * h)
    assert np.abs(ex.atom[:, :, 0] - fd[:, :, 0]).max() < 1e-8 * np.abs(fd[:, :, 0]).max()
    assert np.abs(ex.atom[:, :, 1:] - fd[:, :, 1:]).max() < 1e-7
    assert np.abs(ex.atom[:, :, 0].sum(-1)).max() < 1e-10


def test_periodic_kernel_against_site_potential_differences():
    """A periodic metal with dual spin-orbit coupling and +U (k-points, complex spinors, real
    and skew block channels): the response kernel dN_B/dV_A against differences in a gross site
    potential on A.  Without the skew channels (DFTB+ 25.1) the kernel is 40 % off here."""
    chain = C._cuo_chain()
    labels = chain.get_chemical_symbols()
    p = ParamSet(SKF, max_l={e: C.MAXL[e] for e in set(labels)})

    def run(analysis=None, **extra):
        calc = DFTB(p.skset(labels), scc_tol=1e-12, max_scc=500, analysis=analysis,
                    kpts=monkhorst_pack((2, 1, 1)), spin_orbit=C.XI_CUO,
                    plus_u={"Cu": [("d", 0.15)]}, smearing=("fermi", 1000.0), **extra)
        return calc.calculate(labels, chain.positions / BOHR, chain.cell.array / BOHR,
                              np.ones(3, bool))

    ker = run({"response_kernel": True, "perturbation": EXACT}).analysis.response_kernel
    h = 1e-4
    for a in (0, 1):
        vals = []
        for s in (2, 1, -1, -2):
            r = run(site_potentials={"gross": {a: s * h}})
            vals.append(np.vstack([r.charges, np.zeros((3, len(labels)))
                                   if r.magmoms is None else np.asarray(r.magmoms).T]))
        fd = (8 * (vals[1] - vals[2]) - (vals[0] - vals[3])) / (12 * h)
        assert np.abs(ker.atom[0, a] - fd).max() < 2e-9 * np.abs(fd).max()
    DFTB._response_skew_compat = True
    try:
        compat = run({"response_kernel": True, "perturbation": EXACT}).analysis.response_kernel
    finally:
        DFTB._response_skew_compat = False
    assert np.abs(compat.atom - ker.atom).max() > 0.1 * np.abs(ker.atom).max()


def _ground_state(atoms, **kw):
    calc, labels = _calc(atoms, {"polarisability": True}, **kw)
    st = calc._setup(labels, atoms.positions / BOHR, None, None, False, {})
    sc = calc._scc(st, None, False, {})
    return calc, st, sc


@pytest.mark.parametrize("atoms, kw", [
    (C._oh(), MODELS["dual_u"]),
    (C._cu4co(), {"spin_orbit": C.XI_CU4CO, "smearing": ("fermi", 300.0)}),
], ids=["oh_dual_u", "cu4co_soc"])
def test_chi0_symmetric(atoms, kw):
    """Static chi0 of the four Pauli components on [shells | real | skew channels] (orbital
    rows summed to shells, the Fermi-level move included) is symmetric: real and skew block
    channels (dual spin-orbit coupling with +U), Kramers pairs of a metal."""
    calc, st, sc = _ground_state(atoms, **kw)
    gs = rp.PauliGroundState(calc, st, sc)
    real, skew = rp.block_channels(calc, st)
    x = rp.chi0(gs, [0.0], real=real, skew=skew)[0]
    lay = x.layout
    rows = []
    for b in range(4):
        part = x.full[b * lay.nr:(b + 1) * lay.nr]
        rows.append(np.vstack([gs.shells(part[:lay.norb]), part[lay.norb:]]))
    sq = np.vstack(rows)
    assert sq.shape == (4 * lay.nc, 4 * lay.nc)
    assert np.abs(sq - sq.T).max() < 1e-12 * np.abs(sq).max()
    if real is not None:
        assert lay.nb > 0 and lay.ns > 0


def test_rotation_zero_modes():
    """Without spin-orbit coupling a global rotation of the moments costs no energy: the
    magnetisation response is longitudinal (DFTB+'s answer), rotating the initial moments
    rotates it and leaves the charge response unchanged, and a loose SCC tolerance does not let
    a rotation in (the bordered solve)."""
    atoms = C._oh()

    def response(spins, scc_tol=1e-13):
        r = _run(atoms, {"polarisability": True}, scc_tol=scc_tol,
                 **{**NC, "initial_spins": spins})
        return r.analysis.polarisability.charge_derivatives[0], np.asarray(r.magmoms)

    q, m = response([C.TILT, [0.0, 0.0, 0.0]])
    for i in range(3):
        for a in range(len(atoms)):
            cross = np.cross(q[i, 1:, a], m[a])
            assert np.linalg.norm(cross) < 1e-10 * max(1e-3, np.linalg.norm(m[a]))
    rot = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    q2, m2 = response([rot @ np.array(C.TILT), [0.0, 0.0, 0.0]])
    assert np.abs(m2 - m @ rot.T).max() < 1e-9
    assert np.abs(q2[:, 0] - q[:, 0]).max() < 1e-10
    assert np.abs(q2[:, 1:] - np.einsum("ab,ibA->iaA", rot, q[:, 1:])).max() < 1e-9
    q3, m3 = response([C.TILT, [0.0, 0.0, 0.0]], scc_tol=1e-7)
    for i in range(3):
        for a in range(len(atoms)):
            assert np.linalg.norm(np.cross(q3[i, 1:, a], m3[a])) < 1e-7
    assert np.abs(q3 - q).max() < 1e-6


def test_collinear_moments_match_collinear_spin():
    """Moments along z without spin-orbit coupling: the two-component response is the collinear
    one with a shared Fermi level (charge and m_z; m_x, m_y do not respond)."""
    atoms = C._water()
    base = {"charge": 1.0, "spin_constants": W, "smearing": ("fermi", 300.0)}
    col = _run(atoms, {"polarisability": True, "response_kernel": True}, spin=True,
               unpaired_electrons=1.0, relax_total_spin=True, **base).analysis
    two = _run(atoms, {"polarisability": True, "response_kernel": True}, spin="noncolinear",
               initial_spins=[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
               **base).analysis
    a, b = col.polarisability, two.polarisability
    assert np.abs(a.tensor - b.tensor).max() < 1e-9
    assert np.abs(a.charge_derivatives[0][:, 0] - b.charge_derivatives[0][:, 0]).max() < 1e-9
    assert np.abs(a.charge_derivatives[0][:, 1] - b.charge_derivatives[0][:, 3]).max() < 1e-9
    assert np.abs(b.charge_derivatives[0][:, 1:3]).max() < 1e-12
    ka, kb = col.response_kernel, two.response_kernel
    assert np.abs(ka.atom[0][:, 0] - kb.atom[0][:, 0]).max() < 1e-9
    assert np.abs(ka.atom[0][:, 1] - kb.atom[0][:, 3]).max() < 1e-9
    assert np.abs(ka.net - kb.net).max() < 1e-9


def test_kramers_pairs_and_level_derivatives():
    """Spin-orbit coupling without moments: every level is a Kramers pair; a field keeps the
    pairs degenerate (equal derivatives), the magnetisation does not respond, and the level
    derivatives agree with differences of the levels in a field (pairs at least 0.01 Ha from
    their neighbours: Cu4CO's d pairs 2.5e-4 - 9e-4 Ha apart couple so strongly that the
    differences' h^4 error reaches 1e-6)."""
    atoms = C._cu4co()
    kw = {"spin_orbit": C.XI_CU4CO, "smearing": ("fermi", 300.0)}
    r = _run(atoms, {"polarisability": True, "perturbation": EXACT}, **kw)
    pol = r.analysis.polarisability
    de = pol.eigenvalue_derivatives[0][:, 0]                      # (3, 2N)
    assert np.abs(de[:, 0::2] - de[:, 1::2]).max() < 1e-10
    assert np.abs(pol.charge_derivatives[0][:, 1:]).max() < 1e-8
    pairs = np.asarray(r.eigenvalues[0])[0::2]
    gap = np.minimum(np.r_[np.inf, np.diff(pairs)], np.r_[np.diff(pairs), np.inf])
    apart = np.repeat(gap > 1e-2, 2)
    assert apart.sum() >= 8
    h = 2.5e-4
    for i in range(3):
        lev = []
        for s in (2, 1, -1, -2):
            e = np.zeros(3)
            e[i] = s * h
            lev.append(np.asarray(_run(atoms, electric_field=e, **kw).eigenvalues[0]))
        fd = (8 * (lev[1] - lev[2]) - (lev[0] - lev[3])) / (12 * h)
        assert np.abs(fd - de[i])[apart].max() < 1e-8


def test_skew_compat_switch():
    """DFTB._response_skew_compat leaves the skew block channels out (DFTB+ 25.1's response):
    it changes dual spin-orbit coupling with +U (by ~1e-4) and nothing without the dual model."""
    atoms = C._oh()

    def alpha(kw):
        return _run(atoms, {"polarisability": True}, **kw).analysis.polarisability.tensor[0]

    for model, changed in (("dual_u", True), ("noncolinear", False)):
        kw = {**MODELS[model], "plus_u": {"O": [("p", 0.1)]}}
        a = alpha(kw)
        DFTB._response_skew_compat = True
        try:
            b = alpha(kw)
        finally:
            DFTB._response_skew_compat = False
        if changed:
            assert 1e-6 < np.abs(a - b).max() / np.abs(a).max() < 1e-3
        else:
            assert np.array_equal(a, b)


def test_text_outputs():
    """detailed.out: q/x/y/z rows of dq/dE, one density of states, dE_F/dE as it is (DFTB+
    divides by the full filling, 1); dV_band.out: SPIN 1 and zero blocks for SPIN 2-4;
    dE_band.out: SPIN 1 only."""
    r = _run(C._oh(), {"polarisability": True, "response_kernel": True}, **MODELS["dual"])
    pol = r.analysis.polarisability
    text = io.response_tables(r, part="main", dftbplus_fermi=True).splitlines()
    i = text.index("Atomic charge derivatives (a.u.), d q / d E_x:")
    assert text[i + 1].startswith("   1 q") and [ln[:6] for ln in text[i + 2:i + 5]] == \
        ["     x", "     y", "     z"]
    vals = [float(ln[6:]) for ln in text[i + 1:i + 5]]
    assert np.abs(np.array(vals) - pol.charge_derivatives[-1][0, :, 0]).max() < 1e-12
    assert sum(ln.startswith("Density of states at the Fermi energy") for ln in text) == 1
    j = text.index("Derivative of Fermi energy with respect to electric field")
    assert abs(float(text[j + 1].split(":")[1]) - pol.fermi_derivatives[-1, 0, 0]) < 1e-12
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dV_band.out")
        io.write_band_derivatives(path, r, "potential")
        with open(path) as fh:
            text = fh.read()
        heads = [ln for ln in text.splitlines() if ln.startswith("ATOM")]
        assert [ln.split()[5] for ln in heads] == ["1", "2", "3", "4"] * 2
        blocks = text.split("\n\n")
        assert all(float(x.split()[1]) == 0.0 for b in blocks[1:4] for x in b.splitlines()[1:])
        path = os.path.join(tmp, "dE_band.out")
        io.write_band_derivatives(path, r, "field")
        with open(path) as fh:
            heads = [ln for ln in fh if "DIR" in ln]
        assert len(heads) == 3 and all(ln.split()[5] == "1" for ln in heads)


def test_soc_zero_mode_warning_and_refusals():
    """GdO with tilted moments and weak spin-orbit coupling keeps an exact zero mode (its
    electrons rotate about the axis) next to a soft spin mode: the SVD solve leaves the zero
    mode out with a warning and matches DFTB+ (the gdo_noncolinear_soc_pol case).  Periodic
    coordinate derivatives are refused (molecules only, as in M6g-2)."""
    case = dataclasses.replace(C.CASES_BY_NAME["gdo_noncolinear_soc_pol"], torch=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = refgen.run_pydftb(case, refgen._inputs(case))
    msgs = [str(w.message) for w in caught if "near-zero mode" in str(w.message)]
    assert len(msgs) == 1 and msgs[0].startswith("two-component response: 1 near-zero mode")
    assert np.all(np.isfinite(res.analysis.polarisability.tensor))
    chain = C._cuo_chain()
    calc, labels = _calc(chain, {"coord_derivatives": True}, spin_orbit=C.XI_CUO,
                         smearing=("fermi", 500.0))
    with pytest.raises(ValueError, match="molecules only"):
        calc.calculate(labels, chain.positions / BOHR, chain.cell.array / BOHR,
                       np.ones(3, bool))
