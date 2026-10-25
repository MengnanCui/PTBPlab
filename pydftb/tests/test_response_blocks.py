"""Responses with block potentials and the external charge derivatives (milestone M6g-3a).

The DFTB+ references are the M6g-3a cases of :mod:`pydftb.validation.refgen` (the ``*_pol``,
``*_kernel`` and ``*_dx`` cases with ``plus_u``/``onsite``, the ``*_extq`` cases).  Here: the
polarisability with block potentials against differences of the dipole in a field (also +U
and onsite corrections together, which DFTB+ refuses), the coordinate derivatives against
differences of the dipole with the sum rules, the derivatives with respect to the point
charges' positions against differences of the populations (molecule, blur, Ewald sum), the
symmetry of chi0 with block channels, the unchanged path without block potentials, the text
block and the refused combinations.
"""
import numpy as np
import pytest

from pydftb import DFTB, ParamSet
from pydftb import dftbplus_io as io
from pydftb import response as resp
from pydftb.constants import BOHR
from pydftb.validation import cases as C

from .test_vibrations import DOUBLET, EXACT, _differences, _water

SKF = C.SKF_SETS["ptbp_hcocu"]
U_CU = {"Cu": [("d", 0.15)]}
U_O = {"O": [("p", 0.1)]}
METAL = {"smearing": ("fermi", 3000.0)}
PC = {"positions": np.array([[3.0, 0.2, 0.1], [-2.5, 1.0, 0.5], [0.3, -2.8, 1.2]]) / BOHR,
      "charges": [0.8, -0.6, 0.4]}


def _ons(atoms):
    return {k: v for k, v in C.ONS.items() if k in set(atoms.get_chemical_symbols())}


def _calc(atoms, analysis=None, **kw):
    labels = atoms.get_chemical_symbols()
    p = ParamSet(SKF, max_l={e: C.MAXL[e] for e in set(labels)})
    return DFTB(p.skset(labels), scc_tol=1e-13, max_scc=500, analysis=analysis, **kw), labels


def _run(atoms, analysis=None, cell=None, **kw):
    calc, labels = _calc(atoms, analysis, **kw)
    if cell is None:
        return calc.calculate(labels, atoms.positions / BOHR)
    return calc.calculate(labels, atoms.positions / BOHR, cell, np.ones(3, bool))


def _cases():
    cu4co = C._cu4co()
    return [
        (_water(), {"onsite_corrections": _ons(_water())}, None),
        (cu4co, {"plus_u": U_CU, **METAL}, EXACT),
        (cu4co, {"plus_u": U_CU, "plus_u_functional": "psic", **METAL}, EXACT),
        (cu4co, {"plus_u": U_CU, "onsite_corrections": _ons(cu4co), **METAL}, EXACT),
        (_water(), {**DOUBLET, "plus_u": U_O}, None),
    ]


IDS = ["water_onsite", "cu4co_fll", "cu4co_psic", "cu4co_fll_onsite", "water_doublet_u"]


@pytest.mark.parametrize("atoms, kw, pert", _cases(), ids=IDS)
def test_polarisability_against_field_differences(atoms, kw, pert):
    """alpha_ji = d mu_j / d E_i against fourth-order differences of the dipole in a field:
    the block populations respond (+U and onsite together too: DFTB+ refuses that).  The metal's
    hyperpolarisabilities make the differences' error 3e-7 at h = 1e-3, 1e-9 at 2.5e-4 (h^4)."""
    pol = _run(atoms, {"polarisability": True, "perturbation": pert}, **kw).analysis.polarisability
    alpha = pol.tensor[0]
    h = 2.5e-4
    for i in range(3):
        mu = []
        for s in (2, 1, -1, -2):
            e = np.zeros(3)
            e[i] = s * h
            mu.append(_run(atoms, electric_field=e, **kw).dipole)
        fd = (8 * (mu[1] - mu[2]) - (mu[0] - mu[3])) / (12 * h)
        assert np.abs(fd - alpha[:, i]).max() < 2e-8 * np.abs(alpha).max()
    assert np.abs(alpha - alpha.T).max() < 1e-9 * np.abs(alpha).max()


@pytest.mark.parametrize("atoms, kw, pert", [_cases()[i] for i in (0, 3, 4)],
                         ids=[IDS[i] for i in (0, 3, 4)])
def test_coordinate_derivatives_against_dipole_differences(atoms, kw, pert):
    """Born charges with block potentials (DFTB+ refuses them) against fourth-order differences
    of the dipole; electrons conserved, translation invariance, sum_K Z_K = Q I."""
    cd = _run(atoms, {"coord_derivatives": True, "perturbation": pert},
              **kw).analysis.coord_derivatives
    metal = "smearing" in kw
    fd = _differences(atoms, lambda r: r.dipole, h=4e-4 if metal else 2e-4, **kw)
    assert np.abs(cd.born - fd).max() < 2e-8 * max(1.0, np.abs(fd).max())
    dn = cd.atom[:, :, 0, :]
    assert np.abs(dn.sum(-1)).max() < 1e-10 and np.abs(dn.sum(0)).max() < 1e-10
    np.testing.assert_allclose(cd.born.sum(0), kw.get("charge", 0.0) * np.eye(3), atol=1e-9)


def _pc_differences(atoms, pc, cell=None, h=1e-3, **kw):
    """Fourth-order differences of the electron populations moving each point charge:
    (n_charge, 3, n_atom)."""
    out = np.zeros((len(pc["charges"]), 3, len(atoms)))
    for j in range(len(pc["charges"])):
        for c in range(3):
            vals = []
            for s in (2, 1, -1, -2):
                pos = np.array(pc["positions"], float)
                pos[j, c] += s * h
                vals.append(_run(atoms, cell=cell, point_charges={**pc, "positions": pos},
                                 **kw).charges)
            out[j, c] = (8 * (vals[1] - vals[2]) - (vals[0] - vals[3])) / (12 * h)
    return out


@pytest.mark.parametrize("system", ["water", "water_blur_onsite", "water_box"])
def test_external_charge_derivatives_against_differences(system):
    """dN_B/dR_j against differences of the populations moving the charges: a molecule, blurred
    charges with onsite corrections, a periodic box (the Ewald sum: the development DFTB+ gets
    its sign wrong); the electron count does not change; the orbitals sum to the atoms."""
    atoms, cell, kw = _water(), None, {}
    pc = dict(PC)
    if system == "water_blur_onsite":
        pc = {**PC, "blur": 0.8 / BOHR}
        kw = {"onsite_corrections": _ons(atoms)}
    elif system == "water_box":
        atoms = C._water_box()
        cell = atoms.cell.array / BOHR
        pc = {**PC, "positions": PC["positions"] + 2.0 / BOHR, "blur": 0.3 / BOHR}
    ex = _run(atoms, {"external_charge_derivatives": True}, cell=cell, point_charges=pc,
              **kw).analysis.external_charge_derivatives
    np.testing.assert_array_equal(ex.charges, [0, 1, 2])
    fd = _pc_differences(atoms, pc, cell, **kw)
    dn = ex.atom[:, :, 0, :]
    assert np.abs(dn - fd).max() < 1e-8 * max(1.0, np.abs(fd).max())
    assert np.abs(dn.sum(-1)).max() < 1e-10
    summed = np.zeros_like(dn)
    np.add.at(summed, (slice(None), slice(None), ex.orbital_atom), ex.orbital[:, :, 0, :])
    assert np.abs(summed - dn).max() < 1e-12


def test_external_charge_selection_and_spin():
    """A selection gives the full result's rows (DFTB+'s WrtCharges); with collinear spin the
    magnetisation responds too, the electron count still not."""
    atoms = _water()
    full = _run(atoms, {"external_charge_derivatives": True}, point_charges=PC,
                **DOUBLET).analysis.external_charge_derivatives
    part = _run(atoms, {"external_charge_derivatives": {"charges": [2, 0]}}, point_charges=PC,
                **DOUBLET).analysis.external_charge_derivatives
    np.testing.assert_array_equal(part.charges, [0, 2])
    assert np.abs(part.atom - full.atom[[0, 2]]).max() < 1e-12
    assert full.atom.shape[2] == 2 and np.abs(full.atom[:, :, 1, :]).max() > 1e-4
    assert np.abs(full.atom[:, :, 0, :].sum(-1)).max() < 1e-10
    with pytest.raises(ValueError, match="out of range"):
        _run(atoms, {"external_charge_derivatives": {"charges": [3]}}, point_charges=PC)
    with pytest.raises(ValueError, match="unknown keys"):
        _calc(atoms, {"external_charge_derivatives": {"charge": [0]}}, point_charges=PC)


@pytest.mark.parametrize("atoms, kw", [
    (C._cu4co(), {"plus_u": U_CU, "onsite_corrections": _ons(C._cu4co()), **METAL}),
    (_water(), {**DOUBLET, "plus_u": U_O}),
], ids=["cu4co_fll_onsite", "water_doublet_u"])
def test_block_chi0_symmetric(atoms, kw):
    """Static chi0 on [shells | block channels] per spin channel (orbital rows summed to
    shells, the Fermi-level move included) is symmetric; its orbital part is M6g-1's chi0, bit
    for bit."""
    calc, labels = _calc(atoms, {"polarisability": True}, **kw)
    st = calc._setup(labels, atoms.positions / BOHR, None, None, False, {})
    sc = calc._scc(st, None, False, {})
    gs = resp.GroundState(calc, st, sc)
    ch = resp.block_channels(st)
    x = resp.chi0(gs, [0.0], channels=ch)[0]
    assert np.array_equal(x.orb, resp.chi0(gs, [0.0])[0].orb)
    no, nsh, nb = gs.norb, gs.nshell, ch.size
    for s in range(gs.nspin):
        part = x.full[s * (no + nb):(s + 1) * (no + nb), s * (nsh + nb):(s + 1) * (nsh + nb)]
        sq = np.vstack([gs.shells(part[:no]), part[no:]])
        assert np.abs(sq - sq.T).max() < 1e-12 * np.abs(sq).max()


def test_text_block():
    """The development DFTB+'s detailed.out block: a header per charge, a row per atom in
    e/Angstrom with its column layout."""
    r = _run(_water(), {"external_charge_derivatives": {"charges": [1]}}, point_charges=PC)
    lines = io.external_charge_tables(r).splitlines()
    assert lines[0] == "Derivatives of atomic charges w.r.t. coordinates of external charges"
    assert lines[1] == "Chg#2    x               y               z"
    assert len(lines) == 6 and lines[-1] == "" and lines[2].startswith("At1 ")
    vals = np.array([[float(v) for v in ln[3:].split()] for ln in lines[2:5]])
    ex = r.analysis.external_charge_derivatives
    assert np.abs(vals - ex.atom[0, :, 0, :].T / BOHR).max() < 1e-8
    with pytest.raises(ValueError, match="external_charge_derivatives"):
        io.external_charge_tables(_run(_water(), point_charges=PC))


def test_refused():
    with pytest.raises(ValueError, match="point charges for the external charge derivatives"):
        _calc(_water(), {"external_charge_derivatives": True})
