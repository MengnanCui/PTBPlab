"""Coordinate derivatives, the finite-difference Hessian and normal modes (milestone M6g-2,
DFTB+ ``CoordDerivatives``, ``Driver = SecondDerivatives`` and ``modes``).

The DFTB+ references are the ``*_dx*`` and ``*_hess*`` cases of :mod:`pydftb.validation.refgen`.
Here: the analytic Born charges and charge derivatives against differences of pydftb's dipole
and populations (also where DFTB+ refuses: spin, metals, degenerate levels; or errs: fields,
point charges), the sum rules, the eigenvalue and Fermi-level derivatives, rotations inside
degenerate runs, the Hessian's symmetry and partial Hessians, normal modes and projections,
the text outputs and the refused combinations.
"""
import numpy as np
import pytest
from ase.build import molecule

from pydftb import DFTB, ParamSet
from pydftb import dftbplus_io as io
from pydftb import response as resp
from pydftb.constants import AMU_AU, BOHR, HARTREE_CM, VOLT_PER_ANGSTROM_AU
from pydftb.validation import cases as C
from pydftb.validation.dftbplus import parse_vibrations_tag
from pydftb.vibrations import masses_of, normal_modes, second_derivatives

SKF = C.SKF_SETS["ptbp_hcocu"]
DFTB3 = {"third_order": "full", "hubbard_derivs": C.UD_3OB, "h_damping": 4.0}
DOUBLET = {"charge": 1.0, "spin": True, "unpaired_electrons": 1.0,   # polar, unlike O2
           "spin_constants": {e: C.W_SPIN[e] for e in ("H", "O")}}
FIELD = 0.3 * VOLT_PER_ANGSTROM_AU * np.array([0.2, 1.0, -0.3]) / np.linalg.norm([0.2, 1.0, -0.3])
PC = {"positions": np.array([[3.0, 0.2, 0.1], [-2.5, 1.0, 0.5]]) / BOHR, "charges": [0.4, -0.3]}
EXACT = {"filling_tol": 1e-15}  # every partly filled level (DFTB+'s window: ~1e-7 of a metal's)


def _co2():
    return molecule("CO2")


# The reference water has its H-H distance 4.8e-6 Bohr from a point of the SK grid (0.02 Bohr),
# methanol an O-H distance 1.6e-4 Bohr from one: the derivative of DFTB+'s sliding-window
# interpolation jumps there, and differences across such a point are meaningless.  These
# copies are moved by a few mAngstrom (test_geometries_off_grid).
WATER_SHIFT = [[-0.002, -0.002, 0.003], [-0.004, 0.001, 0.002], [-0.003, -0.004, -0.002]]
MEOH_SHIFT = [[0.0, 0.005, -0.004], [0.004, -0.002, -0.001], [0.003, -0.001, 0.0],
              [-0.005, 0.003, 0.0], [-0.002, 0.003, -0.002], [0.0, -0.004, -0.001]]


def _water():
    atoms = C._water()
    atoms.positions += WATER_SHIFT
    return atoms


def _meoh():
    atoms = C._meoh()
    atoms.positions += MEOH_SHIFT
    return atoms


def _grid_offset(atoms, dr=0.02):
    """Smallest distance (Bohr) of a pair distance from a multiple of ``dr``."""
    pos = atoms.positions / BOHR
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)[np.triu_indices(len(pos), 1)]
    return np.abs(d / dr - np.round(d / dr)).min() * dr


def _calc(atoms, analysis=None, shell_resolved=False, **kw):
    labels = atoms.get_chemical_symbols()
    p = ParamSet(SKF, max_l={e: C.MAXL[e] for e in set(labels)}, shell_resolved=shell_resolved)
    return DFTB(p.skset(labels), scc_tol=1e-13, max_scc=500, analysis=analysis, **kw), labels


def _run(atoms, analysis=None, **kw):
    calc, labels = _calc(atoms, analysis, **kw)
    return calc.calculate(labels, atoms.positions / BOHR)


def _differences(atoms, quantity, h=1e-4, **kw):
    """Fourth-order central differences of ``quantity(result)`` w.r.t. every coordinate:
    (n_atom, 3, ...)."""
    calc, labels = _calc(atoms, **kw)
    x0 = atoms.positions / BOHR
    out = []
    for a in range(len(labels)):
        for c in range(3):
            vals = []
            for s in (2, 1, -1, -2):
                x = x0.copy()
                x[a, c] += s * h
                vals.append(np.asarray(quantity(calc.calculate(labels, x)), float))
            out.append((8 * (vals[1] - vals[2]) - (vals[0] - vals[3])) / (12 * h))
    return np.array(out).reshape((len(labels), 3) + out[0].shape)


@pytest.mark.parametrize("atoms, kw, pert", [
    (_water(), {}, None),
    (_water(), {"shell_resolved": True}, None),
    (_water(), {"scc": False}, None),
    (_meoh(), DFTB3, None),
    (_water(), DOUBLET, None),
    (_water(), {"electric_field": FIELD}, None),
    (_water(), {"point_charges": PC}, None),
    (C._cu4co(), {"smearing": ("fermi", 3000.0)}, EXACT),
    (_co2(), {}, None),
], ids=["water", "water_shell", "water_nonscc", "ch3oh_dftb3", "water_doublet", "water_field",
        "water_pc", "cu4co_3000K", "co2"])
def test_born_charges_against_dipole_differences(atoms, kw, pert):
    """Z[K, i, j] = d mu_j / d x_K,i against fourth-order differences of the dipole: also where
    DFTB+ refuses (spin, metals, degenerate levels) or leaves terms out (fields, point
    charges, the full third order in 25.1)."""
    analysis = {"coord_derivatives": True, "perturbation": pert}
    cd = _run(atoms, analysis, **kw).analysis.coord_derivatives
    fd = _differences(atoms, lambda r: r.dipole, **kw)
    assert np.abs(cd.born - fd).max() < 1e-7 * max(1.0, np.abs(fd).max())


def test_geometries_off_grid():
    """The differences here reach 2e-4 Bohr: every pair distance stays 5e-4 Bohr away from
    the SK grid (and the reference water does not)."""
    for atoms in (_water(), _meoh(), C._o2(), C._cu4co(), _co2()):
        assert _grid_offset(atoms) > 5e-4
    assert _grid_offset(C._water()) < 1e-5


@pytest.mark.parametrize("kw", [{"charge": -1.0, "smearing": ("fermi", 300.0)}, DOUBLET],
                         ids=["anion_300K", "cation_doublet"])
def test_charge_derivatives_and_sum_rules(kw):
    """dN_B/dx against differences of the populations; charge conservation, translation
    invariance and the acoustic sum rule sum_K Z_K = Q I for charged waters (a half-filled
    level at 300 K, a spin doublet)."""
    atoms = _water()
    cd = _run(atoms, {"coord_derivatives": True}, **kw).analysis.coord_derivatives
    fd = _differences(atoms, lambda r: r.charges, **kw)
    assert np.abs(cd.atom[:, :, 0, :] - fd).max() < 1e-7
    assert np.abs(cd.atom[:, :, 0, :].sum(-1)).max() < 1e-11          # electrons conserved
    assert np.abs(cd.atom[:, :, 0, :].sum(0)).max() < 1e-11           # translations
    np.testing.assert_allclose(cd.born.sum(0), kw["charge"] * np.eye(3), atol=1e-10)


def test_eigenvalue_and_fermi_derivatives():
    """d eps / dx (non-degenerate levels) and dE_F / dx of a metal against differences."""
    atoms = _water()
    cd = _run(atoms, {"coord_derivatives": True}).analysis.coord_derivatives
    fd = _differences(atoms, lambda r: np.ravel(r.eigenvalues[0]))
    assert np.abs(cd.eigenvalue_derivatives[:, :, 0, :] - fd).max() < 1e-7
    metal = C._cu4co()
    kw = {"smearing": ("fermi", 3000.0)}
    analysis = {"coord_derivatives": True, "perturbation": EXACT}
    cd = _run(metal, analysis, **kw).analysis.coord_derivatives
    assert cd.metallic and cd.dos_fermi[0] > 0
    fd = _differences(metal, lambda r: r.fermi, **kw)
    assert np.abs(cd.fermi_derivatives[:, :, 0] - fd).max() < 1e-8


def test_filling_window():
    """DFTB+'s window (the default) leaves out the levels within 1e-7 of empty or full: about
    1e-7 of a metal's response at 3000 K; filling_tol 1e-15 keeps every partly filled level
    (exact, test above).  An insulator does not notice."""
    metal = C._cu4co()
    kw = {"smearing": ("fermi", 3000.0)}
    dftbp = _run(metal, {"coord_derivatives": True}, **kw).analysis.coord_derivatives
    exact = _run(metal, {"coord_derivatives": True, "perturbation": EXACT},
                 **kw).analysis.coord_derivatives
    assert 1e-8 < np.abs(dftbp.born - exact.born).max() < 1e-6
    water = _water()
    a = _run(water, {"coord_derivatives": True}).analysis.coord_derivatives
    b = _run(water, {"coord_derivatives": True, "perturbation": EXACT}).analysis.coord_derivatives
    assert np.array_equal(a.born, b.born)
    for bad in (0.0, -1e-3):
        with pytest.raises(ValueError, match="filling_tol must be positive"):
            _calc(water, {"coord_derivatives": True, "perturbation": {"filling_tol": bad}})


def test_degenerate_runs_rotation_free():
    """CO2's degenerate pi levels: the coordinate response does not depend on the
    eigenvectors' rotation inside a run (DFTB+ refuses any degeneracy)."""
    atoms = _co2()
    calc, labels = _calc(atoms, {"coord_derivatives": True})
    st = calc._setup(labels, atoms.positions / BOHR, None, None, True, {})
    sc = calc._scc(st, None, False, {})
    opts = resp.perturbation_options(None)
    z0 = resp.coordinate_derivatives(calc, st, sc, True, opts).born
    orig = resp.GroundState.__init__
    rng = np.random.default_rng(5)

    def rotated(self, *a, **k):
        orig(self, *a, **k)
        for blk in self.blocks:
            for gid in np.unique(blk.group):
                idx = np.flatnonzero(blk.group == gid)
                if idx.size == 2:
                    t = rng.uniform(0, np.pi)
                    u = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                    blk.c[:, idx] = blk.c[:, idx] @ u
                    blk.sc[:, idx] = blk.sc[:, idx] @ u

    resp.GroundState.__init__ = rotated
    try:
        z1 = resp.coordinate_derivatives(calc, st, sc, True, opts).born
    finally:
        resp.GroundState.__init__ = orig
    assert np.abs(z1 - z0).max() < 1e-9


def test_options_and_invariances():
    """A subset of atoms gives the full result's rows; dispersion changes nothing."""
    atoms = _meoh()
    full = _run(atoms, {"coord_derivatives": True}).analysis.coord_derivatives
    part = _run(atoms, {"coord_derivatives": {"atoms": [4, 1]}}).analysis.coord_derivatives
    np.testing.assert_array_equal(part.atoms, [1, 4])
    assert np.abs(part.born - full.born[[1, 4]]).max() < 1e-12
    disp = _run(atoms, {"coord_derivatives": True},
                dispersion={"model": "lennardjones"}).analysis.coord_derivatives
    assert np.array_equal(disp.born, full.born)
    with pytest.raises(ValueError, match="unknown keys"):
        _calc(atoms, {"coord_derivatives": {"atom": [1]}})
    with pytest.raises(ValueError, match="out of range"):
        _run(atoms, {"coord_derivatives": {"atoms": [9]}})


def test_refused():
    with pytest.raises(ValueError, match="molecules only"):
        a = C._water_box()
        calc, labels = _calc(a, {"coord_derivatives": True})
        calc.calculate(labels, a.positions / BOHR, a.cell.array / BOHR, np.ones(3, bool))
    with pytest.raises(ValueError, match="Fermi filling"):
        _calc(_water(), {"coord_derivatives": True}, smearing=("mp", 300.0, 1))
    with pytest.raises(ValueError, match="M6g-3"):
        _calc(_water(), {"coord_derivatives": True}, plus_u={"O": [("p", 0.1)]})


def test_refused_torch():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="analysis"):
        _calc(_water(), {"coord_derivatives": True}, backend="torch")


# ---- second derivatives and normal modes ----------------------------------------------------
def test_hessian_symmetric_and_partial():
    """Methanol's Hessian (off the SK grid) is symmetric; a partial Hessian is the full one's
    block; 6 n_moved + 1 calculations."""
    atoms = _meoh()
    calc, labels = _calc(atoms)
    calls = []
    orig = calc.calculate

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    calc.calculate = counting
    sd = second_derivatives(calc, labels, atoms.positions / BOHR)
    assert len(calls) == 6 * len(labels) + 1
    h = sd.hessian
    assert np.abs(h - h.T).max() < 1e-6 * np.abs(h).max()
    part = second_derivatives(calc, labels, atoms.positions / BOHR, moved_atoms=[2, 3])
    assert part.partial and part.hessian.shape == (18, 6)
    assert np.abs(part.hessian - h[:, 6:12]).max() < 1e-9
    assert sd.dipole_derivatives.shape == (18, 3)
    # the dipole derivatives are the Born charges (pydftb's sign)
    cd = _run(atoms, {"coord_derivatives": True}).analysis.coord_derivatives
    assert np.abs(sd.dipole_derivatives - cd.born.reshape(-1, 3)).max() < 1e-6


def test_hessian_converges_as_delta_squared():
    """Central differences: halving delta divides the change by 4; steps up to 1e-3 Bohr keep
    the shifted water's distances inside their SK grid intervals."""
    atoms = _water()
    calc, labels = _calc(atoms)
    x = atoms.positions / BOHR
    h = [second_derivatives(calc, labels, x, delta=d).hessian for d in (1e-3, 5e-4, 2.5e-4)]
    assert 3.9 < np.abs(h[0] - h[1]).max() / np.abs(h[1] - h[2]).max() < 4.1


def test_polarisability_derivatives():
    """d alpha / dx of the first frequency: one calculation with the analysis per geometry."""
    atoms = _water()
    calc, labels = _calc(atoms, {"polarisability": True})
    sd = second_derivatives(calc, labels, atoms.positions / BOHR, moved_atoms=[0])
    assert sd.polarisability_derivatives.shape == (3, 3, 3)
    h = 1e-4
    x = atoms.positions / BOHR
    for c in range(3):
        xp, xm = x.copy(), x.copy()
        xp[0, c] += h
        xm[0, c] -= h
        a_p = calc.calculate(labels, xp).analysis.polarisability.tensor[0]
        a_m = calc.calculate(labels, xm).analysis.polarisability.tensor[0]
        assert np.abs(sd.polarisability_derivatives[c] - (a_p - a_m) / (2 * h)).max() < 1e-8


def test_normal_modes():
    """Projected rigid motions: 6 zero frequencies for methanol, 5 for linear CO2, without
    intensities; the average triangle equals eigh of the symmetrised mass-weighted Hessian;
    the modes are orthonormal in the mass-weighted metric."""
    for atoms, n_zero in ((_meoh(), 6), (_co2(), 5)):
        calc, labels = _calc(atoms)
        x = atoms.positions / BOHR
        sd = second_derivatives(calc, labels, x)
        m = masses_of(calc, labels)
        nm = normal_modes(sd.hessian, m, x, born=sd.dipole_derivatives,
                          remove_translation=True, remove_rotation=True)
        small = np.abs(nm.wavenumbers) < 0.5
        assert small.sum() == n_zero
        assert not nm.ir[small].any()       # modes: noise where the rounding makes lambda > 0
        assert nm.ir[~small].max() > 1e-3
        plain = normal_modes(sd.hessian, m)
        mw = np.repeat(np.sqrt(np.asarray(m) * AMU_AU), 3)
        d = sd.hessian / np.outer(mw, mw)
        lam = np.linalg.eigvalsh(0.5 * (d + d.T))
        f = plain.frequencies                # the rigid motions' sqrt amplifies the noise
        np.testing.assert_allclose(np.sign(f) * f * f, lam, rtol=0, atol=1e-16)
        v = plain.mass_weighted
        np.testing.assert_allclose(v.T @ v, np.eye(len(v)), atol=1e-10)
    with pytest.raises(ValueError, match="square"):
        normal_modes(np.zeros((9, 6)), [1.0, 1.0, 1.0])
    assert HARTREE_CM == pytest.approx(219474.63137054)


def test_text_outputs(tmp_path):
    atoms = _water()
    calc, labels = _calc(atoms, {"coord_derivatives": True})
    r = calc.calculate(labels, atoms.positions / BOHR)
    text = io.coord_derivative_tables(r)
    assert " Derivatives of atomic Mulliken charges with atom positions" in text
    assert "At1      x               y               z" in text
    assert " Born effective charges (a.u.)" in text
    io.write_band_derivatives(tmp_path / "dBand_dx.out", r, "coordinate")
    assert (tmp_path / "dBand_dx.out").read_text().startswith("Atom 1 dx  KPT ")
    plain, labels = _calc(atoms)
    sd = second_derivatives(plain, labels, atoms.positions / BOHR, atoms=[0, 1, 2],
                            moved_atoms=[1, 2])
    path = io.write_hessian(tmp_path / "hessian.out", sd)
    assert path.endswith("hessian.out.2-3")
    values = np.array(open(path).read().split(), float)        # 4 per line, column by column
    np.testing.assert_allclose(values, sd.hessian.T.ravel(), atol=1e-10)
    assert io.born_table(sd).startswith(" Born charges/dipole derivatives wrt. atom positions")
    full = second_derivatives(plain, labels, atoms.positions / BOHR)
    nm = normal_modes(full.hessian, masses_of(plain, labels), atoms.positions / BOHR,
                      born=full.dipole_derivatives)
    io.write_vibrations_tag(tmp_path / "vibrations.tag", nm)
    tag = parse_vibrations_tag(tmp_path / "vibrations.tag")
    np.testing.assert_allclose(tag["frequencies"], nm.frequencies, rtol=1e-14)
    np.testing.assert_allclose(tag["intensities"], nm.grouped(nm.ir), rtol=1e-14)
