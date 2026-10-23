"""Occupations and overrides of DFTB+'s Hamiltonian block (milestone M6f-2): a fixed Fermi
level, independent k-point filling, custom Hubbard U, custom reference occupations and
truncated SK tables.

The results against DFTB+ are the frozen references *_fixed_ef*, *_indep_k*, *_custom_* and
*_truncated_sk* in test_reference.py; here the rules themselves: the energy whose gradient the
forces and the stress are, the limits in which nothing changes, the copies of the SK set and
the refusals.
"""
import gzip
import os
import warnings

import numpy as np
import pytest
from ase import units

from pydftb.ase_calc import PyDFTB
from pydftb.bands import eigenstates
from pydftb.calculator import DFTB
from pydftb.constants import BOHR, HARTREE
from pydftb.dftbplus_io import ChargeData
from pydftb.occupations import fill, fill_at, fill_independent_k
from pydftb.params import ParamSet
from pydftb.validation import cases, refgen

from .conftest import SKF_HCOCU
from .test_spin import _system


def _case(name, **options):
    """pydftb's result and engine for a reference case (its M6f-2 options included)."""
    case = refgen.CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    res, calc = refgen.run_pydftb(case, inp, engine=True, **options)
    pos = np.array(inp["positions_angstrom"]) / BOHR
    cell = None if inp["cell_angstrom"] is None else np.array(inp["cell_angstrom"]) / BOHR
    return res, calc, inp["labels"], pos, cell


def _fd(energy, h):
    """dE/dx of ``energy(x)`` at 0 by 4-point central differences."""
    return (8.0 * (energy(h) - energy(-h)) - (energy(2 * h) - energy(-2 * h))) / (12.0 * h)


def _energy_at(calc, labels, pos, cell, attr="e_force_related"):
    pbc = None if cell is None else np.ones(3, bool)

    def at_positions(p):
        return getattr(calc.calculate(labels, p, cell, pbc, forces=False), attr)

    def at_strain(a, b, x):
        e = np.zeros((3, 3))
        e[a, b] += 0.5 * x
        e[b, a] += 0.5 * x
        m = np.eye(3) + e
        return getattr(calc.calculate(labels, pos @ m.T, cell @ m.T, pbc, forces=False), attr)

    return at_positions, at_strain


# ---- the filling functions ----------------------------------------------------------------
@pytest.mark.parametrize("smearing", [("fermi", 300.0), ("mp", 2000.0, 2), ("gaussian", 1000.0)])
def test_fill_at_the_fermi_level_of_fill(smearing):
    """At fill's Fermi level, fill_at gives its occupations and TS (fill rescales to the exact
    electron count, a relative change below 1e-13 here)."""
    rng = np.random.default_rng(1)
    ev = np.sort(rng.normal(scale=0.2, size=(3, 12)), axis=1)
    kw = np.array([0.5, 0.3, 0.2])
    occ, mu, ts = fill(ev, kw, 9.3, smearing)
    occ2, mu2, ts2 = fill_at(ev, kw, mu, smearing)
    assert mu2 == mu
    assert np.allclose(occ2, occ, rtol=0.0, atol=1e-12)
    assert ts2 == pytest.approx(ts, rel=1e-12, abs=1e-15)
    assert sum(w * o.sum() for w, o in zip(kw, occ2)) == pytest.approx(9.3, abs=1e-10)
    one, _, _ = fill_at(ev, kw, mu, smearing, spin_factor=1.0)       # one electron per level
    assert np.allclose(2.0 * np.array(one), occ2, rtol=0.0, atol=1e-15)


def test_independent_k_filling():
    """One k-point: exactly fill; several: every k-point holds the electrons on its own, the
    Fermi level and TS are the weighted sums of the k-points' own."""
    rng = np.random.default_rng(2)
    ev = np.sort(rng.normal(scale=0.2, size=(3, 10)), axis=1)
    kw = np.array([0.5, 0.3, 0.2])
    for smearing in (("fermi", 300.0), ("mp", 2000.0, 2)):
        for n in (8.0, 9.3):                       # integer: DFTB+'s mid-gap rule per k-point
            a = fill(ev[:1], [1.0], n, smearing)
            b = fill_independent_k(ev[:1], [1.0], n, smearing)
            assert np.array_equal(a[0][0], b[0][0]) and a[1] == b[1] and a[2] == b[2]
            occ, mu, ts = fill_independent_k(ev, kw, n, smearing)
            own = [fill(e[None], [1.0], n, smearing) for e in ev]
            for o, (ok, _, _) in zip(occ, own):         # mid-gap: N within 1e-7, as DFTB+
                assert np.array_equal(o, ok[0]) and o.sum() == pytest.approx(n, abs=1e-7)
            assert mu == pytest.approx(sum(w * m for w, (_, m, _) in zip(kw, own)), abs=1e-15)
            assert ts == pytest.approx(sum(w * t for w, (_, _, t) in zip(kw, own)), abs=1e-15)


# ---- a fixed Fermi level ------------------------------------------------------------------
def test_fixed_fermi_level_at_the_scc_level_is_the_scc():
    """Fixed at the Fermi level of the normal SCC, the electron count stays and the solution is
    the same; E_FR is then F - N E_f.  Without a fixed level E_FR is F (n_ef None)."""
    ref, _, _, _, _ = _case("cu4co_300K", scc_tol=1e-10)
    assert ref.n_ef is None and ref.e_force_related == ref.free_energy
    r, _, _, _, _ = _case("cu4co_300K", scc_tol=1e-10, fixed_fermi_level=ref.fermi)
    assert r.fermi == ref.fermi
    assert r.n_electrons == pytest.approx(54.0, abs=1e-8)
    assert r.energy == pytest.approx(ref.energy, abs=1e-9)
    assert r.free_energy == pytest.approx(ref.free_energy, abs=1e-9)
    assert r.e_force_related == pytest.approx(r.free_energy - r.n_electrons * ref.fermi,
                                              abs=1e-12)
    assert np.allclose(r.forces, ref.forces, atol=1e-7)
    assert np.allclose(r.charges, ref.charges, atol=1e-8)


@pytest.mark.parametrize("name, atom", [("cu4co_300K_fixed_ef", 0), ("o2_triplet_fixed_ef_1000K", 1),
                                        ("o2_noncolinear_soc_fixed_ef_1000K", 0)])
def test_forces_are_the_gradient_of_the_force_related_energy(name, atom):
    """-dE_FR/dR, E_FR = F - sum_s N_s E_f,s: molecule, collinear (two levels), two-component.
    The Mermin free energy is not what the forces are the gradient of.  SCC to 1e-9 at most
    (the reference of the collinear case stops at 1e-7)."""
    res, calc, labels, pos, cell = _case(name, scc_tol=min(refgen.CASES_BY_NAME[name].scc_tol,
                                                            1e-9))
    at_positions, _ = _energy_at(calc, labels, pos, cell)
    for k in (0, 2):
        def e(x):
            p = pos.copy()
            p[atom, k] += x
            return at_positions(p)
        assert -_fd(e, 1e-3) == pytest.approx(res.forces[atom, k], abs=2e-8)
    if name == "cu4co_300K_fixed_ef":       # dN/dR is not zero: F alone is far off
        at_f, _ = _energy_at(calc, labels, pos, cell, "free_energy")

        def f(x):
            p = pos.copy()
            p[0, 0] += x
            return at_f(p)
        assert abs(-_fd(f, 1e-3) - res.forces[0, 0]) > 1e-3


@pytest.mark.parametrize("name", ["cu4co_box_fixed_ef", "cu_fcc_k6_fixed_ef"])
def test_stress_of_a_charged_cell_is_the_strain_derivative(name):
    """The cell exchanges electrons with the reservoir, so it is charged: the stress is
    -(1/V) dE_FR/d(strain), background term included (DFTB+'s is not, M6f-2 design)."""
    res, calc, labels, pos, cell = _case(name)
    assert abs(res.n_electrons - round(res.n_electrons)) > 1e-3        # really charged
    _, at_strain = _energy_at(calc, labels, pos, cell)
    vol = abs(np.linalg.det(cell))
    for a, b in [(0, 0), (2, 2), (0, 1)]:
        sigma = -_fd(lambda x: at_strain(a, b, x), 1e-4) / vol
        assert sigma == pytest.approx(res.stress[a, b], abs=1e-11)


def test_fixed_fermi_level_band_states_and_charge_files():
    """A band run at the SCC charges fills at the fixed level; a charge file with another
    electron count is accepted (DFTB+ does not check it then)."""
    res, calc, labels, pos, cell = _case("cu_fcc_k6_fixed_ef")
    st = eigenstates(calc, labels, pos, cell, np.ones(3, bool), res.q_in)
    assert np.allclose(st.fermi, [-0.10])
    assert np.allclose(st.occupations[0], np.array(res.occupations), atol=1e-10)
    data = calc.charge_data(labels, res)
    assert data.q[0].sum() == pytest.approx(res.n_electrons, abs=1e-8)
    calc.populations_from_file(labels, data)                           # no check
    plain = DFTB(calc.sk, kpts=calc.kpts, smearing=calc.smearing)
    with pytest.raises(ValueError, match="electrons instead of"):
        plain.populations_from_file(labels, ChargeData(data.q, data.norb_atom))


# ---- independent k-point filling ----------------------------------------------------------
def test_independent_k_filling_forces_stress_and_gamma():
    """Every k-point holds the electron count: the forces and the stress are the gradient of
    the Mermin free energy; at the Gamma point only it is the normal filling, bit for bit."""
    res, calc, labels, pos, cell = _case("cuo_chain_k422_indep_k")
    occ = np.array(res.occupations)
    assert np.allclose(occ.sum(axis=1), 34.0, atol=1e-10)              # every k-point
    at_positions, at_strain = _energy_at(calc, labels, pos, cell, "free_energy")

    def e(x):
        p = pos.copy()
        p[1, 0] += x
        return at_positions(p)
    assert -_fd(e, 1e-3) == pytest.approx(res.forces[1, 0], abs=2e-8)
    vol = abs(np.linalg.det(cell))
    assert -_fd(lambda x: at_strain(0, 0, x), 1e-4) / vol == pytest.approx(res.stress[0, 0],
                                                                           abs=1e-10)
    ref, _, labels, pos, cell = _case("water_box_mp1_1000K")
    r, _, _, _, _ = _case("water_box_mp1_1000K", independent_k_filling=True)
    assert r.energy == ref.energy and np.array_equal(r.forces, ref.forces)
    assert np.array_equal(r.stress, ref.stress)


# ---- custom Hubbard U ---------------------------------------------------------------------
def test_with_hubbard_copies_and_replaces_positive_values_only():
    _, sk, labels, pos, _, _, _ = _system("ch3oh")
    before = {k: v.copy() for k, v in sk.hubbard.items()}
    new = sk.with_hubbard({"O": 0.5, "H": -1.0, "Xx": 3.0})       # Xx: not in this set
    assert new.hubbard["O"].tolist() == [0.5, 0.5]                 # atom-resolved: every shell
    assert np.array_equal(new.hubbard["H"], before["H"])           # <= 0 keeps the SKF's
    assert new.files is sk.files
    assert all(np.array_equal(sk.hubbard[k], v) for k, v in before.items())
    with pytest.raises(ValueError, match="one value per type"):
        sk.with_hubbard({"O": [0.5, 0.6]})
    _, sks, *_ = _system("cu4co_shell_resolved")
    new = sks.with_hubbard({"Cu": [0.3, 0.0, 0.4]})
    assert new.hubbard["Cu"].tolist() == [0.3, sks.hubbard["Cu"][1], 0.4]
    with pytest.raises(ValueError, match="one value per shell"):
        sks.with_hubbard({"Cu": 0.3})


def _rewrite_hubbard(src, dst, label, u):
    """Copies of the SK files of ``src`` with the U of ``label``'s homonuclear file set to
    ``u`` (simple format: line 2 is E_d E_p E_s SPE U_d U_p U_s f_d f_p f_s)."""
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        with gzip.open(os.path.join(src, fn), "rt") as fh:
            lines = fh.read().splitlines()
        if fn == f"{label}-{label}.skf.gz":
            vals = lines[1].replace(",", " ").split()
            vals[4:7] = [repr(float(u))] * 3
            lines[1] = " ".join(vals)
        with open(os.path.join(dst, fn[:-3]), "w") as fh:
            fh.write("\n".join(lines) + "\n")


def test_custom_hubbards_equal_a_rewritten_skf(tmp_path):
    case, sk, labels, pos, _, _, _ = _system("ch3oh")
    _rewrite_hubbard(SKF_HCOCU, str(tmp_path), "O", 0.5)
    max_l = refgen._inputs(case)["max_l"]
    sk2 = ParamSet(str(tmp_path), max_l=max_l).skset(labels)
    assert sk2.hubbard["O"].tolist() == [0.5, 0.5]
    a = DFTB(sk, hubbard={"O": 0.5}).calculate(labels, pos)
    b = DFTB(sk2).calculate(labels, pos)
    assert a.energy == b.energy and np.array_equal(a.forces, b.forces)
    c = DFTB(sk, hubbard={"O": 0.0, "H": -1.0}).calculate(labels, pos)   # nothing replaced
    d = DFTB(sk).calculate(labels, pos)
    assert c.energy == d.energy and np.array_equal(c.forces, d.forces)


# ---- custom reference occupations ----------------------------------------------------------
def test_reference_occupations():
    """The SKF's own values change nothing; O p = 4.2 adds 0.2 electrons, charges, dipole and
    the SCC energy are relative to the new q0; the input is checked."""
    _, sk, labels, pos, _, _, _ = _system("ch3oh")                  # C O H H H H
    plain = DFTB(sk).calculate(labels, pos)
    same = DFTB(sk, reference_occupations={1: {"s": 2.0, "p": 4.0}, 0: [2.0, 2.0]}).calculate(
        labels, pos)
    assert same.energy == plain.energy and np.array_equal(same.charges, plain.charges)
    assert plain.q0_atom is None and same.q0_atom.tolist() == [4.0, 6.0, 1.0, 1.0, 1.0, 1.0]
    r = DFTB(sk, smearing=("fermi", 300.0), reference_occupations={1: {"p": 4.2}}).calculate(
        labels, pos)
    assert r.q0_atom[1] == pytest.approx(6.2)
    assert r.n_electrons == pytest.approx(14.2, abs=1e-10)
    assert (r.q0_atom - r.charges).sum() == pytest.approx(0.0, abs=1e-10)
    assert np.allclose(r.dipole, -((r.charges - r.q0_atom) @ pos), atol=1e-12)
    for bad, msg in [({6: {"s": 1.0}}, "no atom 6"), ({2: {"p": 1.0}}, "no shell 'p' on H"),
                     ({1: [2.0]}, "one number per shell")]:
        with pytest.raises(ValueError, match=msg):
            DFTB(sk, reference_occupations=bad).calculate(labels, pos)


def test_reference_occupations_with_spin_blocks_and_ase():
    """The start of a collinear-spin and of a block-potential SCC from the custom q0; the ASE
    calculator's charges are relative to it."""
    for name, oxygen in [("o2_triplet", 0), ("ch3oh_ons", 1)]:
        case = refgen.CASES_BY_NAME[name]
        r = refgen.run_pydftb(case, refgen._inputs(case), smearing=("fermi", 300.0),
                              reference_occupations={oxygen: {"p": 4.1}})
        assert r.q0_atom[oxygen] == pytest.approx(6.1)
        assert r.n_electrons == pytest.approx(r.q0_atom.sum(), abs=1e-9)
    atoms = cases._meoh()
    calc = PyDFTB(SKF_HCOCU, max_l=refgen._inputs(refgen.CASES_BY_NAME["ch3oh"])["max_l"],
                  smearing=("fermi", 300.0), scc_tol=1e-10,
                  reference_occupations={1: {"p": 4.2}})
    atoms.calc = calc
    q = atoms.get_charges()
    assert q.sum() == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(q, calc.raw.q0_atom - calc.raw.charges)


# ---- truncated SK tables ------------------------------------------------------------------
def test_truncated_sk_set_is_a_cached_copy():
    _, sk, labels, pos, _, _, _ = _system("water_dimer_hdamp")
    files = dict(sk.files)
    tables = {k: f.table.copy() for k, f in sk.files.items()}
    coefs = {k: f._coefficients() for k, f in sk.files.items()}          # filled caches
    t = sk.truncated(6.0)
    assert sk.truncated(6.0) is t and sk.truncated(6.0, False) is not t
    for k, f in sk.files.items():                                       # nothing changed
        assert f is files[k] and np.array_equal(f.table, tables[k]) and f._coef is coefs[k]
    for k, f in t.files.items():
        n = min(int(np.floor(5.0 / f.dr)), tables[k].shape[0])
        assert f.table.shape[0] == n and np.array_equal(f.table, tables[k][:n])
        assert f.cutoff <= 6.0 + 1e-12 and f.repulsive is files[k].repulsive
    assert all(f.cutoff <= 7.0 + 1e-12 for f in sk.truncated(6.0, False).files.values())
    long = sk.truncated(1000.0)
    assert all(long.files[k] is files[k] for k in files)
    with pytest.raises(ValueError, match="shorter than"):
        sk.truncated(0.5)
    with pytest.raises(ValueError, match="interpolation needs 8"):
        sk.truncated(1.1)


def test_truncation_longer_than_the_tables_changes_nothing():
    _, sk, labels, pos, _, _, _ = _system("water_dimer_hdamp")
    with pytest.warns(RuntimeWarning, match="Artificially truncating the SK table"):
        calc = DFTB(sk, truncate_sk_range=1000.0)
    a = calc.calculate(labels, pos)
    b = DFTB(sk).calculate(labels, pos)
    assert a.energy == b.energy and np.array_equal(a.forces, b.forces)
    with pytest.warns(RuntimeWarning):
        c = DFTB(sk, truncate_sk_range={"max_distance": 6.0, "hard_cutoff": False})
    assert c.sk is sk.truncated(6.0, False)
    with pytest.raises(ValueError, match="unknown keys"):
        DFTB(sk, truncate_sk_range={"distance": 6.0})


# ---- the ASE calculator ------------------------------------------------------------------
def test_ase_units_and_force_consistent_energy():
    """eV and Angstrom in, the force-related energy as ASE's force-consistent energy."""
    case = refgen.CASES_BY_NAME["cu4co_300K_fixed_ef"]
    max_l = refgen._inputs(case)["max_l"]
    atoms = cases._cu4co()
    atoms.calc = PyDFTB(SKF_HCOCU, max_l=max_l, smearing=("fermi", 300.0), scc_tol=1e-9,
                        max_scc=500, fixed_fermi_level=-0.140 * HARTREE)
    e = atoms.get_potential_energy()
    f = atoms.get_potential_energy(force_consistent=True)
    raw = atoms.calc.raw
    assert raw.fermi == pytest.approx(-0.140, abs=1e-15)
    assert e == pytest.approx(raw.energy * units.Hartree, abs=1e-12)
    assert f == pytest.approx(raw.e_force_related * units.Hartree, abs=1e-12)
    atoms = cases._water_dimer()
    max_l = refgen._inputs(refgen.CASES_BY_NAME["water_dimer_hdamp"])["max_l"]
    with pytest.warns(RuntimeWarning):
        atoms.calc = PyDFTB(SKF_HCOCU, max_l=max_l, scc_tol=1e-10,
                            truncate_sk_range=6.0 * BOHR)
        e = atoms.get_potential_energy()
    _, sk, labels, pos, _, _, _ = _system("water_dimer_hdamp")
    with pytest.warns(RuntimeWarning):
        ref = DFTB(sk, scc_tol=1e-10, truncate_sk_range=6.0 * BOHR / BOHR).calculate(labels, pos)
    assert e == pytest.approx(ref.energy * units.Hartree, abs=1e-9)


# ---- refusals -----------------------------------------------------------------------------
def test_combinations_that_are_refused():
    _, sk, labels, pos, _, _, _ = _system("h2o")
    w = {"O": -0.028, "H": -0.072}
    with pytest.raises(ValueError, match="relax_total_spin"):
        DFTB(sk, spin=True, spin_constants=w, relax_total_spin=True, fixed_fermi_level=-0.2)
    with pytest.raises(ValueError, match="does not combine with a fixed Fermi level"):
        DFTB(sk, fixed_fermi_level=-0.2, independent_k_filling=True)
    with pytest.raises(ValueError, match="DFTB\\+ 25.1 crashes"):
        DFTB(sk, spin=True, spin_constants=w, independent_k_filling=True)
    with pytest.raises(ValueError, match="needs a periodic system"):
        DFTB(sk, independent_k_filling=True).calculate(labels, pos)
    with pytest.raises(ValueError, match="SP2"):
        DFTB(sk, solver="sp2", fixed_fermi_level=-0.2)
    with pytest.raises(ValueError, match="one value or \\(up, down\\)"):
        DFTB(sk, spin=True, spin_constants=w, fixed_fermi_level=(-0.1, -0.2, -0.3))
    with pytest.raises(ValueError, match="one value, not 2"):
        DFTB(sk, fixed_fermi_level=(-0.2, -0.3))
    assert DFTB(sk, spin=True, spin_constants=w, fixed_fermi_level=-0.2).fixed_fermi_level == (
        -0.2, -0.2)
    with pytest.raises(ValueError, match="fixed electron count"):
        DFTB(sk, fixed_fermi_level=-0.2, casida={"n_excitations": 3})
    with pytest.raises(ValueError, match="Custom occupation not compatible"):
        DFTB(sk, reference_occupations={0: {"p": 4.1}}, casida={"n_excitations": 3})
    with pytest.raises(ValueError, match="needs SCC"):
        DFTB(sk, scc=False, hubbard={"O": 0.5})
    with pytest.raises(ValueError, match="XL-BOMD with a fixed Fermi level"):
        DFTB(sk, fixed_fermi_level=-0.2, force_evaluation="dynamics",
             smearing=("fermi", 300.0)).calculate(labels, pos, xl=True)


@pytest.mark.parametrize("option", [{"fixed_fermi_level": -0.2}, {"independent_k_filling": True},
                                    {"hubbard": {"O": 0.5}},
                                    {"reference_occupations": {0: {"p": 4.1}}},
                                    {"truncate_sk_range": 6.0}])
def test_torch_backend_and_autodiff_refuse(option):
    torch = pytest.importorskip("torch")       # noqa: F841
    from pydftb.autodiff import evaluate
    _, sk, labels, pos, _, _, _ = _system("h2o")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # the truncation's warning
        with pytest.raises(ValueError, match="NumPy backend only"):
            DFTB(sk, backend="torch", **option)
        calc = DFTB(sk, **option)
    with pytest.raises(NotImplementedError, match="NumPy engine only"):
        evaluate(calc, labels, pos)
