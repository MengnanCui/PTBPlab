"""Smooth switching between parameter sets of one element (milestone M5a).

The discrete limit (every weight 0 or 1) against DFTB+ 25.1 is in the frozen references
cu_slab_types_k441 and cu13_types (test_reference.py), where refgen also runs pydftb with the
switch and requires bitwise the explicit labels' results.  Here: the weights and coordination
numbers, the mixing rule, forces and stress as derivatives of the energy inside the switching
windows (with the physics a switch combines with), continuity across a window, energy
conservation in MD, the ASE calculator, band structures and charge files, and the refusals.
Parameters: the cu_types set (validation/make_cu_types_set.py), Cu and Cu2 with different
orbitals and repulsives.
"""
import gzip
import os
import shutil
import warnings

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.cluster import Octahedron

from pydftb.bands import eigenstates
from pydftb.calculator import DFTB, SCCControl
from pydftb.constants import BOHR
from pydftb.hamiltonian import Basis, build_mixed_pairs, build_pairs
from pydftb.kpoints import monkhorst_pack, reduce_time_reversal
from pydftb.params import ParamSet
from pydftb.skf import write_typed_copies
from pydftb.typeswitch import TypeSwitch, check_switches, smoothstep
from pydftb.validation import refgen
from pydftb.validation.cases import SKF_SETS, W_SPIN

SKF_TYPES = SKF_SETS["cu_types"]
SW_MOL = TypeSwitch("Cu", ("Cu2", "Cu"), centre=3.2, width=2.0, cutoff=(4.6, 5.4))


@pytest.fixture(scope="module")
def sk():
    return ParamSet(SKF_TYPES).skset(["Cu", "Cu2"])


def _cu6(seed=21, rattle=0.12):
    """A rattled Cu6 octahedron: its bonds spread over the counting window (4.6, 5.4) Bohr."""
    a = Octahedron("Cu", 2, cutoff=0, latticeconstant=3.61)
    a.rattle(rattle, seed=seed)
    return a.positions / BOHR


def _cu4_cell():
    """A rattled, sheared fcc cell (4 atoms) whose first shell crosses the window (4.4, 5.3)."""
    a = bulk("Cu", "fcc", a=3.61, cubic=True)
    a.rattle(0.15, seed=22)
    a.set_cell(a.cell.array @ np.array([[1.02, 0.01, 0], [0, 0.99, 0.015], [0.0, 0.0, 1.0]]),
               scale_atoms=True)
    return a.positions / BOHR, a.cell.array / BOHR


def _fd(energy, h):
    """dE/dx at 0 by 4-point central differences."""
    return (8.0 * (energy(h) - energy(-h)) - (energy(2 * h) - energy(-2 * h))) / (12.0 * h)


def _three_types(tmp_path):
    """The cu_types set plus Cu3, a copy of Cu2 (hetero-label files without on-site line)."""
    d = str(tmp_path / "three")
    write_typed_copies(SKF_TYPES, d, {"Cu": "Cu", "Cu2": "Cu2", "Cu3": "Cu2"},
                       template="{a}-{b}.skf.gz")
    return d


# ---- weights and coordination numbers ---------------------------------------------------------
def test_smoothstep_and_weights():
    s, ds = smoothstep(np.array([-1.0, 0.0, 0.25, 0.5, 1.0, 2.0]))
    assert list(s[[0, 1, 4, 5]]) == [0.0, 0.0, 1.0, 1.0] and s[3] == 0.5
    assert list(ds[[0, 1, 4, 5]]) == [0.0] * 4 and ds[3] == pytest.approx(1.875)
    sw = TypeSwitch("Cu", ("Cu2", "Cu3", "Cu"), centre=(4.0, 7.0), width=(2.0, 1.0))
    cn = np.linspace(0.0, 10.0, 2001)
    w, dw = sw.weights(cn)
    assert np.allclose(w.sum(1), 1.0, rtol=0, atol=1e-15) and (w >= 0.0).all()
    outside = (cn <= 3.0) | ((cn >= 5.0) & (cn <= 6.5)) | (cn >= 7.5)
    assert set(np.unique(w[outside])) == {0.0, 1.0} and not dw[outside].any()
    assert (w[cn <= 3.0] == [1.0, 0.0, 0.0]).all() and (w[cn >= 7.5] == [0.0, 0.0, 1.0]).all()
    h = 1e-6
    x = np.array([3.3, 4.0, 4.9, 6.6, 7.2])
    fd = (sw.weights(x + h)[0] - sw.weights(x - h)[0]) / (2 * h)
    assert np.allclose(sw.weights(x)[1], fd, rtol=0, atol=1e-8)
    assert sw.reference == "Cu"
    assert TypeSwitch("Cu", ("Cu_a", "Cu_b"), 3.0, 1.0).reference == "Cu_a"


def test_type_switch_checks():
    assert TypeSwitch.make({"element": "Cu", "types": ["Cu2", "Cu"], "centre": 3, "width": 1}) \
        == TypeSwitch("Cu", ("Cu2", "Cu"), 3.0, 1.0)
    bad = [(dict(element="Xx", types=("Cu", "Cu2")), "chemical symbol"),
           (dict(types=("Cu",)), "at least two"), (dict(types=("Cu", "Cu")), "at least two"),
           (dict(types=("Cu", "O2")), "is not Cu"), (dict(types=("Cu", "Cu-2")), "must not"),
           (dict(width=0.0), "positive"), (dict(centre=(3.0, 4.0)), "needs 1 value"),
           (dict(types=("Cu", "Cu2", "Cu3"), centre=(3.0, 3.5), width=(1.0, 1.0)), "overlap"),
           (dict(types=("Cu", "Cu2", "Cu3")), "one value per switching window"),
           (dict(cutoff=(5.0, 4.0)), "r_on < r_off"), (dict(cutoff={"Xx": (1, 2)}), "symbol")]
    for change, msg in bad:
        spec = dict(element="Cu", types=("Cu2", "Cu"), centre=3.0, width=1.0) | change
        with pytest.raises(ValueError, match=msg):
            TypeSwitch(**spec)
    with pytest.raises(ValueError, match="unknown keys"):
        TypeSwitch.make({"element": "Cu", "types": ("Cu2", "Cu"), "center": 3, "width": 1})
    with pytest.raises(ValueError, match="one TypeSwitch per element"):
        check_switches([SW_MOL, SW_MOL])


def test_coordination_numbers_and_their_derivatives():
    sw = TypeSwitch("Cu", ("Cu2", "Cu"), 3.0, 1.0, cutoff=(4.0, 5.0))
    for r, f in [(3.0, 1.0), (4.0, 1.0), (4.5, 0.5), (5.0, 0.0), (6.0, 0.0)]:
        cn = sw.coordination_numbers(["Cu", "Cu"], [[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        assert list(cn) == [f, f]
    pos = _cu6()
    el = ["Cu"] * 6
    cn, (i, j, vec, fr) = SW_MOL.coordination_numbers(el, pos, derivatives=True)
    for a in range(6):
        grad = np.zeros((6, 3))
        g = (fr * (i == a))[:, None] * vec
        np.add.at(grad, j, g)
        np.add.at(grad, i, -g)
        for c in range(18):
            def at(x):
                p = pos.copy()
                p.flat[c] += x
                return SW_MOL.coordination_numbers(el, p)[a]
            assert grad.flat[c] == pytest.approx(_fd(at, 1e-5), abs=1e-9)
    # periodic images: fcc bulk counts its 12 nearest neighbours with the default window
    a = bulk("Cu", "fcc", a=3.61)
    cn = TypeSwitch("Cu", ("Cu2", "Cu"), 10.0, 2.0).coordination_numbers(
        ["Cu"], a.positions / BOHR, a.cell.array / BOHR, np.ones(3, bool))
    assert list(cn) == [12.0]
    # a dict selects the neighbour elements that count (and only the switch's atoms get a CN)
    pos = np.array([[0.0, 0, 0], [3.5, 0, 0], [0, 3.6, 0], [0, 0, 2.0]])
    cn = TypeSwitch("Cu", ("Cu2", "Cu"), 1.0, 1.0, cutoff={"O": (4.0, 5.0)}).coordination_numbers(
        ["Cu", "O", "O", "H"], pos)
    assert list(cn) == [2.0, 0.0, 0.0, 0.0]


def test_mixing_rule(sk):
    """The mixed blocks of a dimer are sum_tu w_At w_Bu of the explicit labels' blocks."""
    pos = np.array([[0.0, 0.0, 0.0], [1.1, 2.9, 3.4]])
    wa, wb = np.array([0.3, 0.7]), np.array([0.8, 0.2])          # columns Cu, Cu2
    basis = Basis.build(["Cu", "Cu"], sk)
    pb, _ = build_mixed_pairs(["Cu", "Cu"], pos, None, None, sk, basis,
                              {"Cu": ["Cu", "Cu2"]}, {"Cu": 0, "Cu2": 1},
                              np.array([wa, wb]), derivatives=True)
    def by_pair(p, values):                       # elements ordered by pair (blocks kept)
        return values[np.argsort(p.pair_of, kind="stable")]

    h = np.zeros_like(pb.h)
    dh = np.zeros_like(pb.dh)
    for t, x in zip(("Cu", "Cu2"), wa):
        for u, y in zip(("Cu", "Cu2"), wb):
            ref = build_pairs([t, u], pos, None, None, sk, basis, derivatives=True)
            h += x * y * by_pair(ref, ref.h)
            dh += x * y * by_pair(ref, ref.dh)
    assert np.allclose(by_pair(pb, pb.h), h, rtol=0, atol=1e-15)
    assert np.allclose(by_pair(pb, pb.dh), dh, rtol=0, atol=1e-13)


# ---- forces and stress inside the windows ----------------------------------------------------
UD = {"Cu": -0.05, "Cu2": -0.05}
MOLECULE_OPTIONS = {
    "scc": dict(smearing=("fermi", 300.0)),
    "non_scc": dict(scc=False, smearing=("fermi", 300.0)),
    "dftb3_spin": dict(third_order="full", hubbard_derivs=UD, spin=True, unpaired_electrons=2.0,
                       spin_constants={"Cu": W_SPIN["Cu"], "Cu2": W_SPIN["Cu"]},
                       smearing=("fermi", 300.0)),
    "noncolinear_soc": dict(spin="noncolinear", spin_constants={"Cu": -0.016},
                            initial_spins=[[0.3, 0.0, 0.4]] * 3 + [[0.0, 0.0, 0.0]] * 3,
                            spin_orbit={"Cu": [0.0, 0.0, 0.01], "Cu2": [0.0, 0.0, 0.01]},
                            smearing=("fermi", 300.0)),
    "plus_u": dict(plus_u={"Cu": [("d", 0.15)]}, smearing=("fermi", 300.0)),
    "fixed_fermi": dict(fixed_fermi_level=-0.165, smearing=("fermi", 600.0)),
    "field": dict(electric_field=[0.01, 0.005, -0.003], smearing=("fermi", 300.0)),
    "shell_resolved": dict(smearing=("fermi", 300.0)),
}


@pytest.mark.parametrize("name", list(MOLECULE_OPTIONS))
def test_forces_are_derivatives_in_the_window(name):
    sk_ = ParamSet(SKF_TYPES, shell_resolved=name == "shell_resolved").skset(["Cu", "Cu2"])
    calc = DFTB(sk_, type_switching=SW_MOL, scc_tol=1e-12, **MOLECULE_OPTIONS[name])
    pos, lab = _cu6(), ["Cu"] * 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = calc.calculate(lab, pos)
    w = res.type_weights["Cu"]
    assert ((w > 0.0) & (w < 1.0)).sum() >= 4 and np.allclose(w + res.type_weights["Cu2"], 1.0)
    for c in (0, 4, 8, 13):
        def energy(x):
            p = pos.copy()
            p.flat[c] += x
            return calc.calculate(lab, p, forces=False).e_force_related
        assert -res.forces.flat[c] == pytest.approx(_fd(energy, 1e-4), abs=1e-8), (name, c)


def test_three_types(tmp_path):
    """Two windows (Cu2 -> Cu -> Cu3, Cu3 a copy of Cu2): the middle type's weight is
    s_1 - s_2; forces are derivatives with atoms in both windows."""
    sk3 = ParamSet(_three_types(tmp_path)).skset(["Cu", "Cu2", "Cu3"])
    sw = TypeSwitch("Cu", ("Cu2", "Cu", "Cu3"), centre=(2.6, 3.6), width=(1.0, 1.0),
                    cutoff=(4.6, 5.4))
    calc = DFTB(sk3, type_switching=[sw], scc_tol=1e-12, smearing=("fermi", 300.0))
    pos, lab = _cu6(), ["Cu"] * 6
    res = calc.calculate(lab, pos)
    w = np.array([res.type_weights[t] for t in sw.types]).T
    assert np.allclose(w.sum(1), 1.0) and ((w > 0) & (w < 1)).any(axis=0).all()
    for c in (1, 6, 11, 15):
        def energy(x):
            p = pos.copy()
            p.flat[c] += x
            return calc.calculate(lab, p, forces=False).free_energy
        assert -res.forces.flat[c] == pytest.approx(_fd(energy, 1e-4), abs=1e-8)


def test_periodic_forces_and_stress_in_the_window(sk):
    """k-points; the stress from 4-point strain differences (the steep switch makes the
    two-point h^2 error 1e-6 at h = 1e-5)."""
    pos, cell = _cu4_cell()
    pbc, lab = np.ones(3, bool), ["Cu"] * 4
    sw = TypeSwitch("Cu", ("Cu2", "Cu"), centre=5.6, width=1.0, cutoff=(4.4, 5.3))
    calc = DFTB(sk, type_switching=[sw], kpts=reduce_time_reversal(*monkhorst_pack((2, 2, 2))),
                scc_tol=1e-12, smearing=("fermi", 1000.0))
    res = calc.calculate(lab, pos, cell, pbc)
    w = res.type_weights["Cu"]
    assert ((w > 0.0) & (w < 1.0)).all()
    for c in (0, 5, 10):
        def energy(x):
            p = pos.copy()
            p.flat[c] += x
            return calc.calculate(lab, p, cell, pbc, forces=False).free_energy
        assert -res.forces.flat[c] == pytest.approx(_fd(energy, 1e-4), abs=1e-8)
    vol = abs(np.linalg.det(cell))
    for a, b in ((0, 0), (2, 2), (0, 1), (1, 2)):
        def strained(x):
            eps = np.zeros((3, 3))
            eps[a, b] += 0.5 * x
            eps[b, a] += 0.5 * x
            m = np.eye(3) + eps
            return calc.calculate(lab, pos @ m.T, cell @ m.T, pbc, forces=False).free_energy
        assert -res.stress[a, b] * vol == pytest.approx(_fd(strained, 1e-5), abs=5e-8)


def test_energy_is_continuous_across_a_window(sk):
    """An atom capping a face of a Cu4 tetrahedron is pulled out along the face normal: its CN
    falls from 3 to 0 through the window [0.5, 2.5] (its three bonds cross the counting window).
    Along the path the slope of the energy is the force (4-point differences at every point,
    in and outside the window), Simpson's rule of the forces gives the energy differences (no
    jump), and outside the window the result is bitwise that of the fixed labels."""
    base = 1.0013 * np.array([[0.0, 0.0, 0.0], [4.8, 0.0, 0.0], [2.4, 4.157, 0.0],
                              [2.4, 1.386, 3.919]])
    centre = base[1:].mean(0)
    normal = (centre - base[0]) / np.linalg.norm(centre - base[0])
    h0 = np.sqrt(4.79 ** 2 - np.linalg.norm(base[1] - centre) ** 2)
    sw = TypeSwitch("Cu", ("Cu2", "Cu"), centre=1.5, width=2.0, cutoff=(5.0, 6.0))
    calc = DFTB(sk, type_switching=[sw], scc_tol=1e-12, smearing=("fermi", 300.0))
    lab = ["Cu"] * 5

    def geometry(s):
        return np.vstack([base, centre + (h0 + s) * normal])

    def energy(s):
        return calc.calculate(lab, geometry(s), forces=False).free_energy

    def off_node(s):            # no distance within 1e-3 Bohr of an SK grid node (0.02 Bohr)
        while True:
            p = geometry(s)
            d = np.linalg.norm(p[:, None] - p[None], axis=2)[np.triu_indices(5, 1)] / 0.02
            if np.abs(d - np.round(d)).min() > 0.05:
                return s
            s += 1.3e-3

    xs = np.array([off_node(s) for s in np.linspace(0.0, 1.7, 18)])
    res = [calc.calculate(lab, geometry(s)) for s in xs]
    cn4 = [sw.coordination_numbers(lab, geometry(s))[4] for s in xs]
    w4 = np.array([r.type_weights["Cu"][4] for r in res])
    assert cn4[0] == 3.0 and cn4[-1] == 0.0 and ((w4 > 0.0) & (w4 < 1.0)).sum() >= 3
    slope = np.array([-(r.forces[4] @ normal) for r in res])
    for s, sl in zip(xs, slope):
        assert sl == pytest.approx(_fd(lambda x: energy(s + x), 1e-4), abs=1e-8), s
    # at the window's edges (CN of the atom 2.5 and 0.5) energy and force are continuous
    for edge in (2.5, 0.5):
        lo, hi = xs[0], xs[-1]
        for _ in range(60):                       # bisection: CN(s) falls monotonically
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if sw.coordination_numbers(lab, geometry(m))[4] > edge else (lo, m)
        below, above = (calc.calculate(lab, geometry(s)) for s in (lo - 1e-7, hi + 1e-7))
        step = -(0.5 * (above.forces[4] + below.forces[4]) @ normal) * (hi - lo + 2e-7)
        assert abs(above.free_energy - below.free_energy - step) < 1e-12
        assert np.abs(above.forces - below.forces).max() < 1e-6
    for s, r in ((xs[0], res[0]), (xs[-1], res[-1])):
        w = r.type_weights["Cu"]
        assert set(np.unique(w)) <= {0.0, 1.0}
        labels = ["Cu" if x == 1.0 else "Cu2" for x in w]
        fixed = DFTB(sk, scc_tol=1e-12, smearing=("fermi", 300.0)).calculate(labels, geometry(s))
        assert fixed.energy == r.energy and np.array_equal(fixed.forces, r.forces)


def test_md_conserves_energy_through_the_window():
    """NVE (velocity Verlet, 60 fs) of a hot Cu6 whose atoms enter and leave the switching window:
    the conserved F + E_kin fluctuates like Verlet's dt^2 error (forces are the free energy's
    gradient) and about as much as without switching."""
    from ase.md.verlet import VelocityVerlet

    from pydftb.ase_calc import PyDFTB
    sw = TypeSwitch("Cu", ("Cu2", "Cu"), centre=3.0, width=2.0, cutoff=(4.2, 6.0))

    def run(dt, switch=True):
        a = Atoms("Cu6", positions=_cu6() * BOHR)
        sigma = np.sqrt(900.0 * units.kB / a.get_masses())[:, None]     # Maxwell-Boltzmann
        a.set_velocities(np.random.default_rng(5).normal(size=(6, 3)) * sigma)
        a.calc = PyDFTB(SKF_TYPES, smearing=("fermi", 300.0), scc_tol=1e-10, solver="eigh",
                        reuse_charges=True, type_switching=[sw] if switch else None)
        etot, frac = [], []
        dyn = VelocityVerlet(a, timestep=dt * units.fs)

        def log():
            etot.append(a.get_potential_energy(force_consistent=True) + a.get_kinetic_energy())
            if switch:
                w = a.calc.raw.type_weights["Cu"]
                frac.append((w > 0.0) & (w < 1.0))
        dyn.attach(log, interval=1)
        dyn.run(int(round(60.0 / dt)))
        return np.ptp(etot), np.array(frac)

    ptp1, frac = run(1.0)
    ptp_half, _ = run(0.5)
    ptp_plain, _ = run(1.0, switch=False)
    assert np.sum(frac[1:] != frac[:-1]) >= 4            # atoms enter and leave the window
    assert ptp_half < ptp1 / 2.5                          # Verlet: error ~ dt^2
    assert ptp1 < 4.0 * ptp_plain and ptp1 < 1e-2         # eV


# ---- the ASE calculator, band structures, charge files -----------------------------------------
def test_ase_calculator(sk):
    from pydftb.ase_calc import PyDFTB
    pos = _cu6()
    a = Atoms("Cu6", positions=pos * BOHR)
    a.calc = PyDFTB(SKF_TYPES, smearing=("fermi", 300.0), scc_tol=1e-10, solver="eigh",
                    type_switching=[{"element": "Cu", "types": ("Cu2", "Cu"), "centre": 3.2,
                                     "width": 2.0, "cutoff": (4.6, 5.4)}])
    ref = DFTB(sk, type_switching=SW_MOL, scc_tol=1e-10, smearing=("fermi", 300.0)).calculate(
        ["Cu"] * 6, pos)
    assert a.get_potential_energy() == pytest.approx(ref.energy * units.Hartree, abs=1e-9)
    assert np.allclose(a.get_forces(), ref.forces * units.Hartree / units.Bohr, atol=1e-8)
    assert np.allclose(a.calc.raw.type_weights["Cu"], ref.type_weights["Cu"], atol=0)
    assert np.allclose(a.get_charges(), 11.0 - ref.charges, atol=1e-10)


def test_types_without_the_element_label(tmp_path):
    """Types "Cu_a"/"Cu_b" (copies of Cu/Cu2): the atoms are still given as "Cu", the engine
    uses the reference type Cu_a, and the results are bitwise those of Cu/Cu2."""
    from pydftb.ase_calc import PyDFTB
    d = str(tmp_path / "ab")
    write_typed_copies(SKF_TYPES, d, {"Cu_a": "Cu", "Cu_b": "Cu2"}, template="{a}-{b}.skf.gz")
    sw_ab = TypeSwitch("Cu", ("Cu_b", "Cu_a"), centre=3.2, width=2.0, cutoff=(4.6, 5.4))
    assert sw_ab.reference == "Cu_b"
    out = []
    for skdir, sw in ((d, sw_ab), (SKF_TYPES, SW_MOL)):
        a = Atoms("Cu6", positions=_cu6() * BOHR)
        a.calc = PyDFTB(skdir, smearing=("fermi", 300.0), scc_tol=1e-10, solver="eigh",
                        type_switching=sw)
        out.append((a.get_potential_energy(), a.get_forces(), a.get_charges()))
    assert out[0][0] == out[1][0] and np.array_equal(out[0][1], out[1][1])
    assert np.array_equal(out[0][2], out[1][2])


def test_band_structure_and_charge_file(sk, tmp_path):
    """Band structures fold the mixed blocks (the SCC k-points give the SCC eigenvalues); a
    charge file written and read back gives the same input populations."""
    pos, cell = _cu4_cell()
    pbc, lab = np.ones(3, bool), ["Cu"] * 4
    sw = TypeSwitch("Cu", ("Cu2", "Cu"), centre=5.6, width=1.0, cutoff=(4.4, 5.3))
    k = reduce_time_reversal(*monkhorst_pack((2, 2, 2)))
    calc = DFTB(sk, type_switching=[sw], kpts=k, scc_tol=1e-10, smearing=("fermi", 1000.0))
    res = calc.calculate(lab, pos, cell, pbc, forces=False)
    st = eigenstates(calc, lab, pos, cell, pbc, res.q_in, scc_fermi=res.fermi)
    assert np.allclose(st.eigenvalues, np.array(res.eigenvalues), rtol=0, atol=1e-12)
    from pydftb.dftbplus_io import write_charges
    path = str(tmp_path / "charges.bin")
    write_charges(path, calc.charge_data(lab, res))
    assert np.allclose(calc.populations_from_file(lab, path), res.q_in, rtol=0, atol=1e-12)


def test_discrete_limit_with_spin_and_dftb3(sk):
    """Weights 0/1 with DFTB3 and collinear spin: bitwise the fixed labels."""
    case = refgen.CASES_BY_NAME["cu13_types"]
    inp = refgen._inputs(case)
    opts = dict(third_order="full", hubbard_derivs=UD, spin=True, unpaired_electrons=1.0,
                spin_constants={"Cu": W_SPIN["Cu"], "Cu2": W_SPIN["Cu"]}, scc_tol=1e-10)
    a = refgen.run_pydftb(case, inp, **opts)
    b = refgen.run_pydftb(case, inp, switching=True, **opts)
    assert a.energy == b.energy and np.array_equal(a.forces, b.forces)
    assert np.array_equal(a.magmoms, b.magmoms) and a.n_scc == b.n_scc


# ---- refusals ----------------------------------------------------------------------------------
def _modified_set(tmp_path, edit, name="mod"):
    """A plain copy of cu_types whose files ``edit(file name, lines)`` may change."""
    d = tmp_path / name
    d.mkdir()
    for f in os.listdir(SKF_TYPES):
        with gzip.open(os.path.join(SKF_TYPES, f), "rt") as fh:
            lines = fh.read().splitlines(keepends=True)
        lines = edit(f[:-3], lines)
        (d / f[:-3]).write_text("".join(lines))
    return str(d)


def test_refusals(sk, tmp_path):
    with pytest.raises(ValueError, match="as 'Cu', not as the type label 'Cu2'"):
        DFTB(sk, type_switching=SW_MOL).calculate(["Cu", "Cu2"], [[0, 0, 0], [0, 0, 4.5]])
    with pytest.raises(ValueError, match="has no type"):
        DFTB(ParamSet(SKF_TYPES).skset(["Cu"]), type_switching=SW_MOL)
    with pytest.raises(ValueError, match="hubbard_derivs of Cu"):
        DFTB(sk, third_order="full", hubbard_derivs={"Cu": -0.05, "Cu2": -0.06},
             type_switching=SW_MOL)
    with pytest.raises(ValueError, match="hubbard of"):
        DFTB(sk, hubbard={"Cu": 0.3}, type_switching=SW_MOL)
    with pytest.raises(ValueError, match="Casida"):
        DFTB(sk, casida={"n_excitations": 2}, type_switching=SW_MOL)
    with pytest.raises(ValueError, match="analysis outputs"):
        DFTB(sk, analysis=True, type_switching=SW_MOL)
    with pytest.raises(ValueError, match="XL-BOMD does not support type switching"):
        DFTB(sk, type_switching=SW_MOL, force_evaluation="dynamics").calculate(
            ["Cu"] * 6, _cu6(), scc_control=SCCControl(), xl=True)
    # unequal on-site energies
    def onsite(name, lines):
        if name == "Cu2-Cu2.skf":
            vals = lines[1].split()
            vals[0] = "-0.185000"
            lines[1] = " ".join(vals) + "\n"
        return lines
    sk_bad = ParamSet(_modified_set(tmp_path, onsite)).skset(["Cu", "Cu2"])
    with pytest.raises(ValueError, match="onsite of Cu2 and Cu"):
        DFTB(sk_bad, type_switching=SW_MOL)
    # a hetero-label file written as homonuclear (with the on-site line)
    d = tmp_path / "plain"
    _modified_set(tmp_path, lambda n, x: x, "plain")
    shutil.copy(d / "Cu-Cu.skf", d / "Cu-Cu2.skf")
    with pytest.raises(ValueError, match="on-site line"):
        DFTB(ParamSet(str(d)).skset(["Cu", "Cu2"]), type_switching=SW_MOL)


def test_numpy_only(sk):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="milestone M5a"):
        DFTB(sk, type_switching=SW_MOL, backend="torch")
    from pydftb.autodiff import api
    with pytest.raises(NotImplementedError, match="type_switching"):
        api.evaluate(DFTB(sk, type_switching=SW_MOL), ["Cu"] * 6, _cu6())
