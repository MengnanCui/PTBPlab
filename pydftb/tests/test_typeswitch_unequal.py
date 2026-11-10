"""Per-type on-site energies and Hubbard U, mixed per atom (A2 channel E1,
:class:`pydftb.typeswitch.OnsiteSwitch`).

The cu_types set (Cu and Cu2, different orbitals) with the on-site line of Cu2-Cu2 changed (Ed,
Es, Ud, Us), so the two types differ in on-site energies and U.  Forces and stress are the
free energy's derivatives inside the switching window, dF/d(shifts) agrees with finite
differences, the Hubbard derivative of the Coulomb kernel (``Coulomb.hubbard_gradient``, every
branch of the short-range function) with finite differences, equal types reproduce M5a, the
unsupported combinations are refused, and the model round-trips through JSON
(:mod:`pydftb.envmodel`).
"""
import gzip
import json
import os

import numpy as np
import pytest
from ase.build import bulk
from ase.cluster import Octahedron

from pydftb import envmodel
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.coulomb import SAME_TAU, Coulomb, _gamma0, _gamma0_dta, _short, _short_dta
from pydftb.kpoints import monkhorst_pack, reduce_time_reversal
from pydftb.params import ParamSet
from pydftb.typeswitch import OnsiteSwitch, TypeSwitch
from pydftb.validation.cases import SKF_SETS

SKF_TYPES = SKF_SETS["cu_types"]
SHIFTS = dict(onsite_shift={"Cu2": {"s": 0.004, "d": -0.003}, "Cu": {"d": 0.002}},
              hubbard_shift={"Cu2": {"s": 0.01}})


def _fd(energy, h):
    """dE/dx at 0 by 4-point central differences."""
    return (8.0 * (energy(h) - energy(-h)) - (energy(2 * h) - energy(-2 * h))) / (12.0 * h)


@pytest.fixture(scope="module")
def unequal_dir(tmp_path_factory):
    """cu_types with Cu2's on-site energies and U changed (Ed, Es, Ud, Us)."""
    d = tmp_path_factory.mktemp("cu_unequal")
    for f in os.listdir(SKF_TYPES):
        with gzip.open(os.path.join(SKF_TYPES, f), "rt") as fh:
            lines = fh.read().splitlines(keepends=True)
        if f.startswith("Cu2-Cu2"):
            v = lines[1].split()             # Ed Ep Es SPE Ud Up Us fd fp fs
            v[0] = repr(float(v[0]) - 0.012)
            v[2] = repr(float(v[2]) + 0.015)
            v[4] = repr(float(v[4]) + 0.03)
            v[6] = repr(float(v[6]) + 0.025)
            lines[1] = " ".join(v) + "\n"
        (d / f[:-3]).write_text("".join(lines))
    return str(d)


def _cu6():
    a = Octahedron("Cu", 2, cutoff=0, latticeconstant=3.61)
    a.rattle(0.12, seed=21)
    return a.positions / BOHR


def _cu4_cell():
    a = bulk("Cu", "fcc", a=3.61, cubic=True)
    a.rattle(0.15, seed=22)
    a.set_cell(a.cell.array @ np.array([[1.02, 0.01, 0], [0, 0.99, 0.015], [0.0, 0.0, 1.0]]),
               scale_atoms=True)
    return a.positions / BOHR, a.cell.array / BOHR


# ---- the Coulomb kernel's Hubbard derivative ----------------------------------------------------
def test_short_range_tau_derivatives():
    r = np.random.default_rng(0).uniform(1.0, 9.0, 40)
    one = np.ones(40)
    for ta, tb in ((1.3 * one, 1.3 * one + 0.2 * SAME_TAU), (1.3 * one, 1.35 * one),
                   (1.3 * one, 2.4 * one), (2.9 * one, 1.1 * one)):      # same, near, general
        h = 1e-6
        fd = (_short(r, ta + h, tb)[0] - _short(r, ta - h, tb)[0]) / (2 * h)
        assert np.allclose(_short_dta(r, ta, tb), fd, rtol=0, atol=1e-9)
    for ta, tb in ((1.3, 1.3), (1.3, 2.1), (3.0, 0.9)):
        h = 1e-6
        fd = (_gamma0(np.array(ta + h), np.array(tb)) - _gamma0(np.array(ta - h), np.array(tb))) / (2 * h)
        assert float(_gamma0_dta(ta, tb)) == pytest.approx(float(fd), abs=1e-9)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("shells", [False, True])
def test_hubbard_gradient(periodic, shells):
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 6, (5, 3))
    site_atom = np.repeat(np.arange(5), 2) if shells else np.arange(5)
    dq = rng.normal(size=len(site_atom)) * 0.3
    u = rng.uniform(0.3, 0.6, len(site_atom))
    cell, pbc = (np.eye(3) * 7.5, np.ones(3, bool)) if periodic else (None, None)
    cut = np.full_like(u, 0.95 * u.min())            # the cut-offs stay fixed while U moves
    g = Coulomb(pos, site_atom, u, cell, pbc, cutoff_hubbard=cut).hubbard_gradient(dq)
    for s in range(len(u)):
        def energy(h):
            v = u.copy()
            v[s] += h
            return Coulomb(pos, site_atom, v, cell, pbc, cutoff_hubbard=cut).energy_shift(dq)[0]
        assert g[s] == pytest.approx(_fd(energy, 1e-4), abs=1e-9)
    # the default cut-offs: gamma bitwise as before
    a = Coulomb(pos, site_atom, u, cell, pbc)
    b = Coulomb(pos, site_atom, u, cell, pbc, cutoff_hubbard=u)
    assert np.array_equal(a.gamma, b.gamma)


# ---- forces, stress and the shifts' gradient -----------------------------------------------------
@pytest.mark.parametrize("shell_resolved", [False, True])
def test_molecule_forces_and_shift_gradient(unequal_dir, shell_resolved):
    sk = ParamSet(unequal_dir, shell_resolved=shell_resolved).skset(["Cu", "Cu2"])
    shifts = dict(SHIFTS)
    if shell_resolved:
        shifts["hubbard_shift"] = {"Cu2": {"s": 0.01, "d": -0.02}, "Cu": {"p": 0.005}}
    sw = OnsiteSwitch("Cu", ("Cu2", "Cu"), centre=3.2, width=2.0, cutoff=(4.6, 5.4), **shifts)
    calc = DFTB(sk, type_switching=[sw], scc_tol=1e-12, smearing=("fermi", 300.0))
    pos, lab = _cu6(), ["Cu"] * 6
    res = calc.calculate(lab, pos)
    w = res.type_weights["Cu"]
    assert ((w > 0.0) & (w < 1.0)).sum() >= 4
    for c in (0, 8, 13):
        def energy(x):
            p = pos.copy()
            p.flat[c] += x
            return calc.calculate(lab, p, forces=False).free_energy
        assert -res.forces.flat[c] == pytest.approx(_fd(energy, 1e-4), abs=2e-9)
    x0 = sw.vector()
    g = sw.gradient_from(res)
    assert g is not None and len(g) == len(x0)
    for k in range(len(x0)):
        def energy(h):
            x = x0.copy()
            x[k] += h
            c2 = DFTB(sk, type_switching=[sw.with_vector(x)], scc_tol=1e-12,
                      smearing=("fermi", 300.0))
            return c2.calculate(lab, pos, forces=False).free_energy
        assert g[k] == pytest.approx(_fd(energy, 1e-4), abs=2e-9), sw.parameter_names()[k]


def test_periodic_forces_stress_and_shift_gradient(unequal_dir):
    sk = ParamSet(unequal_dir).skset(["Cu", "Cu2"])
    pos, cell = _cu4_cell()
    pbc, lab = np.ones(3, bool), ["Cu"] * 4
    k = reduce_time_reversal(*monkhorst_pack((2, 2, 2)))
    sw = OnsiteSwitch("Cu", ("Cu2", "Cu"), centre=5.6, width=1.0, cutoff=(4.4, 5.3),
                      onsite_shift={"Cu2": {"s": 0.004}}, hubbard_shift={"Cu2": {"s": 0.01}})
    calc = DFTB(sk, type_switching=[sw], kpts=k, scc_tol=1e-12, smearing=("fermi", 1000.0))
    res = calc.calculate(lab, pos, cell, pbc)
    w = res.type_weights["Cu"]
    assert ((w > 0.0) & (w < 1.0)).all()
    for c in (0, 10):
        def energy(x):
            p = pos.copy()
            p.flat[c] += x
            return calc.calculate(lab, p, cell, pbc, forces=False).free_energy
        assert -res.forces.flat[c] == pytest.approx(_fd(energy, 1e-4), abs=2e-9)
    vol = abs(np.linalg.det(cell))
    for a, b in ((0, 0), (0, 1)):
        def strained(x):
            eps = np.zeros((3, 3))
            eps[a, b] += 0.5 * x
            eps[b, a] += 0.5 * x
            m = np.eye(3) + eps
            return calc.calculate(lab, pos @ m.T, cell @ m.T, pbc, forces=False).free_energy
        assert -res.stress[a, b] * vol == pytest.approx(_fd(strained, 1e-5), abs=5e-8)
    x0 = sw.vector()
    g = sw.gradient_from(res)
    for i in range(len(x0)):
        def energy(h):
            x = x0.copy()
            x[i] += h
            c2 = DFTB(sk, type_switching=[sw.with_vector(x)], kpts=k, scc_tol=1e-12,
                      smearing=("fermi", 1000.0))
            return c2.calculate(lab, pos, cell, pbc, forces=False).free_energy
        assert g[i] == pytest.approx(_fd(energy, 1e-4), abs=2e-9)


def test_equal_types_reproduce_m5a():
    """With the SKF's equal on-site energies and U and zero shifts, E1 is M5a to rounding."""
    sk = ParamSet(SKF_TYPES).skset(["Cu", "Cu2"])
    spec = dict(element="Cu", types=("Cu2", "Cu"), centre=3.2, width=2.0, cutoff=(4.6, 5.4))
    pos, lab = _cu6(), ["Cu"] * 6
    a = DFTB(sk, type_switching=[TypeSwitch(**spec)], scc_tol=1e-12,
             smearing=("fermi", 300.0)).calculate(lab, pos)
    b = DFTB(sk, type_switching=[OnsiteSwitch(**spec, onsite_shift={"Cu": {"s": 0.0}})],
             scc_tol=1e-12, smearing=("fermi", 300.0)).calculate(lab, pos)
    assert b.free_energy == pytest.approx(a.free_energy, abs=1e-11)
    assert np.allclose(b.forces, a.forces, rtol=0, atol=1e-9)
    assert a.switching is None and set(b.switching) == {"Cu"}


def test_unequal_types_need_the_opt_in(unequal_dir):
    sk = ParamSet(unequal_dir).skset(["Cu", "Cu2"])
    with pytest.raises(ValueError, match="onsite of Cu2 and Cu"):
        DFTB(sk, type_switching=TypeSwitch("Cu", ("Cu2", "Cu"), 3.2, 2.0))
    DFTB(sk, type_switching=OnsiteSwitch("Cu", ("Cu2", "Cu"), 3.2, 2.0))      # allowed


def test_refusals(unequal_dir):
    sk = ParamSet(unequal_dir).skset(["Cu", "Cu2"])
    sw = OnsiteSwitch("Cu", ("Cu2", "Cu"), 3.2, 2.0)
    for opts, msg in ((dict(third_order="full", hubbard_derivs={"Cu": -0.05, "Cu2": -0.05}),
                       "third order"),
                      (dict(plus_u={"Cu": [("d", 0.15)]}), "DFTB\\+U"),
                      (dict(h_damping=4.0), "damping")):
        with pytest.raises(ValueError, match=msg):
            DFTB(sk, type_switching=sw, **opts)
    with pytest.raises(ValueError, match="not one of the types"):
        OnsiteSwitch("Cu", ("Cu2", "Cu"), 3.2, 2.0, onsite_shift={"Cu3": {"s": 0.1}})
    with pytest.raises(ValueError, match="shells are letters"):
        OnsiteSwitch("Cu", ("Cu2", "Cu"), 3.2, 2.0, onsite_shift={"Cu2": {"x": 0.1}})
    with pytest.raises(ValueError, match="one U per atom"):          # atom-resolved SCC
        DFTB(sk, type_switching=OnsiteSwitch("Cu", ("Cu2", "Cu"), 3.2, 2.0,
                                             hubbard_shift={"Cu2": {"s": 0.1, "d": 0.1}})
             ).calculate(["Cu"] * 6, _cu6())


def test_model_protocol_and_json():
    sw = OnsiteSwitch("Cu", ("Cu2", "Cu"), 3.2, 2.0, cutoff=(4.6, 5.4), **SHIFTS)
    assert sw.parameter_names() == ["Cu2.onsite.s", "Cu2.onsite.d", "Cu.onsite.d",
                                    "Cu2.hubbard.s"]
    assert np.allclose(sw.vector(), [0.004, -0.003, 0.002, 0.01])
    x = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.array_equal(sw.with_vector(x).vector(), x)
    back = envmodel.model_from_dict(json.loads(json.dumps(sw.to_dict())))
    assert back == sw and isinstance(back, OnsiteSwitch)
    assert envmodel.engine_kwargs([sw]) == {"type_switching": [sw]}
    assert TypeSwitch.make(sw.to_dict()) == sw
