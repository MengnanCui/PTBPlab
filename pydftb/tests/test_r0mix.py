"""Continuous r0 per atom by SK mixing between r0 nodes (A2 channel E2, :mod:`pydftb.r0mix`).

A synthetic node set: five labels Cu1..Cu5, copies of cu_types' Cu and Cu2 whose H and S tables
are scaled by s_t s_u (s = 1, 1, 0.97, 1.03, 1.01), so neighbouring nodes differ like tables at
different radii.  Checked: the cubic B-spline weights and the clamp, forces and stress as
derivatives of the free energy, dF/dtheta (``DFTBResult.switching``) against finite differences
of the coefficients, E1 shifts on the nodes, band structures, the JSON round trip and the
envmodel protocol, and the refusals.
"""
import json
import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest
from ase.build import bulk
from ase.cluster import Octahedron

from pydftb import envmodel
from pydftb.bands import eigenstates
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.descriptors import CoordinationNumber, Stack
from pydftb.kpoints import monkhorst_pack, reduce_time_reversal
from pydftb.params import ParamSet
from pydftb.r0mix import (R0Switch, bspline3, labels_max_l, node_radii, r0mix_from_dict,
                          smooth_clamp)
from pydftb.regressors import Linear
from pydftb.skf import _expand_fortran, write_typed_copies
from pydftb.typeswitch import check_switches
from pydftb.validation.cases import SKF_SETS

SKF_TYPES = SKF_SETS["cu_types"]
BASE = {"Cu1": "Cu", "Cu2": "Cu2", "Cu3": "Cu", "Cu4": "Cu2", "Cu5": "Cu"}
SCALE = {"Cu1": 1.0, "Cu2": 1.0, "Cu3": 0.97, "Cu4": 1.03, "Cu5": 1.01}
NODES = tuple((4.0 + 0.2 * k, 8.0 + 0.1 * k) for k in range(5))


def _fd(energy, h):
    return (8.0 * (energy(h) - energy(-h)) - (energy(2 * h) - energy(-2 * h))) / (12.0 * h)


def _write_nodes(d, onsite=None):
    """The node set into ``d``; ``onsite`` {label: (dEd, dEs, dUd, dUs)} edits on-site lines."""
    write_typed_copies(SKF_TYPES, d, BASE, template="{a}-{b}.skf.gz")
    for t in BASE:
        for u in BASE:
            f = SCALE[t] * SCALE[u]
            path = os.path.join(d, f"{t}-{u}.skf")
            lines = open(path).read().splitlines(keepends=True)
            first = 1 if lines[0].lstrip().startswith("@") else 0
            if t == u and onsite and t in onsite:
                v = lines[first + 1].split()            # Ed Ep Es SPE Ud Up Us fd fp fs
                for k, dv in zip((0, 2, 4, 6), onsite[t]):
                    v[k] = repr(float(v[k]) + dv)
                lines[first + 1] = " ".join(v) + "\n"
            if f != 1.0:
                n_grid = int(_expand_fortran(lines[first])[1])
                start = first + 1 + (1 if t == u else 0) + 1
                for k in range(start, start + n_grid - 1):
                    lines[k] = " ".join(f"{x * f:.15e}" for x in _expand_fortran(lines[k])) + "\n"
            open(path, "w").writelines(lines)
    return d


@pytest.fixture(scope="module")
def nodes_dir(tmp_path_factory):
    return _write_nodes(str(tmp_path_factory.mktemp("cu_nodes")))


@pytest.fixture(scope="module")
def sk(nodes_dir):
    return ParamSet(nodes_dir).skset(list(BASE))


def _switch(**kw):
    desc = kw.pop("descriptor", CoordinationNumber("Cu", window=(0.8, 1.4)))
    reg = kw.pop("regressor", Linear(desc.size, 1, centre=(3.0,) * desc.size,
                                     scale=(1.0,) * desc.size))
    kw.setdefault("theta", (0.1,) + (0.35,) * desc.size)
    return R0Switch("Cu", tuple(BASE), NODES, descriptor=desc, regressor=reg, **kw)


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


# ---- the weights ------------------------------------------------------------------------------------
def test_bspline_weights_and_clamp():
    x = np.linspace(-2.5, 2.5, 1001)
    b, db = bspline3(x)
    assert (b >= 0.0).all() and b[np.abs(x) >= 2.0].max() == 0.0
    h = 1e-6
    assert np.allclose(db, (bspline3(x + h)[0] - bspline3(x - h)[0]) / (2 * h), atol=1e-8)
    d2 = (bspline3(x + h)[1] - bspline3(x - h)[1]) / (2 * h)     # C2: no jump in B''
    assert np.abs(np.diff(d2)).max() < 0.05
    sw = _switch()
    u = np.linspace(1.0, 3.0, 201)[:, None]
    w, dw = sw.spline_weights(u)
    assert np.allclose(w.sum(1), 1.0, atol=1e-14) and (w >= 0.0).all()
    assert np.allclose(w @ np.array(NODES), np.array([4.0, 8.0]) + u * np.array([0.2, 0.1]),
                       atol=1e-12)                     # linear nodes: the mean r0 is r0(u)
    assert np.allclose(dw[:, :, 0], (sw.spline_weights(u + h)[0] - sw.spline_weights(u - h)[0])
                       / (2 * h), atol=1e-8)
    y, dy = smooth_clamp(np.linspace(-1, 5, 601), 1.0, 3.0, 0.5)
    assert y.min() == 1.0 and y.max() == 3.0 and (np.diff(y) >= 0).all()
    xs = np.linspace(-1, 5, 601)
    assert np.array_equal(y[(xs >= 1.5) & (xs <= 2.5)], xs[(xs >= 1.5) & (xs <= 2.5)])
    assert np.allclose(dy, (smooth_clamp(xs + h, 1, 3, 0.5)[0] - smooth_clamp(xs - h, 1, 3, 0.5)[0])
                       / (2 * h), atol=1e-8)


def test_grid_weights():
    labels = tuple(f"Cu{a}{b}" for a in range(1, 5) for b in range(1, 5))
    nodes = tuple((4.0 + 0.2 * a, 8.0 + 0.4 * b) for a in range(4) for b in range(4))
    sw = R0Switch("Cu", labels, nodes, grid=(4, 4))
    assert sw.regressor.n_out == 2 and sw.u_ref == (1.5, 1.5)
    u = np.random.default_rng(2).uniform(1.0, 2.0, (50, 2))
    w, dw = sw.spline_weights(u)
    assert np.allclose(w.sum(1), 1.0, atol=1e-14) and (w >= 0).all()
    assert np.allclose(w @ np.array(nodes), np.column_stack([4.0 + 0.2 * u[:, 0], 8.0 + 0.4 * u[:, 1]]),
                       atol=1e-12)
    h = 1e-6
    for a in range(2):
        e = np.zeros(2)
        e[a] = h
        assert np.allclose(dw[:, :, a], (sw.spline_weights(u + e)[0] - sw.spline_weights(u - e)[0])
                           / (2 * h), atol=1e-8)


def test_construction():
    sw = R0Switch.around("Li", r0=(4.2, 9.1), spacing=(0.3, 0.6), n=7)
    assert sw.types == tuple(f"Li{k}" for k in range(1, 8)) and sw.nodes[3] == (4.2, 9.1)
    assert sw.u_ref == (3.0,) and sw.reference == "Li1" and not sw.mixes_onsite
    g = R0Switch.around("O", r0=(3.0, 6.0), spacing=0.2, n=5, grid=True)
    assert g.grid == (5, 5) and g.types[0] == "O11" and g.nodes[12] == (3.0, 6.0)
    assert node_radii([sw], {"O": ("O", 2.9, 6.0)})["Li4"] == ("Li", 4.2, 9.1)
    assert labels_max_l([sw], {"Li": "s"})["Li7"] == "s"
    for bad, msg in ((dict(types=("Cu1", "Cu2", "Cu3")), "one \\(r0_w, r0_d\\) per node"),
                     (dict(types=("Cu1", "Cu2", "O3", "Cu4", "Cu5")), "is not Cu"),
                     (dict(types=("Cu", "Cu2", "Cu3", "Cu4", "Cu5")), "carry the label"),
                     (dict(margin=1.5), "n >= 3 \\+ 2 margin"),
                     (dict(regressor=Linear(2, 1)), "maps 2 -> 1")):
        spec = dict(element="Cu", types=tuple(BASE), nodes=NODES) | bad
        with pytest.raises(ValueError, match=msg):
            R0Switch(**spec)
    assert check_switches([sw])[0] is sw


# ---- forces, stress and the coefficients' gradient ---------------------------------------------
@pytest.mark.parametrize("stacked", [False, True])
def test_molecule_forces_and_theta_gradient(sk, stacked):
    kw = {}
    if stacked:
        desc = Stack((CoordinationNumber("Cu", window=(0.8, 1.4)),
                      CoordinationNumber("Cu", window=(1.0, 1.6))))
        kw = dict(descriptor=desc, regressor=Linear(2, 1, centre=(3.0, 4.0)),
                  theta=(0.2, 0.25, -0.1))
    sw = _switch(**kw)
    calc = DFTB(sk, type_switching=[sw], scc_tol=1e-12, smearing=("fermi", 300.0))
    pos, lab = _cu6(), ["Cu"] * 6
    res = calc.calculate(lab, pos)
    terms = res.switching["Cu"]
    assert (terms.u > 1.0).all() and (terms.u < 3.0).all()
    assert np.allclose(terms.weights.sum(1), 1.0) and terms.r0.shape == (6, 2)
    for c in (0, 4, 13):
        def energy(x):
            p = pos.copy()
            p.flat[c] += x
            return calc.calculate(lab, p, forces=False).free_energy
        assert -res.forces.flat[c] == pytest.approx(_fd(energy, 1e-4), abs=2e-9)
    th = np.array(sw.theta)
    g = sw.gradient_from(res)
    assert g is not None and len(g) == len(th)
    for k in range(len(th)):
        def energy(x):
            t = th.copy()
            t[k] += x
            c2 = DFTB(sk, type_switching=[sw.with_vector(t)], scc_tol=1e-12,
                      smearing=("fermi", 300.0))
            return c2.calculate(lab, pos, forces=False).free_energy
        assert g[k] == pytest.approx(_fd(energy, 1e-4), abs=2e-9)
    # energy-only calculations carry the same gradient
    assert np.allclose(sw.gradient_from(calc.calculate(lab, pos, forces=False)), g, atol=1e-10)


def test_periodic_forces_stress_and_bands(sk):
    pos, cell = _cu4_cell()
    pbc, lab = np.ones(3, bool), ["Cu"] * 4
    k = reduce_time_reversal(*monkhorst_pack((2, 2, 2)))
    sw = _switch(theta=(0.1, -0.3))
    calc = DFTB(sk, type_switching=[sw], kpts=k, scc_tol=1e-12, smearing=("fermi", 1000.0))
    res = calc.calculate(lab, pos, cell, pbc)
    for c in (0, 10):
        def energy(x):
            p = pos.copy()
            p.flat[c] += x
            return calc.calculate(lab, p, cell, pbc, forces=False).free_energy
        assert -res.forces.flat[c] == pytest.approx(_fd(energy, 1e-4), abs=2e-9)
    vol = abs(np.linalg.det(cell))
    for a, b in ((0, 0), (1, 2)):
        def strained(x):
            eps = np.zeros((3, 3))
            eps[a, b] += 0.5 * x
            eps[b, a] += 0.5 * x
            m = np.eye(3) + eps
            return calc.calculate(lab, pos @ m.T, cell @ m.T, pbc, forces=False).free_energy
        assert -res.stress[a, b] * vol == pytest.approx(_fd(strained, 1e-5), abs=5e-8)
    st = eigenstates(calc, lab, pos, cell, pbc, res.q_in, scc_fermi=res.fermi)
    assert np.allclose(st.eigenvalues, np.array(res.eigenvalues), rtol=0, atol=1e-10)


def test_onsite_shifts_on_the_nodes(tmp_path):
    """Nodes with different on-site energies and U (mix_onsite) and fitted shifts: forces and
    dF/d(theta, shifts)."""
    d = _write_nodes(str(tmp_path / "e1"), onsite={"Cu4": (-0.01, 0.012, 0.02, 0.015),
                                                   "Cu2": (0.004, 0.0, 0.0, -0.01)})
    sk = ParamSet(d).skset(list(BASE))
    with pytest.raises(ValueError, match="onsite of"):
        DFTB(sk, type_switching=[_switch()])
    sw = _switch(mix_onsite=True, onsite_shift={"Cu3": {"d": 0.003}},
                 hubbard_shift={"Cu2": {"s": 0.01}})
    assert sw.parameter_names() == ["Cu.r0mix.out0.c", "Cu.r0mix.out0.w0", "Cu3.onsite.d",
                                    "Cu2.hubbard.s"]
    calc = DFTB(sk, type_switching=[sw], scc_tol=1e-12, smearing=("fermi", 300.0))
    pos, lab = _cu6(), ["Cu"] * 6
    res = calc.calculate(lab, pos)
    for c in (2, 9):
        def energy(x):
            p = pos.copy()
            p.flat[c] += x
            return calc.calculate(lab, p, forces=False).free_energy
        assert -res.forces.flat[c] == pytest.approx(_fd(energy, 1e-4), abs=2e-9)
    x0, g = sw.vector(), sw.gradient_from(res)
    for k in range(len(x0)):
        def energy(h):
            x = x0.copy()
            x[k] += h
            c2 = DFTB(sk, type_switching=[sw.with_vector(x)], scc_tol=1e-12,
                      smearing=("fermi", 300.0))
            return c2.calculate(lab, pos, forces=False).free_energy
        assert g[k] == pytest.approx(_fd(energy, 1e-4), abs=2e-9), sw.parameter_names()[k]


# ---- protocol, JSON and refusals ---------------------------------------------------------------------
def test_model_protocol_and_json():
    sw = _switch(mix_onsite=True, hubbard_shift={"Cu2": {"s": 0.01}})
    assert envmodel.engine_kwargs([sw]) == {"type_switching": [sw]}
    x = np.array([0.5, -0.2, 0.03])
    assert np.allclose(sw.with_vector(x).vector(), x)
    back = envmodel.model_from_dict(json.loads(json.dumps(sw.to_dict())))
    assert back == sw and r0mix_from_dict(sw.to_dict()) == sw
    frozen = _switch(fit_theta=False)
    assert len(frozen.vector()) == 0 and not frozen.wants_gradient
    assert _switch().with_gradient(False).wants_gradient is False


def test_gradient_without_the_element():
    """A structure without the element: zero gradient (no finite differences needed); a result
    without switch data: None."""
    from types import SimpleNamespace
    sw = _switch()
    none = SimpleNamespace(switching=None, type_weights={t: np.zeros(3) for t in sw.types})
    assert np.array_equal(sw.gradient_from(none), np.zeros(len(sw.vector())))
    assert sw.gradient_from(SimpleNamespace(switching=None, type_weights=None)) is None


@dataclass(frozen=True)
class _NoDerivatives(CoordinationNumber):
    has_derivatives: ClassVar[bool] = False


def test_refusals(sk):
    sw = _switch(descriptor=_NoDerivatives("Cu", window=(0.8, 1.4)))
    calc = DFTB(sk, type_switching=[sw], smearing=("fermi", 300.0))
    calc.calculate(["Cu"] * 6, _cu6(), forces=False)            # band-only use: fine
    with pytest.raises(ValueError, match="forces need a descriptor with derivatives"):
        calc.calculate(["Cu"] * 6, _cu6())
    with pytest.raises(ValueError, match="must be given as 'Cu'"):
        DFTB(sk, type_switching=[_switch()]).calculate(["Cu1"] * 2, [[0, 0, 0], [0, 0, 4.5]])
