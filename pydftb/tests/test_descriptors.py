"""Descriptors of the A-series environment models (A0 section 4, ladder L1-L4): gradients and
virials against finite differences on structures whose neighbours really are inside the
windows, invariances, periodic images, JSON round trips."""
import json

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule

from pydftb.constants import BOHR
from pydftb.descriptors import (PCA, SOAP, Affine, Angular, CoordinationNumber, Expand,
                                MACEFeatures, Radial, Stack, Standardize, compute_atoms,
                                descriptor_from_dict)


def _li_o():
    at = bulk("Li", "bcc", a=3.2, cubic=True).repeat((2, 2, 1))
    at.rattle(0.15, seed=1)
    sym = at.get_chemical_symbols()
    sym[1] = sym[6] = "O"
    at.set_chemical_symbols(sym)
    return at


def _cluster():
    at = molecule("CH3CH2OH")
    at.rattle(0.05, seed=3)
    return at


def _args(at):
    periodic = bool(np.any(at.pbc))
    return (at.get_chemical_symbols(), at.positions / BOHR,
            at.cell.array / BOHR if periodic else None,
            np.ones(3, bool) if periodic else None)


DESCRIPTORS = {
    "radial": Radial("Li", centres=(3.5, 5.0, 6.5), width=1.0, cutoff=9.0),
    "radial_split": Radial("Li", centres=(3.0, 5.5), width=(0.8, 1.2), cutoff=9.0, r_on=2.0,
                           neighbours=("Li", "O"), split=True),
    "g5": Angular("Li", zetas=(1.0, 2.0), lambdas=(-1.0, 1.0), etas=(0.0, 0.02), cutoff=8.0),
    "g4_split": Angular("Li", zetas=(1.0, 4.0), lambdas=(1.0,), etas=(0.01,), cutoff=8.0,
                        neighbours=("Li", "O"), split=True, form="G4"),
    "expand": Expand(CoordinationNumber("Li", neighbours=("Li", "O"), split=True,
                                        window=(0.8, 1.5)),
                     centres=((3.0, 5.0), (0.0, 1.0)), width=(1.0, 0.5),
                     clamp=((1.0, 8.0), None), include_input=True),
    "stack": Stack((Radial("Li", centres=(4.0,), cutoff=8.5),
                    Angular("Li", zetas=(2.0,), lambdas=(1.0,), cutoff=8.0))),
}


def _energy(desc, args, dedv):
    return float((desc.compute(*args).values * dedv).sum())


@pytest.mark.parametrize("name", sorted(DESCRIPTORS))
def test_gradient_and_virial_against_finite_differences(name):
    desc = DESCRIPTORS[name]
    at = _li_o()
    lab, pos, cell, pbc = _args(at)
    f = desc.compute(lab, pos, cell, pbc, derivatives=True)
    assert f.values.shape == (6, desc.size) and len(f.names) == desc.size
    assert len(f.pairs[0]) > 0 and np.abs(f.values).max() > 1e-3
    dedv = np.random.default_rng(0).normal(size=f.values.shape)
    g, vir = f.chain(dedv, len(lab), virial=True)
    assert np.abs(g).max() > 1e-3                            # not a trivially zero gradient
    h = 1e-6
    fd = np.zeros_like(g)
    for a in range(len(lab)):
        for x in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[a, x] += h
            pm[a, x] -= h
            fd[a, x] = (_energy(desc, (lab, pp, cell, pbc), dedv)
                        - _energy(desc, (lab, pm, cell, pbc), dedv)) / (2 * h)
    assert np.abs(g - fd).max() < 1e-6 * max(1.0, np.abs(fd).max())
    fdv = np.zeros((3, 3))
    for x in range(3):
        for y in range(3):
            def e_of(s):
                st = np.eye(3)
                st[x, y] += s
                return _energy(desc, (lab, pos @ st.T, cell @ st.T, pbc), dedv)
            fdv[x, y] = (e_of(h) - e_of(-h)) / (2 * h)
    assert np.abs(vir - fdv).max() < 1e-5 * max(1.0, np.abs(fdv).max())


def test_molecule_gradients():
    at = _cluster()
    lab, pos, _, _ = _args(at)
    for desc in (Radial("C", centres=(2.5, 4.0), cutoff=7.0, neighbours=("H", "O", "C"),
                        split=True),
                 Angular("O", zetas=(1.0, 2.0), lambdas=(-1.0, 1.0), cutoff=7.0, form="G4")):
        f = desc.compute(lab, pos, derivatives=True)
        dedv = np.random.default_rng(1).normal(size=f.values.shape)
        g, _ = f.chain(dedv, len(lab))
        assert np.abs(g).max() > 1e-4
        fd = np.zeros_like(g)
        for a in range(len(lab)):
            for x in range(3):
                pp, pm = pos.copy(), pos.copy()
                pp[a, x] += 1e-6
                pm[a, x] -= 1e-6
                fd[a, x] = (_energy(desc, (lab, pp, None, None), dedv)
                            - _energy(desc, (lab, pm, None, None), dedv)) / 2e-6
        assert np.abs(g - fd).max() < 1e-6


@pytest.mark.parametrize("name", sorted(DESCRIPTORS))
def test_rotation_permutation_and_supercell(name):
    desc = DESCRIPTORS[name]
    at = _li_o()
    v0 = compute_atoms(desc, at).values
    # rotation of positions and cell
    rot = at.copy()
    rot.rotate(37.0, (1.0, 2.0, 0.5), rotate_cell=True)
    assert np.abs(compute_atoms(desc, rot).values - v0).max() < 1e-9
    # permutation: the centre atoms come in the new order
    perm = np.random.default_rng(2).permutation(len(at))
    pa = at[perm]
    f = compute_atoms(desc, pa)
    order = {a: k for k, a in enumerate(compute_atoms(desc, at).atoms)}
    want = np.array([v0[order[int(perm[a])]] for a in f.atoms])
    assert np.abs(f.values - want).max() < 1e-9
    # a supercell repeats the values of the cell
    sc = at.repeat((2, 1, 1))
    vs = compute_atoms(desc, sc).values
    assert np.abs(vs - np.vstack([v0, v0])).max() < 1e-9


def test_angular_values_by_hand():
    """G5 with zeta 1, lambda +1, eta 0 for three atoms: 2 (1 + cos theta) f_c(r1) f_c(r2)."""
    lab = ["O", "H", "H"]
    pos = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 2.0, 0.3]])
    d = Angular("O", zetas=(1.0,), lambdas=(1.0,), etas=(0.0,), cutoff=6.0)
    v = d.compute(lab, pos).values[0, 0]
    from pydftb.descriptors import _cutoff
    r1, r2 = np.linalg.norm(pos[1]), np.linalg.norm(pos[2])
    c = pos[1] @ pos[2] / (r1 * r2)
    assert v == pytest.approx(2.0 * (1.0 + c) * _cutoff(r1, 6.0)[0] * _cutoff(r2, 6.0)[0])


def test_transforms_fitted_on_frames():
    frames = []
    for k in range(4):
        a = _li_o()
        a.rattle(0.1, seed=10 + k)
        frames.append(a)
    inner = Stack((Radial("Li", centres=(3.5, 5.0), cutoff=9.0),
                   CoordinationNumber("Li", neighbours=("O",))))
    std = Standardize(inner, frames)
    x = np.vstack([compute_atoms(std, a).values for a in frames])
    assert np.abs(x.mean(axis=0)).max() < 1e-12 and np.abs(x.std(axis=0) - 1).max() < 1e-12
    pca = PCA(inner, frames, n_components=2)
    assert pca.size == 2
    xp = np.vstack([compute_atoms(pca, a).values for a in frames])
    c = np.cov(xp.T)
    assert abs(c[0, 1]) < 1e-10 * max(1.0, abs(c[0, 0]))       # uncorrelated components
    at = frames[0]
    lab, pos, cell, pbc = _args(at)
    f = pca.compute(lab, pos, cell, pbc, derivatives=True)
    dedv = np.ones_like(f.values)
    g, _ = f.chain(dedv, len(lab))
    fd = np.zeros_like(g)
    for a in range(len(lab)):
        for xx in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[a, xx] += 1e-6
            pm[a, xx] -= 1e-6
            fd[a, xx] = (_energy(pca, (lab, pp, cell, pbc), dedv)
                         - _energy(pca, (lab, pm, cell, pbc), dedv)) / 2e-6
    assert np.abs(g).max() > 1e-4 and np.abs(g - fd).max() < 1e-6


@pytest.mark.parametrize("name", sorted(DESCRIPTORS))
def test_json_round_trip(name):
    desc = DESCRIPTORS[name]
    back = descriptor_from_dict(json.loads(json.dumps(desc.to_dict())))
    at = _li_o()
    assert np.array_equal(compute_atoms(back, at).values, compute_atoms(desc, at).values)
    frames = [_li_o()]
    aff = Affine.pca(Radial("Li", centres=(3.5, 5.0, 6.0), cutoff=9.0), frames, n_components=2)
    back = descriptor_from_dict(json.loads(json.dumps(aff.to_dict())))
    assert np.array_equal(compute_atoms(back, at).values, compute_atoms(aff, at).values)


def test_refusals():
    with pytest.raises(ValueError):
        Radial("Xx", centres=(1.0,))
    with pytest.raises(ValueError):
        Angular("Li", zetas=(0.5,))
    with pytest.raises(ValueError):
        Radial("Li", centres=(1.0,), split=True)
    with pytest.raises(ValueError):
        Expand(CoordinationNumber("Li"), centres=((1.0,), (2.0,)))


def test_soap_optional():
    pytest.importorskip("dscribe")
    at = _li_o()
    d = SOAP("Li", species=("Li", "O"), n_max=3, l_max=2)
    f = compute_atoms(d, at)
    assert f.values.shape == (6, d.size) and not d.has_derivatives
    rot = at.copy()
    rot.rotate(25.0, (0.3, 1.0, 0.2), rotate_cell=True)
    assert np.abs(compute_atoms(d, rot).values - f.values).max() < 1e-6


def test_mace_optional():
    pytest.importorskip("mace")
    at = _li_o()
    d = MACEFeatures("Li", model="small").with_size(at)
    f = compute_atoms(d, at)
    assert f.values.shape == (6, d.size)
