"""BandRef: k-point sets, JSON, edges, semicore counting and VASP readers (A series, A0)."""
import numpy as np
import pytest
from ase.build import bulk

from pydftb.bandref import (BandRef, count_semicore, from_eigenval, special_kpoints,
                            vasp_kpoints_text)


@pytest.mark.parametrize("name, a", [("Cu", 3.6), ("Li", 3.44)])
def test_special_kpoints_cover_the_standard_path(name, a):
    atoms = bulk(name, a=a)
    k, labels, level, segments = special_kpoints(atoms, fractions=(0.05, 0.1),
                                                 points_per_segment=11)
    bp = atoms.cell.bandpath(npoints=0)
    for lab, kp in bp.special_points.items():
        hits = [i for i, x in enumerate(labels) if x == lab and level[i] == 1]
        if lab in "".join(bp.path):
            assert hits and np.allclose(k[hits[0]], kp)
    n_seg = len(segments)
    assert n_seg > 0
    # two fractions from both ends of every segment
    assert int((level == 1).sum()) == len({x for x in labels if x and "+" not in x}) + 4 * n_seg
    for a_lab, b_lab, i0, i1 in segments:
        assert i1 - i0 + 1 == 11
        assert np.allclose(k[i0], bp.special_points[a_lab])
        assert np.allclose(k[i1], bp.special_points[b_lab])
        assert np.all(level[i0:i1 + 1] == 2)


def _toy(kind="path", metal=False):
    nk = 4
    e = np.array([[[-10.0, -2.0, -1.0, 3.0, 4.0]] * nk])
    occ = np.array([[[1, 1, 1, 0, 0]] * nk], float)
    if metal:
        occ[0, :, 2] = 0.5
    return BandRef(name="toy", kind=kind, kpts=np.zeros((nk, 3)) + np.arange(nk)[:, None] * 0.1,
                   kweights=np.ones(nk), energies=e, occupations=occ, fermi=0.5, n_semicore=1,
                   level=np.array([1, 1, 2, 2]), labels=["G", "", "", "X"],
                   segments=[["G", "X", 0, 3]])


def test_json_round_trip(tmp_path):
    ref = _toy().set_atoms(bulk("Cu", a=3.6))
    path = ref.to_json(str(tmp_path / "toy.json"))
    back = BandRef.from_json(path)
    assert back.name == "toy" and back.kind == "path" and back.n_semicore == 1
    assert np.array_equal(back.energies, ref.energies)
    assert np.array_equal(back.level, ref.level) and back.segments == ref.segments
    assert back.atoms().get_chemical_symbols() == ["Cu"]


def test_edges_insulator_and_metal():
    ins = _toy().edges()
    assert ins["vbm"] == -1.0 and ins["cbm"] == 3.0 and ins["gap"] == 4.0 and not ins["metal"]
    assert _toy().edges(n_valence_bands=2)["gap"] == 4.0
    met = _toy(metal=True).edges()
    assert met["metal"] and met["vbm"] == met["cbm"] == 0.5 and met["gap"] == 0.0


def test_occupations_on_a_zero_to_two_scale_are_fractions():
    ref = _toy()
    ref2 = BandRef(name="x", kind="ibz", kpts=ref.kpts, kweights=ref.kweights,
                   energies=ref.energies, occupations=2 * ref.occupations)
    assert ref2.occupations.max() == 1.0


def test_semicore_count():
    from ase import Atoms
    atoms = Atoms("OLi2", positions=np.zeros((3, 3)))     # Li_sv: one 1s band per Li
    assert count_semicore(atoms, {"Li": 3, "O": 6}, {"Li": 1, "O": 6}) == 2
    with pytest.raises(ValueError):                                   # one extra electron
        count_semicore(Atoms("OLi", positions=np.zeros((2, 3))), {"Li": 2, "O": 6},
                       {"Li": 1, "O": 6})


def test_vasp_kpoints_text():
    txt = vasp_kpoints_text([[0, 0, 0], [0.5, 0, 0]])
    lines = txt.splitlines()
    assert lines[1] == "2" and lines[2] == "Reciprocal" and lines[4].endswith("1.0")


def test_eigenval_reader(tmp_path):
    # VASP 6 EIGENVAL, ISPIN 1, 2 k-points, 3 bands, with occupations
    text = """    1    1    1    1
  0.1E+02  0.1E-09  0.1E-09  0.1E-09  0.5E-15
  1.0E-004
  CAR
 unknown system
      2      2      3

  0.0000000E+00  0.0000000E+00  0.0000000E+00  0.5000000E+00
    1      -5.000000   1.000000
    2      -1.000000   1.000000
    3       2.000000   0.000000

  0.5000000E+00  0.0000000E+00  0.0000000E+00  0.5000000E+00
    1      -4.500000   1.000000
    2      -0.500000   1.000000
    3       2.500000   0.000000
"""
    p = tmp_path / "EIGENVAL"
    p.write_text(text)
    ref = from_eigenval(str(p), fermi=0.3)
    assert ref.energies.shape == (1, 2, 3)
    assert np.allclose(ref.energies[0, 1], [-4.5, -0.5, 2.5])
    assert np.allclose(ref.occupations[0, :, 2], 0.0) and ref.fermi == 0.3
    assert np.allclose(ref.kweights, 0.5)
