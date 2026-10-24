"""Shells from several SK files (milestone M6f-5, DFTB+ ``SelectedShells``).

The DFTB+ references are the ``*_selected*`` cases of :mod:`pydftb.validation.refgen`.  Here:
the bookkeeping (groups, file names and formats, shell names), the default path bitwise
unchanged with one group, Cu {"sd" "p"} equal to plain Cu (the same basis reordered), forces and
stress against finite differences, and the refused combinations.
"""
import numpy as np
import pytest

from pydftb import DFTB, ParamSet
from pydftb.api import make_kpoints
from pydftb.constants import BOHR
from pydftb.skf import parse_selected_shells, shell_names, write_typed_copies
from pydftb.validation import cases as C

SRC = C.SKF_SETS["ptbp_hcocu"]


def _typed(tmp_path, base_of):
    d = tmp_path / "skf"
    write_typed_copies(SRC, str(d), base_of, template="{a}-{b}.skf.gz")
    return str(d)


@pytest.fixture(scope="module")
def cu_dir(tmp_path_factory):
    return _typed(tmp_path_factory.mktemp("cu"), {"Cu": "Cu", "Cu2": "Cu", "C": "C", "O": "O"})


@pytest.fixture(scope="module")
def ox_dir(tmp_path_factory):
    return _typed(tmp_path_factory.mktemp("ox"), {"O": "O", "Ox": "H", "H": "H"})


def _run(skdir, atoms, sel=None, max_l=None, shell_resolved=False, **kw):
    p = ParamSet(skdir, max_l=max_l, selected_shells=sel, shell_resolved=shell_resolved)
    labels = atoms.get_chemical_symbols()
    periodic = bool(atoms.pbc.any())
    calc = DFTB(p.skset(labels), scc_tol=1e-10, max_scc=300, **kw)
    cell = atoms.cell.array / BOHR if periodic else None
    return calc, calc.calculate(labels, atoms.positions / BOHR, cell,
                                np.ones(3, bool) if periodic else None)


SEL_O = {"O": (("O", "sp"), ("Ox", "s"))}


def test_parse_and_shell_names():
    assert parse_selected_shells({"Si": ("sp", "s")}) == {"Si": [("Si", [0, 1]), ("Si2", [0])]}
    assert parse_selected_shells({"O": (("O", "SP"), ("Ox", [0]))}) == {
        "O": [("O", [0, 1]), ("Ox", [0])]}
    assert shell_names([0, 1, 0, 2, 1]) == ["s", "p", "s2", "d", "p2"]
    for bad in ({"O": ("spp",)}, {"O": ("x",)}, {"O": ("spdfs",)}, {"O": ()}):
        with pytest.raises(ValueError):
            parse_selected_shells(bad)
    with pytest.raises(ValueError, match="not both"):
        ParamSet(SRC, max_l={"O": 1}, selected_shells={"O": ("sp",)})


def test_sk_set_layout(ox_dir):
    sk = ParamSet(ox_dir, selected_shells=SEL_O, max_l={"H": 0}).skset(["O", "H"])
    assert sk.shells("O") == [0, 1, 0] and sk.norb("O") == 5 and sk.selected
    h = sk.files[("H", "H")]
    o = sk.group_files[("O", "O")][0][0]
    x = sk.group_files[("O", "O")][1][1]                         # Ox-Ox = H-H, homonuclear
    assert sk.group_files[("O", "O")][0][1].homo is False        # O-Ox: heteronuclear format
    assert x.homo and np.array_equal(x.table, h.table)
    np.testing.assert_array_equal(sk.onsite["O"], [o.onsite[0], o.onsite[1], h.onsite[0]])
    np.testing.assert_array_equal(sk.occupation["O"], [o.occupation[0], o.occupation[1],
                                                       h.occupation[0]])
    assert np.all(sk.hubbard["O"] == o.hubbard[0])               # atom-resolved: the first U
    assert sk.mass["O"] == o.mass
    shell = ParamSet(ox_dir, selected_shells=SEL_O, max_l={"H": 0},
                     shell_resolved=True).skset(["O", "H"])
    np.testing.assert_array_equal(shell.hubbard["O"], [o.hubbard[0], o.hubbard[1], h.hubbard[0]])


def test_one_group_is_the_default_path(ox_dir):
    """A label whose only group is its own file with s..l_max is a plain label: bitwise."""
    atoms = C._water()
    _, r0 = _run(ox_dir, atoms, max_l={"O": 1, "H": 0})
    _, r1 = _run(ox_dir, atoms, sel={"O": ("sp",), "H": ("s",)})
    assert r1.energy == r0.energy and np.array_equal(r1.forces, r0.forces)
    assert np.array_equal(r1.charges, r0.charges)


def test_cu_sd_p_is_cu_reordered(cu_dir):
    """Cu {"sd" "p"} with the p group from a copy of Cu: the plain basis in another order."""
    atoms = C._cu4co()
    _, r0 = _run(cu_dir, atoms, max_l={"Cu": 2, "C": 1, "O": 1}, smearing=("fermi", 300.0))
    calc, r1 = _run(cu_dir, atoms, sel={"Cu": ("sd", "p")}, max_l={"C": 1, "O": 1},
                    smearing=("fermi", 300.0))
    assert calc.sk.shells("Cu") == [0, 2, 1]
    assert abs(r1.free_energy - r0.free_energy) < 1e-12 and r1.n_scc == r0.n_scc
    assert np.abs(r1.forces - r0.forces).max() < 1e-11
    assert np.abs(r1.charges - r0.charges).max() < 1e-11


def test_cu_fcc_sd_p_stress(cu_dir):
    atoms = C._cu_fcc()
    atoms.set_cell(atoms.cell.array @ np.array([[1.0, 0.02, 0.0], [0.0, 1.0, 0.01],
                                                [0.0, 0.0, 0.98]]), scale_atoms=True)
    kw = dict(kpts=make_kpoints((3, 3, 3)), smearing=("fermi", 1000.0))
    _, r0 = _run(cu_dir, atoms, max_l={"Cu": 2}, **kw)
    _, r1 = _run(cu_dir, atoms, sel={"Cu": ("sd", "p")}, **kw)
    assert abs(r1.free_energy - r0.free_energy) < 1e-12
    assert np.abs(r1.stress - r0.stress).max() < 1e-12


def test_forces_and_derivatives(ox_dir):
    """Forces of O {sp s*} (shell-resolved) against central differences of the free energy.  The
    step is small because the SK interpolation's derivative jumps at grid points (here the H-H
    distance lies 5e-6 Bohr below one); tight SCC keeps the energies' noise below the test."""
    atoms = C._water()
    calc, _ = _run(ox_dir, atoms, sel=SEL_O, max_l={"H": 0}, shell_resolved=True,
                   reference_occupations={0: {"s2": 0.0}}, smearing=("fermi", 300.0))
    calc.scc_tol = 1e-12
    labels = atoms.get_chemical_symbols()
    pos = atoms.positions / BOHR
    r = calc.calculate(labels, pos)
    h = 2e-6
    for a in range(3):
        for k in range(3):
            e = []
            for s in (h, -h):
                p = pos.copy()
                p[a, k] += s
                e.append(calc.calculate(labels, p).free_energy)
            assert abs(-(e[0] - e[1]) / (2 * h) - r.forces[a, k]) < 2e-8


def test_reference_occupation_names(ox_dir):
    atoms = C._water()
    _, r = _run(ox_dir, atoms, sel=SEL_O, max_l={"H": 0}, reference_occupations={0: {"s2": 0.0}})
    assert abs(r.charges.sum() - 8.0) < 1e-9                    # O 6 + 2 H, s2 without electrons
    with pytest.raises(ValueError, match="no shell 's3'"):
        _run(ox_dir, atoms, sel=SEL_O, max_l={"H": 0}, reference_occupations={0: {"s3": 0.0}})


def test_checks(tmp_path, ox_dir):
    # a file between two groups of one label with an on-site line: refused (DFTB+ misreads it)
    d = tmp_path / "bad"
    write_typed_copies(SRC, str(d), {"O": "O", "Ox": "H", "H": "H"}, template="{a}-{b}.skf.gz")
    (d / "O-Ox.skf").write_text((d / "Ox-Ox.skf").read_text())     # homonuclear format
    with pytest.raises(ValueError, match="heteronuclear format"):
        ParamSet(str(d), selected_shells=SEL_O, max_l={"H": 0}).skset(["O", "H"])
    with pytest.raises(FileNotFoundError):
        ParamSet(SRC, selected_shells=SEL_O, max_l={"H": 0}).skset(["O", "H"])
    with pytest.raises(ValueError, match="hubbard_from"):
        p = ParamSet(ox_dir, selected_shells=SEL_O, max_l={"H": 0}, hubbard_from="lmax")
        p.skset(["O", "H"])


@pytest.mark.parametrize("opts, what", [
    ({"spin_orbit": {"O": [0.0, 0.01, 0.0]}}, "spin-orbit"),
    ({"plus_u": {"O": [("p", 0.1)]}}, "DFTB\\+U"),
    ({"casida": {"n_excitations": 2}}, "Casida"),
])
def test_refused(ox_dir, opts, what):
    sk = ParamSet(ox_dir, selected_shells=SEL_O, max_l={"H": 0}).skset(["O", "H"])
    with pytest.raises(ValueError, match=what):
        DFTB(sk, **opts)


def test_refused_torch(ox_dir):
    pytest.importorskip("torch")
    sk = ParamSet(ox_dir, selected_shells=SEL_O, max_l={"H": 0}).skset(["O", "H"])
    with pytest.raises(ValueError, match="selected_shells"):
        DFTB(sk, backend="torch")
