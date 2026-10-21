"""Band structures, DOS and PDOS at frozen charges (pydftb.bands, milestone M2e).

DFTB+ parity (eigenvalues, occupations, region weights, dp_dos curves) is checked in
test_reference.py against the frozen *_bands / cu4co_pdos references; these tests check the
rules and identities without DFTB+.
"""
import numpy as np
import pytest
from ase.build import bulk

import pydftb
from pydftb.ase_calc import PyDFTB
from pydftb.bands import broaden, eigenstates, klines
from pydftb.constants import BOHR
from pydftb.validation import refgen

from .conftest import SKF_HCOCU


def _case(name):
    """pydftb result and engine for a frozen reference case (no DFTB+ needed)."""
    doc = refgen.load(name)
    inp = doc["input"]
    res, engine = refgen.run_pydftb(refgen.CASES_BY_NAME[name], inp, engine=True)
    periodic = inp["cell_angstrom"] is not None
    geo = (inp["labels"], np.array(inp["positions_angstrom"]) / BOHR,
           np.array(inp["cell_angstrom"]) / BOHR if periodic else None,
           np.ones(3, bool) if periodic else None)
    return res, engine, geo


# ---- k-point paths ------------------------------------------------------------
def test_klines_rules():
    k, w = klines([(1, (0, 0, 0)), (2, (0.5, 0, 0)), (0, (0, 0, 0.5)), (2, (0, 0.5, 0.5))])
    expected = [[0, 0, 0],                      # 1 point from the implicit Gamma: Gamma itself
                [0.25, 0, 0], [0.5, 0, 0],      # start excluded, end included
                [0, 0.25, 0.5], [0, 0.5, 0.5]]  # n = 0 jumps to (0, 0, 0.5) without a point
    assert np.allclose(k, expected) and np.allclose(w, 0.2)
    k, _ = klines([(0, (0.5, 0.5, 0.5)), (2, (0, 0, 0))])    # a leading jump: starts at L
    assert np.allclose(k, [[0.25, 0.25, 0.25], [0, 0, 0]])
    for bad in ([], [(-1, (0, 0, 0))], [(0, (0, 0, 0))]):
        with pytest.raises(ValueError):
            klines(bad)


def test_path_kpoints_keep_order_and_are_not_reduced():
    path = bulk("Cu", "fcc", a=3.61).cell.bandpath("GXG", npoints=9)
    for spec in (path, path.kpts):
        k, w = pydftb.make_kpoints(spec)
        assert np.array_equal(k, path.kpts) and np.allclose(w, 1 / 9)   # X->G mirrors G->X
    k, w = pydftb.make_kpoints((4, 4, 4))
    assert len(k) < 64                                                  # grids still reduced


# ---- frozen-charge eigenstates ------------------------------------------------
@pytest.mark.parametrize("name", ["li_bcc_k666_1000K", "cuo_chain_spin_k422", "h2o_dftb3",
                                  "cu4co_shell_resolved"])
def test_scc_grid_reproduces_scc(name):
    """On the SCC's own k-points, H built once from q_in is the SCC's last Hamiltonian."""
    res, engine, geo = _case(name)
    st = eigenstates(engine, *geo, res.q_in, projections=True, scc_fermi=res.fermi)
    assert np.abs(st.eigenvalues.reshape(-1) - np.concatenate(res.eigenvalues)).max() < 1e-12
    assert np.abs(st.occupations.reshape(-1) - np.concatenate(res.occupations)).max() < 1e-10
    # projections: every state sums to 1; occupation- and k-weighted they are Mulliken
    assert np.abs(st.projections.sum(axis=2) - 1.0).max() < 1e-12
    q_orb = np.einsum("k,skn,skmn->m", st.kweights, st.occupations, st.projections)
    assert np.abs(q_orb - res.q_orbital).max() < 1e-10


def test_chunking_does_not_change_states():
    res, engine, geo = _case("cuo_chain_spin_k422")
    path = klines([(1, (0, 0, 0)), (7, (0.5, 0, 0))])
    a = eigenstates(engine, *geo, res.q_in, kpts=path, projections=True)
    b = eigenstates(engine, *geo, res.q_in, kpts=path, projections=True, chunk=3)
    assert np.abs(a.eigenvalues - b.eigenvalues).max() < 1e-13
    assert np.abs(a.weights(atoms=[0, 2]) - b.weights(atoms=[0, 2])).max() < 1e-12


def test_populations_are_checked():
    res, engine, geo = _case("cuo_chain_spin_k422")
    with pytest.raises(ValueError, match="populations"):
        eigenstates(engine, *geo, res.charges)        # atom charges, not [up, down] per shell


# ---- regions, DOS, PDOS -------------------------------------------------------
def _integral(y, x):
    """Trapezoidal rule (np.trapezoid needs NumPy >= 2)."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def test_region_orbitals_follow_dftbplus_naming():
    res, engine, geo = _case("cu4co_shell_resolved")
    st = eigenstates(engine, *geo, res.q_in, projections=True)
    # Cu: s, p, d -> 9 orbitals; DFTB+ file cu.3.-2.out = first d orbital (m = -2)
    assert list(st.orbitals(atoms=[1], shell=2, m=-2)) == [9 + 4]
    assert list(st.orbitals(atoms=[0], shell=1)) == [1, 2, 3]
    assert len(st.orbitals()) == st.basis.norb
    with pytest.raises(ValueError):
        st.orbitals(m=0)                            # m needs a shell
    with pytest.raises(ValueError):
        st.orbitals(atoms=[4], shell=2)             # C has no d shell
    shells = sum(st.weights(atoms=[0, 1, 2, 3], shell=i) for i in range(3))
    assert np.abs(shells - st.weights(atoms=[0, 1, 2, 3])).max() < 1e-14


def test_dos_integrates_to_the_number_of_states():
    res, engine, geo = _case("li_bcc_k666_1000K")
    st = eigenstates(engine, *geo, res.q_in, kpts=pydftb.make_kpoints((8, 8, 8)),
                     projections=True)
    nstate = st.eigenvalues.shape[2]
    x, y = st.dos()                                  # Gaussian, cut at 3.5 widths
    assert _integral(y[0], x) == pytest.approx(nstate, rel=1e-5)
    x, y = st.dos(kind="fermi", width=0.05)          # cut at 7 widths: tanh(3.5) of the weight,
    integral = _integral(y[0], x)                    # plus the window's rounding to the grid
    assert nstate * np.tanh(3.5) < integral < nstate * (np.tanh(3.5) + 5e-4)
    x, y = st.dos(occupied=True)                     # one electron per Li
    assert _integral(y[0], x) == pytest.approx(1.0, rel=1e-4)
    xp, yp = st.pdos(atoms=[0])                      # all orbitals: the total DOS
    assert np.array_equal(xp, st.dos()[0]) and np.abs(yp - st.dos()[1]).max() < 1e-12


def _dp_dos_loop(e, kw, w, sigma, cutoff, grid):
    """dp_dos's Gaussian rules written as its plain loops (for checking the vectorised code)."""
    dsig = cutoff * sigma
    xmin = np.floor((e.min() - dsig) / grid) * grid
    xmax = np.ceil((e.max() + dsig) / grid) * grid
    x = np.arange(xmin, xmax + grid, grid, dtype=float)
    y = np.zeros((e.shape[0], len(x)))
    for s in range(e.shape[0]):
        for k in range(e.shape[1]):
            for ev, occ in zip(e[s, k], w[s, k]):
                lo = int(np.floor((ev - dsig - xmin) / grid))
                up = int(np.ceil((ev + dsig - xmin) / grid))
                y[s, lo:up + 1] += (kw[k] / (sigma * np.sqrt(np.pi)) * occ
                                    * np.exp(-((ev - x[lo:up + 1]) / sigma) ** 2))
    return x, y


def test_broaden_matches_the_dp_dos_rules():
    rng = np.random.default_rng(5)
    e = np.sort(rng.uniform(-12, 6, (2, 7, 11)), axis=2)
    kw = rng.uniform(0.5, 1.5, 7)
    kw /= kw.sum()
    w = rng.uniform(0, 1, e.shape)
    x, y = broaden(e, kw, w, width=0.15, cutoff=4.0, grid=0.02)
    xr, yr = _dp_dos_loop(e, kw, w, 0.15, 4.0, 0.02)
    assert np.array_equal(x, xr) and np.abs(y - yr).max() < 1e-12 * np.abs(yr).max()
    with pytest.raises(ValueError):
        broaden(e, kw, kind="lorentz")


# ---- ASE ----------------------------------------------------------------------
def test_ase_band_structure_at_scc_charges():
    cu = bulk("Cu", "fcc", a=3.61)
    calc = PyDFTB(SKF_HCOCU, kpts=(3, 3, 3), temperature=1000.0, scc_tol=1e-10)
    cu.calc = calc
    cu.get_potential_energy()
    bs = calc.band_structure(path="GXWL", npoints=12)
    assert bs.energies.shape == (1, 12, 9) and bs.reference == pytest.approx(calc.get_fermi_level())
    # Gamma is in the 3x3x3 grid: the path's first point agrees with the SCC eigenvalues there
    gamma = int(np.nonzero(~np.any(calc.get_ibz_k_points(), axis=1))[0][0])
    assert np.abs(bs.energies[0, 0] - calc.get_eigenvalues(kpt=gamma)).max() < 1e-10
    st = calc.states(kpts=(6, 6, 6))
    assert st.eigenvalues.shape[1] == len(pydftb.make_kpoints((6, 6, 6))[0])
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    bs.plot()


def test_ase_kpts_band_path_as_with_dftbplus():
    """kpts = band path: the points in order and ASE's own band_structure() (non-SCC here)."""
    cu = bulk("Cu", "fcc", a=3.61)
    calc = PyDFTB(SKF_HCOCU, kpts={"path": "GXW", "npoints": 10}, scc=False)
    cu.calc = calc
    cu.get_potential_energy()
    assert np.allclose(calc.get_ibz_k_points(), cu.cell.bandpath("GXW", npoints=10).kpts)
    bs = calc.band_structure()
    assert bs.energies.shape == (1, 10, 9)
    # the same points through band_structure(path=...) at frozen (here: reference) charges
    assert np.abs(calc.band_structure(path="GXW", npoints=10).energies - bs.energies).max() < 1e-10
