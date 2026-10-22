"""Occupations, smearing conventions, mixers and the ASE calculator."""
import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule

import pydftb
from pydftb.ase_calc import PyDFTB
from pydftb.constants import BOHR
from pydftb.mixer import BroydenMixer
from pydftb.occupations import Smearing, fill

from .conftest import SKF_HCOCU


def test_smearing_specs():
    assert Smearing.from_spec(300.0) == Smearing("fermi", 300.0)
    assert Smearing.from_spec(("mp", 2000, 2)).mp_order == 2
    assert Smearing.from_spec(("gaussian", 100)).mp_order == 0
    assert Smearing.from_spec(("MethfesselPaxton", 100, 3)).order == 3
    assert Smearing("fermi", 0.0).kt == pytest.approx(1e-8)       # DFTB+ minTemp
    with pytest.raises(ValueError):
        Smearing("mp", 100.0, 0)


@pytest.mark.parametrize("spec", [("fermi", 0.0), ("fermi", 2000.0), ("gaussian", 1500.0),
                                  ("mp", 2000.0, 1), ("mp", 3000.0, 2)])
def test_fill_conserves_electrons(spec):
    rng = np.random.default_rng(1)
    ev = np.sort(rng.normal(size=(5, 12)) * 0.2, axis=1)
    w = rng.uniform(0.5, 1.5, 5)
    w /= w.sum()
    occ, mu, ts = fill(ev, w, 9.0, spec)
    assert sum(wk * o.sum() for wk, o in zip(w, occ)) == pytest.approx(9.0, abs=1e-9)


def test_insulator_fermi_level_mid_gap():
    ev = np.array([[-0.5, -0.4, -0.3, 0.1, 0.2]])
    occ, mu, ts = fill(ev, np.ones(1), 6.0, ("fermi", 0.0))
    assert mu == pytest.approx(-0.1)                 # half-way between -0.3 and 0.1
    assert np.allclose(occ[0], [2, 2, 2, 0, 0]) and ts == 0.0


def test_insulator_mid_gap_with_inexact_k_weights():
    """416 weights of 1/13 add up to 31.999999999999904, not 32 (the water box on a 13-point
    path): DFTB+'s middleGap then misses the gap (M2e report); pydftb allows for the rounding
    and stays mid-gap."""
    rng = np.random.default_rng(2)
    ev = np.sort(np.concatenate([rng.uniform(-0.5, -0.2, (13, 32)), rng.uniform(0.1, 0.4, (13, 4))],
                                axis=1), axis=1)
    w = np.full(13, 1.0 / 13)
    assert np.cumsum(np.repeat(w, 32))[-1] < 32.0                 # the rounding in question
    occ, mu, _ = fill(ev, w, 64.0, ("fermi", 0.0))
    assert mu == pytest.approx(0.5 * (ev[:, 31].max() + ev[:, 32].min()), abs=1e-15)
    assert np.allclose(np.array(occ)[:, :32], 2.0) and np.allclose(np.array(occ)[:, 32:], 0.0)


def test_fermi_level_exact_keeps_the_electron_count():
    """With an integer count DFTB+ puts mu mid-gap if the count there is within 1e-7: here
    the level 0.06 Ha below takes 5.6e-8 electrons out.  fermi_level='exact' bisects to the
    exact count (F is then the fixed-N Mermin free energy, see pydftb.autodiff)."""
    ev = np.array([[-0.5, -0.06, -0.025, 0.025, 0.4]])
    occ, mu, _ = fill(ev, np.ones(1), 6.0, Smearing("fermi", 1000.0))
    assert mu == 0.0 and 1e-8 < 6.0 - occ[0].sum() < 1e-7
    occ, mu, _ = fill(ev, np.ones(1), 6.0, Smearing("fermi", 1000.0, fermi_level="exact"))
    assert mu != 0.0 and abs(6.0 - occ[0].sum()) < 1e-14


def _cluster():
    a = Atoms("Cu4CO", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.1, 0], [1.25, 0.7, 2.0],
                                   [1.25, 0.7, 3.9], [1.25, 0.75, 5.05]])
    a.rattle(0.03, seed=3)
    return a


def _fd_free_energy_error(ts_convention):
    a = _cluster()
    sm = Smearing("mp", 3000.0, 2, ts_convention)
    params = pydftb.ParamSet(SKF_HCOCU)
    sym = a.get_chemical_symbols()
    calc = pydftb.DFTB(params.skset(sym), smearing=sm, scc_tol=1e-11)
    pos = a.positions / BOHR
    f = calc.calculate(sym, pos).forces
    h = 1e-4
    err = 0.0
    for atom, k in [(0, 0), (3, 2), (5, 1)]:
        p, m = pos.copy(), pos.copy()
        p[atom, k] += h
        m[atom, k] -= h
        fd = -(calc.calculate(sym, p, forces=False).free_energy
               - calc.calculate(sym, m, forces=False).free_energy) / (2 * h)
        err = max(err, abs(fd - f[atom, k]))
    return err


def test_mp_entropy_convention_documented():
    """Methfessel-Paxton order 2: only the 'variational' TS makes the Mermin energy the
    exact antiderivative of the forces; DFTB+'s convention (default, for parity) does not."""
    err_var = _fd_free_energy_error("variational")
    err_dftb = _fd_free_energy_error("dftb+")
    assert err_var < 1e-6
    assert err_dftb > 10 * err_var


def test_broyden_solves_linear_fixed_point():
    """On a linear map the Broyden mixer converges in far fewer steps than simple mixing."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6)) * 0.3
    b = rng.normal(size=6)
    g = lambda x: a @ x + b              # noqa: E731  fixed point x* = (1-a)^-1 b
    x_star = np.linalg.solve(np.eye(6) - a, b)
    m, x = BroydenMixer(), np.zeros(6)
    for it in range(60):
        x = m.mix(x, g(x))
        if np.abs(x - x_star).max() < 1e-10:
            break
    assert it < 40


def test_pydftb_ase_calculator_features():
    cu = bulk("Cu", "fcc", a=3.61)
    calc = PyDFTB(SKF_HCOCU, kpts=(4, 4, 4), smearing=("mp", 2000.0, 2), threads=1)
    cu.calc = calc
    e = cu.get_potential_energy()
    assert np.isfinite(e)
    assert len(calc.get_ibz_k_points()) == 32 and calc.get_k_point_weights().sum() == pytest.approx(1)
    assert calc.get_number_of_spins() == 1
    assert len(calc.get_eigenvalues(0)) == 9
    assert np.isfinite(calc.get_fermi_level())
    # explicit k-point list, no time-reversal reduction
    calc2 = PyDFTB(SKF_HCOCU, kpts={"kpts": [[0, 0, 0], [0.25, 0.25, 0.25]], "weights": [1, 1]},
                   time_reversal=False)
    cu.calc = calc2
    cu.get_potential_energy()
    assert len(calc2.get_ibz_k_points()) == 2


def test_unconverged_scc_warns_but_returns():
    w = molecule("H2O")
    w.calc = PyDFTB(SKF_HCOCU, max_scc=2, scc_tol=1e-12)
    with pytest.warns(RuntimeWarning, match="SCC not converged"):
        e = w.get_potential_energy()
    assert np.isfinite(e) and not w.calc.raw.converged


def test_pydftb_typed_labels(tmp_path):
    from pydftb.skf import write_typed_copies
    write_typed_copies(SKF_HCOCU, str(tmp_path), {"O": "O", "O2": "O", "H": "H"},
                       template="{a}-{b}.skf.gz")
    a = molecule("H2O")
    b = molecule("H2O")
    b.translate([0.3, 0.1, 2.9])
    a += b
    a.calc = PyDFTB(str(tmp_path), labels=["O", "H", "H", "O2", "H", "H"])
    e_typed = a.get_potential_energy()
    a.calc = PyDFTB(str(tmp_path))
    assert e_typed == pytest.approx(a.get_potential_energy(), abs=1e-10)
