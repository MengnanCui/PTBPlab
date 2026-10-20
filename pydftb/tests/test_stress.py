"""Stress tensor: strain finite differences, ASE conventions, lattice relaxation."""
import numpy as np
import pytest
from ase.build import bulk

from pydftb.ase_calc import PyDFTB
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.params import ParamSet
from pydftb.validation import refgen

from .conftest import SKF_ORBIT


@pytest.mark.parametrize("name", ["cuo_chain_k422", "water_box_gamma"])
def test_virial_matches_strain_finite_differences(name):
    case = refgen.CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    sk = ParamSet(refgen.SKF_SETS[case.skf_set], max_l=inp["max_l"]).skset(inp["labels"])
    kpts = None if inp["kpoints"] is None else (np.array(inp["kpoints"]), np.array(inp["kweights"]))
    calc = DFTB(sk, kpts=kpts, smearing=tuple(case.filling), scc_tol=1e-12, max_scc=500)
    pos = np.array(inp["positions_angstrom"]) / BOHR
    cell = np.array(inp["cell_angstrom"]) / BOHR
    pbc = np.ones(3, bool)
    w = -calc.calculate(inp["labels"], pos, cell, pbc).stress * abs(np.linalg.det(cell))
    h = 1e-5
    for a, b in [(0, 0), (1, 2), (0, 1)]:
        e = np.zeros((3, 3))
        e[a, b] += h / 2
        e[b, a] += h / 2
        vals = []
        for sgn in (1, -1):
            f = np.eye(3) + sgn * e
            vals.append(calc.calculate(inp["labels"], pos @ f.T, cell @ f.T, pbc,
                                       forces=False).free_energy)
        assert abs((vals[0] - vals[1]) / (2 * h) - w[a, b]) < 5e-8


def _li_bcc(a):
    atoms = bulk("Li", "bcc", a=a)
    atoms.calc = PyDFTB(SKF_ORBIT, kpts=(6, 6, 6), smearing=("fermi", 1000.0), scc_tol=1e-10)
    return atoms


def test_ase_stress_is_the_volume_derivative():
    """ASE convention: stress = +(1/V) dE/d(strain); a uniform strain eps gives (1/3V) dE/d(eps)."""
    a0, h = 3.4, 1e-5                       # below the E(V) minimum of this set (a = 3.556)
    atoms = _li_bcc(a0)
    s = atoms.get_stress()
    e = [_li_bcc(a0 * (1 + x)).get_potential_energy(force_consistent=True) for x in (h, -h)]
    p_fd = (e[0] - e[1]) / (2 * h) / (3 * atoms.get_volume())
    assert s.shape == (6,)
    assert np.allclose(s[:3], p_fd, rtol=1e-6)
    assert np.abs(s[3:]).max() < 1e-10
    assert s[0] < 0                         # compressed: the crystal pushes outwards


def test_cell_relaxation_finds_the_energy_volume_minimum(monkeypatch):
    """ASE cell relaxation needs one pydftb run per step and ends at the minimum of E(V)."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS
    runs = []
    calculate = PyDFTB.calculate

    def counted(self, *args, **kwargs):
        runs.append(args)
        return calculate(self, *args, **kwargs)

    monkeypatch.setattr(PyDFTB, "calculate", counted)
    atoms = _li_bcc(3.4)
    opt = BFGS(FrechetCellFilter(atoms), logfile=None)
    assert opt.run(fmax=1e-5)
    assert len(runs) == opt.nsteps + 1      # forces and stress come from the same run
    assert np.abs(atoms.get_stress()).max() < 1e-6
    a_opt = (2 * atoms.get_volume()) ** (1 / 3)      # bcc primitive cell: V = a^3 / 2
    scan = a_opt + np.linspace(-0.04, 0.04, 9)
    energies = [_li_bcc(a).get_potential_energy(force_consistent=True) for a in scan]
    assert np.argmin(energies) == 4
    roots = np.roots(np.polyder(np.polyfit(scan, energies, 4)))
    a_fit = min((r.real for r in roots if abs(r.imag) < 1e-9), key=lambda r: abs(r - a_opt))
    # E(a) is only piecewise smooth: as in DFTB+, the SK tables are interpolated with a
    # polynomial that changes at every grid node, which kinks dE/da by up to ~1e-3 eV/A.
    assert abs(a_opt - a_fit) < 2e-3
