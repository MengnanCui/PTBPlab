"""Band-structure optimisation (`--optimization_option bandstructure`).

The fast tests need ase and hotcent importable, like the rest of the suite,
but not DFTB+. The slow ones run hotcent and DFTB+ against the GPAW reference
of examples/07_bandstructure_cu.
"""
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.data import covalent_radii
from ase.io.jsonio import read_json
from ase.spectrum.band_structure import BandStructure
from ase.units import Bohr

import utils.loss
from utils.calculator import (band_json_name, check_band_reference,
                              delta_band_binary, dftbplus_calculator,
                              get_band_path, path_dict)
from utils.loss import Loss

REPO = Path(__file__).resolve().parent.parent
EXAMPLE_REF = REPO / 'examples' / '07_bandstructure_cu' / 'ref'
CONF = [5.2, 5.2, 2.2]                 # [r0_w, r0_d, p] of a band-structure step
NKPTS = 41


def cu_fcc():
    atoms = bulk('Cu', 'fcc', a=3.6366)
    atoms.info['structure_name'] = 'fcc'
    return atoms


def band_args(monkeypatch, *extra):
    """Parsed legacy CLI arguments of a band-structure run on example 07."""
    from cli.arg_parser import build_default_arg_parser
    monkeypatch.setattr(sys, 'argv', [
        'run.py', '--optimization_option', 'bandstructure',
        '--ref_dir', str(EXAMPLE_REF), '--dft_file', 'cu_fcc.xyz',
        '--dft_band_file', 'cu_fcc_band.json', *extra])
    return build_default_arg_parser()


# ----------------------------------------------------------------------------
# Band path of the DFTB calculation
# ----------------------------------------------------------------------------
def test_band_path_of_path_dict_is_accepted_by_dftb():
    atoms = cu_fcc()
    bandpath = get_band_path(atoms, 'fcc')
    assert bandpath.path == path_dict['fcc']['path']
    assert len(bandpath.kpts) == path_dict['fcc']['npoints']

    dftb_args = {'calc_type': 'static', 'poly_rep': 'Yes', 'SCC': 'Yes', 'slater_p': './',
                 'kpts': {'path': bandpath.path, 'kpts': bandpath.kpts}}
    calc = dftbplus_calculator(atoms, dftb_args)
    # explicit list of k-points: KPointsAndWeights = Klines { 1 kx ky kz ... }
    assert calc.parameters['Hamiltonian_KPointsAndWeights_'].strip() == 'Klines'
    np.testing.assert_allclose(calc.get_ibz_k_points(), bandpath.kpts)


def test_band_path_prefers_the_reference_path():
    atoms = cu_fcc()
    ref_bandpath = atoms.cell.bandpath('GXL', npoints=17)
    assert get_band_path(atoms, 'fcc', ref_bandpath) is ref_bandpath


def test_band_path_without_path_dict_entry_is_the_standard_one():
    atoms = cu_fcc()
    bandpath = get_band_path(atoms, 'band_0')
    assert bandpath.path == atoms.cell.bandpath(npoints=0).path
    assert len(bandpath.kpts) > 0


def test_band_path_special_points_are_in_the_basis_of_the_cell():
    """The 'dimer' cell of dft/dft.xyz is tetragonal with its unique axis along x;
    the standard tetragonal cell has it along z."""
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.2, 0, 0]], cell=[9.9, 7.92, 7.92], pbc=True)
    bandpath = get_band_path(atoms, 'dimer')
    assert bandpath.path == path_dict['dimer']['path']
    reciprocal = atoms.cell.reciprocal()
    z_point = bandpath.special_points['Z'] @ reciprocal
    m_point = bandpath.special_points['M'] @ reciprocal
    assert abs(z_point[1]) < 1e-12 and abs(z_point[2]) < 1e-12   # along the unique axis
    assert abs(m_point[0]) < 1e-12                               # in the square plane


# ----------------------------------------------------------------------------
# delta_band_binary on synthetic band structures (Cu: 11 electrons, so band 5
# is the highest occupied band and band 6 the next one)
# ----------------------------------------------------------------------------
def cu_bands(band5, band6):
    x = np.linspace(0, 1, NKPTS)
    bands = [-4.0 + 0.5 * n + 0.1 * np.cos(2 * np.pi * x) for n in range(5)]
    bands += [band5(x), band6(x), 5.0 + x, 6.0 + x]
    return np.array(bands).T[np.newaxis]     # (spin, k-point, band)


def write_bands(filename, bandpath, relative_energies, reference):
    BandStructure(bandpath, relative_energies + reference, reference).write(filename)


def run_delta(tmp_path, dft_rel, dftb_rel, dft_ef=7.76, dftb_ref=-70.9, dftb_bandpath=None):
    """delta_band_binary for a DFT reference and a DFTB band structure as
    calc_band_binary writes it (energies referenced to dftb_ref)."""
    atoms = cu_fcc()
    bandpath = atoms.cell.bandpath('WXGLKGL', npoints=NKPTS)
    write_bands(tmp_path / 'dft.json', bandpath, dft_rel, dft_ef)
    if dftb_bandpath is None:
        dftb_bandpath = bandpath
    write_bands(tmp_path / band_json_name('fcc', CONF), dftb_bandpath, dftb_rel, dftb_ref)
    return delta_band_binary(CONF, tmp_path / 'dft.json', [atoms], [1.0])


def test_delta_band_binary_identical_bands_of_a_metal(tmp_path, monkeypatch):
    """Band 5 crosses the Fermi level, band 6 lies above all of band 5: still a
    metal, so both band structures stay referenced to their Fermi level."""
    monkeypatch.chdir(tmp_path)
    bands = cu_bands(lambda x: -2.0 + 4.0 * x, lambda x: 3.0 + x)
    mu_homo, mu_lumo = run_delta(tmp_path, bands, bands)
    assert mu_homo == pytest.approx(0.0, abs=1e-9)
    assert mu_lumo == pytest.approx(0.0, abs=1e-9)
    assert (tmp_path / 'pic_5.2_5.2_2.2.png').exists()


def test_delta_band_binary_compares_the_bands_within_1_eV(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dft = cu_bands(lambda x: -2.0 + 4.0 * x, lambda x: 3.0 + x)
    dftb = dft.copy()
    dftb[0, :, 5] += 0.1
    mu_homo, mu_lumo = run_delta(tmp_path, dft, dftb)
    assert mu_homo == pytest.approx(0.1)
    # no point of band 6 within 1 eV of the Fermi level: 0, not NaN
    assert mu_lumo == 0.0


def test_delta_band_binary_references_a_gapped_reference_to_its_valence_band_maximum(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bands = cu_bands(lambda x: -2.0 + x, lambda x: 0.5 + x)       # gap from -1.0 to 0.5
    vbm = bands[0, :, 5].max()
    # DFT: Fermi level in the gap; DFTB: referenced to its valence band maximum
    mu_homo, mu_lumo = run_delta(tmp_path, bands, bands - vbm, dft_ef=7.76, dftb_ref=-70.9)
    assert mu_homo == pytest.approx(0.0, abs=1e-9)
    assert mu_lumo == pytest.approx(0.0, abs=1e-9)


def test_delta_band_binary_needs_the_same_k_points(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bands = cu_bands(lambda x: -2.0 + 4.0 * x, lambda x: 3.0 + x)
    other_path = cu_fcc().cell.bandpath('GXL', npoints=NKPTS)
    with pytest.raises(ValueError, match='k-points'):
        run_delta(tmp_path, bands, bands, dftb_bandpath=other_path)


# ----------------------------------------------------------------------------
# DFT reference checks
# ----------------------------------------------------------------------------
def test_check_band_reference_accepts_the_example_reference():
    ref = read_json(EXAMPLE_REF / 'cu_fcc_band.json')
    atoms = ase.io.read(EXAMPLE_REF / 'cu_fcc.xyz')
    check_band_reference(ref, atoms)
    assert atoms.info['structure_name'] == 'fcc'


@pytest.mark.parametrize('case, match', [
    ('not_bands', 'not an ase BandStructure'),
    ('other_cell', 'cell'),
    ('few_bands', 'bands'),
    ('semicore', 'semicore'),
])
def test_check_band_reference_rejects(case, match):
    atoms = cu_fcc()
    bandpath = atoms.cell.bandpath('WXGLKGL', npoints=NKPTS)
    bands = cu_bands(lambda x: -2.0 + 4.0 * x, lambda x: 3.0 + x)
    if case == 'not_bands':
        ref = atoms
    elif case == 'other_cell':
        ref = BandStructure(bulk('Cu', 'fcc', a=3.9).cell.bandpath('WXGLKGL', npoints=NKPTS), bands, 0.0)
    elif case == 'few_bands':
        ref = BandStructure(bandpath, bands[:, :, :6], 0.0)
    else:
        # band 6 occupied: the reference counts more electrons as valence
        ref = BandStructure(bandpath, bands - 5.0, 0.0)
    with pytest.raises(ValueError, match=match):
        check_band_reference(ref, atoms)


# ----------------------------------------------------------------------------
# Loss.BandStructure and the Bayesian optimisation seed
# ----------------------------------------------------------------------------
def test_band_structure_step_scores_failed_skf_generation(tmp_path, monkeypatch):
    """No SKF (hotcent failed for these parameters) is scored, not raised."""
    monkeypatch.chdir(tmp_path)
    args = band_args(monkeypatch, '--superposition', 'density')
    loss = Loss(args_input=args, dft_input=ase.io.read(EXAMPLE_REF / 'cu_fcc.xyz', ':'),
                bolt_f11=[1] * 11, path=tmp_path)
    monkeypatch.setattr(loss.hp, 'full_hotcent', lambda *a, **k: None)

    def no_skf(*args, **kwargs):
        raise FileNotFoundError("[Errno 2] No such file or directory: './Cu-Cu.skf'")
    monkeypatch.setattr(utils.loss, 'calc_band_binary', no_skf)

    assert loss.Loss_BandStructure([5.2, 2.2]) == 20
    row = (tmp_path / 'results' / 'opt_5.200_2.200' / 'par.out').read_text()
    assert float(row.split(',')[-1]) == 20


def test_band_structure_step_raises_without_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = band_args(monkeypatch, '--superposition', 'density', '--dft_band_file', 'missing.json')
    loss = Loss(args_input=args, dft_input=ase.io.read(EXAMPLE_REF / 'cu_fcc.xyz', ':'),
                bolt_f11=[1] * 11, path=tmp_path)
    with pytest.raises(FileNotFoundError):
        loss.Loss_BandStructure([5.2, 2.2])


@pytest.mark.parametrize('symbol, seed', [('Cu', [5.2, 2.2]),    # QUASINANO2013 (r0, p)
                                          ('Ge', [4.5, 11.0])])  # p = 13.4 clipped
def test_bayesian_optimization_seed_of_band_structure(symbol, seed, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = band_args(monkeypatch)
    atoms = bulk(symbol, 'fcc', a=4.0)
    cov_radius = covalent_radii[atoms.numbers[0]] / Bohr
    loss = Loss(args_input=args, dft_input=[atoms], bolt_f11=[1] * 11, path=tmp_path)

    captured = {}

    def fake_gp_minimize(**kwargs):
        captured.update(kwargs)
        return 'result'
    monkeypatch.setattr(utils.loss, 'gp_minimize', fake_gp_minimize)

    assert loss.Bayesian_optimization(cov_radius, checkpoint_saver=None, n_calls=11) == 'result'
    assert captured['x0'] == pytest.approx(seed)
    assert captured['func'] == loss.Loss_BandStructure
    assert isinstance(captured['random_state'], int)
    for value, dimension in zip(captured['x0'], captured['dimensions']):
        assert dimension.low <= value <= dimension.high


# ----------------------------------------------------------------------------
# cli/run.py (the legacy flat CLI), with the expensive loss replaced
# ----------------------------------------------------------------------------
def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['run.py', *argv])
    runpy.run_path(str(REPO / 'cli' / 'run.py'), run_name='__main__')


def fake_band_loss(calls):
    """Quadratic stand-in for Loss.Loss_BandStructure that creates the step's
    results folder, as the real one does."""
    def loss_band_structure(self, parameters):
        r0, p = parameters
        calls.append((list(parameters), self.args_input.superposition))
        folder = Path(self.path) / self.args_input.results_dir / f"opt_{r0:.3f}_{p:.3f}"
        folder.mkdir(parents=True, exist_ok=True)
        return (r0 - 5.0) ** 2 + (p - 3.0) ** 2
    return loss_band_structure


BAND_ARGV = ['--optimization_option', 'bandstructure', '--ref_dir', str(EXAMPLE_REF),
             '--dft_file', 'cu_fcc.xyz', '--dft_band_file', 'cu_fcc_band.json']


def test_cli_bandstructure_bayesian_optimization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(Loss, 'Loss_BandStructure', fake_band_loss(calls))

    run_cli(monkeypatch, *BAND_ARGV, '--superposition', 'density', '--n_calls', '11')

    assert len(calls) == 11
    assert calls[0] == ([5.2, 2.2], 'density')       # QUASINANO2013 seed for Cu
    assert (tmp_path / 'par.out').read_text().startswith('#r0_w,r0_d,p,cov_radius,mu_homo,mu_lumo,mu')
    for plot in ('convergence.png', 'eval.png', 'obj.png'):
        assert (tmp_path / plot).exists(), plot
    assert (tmp_path / 'checkpoint.pkl').exists()
    assert (tmp_path / 'results.tgz').exists()
    assert not (tmp_path / 'results').exists()
    assert len(list(tmp_path.glob('opt_*_*'))) == 1    # copy of the best step


def test_cli_bandstructure_pso_defaults_to_density_superposition(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(Loss, 'Loss_BandStructure', fake_band_loss(calls))

    run_cli(monkeypatch, *BAND_ARGV, '--optimizer', 'pso', '--n_particles', '2', '--n_calls', '4')

    assert len(calls) == 4
    assert {superposition for _, superposition in calls} == {'density'}
    assert (tmp_path / 'convergence.png').exists()
    assert (tmp_path / 'results.tgz').exists()


def test_cli_bandstructure_needs_a_single_structure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ase.io.write(tmp_path / 'two.xyz', [cu_fcc(), cu_fcc()])
    shutil.copy(EXAMPLE_REF / 'cu_fcc_band.json', tmp_path)
    with pytest.raises(AssertionError, match='one structure'):
        run_cli(monkeypatch, '--optimization_option', 'bandstructure', '--ref_dir', str(tmp_path),
                '--dft_file', 'two.xyz', '--dft_band_file', 'cu_fcc_band.json')


def test_cli_repulsion_only_evaluates_the_given_confinement(tmp_path, monkeypatch):
    """--parameters sigma_rep: one Loss_EnergyGeometry([r0_w, r0_d]) call."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def loss_energy_geometry(self, parameters):
        calls.append(list(parameters))
        return 0.5
    monkeypatch.setattr(Loss, 'Loss_EnergyGeometry', loss_energy_geometry)

    run_cli(monkeypatch, '--optimization_option', 'energygeometry', '--parameters', 'sigma_rep',
            '--confinement_parameters', '3.0', '6.0', '2.0',
            '--ref_dir', str(REPO / 'dft'), '--dft_file', 'dft.xyz')

    assert calls == [[3.0, 6.0]]


# ----------------------------------------------------------------------------
# tools/calc_dftb.py
# ----------------------------------------------------------------------------
def test_run_one_by_one_reads_the_given_file(tmp_path, monkeypatch):
    import tools.calc_dftb as calc_dftb
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'data').mkdir()
    infile = tmp_path / 'data' / 'two.xyz'
    ase.io.write(infile, [bulk('Cu'), bulk('Cu', a=3.7)])
    monkeypatch.setattr(calc_dftb, 'run_calculator', lambda atoms, args: (atoms, 0.0, None))

    atoms_output, atoms_error = calc_dftb.run_one_by_one(str(infile), [], {})

    assert len(atoms_output) == 2 and atoms_error == []
    assert len(ase.io.read(tmp_path / 'two_dftb.xyz', ':')) == 2


# ----------------------------------------------------------------------------
# Slow: hotcent + DFTB+
# ----------------------------------------------------------------------------
needs_dftb = pytest.mark.skipif(shutil.which('dftb+') is None, reason='needs dftb+ on PATH')


@pytest.mark.slow
@needs_dftb
def test_band_structure_step_against_the_gpaw_reference(tmp_path, monkeypatch):
    pytest.importorskip('hotcent')
    monkeypatch.delenv('DFTB_PREFIX', raising=False)
    monkeypatch.chdir(tmp_path)
    args = band_args(monkeypatch, '--superposition', 'density')
    loss = Loss(args_input=args, dft_input=ase.io.read(EXAMPLE_REF / 'cu_fcc.xyz', ':'),
                bolt_f11=[1] * 11, path=tmp_path)

    mu = loss.Loss_BandStructure([5.2, 2.2])

    assert np.isfinite(mu) and 0.0 < mu < 20
    step = tmp_path / 'results' / 'opt_5.200_2.200'
    assert float((step / 'par.out').read_text().split(',')[-1]) == pytest.approx(mu)
    dftb = read_json(step / band_json_name('fcc', CONF))
    ref = read_json(EXAMPLE_REF / 'cu_fcc_band.json')
    np.testing.assert_allclose(dftb.path.kpts, ref.path.kpts)
    assert dftb.energies.shape[:2] == ref.energies.shape[:2]


@pytest.mark.slow
@needs_dftb
def test_ptbp_bandstructure_cli(tmp_path):
    pytest.importorskip('hotcent')
    env = {key: value for key, value in os.environ.items() if key != 'DFTB_PREFIX'}
    cmd = [sys.executable, str(REPO / 'parameterize.py'), *BAND_ARGV,
           '--superposition', 'density',
           '--optimizer', 'parallel_bo', '--n_particles', '2', '--n_calls', '2']

    result = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=1800)

    assert result.returncode == 0, (
        f"ptbp exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-3000:]}\n--- stderr ---\n{result.stderr[-3000:]}")
    assert 'Best parameter' in result.stdout
    assert (tmp_path / 'convergence.png').exists()
    assert (tmp_path / 'results.tgz').exists()
