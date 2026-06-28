"""utils._compat applies two runtime patches; verify they survive."""
import multiprocessing as mp


def test_fork_start_method_is_active():
    assert mp.get_start_method(allow_none=True) == 'fork'


def test_ase_io_read_returns_atoms_with_legacy_info(tmp_path):
    """A round-trip through ase.io.write + ase.io.read should leave the
    energy retrievable via both `atoms.info['energy']` (legacy code path)
    and `atoms.get_potential_energy()` (modern path)."""
    import ase
    import ase.io
    import numpy as np
    from ase.calculators.singlepoint import SinglePointCalculator

    a = ase.Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]],
                  cell=np.eye(3) * 5.0, pbc=True)
    a.calc = SinglePointCalculator(a, energy=-1.234, forces=np.zeros((2, 3)))
    path = tmp_path / "tmp.xyz"
    ase.io.write(str(path), a)
    loaded = ase.io.read(str(path))
    assert 'energy' in loaded.info, "compat shim should mirror calc.results['energy'] into info"
    assert loaded.info['energy'] == -1.234
    assert loaded.get_potential_energy() == -1.234
