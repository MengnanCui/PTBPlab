"""DFTB engine switch (utils/engines.py): DFTB+ through ASE or pydftb."""
import gzip
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase.build import bulk

import utils.engines as eng

ROOT = Path(__file__).resolve().parents[1]
CU_SKF_GZ = ROOT / "pydftb" / "tests" / "data" / "skf" / "ptbp_hcocu" / "Cu-Cu.skf.gz"
HAVE_DFTBPLUS = shutil.which("dftb+") is not None or os.path.exists(os.environ.get("DFTBPLUS", "/nonexistent"))


@pytest.fixture(autouse=True)
def _restore_engine_settings():
    saved = eng.settings()
    yield
    eng._SETTINGS.clear()
    eng._SETTINGS.update(saved)


@pytest.fixture
def cu_skf_dir(tmp_path):
    with gzip.open(CU_SKF_GZ, "rb") as fi, open(tmp_path / "Cu-Cu.skf", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    return tmp_path


def test_configure_validates():
    assert eng.configure(engine="pydftb")["engine"] == "pydftb"
    with pytest.raises(ValueError):
        eng.configure(engine="vasp")
    with pytest.raises(ValueError):
        eng.configure(smearing="mp", mp_order=0)
    with pytest.raises(KeyError):
        eng.configure(colour="blue")


def test_default_keeps_historical_dftbplus_input():
    assert eng.settings()["engine"] == "dftbplus"
    assert eng.dftbplus_filling_kwargs() is None
    assert eng.expected_filling() is None


def test_dftbplus_filling_keywords_account_for_parser_version_1():
    eng.configure(smearing="mp", smearing_temperature=2000.0, mp_order=2)
    kw = eng.dftbplus_filling_kwargs()
    assert kw["Hamiltonian_Filling_"] == "MethfesselPaxton"
    assert kw["Hamiltonian_Filling_Order"] == 3          # DFTB+ subtracts 1 for ParserVersion < 14
    assert kw["Hamiltonian_Filling_Temperature"] == pytest.approx(2000.0 * eng.KB_HARTREE)
    assert eng.expected_filling() == ("MethfesselPaxton", pytest.approx(2000.0 * eng.KB_HARTREE), 2)


def test_check_dftbplus_filling(tmp_path):
    (tmp_path / "dftb_pin.hsd").write_text(
        "Hamiltonian = DFTB {\n  Filling = MethfesselPaxton {\n    Temperature = 6.3336E-003\n"
        "    Order = 2\n    IndependentKFilling = No\n  }\n}\n")
    eng.check_dftbplus_filling(str(tmp_path), ("MethfesselPaxton", 6.3336e-3, 2))
    with pytest.raises(RuntimeError):
        eng.check_dftbplus_filling(str(tmp_path), ("MethfesselPaxton", 6.3336e-3, 1))
    with pytest.raises(RuntimeError):
        eng.check_dftbplus_filling(str(tmp_path), ("Fermi", 6.3336e-3, None))


def test_pydftb_calculator_settings(cu_skf_dir):
    from pydftb.ase_calc import PyDFTB
    eng.configure(engine="pydftb")
    args = dict(calc_type="static", poly_rep="Yes", SCC="Yes", kpts=[3, 4, 4],
                slater_p=str(cu_skf_dir) + "/")
    calc = eng.make_calculator(bulk("Cu"), args)
    assert isinstance(calc, PyDFTB)
    opts = calc.engine_opts
    assert opts["repulsive"] == "polynomial" and opts["scc_tol"] == 1e-5 and opts["max_scc"] == 1000
    assert opts["smearing"] == ("fermi", 0.0) and opts["fd_delta"] == eng.LEGACY_FD_DELTA
    assert calc.params.shell_resolved and calc.kpts_spec == (3, 4, 4)
    eng.configure(pydftb_forces="exact", smearing="mp", mp_order=1, smearing_temperature=500)
    calc = eng.make_calculator(bulk("Cu"), args)
    assert calc.engine_opts["fd_delta"] is None
    assert calc.engine_opts["smearing"] == ("mp", 500.0, 1)
    with pytest.raises(NotImplementedError):
        eng.make_calculator(bulk("Cu"), dict(args, calc_type="xtb"))


def test_dftbplus_calculator_alias_dispatches(cu_skf_dir):
    from pydftb.ase_calc import PyDFTB
    from utils.calculator import _CheckedDftb, dftbplus_calculator
    args = dict(calc_type="static", poly_rep="Yes", SCC="Yes", kpts=[2, 2, 2],
                slater_p=str(cu_skf_dir) + "/")
    assert isinstance(dftbplus_calculator(bulk("Cu"), args), _CheckedDftb)
    eng.configure(engine="pydftb")
    assert isinstance(dftbplus_calculator(bulk("Cu"), args), PyDFTB)
    assert isinstance(dftbplus_calculator(bulk("Cu"), dict(args, engine="dftbplus")), _CheckedDftb)


def test_cli_optimize_passes_engine_flags(tmp_path, monkeypatch):
    dataset = tmp_path / "my_data.xyz"
    dataset.write_text("0\n\n")
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_run_path(path, run_name=None):
        captured['argv'] = list(sys.argv)

    with patch('cli.cmd_optimize.runpy.run_path', side_effect=fake_run_path):
        from cli.main import dispatch
        rc = dispatch(['optimize', str(dataset), '--output', str(tmp_path / 'run'),
                       '--mode', 'dataset', '--engine', 'pydftb', '--smearing', 'mp',
                       '--smearing-temperature', '1500', '--mp-order', '1',
                       '--pydftb-forces', 'exact'])
    assert rc == 0
    argv = captured['argv']
    assert argv[argv.index('--engine') + 1] == 'pydftb'
    assert argv[argv.index('--smearing') + 1] == 'mp'
    assert float(argv[argv.index('--smearing_temperature') + 1]) == 1500.0
    assert argv[argv.index('--mp_order') + 1] == '1'
    assert argv[argv.index('--pydftb_forces') + 1] == 'exact'
    # the legacy parser accepts the forwarded flags
    from cli.arg_parser import build_default_arg_parser
    monkeypatch.setattr(sys, 'argv', argv)
    parsed = build_default_arg_parser()
    assert parsed.engine == 'pydftb' and parsed.smearing == 'mp' and parsed.mp_order == 1


def test_yaml_and_tui_know_the_engine():
    from cli import config
    from cli.tui.state import SetupState, assemble_optimize_yaml
    for key in ('engine', 'smearing', 'smearing_temperature', 'mp_order', 'pydftb_forces'):
        assert key in config.YAML_KEY_TO_ARG
    s = SetupState()
    s.engine = 'pydftb'
    assert assemble_optimize_yaml(s)['engine'] == 'pydftb'


@pytest.mark.skipif(not HAVE_DFTBPLUS, reason="dftb+ binary not available")
def test_engines_agree_on_ptbp_dataset(cu_skf_dir, monkeypatch):
    """Both engines on PTBP's real settings (dft/dft.xyz, PTBP k-grid)."""
    from ase.io import read
    from utils.calculator import dftbplus_calculator, monkhorstpack2kptdensity
    monkeypatch.chdir(cu_skf_dir)
    images = read(ROOT / "dft" / "dft.xyz", ":")
    for i in (0, 7):
        at = images[i]
        _, kpts = monkhorstpack2kptdensity(at, 5.0)
        args = dict(calc_type="static", poly_rep="Yes", SCC="Yes", kpts=kpts, slater_p="./")
        out = {}
        for engine in eng.ENGINES:
            eng.configure(engine=engine)
            a = at.copy()
            a.calc = dftbplus_calculator(a, args)
            out[engine] = (a.get_potential_energy(), a.get_forces())
        assert abs(out["pydftb"][0] - out["dftbplus"][0]) < 1e-6           # eV
        assert np.abs(out["pydftb"][1] - out["dftbplus"][1]).max() < 1e-4  # eV/A
