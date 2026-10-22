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


def test_pydftb_backend_options(cu_skf_dir):
    """M4d: the torch backend reaches PyDFTB; the default calls are unchanged."""
    pytest.importorskip("torch")
    args = dict(calc_type="static", poly_rep="Yes", SCC="Yes", kpts=[2, 2, 2],
                slater_p=str(cu_skf_dir) + "/", k_threads=1)
    eng.configure(engine="pydftb")
    calc = eng.make_calculator(bulk("Cu"), args)
    assert calc.engine_opts["backend"] is None and not eng.is_gpu()
    eng.configure(pydftb_backend="torch", pydftb_dtype="float32")
    calc = eng.make_calculator(bulk("Cu"), args)
    assert (calc.engine_opts["backend"], calc.engine_opts["device"],
            calc.engine_opts["dtype"]) == ("torch", "cpu", "float32")
    assert not eng.is_gpu()
    eng.configure(pydftb_device="cuda:1")
    assert eng.is_gpu()                         # the DFTB stage then runs in this process


@pytest.mark.parametrize("bad, match", [
    (dict(engine="pydftb", pydftb_device="cuda"), "need --pydftb_backend torch"),
    (dict(engine="pydftb", pydftb_device="auto"), "need --pydftb_backend torch"),
    (dict(engine="pydftb", pydftb_dtype="float32"), "need --pydftb_backend torch"),
    (dict(engine="dftbplus", pydftb_backend="torch"), "needs --engine pydftb"),
    (dict(engine="pydftb", pydftb_backend="torch", pydftb_device="tpu"), "unknown device"),
    (dict(engine="pydftb", pydftb_backend="torch", pydftb_device="mps"), "no float64"),
    (dict(engine="pydftb", pydftb_backend="jax"), "pydftb_backend must be"),
])
def test_pydftb_backend_options_are_checked(bad, match):
    with pytest.raises(ValueError, match=match):
        eng.configure(**bad)


def test_gpu_stage_runs_in_this_process(monkeypatch):
    from utils import dftb_stage
    from utils.parallel import CpuBudget, set_plan, simple_plan
    set_plan(simple_plan(CpuBudget(4, {"--ncores": 4}), 8, "pydftb"))
    try:
        assert dftb_stage._stage_settings()[0] == 4
        monkeypatch.setattr(eng, "is_gpu", lambda: True)
        assert dftb_stage._stage_settings() == (1, 1, 1, True)
    finally:
        set_plan(None)


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
    assert '--pydftb_backend' not in argv                  # the default adds nothing (M4d)
    assert (parsed.pydftb_backend, parsed.device, parsed.dtype) == ('numpy', 'cpu', 'float64')
    with patch('cli.cmd_optimize.runpy.run_path', side_effect=fake_run_path):
        rc = dispatch(['optimize', str(dataset), '--output', str(tmp_path / 'run2'),
                       '--mode', 'dataset', '--engine', 'pydftb', '--pydftb-backend', 'torch',
                       '--device', 'cuda:1', '--dtype', 'float32'])
    assert rc == 0
    monkeypatch.setattr(sys, 'argv', captured['argv'])
    parsed = build_default_arg_parser()
    assert (parsed.pydftb_backend, parsed.device, parsed.dtype) == ('torch', 'cuda:1', 'float32')


def test_yaml_and_tui_know_the_engine():
    from cli import config
    from cli.tui.state import SetupState, assemble_optimize_yaml
    for key in ('engine', 'smearing', 'smearing_temperature', 'mp_order', 'pydftb_forces',
                'pydftb_backend', 'device', 'dtype'):
        assert key in config.YAML_KEY_TO_ARG
    s = SetupState()
    s.engine = 'pydftb'
    out = assemble_optimize_yaml(s)
    assert out['engine'] == 'pydftb' and 'pydftb_backend' not in out
    s.pydftb_device = 'cuda'
    out = assemble_optimize_yaml(s)
    assert out['pydftb_backend'] == 'torch' and out['device'] == 'cuda' and 'dtype' not in out
    s.pydftb_device = 'mps'
    assert assemble_optimize_yaml(s)['dtype'] == 'float32'


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
