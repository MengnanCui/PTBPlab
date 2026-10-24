"""The ``pydftb`` command (P1): dispatch, --help, --version, errors, argument pass-through."""
import importlib
import inspect
import os
import subprocess
import sys

import pytest

import pydftb
from pydftb import cli

from .conftest import DATA

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _exit_code(argv) -> int:
    with pytest.raises(SystemExit) as info:
        cli.main(argv)
    return info.value.code


def test_help_lists_every_command(capsys):
    for argv in (["--help"], ["-h"], [], ["help"]):
        assert _exit_code(argv) == 0
        out = capsys.readouterr().out
        assert out.startswith("usage: pydftb")
        for name in [*cli.COMMANDS, *cli.ALIASES]:
            assert name in out, (argv, name)


def test_version(capsys):
    assert _exit_code(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"pydftb {pydftb.__version__}"


@pytest.mark.parametrize("argv", [["refgenx"], ["--bogus"], ["--", "refgen"], ["--"]])
def test_usage_errors(argv, capsys):
    assert _exit_code(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: pydftb") and "error:" in err


def test_unknown_command_names_the_choices(capsys):
    assert _exit_code(["refgenx"]) == 2
    err = capsys.readouterr().err
    assert "invalid choice: 'refgenx'" in err and "gpu-check" in err


@pytest.mark.parametrize("name", sorted(cli.COMMANDS))
def test_every_command_takes_argv_and_prog(name):
    module = cli.COMMANDS[name][0]
    if name == "gradcheck":
        pytest.importorskip("torch")                 # imported at module level
    main = importlib.import_module(module).main
    assert list(inspect.signature(main).parameters)[:2] == ["argv", "prog"]


@pytest.mark.parametrize("name", sorted(cli.COMMANDS))
def test_command_help_uses_the_command_name(name, capsys):
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if name == "gradcheck":                      # needs torch already to import
            assert cli.main([name, "--help"]) == 2
            assert "pip install 'pydftb[torch]'" in capsys.readouterr().err
            return
    assert _exit_code([name, "--help"]) == 0
    assert capsys.readouterr().out.startswith(f"usage: pydftb {name}")


def test_help_subcommand(capsys):
    assert _exit_code(["help", "dynamics"]) == 0
    assert capsys.readouterr().out.startswith("usage: pydftb dynamics")


def _fake_main(monkeypatch, module, result=None, raises=None):
    calls = []

    def main(argv=None, prog=None):
        calls.append((argv, prog))
        if raises is not None:
            raise raises
        return result

    monkeypatch.setattr(importlib.import_module(module), "main", main)
    return calls


def test_arguments_reach_the_tool_unchanged(monkeypatch):
    calls = _fake_main(monkeypatch, "pydftb.validation.refgen", result=7)
    argv = ["--check", "h2o", "--help", "--", "-x", "pydftb"]
    assert cli.main(["refgen", *argv]) == 7
    assert calls == [(argv, "pydftb refgen")]


def test_alias_and_none_result(monkeypatch):
    calls = _fake_main(monkeypatch, "pydftb.validation.gpu_check", result=None)
    assert cli.main(["gpu_check", "--quick"]) == 0     # refgen/dynamics/gradcheck return None
    assert calls == [(["--quick"], "pydftb gpu-check")]


def test_system_exit_passes_through(monkeypatch):
    _fake_main(monkeypatch, "pydftb.validation.golden", raises=SystemExit(3))
    assert _exit_code(["golden", "--base", "HEAD"]) == 3


def test_missing_torch_is_a_short_message(monkeypatch, capsys):
    missing = ModuleNotFoundError("No module named 'torch'", name="torch")
    _fake_main(monkeypatch, "pydftb.validation.gpu_check", raises=missing)
    assert cli.main(["gpu-check"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("pydftb gpu-check: needs PyTorch") and "Traceback" not in err
    other = ModuleNotFoundError("No module named 'hotcent'", name="hotcent")
    _fake_main(monkeypatch, "pydftb.validation.gpu_check", raises=other)
    with pytest.raises(ModuleNotFoundError, match="hotcent"):
        cli.main(["gpu-check"])


def test_advisor_through_the_command(capsys):
    """A real tool behind the command: exit status and output as with python -m."""
    assert cli.main(["advisor", os.path.join(DATA, "advisor", "klines")]) == 1
    assert "[B01] error" in capsys.readouterr().out
    assert cli.main(["advisor", "--explain", "p03"]) == 0
    assert "IgnoreUnprocessedNodes" in capsys.readouterr().out


def test_python_m_pydftb():
    env = dict(os.environ, PYTHONPATH=ROOT)
    run = subprocess.run([sys.executable, "-m", "pydftb", "--version"], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0 and run.stdout.strip() == f"pydftb {pydftb.__version__}"
    run = subprocess.run([sys.executable, "-m", "pydftb", "advisor", "--list-rules"], cwd=ROOT,
                         env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0 and run.stdout.startswith("P01")
