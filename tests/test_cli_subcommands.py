"""Subcommand dispatch tests.

These tests poke the new CLI surface without actually executing the
expensive cli/run.py legacy entry point — we verify the argparse layer
parses correctly and the translation to legacy sys.argv is sane.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


def test_top_parser_lists_three_subcommands():
    from cli.main import _build_top_parser, SUBCOMMANDS
    p = _build_top_parser()
    # Help text mentions all three subcommands.
    help_text = p.format_help()
    for sc in SUBCOMMANDS:
        assert sc in help_text, f"top-level help missing subcommand {sc!r}"


def test_gen_args_translate_to_legacy_skf_generator():
    """`ptbp gen H O --params PTBP` translates to the legacy
    `--skf_generator full --symbols H O --known_parameters PTBP` flag set."""
    from cli import cmd_gen

    captured = {}

    def fake_run_path(path, run_name=None):
        captured['argv'] = list(sys.argv)

    with patch('cli.cmd_gen.runpy.run_path', side_effect=fake_run_path):
        from cli.main import dispatch
        rc = dispatch(['gen', 'H', 'O', '--params', 'PTBP'])

    assert rc == 0
    argv = captured['argv']
    assert argv[0] == 'run.py'
    assert '--skf_generator' in argv
    assert argv[argv.index('--skf_generator') + 1] == 'full'
    assert '--symbols' in argv
    sym_idx = argv.index('--symbols')
    assert argv[sym_idx + 1: sym_idx + 3] == ['H', 'O']
    assert '--known_parameters' in argv
    assert argv[argv.index('--known_parameters') + 1] == 'PTBP'


def test_optimize_args_translate_to_legacy_optimization_option(tmp_path, monkeypatch):
    """`ptbp optimize my_data.xyz --mode dataset --n-calls 4 --E0s '{...}'`
    translates correctly."""
    # Make a fake dataset file so cmd_optimize doesn't reject the path.
    dataset = tmp_path / "my_data.xyz"
    dataset.write_text("0\n\n")  # technically invalid xyz, but resolve() check passes
    output = tmp_path / "run_test"
    monkeypatch.chdir(tmp_path)  # so any default ./run_<ts>/ lands in tmp

    captured = {}

    def fake_run_path(path, run_name=None):
        captured['argv'] = list(sys.argv)

    with patch('cli.cmd_optimize.runpy.run_path', side_effect=fake_run_path):
        from cli.main import dispatch
        rc = dispatch([
            'optimize', str(dataset),
            '--output', str(output),
            '--mode', 'dataset',
            '--n-calls', '4',
            '--E0s', '{"6": -37.8}',
            '--optimizer', 'pso',
            '--n-particles', '3',
        ])

    assert rc == 0
    assert output.exists() and output.is_dir(), \
        "--output dir should be created"
    assert (output / 'dft').exists(), \
        "--output dir should contain a dft/ symlink to the dataset's parent"
    argv = captured['argv']
    # ref_dir + dft_file split
    assert '--ref_dir' in argv
    assert argv[argv.index('--ref_dir') + 1] == str(tmp_path)
    assert argv[argv.index('--dft_file') + 1] == 'my_data.xyz'
    # mode
    assert argv[argv.index('--optimization_option') + 1] == 'dataset'
    # numeric flags
    assert argv[argv.index('--n_calls') + 1] == '4'
    assert argv[argv.index('--n_particles') + 1] == '3'
    # optimizer
    assert argv[argv.index('--optimizer') + 1] == 'pso'
    # E0s passed through verbatim
    assert argv[argv.index('--E0s') + 1] == '{"6": -37.8}'


def test_optimize_missing_dataset_returns_error_code(tmp_path):
    from cli.main import dispatch
    rc = dispatch(['optimize', str(tmp_path / 'does_not_exist.xyz')])
    assert rc == 2


def test_postprocess_missing_run_dir_returns_error_code(tmp_path):
    from cli.main import dispatch
    rc = dispatch(['postprocess', str(tmp_path / 'no_such_dir')])
    assert rc == 2


def test_postprocess_run_dir_without_result_pkl_errors(tmp_path):
    from cli.main import dispatch
    (tmp_path / 'dft').mkdir()
    rc = dispatch(['postprocess', str(tmp_path)])
    assert rc == 2  # result.pkl missing


def test_parameterize_main_routes_legacy_to_run_py():
    """Anything that's not a known subcommand should fall through to the
    legacy `cli/run.py` runner."""
    import parameterize

    routed = {'legacy': False, 'new': False}

    def fake_run_path(path, run_name=None):
        routed['legacy'] = True

    def fake_dispatch(argv):
        routed['new'] = True
        return 0

    with patch('parameterize.runpy.run_path', side_effect=fake_run_path), \
         patch('cli.main.dispatch', side_effect=fake_dispatch):
        with patch.object(sys, 'argv', ['ptbp', '--skf_generator', 'full',
                                         '--symbols', 'H',
                                         '--known_parameters', 'PTBP']):
            parameterize.main()

    assert routed['legacy'], "legacy CLI fell through path was not taken"
    assert not routed['new'], "should not have hit the new dispatcher"


def test_parameterize_main_routes_subcommand_to_dispatch():
    import parameterize

    routed = {'legacy': False, 'new': False, 'argv': None}

    def fake_run_path(path, run_name=None):
        routed['legacy'] = True

    def fake_dispatch(argv):
        routed['new'] = True
        routed['argv'] = list(argv)
        return 0

    with patch('parameterize.runpy.run_path', side_effect=fake_run_path), \
         patch('cli.main.dispatch', side_effect=fake_dispatch):
        with patch.object(sys, 'argv', ['ptbp', 'gen', 'H', 'O']):
            parameterize.main()

    assert routed['new'], "new dispatcher was not invoked"
    assert not routed['legacy']
    assert routed['argv'] == ['gen', 'H', 'O']
