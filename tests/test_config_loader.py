"""YAML config + CLI-override merge tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest


def test_load_yaml_basic(tmp_path):
    from cli import config
    p = tmp_path / "run.yaml"
    p.write_text(
        "mode: dataset\n"
        "n_calls: 24\n"
        "optimizer: pso\n"
        "E0s:\n"
        "  6: -37.8\n"
        "  1: -0.5\n"
    )
    out = config.load_yaml(p)
    assert out['mode'] == 'dataset'
    assert out['n_calls'] == 24
    assert out['optimizer'] == 'pso'
    assert out['E0s'] == {6: -37.8, 1: -0.5}


def test_load_yaml_missing_returns_friendly_error(tmp_path):
    from cli import config
    with pytest.raises(FileNotFoundError):
        config.load_yaml(tmp_path / "nonexistent.yaml")


def test_load_yaml_non_mapping_raises(tmp_path):
    from cli import config
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        config.load_yaml(p)


def _make_parser():
    """Mini parser whose defaults exactly mirror cmd_optimize for the keys
    we care about."""
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='dataset')
    p.add_argument('--optimizer', default='bo')
    p.add_argument('--n-calls', dest='n_calls', type=int, default=None)
    p.add_argument('--n-particles', dest='n_particles', type=int, default=4)
    p.add_argument('--E0s', default=None)
    p.add_argument('--params', dest='parameters', nargs='+',
                   default=['r0_w', 'r0_d', 'sigma_rep'])
    return p


def test_yaml_fills_in_when_cli_at_default():
    """Both --mode (default 'dataset') and --optimizer (default 'bo') are
    left at default → YAML's values should fill in."""
    from cli import config
    parser = _make_parser()
    args = parser.parse_args([])
    yaml_dict = {'mode': 'reaction', 'optimizer': 'pso', 'n_calls': 30}
    defaults = config.get_parser_defaults(parser)
    config.merge_into_namespace(args, yaml_dict, defaults)
    assert args.mode == 'reaction'
    assert args.optimizer == 'pso'
    assert args.n_calls == 30


def test_cli_explicit_overrides_yaml():
    """User explicitly passes a non-default --optimizer parallel_bo on the
    CLI; the YAML's 'pso' must not win.

    Caveat (documented): CLI values that *match* the parser's default are
    indistinguishable from "user didn't pass" with stock argparse, so the
    YAML wins in that case. To force a value over YAML, pass a non-default
    value or remove the YAML entry.
    """
    from cli import config
    parser = _make_parser()
    args = parser.parse_args(['--optimizer', 'parallel_bo', '--n-calls', '7'])
    yaml_dict = {'optimizer': 'pso', 'n_calls': 30}
    defaults = config.get_parser_defaults(parser)
    config.merge_into_namespace(args, yaml_dict, defaults)
    assert args.optimizer == 'parallel_bo'  # CLI wins
    assert args.n_calls == 7                 # CLI wins (default was None)


def test_yaml_E0s_dict_serialised_to_json_string():
    """YAML provides a native dict; merge_into_namespace converts it to the
    JSON string that the legacy --E0s flag expects."""
    from cli import config
    import json
    parser = _make_parser()
    args = parser.parse_args([])
    yaml_dict = {'E0s': {6: -37.8, 1: -0.5}}
    config.merge_into_namespace(args, yaml_dict, config.get_parser_defaults(parser))
    assert isinstance(args.E0s, str)
    parsed = json.loads(args.E0s)
    assert parsed == {'6': -37.8, '1': -0.5}


def test_unknown_yaml_keys_warn_not_error(tmp_path, capsys):
    from cli import config
    parser = _make_parser()
    args = parser.parse_args([])
    yaml_dict = {'mode': 'dataset', 'sneaky_unknown_key': 42}
    config.merge_into_namespace(args, yaml_dict, config.get_parser_defaults(parser))
    err = capsys.readouterr().err
    assert 'sneaky_unknown_key' in err
    assert args.mode == 'dataset'  # known keys still applied


def test_write_effective_roundtrip(tmp_path):
    """Writing then reading back should give the same dict (modulo type
    coercions for E0s)."""
    from cli import config
    args = argparse.Namespace(
        dataset='/tmp/data.xyz',
        mode='dataset',
        output=str(tmp_path / 'run'),
        optimizer='pso',
        n_calls=24,
        n_particles=4,
        E0s='{"6": -37.8}',
        eos_points=11,
        xc='GGA_X_PBE+GGA_C_PBE',
        kpt_density=5.0,
        parameters=['r0_w', 'r0_d', 'sigma_rep'],
        multi_element=None,
        superposition='density',
        checkpoint='checkpoint.pkl',
    )
    target_dir = tmp_path / 'run'
    target_dir.mkdir()
    config.write_effective(args, target_dir)
    snapshot = config.load_yaml(target_dir / 'ptbp_run.yaml')
    assert snapshot['mode'] == 'dataset'
    assert snapshot['n_calls'] == 24
    # E0s gets normalised to a dict in the snapshot (more readable than JSON
    # string when humans edit the file).
    assert snapshot['E0s'] == {6: -37.8}
