"""SetupState assembler tests — no Textual interaction needed."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_assemble_optimize_yaml_minimum():
    from cli.tui.state import SetupState, assemble_optimize_yaml
    s = SetupState()
    s.dataset_path = Path("/data/x.xyz")
    s.mode = 'dataset'
    s.optimizer = 'bo'

    out = assemble_optimize_yaml(s)
    assert out['mode'] == 'dataset'
    assert out['optimizer'] == 'bo'
    assert out['dataset'] == '/data/x.xyz'
    # n_calls is optional and unset (None) — must NOT appear in YAML.
    assert 'n_calls' not in out
    # Default parameters are present.
    assert out['parameters'] == ['r0_w', 'r0_d', 'sigma_rep']
    assert out['superposition'] == 'density'


def test_assemble_optimize_yaml_with_e0s_manual_dict():
    from cli.tui.state import SetupState, assemble_optimize_yaml
    s = SetupState()
    s.dataset_path = Path("/data/x.xyz")
    s.e0s_strategy = 'manual'
    s.e0s_manual_json = '{"6": -37.8, "1": -0.5}'

    out = assemble_optimize_yaml(s)
    assert out['E0s'] == {"6": -37.8, "1": -0.5}


def test_assemble_optimize_yaml_e0s_auto_omits_key():
    """Auto strategy: ptbp optimize handles E0s itself, so YAML should
    NOT contain an E0s key."""
    from cli.tui.state import SetupState, assemble_optimize_yaml
    s = SetupState()
    s.dataset_path = Path("/data/x.xyz")
    s.e0s_strategy = 'auto'
    out = assemble_optimize_yaml(s)
    assert 'E0s' not in out


def test_assemble_optimize_command_uses_config():
    from cli.tui.state import SetupState, assemble_optimize_command
    s = SetupState()
    s.dataset_path = Path("/data/x.xyz")
    s.output_yaml = Path("/work/run.yaml")
    cmd = assemble_optimize_command(s)
    assert cmd == "ptbp optimize /data/x.xyz --config /work/run.yaml"


def test_assemble_gen_command_minimum():
    from cli.tui.state import SetupState, assemble_gen_command
    s = SetupState()
    s.gen_symbols = ['H', 'O']
    s.gen_params = 'PTBP'
    cmd = assemble_gen_command(s)
    assert cmd == "ptbp gen H O --params PTBP"


def test_assemble_gen_command_with_personal_values():
    from cli.tui.state import SetupState, assemble_gen_command
    s = SetupState()
    s.gen_symbols = ['H']
    s.gen_params = 'Personal'
    s.gen_r0 = [2.0, 5.0, 2.0]
    s.gen_rep = [0.6, 0.0]
    s.gen_out = Path('./skfs')
    cmd = assemble_gen_command(s)
    assert cmd.startswith("ptbp gen H --params Personal")
    assert "--out skfs" in cmd
    assert "--r0 2.0 5.0 2.0" in cmd
    assert "--rep 0.6 0.0" in cmd


def test_assemble_gen_command_skf_mode_and_xc():
    from cli.tui.state import SetupState, assemble_gen_command
    s = SetupState()
    s.gen_symbols = ['H', 'O']
    s.gen_params = 'PTBP'
    s.gen_skf_mode = 'band'
    s.gen_xc = 'LDA_X+LDA_C_PW'
    cmd = assemble_gen_command(s)
    assert "--mode band" in cmd
    assert "--xc LDA_X+LDA_C_PW" in cmd


def test_setup_state_defaults_round_trip():
    """Default-constructed SetupState assembles a sensible YAML."""
    from cli.tui.state import SetupState, assemble_optimize_yaml
    s = SetupState()
    s.dataset_path = Path("/x")
    out = assemble_optimize_yaml(s)
    # Defaults that should always be present
    for key in ('mode', 'optimizer', 'n_particles', 'eos_points', 'xc',
                'kpt_density', 'parameters', 'superposition'):
        assert key in out, f"default key {key!r} missing from YAML"


def test_assemble_optimize_argv_pairs_with_command_string():
    """argv list and command string must encode the same invocation."""
    import shlex
    from cli.tui.state import (SetupState, assemble_optimize_argv,
                                assemble_optimize_command)
    s = SetupState()
    s.dataset_path = Path("/data with space/x.xyz")
    s.output_yaml = Path("/work/run.yaml")

    argv = assemble_optimize_argv(s)
    cmd  = assemble_optimize_command(s)
    # The command string is the shell-escaped form of argv.
    assert ' '.join(shlex.quote(a) for a in argv) == cmd
    assert argv[0] == 'ptbp'
    assert argv[1] == 'optimize'
    assert argv[2] == '/data with space/x.xyz'
    assert '--config' in argv
    assert argv[argv.index('--config') + 1] == '/work/run.yaml'


def test_assemble_gen_argv_personal_with_r0_and_rep():
    from cli.tui.state import (SetupState, assemble_gen_argv,
                                assemble_gen_command)
    s = SetupState()
    s.gen_symbols = ['H', 'O']
    s.gen_params = 'Personal'
    s.gen_r0 = [2.0, 5.0, 2.0, 3.0, 5.0, 2.0]
    s.gen_rep = [0.6, 0.0, 0.6, 0.0]
    argv = assemble_gen_argv(s)
    assert argv[:4] == ['ptbp', 'gen', 'H', 'O']
    assert '--r0' in argv
    r0_idx = argv.index('--r0')
    assert argv[r0_idx + 1: r0_idx + 7] == ['2.0', '5.0', '2.0', '3.0', '5.0', '2.0']
    assert '--rep' in argv
    rep_idx = argv.index('--rep')
    assert argv[rep_idx + 1: rep_idx + 5] == ['0.6', '0.0', '0.6', '0.0']

    # Command string is the shlex-quoted argv.
    import shlex
    assert ' '.join(shlex.quote(a) for a in argv) == assemble_gen_command(s)
