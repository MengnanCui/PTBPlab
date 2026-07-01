"""YAML config support for `ptbp optimize`.

Layered semantics: CLI flags override YAML values. The merged config is
written back to `<run_dir>/ptbp_run.yaml` after every successful run for
reproducibility.

Supported YAML keys (all optional; correspond to `ptbp optimize` flags):

    dataset:        # path string
    mode:           # energygeometry / dataset / reaction / bandstructure
    output:         # run-folder path
    optimizer:      # bo / parallel_bo / pso
    n_calls:        # int
    n_particles:    # int
    E0s:            # dict {atomic_number: eV} or string '{"6": -37.8}'
    eos_points:     # int
    xc:             # str
    kpt_density:    # float
    parameters:     # list[str]
    multi_element:  # list[str]
    superposition:  # density / potential
    checkpoint:     # str
    seed:           # int (random seed for reproducibility)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional


# Map: YAML key  →  argparse Namespace attribute.
# Almost all keys are 1:1, but `kpts` ↔ `kpt_density` etc. are flagged.
YAML_KEY_TO_ARG: Dict[str, str] = {
    'dataset':       'dataset',
    'mode':          'mode',
    'output':        'output',
    'optimizer':     'optimizer',
    'n_calls':       'n_calls',
    'n_particles':   'n_particles',
    'E0s':           'E0s',
    'eos_points':    'eos_points',
    'xc':            'xc',
    'kpt_density':   'kpt_density',
    'parameters':    'parameters',
    'multi_element': 'multi_element',
    'superposition': 'superposition',
    'checkpoint':    'checkpoint',
    'seed':          'seed',
}


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Read YAML file → dict. Returns {} if path is empty/None."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    import yaml
    with open(p) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must be a YAML mapping at top level, got "
                         f"{type(data).__name__} from {p}")
    return data


def merge_into_namespace(
    args: argparse.Namespace,
    yaml_dict: Dict[str, Any],
    parser_defaults: Dict[str, Any],
) -> None:
    """Merge yaml_dict into args in place, with CLI overrides taking
    priority.

    Rule: a CLI value is considered explicit (override) iff it differs from
    the parser's default for that flag. Otherwise the YAML value wins.
    """
    # Warn about YAML keys we don't know how to map.
    for k in yaml_dict:
        if k not in YAML_KEY_TO_ARG:
            print(f"[config] warning: unknown key {k!r} in YAML, skipping",
                  file=sys.stderr)

    for yaml_key, attr in YAML_KEY_TO_ARG.items():
        if yaml_key not in yaml_dict:
            continue
        if not hasattr(args, attr):
            continue
        cli_value = getattr(args, attr)
        default = parser_defaults.get(attr)
        # If CLI matches default, treat as 'not specified' → take YAML.
        if cli_value == default:
            yaml_value = yaml_dict[yaml_key]
            # E0s arrives as JSON string from CLI; if YAML provides a dict,
            # turn it into the same JSON string the legacy parser expects.
            if attr == 'E0s' and isinstance(yaml_value, dict):
                import json
                yaml_value = json.dumps({str(k): v for k, v in yaml_value.items()})
            setattr(args, attr, yaml_value)


def write_effective(args: argparse.Namespace, run_dir: Path) -> None:
    """Snapshot the resolved settings to `<run_dir>/ptbp_run.yaml`.

    Writing this every successful run lets the user re-run with `--config
    ptbp_run.yaml` and gives `ptbp postprocess RUN_DIR` everything it
    needs without extra CLI flags.
    """
    import yaml
    out = {}
    for yaml_key, attr in YAML_KEY_TO_ARG.items():
        if hasattr(args, attr):
            v = getattr(args, attr)
            if v is None:
                continue
            # E0s: prefer dict in YAML for human readability.
            if yaml_key == 'E0s' and isinstance(v, str) and v.lstrip().startswith('{'):
                import json
                try:
                    v = {int(k): float(val) for k, val in json.loads(v).items()}
                except (ValueError, TypeError):
                    pass
            out[yaml_key] = v
    target = run_dir / 'ptbp_run.yaml'
    with open(target, 'w') as fh:
        yaml.safe_dump(out, fh, default_flow_style=False, sort_keys=True)


def get_parser_defaults(parser: argparse.ArgumentParser) -> Dict[str, Any]:
    """Return {arg_attr: default_value} for every action in the parser."""
    return {a.dest: a.default for a in parser._actions if a.dest != 'help'}
