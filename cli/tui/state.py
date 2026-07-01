"""Shared state passed between wizard screens.

The Textual App holds a single SetupState instance; each screen reads
defaults from it on mount and writes user choices back on Next. Keeping
state in a plain dataclass (not on the App as reactive vars) makes
testing trivial — assemble_optimize_yaml() / assemble_gen_command() can
be exercised without touching Textual at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from cli.dataset_inspect import DatasetReport


@dataclass
class SetupState:
    """Mutable bag the wizard fills in step-by-step."""

    # Top-level choice
    flow: str = ''                       # 'optimize' or 'gen' (set by welcome)

    # ----- optimize flow -----
    dataset_path: Optional[Path] = None
    dataset_report: Optional[DatasetReport] = None
    mode: str = 'auto'                   # auto / dataset / reaction / ...
    optimizer: str = 'bo'                # bo / parallel_bo / pso
    n_calls: Optional[int] = None
    n_particles: int = 4
    e0s_strategy: str = 'auto'           # auto / cache / manual
    e0s_cache_path: Optional[Path] = None
    e0s_manual_json: str = ''
    xc: str = 'GGA_X_PBE+GGA_C_PBE'
    kpt_density: float = 5.0
    eos_points: int = 11
    parameters: List[str] = field(default_factory=lambda: ['r0_w', 'r0_d', 'sigma_rep'])
    multi_element: List[str] = field(default_factory=list)
    superposition: str = 'density'
    seed: Optional[int] = None            # random seed; None → run.py default (123)

    # ----- gen flow -----
    gen_symbols: List[str] = field(default_factory=list)
    gen_params: str = 'PTBP'             # PTBP / QNplusRep / Prior / Personal
    gen_skf_mode: str = 'full'           # full / band / rep
    gen_r0: List[float] = field(default_factory=list)   # for Personal
    gen_rep: List[float] = field(default_factory=list)  # for Personal
    gen_out: Optional[Path] = None
    gen_xc: str = 'GGA_X_PBE+GGA_C_PBE'

    # ----- output (where the wizard writes ptbp_run.yaml for optimize) -----
    output_yaml: Path = field(default_factory=lambda: Path.cwd() / 'ptbp_run.yaml')


def assemble_optimize_yaml(state: SetupState) -> dict:
    """Translate the optimize-flow state into a YAML-ready dict that's
    accepted by `cli.config.YAML_KEY_TO_ARG` (so `ptbp optimize --config
    <file>` consumes it directly)."""
    out: dict = {
        'mode':          state.mode,
        'optimizer':     state.optimizer,
        'n_particles':   state.n_particles,
        'eos_points':    state.eos_points,
        'xc':            state.xc,
        'kpt_density':   state.kpt_density,
        'parameters':    list(state.parameters),
        'superposition': state.superposition,
    }
    if state.dataset_path:
        out['dataset'] = str(state.dataset_path)
    if state.n_calls is not None:
        out['n_calls'] = state.n_calls
    if state.seed is not None:
        out['seed'] = state.seed
    if state.multi_element:
        out['multi_element'] = list(state.multi_element)

    # E0s: only emit when the user chose explicit / cache. 'auto' means
    # leave the key out so `ptbp optimize` runs its auto-fit + cache.
    if state.e0s_strategy == 'manual' and state.e0s_manual_json.strip():
        try:
            import json
            out['E0s'] = json.loads(state.e0s_manual_json)
        except Exception:
            out['E0s'] = state.e0s_manual_json   # leave string; ptbp will reparse
    elif state.e0s_strategy == 'cache' and state.e0s_cache_path:
        # Cache path lives under <ref_dir>/E0s_dft.json; auto-load happens
        # automatically when --E0s is omitted *and* the cache exists.
        # Nothing to write here — but warn user via a comment field.
        pass

    return out


def assemble_optimize_argv(state: SetupState) -> list[str]:
    """argv list (suitable for os.execvp / subprocess.run) for the
    optimize-flow command. Uses --config so the rest of the settings
    stay in YAML; the YAML must be written to disk before this is
    executed.
    """
    if not state.dataset_path:
        return ['ptbp', 'optimize', '<dataset>']
    return ['ptbp', 'optimize', str(state.dataset_path),
            '--config', str(state.output_yaml)]


def assemble_optimize_command(state: SetupState) -> str:
    """Human-displayable shell version of assemble_optimize_argv. Uses
    shlex.quote so paths with spaces remain copy-paste-safe."""
    import shlex
    return ' '.join(shlex.quote(a) for a in assemble_optimize_argv(state))


def assemble_gen_argv(state: SetupState) -> list[str]:
    """argv list for the gen-flow command. Gen is parameter-light enough
    that we don't bother writing a YAML — the argv carries everything."""
    parts: list[str] = ['ptbp', 'gen']
    parts += list(state.gen_symbols)
    parts += ['--params', state.gen_params]
    if state.gen_skf_mode != 'full':
        parts += ['--mode', state.gen_skf_mode]
    if state.gen_xc != 'GGA_X_PBE+GGA_C_PBE':
        parts += ['--xc', state.gen_xc]
    if state.gen_out:
        parts += ['--out', str(state.gen_out)]
    if state.gen_params == 'Personal':
        if state.gen_r0:
            parts += ['--r0'] + [str(v) for v in state.gen_r0]
        if state.gen_rep:
            parts += ['--rep'] + [str(v) for v in state.gen_rep]
    return parts


def assemble_gen_command(state: SetupState) -> str:
    import shlex
    return ' '.join(shlex.quote(a) for a in assemble_gen_argv(state))
