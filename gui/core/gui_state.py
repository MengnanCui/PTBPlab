"""Bridge between the GUI forms and the existing `cli.tui.state.SetupState`.

The GUI is a graphical skin over `SetupState`: the optimize / generate views
bind their widgets to a `SetupState` instance, then this module reuses the
already-tested `assemble_*` helpers to produce the YAML + shell command. No
command-building logic is duplicated here — only the user-facing label maps
(e.g. "Band gap" → mode `bandstructure`) and a couple of preview helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from cli.tui.state import (
    SetupState,
    assemble_gen_command,
    assemble_optimize_command,
    assemble_optimize_yaml,
)

# ---- user-facing option maps (label shown in the UI → value the CLI wants) ----

# Optimisation target == the loss/mode. Kept human-readable for the combo box.
TARGET_TO_MODE: Dict[str, str] = {
    "Auto-detect": "auto",
    "Energy + Geometry (EOS)": "energygeometry",
    "Energy + Forces (dataset)": "dataset",
    "Reaction (multi-formula)": "reaction",
    "Band gap (bandstructure)": "bandstructure",
}
MODE_TO_TARGET: Dict[str, str] = {v: k for k, v in TARGET_TO_MODE.items()}

# Outer-loop optimiser.
OPTIMIZER_TO_VALUE: Dict[str, str] = {
    "Bayesian — GP (bo)": "bo",
    "Parallel BO (parallel_bo)": "parallel_bo",
    "Particle Swarm (pso)": "pso",
}
VALUE_TO_OPTIMIZER: Dict[str, str] = {v: k for k, v in OPTIMIZER_TO_VALUE.items()}

SUPERPOSITIONS: List[str] = ["density", "potential"]
OPTIMIZABLE_PARAMS: List[str] = ["r0_w", "r0_d", "sigma_rep"]
E0S_STRATEGIES: List[str] = ["auto", "cache", "manual"]

# Gen flow.
PARAM_SETS: List[str] = ["PTBP", "QNplusRep", "QUASINANO2013", "Prior", "Personal"]
SKF_MODES: List[str] = ["full", "band", "rep"]

DEFAULT_XC = "GGA_X_PBE+GGA_C_PBE"


def new_optimize_state() -> SetupState:
    """A fresh SetupState primed for the optimize flow."""
    s = SetupState()
    s.flow = "optimize"
    return s


def new_gen_state() -> SetupState:
    """A fresh SetupState primed for the gen flow."""
    s = SetupState()
    s.flow = "gen"
    return s


def optimize_yaml_text(state: SetupState) -> str:
    """Render the effective optimize YAML exactly as `ptbp optimize --config`
    would consume it."""
    import yaml

    return yaml.safe_dump(
        assemble_optimize_yaml(state), default_flow_style=False, sort_keys=True
    )


def optimize_preview(state: SetupState) -> Tuple[str, str]:
    """Return (yaml_text, shell_command) for the optimize review panel."""
    return optimize_yaml_text(state), assemble_optimize_command(state)


def gen_preview(state: SetupState) -> str:
    """Return the shell command for the gen review panel."""
    return assemble_gen_command(state)


def write_optimize_yaml(state: SetupState, path: str | Path) -> Path:
    """Persist the optimize YAML to disk so the Run button can pass
    `--config <path>`."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(optimize_yaml_text(state))
    return out


def validate_optimize(state: SetupState) -> List[str]:
    """Cheap, non-fatal pre-flight checks surfaced as UI warnings."""
    problems: List[str] = []
    if not state.dataset_path:
        problems.append("No dataset selected.")
    elif not Path(state.dataset_path).exists():
        problems.append(f"Dataset not found: {state.dataset_path}")
    if not state.parameters:
        problems.append("No parameters selected to optimise.")
    if state.optimizer in ("pso", "parallel_bo") and state.n_particles < 2:
        problems.append("n_particles must be >= 2 for pso / parallel_bo.")
    if state.e0s_strategy == "manual" and not state.e0s_manual_json.strip():
        problems.append("E0s strategy is 'manual' but no JSON was provided.")
    return problems


def validate_gen(state: SetupState) -> List[str]:
    """Pre-flight checks for the gen flow."""
    problems: List[str] = []
    if not state.gen_symbols:
        problems.append("No element symbols given.")
    if state.gen_params == "Personal":
        need = 3 * len(state.gen_symbols)
        if len(state.gen_r0) != need:
            problems.append(
                f"Personal params need {need} r0 values "
                f"(3 per symbol: r0_w r0_d p), got {len(state.gen_r0)}."
            )
    return problems
