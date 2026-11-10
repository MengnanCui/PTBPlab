"""Tests for gui.core.gui_state (command/YAML assembly + label maps)."""
import yaml

from gui.core import gui_state as gs


def test_new_states_have_flow():
    assert gs.new_optimize_state().flow == "optimize"
    assert gs.new_gen_state().flow == "gen"


def test_target_and_optimizer_maps_roundtrip():
    for label, mode in gs.TARGET_TO_MODE.items():
        assert gs.MODE_TO_TARGET[mode] == label
    for label, val in gs.OPTIMIZER_TO_VALUE.items():
        assert gs.VALUE_TO_OPTIMIZER[val] == label


def test_optimize_yaml_and_command():
    s = gs.new_optimize_state()
    s.dataset_path = "dft/dft.xyz"
    s.mode = "dataset"
    s.optimizer = "pso"
    s.n_particles = 4
    s.n_calls = 12
    yaml_text, cmd = gs.optimize_preview(s)
    d = yaml.safe_load(yaml_text)
    assert d["mode"] == "dataset"
    assert d["optimizer"] == "pso"
    assert d["n_calls"] == 12
    assert d["dataset"] == "dft/dft.xyz"
    assert "ptbp optimize" in cmd
    assert "--config" in cmd


def test_seed_appears_in_yaml_only_when_set():
    s = gs.new_optimize_state()
    s.dataset_path = "dft/dft.xyz"
    assert "seed" not in yaml.safe_load(gs.optimize_yaml_text(s))
    s.seed = 42
    assert yaml.safe_load(gs.optimize_yaml_text(s))["seed"] == 42


def test_validate_optimize_flags_missing_dataset():
    s = gs.new_optimize_state()
    problems = gs.validate_optimize(s)
    assert any("dataset" in p.lower() for p in problems)


def test_validate_optimize_particles_guard():
    s = gs.new_optimize_state()
    s.dataset_path = "dft/dft.xyz"
    s.optimizer = "pso"
    s.n_particles = 1
    assert any("n_particles" in p for p in gs.validate_optimize(s))


def test_gen_preview_and_validation():
    s = gs.new_gen_state()
    s.gen_symbols = ["H", "O"]
    s.gen_params = "PTBP"
    cmd = gs.gen_preview(s)
    assert cmd.startswith("ptbp gen")
    assert "H" in cmd and "O" in cmd
    assert gs.validate_gen(s) == []


def test_validate_gen_personal_requires_r0():
    s = gs.new_gen_state()
    s.gen_symbols = ["H", "O"]
    s.gen_params = "Personal"
    s.gen_r0 = [3.0, 5.0, 2.0]  # only 3, need 6
    assert any("Personal" in p for p in gs.validate_gen(s))
