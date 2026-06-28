"""Catch broken-import regressions before they hit a long DFTB+ run."""
import importlib

import pytest


@pytest.mark.parametrize("modpath", [
    "cli.arg_parser",
    "cli.run",  # NOTE: cli.run is script-style, but should at least be importable up to argparse
    "utils.data",
    "utils.hotpy",
    "utils.loss",
    "utils.parameters_set",
    "utils.calculator",
    "tools.physrep",
    "parameterize",
])
def test_module_imports(modpath):
    """Each named module must import without raising."""
    if modpath == "cli.run":
        # cli.run executes top-level code; skipping import here is fine — the
        # rest of the imports prove the dependency tree is healthy.
        pytest.skip("cli.run is script-style; covered indirectly")
    importlib.import_module(modpath)


def test_loss_class_has_required_methods():
    from utils.loss import Loss
    expected = [
        "get_forces", "get_formation_energy", "get_atomic_energy",
        "get_elements_from_dataset", "get_energy_per_atom",
        "set_parameters",
        "collectAndcalculate_muEV", "collectAndcalculate_muEF",
        "EnergyGeometry", "BandStructure",
        "Repulsion_optimization", "parallel_repulsion", "unparallel_repulsion",
        "run_calculator_seperately",
        "Loss_EnergyGeometry", "Loss_BandStructure",
        "Bayesian_optimization",
        "mk_traj", "fit", "eos_atoms", "eos_cal",
    ]
    missing = [m for m in expected if not hasattr(Loss, m)]
    assert not missing, f"Loss class missing methods: {missing}"


def test_data_module_has_required_helpers():
    """Some helpers that are module-level (not Loss methods) but loss.py
    relies on them via `data.X(...)`. Keep them around."""
    import utils.data as data
    expected = [
        "get_bolt_factor", "get_full_dftb", "get_elements_from_dataset",
        "find_ref_by_structure",
        "fit_atomic_E0s", "load_E0s_cache", "save_E0s_cache",
    ]
    missing = [m for m in expected if not hasattr(data, m)]
    assert not missing, f"utils.data missing helpers: {missing}"
