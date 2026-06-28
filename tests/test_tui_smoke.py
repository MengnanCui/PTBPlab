"""Textual smoke tests using App.run_test() — exercises screen mounting
without needing a real terminal."""
from __future__ import annotations

from pathlib import Path

import ase
import ase.io
import numpy as np
import pytest


pytestmark = pytest.mark.asyncio


def _h2_dataset(tmp_path: Path) -> Path:
    p = tmp_path / "h2.xyz"
    atoms_list = []
    for i in range(3):
        a = ase.Atoms('H2', positions=[[0, 0, 0], [0.74 + 0.01 * i, 0, 0]],
                      cell=np.eye(3) * 10.0, pbc=False)
        a.info['energy'] = -1.0 - 0.05 * i
        atoms_list.append(a)
    ase.io.write(str(p), atoms_list)
    return p


async def test_app_starts_at_welcome_screen():
    from cli.tui.app import PtbpSetupApp
    app = PtbpSetupApp()
    async with app.run_test() as pilot:
        # The welcome screen should be the active screen.
        from cli.tui.screens.welcome import WelcomeScreen
        assert isinstance(app.screen, WelcomeScreen)


async def test_initial_flow_optimize_skips_welcome():
    from cli.tui.app import PtbpSetupApp
    from cli.tui.screens.optimize_step1_dataset import OptimizeStep1Dataset
    app = PtbpSetupApp(initial_flow='optimize')
    async with app.run_test() as pilot:
        assert isinstance(app.screen, OptimizeStep1Dataset)
        assert app.state.flow == 'optimize'


async def test_initial_flow_gen_skips_welcome():
    from cli.tui.app import PtbpSetupApp
    from cli.tui.screens.gen_step1_symbols import GenStep1Symbols
    app = PtbpSetupApp(initial_flow='gen')
    async with app.run_test() as pilot:
        assert isinstance(app.screen, GenStep1Symbols)
        assert app.state.flow == 'gen'


async def test_optimize_dataset_picker_validates_existing_file(tmp_path):
    """Step 1 → fill dataset path → Next → state.dataset_report populated."""
    from cli.tui.app import PtbpSetupApp
    from cli.tui.screens.optimize_step2_validation import OptimizeStep2Validation
    from textual.widgets import Input, Button

    ds = _h2_dataset(tmp_path)
    app = PtbpSetupApp(initial_flow='optimize')
    async with app.run_test() as pilot:
        # Type the dataset path
        app.screen.query_one('#path-input', Input).value = str(ds)
        # Click Next
        await pilot.click('#btn-next')
        # Should advance to step 2.
        assert isinstance(app.screen, OptimizeStep2Validation)
        assert app.state.dataset_path == ds
        assert app.state.dataset_report is not None
        assert app.state.dataset_report.n_structures == 3
        assert app.state.dataset_report.detected_mode == 'dataset'


async def test_optimize_dataset_picker_rejects_missing_file(tmp_path):
    """Bad path → stays on step 1 (does not advance)."""
    from cli.tui.app import PtbpSetupApp
    from cli.tui.screens.optimize_step1_dataset import OptimizeStep1Dataset
    from textual.widgets import Input

    app = PtbpSetupApp(initial_flow='optimize')
    async with app.run_test() as pilot:
        app.screen.query_one('#path-input', Input).value = str(
            tmp_path / "does_not_exist.xyz")
        await pilot.click('#btn-next')
        # Did not advance — still on step 1, dataset_report not populated.
        assert isinstance(app.screen, OptimizeStep1Dataset)
        assert app.state.dataset_report is None


async def test_gen_step1_rejects_unknown_symbol():
    from cli.tui.app import PtbpSetupApp
    from cli.tui.screens.gen_step1_symbols import GenStep1Symbols
    from textual.widgets import Input

    app = PtbpSetupApp(initial_flow='gen')
    async with app.run_test() as pilot:
        app.screen.query_one('#syms', Input).value = "H Xx"
        await pilot.click('#btn-next')
        # Stayed on step 1, symbols not committed.
        assert isinstance(app.screen, GenStep1Symbols)
        assert app.state.gen_symbols == []


async def test_gen_step1_accepts_known_symbols_and_advances():
    from cli.tui.app import PtbpSetupApp
    from cli.tui.screens.gen_step2_params import GenStep2Params
    from textual.widgets import Input

    app = PtbpSetupApp(initial_flow='gen')
    async with app.run_test() as pilot:
        app.screen.query_one('#syms', Input).value = "H O Cu"
        await pilot.click('#btn-next')
        assert isinstance(app.screen, GenStep2Params)
        assert app.state.gen_symbols == ['H', 'O', 'Cu']
