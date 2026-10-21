"""Loss penalty: the covalent radius of the element being parametrised.

Until the fix, ``Loss`` was never given the radius and the penalty always used its
default of 0.5 Bohr; the published PTBP losses (utils/parameters_set.py) were computed
with the element's own radius, which the oracle test below checks for all 75 elements.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.data import covalent_radii
from ase.units import Bohr

ROOT = Path(__file__).resolve().parents[1]


def test_published_losses_use_the_element_radius():
    from utils.loss import r0_penalty
    from utils.parameters_set import get_PTBP
    table = get_PTBP()
    assert len(table) == 75
    for sym, p in table.items():
        r = covalent_radii[p["atomic_number"]] / Bohr
        assert p["cov_radius"] == pytest.approx(r, abs=5e-4), sym
        expected = (0.5 * p["mu_V"] + 0.5 * p["mu_E"]) * r0_penalty(p["r0_w"], p["r0_d"], r)
        assert expected == pytest.approx(p["loss"], rel=1e-4), sym
        # the old default radius reproduces none of them
        old = (0.5 * p["mu_V"] + 0.5 * p["mu_E"]) * r0_penalty(p["r0_w"], p["r0_d"], 0.5)
        assert old != pytest.approx(p["loss"], rel=1e-2), sym


def test_penalty_minimum():
    from utils.loss import r0_penalty
    assert r0_penalty(2 * 1.3, 4 * 1.3, 1.3) == pytest.approx(4.0)
    assert r0_penalty(2.749, 12.036, covalent_radii[29] / Bohr) == pytest.approx(6.0235, rel=1e-4)


def _cell(formula, a=3.0):
    return Atoms(formula, cell=np.eye(3) * a, pbc=True)


def test_penalty_radius_choice():
    from utils.loss import penalty_radius
    rc = lambda z: covalent_radii[z] / Bohr                       # noqa: E731
    assert penalty_radius([_cell("Cu"), _cell("Cu2")]) == rc(29)
    # several elements: the heaviest, as the BO bounds use
    assert penalty_radius([_cell("H2O"), _cell("NaN")]) == rc(11)
    # structures with different element sets (np.unique on ragged lists used to fail)
    assert penalty_radius([_cell("H2"), _cell("H2O")]) == rc(8)
    # multi-element mode: the element being optimised, as symbol or atomic number
    assert penalty_radius([_cell("CuO")], ["O"]) == rc(8)
    assert penalty_radius([_cell("CuO")], [8]) == rc(8)
    with pytest.raises(ValueError):
        penalty_radius([])


def test_loss_gets_the_dataset_radius():
    from utils.loss import Loss
    loss = Loss(args_input=None, dft_input=[_cell("Cu")], bolt_f11=[1.0])
    assert loss.cov_radius == covalent_radii[29] / Bohr
    assert Loss(dft_input=[_cell("Cu")], cov_radius=2.0).cov_radius == 2.0
    with pytest.raises(ValueError):
        Loss().penalty(2.0, 4.0)


def test_engine_parity_builds_the_loss_with_the_radius(tmp_path):
    """tools/engine_parity.py builds PTBP's Loss like cli/run.py does."""
    from tools.engine_parity import build_loss
    for mode in ("energygeometry", "dataset"):
        loss, rc = build_loss(ROOT / "dft" / "dft.xyz", mode, tmp_path, 11)
        assert rc == loss.cov_radius == covalent_radii[29] / Bohr


def test_dataset_loss_logs_and_applies_the_penalty(tmp_path, monkeypatch):
    from utils.loss import Loss, r0_penalty
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "log.out"
    args = argparse.Namespace(log_file=str(log), optimization_option="dataset",
                              E0s={29: -10.0})

    def structure(e, f):
        a = _cell("Cu2")
        a.info["energy"] = e
        a.set_array("forces", np.full((2, 3), f))
        return a

    dft = [structure(-21.0, 0.1), structure(-20.5, -0.2)]
    dftb = [structure(-21.2, 0.15), structure(-20.4, -0.1)]
    loss = Loss(args_input=args, dft_input=dft, bolt_f11=[1.0, 1.0])
    monkeypatch.setattr(Loss, "get_atomic_energy", lambda self, numbers: {29: -10.1})
    r0_w, r0_d = 3.0, 9.0
    mu = loss.collectAndcalculate_muEF(dft, dftb, [0.6], r0_w, r0_d)
    penalty = r0_penalty(r0_w, r0_d, covalent_radii[29] / Bohr)
    mu_e = abs((-21.2 + 20.2) / 2 - (-21.0 + 20.0) / 2) + abs((-20.4 + 20.2) / 2 - (-20.5 + 20.0) / 2)
    mu_f = np.mean(np.abs([0.05] * 6 + [0.1] * 6))
    assert mu == pytest.approx((0.9 * mu_e + 0.1 * mu_f) * penalty)
    columns = [float(x) for x in log.read_text().split(",")]
    assert columns[4] == pytest.approx(covalent_radii[29] / Bohr, abs=1e-8)
    assert columns[8] == pytest.approx(penalty, abs=1e-5)


def test_old_checkpoints_are_not_resumed(tmp_path):
    from utils.loss import LOSS_VERSION, Loss
    ckpt = tmp_path / "checkpoint.pkl"
    ckpt.write_bytes(b"")
    loss = Loss(args_input=argparse.Namespace(checkpoint=str(ckpt)), dft_input=[_cell("Cu")])
    meta = Path(f"{ckpt}.meta.json")
    with pytest.raises(RuntimeError, match="rescale_checkpoint"):
        loss._check_checkpoint_meta(meta)                     # written before the fix
    meta.write_text(json.dumps({"ptbp_loss_version": LOSS_VERSION, "penalty_radius_bohr": 0.5}))
    with pytest.raises(RuntimeError):
        loss._check_checkpoint_meta(meta)                     # another radius
    meta.write_text(json.dumps({"ptbp_loss_version": LOSS_VERSION,
                                "penalty_radius_bohr": round(loss.cov_radius, 6)}))
    loss._check_checkpoint_meta(meta)                         # the radius as typed for the tool


def test_rescale_checkpoint_is_exact():
    from types import SimpleNamespace

    from tools.rescale_checkpoint import rescale
    from utils.loss import r0_penalty
    xs = [[2.749, 12.036], [3.0, 6.0], [4.0, 9.0]]
    rc = covalent_radii[29] / Bohr
    old = [r0_penalty(x[0], x[1], 0.5) * k for x, k in zip(xs, (0.05, 0.3, 1.0))]
    old[2] = 98.0                                             # failed evaluation marker
    res = rescale(SimpleNamespace(x_iters=xs, func_vals=np.array(old), fun=None, x=None), rc)
    assert res.func_vals[0] == pytest.approx(0.05 * r0_penalty(2.749, 12.036, rc))
    assert res.func_vals[1] == pytest.approx(0.3 * r0_penalty(3.0, 6.0, rc))
    assert res.func_vals[2] == 98.0
    assert res.x == xs[0] and res.fun == res.func_vals[0]
