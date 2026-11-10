"""`ptbp scan refine`: Nelder–Mead (default) and Bayesian optimisation (--method bo), on a toy
objective.  The SK generation and DFTB stages are replaced by a quadratic in the radii, so
the tests check the bookkeeping only: new point indices from 100000 on, never reused, one
record per BO seed, the best over seeds, and a clear error without scikit-optimize."""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest


def _toy(tmp_path, monkeypatch):
    from utils import accuracy as acc
    cfg_path = tmp_path / "toy.json"
    cfg_path.write_text(json.dumps({
        "name": "toy", "out": str(tmp_path / "runs"),
        "space": {"elements": ["Li"], "fit": ["Li"], "bounds": {"around": [0.8, 1.2]},
                  "kind": "random", "n": 1, "seed": 0, "include_published": True},
        "data": {"e0s": None}}))
    cfg = acc.load_config(str(cfg_path))
    sp = acc.space_of(cfg)
    lo, hi = sp.lower_upper()
    target = lo + 0.6 * (hi - lo)

    def loss(x):
        return float(np.sum(((np.asarray(x) - target) / (hi - lo)) ** 2))

    xs = {}

    def fake_evaluate(cfg_, idx, x=None, force=False, log=print):
        xs[idx] = np.asarray(x, float)
        os.makedirs(acc.point_dir(cfg_, idx), exist_ok=True)
        return {}

    def fake_score(cfg_, idx, force=False):
        return {"index": idx, "strategies": {"S1": {"loss": loss(xs[idx])}}}

    monkeypatch.setattr(acc, "evaluate_point", fake_evaluate)
    monkeypatch.setattr(acc, "score_point", fake_score)
    # one scored grid point: the start of the refinement
    x0 = lo + 0.4 * (hi - lo)
    d = acc.point_dir(cfg, 0)
    os.makedirs(d)
    with open(os.path.join(d, "losses.json"), "w") as fh:
        json.dump({"index": 0, "radii": sp.radii(x0), "strategies": {"S1": {"loss": loss(x0)}}}, fh)
    return acc, cfg, loss, x0


def test_refine_nm_indices_and_record(tmp_path, monkeypatch):
    acc, cfg, loss, x0 = _toy(tmp_path, monkeypatch)
    out = acc.refine(cfg, "S1", maxfev=12, log=lambda *a: None)
    assert out["loss"] < loss(x0)
    idx = [h["index"] for h in out["history"]]
    assert idx == list(range(100000, 100000 + len(idx)))
    assert os.path.exists(os.path.join(cfg["out"], "refine_S1.json"))
    out2 = acc.refine(cfg, "S1", maxfev=4, log=lambda *a: None)       # a second run: new indices
    assert min(h["index"] for h in out2["history"]) > max(idx)


@pytest.mark.skipif(importlib.util.find_spec("skopt") is not None,
                    reason="scikit-optimize is installed")
def test_refine_bo_needs_skopt(tmp_path, monkeypatch):
    acc, cfg, _, _ = _toy(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="scikit-optimize"):
        acc.refine(cfg, "S1", maxfev=4, method="bo", seeds=2)


@pytest.mark.skipif(importlib.util.find_spec("skopt") is None, reason="needs scikit-optimize")
def test_refine_bo_seeds(tmp_path, monkeypatch):
    acc, cfg, loss, x0 = _toy(tmp_path, monkeypatch)
    out = acc.refine(cfg, "S1", maxfev=6, method="bo", seeds=2, box=0.3, log=lambda *a: None)
    assert [r["seed"] for r in out["seeds"]] == [0, 1]
    all_idx = [h["index"] for r in out["seeds"] for h in r["history"]]
    assert len(all_idx) == len(set(all_idx)) and min(all_idx) >= 100000
    assert out["best"]["loss"] == min(r["loss"] for r in out["seeds"])
    assert out["best"]["loss"] <= loss(x0) + 1e-12
    assert os.path.exists(os.path.join(cfg["out"], "refine_S1_bo.json"))


def test_refine_rejects_unknown_method(tmp_path, monkeypatch):
    acc, cfg, _, _ = _toy(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="nm"):
        acc.refine(cfg, "S1", method="cmaes")


def test_scan_refine_cli_parses_bo():
    import argparse
    from cli import cmd_scan
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    cmd_scan.add_parser(sub)
    a = p.parse_args(["scan", "refine", "x.yaml", "--method", "bo", "--seeds", "3", "--box", "0.2"])
    assert (a.method, a.seeds, a.box, a.maxfev) == ("bo", 3, 0.2, 30)
    a = p.parse_args(["scan", "refine", "x.yaml"])
    assert a.method == "nm"
