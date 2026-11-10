"""Typed SK sets for the r0 nodes of A2 channel E2 (utils.skgrid.TypedSKCache).

The plan and the key without hotcent; with PTBP's hotcent driver one tiny set (two Li nodes):
every ordered label pair, the same-element hetero-label files in the heteronuclear format, the
homo-label file byte-identical to the A1 cache's file at the same radii, and pydftb runs a
type switch on it.
"""
import filecmp
import os

import pytest

from utils.skgrid import (SKCache, SKSettings, TypedSKCache, published, typed_files, typed_plan,
                          typed_spec)


def test_spec_plan_and_key(tmp_path):
    from pydftb.r0mix import R0Switch, node_radii
    sw = R0Switch.around("Li", r0=(4.2, 9.1), spacing=(0.3, 0.6), n=7)
    types = node_radii([sw], {"O": ("O", 2.9, 6.0)})
    spec = typed_spec(types)
    assert spec["Li4"][:3] == ("Li", 4.2, 9.1) and spec["O"][3] == published("O")["p"]
    assert len(typed_files(types)) == 64 and "Li1-Li7.skf" in typed_files(types)
    plan = typed_plan(types, SKSettings(workers=8))
    assert plan["tables"] == 36 and plan["same_element_hetero_tables"] == 21
    assert plan["files"] == 64 and plan["wall_seconds"] < plan["serial_seconds"]
    cache = TypedSKCache(str(tmp_path), SKSettings())
    assert cache.key(types) == cache.key(dict(reversed(list(types.items()))))
    other = dict(types, O=("O", 3.0, 6.0))
    assert cache.key(other) != cache.key(types) and not cache.has(types)
    assert cache.plan(types)["cached"] is False
    with pytest.raises(ValueError, match="is not of element"):
        typed_spec({"Li1": ("O", 3.0, 6.0)})


def test_tiny_typed_set(tmp_path):
    pytest.importorskip("hotcent")
    pytest.importorskip("pylibxc")
    from utils import hotcent_parallel
    if os.environ.get("PTBP_HOTCENT_REFERENCE") or not hotcent_parallel.sharding_supported():
        pytest.skip("typed sets need PTBP's sharded hotcent driver")
    pub = published("Li")
    types = {"Li1": ("Li", pub["r0_w"], pub["r0_d"]), "Li2": ("Li", 1.1 * pub["r0_w"], pub["r0_d"])}
    d = TypedSKCache(str(tmp_path / "typed"), SKSettings(workers=2)).get(types)
    assert sorted(f for f in os.listdir(d) if f.endswith(".skf")) == sorted(typed_files(types))
    a = SKCache(str(tmp_path / "a1"), SKSettings(workers=2)).get(
        {"Li": [pub["r0_w"], pub["r0_d"]]}, {"Li": pub["p"]})
    assert filecmp.cmp(os.path.join(d, "Li1-Li1.skf"), os.path.join(a, "Li-Li.skf"), shallow=False)
    from pydftb.calculator import DFTB
    from pydftb.params import ParamSet
    from pydftb.typeswitch import TypeSwitch
    sk = ParamSet(d, max_l={"Li1": "s", "Li2": "s"}).skset(["Li1", "Li2"])
    res = DFTB(sk, type_switching=[TypeSwitch("Li", ("Li2", "Li1"), 1.0, 1.0)]).calculate(
        ["Li", "Li"], [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    assert res.converged and set(res.type_weights) == {"Li1", "Li2"}
