"""Paths shared by the A-series scripts (examples/08_accuracy).

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from common.paths import REPO, EX07_DATA, setup
    setup()          # PTBP's utils/ and tools/, examples/07's vasp_common on sys.path

Every path can be moved with an environment variable, so the same scripts run on the remote
server: PTBP_REPO, PTBP_EX07_DATA, PTBP_PEROVSKITE (the perovskite ink project whose stage-1
structures and MACE trajectories the pvk_ink scripts read; no default), PTBP_OFFLOAD (where
files of 20 MB or more go; default examples/08_accuracy/large, which git ignores).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ACCURACY = os.path.dirname(HERE)                                     # examples/08_accuracy
REPO = os.environ.get("PTBP_REPO") or os.path.abspath(os.path.join(ACCURACY, "..", ".."))
EX07 = os.path.join(REPO, "examples", "07_battery_li_li2o")
EX07_DATA = os.environ.get("PTBP_EX07_DATA") or os.path.join(EX07, "data")
PEROVSKITE = os.environ.get("PTBP_PEROVSKITE") or ""                # set it (or --perovskite)
OFFLOAD = os.environ.get("PTBP_OFFLOAD") or os.path.join(ACCURACY, "large")


def setup() -> None:
    """Put PTBP (utils, tools, cli) and examples/07 (vasp_common) on sys.path."""
    for p in (REPO, EX07):
        if p not in sys.path:
            sys.path.insert(0, p)
