"""Convert a PTBP skopt checkpoint written before the penalty-radius fix.

Until the fix, the loss penalty always used a covalent radius of 0.5 Bohr
(``utils/loss.py``: the radius was never passed to ``Loss``).  The penalty multiplies
the whole loss and does not depend on sigma_rep, so every successful evaluation
converts exactly:

    y_new = y_old * r0_penalty(r0_w, r0_d, radius) / r0_penalty(r0_w, r0_d, 0.5)

Failed evaluations (the markers 97, 98, 99) are left as they are; resuming in the dataset
or reaction mode gives them that mode's failure value (``Loss.failure_loss``).  The
original file is kept as ``<checkpoint>.v1.pkl``; ``<checkpoint>.meta.json`` records the
new scale.

    python tools/rescale_checkpoint.py run/checkpoint.pkl --radius 2.494438
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OLD_RADIUS = 0.5
FAILURE_MARKERS = (97.0, 98.0, 99.0)


def rescale(res, radius: float, old_radius: float = OLD_RADIUS):
    """Scale ``res.func_vals`` (and ``fun``/``x``) of an skopt result in place."""
    from utils.loss import r0_penalty
    vals = np.array(res.func_vals, dtype=float)
    for i, (x, y) in enumerate(zip(res.x_iters, vals)):
        if y in FAILURE_MARKERS:
            continue
        vals[i] = y * r0_penalty(x[0], x[1], radius) / r0_penalty(x[0], x[1], old_radius)
    res.func_vals = vals
    best = int(np.argmin(vals))
    res.fun, res.x = float(vals[best]), list(res.x_iters[best])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--radius", type=float, required=True,
                    help="penalty radius in Bohr, as printed at the start of the new run")
    ap.add_argument("--mode", default="energygeometry",
                    help="optimization option of the run (recorded in the meta file)")
    a = ap.parse_args(argv)

    from skopt import dump, load
    from utils.loss import LOSS_VERSION

    meta_path = Path(f"{a.checkpoint}.meta.json")
    if meta_path.exists():
        sys.exit(f"{meta_path} exists: {a.checkpoint} already uses loss version "
                 f"{json.loads(meta_path.read_text()).get('ptbp_loss_version')}")
    backup = a.checkpoint.with_suffix(".v1.pkl")
    shutil.copy2(a.checkpoint, backup)
    res = rescale(load(a.checkpoint), a.radius)
    dump(res, a.checkpoint, compress=9, store_objective=False)
    meta_path.write_text(json.dumps({"ptbp_loss_version": LOSS_VERSION,
                                     "penalty_radius_bohr": a.radius, "mode": a.mode}))
    print(f"rescaled {len(res.func_vals)} evaluations; best {res.fun:.8g} at {res.x}; "
          f"original kept as {os.fspath(backup)}")


if __name__ == "__main__":
    main()
