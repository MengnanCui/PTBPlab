"""The NumPy path never imports torch (milestone M4): importing pydftb and running a
calculation with the default backend must leave ``torch`` out of ``sys.modules``."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT = r"""
import sys
import numpy as np
import pydftb
from pydftb import DFTB, ParamSet, ase_calc, bands, calculator, solvers, perfmodel  # noqa
from pydftb.validation import cases
p = ParamSet(cases.SKF_SETS["ptbp_hcocu"], max_l={"O": 1, "H": 0})
res = DFTB(p.skset(["O", "H", "H"]), scc_tol=1e-8).calculate(
    ["O", "H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]]))
assert abs(res.energy) > 1.0
print("torch" in sys.modules)
"""


def test_numpy_path_does_not_import_torch():
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, env=env, check=True,
                         capture_output=True, text=True).stdout.split()
    assert out[-1] == "False"
