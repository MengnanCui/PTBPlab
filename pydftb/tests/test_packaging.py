"""pydftb as a package of its own (P1): the package list of ``pydftb/pyproject.toml`` matches
the tree, and no pydftb module imports PTBP (``utils``, ``cli``, ``tools``, ``parameterize``)."""
import json
import os
import subprocess
import sys

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # the pydftb folder
ROOT = os.path.dirname(PKG)
PYPROJECT = os.path.join(PKG, "pyproject.toml")
PTBP = ("utils", "cli", "tools", "parameterize")


def _tree_packages() -> list:
    """Every folder with an ``__init__.py`` under pydftb/, as a dotted name (build output,
    egg-info and caches left by local builds are not part of the tree)."""
    found = []
    for path, dirs, files in os.walk(PKG):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "build", "dist")
                   and not d.endswith(".egg-info") and not d.startswith(".")]
        if "__init__.py" in files:
            rel = os.path.relpath(path, PKG)
            found.append("pydftb" if rel == "." else "pydftb." + rel.replace(os.sep, "."))
    return sorted(found)


def test_pyproject_lists_every_package():
    if not os.path.exists(PYPROJECT):
        pytest.skip("installed copy: pyproject.toml is not part of the package")
    tomllib = pytest.importorskip("tomllib")
    with open(PYPROJECT, "rb") as fh:
        conf = tomllib.load(fh)
    setup = conf["tool"]["setuptools"]
    assert sorted(setup["packages"]) == _tree_packages()
    assert setup["package-dir"] == {"pydftb": "."}
    assert setup["include-package-data"] is False             # docs/ and bench/ stay out
    assert conf["project"]["scripts"] == {"pydftb": "pydftb.cli:main"}
    assert "pytest" not in conf.get("tool", {})                # would move pytest's rootdir


SCRIPT = r"""
import importlib, importlib.abc, json, pkgutil, sys
PTBP = set(sys.argv[1].split(","))
attempts = []

class Block(importlib.abc.MetaPathFinder):
    # records and refuses every import of a PTBP top-level module, wherever it would be found
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in PTBP:
            attempts.append(name)
            raise ModuleNotFoundError(f"No module named {name!r} (blocked)", name=name)
        return None

sys.meta_path.insert(0, Block())
import pydftb
names, optional, failed = [], [], []

def classify(name, err):
    # a missing optional package (torch, ...) or pytest.importorskip: skipped; else a failure
    if isinstance(err, ModuleNotFoundError):
        ok = (err.name or "?").split(".")[0] not in PTBP
    else:
        ok = type(err).__name__ == "Skipped"
    (optional if ok else failed).append(f"{name}: {type(err).__name__}: {err}")

for info in pkgutil.walk_packages(pydftb.__path__, "pydftb.",
                                  onerror=lambda name: classify(name, sys.exc_info()[1])):
    names.append(info.name)
    try:
        importlib.import_module(info.name)
    except BaseException as err:
        classify(info.name, err)
print(json.dumps({"file": pydftb.__file__, "names": names, "optional": optional,
                  "failed": failed, "attempts": attempts,
                  "loaded": sorted(m for m in sys.modules if m.split(".")[0] in PTBP)}))
"""


def test_pydftb_never_imports_ptbp():
    """Imports every module of the package in a fresh interpreter whose import system refuses
    PTBP's top-level names (they are importable from a checkout, where pydftb/ sits next to
    utils/ and cli/).  Modules that need a missing optional package (torch, ...) are
    skipped."""
    env = dict(os.environ, PYTHONPATH=ROOT)
    run = subprocess.run([sys.executable, "-c", SCRIPT, ",".join(PTBP)], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert os.path.realpath(out["file"]) == os.path.realpath(os.path.join(PKG, "__init__.py"))
    assert {"pydftb.cli", "pydftb.validation.refgen", "pydftb.tests.test_cli"} <= set(out["names"])
    assert out["failed"] == [] and out["attempts"] == [] and out["loaded"] == []
    assert len(out["optional"]) < len(out["names"]) / 2, out["optional"]
