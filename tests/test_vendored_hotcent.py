"""third_party/hotcent: the v2.1.0 source PTBP installs offline (third_party/README.md)."""
import hashlib
import importlib.util
import re
from pathlib import Path

import pytest

VENDORED = Path(__file__).resolve().parents[1] / "third_party" / "hotcent"


def _digests(pkg: Path) -> dict:
    return {p.relative_to(pkg).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(pkg.rglob("*")) if p.is_file() and p.suffix in (".py", ".c", ".pyx")}


def test_vendored_copy_is_complete():
    init = (VENDORED / "hotcent" / "__init__.py").read_text()
    assert re.search(r"__version__ = '2\.1\.0'", init)
    for name in ("LICENSE", "setup.py", "pyproject.toml", "hotcent/extensions.c",
                 "hotcent/atomic_dft.py", "hotcent/offsite_twocenter.py"):
        assert (VENDORED / name).is_file(), name


def test_installed_hotcent_2_1_0_is_the_vendored_source():
    spec = importlib.util.find_spec("hotcent")
    if spec is None or spec.origin is None:
        pytest.skip("hotcent is not installed")
    installed = Path(spec.origin).parent
    if Path(spec.origin).resolve().is_relative_to(VENDORED.resolve()):
        pytest.skip("hotcent is imported from third_party itself")
    if "__version__ = '2.1.0'" not in (installed / "__init__.py").read_text():
        pytest.skip("installed hotcent is not 2.1.0")
    assert _digests(installed) == _digests(VENDORED / "hotcent")
