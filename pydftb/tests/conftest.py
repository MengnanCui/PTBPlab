"""pydftb test configuration (independent of PTBP's tests/conftest.py).

Tests marked ``dftbplus`` need the dftb+ binary and are skipped without it.
"""
import os

import pytest

DATA = os.path.join(os.path.dirname(__file__), "data")
SKF_ORBIT = os.path.join(DATA, "skf", "orbit")
SKF_HCOCU = os.path.join(DATA, "skf", "ptbp_hcocu")


def pytest_configure(config):
    config.addinivalue_line("markers", "dftbplus: needs the dftb+ binary (skipped otherwise)")


def pytest_collection_modifyitems(config, items):
    from pydftb.validation import dftbplus
    if dftbplus.available():
        return
    skip = pytest.mark.skip(reason="dftb+ binary not found (set $DFTBPLUS)")
    for item in items:
        if "dftbplus" in item.keywords:
            item.add_marker(skip)
