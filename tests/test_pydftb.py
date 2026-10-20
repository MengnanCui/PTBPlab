"""Smoke test that the pydftb sub-package imports from the PTBP test suite.

pydftb's own tests live in pydftb/tests (run: pytest pydftb/tests).
"""


def test_pydftb_imports():
    import pydftb

    assert hasattr(pydftb, "SCC") and hasattr(pydftb, "ParamSet")
