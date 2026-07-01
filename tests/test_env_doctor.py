"""Tests for gui.core.env_doctor (must degrade gracefully, never raise)."""
import sys

from gui.core import env_doctor


def test_check_returns_readiness_without_raising():
    r = env_doctor.check()
    assert r.probes  # non-empty
    ase = r.by_name("ASE")
    assert ase is not None and ase.ok  # ASE is a hard dep, always present in tests
    # structures only need ASE + NumPy → ready in the test env
    assert r.structures_ready is True


def test_missing_module_reported_not_raised():
    # hotcent / pylibxc are conda-only and absent in CI — must be False, no crash
    r = env_doctor.check()
    hot = r.by_name("hotcent")
    assert hot is not None and hot.ok is False
    assert r.skf_ready is False


def test_probe_dftb_honours_env(monkeypatch):
    # Point ASE_DFTB_COMMAND at a real executable → probe resolves it.
    monkeypatch.setenv("ASE_DFTB_COMMAND", sys.executable)
    p = env_doctor.probe_dftb()
    assert p.ok is True
    assert sys.executable in p.detail


def test_probe_dftb_missing(monkeypatch):
    monkeypatch.delenv("ASE_DFTB_COMMAND", raising=False)
    monkeypatch.setattr(env_doctor.shutil, "which", lambda _name: None)
    p = env_doctor.probe_dftb()
    assert p.ok is False
