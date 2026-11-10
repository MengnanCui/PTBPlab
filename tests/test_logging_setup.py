"""Tests for gui.core.logging_setup (ANSI stripping, logger, Tk exc filter)."""
import logging
import sys

from gui.core import logging_setup as ls


def test_strip_ansi():
    coloured = "\x1b[31mred\x1b[0m \x1b[1;32mbold-green\x1b[0m plain"
    assert ls.strip_ansi(coloured) == "red bold-green plain"
    # already-clean text is unchanged
    assert ls.strip_ansi("no colour here") == "no colour here"


def test_configure_logging_idempotent(tmp_path):
    log_file = tmp_path / "gui.log"
    a = ls.configure_logging(log_file=log_file)
    n1 = len(a.handlers)
    b = ls.configure_logging(log_file=log_file)
    n2 = len(b.handlers)
    assert a is b                      # same logger
    assert n2 == n1                    # no handler stacking on repeat
    a.info("hello-log")
    for h in a.handlers:
        h.flush()
    assert "hello-log" in log_file.read_text()


def _fake_tb(func_name):
    """Build a traceback whose deepest frame has function name `func_name`."""
    code = compile(
        f"def {func_name}():\n    raise AttributeError("
        "\"'NoneType' object has no attribute 'winfo_exists'\")\n"
        f"{func_name}()",
        "<ctk_fake>", "exec",
    )
    try:
        exec(code, {})
    except AttributeError:
        return sys.exc_info()
    raise AssertionError("expected AttributeError")


def test_benign_ctk_error_detected():
    exc_type, exc_value, tb = _fake_tb("_draw")
    assert ls.is_benign_ctk_error(exc_type, exc_value, tb) is True


def test_real_error_not_classified_benign():
    # AttributeError from an unrelated function is NOT benign.
    exc_type, exc_value, tb = _fake_tb("my_business_logic")
    assert ls.is_benign_ctk_error(exc_type, exc_value, tb) is False
    # A different exception type is never benign.
    try:
        raise ValueError("boom")
    except ValueError:
        et, ev, t = sys.exc_info()
    assert ls.is_benign_ctk_error(et, ev, t) is False


class _FakeRoot:
    report_callback_exception = None


def test_install_handler_swallows_benign_logs_real(caplog):
    root = _FakeRoot()
    logger = logging.getLogger(ls.LOGGER_NAME)
    ls.install_tk_exception_handler(root, logger)
    assert callable(root.report_callback_exception)

    # benign → no error record
    with caplog.at_level(logging.ERROR, logger=ls.LOGGER_NAME):
        root.report_callback_exception(*_fake_tb("_draw"))
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    # real → one clean error record
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=ls.LOGGER_NAME):
        root.report_callback_exception(*_fake_tb("my_business_logic"))
    errs = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errs) == 1
    assert "UI callback error" in errs[0].message
