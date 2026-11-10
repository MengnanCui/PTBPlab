"""Clean, centralised logging for the GUI (Tk-free).

Two jobs:

1. `configure_logging` / `get_logger` — a concise, leveled logger ("ptbp.gui")
   writing to the console and, optionally, a file. `strip_ansi` keeps colour
   codes emitted by child processes out of both.

2. `install_tk_exception_handler` — customtkinter 6.0.0 raises a benign
   `AttributeError: 'NoneType' object has no attribute 'winfo_exists'` from its
   internal `_draw` when a `<Configure>` event fires while a frame's `_canvas`
   is momentarily `None`. Tk prints these as raw "Exception in Tkinter
   callback" tracebacks, which is noise. This installs a
   `report_callback_exception` hook that swallows that specific case (logged at
   DEBUG) and formats any *real* callback error as one clean line.

Nothing here imports tkinter; the handler operates on the root object passed in
(customtkinter's CTk subclasses tkinter.Tk and exposes the hook).
"""
from __future__ import annotations

import logging
import re
import traceback
from pathlib import Path
from typing import Optional

LOGGER_NAME = "ptbp.gui"

# Matches CSI ("\x1b[...m" etc.) and a few other ANSI escape sequences.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# The customtkinter internal frames whose AttributeError is known-benign.
_BENIGN_FRAMES = {"_draw", "_update_dimensions_event", "_set_dimensions"}
_BENIGN_TOKENS = ("winfo_exists", "_canvas", "NoneType")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences so logs/panels render as plain text."""
    return _ANSI_RE.sub("", text)


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def configure_logging(log_file: Optional[str | Path] = None,
                      level: int = logging.INFO) -> logging.Logger:
    """Set up the `ptbp.gui` logger. Idempotent: repeated calls do not stack
    handlers. Adds a file handler only the first time a given path is seen.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s",
                            datefmt="%H:%M:%S")

    has_console = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_file is not None:
        target = str(Path(log_file).resolve())
        already = any(
            isinstance(h, logging.FileHandler)
            and Path(getattr(h, "baseFilename", "")).resolve() == Path(target)
            for h in logger.handlers
        )
        if not already:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(target, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


def is_benign_ctk_error(exc_type, exc_value, tb) -> bool:
    """True for the known-benign customtkinter `_canvas.winfo_exists` race."""
    if not issubclass(exc_type, AttributeError):
        return False
    msg = str(exc_value)
    if not any(tok in msg for tok in _BENIGN_TOKENS):
        return False
    # Any frame in the traceback from customtkinter's draw/resize path?
    for frame in traceback.extract_tb(tb):
        if frame.name in _BENIGN_FRAMES:
            return True
    return False


def install_tk_exception_handler(root, logger: Optional[logging.Logger] = None):
    """Route Tk callback exceptions through the logger, muting the benign
    customtkinter draw-race noise. `root` is a tkinter.Tk (CTk) instance."""
    log = logger or get_logger()

    def handler(exc_type, exc_value, tb):
        if is_benign_ctk_error(exc_type, exc_value, tb):
            log.debug("suppressed benign customtkinter redraw race: %s", exc_value)
            return
        log.error("UI callback error: %s: %s", exc_type.__name__, exc_value)
        log.debug("".join(traceback.format_exception(exc_type, exc_value, tb)))

    root.report_callback_exception = handler
    return handler
