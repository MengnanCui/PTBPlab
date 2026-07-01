"""`python -m gui` → launch the desktop app."""
from __future__ import annotations

import sys

from gui.app import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
