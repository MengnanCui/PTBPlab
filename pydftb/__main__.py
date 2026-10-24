"""``python -m pydftb COMMAND ARGS...``: the ``pydftb`` command (see :mod:`pydftb.cli`)."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
