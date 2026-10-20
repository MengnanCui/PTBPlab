"""Compare pydftb with a fresh DFTB+ run on all validation cases.

Thin wrapper kept for backward compatibility; the cases now live in
pydftb/validation/cases.py.  Equivalent to::

    python -m pydftb.validation.refgen --live
"""
from pydftb.validation.refgen import live

if __name__ == "__main__":
    live()
