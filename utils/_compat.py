"""Runtime compatibility shims, loaded automatically by utils/__init__.py.

These shims paper over two environmental mismatches between the original
PTBP code (Linux + ASE ~3.20) and a modern macOS + ASE 3.28 setup, with
no edits to the rest of the source tree.

1. macOS multiprocessing default
   --------------------------------
   `utils.hotpy.full_hotcent` and `utils.loss.parallel_repulsion` use
   `multiprocessing.Pool`. On macOS Python 3.8+ defaults to the 'spawn'
   start method, under which these pools hang silently (children exit
   before producing output, `error_callback` swallows exceptions). Force
   'fork' instead.

2. ASE 3.20+ extxyz layout
   ------------------------
   xyz `energy=...` headers used to land on `atoms.info`; new ASE puts
   them on `atoms.calc.results`. `forces=...` similarly moved out of
   `atoms.arrays`. PTBP code uses both styles in different places —
   `info['energy']`, `get_potential_energy()`, `get_array('forces')`.

   - On read: copy calc.results into info/arrays so legacy reads work
     (keep calc itself for `get_potential_energy()` callers).
   - On write: drop info/arrays keys that duplicate calc.results so
     ASE's collision check at write time does not raise.

3. hotcent from the repository
   ----------------------------
   hotcent is not on PyPI.  Its source (v2.1.0) is kept in third_party/hotcent:
   `pip install --no-build-isolation ./third_party/hotcent` installs it offline (with
   its C extension).  When hotcent is not installed, a checkout imports it from
   third_party/hotcent directly (slower: the C extension needs the install).

Idempotent: re-importing this module is a no-op.
"""

import multiprocessing as mp


def _set_fork_start_method():
    try:
        if mp.get_start_method(allow_none=True) != 'fork':
            mp.set_start_method('fork', force=True)
    except RuntimeError:
        # Start method already locked in by an earlier user — leave it.
        pass


def _patch_ase_io():
    import ase.io
    from ase.atoms import Atoms

    if getattr(ase.io.read, '_ptbp_compat', False):
        return  # already patched

    _orig_read = ase.io.read
    _orig_write = ase.io.write

    def _mirror_calc_to_legacy(atoms):
        if atoms.calc is None:
            return atoms
        n = len(atoms)
        for k, v in list(atoms.calc.results.items()):
            is_per_atom = (
                hasattr(v, 'shape')
                and len(getattr(v, 'shape', ())) >= 1
                and v.shape[0] == n
            )
            if is_per_atom:
                if k not in atoms.arrays:
                    atoms.set_array(k, v)
            else:
                if k not in atoms.info:
                    atoms.info[k] = v
        return atoms

    def _read_with_legacy_mirror(*args, **kwargs):
        result = _orig_read(*args, **kwargs)
        if isinstance(result, list):
            return [_mirror_calc_to_legacy(a) for a in result]
        return _mirror_calc_to_legacy(result)
    _read_with_legacy_mirror._ptbp_compat = True

    def _drop_dup(atoms):
        if atoms.calc is None:
            return atoms
        for k in list(atoms.calc.results.keys()):
            atoms.info.pop(k, None)
            if k in atoms.arrays:
                del atoms.arrays[k]
        return atoms

    def _write_drop_dup(filename, images, *args, **kwargs):
        if isinstance(images, Atoms):
            _drop_dup(images)
        else:
            try:
                for a in images:
                    _drop_dup(a)
            except TypeError:
                pass
        return _orig_write(filename, images, *args, **kwargs)
    _write_drop_dup._ptbp_compat = True

    ase.io.read = _read_with_legacy_mirror
    ase.io.write = _write_drop_dup


def _vendored_hotcent():
    """Put third_party/hotcent on sys.path when hotcent is not installed (a checkout)."""
    import importlib.util
    import os
    import sys
    if importlib.util.find_spec('hotcent') is not None:
        return
    vendored = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'third_party', 'hotcent')
    if not os.path.isfile(os.path.join(vendored, 'hotcent', '__init__.py')):
        return
    sys.path.insert(0, vendored)
    os.environ['PYTHONPATH'] = os.pathsep.join(           # spawned workers and subprocesses
        p for p in (vendored, os.environ.get('PYTHONPATH')) if p)
    if importlib.util.find_spec('_hotcent') is None:
        print("[hotcent] using third_party/hotcent without its C extension: slower, and the SKF "
              "values differ from the C build by ~1e-7; "
              "install it: pip install --no-build-isolation ./third_party/hotcent",
              file=sys.stderr)


_set_fork_start_method()
_patch_ase_io()
_vendored_hotcent()
