# third_party

Source of dependencies that cannot be downloaded on every machine.  Each directory is an
unmodified copy of the upstream release and keeps the upstream licence.

## hotcent 2.1.0

- Upstream: https://gitlab.com/mvdb/hotcent, tag `v2.1.0` (commit `685bf7d`).
- Licence: **GPL-3.0-or-later** (`hotcent/LICENSE`), not the MIT licence of the rest of this
  repository.  PTBP only calls hotcent; the copy is kept here unchanged so that it can be
  installed without network access.
- Left out of the copy: upstream's `tests/`, `examples/` and `tutorials/` (1.6 MB).

Install it (offline: no download; needs a C compiler for the `_hotcent` extension, and
numpy, scipy, ase, matplotlib, pyyaml, lxml already in the environment):

```bash
pip install --no-build-isolation ./third_party/hotcent          # add --no-deps when offline
```

The extension is compiled with upstream's `-O3 -march=native`: build it on the kind of
machine that runs the calculations (a login node with an older CPU than the compute nodes
can give an "illegal instruction" there).

A checkout of PTBP that cannot install it still runs: when `import hotcent` fails,
`utils/_compat.py` imports `third_party/hotcent` directly.  hotcent then works without its C
extension, as upstream does: slower (about 2 times for the SKF tables) and the table values
differ from the C build by about 1e-7, so the installed copy is the one to use for results.

Check (the installed copy gave byte-identical SKF files to the GitLab install of v2.1.0):

```bash
python -c "import hotcent, _hotcent; print(hotcent.__version__, hotcent.__file__)"
```

pylibxc (the libxc bindings hotcent uses for GGA functionals such as PBE) is not copied
here: it needs the libxc C library (`conda install -c conda-forge libxc pylibxc`).
