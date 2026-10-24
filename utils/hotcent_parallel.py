"""Slater-Koster tables of a BO step with hotcent: every table once, its rows in parallel.

``hotpy.full_hotcent`` used to start one process per pair of elements, and each pair
recomputed its atoms and all its tables -- for an elementary system the loop over
``['Cu', 'Cu']`` computed the Cu-Cu table three times.  Here (milestone M3b)

* every element's confined atom (``AtomicDFT``) is computed once, in this process;
* every table is computed once: its distance rows are split into interleaved shards
  (row i goes to shard i mod K) that run in forked workers (``utils.parallel``) and
  record, per row, the grid size and hotcent's (S, H, H2);
* hotcent's own ``Offsite2cTable.run()`` then replays the recorded rows, so the table,
  its tail smoothening and the SKF files come from hotcent itself: byte-identical to
  the serial computation, whatever the number of shards.

Which files are written, the ``pot.out`` log and the error handling (a pair whose atom
or table fails writes none of its files) are those of ``hotpy.generate_skf``.
Sharding relies on the internals of hotcent 2.1's ``run()``; with another version the
tables are computed by hotcent directly (``sharding_supported``).

Type labels (milestone M5b): the keys of hotpy's confinement may be labels of types
(``Cu2``) with their element in ``hp.elements``.  Every label's atom is computed for its
element with the label's confinement, every file is named after the labels, and the table
of two labels of one element (``Cu-Cu2``), which hotcent 2.1 treats as one atom (one
direction, the wave-function range of the first atom, the files under the element symbol
with an on-site line), is computed for both directions with the range of both atoms and
written in the heteronuclear format, without the on-site line, as
``pydftb/validation/make_cu_types_set.py`` does.  Without labels nothing changes.
"""
from __future__ import annotations

import hashlib
import inspect
import os
from itertools import combinations

import numpy as np
from hotcent.offsite_twocenter import Offsite2cTable

from utils.parallel import ForkBackend, SerialBackend, map_tasks

RMIN, DR, NPOINTS = 0.4, 0.02, 700           # hotpy.generate_skf
# Offsite2cTable.run of hotcent 2.1.0, which the record/replay below was checked against
RUN_SHA256 = "e211b26dfcb5246996222b92bb6ed9d70b46712ee45d798fb3709aea840511ea"


def sharding_supported() -> bool:
    try:
        src = inspect.getsource(Offsite2cTable.run)
    except (OSError, TypeError):
        return False
    return hashlib.sha256(src.encode()).hexdigest() == RUN_SHA256


class ShardedOffsite2c(Offsite2cTable):
    """``Offsite2cTable`` that computes only some rows and records them ("compute"),
    or plays recorded rows back into hotcent's ``run()`` ("replay")."""

    def configure(self, mode: str, shard: int = 0, nshard: int = 1, records=None):
        assert mode in ("compute", "replay")
        self.mode, self.shard, self.nshard = mode, shard, nshard
        self.records = {} if records is None else records
        self._row = -1
        return self

    def make_grid(self, *args, **kwargs):
        self._row += 1                          # run() asks for one grid per row, in order
        i = self._row
        if self.mode == "replay":
            n = self.records[("grid", i)]
            return np.zeros((n, 2)), np.zeros(n)
        if i % self.nshard != self.shard:
            return np.zeros((0, 2)), np.zeros(0)   # not ours: run() skips empty grids
        grid, area = Offsite2cTable.make_grid(self, *args, **kwargs)
        self.records[("grid", i)] = len(grid)
        return grid, area

    def calculate(self, selected, e1, e2, R, grid, area, **kwargs):
        key = (self._row, self.pairs.index((e1, e2)))
        if self.mode == "replay":
            return self.records[key]
        out = Offsite2cTable.calculate(self, selected, e1, e2, R, grid, area, **kwargs)
        self.records[key] = out
        return out


class ShardedCross(ShardedOffsite2c):
    """The table of two atoms of one element with different confinements (two type labels,
    M5b): both directions and the wave-function range of both atoms, as hotcent does for two
    elements (``make_cu_types_set.CrossTable``)."""

    def __init__(self, ela, elb, **kwargs):
        super().__init__(ela, elb, **kwargs)
        self.nel, self.pairs, self.elements = 2, [(ela, elb), (elb, ela)], [ela, elb]


# ---------------------------------------------------------------------------
def skf_jobs(symbols, opt_elem=()) -> list[list[str]]:
    """The pairs of elements ``hotpy.full_hotcent`` runs ``generate_skf`` for."""
    opt_symbol = list(opt_elem)
    if not opt_symbol:
        if len(symbols) == 1:
            return [[symbols[0], symbols[0]]]
        return [list(c) for c in combinations(symbols, 2)]
    return [[elx] + opt_symbol for elx in symbols if elx not in opt_symbol]


def job_tables(pair) -> list[tuple[str, str]]:
    """The tables ``generate_skf`` computes for a pair, in its order (with repeats)."""
    return [(pair[i], pair[k]) for i in range(len(pair)) for k in range(i + 1)]


def _compute_shard(task):
    """Worker: rows of one table (``task`` = (atom_a, atom_b, shard, nshard, kw[, cross]))."""
    atom_a, atom_b, shard, nshard, kw = task[:5]
    table = ShardedCross if len(task) > 5 and task[5] else ShardedOffsite2c
    sk = table(atom_a, atom_b, txt=None).configure("compute", shard, nshard)
    sk.run(RMIN, DR, NPOINTS, **kw)
    return sk.records


def _element_of(hp):
    """label -> element symbol of hotpy's keys (identity without type labels)."""
    of = getattr(hp, "element_of", None)
    return of if of is not None else (lambda s: s)


def generate_all(hp, superposition: str, opt_elem=(), label: str = "ptbp", workers: int = 1,
                 outdir: str = ".", only=None) -> None:
    """``hotpy.full_hotcent``: SKF files of all pairs of ``hp.symbols`` in ``outdir``.
    ``only``: just the tables that involve one of these keys (the hotcent tangent of one
    type's radii, M5b)."""
    xc, r0, p, symbols = hp.xc, hp.confinement_r0, hp.confinement_p, hp.symbols
    element = _element_of(hp)
    jobs = skf_jobs(symbols, opt_elem)
    if only is not None:
        # the tables of the full run that involve `only`, in its order and orientation
        wanted = [t for t in dict.fromkeys(t for job in jobs for t in job_tables(job))
                  if t[0] in only or t[1] in only]
        return _generate_tables(hp, superposition, label, workers, outdir, wanted, element)
    log_path = os.path.join(outdir, "pot.out")
    with open(log_path, "w") as log:
        log.write('#---------------rconf=' + str(r0) + '---------------\n')
    print("CPU cores: " + str(os.cpu_count()))
    print("Number of jobs: " + str(len(jobs)))

    # atoms, once per element (as hot_hamiltonian computes them in every job); with type
    # labels once per label, for its element
    atoms, failed = {}, {}
    for sym in dict.fromkeys(s for job in jobs for s in job):
        try:
            atoms[sym] = hp.hot_hamiltonian(r0[sym], p[sym], xc, element(sym), label)
        except Exception as exc:                    # noqa: BLE001  (reported per job below)
            failed[sym] = exc

    # tables, once each, rows sharded over the workers
    tables = list(dict.fromkeys(t for job in jobs if not any(s in failed for s in job)
                                for t in job_tables(job)))
    kw = dict(superposition=superposition, xc=xc)
    nshard = max(1, -(-workers // max(1, len(tables))))
    print(f"[hotcent] {len(tables)} table(s) x {nshard} shard(s) on {workers} process(es)")
    tasks = [(t, s) for t in tables for s in range(nshard)]
    # forked workers even for one: a crash in hotcent fails the pair, not PTBP
    backend = ForkBackend(workers) if hasattr(os, "fork") else SerialBackend()

    def task(a, b, s):
        if a != b and element(a) == element(b):       # two types of one element (M5b)
            return (atoms[a][0], atoms[b][0], s, nshard, kw, True)
        return (atoms[a][0], atoms[b][0], s, nshard, kw)
    results = map_tasks(_compute_shard, [task(a, b, s) for (a, b), s in tasks], workers,
                        backend=backend)
    records = {t: {} for t in tables}
    for (table, _), res in zip(tasks, results):
        if isinstance(records[table], Exception):
            continue
        if res.ok:
            records[table].update(res.value)
        else:
            records[table] = res.error

    written = set()
    for job in jobs:
        _write_job(job, atoms, failed, records, written, log_path, outdir, label, kw,
                   element=element)


def _generate_tables(hp, superposition, label, workers, outdir, tables, element):
    """The files of the given tables only (``generate_all(..., only=...)``): the atoms of
    their labels, every table once with its rows sharded, written as generate_all writes
    them.  Errors are raised (a tangent needs every file)."""
    xc, r0, p = hp.xc, hp.confinement_r0, hp.confinement_p
    atoms = {sym: hp.hot_hamiltonian(r0[sym], p[sym], xc, element(sym), label)
             for sym in dict.fromkeys(s for t in tables for s in t)}
    kw = dict(superposition=superposition, xc=xc)
    nshard = max(1, -(-workers // max(1, len(tables))))
    tasks = [(t, s) for t in tables for s in range(nshard)]
    backend = ForkBackend(workers) if hasattr(os, "fork") else SerialBackend()
    payload = []
    for (a, b), s in tasks:
        cross = a != b and element(a) == element(b)
        payload.append((atoms[a][0], atoms[b][0], s, nshard, kw) + ((True,) if cross else ()))
    results = map_tasks(_compute_shard, payload, workers, backend=backend)
    records = {t: {} for t in tables}
    for (table, _), res in zip(tasks, results):
        if not res.ok:
            raise res.error
        records[table].update(res.value)
    for table in tables:
        if all(element(s) == s for s in table):
            _replay_and_write(table, atoms, records[table], outdir, kw)
        else:
            _replay_and_write_labels(table, atoms, records[table], outdir, kw, element)


def _write_job(job, atoms, failed, records, written, log_path, outdir, label, kw, element=None):
    """One ``generate_skf`` call: its log lines, then its files -- or none on an error."""
    homonuclear = len(job) == 2 and job[0] == job[1]
    log = open(log_path, "a")
    try:
        for sym in (job[:1] if homonuclear else job):
            log.write(f"calculate {sym}'s hamitonian{' for elementary' if homonuclear else ''}\n")
            log.flush()
            if sym in failed:
                raise failed[sym]
        log.write("###Write skf file###\n")
        pending = []
        for a, b in job_tables(job):
            print(f"\nWorking on {a}-{b}.skf\n")
            rec = records[(a, b)]
            if isinstance(rec, Exception):
                raise rec
            pending.append((a, b))
            config_a, r0_a = atoms[a][1], atoms[a][2]
            config_b, r0_b = atoms[b][1], atoms[b][2]
            log.write('config_a=' + str(config_a) + '\n')
            log.write('config_b=' + str(config_b) + '\n')
            log.write('#r0 for eigenvalue calculation:a,b=' + str(r0_a) + ',' + str(r0_b) + '\n')
            log.flush()
        for table in dict.fromkeys(pending):            # the job succeeded: its files
            if table not in written:
                if element is None or all(element(s) == s for s in table):
                    _replay_and_write(table, atoms, records[table], outdir, kw)
                else:
                    _replay_and_write_labels(table, atoms, records[table], outdir, kw, element)
                written.add(table)
    except Exception as exc:                            # noqa: BLE001  as hotpy.error_callback
        print('Exception:', exc)
    finally:
        log.close()


def _replay_and_write(table, atoms, records, outdir, kw):
    a, b = table
    atom_a, atom_b = atoms[a][0], atoms[b][0]
    sk = ShardedOffsite2c(atom_a, atom_b).configure("replay", records=records)
    sk.run(RMIN, DR, NPOINTS, **kw)
    template = os.path.join(outdir.replace("{", "{{").replace("}", "}}"), "{el1}-{el2}.skf")
    if a == b:
        sk.write(eigenvalues=atom_a.info['eigenvalues'], hubbardvalues=atom_a.info['hubbardvalues'],
                 occupations=atom_a.info['occupations'], spe=0., filename_template=template)
    else:
        sk.write(filename_template=template)


def _replay_and_write_labels(table, atoms, records, outdir, kw, element):
    """The files of a table with a type label (M5b), named after the labels.

    Two types of one element: both directions in the heteronuclear format (no on-site line:
    DFTB+ and pydftb take the format from the labels), written with hotcent's ``write_skf`` as
    ``make_cu_types_set.skf_text``.  Otherwise hotcent writes the files under the element
    symbols into a scratch directory, and they are renamed after the labels."""
    import shutil
    import tempfile

    from ase.data import atomic_masses, atomic_numbers
    from hotcent.slako import write_skf
    a, b = table
    atom_a, atom_b = atoms[a][0], atoms[b][0]
    cross = a != b and element(a) == element(b)
    sk = (ShardedCross if cross else ShardedOffsite2c)(atom_a, atom_b).configure(
        "replay", records=records)
    sk.run(RMIN, DR, NPOINTS, **kw)
    if any(len(e.basis_sets) != 1 for e in (atom_a, atom_b)):
        raise NotImplementedError("type labels with more than one basis set per atom")
    if cross:
        mass = atomic_masses[atomic_numbers[element(a)]]
        extended = any(nl[1] == 'f' for e in (atom_a, atom_b) for nl in e.basis_sets[0])
        for p, (x, y) in enumerate(((a, b), (b, a))):
            with open(os.path.join(outdir, f"{x}-{y}.skf"), "w") as fh:
                write_skf(fh, sk.Rgrid, sk.tables[(p, 0, 0)], False, extended, {}, {}, {}, 0.0,
                          mass, False, {}, {})
        return
    scratch = tempfile.mkdtemp(prefix="skf_", dir=outdir)
    try:
        template = os.path.join(scratch.replace("{", "{{").replace("}", "}}"), "{el1}-{el2}.skf")
        if a == b:
            sk.write(eigenvalues=atom_a.info['eigenvalues'], hubbardvalues=atom_a.info['hubbardvalues'],
                     occupations=atom_a.info['occupations'], spe=0., filename_template=template)
            names = [(element(a), element(a), a, a)]
        else:
            sk.write(filename_template=template)
            names = [(element(a), element(b), a, b), (element(b), element(a), b, a)]
        for ea, eb, x, y in names:
            os.replace(os.path.join(scratch, f"{ea}-{eb}.skf"), os.path.join(outdir, f"{x}-{y}.skf"))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
