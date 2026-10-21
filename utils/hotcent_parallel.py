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
    """Worker: rows of one table (``task`` = (atom_a, atom_b, shard, nshard, kw))."""
    atom_a, atom_b, shard, nshard, kw = task
    sk = ShardedOffsite2c(atom_a, atom_b, txt=None).configure("compute", shard, nshard)
    sk.run(RMIN, DR, NPOINTS, **kw)
    return sk.records


def generate_all(hp, superposition: str, opt_elem=(), label: str = "ptbp", workers: int = 1,
                 outdir: str = ".") -> None:
    """``hotpy.full_hotcent``: SKF files of all pairs of ``hp.symbols`` in ``outdir``."""
    xc, r0, p, symbols = hp.xc, hp.confinement_r0, hp.confinement_p, hp.symbols
    jobs = skf_jobs(symbols, opt_elem)
    log_path = os.path.join(outdir, "pot.out")
    with open(log_path, "w") as log:
        log.write('#---------------rconf=' + str(r0) + '---------------\n')
    print("CPU cores: " + str(os.cpu_count()))
    print("Number of jobs: " + str(len(jobs)))

    # atoms, once per element (as hot_hamiltonian computes them in every job)
    atoms, failed = {}, {}
    for sym in dict.fromkeys(s for job in jobs for s in job):
        try:
            atoms[sym] = hp.hot_hamiltonian(r0[sym], p[sym], xc, sym, label)
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
    results = map_tasks(_compute_shard, [(atoms[a][0], atoms[b][0], s, nshard, kw)
                                         for (a, b), s in tasks], workers, backend=backend)
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
        _write_job(job, atoms, failed, records, written, log_path, outdir, label, kw)


def _write_job(job, atoms, failed, records, written, log_path, outdir, label, kw):
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
                _replay_and_write(table, atoms, records[table], outdir, kw)
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
