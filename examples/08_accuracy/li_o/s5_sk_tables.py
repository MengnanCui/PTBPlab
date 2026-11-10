"""S5 (A0 plan, section 3; optional diagnostic): keep DFTB's two-centre form, fit the SK tables.

Question.  A1 changes the SK tables only through the confinement radii (r0_w, r0_d per element).
When the best A1 point misses the acceptance line even with a flexible two-body repulsive (the
scan's R2b and oracle rows), is the limit the confinement-radius parameterisation or the
two-centre form itself?  S5 lets the H and S tables of that point change freely but smoothly and
refits the repulsive.  If S5 reaches tier A/B on test/ood where R2b did not, the parameterisation
was the limit.  If it does not, the two-centre form plus a two-body repulsive is the limit, and
environment dependence (A2, A3) is the way on.

What is fitted.  Every SK integral column the basis uses (H and S; per element pair and channel
ss0, sp0, pp0, pp1, ...) gets a correction c(r) = sum_m beta_m B_m(r).  The B_m are n_coef uniform
cubic B-splines whose supports lie inside [r_lo, r_hi], so c is C2 and exactly zero outside.  It
acts as T(r) (1 + c(r)) (``form: multiplicative``, default) or T(r) + c(r) max|T|
(``additive``).  For a pair A != B, the l1 = l2 columns of A-B.skf and B-A.skf hold the same
integral and share their coefficients.  On-site lines, grids, the repulsive blocks and every other
line of the files are never changed.  Penalties: ridge on beta, and second differences of beta
within a channel.

Repulsive.  Refitted on the train set at every outer step: ``r2b`` (default; pydftb.repfit pair
splines with the scan's repfit section, like the scan's R2b row) or ``physrep`` (one sigma per
fitted element by the scan's L_EF, like S1).  R2b is the default because it keeps a weak
repulsive form from hiding what the electronic tables can do.

Loss.  The accept/reject decisions and the reports use the scan's L_EF on the train role
(utils.accuracy.loss_ef) exactly.  The optimiser sees the same terms per structure, with |x|
smoothed to sqrt(x^2 + d^2) - d (``optimizer.abs_smoothing``, eV/atom).  Optional band term
(``bands.weight``): the scan's band loss of the corrected tables, value only, added to the
accept/reject objective.

Optimisation (block coordinates with a trust region).  Outer step k: refit the repulsive for the
current tables.  Then run a few L-BFGS-B iterations on the tables with that repulsive frozen,
inside |beta - beta_k| <= trust.  Accept if the full objective (repulsive refitted, exact L_EF,
penalties, band term) decreased; otherwise halve the trust radius.  With ``physrep`` the
frozen-sigma gradient is the gradient of the full objective, since sigma minimises L_EF.  With
``r2b`` it is not, because R2b minimises its own least squares; that is why every step is checked
after the refit.

Gradients (``optimizer.gradient``):

* ``autodiff``: pydftb.autodiff (milestone M4b), mode "implicit".  The SK tables are leaves of
  pydftb.autodiff.Params, and the implicit SCC derivative gives d(E - TS/2)/dbeta and
  d(forces)/dbeta exactly, with one backward pass per structure.  Needs torch.
* ``fd``: central differences of the full SCC in each coefficient whose support holds a pair
  distance of the structure.  No torch; costs 2 SCCs per such coefficient and structure.
* ``none``: derivative-free Powell on the inner problem.

Stages:

    python s5_sk_tables.py dry-run   s5_li_o.yaml                  # channels, parameters, cost
    python s5_sk_tables.py gradcheck s5_li_o.yaml --structures 2 --coefs 4
    python s5_sk_tables.py fit       s5_li_o.yaml --jobs 64         # resumable (state.json)
    python s5_sk_tables.py report    s5_li_o.yaml --jobs 64         # vs the scan's S1/R2b/oracle
    python s5_sk_tables.py export    s5_li_o.yaml --dest models/S5  # SK files + repulsive splines

Outputs in ``out``:

* ``state.json``: the checkpoint.
* ``steps/kNNN/``: the corrected tables of every accepted step (``skf/``, a full SK set; the start
  point's own files are never touched), the train electronic energies and forces
  (``train.npz``), and the repulsive (``rep.json``).
* ``s5_report.md`` and ``s5_report.json``.
* ``logs/<stage>/...``: run.log, meta.json, events.jsonl, status.json.

``--max-structures N`` keeps the first N structures of every role (smoke tests only).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import setup  # noqa: E402

setup()

L_LETTERS = "spdf"
DEFAULTS = {
    "start": "best:S1",
    "corrections": {"form": "multiplicative", "kinds": ["H", "S"], "n_coef": 4, "ranges": "auto",
                    "channels": "all", "beta_max": 0.5},
    "penalties": {"ridge": 1.0e-3, "smooth": 1.0e-2},
    "repulsive": {"kind": "r2b"},
    "optimizer": {"gradient": "autodiff", "outer": 20, "inner": 5, "trust": 0.2,
                  "trust_max": 0.5, "trust_min": 1.0e-3, "fd_step": 1.0e-3,
                  "abs_smoothing": 1.0e-4, "accept_tol": 1.0e-6},
    "bands": {"weight": 0.0},
}
GRADIENTS = ("autodiff", "fd", "none")
REPULSIVES = ("r2b", "physrep")
BLAS_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS")


# ---- configuration -------------------------------------------------------------------------
def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_s5(path: str) -> dict:
    """The S5 config with defaults filled in; ``scan`` and ``out`` made absolute against the
    config's directory."""
    import yaml
    path = os.path.abspath(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _merge(DEFAULTS, raw)
    base = os.path.dirname(path)

    def absolute(p):
        p = os.path.expandvars(os.path.expanduser(str(p)))
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    cfg["_path"] = path
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    if not cfg.get("scan"):
        raise ValueError("s5 config: 'scan' (the A1 scan config) is required")
    cfg["scan"] = absolute(cfg["scan"])
    cfg["out"] = absolute(cfg.get("out") or os.path.join("runs", cfg["name"]))
    if cfg["optimizer"]["gradient"] not in GRADIENTS:
        raise ValueError(f"optimizer.gradient must be one of {GRADIENTS}")
    if cfg["repulsive"]["kind"] not in REPULSIVES:
        raise ValueError(f"repulsive.kind must be one of {REPULSIVES}")
    if cfg["corrections"]["form"] not in ("multiplicative", "additive"):
        raise ValueError("corrections.form must be multiplicative or additive")
    return cfg


def start_index(scan, spec) -> int:
    """A point index, or ``best:<strategy>`` over the scan's scored points."""
    from utils import accuracy as acc
    s = str(spec)
    if s.isdigit():
        return int(s)
    if s.startswith("best:"):
        strategy = s.split(":", 1)[1]
        best = acc.best_points(acc._scored(scan), [strategy])
        if strategy not in best:
            raise SystemExit(f"no scored point of {scan['name']} has a finite {strategy} loss: "
                             "run `ptbp scan losses` first")
        return int(best[strategy]["index"])
    raise ValueError(f"start must be a point index or best:<strategy>, not {spec!r}")


def point_records(scan, index: int):
    """(result.json, losses.json or None) of a scan point."""
    from utils import accuracy as acc
    pdir = acc.point_dir(scan, index)
    path = os.path.join(pdir, "result.json")
    if not os.path.exists(path):
        raise SystemExit(f"point {index} of {scan['name']} has not run ({path} is missing)")
    with open(path) as fh:
        res = json.load(fh)
    losses = None
    if os.path.exists(os.path.join(pdir, "losses.json")):
        with open(os.path.join(pdir, "losses.json")) as fh:
            losses = json.load(fh)
    return res, losses


# ---- the corrections -----------------------------------------------------------------------
def _cubic(x):
    """The uniform cubic B-spline on [0, 4)."""
    y = np.zeros_like(x)
    a = (x >= 0) & (x < 1)
    y[a] = x[a] ** 3 / 6.0
    b = (x >= 1) & (x < 2)
    xb = x[b]
    y[b] = (-3 * xb ** 3 + 12 * xb ** 2 - 12 * xb + 4) / 6.0
    c = (x >= 2) & (x < 3)
    xc = x[c]
    y[c] = (3 * xc ** 3 - 24 * xc ** 2 + 60 * xc - 44) / 6.0
    d = (x >= 3) & (x < 4)
    y[d] = (4 - x[d]) ** 3 / 6.0
    return y


def bspline_basis(r, r_lo: float, r_hi: float, n: int) -> np.ndarray:
    """(len(r), n): n uniform cubic B-splines with supports inside [r_lo, r_hi]; any combination
    and its first two derivatives vanish at both ends (C2, zero outside)."""
    r = np.asarray(r, float)
    h = (r_hi - r_lo) / (n + 3)
    return np.stack([_cubic((r - (r_lo + m * h)) / h) for m in range(n)], axis=1)


def _support(ch, m):
    h = (ch.r_hi - ch.r_lo) / (ch.n_coef + 3)
    return ch.r_lo + m * h, ch.r_lo + (m + 4) * h


@dataclasses.dataclass
class Channel:
    name: str              # e.g. "Li-O.H.sp0": H integral, s on Li, p on O, |m| = 0
    pair: tuple            # the two elements (sorted), for the distances
    kind: str              # "H" or "S"
    targets: list          # [((a, b) file key, stored column)]
    r_lo: float            # Bohr
    r_hi: float
    n_coef: int
    start: int             # its first coefficient in the vector

    @property
    def slice(self) -> slice:
        return slice(self.start, self.start + self.n_coef)


def _stored(f, full: int):
    cols = np.arange(f.table.shape[1]) if f.columns is None else np.asarray(f.columns)
    hit = np.flatnonzero(cols == full)
    return int(hit[0]) if len(hit) else None


def _grid(f) -> np.ndarray:
    """r (Bohr) of the table rows: row k is at (k + 1) dr (pydftb.skf, DFTB+)."""
    return (np.arange(f.table.shape[0]) + 1) * f.dr


AUTO_FRACTION = 1e-3      # "auto" ranges end where the H integrals fall below this fraction


def _auto_range(fs, d_min: float):
    """For the files of one element pair (A-B and B-A): r_lo = 0.9 x the shortest train
    distance; r_hi where the largest H integral has fallen below AUTO_FRACTION of its value at
    r_lo (the larger of the two files; at most the table end minus 1 Bohr)."""
    lo, hi = [], []
    for f in fs:
        r = _grid(f)
        cols = np.arange(f.table.shape[1]) if f.columns is None else np.asarray(f.columns)
        h_cols = [j for j, full in enumerate(cols) if full < 20]
        mag = np.abs(f.table[:, h_cols]).max(axis=1)
        r_lo = max(float(r[0]), 0.9 * d_min)
        ref = mag[min(int(np.searchsorted(r, r_lo)), len(mag) - 1)]
        below = np.flatnonzero((r > r_lo) & (mag < AUTO_FRACTION * ref))
        r_hi = float(r[below[0]]) if len(below) else float(r[-1])
        lo.append(r_lo)
        hi.append(min(r_hi, float(r[-1]) - 1.0))
    r_lo, r_hi = max(lo), max(hi)
    return r_lo, max(r_hi, r_lo + 2.0)


def _wanted(tokens, name, label, kind, short):
    if tokens in (None, "all"):
        return True
    keys = {name, label, short, f"{kind}.{short}", kind}
    return any(t in keys for t in tokens)


def build_channels(files, shells: dict, ccfg: dict, d_min: dict, log=print) -> list:
    """The corrected SK integral columns (module docstring); ranges in Bohr."""
    from ase.data import atomic_numbers
    from pydftb.constants import BOHR
    from pydftb.skf import SK_COLUMN
    kinds = [str(k).upper() for k in ccfg.get("kinds", ["H", "S"])]
    tokens = ccfg.get("channels", "all")
    ranges = ccfg.get("ranges", "auto")
    n_coef = int(ccfg.get("n_coef", 4))
    els = sorted(shells, key=lambda e: (atomic_numbers[e], e))
    combos = []
    for ia, a in enumerate(els):
        for b in els[ia:]:
            if (a, b) not in files:
                continue
            for (l1, l2, m), col in sorted(SK_COLUMN.items(), key=lambda kv: kv[1]):
                if l1 in shells[a] and l2 in shells[b]:
                    keys = [(a, b)]
                    if a != b and l1 == l2 and (b, a) in files:
                        keys.append((b, a))         # the same integral, both files
                    combos.append((f"{a}-{b}", (a, b), (l1, l2, m), col, keys))
                if a != b and l1 < l2 and l1 in shells[b] and l2 in shells[a] and (b, a) in files:
                    combos.append((f"{b}-{a}", (a, b), (l1, l2, m), col, [(b, a)]))
    out, start, skipped = [], 0, set()
    for label, pair, (l1, l2, m), col, keys in combos:
        key = tuple(sorted(pair, key=lambda e: (atomic_numbers[e], e)))
        dist = d_min.get(key)
        if dist is None:
            skipped.add("-".join(key))
            continue
        given = None
        if isinstance(ranges, dict):
            given = ranges.get(f"{key[0]}-{key[1]}") or ranges.get(f"{key[1]}-{key[0]}")
        if given is not None:
            r_lo, r_hi = float(given[0]) / BOHR, float(given[1]) / BOHR
        else:
            fs = [files[k] for k in ((key[0], key[1]), (key[1], key[0])) if k in files]
            r_lo, r_hi = _auto_range(fs, dist)
        if not r_hi > r_lo:
            raise ValueError(f"corrections.ranges {label}: needs r_lo < r_hi")
        short = f"{L_LETTERS[l1]}{L_LETTERS[l2]}{m}"
        for kind in kinds:
            name = f"{label}.{kind}.{short}"
            if not _wanted(tokens, name, label, kind, short):
                continue
            full = col + (0 if kind == "H" else 20)
            targets = [(k, j) for k in keys if (j := _stored(files[k], full)) is not None]
            if not targets:
                continue
            out.append(Channel(name, key, kind, targets, r_lo, r_hi, n_coef, start))
            start += n_coef
    for p in sorted(skipped):
        log(f"[s5] pair {p}: no train distance inside the SK range, not corrected")
    if not out:
        raise ValueError("no SK channel to correct (check corrections.kinds/channels and the data)")
    return out


class Corrections:
    """The corrections c(r) of every channel and their action on the tables of an SK set."""

    def __init__(self, channels, form: str, files):
        self.channels = channels
        self.form = form
        self.n = sum(ch.n_coef for ch in channels)
        self.by_key = {}
        self.scale = {}
        for ch in channels:
            for key, j in ch.targets:
                f = files[key]
                basis = bspline_basis(_grid(f), ch.r_lo, ch.r_hi, ch.n_coef)
                self.by_key.setdefault(key, []).append((j, basis, ch.slice))
                s = float(np.abs(f.table[:, j]).max())
                self.scale[(key, j)] = s if s > 0.0 else 1.0

    def table(self, key, t0, c) -> np.ndarray:
        t0 = np.asarray(t0, float)
        t = t0.copy()
        c = np.asarray(c, float)
        for j, basis, sl in self.by_key.get(key, []):
            v = basis @ c[sl]
            t[:, j] = t0[:, j] * (1.0 + v) if self.form == "multiplicative" else \
                t0[:, j] + v * self.scale[(key, j)]
        return t

    def tables(self, files, c) -> dict:
        return {k: self.table(k, f.table, c) for k, f in files.items() if k in self.by_key}

    def skset(self, sk, c):
        """A copy of the SK set (pydftb.skf.SKSet) with corrected tables (as
        pydftb.autodiff.Params.to_skset does)."""
        out = copy.copy(sk)
        files = dict(sk.files)
        for key, f in sk.files.items():
            if key in self.by_key:
                files[key] = dataclasses.replace(f, table=self.table(key, f.table, c), _coef=None)
        out.files = files
        return out

    def torch_tables(self, sk, c_t) -> dict:
        """The corrected tables as torch tensors carrying the graph of ``c_t``."""
        import torch
        out = {}
        for key, f in sk.files.items():
            if key not in self.by_key:
                continue
            t0 = torch.as_tensor(np.asarray(f.table, float), dtype=torch.float64)
            cols = [t0[:, j] for j in range(t0.shape[1])]
            for j, basis, sl in self.by_key[key]:
                v = torch.as_tensor(basis, dtype=torch.float64) @ c_t[sl]
                cols[j] = t0[:, j] * (1.0 + v) if self.form == "multiplicative" else \
                    t0[:, j] + v * self.scale[(key, j)]
            out[key] = torch.stack(cols, dim=1)
        return out

    def penalty(self, c, ridge: float, smooth: float):
        """ridge |beta|^2 + smooth sum over channels |second differences|^2, and its gradient."""
        c = np.asarray(c, float)
        val = ridge * float(c @ c)
        grad = 2.0 * ridge * c
        for ch in self.channels:
            b = c[ch.slice]
            if len(b) < 3:
                continue
            d = b[2:] - 2.0 * b[1:-1] + b[:-2]
            val += smooth * float(d @ d)
            g = np.zeros_like(b)
            g[2:] += 2.0 * smooth * d
            g[1:-1] -= 4.0 * smooth * d
            g[:-2] += 2.0 * smooth * d
            grad[ch.slice] += g
        return val, grad


def write_sk_dir(src_dir: str, dst_dir: str, tables: dict, note: dict | None = None) -> str:
    """A copy of ``src_dir`` with the integral rows of the corrected files replaced.  Every other
    line (on-site, mass and polynomial, Spline) and every other file stay byte for byte; the
    written tables are read back and compared."""
    from pydftb.skf import SKFile, _expand_fortran, _open_text
    tmp = dst_dir + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for name in os.listdir(src_dir):
        p = os.path.join(src_dir, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(tmp, name))
    for (a, b), t in tables.items():
        with _open_text(os.path.join(src_dir, f"{a}-{b}.skf")) as fh:
            lines = fh.read().splitlines()
        off = 1 if lines[0].lstrip().startswith("@") else 0
        first = off + (3 if a == b else 2)          # grid line, (on-site line), mass line
        for k in range(t.shape[0]):
            vals = _expand_fortran(lines[first + k])
            vals[: t.shape[1]] = [float(v) for v in t[k]]
            lines[first + k] = " ".join(f"{v: .15e}" for v in vals)
        path = os.path.join(tmp, f"{a}-{b}.skf")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        back = SKFile.read(path)
        err = float(np.max(np.abs(back.table - t))) if t.size else 0.0
        if err > 1e-12 * max(1.0, float(np.max(np.abs(t)))):
            raise RuntimeError(f"{path}: the written table differs from the corrected one by {err}")
    with open(os.path.join(tmp, "s5.json"), "w") as fh:
        json.dump({"source": os.path.abspath(src_dir),
                   "corrected": sorted(f"{a}-{b}" for a, b in tables), **(note or {})}, fh, indent=1)
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.replace(tmp, dst_dir)
    return dst_dir


# ---- the problem (built in the parent and in every worker) -----------------------------------
class Problem:
    def __init__(self, s5_path: str, index=None, max_structures=None, overrides=None, log=print):
        from pydftb.skf import SKSet
        from utils import accuracy as acc
        self.cfg = _merge(load_s5(s5_path), overrides or {})
        self.overrides = dict(overrides or {})
        self.scan = acc.load_config(self.cfg["scan"])
        self.index = start_index(self.scan, self.cfg["start"]) if index is None else int(index)
        self.res, self.losses = point_records(self.scan, self.index)
        self.skdir = self.res["skdir"]
        self.dcfg = self.scan["dftb"]
        self.lcfg = self.scan["loss"]
        self.frames = acc.load_data(self.scan)
        if self.res["names"] != [a.info["name"] for a in self.frames]:
            raise RuntimeError(f"the data changed since point {self.index} ran (rerun the scan)")
        self.e0_ref = acc.load_e0s(self.scan)
        self.e0_pred = self.res["e0_dftb"]     # isolated atoms: no pairs, unchanged by S5
        self.roles = {r: [i for i, a in enumerate(self.frames) if a.info["role"] == r]
                      for r in ("train", "val", "test", "ood")}
        if max_structures:
            self.roles = {r: v[: int(max_structures)] for r, v in self.roles.items()}
        self.e_ref = np.array([a.get_potential_energy() for a in self.frames])
        self.f_ref = []
        for a in self.frames:
            try:
                self.f_ref.append(np.array(a.get_forces()))
            except Exception:                 # noqa: BLE001
                self.f_ref.append(None)
        syms = [a.get_chemical_symbols() for a in self.frames]
        self.a_ref = np.array([(e - sum(self.e0_ref[s] for s in sy)) / len(sy)
                               for e, sy in zip(self.e_ref, syms)])
        self.e0sum_pred = np.array([sum(self.e0_pred[s] for s in sy) for sy in syms])
        elements = sorted({s for sy in syms for s in sy})
        self.sk = SKSet(self.skdir, elements, max_l=self.dcfg.get("max_l"))
        shells = {el: list(self.sk.shells(el)) for el in elements}
        self._distances = {}
        d_min = self._min_distances(self.roles["train"], float(self.sk.cutoff))
        self.channels = build_channels(self.sk.files, shells, self.cfg["corrections"], d_min, log)
        self.corr = Corrections(self.channels, self.cfg["corrections"]["form"], self.sk.files)
        self.n = self.corr.n
        self._relevant = {}
        self._weights()

    # -- geometry ----------------------------------------------------------------------------
    def geometry(self, i):
        from pydftb.constants import BOHR
        a = self.frames[i]
        periodic = bool(np.any(a.pbc))
        return (a.positions / BOHR, a.cell.array / BOHR if periodic else None,
                a.pbc if periodic else None)

    def pair_distances(self, i, cutoff: float) -> dict:
        """{(A, B) sorted: distances (Bohr) up to ``cutoff``} of one structure."""
        from ase.data import atomic_numbers
        from pydftb.neighbors import neighbor_list
        key = (i, round(cutoff, 6))
        if key not in self._distances:
            a = self.frames[i]
            pos, cell, pbc = self.geometry(i)
            out = {}
            if len(a) > 1 or np.any(a.pbc):
                ii, jj, _, _, d = neighbor_list(pos, cell, pbc, cutoff)
                sym = np.array(a.get_chemical_symbols())
                order = {s: (atomic_numbers[s], s) for s in set(sym)}
                for x in set(sym):
                    for y in set(sym):
                        if order[x] > order[y]:
                            continue
                        m = ((sym[ii] == x) & (sym[jj] == y)) | ((sym[ii] == y) & (sym[jj] == x))
                        if m.any():
                            out[(x, y)] = d[m]
            self._distances[key] = out
        return self._distances[key]

    def _min_distances(self, idx, cutoff):
        out = {}
        for i in idx:
            for k, d in self.pair_distances(i, cutoff).items():
                out[k] = min(out.get(k, np.inf), float(d.min()))
        return out

    def relevant_coefficients(self, i) -> list:
        """The coefficients whose B-spline support holds a pair distance of structure i."""
        if i not in self._relevant:
            rc = max(ch.r_hi for ch in self.channels)
            dist = self.pair_distances(i, rc)
            js = []
            for ch in self.channels:
                d = dist.get(ch.pair, np.zeros(0))
                for m in range(ch.n_coef):
                    lo, hi = _support(ch, m)
                    if np.any((d > lo) & (d < hi)):
                        js.append(ch.start + m)
            self._relevant[i] = js
        return self._relevant[i]

    # -- the loss terms per structure --------------------------------------------------------
    def _weights(self):
        self.eps_e = float(self.lcfg.get("eps_e", 0.010))
        self.eps_f = float(self.lcfg.get("eps_f", 0.10))
        self.lam = float(self.lcfg.get("lambda_f", 1.0))
        fams = {}
        for i in self.roles["train"]:
            fams.setdefault(self.frames[i].info["family"], []).append(i)
        nf = max(1, len(fams))
        self.w_e, self.w_f = {}, {}
        for members in fams.values():
            withf = {i for i in members if self.f_ref[i] is not None and len(self.frames[i]) > 1}
            for i in members:
                self.w_e[i] = 1.0 / (nf * len(members))
                self.w_f[i] = 1.0 / (nf * len(withf)) if i in withf else 0.0

    def loss_np(self, i, e, f, rep_e, rep_f, delta) -> float:
        """Structure i's share of L_EF (|x| smoothed) for electronic e (eV), f (eV/A) plus a
        frozen repulsive."""
        n = len(self.frames[i])
        da = (e + rep_e - self.e0sum_pred[i]) / n - self.a_ref[i]
        val = self.w_e.get(i, 0.0) / self.eps_e * (np.sqrt(da * da + delta * delta) - delta)
        wf = self.w_f.get(i, 0.0)
        if wf > 0.0:
            df = f + rep_f - self.f_ref[i]
            val += wf * self.lam / self.eps_f * np.sqrt(np.mean(df * df))
        return float(val)

    # -- calculations ------------------------------------------------------------------------
    def engine(self, i, attempt: int):
        """(pydftb.DFTB, labels, SK set) of structure i with the scan's settings (attempt 1, 2:
        the scan's slower-mixing retries)."""
        from utils import accuracy as acc
        pc = acc.calculator(self.skdir, self.dcfg, self.frames[i], attempt)
        return pc._engine(self.frames[i])

    def value(self, i, c, attempt: int = 0):
        """E(sigma -> 0) in eV and forces in eV/A of structure i with corrected tables (as the
        scan: raw.e_zero x HARTREE_EV, forces x ase Hartree/Bohr as PyDFTB)."""
        from ase import units
        from utils.accuracy import HARTREE_EV
        eng, labels, sk = self.engine(i, attempt)
        eng = copy.copy(eng)
        eng.sk = self.corr.skset(sk, c)
        pos, cell, pbc = self.geometry(i)
        r = eng.calculate(labels, pos, cell, pbc, forces=True, stress=False)
        if not r.converged:
            raise RuntimeError("SCC not converged")
        f = np.asarray(r.forces, float) * (units.Hartree / units.Bohr)
        return float(r.e_zero) * HARTREE_EV, f

    def autodiff(self, i, c, attempt, rep_e, rep_f, delta):
        """(e, f, l_i, dl_i/dbeta) with pydftb.autodiff, mode "implicit"."""
        import torch
        from ase import units
        from pydftb.autodiff import Params, evaluate
        from utils.accuracy import HARTREE_EV
        eng, labels, sk = self.engine(i, attempt)
        c_t = torch.tensor(np.asarray(c, float), dtype=torch.float64, requires_grad=True)
        params = Params.from_skset(sk, eng.repulsive, eng.hubbard_derivs, eng.spin_constants)
        params.tables.update(self.corr.torch_tables(sk, c_t))
        pos, cell, pbc = self.geometry(i)
        r = evaluate(eng, labels, pos, cell, pbc, params, mode="implicit", forces=True,
                     stress=False)
        e_t = 0.5 * (r.energy + r.free_energy) * HARTREE_EV          # E - TS/2
        f_t = r.forces * (units.Hartree / units.Bohr)
        n = len(self.frames[i])
        da = (e_t + rep_e - float(self.e0sum_pred[i])) / n - float(self.a_ref[i])
        l_t = self.w_e.get(i, 0.0) / self.eps_e * (torch.sqrt(da * da + delta * delta) - delta)
        wf = self.w_f.get(i, 0.0)
        if wf > 0.0:
            df = f_t + torch.as_tensor(np.asarray(rep_f, float) - self.f_ref[i])
            l_t = l_t + wf * self.lam / self.eps_f * torch.sqrt((df * df).mean())
        g, = torch.autograd.grad(l_t, c_t, allow_unused=True)
        g = np.zeros(len(c)) if g is None else g.detach().numpy().astype(float)
        return float(e_t.detach()), f_t.detach().numpy().astype(float), float(l_t.detach()), g


_PROBLEM = None


def _init_worker(args):
    global _PROBLEM
    for var in BLAS_VARS:
        os.environ.setdefault(var, "1")
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:                         # noqa: BLE001
        pass
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:                         # noqa: BLE001
        pass
    _PROBLEM = Problem(*args, log=lambda *a: None)


def _task(args):
    """One structure: mode "value" (e, f), "loss" (+ l_i) or "grad" (+ dl_i/dbeta); the SCC is
    retried with the scan's slower mixings, then recorded as failed."""
    i, c, mode, rep_e, rep_f = args
    P = _PROBLEM
    opt = P.cfg["optimizer"]
    method, delta, h = opt["gradient"], float(opt["abs_smoothing"]), float(opt["fd_step"])
    if rep_f is None:
        rep_f = np.zeros((len(P.frames[i]), 3))
    t0 = time.perf_counter()
    err = None
    for attempt in range(3):
        try:
            g = l = None
            if mode == "grad" and method == "autodiff":
                e, f, l, g = P.autodiff(i, c, attempt, rep_e, rep_f, delta)
            else:
                e, f = P.value(i, c, attempt)
                if mode != "value":
                    l = P.loss_np(i, e, f, rep_e, rep_f, delta)
                if mode == "grad":
                    g = np.zeros(len(c))
                    for j in P.relevant_coefficients(i):
                        cp = np.array(c, float)
                        cp[j] += h
                        cm = np.array(c, float)
                        cm[j] -= h
                        lp = P.loss_np(i, *P.value(i, cp, attempt), rep_e, rep_f, delta)
                        lm = P.loss_np(i, *P.value(i, cm, attempt), rep_e, rep_f, delta)
                        g[j] = (lp - lm) / (2.0 * h)
            return {"i": i, "e": e, "f": f, "l": l, "grad": g, "attempts": attempt + 1,
                    "error": None, "seconds": time.perf_counter() - t0}
        except Exception as exc:              # noqa: BLE001  (retried, then recorded)
            err = f"{type(exc).__name__}: {exc}"
    return {"i": i, "e": np.nan, "f": None, "l": np.nan, "grad": None, "attempts": 3,
            "error": err, "seconds": time.perf_counter() - t0}


class Runner:
    """Structure tasks in ``jobs`` processes (fork on Linux, spawn elsewhere), or in-process."""

    def __init__(self, problem: Problem, max_structures, jobs: int):
        global _PROBLEM
        self.jobs = max(1, int(jobs))
        self.ex = None
        if self.jobs > 1:
            import concurrent.futures as cf
            import multiprocessing as mp
            for var in BLAS_VARS:                # inherited by the workers before numpy loads
                os.environ.setdefault(var, "1")
            ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
            self.ex = cf.ProcessPoolExecutor(
                self.jobs, mp_context=ctx, initializer=_init_worker,
                initargs=((problem.cfg["_path"], problem.index, max_structures,
                           problem.overrides),))
        else:
            _PROBLEM = problem

    def run(self, tasks) -> list:
        if self.ex is None:
            return [_task(t) for t in tasks]
        return list(self.ex.map(_task, tasks, chunksize=1))

    def close(self):
        if self.ex is not None:
            self.ex.shutdown()


def _collect(P: Problem, results):
    e = np.full(len(P.frames), np.nan)
    f = [None] * len(P.frames)
    for r in results:
        e[r["i"]] = r["e"]
        f[r["i"]] = r["f"]
    return e, f


# ---- the repulsive -------------------------------------------------------------------------
class Repulsive:
    """The repulsive refitted on ``fit_idx`` for these electronic energies and forces."""

    def __init__(self, kind: str, P: Problem, e_el, f_el, fit_idx):
        self.kind, self.P = kind, P
        if kind == "r2b":
            self._r2b(e_el, f_el, fit_idx)
        else:
            self._physrep(e_el, f_el, fit_idx)

    def _r2b(self, e_el, f_el, fit_idx):          # = utils.accuracy._repfit_control
        from pydftb.repfit import RepSample, fit_pair_splines
        P = self.P
        rcfg = P.scan["repfit"]
        if not rcfg.get("pairs"):
            raise ValueError("repulsive r2b needs the scan's repfit.pairs")
        pairs = {tuple(k.split("-")): tuple(v) for k, v in rcfg["pairs"].items()}
        samples = []
        for i in fit_idx:
            if not np.isfinite(e_el[i]):
                continue
            a = P.frames[i]
            sy = a.get_chemical_symbols()
            target = ((P.e_ref[i] - sum(P.e0_ref[s] for s in sy))
                      - (e_el[i] - sum(P.e0_pred[s] for s in sy)))
            tf = None if P.f_ref[i] is None or f_el[i] is None else P.f_ref[i] - f_el[i]
            samples.append(RepSample(a, target, tf, float(rcfg.get("weights", {}).get(
                a.info["family"], 1.0)), a.info["family"], a.info["name"]))
        self.fit = fit_pair_splines(samples, pairs, int(rcfg.get("n_knots", 12)),
                                    float(rcfg.get("smooth", 1e-3)),
                                    float(rcfg.get("force_weight", 1.0)),
                                    bool(rcfg.get("monotonic", True)), rcfg.get("offsets", "none"))

    def _physrep(self, e_el, f_el, fit_idx):     # = the scan's S1 sigma search
        from utils import accuracy as acc
        P = self.P
        pcfg = P.scan["physrep"]
        lo, hi = (float(v) for v in pcfg.get("sigma_bounds", [0.1, 0.8]))
        self.phys = acc._Physrep(P.frames, pcfg.get("fit", acc.space_of(P.scan).fit),
                                 pcfg.get("fixed", {}))

        def obj(s):
            er, fr = self.phys.evaluate(s, fit_idx)
            e = np.full(len(P.frames), np.nan)
            f = [None] * len(P.frames)
            for k, i in enumerate(fit_idx):
                e[i] = e_el[i] + er[k]
                f[i] = None if f_el[i] is None else f_el[i] + fr[k]
            return acc.loss_ef(P.frames, fit_idx, e, f, P.e_ref, P.f_ref, P.e0_ref, P.e0_pred,
                               P.lcfg)
        self.sigma, _, _ = acc.sigma_search(obj, len(self.phys.fit), lo, hi,
                                            int(pcfg.get("grid", 11)), int(pcfg.get("maxiter", 25)))

    def energy_forces(self, i):
        a = self.P.frames[i]
        if self.kind == "r2b":
            er, fr = self.fit.predict(a)
            return float(er + self.fit.offset_energy(a)), np.asarray(fr, float)
        er, fr = self.phys.evaluate(self.sigma, [i])
        return float(er[0]), np.asarray(fr[0], float)

    def save(self, path: str):
        if self.kind == "r2b":
            self.fit.to_json(path)
        else:
            with open(path, "w") as fh:
                json.dump({"kind": "physrep", "sigma": self.phys.sigma_dict(self.sigma)}, fh,
                          indent=1)

    def describe(self) -> dict:
        if self.kind == "r2b":
            return {"kind": "r2b", "stats": self.fit.stats}
        return {"kind": "physrep", "sigma": self.phys.sigma_dict(self.sigma)}


def totals(P: Problem, rep: Repulsive, e_el, f_el, idx):
    e = np.full(len(P.frames), np.nan)
    f = [None] * len(P.frames)
    for i in idx:
        if np.isfinite(e_el[i]):
            er, fr = rep.energy_forces(i)
            e[i] = e_el[i] + er
            f[i] = f_el[i] + fr
    return e, f


def full_objective(P: Problem, c, e_el, f_el):
    """(exact L_EF on train with the refitted repulsive, penalty, repulsive)."""
    from utils import accuracy as acc
    idx = P.roles["train"]
    pen, _ = P.corr.penalty(c, float(P.cfg["penalties"]["ridge"]),
                            float(P.cfg["penalties"]["smooth"]))
    if not np.all(np.isfinite(e_el[idx])):
        return np.nan, pen, None
    rep = Repulsive(P.cfg["repulsive"]["kind"], P, e_el, f_el, idx)
    e, f = totals(P, rep, e_el, f_el, idx)
    lef = acc.loss_ef(P.frames, idx, e, f, P.e_ref, P.f_ref, P.e0_ref, P.e0_pred, P.lcfg)
    return float(lef), float(pen), rep


def band_term(P: Problem, skdir: str):
    """The scan's band loss with these SK files (as utils.accuracy.evaluate_point), or None
    without band references."""
    from pydftb.bandloss import BandSettings, band_loss, compare, dftb_bands
    from utils import accuracy as acc
    refs = acc.load_refs(P.scan)
    if not refs:
        return None, []
    bset = BandSettings.from_dict(P.scan["bands"].get("settings"))
    out = []
    for ref in refs:
        at = ref.atoms()
        at.info["kspacing"] = ref.meta.get("kspacing", P.dcfg.get("kspacing", 0.15))
        try:
            calc = acc.calculator(skdir, P.dcfg, at)
            at.calc = calc
            out.append(compare(ref, dftb_bands(calc, at, ref), bset,
                               group=ref.meta.get("group") or ref.name))
        except Exception as exc:              # noqa: BLE001
            out.append({"name": ref.name, "group": ref.meta.get("group") or ref.name,
                        "loss": np.nan, "error": f"{type(exc).__name__}: {exc}"})
    lb = band_loss([{**b, "loss": np.nan if b.get("loss") is None else b["loss"]} for b in out])
    return float(lb), out


# ---- checkpoints ---------------------------------------------------------------------------
def _key(P: Problem, max_structures) -> str:
    keep = {k: v for k, v in P.cfg.items()
            if k in ("corrections", "penalties", "repulsive", "bands", "scan")}
    blob = json.dumps({"s5": keep, "start": P.index, "skdir": P.skdir,
                       "max_structures": max_structures,
                       "channels": [ch.name for ch in P.channels]}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _clean(obj):
    """JSON-ready: numpy scalars and arrays to Python, NaN/inf to None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def _atomic_json(path, obj):
    with open(path + ".tmp", "w") as fh:
        json.dump(_clean(obj), fh, indent=1)
    os.replace(path + ".tmp", path)


def step_dir(out: str, k: int) -> str:
    return os.path.join(out, "steps", f"k{int(k):03d}")


def write_step(P: Problem, out, k, c, e_el, f_el, rep, info, skdir=None) -> str:
    """steps/kNNN: the corrected SK set, the train electronic energies and forces, the
    repulsive, step.json."""
    d = step_dir(out, k)
    os.makedirs(d, exist_ok=True)
    if skdir is None:
        write_sk_dir(P.skdir, os.path.join(d, "skf"), P.corr.tables(P.sk.files, c),
                     {"s5_step": int(k), "form": P.corr.form})
    elif os.path.abspath(skdir) != os.path.abspath(os.path.join(d, "skf")):
        if os.path.exists(os.path.join(d, "skf")):
            shutil.rmtree(os.path.join(d, "skf"))
        shutil.move(skdir, os.path.join(d, "skf"))
    idx = P.roles["train"]
    np.savez_compressed(os.path.join(d, "train.npz"), index=np.array(idx),
                        energies=np.array([e_el[i] for i in idx]),
                        forces=np.concatenate([np.ravel(f_el[i]) if f_el[i] is not None
                                               else np.full(3 * len(P.frames[i]), np.nan)
                                               for i in idx]) if idx else np.zeros(0),
                        natoms=np.array([len(P.frames[i]) for i in idx]))
    if rep is not None:
        rep.save(os.path.join(d, "rep.json"))
    _atomic_json(os.path.join(d, "step.json"), {"k": int(k), "c": list(map(float, c)), **info})
    return d


def read_step_train(P: Problem, out, k):
    z = np.load(os.path.join(step_dir(out, k), "train.npz"))
    e = np.full(len(P.frames), np.nan)
    f = [None] * len(P.frames)
    off = 0
    for i, ei, n in zip(z["index"], z["energies"], z["natoms"]):
        e[int(i)] = ei
        f[int(i)] = z["forces"][off:off + 3 * n].reshape(n, 3)
        off += 3 * n
    return e, f


# ---- stages --------------------------------------------------------------------------------
def _check_method(method):
    if method == "autodiff":
        try:
            import torch  # noqa: F401
            from pydftb.autodiff import evaluate  # noqa: F401
        except ImportError as exc:
            raise SystemExit(f"optimizer.gradient autodiff needs torch ({exc}); install torch "
                             "(CPU is enough) or use --gradient fd")


def _channel_table(P: Problem) -> list:
    from pydftb.constants import BOHR
    return [{"name": ch.name, "files": [f"{a}-{b}.skf" for (a, b), _ in ch.targets],
             "range_A": [round(ch.r_lo * BOHR, 3), round(ch.r_hi * BOHR, 3)],
             "n_coef": ch.n_coef, "first": ch.start} for ch in P.channels]


def dry_run(cfg, args, log):
    P = Problem(cfg["_path"], None, args.max_structures)
    opt = cfg["optimizer"]
    train = P.roles["train"]
    t_el = float(P.res.get("seconds", {}).get("electronic") or np.nan)
    t_s = t_el / max(1, len(P.res["names"]))
    n_rel = np.mean([len(P.relevant_coefficients(i)) for i in train]) if train else 0.0
    per_value = t_s * len(train) / max(1, args.jobs)
    evals = int(opt["outer"]) * int(opt["inner"]) * (P.n + 1 if opt["gradient"] == "none" else 3)
    est = {"value": per_value, "fd": per_value * (1 + 2 * n_rel),
           "autodiff_low": 3 * per_value, "autodiff_high": 10 * per_value}
    out = {"start": P.index, "radii": P.res.get("radii"), "skdir": P.skdir,
           "n_parameters": P.n, "channels": _channel_table(P),
           "roles": {r: len(v) for r, v in P.roles.items()}, "jobs": args.jobs,
           "scan_seconds_electronic_all": t_el, "mean_relevant_coefficients": float(n_rel),
           "seconds_per_inner_evaluation": est, "inner_evaluations_at_most": evals,
           "gradient": opt["gradient"], "repulsive": cfg["repulsive"]["kind"]}
    print(f"[s5] start point {P.index} of {P.scan['name']} (radii {P.res.get('radii')})")
    print(f"[s5] {P.n} coefficients in {len(P.channels)} channels "
          f"(form {P.corr.form}, n_coef {cfg['corrections']['n_coef']}):")
    for row in out["channels"]:
        print(f"    {row['name']:<16} {','.join(row['files']):<22} r {row['range_A'][0]:.2f}–"
              f"{row['range_A'][1]:.2f} A  beta[{row['first']}:{row['first'] + row['n_coef']}]")
    print(f"[s5] structures: " + ", ".join(f"{r} {n}" for r, n in out["roles"].items()))
    print(f"[s5] one inner evaluation on {len(train)} train structures, {args.jobs} process(es): "
          f"value {per_value:.1f} s; fd {est['fd']:.1f} s (mean {n_rel:.1f} coefficients per "
          f"structure); autodiff about {est['autodiff_low']:.1f}–{est['autodiff_high']:.1f} s "
          "(measure with gradcheck)")
    print(f"[s5] at most {evals} inner evaluations ({opt['outer']} outer steps, "
          f"{opt['inner']} inner iterations each)")
    if P.losses:
        s1 = (P.losses.get("strategies", {}).get("S1") or {}).get("summary", {})
        r2b = (P.losses.get("R2b") or {}).get("summary", {})
        print(f"[s5] the scan at this point: S1 tier {s1.get('tier')}, R2b tier {r2b.get('tier')}")
    log.save_json("dry_run.json", out)
    return out


def gradcheck(cfg, args, log):
    """The configured gradient against central differences of the same per-structure terms."""
    global _PROBLEM
    method = args.gradient or cfg["optimizer"]["gradient"]
    _check_method(method)
    P = Problem(cfg["_path"], None, args.max_structures, {"optimizer": {"gradient": method}})
    _PROBLEM = P
    idx = P.roles["train"][: args.structures]
    rel = sorted(set().union(*[P.relevant_coefficients(i) for i in idx]))[: args.coefs]
    rng = np.random.default_rng(0)
    c = np.zeros(P.n)
    c[rel] = rng.uniform(-0.05, 0.05, len(rel))
    delta = float(cfg["optimizer"]["abs_smoothing"])
    h = float(args.step)
    rows = []
    for i in idx:
        zero = np.zeros((len(P.frames[i]), 3))
        t0 = time.perf_counter()
        r = _task((i, c, "value", 0.0, zero))
        t_value = time.perf_counter() - t0
        t0 = time.perf_counter()
        r = _task((i, c, "grad", 0.0, zero))
        t_grad = time.perf_counter() - t0
        if r["error"]:
            raise RuntimeError(f"{P.frames[i].info['name']}: {r['error']}")
        for j in rel:
            cp = c.copy()
            cp[j] += h
            cm = c.copy()
            cm[j] -= h
            lp = P.loss_np(i, *P.value(i, cp), 0.0, zero, delta)
            lm = P.loss_np(i, *P.value(i, cm), 0.0, zero, delta)
            fd = (lp - lm) / (2 * h)
            channel = next(ch.name for ch in P.channels if ch.start <= j < ch.start + ch.n_coef)
            rows.append({"structure": P.frames[i].info["name"], "coef": int(j), "channel": channel,
                         "gradient": float(r["grad"][j]), "central_difference": fd,
                         "abs_err": abs(float(r["grad"][j]) - fd)})
        rows[-1]["seconds_value"] = t_value
        rows[-1]["seconds_gradient"] = t_grad
        log.event("gradcheck", structure=P.frames[i].info["name"], seconds_value=t_value,
                  seconds_gradient=t_grad)
    scale = max(1e-12, max(abs(r["central_difference"]) for r in rows))
    worst = max(r["abs_err"] for r in rows) / scale
    print(f"[s5] gradcheck ({method}, step {h}): {len(rows)} entries, max |error| / max |gradient| "
          f"= {worst:.2e}")
    for r in rows:
        print(f"    {r['structure']:<28} {r['channel']:<14} beta[{r['coef']}]  "
              f"{r['gradient']: .6e}  {r['central_difference']: .6e}")
    out = {"method": method, "step": h, "rows": rows, "relative_error": worst}
    log.save_json("gradcheck.json", out)
    os.makedirs(cfg["out"], exist_ok=True)
    _atomic_json(os.path.join(cfg["out"], f"gradcheck_{method}.json"), out)
    return out


def fit(cfg, args, log):
    from scipy.optimize import minimize
    from utils import accuracy as acc
    if args.gradient:
        cfg["optimizer"]["gradient"] = args.gradient
    P = Problem(cfg["_path"], None, args.max_structures,
                {"optimizer": {"gradient": args.gradient}} if args.gradient else None)
    opt = cfg["optimizer"]
    method = opt["gradient"]
    _check_method(method)
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    key = _key(P, args.max_structures)
    state_path = os.path.join(out, "state.json")
    state = None
    if os.path.exists(state_path):
        with open(state_path) as fh:
            state = json.load(fh)
        if state.get("key") != key:
            if not args.restart:
                raise SystemExit(f"{state_path} belongs to another setting (config, start point or "
                                 "channels changed): use --restart to start over")
            os.replace(state_path, state_path + f".{int(time.time())}.bak")
            state = None
    train = P.roles["train"]
    ridge, smooth = float(cfg["penalties"]["ridge"]), float(cfg["penalties"]["smooth"])
    beta_max = float(cfg["corrections"].get("beta_max", 0.5))
    wb = float(cfg["bands"].get("weight", 0.0))
    outer = int(args.outer if args.outer is not None else opt["outer"])
    runner = Runner(P, args.max_structures, args.jobs)
    try:
        if state is None:
            c = np.zeros(P.n)
            with log.step("start", n_train=len(train), n_coef=P.n, method=method):
                rs = runner.run([(i, c, "value", 0.0, None) for i in train])
            e_el, f_el = _collect(P, rs)
            bad = [P.frames[r["i"]].info["name"] for r in rs if r["error"]]
            if bad:
                raise SystemExit(f"SCC failed at the start point for {bad[:5]}")
            _, e_scan, f_scan, _ = acc._point_records(P.scan, P.index)
            de = float(np.nanmax(np.abs(e_el[train] - e_scan[train])))
            dfm = float(max(np.abs(f_el[i] - f_scan[i]).max() for i in train))
            log.event("start_check", max_de_eV=de, max_df_eVA=dfm)
            print(f"[s5] start: the uncorrected tables reproduce the scan's electronic part to "
                  f"{de:.1e} eV, {dfm:.1e} eV/A")
            lef, pen, rep = full_objective(P, c, e_el, f_el)
            s1 = ((P.losses or {}).get("strategies", {}).get("S1") or {}).get("loss_raw")
            if cfg["repulsive"]["kind"] == "physrep" and s1 is not None:
                print(f"[s5] start: L_EF {lef:.6f} with physrep; the scan's S1 at this point: "
                      f"{s1:.6f} (equal unless --max-structures)")
                log.event("start_physrep_check", l_ef=lef, scan_s1=s1)
            band = band_term(P, P.skdir)[0] if wb > 0 else None
            L = lef + pen + (wb * band if band is not None else 0.0)
            write_step(P, out, 0, c, e_el, f_el, rep, {"L": L, "L_ef": lef, "penalty": pen,
                                                       "band": band, "repulsive": rep.describe()})
            state = {"key": key, "start": P.index, "skdir": P.skdir, "n": P.n,
                     "channels": _channel_table(P), "c": c.tolist(), "L": L, "L_ef": lef,
                     "penalty": pen, "band": band, "trust": float(opt["trust"]), "outer_done": 0,
                     "best_step": 0, "history": [{"k": 0, "accepted": True, "L": L, "L_ef": lef}]}
            _atomic_json(state_path, state)
            print(f"[s5] step 0: L_EF {lef:.4f} (penalty {pen:.2e}, band {band})")
        c = np.array(state["c"], float)
        trust = float(state["trust"])
        for k in range(int(state["outer_done"]) + 1, outer + 1):
            t_outer = time.perf_counter()
            e_el, f_el = read_step_train(P, out, state["best_step"])
            rep = Repulsive(cfg["repulsive"]["kind"], P, e_el, f_el, train)
            frozen = {i: rep.energy_forces(i) for i in train}
            cache, n_eval = {}, [0]
            mode = "loss" if method == "none" else "grad"

            def fun(x):
                x = np.clip(np.asarray(x, float), -beta_max, beta_max)
                t0 = time.perf_counter()
                rs = runner.run([(i, x, mode, frozen[i][0], frozen[i][1]) for i in train])
                n_eval[0] += 1
                ee, ff = _collect(P, rs)
                if len(cache) > 8:
                    cache.clear()
                cache[x.tobytes()] = (ee, ff)
                pen, gpen = P.corr.penalty(x, ridge, smooth)
                failed = [r for r in rs if r["error"]]
                val = 1e6 if failed else float(sum(r["l"] for r in rs)) + pen
                log.event("inner", k=k, n=n_eval[0], value=val, failed=len(failed),
                          seconds=round(time.perf_counter() - t0, 2))
                if mode == "loss":
                    return val
                if failed:
                    return val, np.zeros_like(x)
                return val, np.sum([r["grad"] for r in rs], axis=0) + gpen

            bounds = [(max(-beta_max, cj - trust), min(beta_max, cj + trust)) for cj in c]
            inner = int(opt["inner"])
            if method == "none":
                res = minimize(fun, c, method="Powell", bounds=bounds,
                               options={"maxfev": inner * (P.n + 1), "xtol": 1e-4, "ftol": 1e-8})
            else:
                res = minimize(fun, c, jac=True, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": inner, "maxfun": 3 * inner})
            x = np.clip(np.asarray(res.x, float), -beta_max, beta_max)
            if x.tobytes() in cache:
                ee, ff = cache[x.tobytes()]
            else:
                ee, ff = _collect(P, runner.run([(i, x, "value", 0.0, None) for i in train]))
            lef, pen, rep_new = full_objective(P, x, ee, ff)
            band, try_dir = None, None
            if wb > 0 and np.isfinite(lef):
                try_dir = write_sk_dir(P.skdir, os.path.join(out, "steps", "try", "skf"),
                                       P.corr.tables(P.sk.files, x))
                band = band_term(P, try_dir)[0]
            L_try = lef + pen + (wb * band if band is not None else 0.0)
            accepted = bool(np.isfinite(L_try) and L_try < state["L"] - float(opt["accept_tol"]))
            rec = {"k": k, "accepted": accepted, "L_try": L_try, "L_ef_try": lef, "penalty": pen,
                   "band": band, "trust": trust, "inner_evaluations": n_eval[0],
                   "optimizer": str(res.message), "seconds": round(time.perf_counter() - t_outer, 1)}
            if accepted:
                write_step(P, out, k, x, ee, ff, rep_new,
                           {"L": L_try, "L_ef": lef, "penalty": pen, "band": band,
                            "repulsive": rep_new.describe()}, skdir=try_dir)
                c = x
                state.update(c=c.tolist(), L=L_try, L_ef=lef, penalty=pen, band=band, best_step=k)
                trust = min(trust * 1.5, float(opt["trust_max"]))
            else:
                trust *= 0.5
                if try_dir and os.path.exists(try_dir):
                    shutil.rmtree(try_dir)
            state["history"].append(rec)
            state.update(trust=trust, outer_done=k)
            _atomic_json(state_path, state)
            log.event("outer", **rec)
            print(f"[s5] step {k}: {'accepted' if accepted else 'rejected'}  L_EF {lef:.4f}  "
                  f"(best {state['L_ef']:.4f})  trust {trust:.3g}  {n_eval[0]} evaluations")
            if trust < float(opt["trust_min"]):
                print("[s5] trust radius below trust_min: stopped")
                break
    finally:
        runner.close()
    log.save_json("state.json", state)
    return state


def _role_row(summ: dict | None, role: str) -> dict:
    b = ((summ or {}).get("energies", summ or {}).get("by_role") or {}).get(role) or {}
    return {k: b.get(k) for k in ("relative_mae", "atomization_mae", "force_mae", "n", "failed")}


def _decisive(summ):
    rx = [r for r in (summ or {}).get("reactions", []) if r.get("decisive")]
    return f"{sum(bool(r.get('sign_ok')) for r in rx)}/{len(rx)}" if rx else "—"


def _fmt(v, scale=1.0, digits=1):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else \
        f"{v * scale:.{digits}f}"


def _summary(P, e, f, idx, errors=None):
    from pydftb import assess
    reactions = assess.load_reactions(P.scan["reactions"]) if P.scan.get("reactions") else None
    frames = [P.frames[i] for i in idx]
    rec = assess.records_from_arrays(frames, np.array([e[i] for i in idx]), [f[i] for i in idx],
                                     None if errors is None else [errors.get(i) for i in idx])
    return assess.summarize(rec, P.e0_ref, P.e0_pred, reactions)


def report(cfg, args, log):
    from utils import accuracy as acc
    P = Problem(cfg["_path"], None, args.max_structures)
    out = cfg["out"]
    with open(os.path.join(out, "state.json")) as fh:
        state = json.load(fh)
    k = int(state["best_step"]) if args.step == "best" else int(args.step)
    with open(os.path.join(step_dir(out, k), "step.json")) as fh:
        step = json.load(fh)
    c = np.array(step["c"], float)
    skdir = os.path.join(step_dir(out, k), "skf")
    kind = cfg["repulsive"]["kind"]
    idx = sorted({i for r in ("train", "val", "test", "ood") for i in P.roles[r]})
    runner = Runner(P, args.max_structures, args.jobs)
    try:
        with log.step("electronic", n=len(idx), step=k):
            rs = runner.run([(i, c, "value", 0.0, None) for i in idx])
    finally:
        runner.close()
    e_el, f_el = _collect(P, rs)
    errors = {r["i"]: r["error"] for r in rs if r["error"]}
    # the written files reproduce the in-memory corrections (the scan's own code path)
    checks = []
    for i in P.roles["train"][:2]:
        ef, ff, _, err = acc.electronic(skdir, P.dcfg, P.frames[i])
        if err is None:
            checks.append({"structure": P.frames[i].info["name"], "de_eV": abs(ef - e_el[i]),
                           "df_eVA": float(np.abs(ff - f_el[i]).max())})
    rows = {}
    rep = Repulsive(kind, P, e_el, f_el, P.roles["train"])
    e, f = totals(P, rep, e_el, f_el, idx)
    rows["S5 (repulsive on train)"] = _summary(P, e, f, idx, errors)
    orc = Repulsive(kind, P, e_el, f_el, idx)
    e, f = totals(P, orc, e_el, f_el, idx)
    rows["S5 oracle (repulsive on all)"] = _summary(P, e, f, idx, errors)
    _, e_scan, f_scan, _ = acc._point_records(P.scan, P.index)
    rep0 = Repulsive(kind, P, e_scan, f_scan, P.roles["train"])
    e, f = totals(P, rep0, e_scan, f_scan, idx)
    rows[f"start tables, same refit ({kind})"] = _summary(P, e, f, idx)
    scan_rows = {}
    if P.losses:
        scan_rows["scan S1 at the start point"] = (P.losses.get("strategies", {}).get("S1") or {}).get("summary")
        scan_rows["scan R2b at the start point"] = (P.losses.get("R2b") or {}).get("summary")
        scan_rows["scan oracle at the start point"] = (P.losses.get("oracle") or {}).get("summary")
    best = acc.best_points(acc._scored(P.scan), ["S1"]).get("S1")
    if best is not None and int(best["index"]) != P.index:
        scan_rows[f"scan best S1 (point {best['index']})"] = best["strategies"]["S1"].get("summary")
    band, band_rows = band_term(P, skdir)
    band0 = (P.losses or {}).get("bands", {}).get("loss")
    result = {"start": P.index, "step": k, "n_parameters": P.n, "repulsive": kind,
              "channels": _channel_table(P), "file_check": checks, "band_loss": band,
              "band_loss_start": band0, "band_results": band_rows, "history": state.get("history"),
              "rows": {n: {"tier": s.get("tier") if s else None,
                           **{r: _role_row(s, r) for r in ("train", "test", "ood")},
                           "decisive": _decisive(s)} for n, s in {**rows, **scan_rows}.items()},
              "summaries": rows, "max_structures": args.max_structures}
    _atomic_json(os.path.join(out, "s5_report.json"), result)
    lines = [f"# S5: SK tables refitted ({P.scan['name']}, start point {P.index}, step {k})", "",
             f"{P.n} coefficients in {len(P.channels)} channels, form {P.corr.form}, repulsive "
             f"{kind} refitted on train; gradient {cfg['optimizer']['gradient']}.", ""]
    if args.max_structures:
        lines += [f"**Smoke run: only the first {args.max_structures} structures of each role.**", ""]
    lines += ["| model | tier | train rel | test rel | ood rel | train F | test F | ood F | decisive |",
              "|---|---|---|---|---|---|---|---|---|"]
    for n, r in result["rows"].items():
        lines.append(f"| {n} | {r['tier'] or '—'} | " + " | ".join(
            _fmt(r[role]["relative_mae"], 1e3) for role in ("train", "test", "ood")) + " | " +
            " | ".join(_fmt(r[role]["force_mae"], 1.0, 3) for role in ("train", "test", "ood")) +
            f" | {r['decisive']} |")
    lines += ["", "Energies: within-family relative MAE in meV/atom; forces: MAE in eV/Å; tier "
              "on test and ood (pydftb.assess).", ""]
    lines.append(f"Band loss: start {_fmt(band0, 1.0, 4)}, S5 {_fmt(band, 1.0, 4)}.")
    if checks:
        lines.append("File check (written tables vs in-memory corrections): max "
                     f"{max(x['de_eV'] for x in checks):.1e} eV, "
                     f"{max(x['df_eVA'] for x in checks):.1e} eV/Å.")
    lines += ["", "Reading (A0 section 3, S5): S5 at tier A/B on test/ood where the scan's R2b "
              "row is not means the confinement-radius parameterisation limits A1.  S5 no better "
              "than R2b means the two-centre form with a two-body repulsive is the limit (go to "
              "A2/A3).  Better on train only means overfitting: raise the penalties or narrow the "
              "ranges."]
    with open(os.path.join(out, "s5_report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return result


def export(cfg, args, log):
    """A model directory like utils.accuracy.export_model: the step's corrected tables with the
    repulsive (R2b or physrep) as Spline blocks, and model.json."""
    from ase.data import atomic_numbers
    from pydftb.constants import BOHR, HARTREE
    from pydftb.repfit import RepFit, skf_block_from_grid, write_spline_blocks
    P = Problem(cfg["_path"], None, args.max_structures)
    out = cfg["out"]
    with open(os.path.join(out, "state.json")) as fh:
        state = json.load(fh)
    k = int(state["best_step"]) if args.step == "best" else int(args.step)
    d = step_dir(out, k)
    src = os.path.join(d, "skf")
    kind = cfg["repulsive"]["kind"]
    dest = os.path.abspath(args.dest)
    if kind == "r2b":
        RepFit.from_json(os.path.join(d, "rep.json")).write_skf(src, dest)
        sigma = None
    else:
        from utils import accuracy as acc
        with open(os.path.join(d, "rep.json")) as fh:
            sigma = json.load(fh)["sigma"]
        els = sorted(sigma, key=lambda s: atomic_numbers[s])
        pr = acc._Physrep(P.frames, [], sigma).model([])
        blocks = {}
        for i, a in enumerate(els):
            for b in els[: i + 1]:
                key = f"{atomic_numbers[a]}_{atomic_numbers[b]}"
                r = np.linspace(0.6, float(pr.rcuts[key]) * BOHR, 200)
                blocks[(a, b)] = skf_block_from_grid(r, np.array(pr.Vreps[key](r / BOHR)) * HARTREE)
        write_spline_blocks(src, dest, blocks)
    with open(os.path.join(dest, "model.json"), "w") as fh:
        json.dump({"scan": P.scan["name"], "index": P.index, "strategy": "S5", "repulsive": kind,
                   "radii": P.res.get("radii"), "sigma": sigma, "max_l": P.dcfg.get("max_l"),
                   "skdir": src, "s5": {"step": k, "config": cfg["_path"], "n_parameters": P.n}},
                  fh, indent=1)
    print(f"[s5] step {k} -> {dest} (use with PyDFTB(dest, repulsive='skf'), e.g. md_check.py --model)")
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["dry-run", "gradcheck", "fit", "report", "export"])
    ap.add_argument("config", help="the S5 config (e.g. s5_li_o.yaml)")
    ap.add_argument("--jobs", type=int, default=1, help="processes (one structure each)")
    ap.add_argument("--max-structures", type=int, default=None,
                    help="first N structures of every role (smoke tests)")
    ap.add_argument("--gradient", choices=GRADIENTS, default=None, help="override optimizer.gradient")
    ap.add_argument("--outer", type=int, default=None, help="override optimizer.outer")
    ap.add_argument("--restart", action="store_true", help="fit: start over (keeps a .bak state)")
    ap.add_argument("--step", default="best", help="report/export: a step number or 'best'")
    ap.add_argument("--dest", default=None, help="export: the model directory")
    ap.add_argument("--structures", type=int, default=2, help="gradcheck: train structures")
    ap.add_argument("--coefs", type=int, default=4, help="gradcheck: coefficients")
    ap.add_argument("--fd-check-step", dest="step_check", type=float, default=1e-4,
                    help="gradcheck: central-difference step")
    args = ap.parse_args(argv)
    cfg = load_s5(args.config)
    if args.stage == "export" and not args.dest:
        ap.error("export needs --dest")
    if args.stage == "gradcheck":
        args.step = args.step_check
    from utils.runlog import session
    name = f"s5-{args.stage}"
    with session(name, vars(args), root=os.path.join(cfg["out"], "logs")) as log:
        stage = {"dry-run": dry_run, "gradcheck": gradcheck, "fit": fit, "report": report,
                 "export": export}[args.stage]
        stage(cfg, args, log)


if __name__ == "__main__":
    main()
