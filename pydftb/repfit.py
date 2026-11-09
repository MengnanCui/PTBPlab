"""Two-body spline repulsives by linear least squares, for several element pairs (A series).

This is the R₂b control and the oracle of A1 (design ``docs/design/A0-accuracy-programme.html``,
section 3), and the R2 repulsive of A3.  It generalises ``pydftb.bench.cu_joint_repulsive``
(one Cu-Cu pair) to every element pair of a data set; that bench script stays as it is.

For each pair (A, B), V_AB(r) is a cubic B-spline on [r_lo, r_cut] whose value and slope vanish
at r_cut.  Below r_lo it continues as the exponential exp(-a1 r + a2) that matches the value and
slope at r_lo; this is the head of the SKF ``Spline`` block.  For fixed electronic energies and
forces (the SK tables with ``repulsive="none"``) the coefficients solve one linear problem:

    energy rows  sqrt(w_s) [ sum_pairs V(r)/N_s + sum_X n_X,s mu_X / N_s - y_s ]
    force rows   sqrt(w_s) * force_weight / sqrt(3 N_s) * [ F_rep,s(c) - (F_DFT - F_el)_s ]
    smoothness   sqrt(smooth) * second differences of each pair's coefficients

Here y_s = (E_DFT - E_el)_s / N_s is in eV/atom, with the same atomic references on both sides
(the caller's choice: atomization energies normally).  The offsets mu_X are fitted only with
``offsets="per_element"``; ``"global"`` fits one constant and ``"none"`` none.
``monotonic`` writes c = U p with p >= 0, which makes every V_AB non-negative and
non-increasing (a B-spline's slope is a positive combination of its coefficient differences).

``RepFit.predict(atoms)`` gives (E_rep in eV, F_rep in eV/Å); ``RepFit.write_skf`` writes the
splines into copies of SK files (pydftb and DFTB+ read them with ``repulsive="skf"``).
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BSpline, CubicSpline

from .constants import BOHR, HARTREE


@dataclass
class RepSample:
    """One structure: the part of the DFT energy and forces the repulsive must give."""
    atoms: object                         # ase.Atoms
    target_energy: float                  # eV, total: E_ref - E_el (same atomic references)
    target_forces: np.ndarray | None = None   # eV/Å, (N, 3): F_ref - F_el
    weight: float = 1.0
    group: str = ""
    name: str = ""


def pair_key(a: str, b: str) -> tuple:
    return tuple(sorted((a, b)))


def basis(r_lo: float, r_cut: float, n_knots: int):
    """Cubic B-splines on [r_lo, r_cut] whose value and slope vanish at r_cut:
    (knots, number of splines, kept indices)."""
    inner = np.linspace(r_lo, r_cut, n_knots)
    t = np.concatenate([[r_lo] * 3, inner, [r_cut] * 3])
    n = len(t) - 4
    return t, n, list(range(n - 2))


def _unit(t, n, k):
    c = np.zeros(n)
    c[k] = 1.0
    return BSpline(t, c, 3, extrapolate=False)


def _neighbours(atoms, cutoff):
    from ase.neighborlist import neighbor_list
    i, j, d, vec = neighbor_list("ijdD", atoms, cutoff)
    return i, j, d, vec


@dataclass
class PairSpline:
    """V(r) of one pair (eV, Å), with its exponential head below r_lo."""
    pair: tuple
    r_lo: float
    r_cut: float
    knots: np.ndarray
    coef: np.ndarray                      # full B-spline coefficient vector

    def _spline(self):
        return BSpline(self.knots, self.coef, 3, extrapolate=False)

    def head(self):
        """(a1, a2) of exp(-a1 r + a2) matching value and slope at r_lo (None if not repulsive)."""
        spl = self._spline()
        v0 = float(np.nan_to_num(spl(self.r_lo)))
        d1 = float(np.nan_to_num(spl.derivative()(self.r_lo)))
        if not (v0 > 0 and d1 < 0):
            return None
        a1 = -d1 / v0
        return a1, np.log(v0) + a1 * self.r_lo

    def value(self, r, derivative: int = 0):
        r = np.asarray(r, float)
        spl = self._spline()
        out = np.nan_to_num(spl(r) if derivative == 0 else spl.derivative()(r))
        low = r < self.r_lo
        if np.any(low):
            h = self.head()
            if h is None:                 # linear continuation of value and slope
                v0 = float(np.nan_to_num(spl(self.r_lo)))
                d1 = float(np.nan_to_num(spl.derivative()(self.r_lo)))
                out[low] = v0 + d1 * (r[low] - self.r_lo) if derivative == 0 else d1
            else:
                a1, a2 = h
                ex = np.exp(-a1 * r[low] + a2)
                out[low] = ex if derivative == 0 else -a1 * ex
        out[r >= self.r_cut] = 0.0
        return out

    def to_dict(self):
        return {"pair": list(self.pair), "r_lo": self.r_lo, "r_cut": self.r_cut,
                "knots": self.knots.tolist(), "coef": self.coef.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(d["pair"]), float(d["r_lo"]), float(d["r_cut"]),
                   np.asarray(d["knots"], float), np.asarray(d["coef"], float))

    def skf_block(self, grid_points: int = 120) -> list:
        """The SKF ``Spline`` block (Hartree, Bohr) of this pair (:func:`skf_block_from_grid`)."""
        grid = np.linspace(self.r_lo, self.r_cut, grid_points)
        try:
            return skf_block_from_grid(grid, self.value(grid))
        except ValueError as exc:
            raise ValueError(f"{self.pair}: {exc}; use monotonic=True or a smaller r_lo") from None


def skf_block_from_grid(r, v) -> list:
    """The SKF ``Spline`` block (Hartree, Bohr) of V(r) sampled on an increasing grid (Å, eV)
    whose last point is the cut-off: cubic pieces with zero value and slope at the cut-off and
    an exponential head exp(-a1 r + a2) matching value and slope at the first point (as
    ``pydftb.bench.cu_dimer_repulsive.spline_block``)."""
    r = np.asarray(r, float)
    v = np.array(v, float)
    v[-1] = 0.0
    spl = CubicSpline(r / BOHR, v / HARTREE, bc_type=("not-a-knot", (1, 0.0)))
    x, c = spl.x, spl.c
    v0, d1 = float(spl(x[0])), float(spl(x[0], 1))
    if not (v0 > 0.0 and d1 < 0.0):
        raise ValueError(f"no repulsive exponential head at r = {r[0]:.3f} A "
                         f"(value {v0}, slope {d1})")
    a1 = -d1 / v0
    a2 = np.log(v0) + a1 * x[0]
    f = repr
    lines = ["Spline", f"{len(x) - 1} {f(float(x[-1]))}", f"{f(float(a1))} {f(float(a2))} 0.0"]
    for i in range(len(x) - 1):
        cc = [c[3, i], c[2, i], c[1, i], c[0, i]]
        tail = " 0.0 0.0" if i == len(x) - 2 else ""
        lines.append(f"{f(float(x[i]))} {f(float(x[i + 1]))} "
                     + " ".join(f(float(q)) for q in cc) + tail)
    return lines


def write_spline_blocks(src_dir: str, dst_dir: str, blocks: dict,
                        template: str = "{a}-{b}.skf") -> list:
    """Copy the SK files of ``src_dir`` to ``dst_dir`` and put ``blocks[(a, b)]`` (lines of a
    Spline block) into both files of each pair, replacing an existing Spline block (up to
    ``<Documentation>`` or the end) or appending one."""
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        if name.endswith(".skf"):
            shutil.copy2(os.path.join(src_dir, name), os.path.join(dst_dir, name))
    written = []
    for (a, b), block in blocks.items():
        for x, y in {(a, b), (b, a)}:
            path = os.path.join(dst_dir, template.format(a=x, b=y))
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with open(path) as fh:
                lines = fh.read().split("\n")
            i = next((k for k, ln in enumerate(lines) if ln.strip().startswith("Spline")), None)
            j = next((k for k, ln in enumerate(lines)
                      if ln.strip().startswith("<Documentation>")), len(lines))
            new = (lines[:i] + block + lines[j:]) if i is not None else (
                lines[:j] + block + lines[j:])
            with open(path, "w") as fh:
                fh.write("\n".join(new).rstrip("\n") + "\n")
            written.append(path)
    return written


@dataclass
class RepFit:
    """The fitted splines, offsets and fit statistics."""
    splines: dict                          # pair -> PairSpline
    offsets: dict = field(default_factory=dict)   # element -> eV/atom (or "_global")
    settings: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    @property
    def cutoff(self) -> float:
        return max(s.r_cut for s in self.splines.values())

    def predict(self, atoms):
        """(E_rep eV, F_rep eV/Å (N, 3)) of one structure (no offsets)."""
        i, j, d, vec = _neighbours(atoms, self.cutoff)
        sym = np.array(atoms.get_chemical_symbols())
        e, f = 0.0, np.zeros((len(atoms), 3))
        for key, sp in self.splines.items():
            m = ((sym[i] == key[0]) & (sym[j] == key[1])) | ((sym[i] == key[1]) & (sym[j] == key[0]))
            if key[0] == key[1]:
                m = (sym[i] == key[0]) & (sym[j] == key[0])
            if not m.any():
                continue
            v = sp.value(d[m])
            dv = sp.value(d[m], 1)
            e += 0.5 * v.sum()
            np.add.at(f, i[m], (dv / d[m])[:, None] * vec[m])
        return float(e), f

    def offset_energy(self, atoms) -> float:
        """The fitted offsets of a structure (eV, total)."""
        if not self.offsets:
            return 0.0
        if "_global" in self.offsets:
            return self.offsets["_global"] * len(atoms)
        return float(sum(self.offsets.get(s, 0.0) for s in atoms.get_chemical_symbols()))

    def to_json(self, path: str) -> str:
        with open(path, "w") as fh:
            json.dump({"splines": [s.to_dict() for s in self.splines.values()],
                       "offsets": self.offsets, "settings": self.settings, "stats": self.stats},
                      fh, indent=1)
        return path

    @classmethod
    def from_json(cls, path: str) -> "RepFit":
        with open(path) as fh:
            d = json.load(fh)
        sp = [PairSpline.from_dict(s) for s in d["splines"]]
        return cls({s.pair: s for s in sp}, d.get("offsets", {}), d.get("settings", {}),
                   d.get("stats", {}))

    def write_skf(self, src_dir: str, dst_dir: str, template: str = "{a}-{b}.skf") -> list:
        """Copy every SK file of ``src_dir`` to ``dst_dir`` and put each fitted pair's Spline
        block into both of its files (pairs absent from the data keep their files)."""
        blocks = {pair: sp.skf_block() for pair, sp in self.splines.items() if np.any(sp.coef)}
        return write_spline_blocks(src_dir, dst_dir, blocks, template)


def _design(samples, pairs, n_knots, force_weight):
    """Energy rows (per atom), force rows, and the column layout."""
    layout, col = {}, 0
    for key, (r_lo, r_cut) in pairs.items():
        t, n, keep = basis(r_lo, r_cut, n_knots)
        layout[key] = (t, n, keep, col)
        col += len(keep)
    ncol = col
    cutoff = max(rc for _, rc in pairs.values())
    a_e = np.zeros((len(samples), ncol))
    f_blocks, f_rhs, f_index = [], [], []
    for s_idx, s in enumerate(samples):
        i, j, d, vec = _neighbours(s.atoms, cutoff)
        sym = np.array(s.atoms.get_chemical_symbols())
        nat = len(s.atoms)
        want_f = force_weight > 0 and s.target_forces is not None
        fb = np.zeros((nat, 3, ncol)) if want_f else None
        for key, (t, n, keep, c0) in layout.items():
            if key[0] == key[1]:
                m = (sym[i] == key[0]) & (sym[j] == key[0])
            else:
                m = (((sym[i] == key[0]) & (sym[j] == key[1]))
                     | ((sym[i] == key[1]) & (sym[j] == key[0])))
            m &= d < pairs[key][1]
            if not m.any():
                continue
            dm, vm, im = d[m], vec[m], i[m]
            if dm.min() < pairs[key][0]:
                raise ValueError(f"{s.name or s_idx}: a {key[0]}-{key[1]} distance "
                                 f"{dm.min():.3f} A is below r_lo {pairs[key][0]} A; the spline "
                                 "basis is zero there, so lower r_lo")
            for c, k in enumerate(keep):
                u = _unit(t, n, k)
                a_e[s_idx, c0 + c] = 0.5 * np.nan_to_num(u(dm)).sum() / nat
                if want_f:
                    dv = np.nan_to_num(u.derivative()(dm))
                    np.add.at(fb[:, :, c0 + c], im, (dv / dm)[:, None] * vm)
        if want_f:
            scale = force_weight / np.sqrt(3 * nat)
            f_blocks.append(scale * fb.reshape(-1, ncol))
            f_rhs.append(scale * np.asarray(s.target_forces, float).ravel())
            f_index.append((s_idx, scale, 3 * nat))
    return a_e, f_blocks, f_rhs, f_index, layout, ncol


def fit_pair_splines(samples, pairs: dict, n_knots: int = 12, smooth: float = 1e-3,
                     force_weight: float = 1.0, monotonic: bool = True,
                     offsets: str = "none") -> RepFit:
    """Fit V_AB for ``pairs`` = {(A, B): (r_lo, r_cut)} (Å) to the samples (see the module
    docstring).  Pairs absent from every structure keep zero coefficients."""
    from scipy.optimize import lsq_linear
    if offsets not in ("none", "global", "per_element"):
        raise ValueError(f"offsets must be none, global or per_element, got {offsets!r}")
    pairs = {pair_key(*k): tuple(float(x) for x in v) for k, v in pairs.items()}
    samples = list(samples)
    a_e, f_blocks, f_rhs, f_index, layout, ncol = _design(samples, pairs, n_knots, force_weight)
    w = np.array([max(float(s.weight), 0.0) for s in samples])
    y = np.array([s.target_energy / len(s.atoms) for s in samples])
    elements = sorted({e for s in samples for e in s.atoms.get_chemical_symbols()})
    if offsets == "per_element":
        off = np.array([[s.atoms.get_chemical_symbols().count(e) / len(s.atoms) for e in elements]
                        for s in samples])
    elif offsets == "global":
        off = np.ones((len(samples), 1))
    else:
        off = np.zeros((len(samples), 0))
    n_off = off.shape[1]
    # c = U p per pair block (monotonic: cumulative from the cut-off side)
    u = np.zeros((ncol, ncol))
    dd_rows = []
    for key, (t, n, keep, c0) in layout.items():
        m = len(keep)
        blk = np.triu(np.ones((m, m))) if monotonic else np.eye(m)
        u[c0:c0 + m, c0:c0 + m] = blk
        if m > 2:
            d2 = np.zeros((m - 2, ncol))
            d2[:, c0:c0 + m] = np.diff(np.eye(m), 2, axis=0)
            dd_rows.append(d2)
    sw = np.sqrt(w)[:, None]
    rows = [np.hstack([sw * (a_e @ u), sw * off])]
    rhs = [np.sqrt(w) * y]
    for blk, r, (s_idx, _, _) in zip(f_blocks, f_rhs, f_index):
        rows.append(np.hstack([np.sqrt(w[s_idx]) * (blk @ u), np.zeros((blk.shape[0], n_off))]))
        rhs.append(np.sqrt(w[s_idx]) * r)
    if dd_rows and smooth > 0:
        dd = np.vstack(dd_rows) @ u
        rows.append(np.hstack([np.sqrt(smooth) * dd, np.zeros((dd.shape[0], n_off))]))
        rhs.append(np.zeros(dd.shape[0]))
    lo = np.concatenate([np.zeros(ncol) if monotonic else np.full(ncol, -np.inf),
                         np.full(n_off, -np.inf)])
    sol = lsq_linear(np.vstack(rows), np.concatenate(rhs), bounds=(lo, np.inf),
                     lsmr_tol="auto", max_iter=10000)
    p = sol.x
    c_all = u @ p[:ncol]
    splines = {}
    for key, (t, n, keep, c0) in layout.items():
        coef = np.zeros(n)
        coef[keep] = c_all[c0:c0 + len(keep)]
        splines[key] = PairSpline(key, pairs[key][0], pairs[key][1], t, coef)
    if offsets == "per_element":
        offs = {e: float(v) for e, v in zip(elements, p[ncol:])}
    elif offsets == "global":
        offs = {"_global": float(p[ncol])}
    else:
        offs = {}
    fit = RepFit(splines, offs, settings={"pairs": {f"{a}-{b}": v for (a, b), v in pairs.items()},
                                          "n_knots": n_knots, "smooth": smooth,
                                          "force_weight": force_weight, "monotonic": monotonic,
                                          "offsets": offsets})
    e_res = a_e @ c_all + off @ p[ncol:] - y
    fit.stats = {"energy_rmse_eV_per_atom": float(np.sqrt(np.mean(e_res ** 2))),
                 "energy_mae_eV_per_atom": float(np.mean(np.abs(e_res))),
                 "status": int(sol.status), "message": str(sol.message), "n_samples": len(samples)}
    if f_blocks:
        fr = np.concatenate([(blk @ c_all - r) / sc for blk, r, (_, sc, _) in
                             zip(f_blocks, f_rhs, f_index)])
        fit.stats["force_rmse_eV_per_A"] = float(np.sqrt(np.mean(fr ** 2)))
        fit.stats["force_mae_eV_per_A"] = float(np.mean(np.abs(fr)))
    return fit
