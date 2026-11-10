"""Band-structure loss between DFTB and a DFT reference (A series, A0 section 3).

For one structure, :func:`compare` matches DFTB eigenvalues with a :class:`pydftb.bandref.BandRef`.
The steps are:

1. **Window.** Only states between VBM - ``below`` and CBM + ``above`` count (HOMO/LUMO for
   molecules, the Fermi level for metals; default -3 / +1 eV, the user's choice).  The edges
   are logistic steps of width ``edge`` (0.1 eV), so a band that crosses an edge does not make
   the loss jump.
2. **Valence bands.** The DFT semicore bands are removed by count (``BandRef.n_semicore``).
   Then the lowest ``n_valence`` states at every k-point are paired by index on both sides,
   where ``n_valence`` is the number of DFTB electrons / 2.
3. **Conduction states.** Each DFTB state above them is paired with the nearest DFT state above
   them at the same k-point.  Matching is one-way: DFT states that the minimal basis cannot
   represent are not penalised, but spurious DFTB states are.  These pairs have weight
   ``conduction_weight`` (0.4).
4. **Shift.** One free shift Δ per structure, the weighted mean difference (closed form,
   iterated with the conduction matching).  ``shift="vbm"`` or ``"fermi"`` fixes it instead.
5. **Loss.** loss = RMS of ε_DFTB − ε_DFT − Δ, plus ``gap_weight`` · |gap error|.  The RMS is
   weighted by k-point weight × window × pair weight.  Only the k-points of
   ``levels`` enter: level 1 = special points and points next to them (default), level 2 = the
   segments between them.

Also reported: the RMS on level 1, the mean absolute error on every path segment (level 2) and
their mean, gaps, occupied band widths, Δ, and DOVE's score 𝒜 = 1 − mean|B̃_DFTB − B̃_DFT|
on Gaussian-broadened (k, E) maps (Song et al., arXiv 2605.19977), for comparison only.

The DFTB eigenvalues come from :func:`dftb_bands`, at the converged SCC charges of the same
calculation that gives the energy (``PyDFTB.states``).  DSKO's band objective (Samtsevych et
al., ChemRxiv 10.26434/chemrxiv-2025-4cmx9) could not be read; the two levels follow the
user's description: points near the special points first, then the lines between them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bandref import BandRef

HARTREE_EV = 27.211386245988


@dataclass(frozen=True)
class BandSettings:
    below: float = 3.0              # eV below VBM / HOMO / E_F
    above: float = 1.0              # eV above CBM / LUMO / E_F
    edge: float = 0.1               # eV, width of the logistic window edges
    conduction_weight: float = 0.4  # weight of the one-way matched states above n_valence
    gap_weight: float = 1.0         # beta in loss = RMS + beta |gap error|
    levels: tuple = (1,)            # path k-points in the loss: 1 near special points, 2 lines
    shift: str = "free"             # "free" | "vbm" | "fermi"
    band_bottom: bool = False       # metals: extend the window down to the valence bottom
    dove_sigma_e: float = 0.1       # eV, DOVE map broadening in energy
    dove_sigma_k: float = 1.0       # path points, DOVE map broadening along the path
    dove_grid: float = 0.01         # eV
    iterations: int = 6             # shift <-> conduction matching

    def __post_init__(self):
        if self.shift not in ("free", "vbm", "fermi"):
            raise ValueError(f"shift must be free, vbm or fermi, got {self.shift!r}")

    @classmethod
    def from_dict(cls, d: dict | None) -> "BandSettings":
        d = dict(d or {})
        if "levels" in d:
            d["levels"] = tuple(int(v) for v in np.atleast_1d(d["levels"]))
        return cls(**d)


@dataclass
class DFTBBands:
    """DFTB eigenvalues at a reference's k-points."""
    energies: np.ndarray     # (n_spin, n_k, n_state) eV
    n_valence: int           # bands paired by index: ceil(electrons / 2) without spin
    fermi: float             # eV, the SCC's Fermi level
    n_electrons: float | None = None   # odd counts (e.g. a one-atom Li cell) are metals


def _logistic(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def window_weight(e, lo: float, hi: float, edge: float):
    """1 inside [lo, hi], logistic steps of width ``edge`` at both ends."""
    return _logistic((np.asarray(e) - lo) / edge) * _logistic((hi - np.asarray(e)) / edge)


def dftb_bands(calc, atoms, ref: BandRef) -> DFTBBands:
    """DFTB eigenvalues (eV) at the k-points of ``ref``, at the charges of the SCC run of
    ``atoms`` with ``calc`` (a :class:`pydftb.ase_calc.PyDFTB`; the SCC runs first if needed).

    Periodic structures get exactly the reference's k-points, in its order, with no
    time-reversal reduction; molecules (and ``kind="molecule"``) the Gamma point."""
    periodic = bool(np.any(atoms.pbc))
    if ref.kind == "molecule" or not periodic:
        st = calc.states(atoms=atoms)
    else:
        from ase.dft.kpoints import BandPath
        path = BandPath(atoms.cell, kpts=np.asarray(ref.kpts, float), special_points={})
        st = calc.states(kpts=path, atoms=atoms)
    raw = calc.raw
    if st.eigenvalues.shape[0] != 1 or st.two_component:
        raise NotImplementedError("band fits support spin-unpolarized DFTB only")
    n_el = float(raw.n_electrons)
    nv = int(np.ceil(n_el / 2.0 - 1e-6))          # a half-filled band is paired by index too
    fermi = float(np.atleast_1d(raw.fermi)[0]) * HARTREE_EV
    return DFTBBands(energies=np.asarray(st.eigenvalues, float) * HARTREE_EV, n_valence=nv,
                     fermi=fermi, n_electrons=n_el)


def _edges(e, nv):
    vbm = float(e[:, :, :nv].max())
    cbm = float(e[:, :, nv:].min()) if e.shape[2] > nv else np.nan
    return vbm, cbm


def _pairs(e_d, e_b, nv, kw, lo, hi, s: BandSettings, delta):
    """Residuals and weights of the valence (index) and conduction (one-way nearest) pairs."""
    ev_d, ev_b = e_d[:, :, :nv], e_b[:, :, :nv]
    wv = kw[None, :, None] * window_weight(ev_d, lo, hi, s.edge)
    rv = ev_b - ev_d - delta
    ec_b, ec_d_all = e_b[:, :, nv:], e_d[:, :, nv:]
    if ec_b.shape[2] and ec_d_all.shape[2]:
        tgt = ec_b - delta
        idx = np.abs(ec_d_all[:, :, None, :] - tgt[:, :, :, None]).argmin(axis=3)
        ec_d = np.take_along_axis(ec_d_all, idx, axis=2)
        wc = s.conduction_weight * kw[None, :, None] * window_weight(tgt, lo, hi, s.edge)
        rc = ec_b - ec_d - delta
    else:
        wc = rc = np.zeros((e_b.shape[0], e_b.shape[1], 0))
    return rv, wv, rc, wc


def _rms(rv, wv, rc, wc):
    sw = wv.sum() + wc.sum()
    if sw <= 0:
        return np.nan, 0.0
    return float(np.sqrt(((wv * rv ** 2).sum() + (wc * rc ** 2).sum()) / sw)), float(sw)


def _dove_map(e, energy, sigma_e, sigma_k):
    m = np.exp(-(energy[None, None, :] - e[:, :, None]) ** 2 / (2 * sigma_e ** 2)).sum(axis=1)
    if sigma_k > 0 and m.shape[0] > 1:
        half = int(np.ceil(3 * sigma_k))
        g = np.exp(-np.arange(-half, half + 1) ** 2 / (2 * sigma_k ** 2))
        g /= g.sum()
        pad = np.pad(m, ((half, half), (0, 0)), mode="edge")
        m = np.stack([np.tensordot(g, pad[i:i + 2 * half + 1], axes=(0, 0))
                      for i in range(m.shape[0])])
    top = m.max()
    return m / top if top > 0 else m


def dove_score(e_d, e_b, delta, lo, hi, s: BandSettings) -> float:
    """DOVE's band accuracy 𝒜 = 1 − mean |B̃_DFTB − B̃_DFT| on the energy window (one spin
    channel; ``e_*`` (n_k, n_band) along the path, DFTB shifted by ``delta``)."""
    energy = np.arange(lo, hi + s.dove_grid / 2, s.dove_grid)
    bd = _dove_map(e_d, energy, s.dove_sigma_e, s.dove_sigma_k)
    bb = _dove_map(e_b - delta, energy, s.dove_sigma_e, s.dove_sigma_k)
    return float(1.0 - np.abs(bb - bd).mean())


def compare(ref: BandRef, dftb: DFTBBands, settings: BandSettings | None = None,
            group: str | None = None) -> dict:
    """Band loss and metrics of one structure (see the module docstring)."""
    s = settings or BandSettings()
    e_d, _ = ref.valence()
    e_b = np.asarray(dftb.energies, float)
    if e_d.shape[:2] != e_b.shape[:2]:
        raise ValueError(f"{ref.name}: DFT {e_d.shape[:2]} and DFTB {e_b.shape[:2]} "
                         "spin/k-point shapes differ")
    nv = int(dftb.n_valence)
    if nv > e_d.shape[2]:
        raise ValueError(f"{ref.name}: DFTB has {nv} valence bands, the reference only "
                         f"{e_d.shape[2]} above its semicore (n_semicore {ref.n_semicore})")
    ed = ref.edges(nv)
    odd = dftb.n_electrons is not None and abs(dftb.n_electrons / 2 - round(dftb.n_electrons / 2)) > 1e-6
    metal = ed["metal"] or odd
    if metal and not ed["metal"]:
        f = ref.fermi if ref.fermi is not None else ed["vbm"]
        ed = {**ed, "vbm": float(f), "cbm": float(f), "gap": 0.0, "metal": True}
    vbm_b, cbm_b = _edges(e_b, nv)
    if metal:
        fermi_d = ref.fermi if ref.fermi is not None else ed["vbm"]
        lo_ref = hi_ref = float(fermi_d)
        b_ref = dftb.fermi
    else:
        lo_ref = ed["vbm"]
        hi_ref = ed["cbm"] if np.isfinite(ed["cbm"]) else ed["vbm"]
        b_ref = vbm_b
    lo, hi = lo_ref - s.below, hi_ref + s.above
    if s.band_bottom:
        lo = min(lo, float(e_d[:, :, 0].min()) - 2 * s.edge)

    sel = ref.select(s.levels)
    if len(sel) == 0:
        raise ValueError(f"{ref.name}: no k-points of levels {s.levels}")
    kw_all = ref.kweights if ref.kind != "path" else np.ones(len(ref.kpts))

    def subset(idx):
        w = kw_all[idx]
        return e_d[:, idx], e_b[:, idx], w / w.sum()

    ld, lb, lw = subset(sel)
    if s.shift == "vbm":
        delta = (dftb.fermi - lo_ref) if metal else (vbm_b - lo_ref)
    elif s.shift == "fermi":
        if ref.fermi is None:
            raise ValueError(f"{ref.name}: shift='fermi' needs the reference's Fermi level")
        delta = dftb.fermi - float(ref.fermi)
    else:
        delta = b_ref - lo_ref
        for _ in range(max(1, s.iterations)):
            rv, wv, rc, wc = _pairs(ld, lb, nv, lw, lo, hi, s, delta)
            sw = wv.sum() + wc.sum()
            if sw <= 0:
                break
            step = ((wv * rv).sum() + (wc * rc).sum()) / sw
            delta += step
            if abs(step) < 1e-7:
                break
    rv, wv, rc, wc = _pairs(ld, lb, nv, lw, lo, hi, s, delta)
    rms, weight = _rms(rv, wv, rc, wc)
    gap_d = 0.0 if metal else ed["gap"]
    gap_b = 0.0 if metal else (cbm_b - vbm_b if np.isfinite(cbm_b) else np.nan)
    gap_err = 0.0 if metal else float(gap_b - gap_d)
    loss = rms + s.gap_weight * abs(gap_err) if np.isfinite(gap_err) else np.nan

    out = {"name": ref.name, "group": group or ref.meta.get("group") or ref.name, "kind": ref.kind,
           "metal": metal, "loss": float(loss), "rms": rms, "weight": weight,
           "shift": float(delta), "gap_ref": float(gap_d), "gap_dftb": float(gap_b),
           "gap_err": gap_err, "n_valence": nv, "levels": list(s.levels),
           "window": [float(lo), float(hi)]}
    # occupied band width (valence bottom to VBM / E_F)
    top_d = lo_ref
    top_b = dftb.fermi if metal else vbm_b
    out["bandwidth_ref"] = float(top_d - e_d[:, :, 0].min())
    out["bandwidth_dftb"] = float(top_b - e_b[:, :, 0].min())
    out["bandwidth_err"] = out["bandwidth_dftb"] - out["bandwidth_ref"]
    if ref.kind == "path":
        l1 = ref.select((1,))
        if len(l1):
            d1, b1, w1 = subset(l1)
            out["rms_level1"] = _rms(*_pairs(d1, b1, nv, w1, lo, hi, s, delta))[0]
        seg = {}
        for a, b, i0, i1 in ref.segments:
            idx = np.arange(int(i0), int(i1) + 1)
            dd, bb, ww = subset(idx)
            r_v, w_v, r_c, w_c = _pairs(dd, bb, nv, ww, lo, hi, s, delta)
            sw = w_v.sum() + w_c.sum()
            seg[f"{a}-{b}"] = (float(((w_v * np.abs(r_v)).sum() + (w_c * np.abs(r_c)).sum()) / sw)
                               if sw > 0 else np.nan)
        if seg:
            out["segment_mae"] = seg
            vals = [v for v in seg.values() if np.isfinite(v)]
            out["mae_level2"] = float(np.mean(vals)) if vals else np.nan
            l2 = np.concatenate([np.arange(int(i0), int(i1) + 1) for _, _, i0, i1 in ref.segments])
            out["dove_A"] = dove_score(e_d[0, l2], e_b[0, l2], delta, lo, hi, s)
    return out


def band_loss(results, by: str = "group") -> float:
    """Mean over groups of the mean loss of their structures (NaN if any structure failed)."""
    groups = {}
    for r in results:
        groups.setdefault(r.get(by, r["name"]), []).append(r["loss"])
    vals = [float(np.mean(v)) for v in groups.values()]
    return float(np.mean(vals)) if vals else np.nan


def evaluate_refs(refs, make_calc, settings: BandSettings | None = None) -> list:
    """Compare every reference with DFTB: ``make_calc(ref) -> (calc, atoms)`` sets up the SCC
    (k-point grid, smearing) for that structure.  A failed SCC gives ``loss = NaN`` and its
    error text instead of raising."""
    out = []
    for ref in refs:
        try:
            calc, atoms = make_calc(ref)
            out.append(compare(ref, dftb_bands(calc, atoms, ref), settings))
        except Exception as exc:              # noqa: BLE001  (recorded per structure)
            out.append({"name": ref.name, "group": ref.meta.get("group") or ref.name,
                        "kind": ref.kind, "loss": np.nan, "error": f"{type(exc).__name__}: {exc}"})
    return out


# ---- least-squares form (fitting, A series A2) ------------------------------------------------------
def residuals(ref: BandRef, dftb: DFTBBands, settings: BandSettings | None = None) -> dict:
    """:func:`compare` as least-squares residuals, for fits with a Jacobian.

    With compare's window, matching and shift Δ, every valence and conduction pair gives
    sqrt(w_i / Σw) (ε_DFTB,i − ε_DFT,i − Δ) (their squares sum to rms²), followed by the gap
    term ``gap_weight`` · gap error (insulators and molecules).  Returns
    ``{"r": (m,) eV, "states": (m_pairs, 3) (spin, k, DFTB state) of every pair, "sqrt_w":
    (m_pairs,), "gap": None or {"cbm": (s, k, n), "vbm": (s, k, n), "weight"}, "free": shift
    is free, "compare": the compare dict}``; :func:`residual_jacobian` turns eigenvalue
    derivatives into the Jacobian of ``r``."""
    s = settings or BandSettings()
    comp = compare(ref, dftb, s)
    e_d, _ = ref.valence()
    e_b = np.asarray(dftb.energies, float)
    nv = int(dftb.n_valence)
    lo, hi = comp["window"]
    delta = comp["shift"]
    sel = ref.select(s.levels)
    kw_all = ref.kweights if ref.kind != "path" else np.ones(len(ref.kpts))
    w = kw_all[sel]
    ld, lb, lw = e_d[:, sel], e_b[:, sel], w / w.sum()
    rv, wv, rc, wc = _pairs(ld, lb, nv, lw, lo, hi, s, delta)
    sw = wv.sum() + wc.sum()
    ns = e_b.shape[0]
    st_v = np.array([(sp, sel[k], n) for sp in range(ns) for k in range(len(sel))
                     for n in range(nv)], int).reshape(-1, 3)
    st_c = np.array([(sp, sel[k], nv + m) for sp in range(ns) for k in range(len(sel))
                     for m in range(rc.shape[2])], int).reshape(-1, 3)
    if sw > 0:
        wh = np.concatenate([wv.ravel(), wc.ravel()]) / sw
        raw = np.concatenate([rv.ravel(), rc.ravel()])
    else:
        wh = raw = np.zeros(0)
        st_v = st_v[:0]
        st_c = st_c[:0]
    keep = wh > 0
    states = np.vstack([st_v, st_c])[keep] if sw > 0 else np.zeros((0, 3), int)
    sqrt_w = np.sqrt(wh[keep])
    r = sqrt_w * raw[keep]
    gap = None
    if not comp["metal"] and np.isfinite(comp["gap_err"]) and e_b.shape[2] > nv and s.gap_weight:
        iv = np.unravel_index(np.argmax(e_b[:, :, :nv]), e_b[:, :, :nv].shape)
        ic = np.unravel_index(np.argmin(e_b[:, :, nv:]), e_b[:, :, nv:].shape)
        gap = {"vbm": (int(iv[0]), int(iv[1]), int(iv[2])),
               "cbm": (int(ic[0]), int(ic[1]), int(nv + ic[2])), "weight": float(s.gap_weight)}
        r = np.append(r, s.gap_weight * comp["gap_err"])
    return {"r": r, "states": states, "sqrt_w": sqrt_w, "gap": gap, "free": s.shift == "free",
            "compare": comp}


def residual_states(res: dict) -> np.ndarray:
    """The DFTB state indices whose derivatives :func:`residual_jacobian` needs."""
    idx = [res["states"][:, 2]]
    if res["gap"] is not None:
        idx.append(np.array([res["gap"]["vbm"][2], res["gap"]["cbm"][2]]))
    return np.unique(np.concatenate(idx)).astype(int) if idx else np.zeros(0, int)


def residual_jacobian(res: dict, deps) -> np.ndarray:
    """Jacobian (m, n_x) of :func:`residuals`' ``r`` from eigenvalue derivatives ``deps``
    (n_spin, n_k, n_state, n_x) in eV per unit x.  The free shift is eliminated (its
    derivative is the weighted mean); window weights and matching are held fixed (they change
    only where a state crosses a window edge or another state)."""
    deps = np.asarray(deps, float)
    st = res["states"]
    d = deps[st[:, 0], st[:, 1], st[:, 2]] if len(st) else np.zeros((0, deps.shape[-1]))
    sw = res["sqrt_w"]
    if res["free"] and len(st):
        d = d - (sw ** 2) @ d
    jac = sw[:, None] * d
    if res["gap"] is not None:
        g = res["gap"]
        row = g["weight"] * (deps[g["cbm"]] - deps[g["vbm"]])
        jac = np.vstack([jac, row[None, :]])
    return jac
