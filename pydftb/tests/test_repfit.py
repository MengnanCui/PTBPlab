"""Two-body spline repulsives by linear least squares (A series): recovery of a known model,
offsets, monotonicity, forces and the SKF Spline block."""
import numpy as np
import pytest
from ase import Atoms

from pydftb.repfit import PairSpline, RepFit, RepSample, basis, fit_pair_splines

PAIRS = {("Li", "Li"): (1.1, 3.2), ("Li", "O"): (1.1, 2.8), ("O", "O"): (1.0, 2.6)}


def _truth():
    """A known monotonic repulsive for three pairs."""
    sp = {}
    for key, (lo, hi) in PAIRS.items():
        t, n, keep = basis(lo, hi, 10)
        c = np.zeros(n)
        steps = np.linspace(1.0, 0.1, len(keep)) * (2.0 if key == ("O", "O") else 1.0)
        c[keep] = np.cumsum(steps[::-1])[::-1] * 0.3
        sp[key] = PairSpline(key, lo, hi, t, c)
    return RepFit(sp)


def _structures(rng):
    out = []
    for k in range(40):
        n = rng.integers(2, 6)
        sym = rng.choice(["Li", "O"], size=n)
        pos = rng.uniform(0, 3.0, size=(n, 3))
        at = Atoms(list(sym), positions=pos, cell=[12, 12, 12], pbc=False)
        d = at.get_all_distances()[np.triu_indices(n, 1)]
        if d.min() < 1.2:
            continue
        out.append(at)
    return out


def test_fit_recovers_a_known_repulsive():
    rng = np.random.default_rng(3)
    truth = _truth()
    samples = []
    for at in _structures(rng):
        e, f = truth.predict(at)
        samples.append(RepSample(at, e + 0.05 * len(at), f))   # plus a global offset
    fit = fit_pair_splines(samples, PAIRS, n_knots=10, smooth=0.0, force_weight=1.0,
                           monotonic=True, offsets="global")
    assert fit.stats["energy_rmse_eV_per_atom"] < 1e-6
    assert fit.stats["force_rmse_eV_per_A"] < 1e-5
    assert fit.offsets["_global"] == pytest.approx(0.05, abs=1e-6)
    r = np.linspace(1.45, 3.0, 30)          # inside the distances of the data (>= 1.2 A)
    for key in PAIRS:
        assert np.allclose(fit.splines[key].value(r), truth.splines[key].value(r), atol=1e-5)


def test_per_element_offsets_and_forces_match_finite_differences():
    rng = np.random.default_rng(7)
    truth = _truth()
    mu = {"Li": -0.2, "O": 0.4}
    samples = []
    for at in _structures(rng):
        e, f = truth.predict(at)
        samples.append(RepSample(at, e + sum(mu[s] for s in at.get_chemical_symbols()), f))
    fit = fit_pair_splines(samples, PAIRS, n_knots=10, smooth=0.0, offsets="per_element")
    assert fit.offsets["Li"] == pytest.approx(-0.2, abs=1e-5)
    assert fit.offsets["O"] == pytest.approx(0.4, abs=1e-5)
    at = samples[0].atoms.copy()
    e0, f0 = fit.predict(at)
    h = 1e-5
    p = at.positions.copy()
    p[0, 0] += h
    at.positions = p
    e1, _ = fit.predict(at)
    assert -(e1 - e0) / h == pytest.approx(f0[0, 0], abs=1e-4)


def test_monotonic_fit_is_non_increasing_and_non_negative():
    rng = np.random.default_rng(11)
    samples = [RepSample(at, rng.normal(), None) for at in _structures(rng)]   # noise
    fit = fit_pair_splines(samples, PAIRS, n_knots=8, monotonic=True, force_weight=0.0)
    for sp in fit.splines.values():
        r = np.linspace(sp.r_lo, sp.r_cut, 200)
        v = sp.value(r)
        assert np.all(v >= -1e-12) and np.all(np.diff(v) <= 1e-12)


def test_r_lo_above_a_distance_in_the_data_is_an_error():
    at = Atoms("Li2", positions=[[0, 0, 0], [0, 0, 1.0]], cell=[10, 10, 10])
    with pytest.raises(ValueError, match="below r_lo"):
        fit_pair_splines([RepSample(at, 1.0)], {("Li", "Li"): (1.1, 3.0)})


def test_skf_spline_block_replaces_and_appends(tmp_path):
    truth = _truth()
    src = tmp_path / "src"
    src.mkdir()
    body = "0.02 10\n1 2 3\n"
    for a in ("Li", "O"):
        for b in ("Li", "O"):
            text = body + ("Spline\n1 5.0\n1 1 0\n4.0 5.0 0 0 0 0 0 0\n" if a == b else "")
            (src / f"{a}-{b}.skf").write_text(text)
    out = truth.write_skf(str(src), str(tmp_path / "dst"))
    assert len(out) == 4 + 0 * len(out)  # Li-Li, O-O, Li-O, O-Li
    li = (tmp_path / "dst" / "Li-Li.skf").read_text().splitlines()
    assert li.count("Spline") == 1 and li[0] == "0.02 10"
    lio = (tmp_path / "dst" / "Li-O.skf").read_text().splitlines()
    assert "Spline" in lio
    head = lio[lio.index("Spline") + 2].split()
    assert float(head[0]) > 0                                 # a1 > 0: a decaying head
    back = RepFit.from_json(truth.to_json(str(tmp_path / "rep.json")))
    assert set(back.splines) == set(truth.splines)
