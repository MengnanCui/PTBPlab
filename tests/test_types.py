"""PTBP's parameters by type (milestone M5b): the type model, physrep and hotcent with type
labels, the mixed repulsive of switched types, the sigma_rep search of several parameter sets,
the gradient with a sigma vector, the CLI/YAML/TUI and the Loss of parameters per type."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.data import covalent_radii
from ase.units import Bohr

import utils.data as data
import utils.engines as eng
import utils.loss as L
from utils import dftb_stage
from utils import loss_grad as LG
from utils import typemodel as TM
from utils.parameters_set import get_PTBP

ROOT = Path(__file__).resolve().parents[1]
SKF = ROOT / "pydftb" / "tests" / "data" / "skf"
linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork backend")


@pytest.fixture(autouse=True)
def _restore_engine():
    saved = eng.settings()
    switches = eng.type_switching()
    yield
    eng._SETTINGS.clear()
    eng._SETTINGS.update(saved)
    eng.set_type_switching(switches)


@pytest.fixture(scope="module")
def images():
    return ase.io.read(ROOT / "dft" / "dft.xyz", ":")


def _box(symbols, positions, a=9.0):
    atoms = Atoms(symbols, positions=positions, cell=np.eye(3) * a, pbc=True)
    return atoms


def _bnna():
    """Two structures with B, N and Na (the elements of examples/04)."""
    return [_box(["Na", "N", "N", "N"], [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]]),
            _box(["B", "B", "N", "N"], [[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]])]


SWITCH = {"element": "Cu", "labels": ["Cu2", "Cu"], "centre": 11.0, "width": 1.0}


# ---- the type model ----------------------------------------------------------------
def test_default_is_one_set_per_element_with_its_bounds_and_seed():
    m = TM.TypeModel.from_dataset(_bnna())
    assert m.labels == ["B", "N", "Na"] and not m.legacy and not m.shared
    assert m.dimension_names() == ["r0_w_B", "r0_d_B", "r0_w_N", "r0_d_N", "r0_w_Na", "r0_d_Na"]
    lo, hi = m.bounds()
    for k, el in enumerate(("B", "N", "Na")):
        r = covalent_radii[{"B": 5, "N": 7, "Na": 11}[el]] / Bohr
        assert lo[2 * k:2 * k + 2] == [2.1, 3.4]
        assert hi[2 * k:2 * k + 2] == pytest.approx([7.0 * r, 9.0 * r], abs=0)
    seed = m.seed()
    assert seed[0:2] == [get_PTBP()["B"]["r0_w"], get_PTBP()["B"]["r0_d"]]
    assert seed[2:4] == [get_PTBP()["N"]["r0_w"], 3.4]      # PTBP's 2.538 is below the box
    assert m.n_sigma == 3 and m.n_params == 6


def test_shared_parameters_and_one_element_are_the_layout_of_before():
    shared = TM.TypeModel.from_dataset(_bnna(), shared=True)
    assert shared.legacy and shared.dimension_names() == ["r0_w", "r0_d"]
    assert shared.sets[0].radius == L.penalty_radius(_bnna(), [])
    cu = TM.TypeModel.from_dataset([_box(["Cu", "Cu"], [[0, 0, 0], [2.5, 0, 0]])])
    assert cu.legacy and cu.seed() == [2.749, 12.036]
    loss = L.Loss(args_input=argparse.Namespace(optimization_option="energygeometry"),
                  dft_input=[_box(["Cu"], [[0, 0, 0]])], types=cu)
    assert loss.typed is False                                  # the code of before M5b
    assert cu.directory_name([2.749, 12.036]) == "opt_2.749_12.036"


def test_switched_types_fixed_types_and_errors():
    cu = [_box(["Cu", "Cu"], [[0, 0, 0], [2.5, 0, 0]])]
    m = TM.TypeModel.from_dataset(cu, declarations=[SWITCH])
    assert m.labels == ["Cu2", "Cu"] and m.element_of == {"Cu2": "Cu", "Cu": "Cu"}
    assert [s.windows for s in m.switches] == [[(10.5, 11.5)]]
    fixed = TM.TypeModel.from_dataset(_bnna(), multi_element=["N"])
    assert [s.name for s in fixed.sets] == ["N"]
    na = get_PTBP()["Na"]
    assert fixed.r0([3.0, 4.0])["Na"] == (na["r0_w"], na["r0_d"])
    assert fixed.sigma([0.5])["Na"] == na["sigma_rep"] and fixed.sigma([0.5])["N"] == 0.5
    with pytest.raises(ValueError, match="identical"):
        TM.TypeModel.from_dataset(cu, shared=True, declarations=[SWITCH])
    with pytest.raises(ValueError, match="does not contain"):
        TM.TypeModel.from_dataset(_bnna(), declarations=[SWITCH])
    with pytest.raises(ValueError, match="unknown types"):
        TM.TypeModel.from_dataset(_bnna(), starts={"N2": (2.0, 4.0)})
    with pytest.raises(ValueError, match="not in the dataset"):
        TM.TypeModel.from_dataset(_bnna(), multi_element=["Cu"])


def test_penalty_is_the_geometric_mean_and_its_derivative():
    m = TM.TypeModel.from_dataset(_bnna())
    x = [2.6, 4.9, 2.5, 3.9, 5.5, 13.0]
    each = [L.r0_penalty(x[2 * k], x[2 * k + 1], s.radius) for k, s in enumerate(m.sets)]
    assert m.penalty(x) == pytest.approx(math.prod(each) ** (1 / 3), rel=1e-15)
    for j, d in enumerate(m.dlnp(x)):
        h = np.zeros(6)
        h[j] = 1e-6
        fd = (math.log(m.penalty(np.array(x) + h)) - math.log(m.penalty(np.array(x) - h))) / 2e-6
        assert d == pytest.approx(fd, abs=1e-8)
    # two types of one element at equal radii: the penalty of the single type
    cu = TM.TypeModel.from_dataset([_box(["Cu", "Cu"], [[0, 0, 0], [2.5, 0, 0]])],
                                   declarations=[SWITCH])
    assert cu.penalty([3.0, 9.0, 3.0, 9.0]) == pytest.approx(
        L.r0_penalty(3.0, 9.0, covalent_radii[29] / Bohr), rel=1e-15)


def test_meta_records_the_types():
    m = TM.TypeModel.from_dataset(_bnna())
    meta = json.loads(json.dumps(m.meta("reaction")))
    assert meta["ptbp_loss_version"] == L.LOSS_VERSION == 3 and m.matches_meta(meta)
    other = TM.TypeModel.from_dataset(_bnna(), multi_element=["N", "B"])
    assert not other.matches_meta(meta) and not m.matches_meta(None)
    assert not m.matches_meta({"ptbp_loss_version": 2, "penalty_radius_bohr": 1.0})


def test_logs_have_a_column_per_type():
    m = TM.TypeModel.from_dataset(_bnna())
    head = m.log_header("reaction")
    assert head.startswith("#r0_w_B,r0_d_B,sigma_rep_B,r0_w_N") and head.endswith("penalty,mu\n")
    line = m.log_line([1, 2, 3, 4, 5, 6], [0.3, 0.4, 0.5], 0.1, 0.2, 0.0, 4.5, 1.25)
    assert len(line.split(",")) == len(head.split(","))
    assert m.directory_name([2.49, 4.358, 2.349, 3.4, 5.01, 14.114]) == \
        "opt_2.490_4.358_2.349_3.400_5.010_14.114"


# ---- physrep with labels and the mixed repulsive --------------------------------------
def test_physrep_with_labels_has_the_splines_of_atomic_numbers():
    from tools.physrep import physrep
    a = physrep([7, 11], rmin=0.5, rmax=20.0, npoints=700, scale_cov=[0.4, 0.55], k_xc=[0, 0])
    b = physrep(["N", "Na"], rmin=0.5, rmax=20.0, npoints=700, scale_cov=[0.4, 0.55],
                k_xc=[0, 0], numbers={"N": 7, "Na": 11})
    for (za, zb), (la, lb) in (((7, 11), ("N", "Na")), ((11, 11), ("Na", "Na"))):
        sa, sb = a.Vreps[f"{za}_{zb}"], b.Vreps[f"{la}_{lb}"]
        assert np.array_equal(sa.c, sb.c) and np.array_equal(sa.x, sb.x)
        assert a.rcuts[f"{za}_{zb}"] == b.rcuts[f"{la}_{lb}"]


def _cu_cluster():
    pos = np.array([[0, 0, 0], [2.4, 0.1, 0], [1.1, 2.1, 0.2], [1.2, 0.7, 2.0], [3.3, 2.0, 1.4]])
    return Atoms(["Cu"] * 5, positions=pos, cell=np.eye(3) * 14.0, pbc=True)


def _switched(centre=3.0, width=2.0, cutoff=(4.4, 5.4), sig=(0.42, 0.45)):
    from pydftb.typeswitch import TypeSwitch

    from tools.physrep import physrep, switched_physrep
    sw = TypeSwitch("Cu", ("Cu2", "Cu"), centre=centre, width=width, cutoff=cutoff)
    pr = physrep(["Cu2", "Cu"], rmin=0.5, rmax=20.0, npoints=700, scale_cov=list(sig),
                 k_xc=[0, 0], numbers={"Cu2": 29, "Cu": 29})
    return switched_physrep(pr, [sw])


def test_switched_repulsive_forces_are_its_gradient():
    """Weights inside the window: the forces include the weights' part (pydftb's
    weight_gradient); 4-point differences of the energy."""
    rep = _switched()
    atoms = _cu_cluster()
    w = rep.weights(atoms)
    assert np.any((w["Cu"] > 0.01) & (w["Cu"] < 0.99))         # some atoms are mixed
    assert np.allclose(w["Cu"] + w["Cu2"], 1.0)
    e, f = rep.evaluate_pbc(atoms)
    h = 1e-4
    for i in range(len(atoms)):
        for c in range(3):
            vals = []
            for s in (2, 1, -1, -2):
                b = atoms.copy()
                b.positions[i, c] += s * h
                vals.append(rep.evaluate_pbc(b)[0])
            fd = -(-vals[0] + 8 * vals[1] - 8 * vals[2] + vals[3]) / (12 * h)
            assert f[i, c] == pytest.approx(fd, abs=2e-6 * max(1.0, abs(fd)))


def test_switched_repulsive_discrete_limit_is_physrep():
    """Every atom in one type (weights 0/1): the energy of plain physrep at that type's
    sigma (pydftb's Bohr instead of ASE's: ~1e-7)."""
    from tools.physrep import physrep
    atoms = _cu_cluster()
    rep = _switched(centre=20.0, width=1.0)                    # CN < 19.5: all Cu2
    ref = physrep([29], rmin=0.5, rmax=20.0, npoints=700, scale_cov=[0.42], k_xc=[0])
    e, f = rep.evaluate_pbc(atoms)
    e0, f0 = ref.evaluate_pbc(atoms)
    assert e == pytest.approx(e0, rel=1e-5)
    assert np.allclose(f, f0, rtol=1e-5, atol=1e-7)


def test_full_dftb_keeps_the_order_of_the_atoms():
    """Atoms(get_chemical_formula(), ...) wrote [Na, N, N, N] as N3Na: symbols on the wrong
    positions."""
    nan3 = _bnna()[0]
    band = nan3.copy()
    band.info["energy"] = -1.0
    band.set_array("forces", np.zeros((4, 3)))

    class Zero:
        def evaluate_pbc(self, atoms):
            return 0.5, np.ones((len(atoms), 3))
    out = data.get_full_dftb([band], [nan3], Zero(), True)[0]
    assert out.get_chemical_symbols() == ["Na", "N", "N", "N"]
    assert np.array_equal(out.positions, nan3.positions)
    assert out.info["energy"] == -0.5 and np.array_equal(out.get_array("rep_forces"), np.ones((4, 3)))


# ---- engines -------------------------------------------------------------------------
def test_switching_goes_to_pydftb_and_dftbplus_refuses():
    eng.set_type_switching([TM.switch_from_declaration(SWITCH)])
    atoms = _cu_cluster()
    eng.configure(engine="pydftb")
    calc = eng.make_calculator(atoms, {"calc_type": "static", "SCC": "Yes", "poly_rep": "Yes",
                                       "slater_p": str(SKF / "cu_types") + "/", "kpts": [1, 1, 1]})
    assert [s.types for s in calc.type_switching] == [("Cu2", "Cu")]
    eng.configure(engine="dftbplus")
    with pytest.raises(ValueError, match="pydftb"):
        eng.make_calculator(atoms, {"calc_type": "static", "SCC": "Yes", "poly_rep": "Yes",
                                    "slater_p": "./", "kpts": [1, 1, 1]})
    eng.set_type_switching(())
    eng.configure(engine="pydftb")
    assert eng.make_calculator(atoms, {"slater_p": str(SKF / "cu_types") + "/",
                                       "kpts": [1, 1, 1]}).type_switching is None


# ---- hotcent with labels (the H atom is cheap) ------------------------------------------
def _hotcent_labels(tmp_path, name, r0, elements, only=None, base=None):
    from utils import hotcent_parallel as hcp
    from utils.hotpy import hotpy
    out = tmp_path / name
    out.mkdir()
    hp = hotpy()
    hp.set_variables(r0, {s: 2.0 for s in r0}, xc="GGA_X_PBE+GGA_C_PBE",
                     superposition="density", elements=elements)
    if only is None:
        hcp.generate_all(hp, "density", [], "ptbp", workers=2, outdir=str(out))
    else:
        from utils.hotcent_tangent import run_hotcent
        run_hotcent({"xc": "GGA_X_PBE+GGA_C_PBE", "superposition": "density",
                     "elements": elements}, (r0, {s: 2.0 for s in r0}), str(out), 2,
                    only=only, base_dir=str(base))
    return out


@pytest.mark.slow
@linux
def test_hotcent_types_of_one_element(tmp_path):
    pytest.importorskip("hotcent")
    pytest.importorskip("pylibxc")
    from pydftb.skf import SKFile
    from utils import hotcent_parallel as hcp
    if not hcp.sharding_supported():
        pytest.skip("hotcent is not 2.1.0")
    el = {"H": "H", "H2": "H"}
    same = _hotcent_labels(tmp_path, "same", {"H": [2.2, 5.0], "H2": [2.2, 5.0]}, el)
    assert sorted(p.name for p in same.glob("*.skf")) == ["H-H.skf", "H-H2.skf", "H2-H.skf",
                                                          "H2-H2.skf"]
    homo = SKFile.read(str(same / "H-H.skf"), homo=True)
    for name in ("H-H2.skf", "H2-H.skf"):                    # heteronuclear format, no on-site
        lines = (same / name).read_text().splitlines()
        assert len(lines[1].replace(",", " ").split()) not in (10, 13)
        cross = SKFile.read(str(same / name), homo=False)
        assert np.array_equal(cross.table, homo.table)        # equal confinements: one table
    assert (same / "H2-H2.skf").read_text() == (same / "H-H.skf").read_text()
    # a type's own tables only, the others copied: the files of the full run
    diff = _hotcent_labels(tmp_path, "diff", {"H": [2.2, 5.0], "H2": [2.6, 5.0]}, el)
    part = _hotcent_labels(tmp_path, "part", {"H": [2.2, 5.0], "H2": [2.6, 5.0]}, el,
                           only={"H2"}, base=same)
    for p in diff.glob("*.skf"):
        assert (part / p.name).read_bytes() == p.read_bytes(), p.name
    a = SKFile.read(str(diff / "H-H2.skf"), homo=False).table
    b = SKFile.read(str(diff / "H2-H.skf"), homo=False).table
    assert not np.array_equal(a, SKFile.read(str(diff / "H-H.skf"), homo=True).table)
    assert np.allclose(a[:, -1], b[:, -1], rtol=1e-10, atol=1e-14)   # ss overlap is symmetric


# ---- the sigma_rep search of several parameter sets -----------------------------------------
def _no_files_step(cls, loss, *a, **k):
    """StepData stand-in for the sigma search on a synthetic loss (no structures, no files)."""
    return SimpleNamespace(dft=[], physrep_all=lambda sigma: SimpleNamespace(
        write_skf=lambda known_parameters: None))


def test_sigma_search_of_several_sets(tmp_path, monkeypatch):
    """1-D search for a common sigma, a grid per set, joint L-BFGS-B: the minimum of a
    smooth function of two sigma, one log line per evaluation."""
    m = TM.TypeModel.from_dataset(_bnna(), multi_element=["B", "N"])
    target = np.array([0.33, 0.61])

    def fake_mu(step, sigma, parts=None):
        s = np.asarray(sigma, float)
        v = 0.05 + 40 * float(np.sum((s - target) ** 2 * [3.0, 1.5])) \
            + 8.0 * (s[0] - 0.33) * (s[1] - 0.61)
        if parts is not None:
            parts.update(mu_E=v, mu_F=0.0, penalty=4.0, diffs=np.zeros(2))
        return v

    monkeypatch.setattr(LG, "mu", fake_mu)
    monkeypatch.setattr(LG.StepData, "from_files",
                        classmethod(_no_files_step))
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(optimization_option="dataset", log_file=str(tmp_path / "log.out"))
    loss = L.Loss(args_input=args, dft_input=_bnna(), types=m)
    assert loss.typed
    svec, mu = loss.Repulsion_optimization_types([3.0, 4.0, 2.5, 3.6])
    assert np.allclose(svec, target, atol=5e-3) and mu == pytest.approx(0.05, abs=2e-3)
    lines = (tmp_path / "log.out").read_text().splitlines()
    assert len(lines) >= 11 + 1 + 22 and all(len(x.split(",")) == 3 * 3 + 5 for x in lines)


def test_sigma_search_rounds_find_a_plateau(tmp_path, monkeypatch):
    """A sigma on a plateau that pays only once the other sigma has moved (as Na's sigma at the
    seed of the reaction example): the second round of grids finds it after L-BFGS-B (here a
    stand-in that stops in the well, as at a kink), a third round finds nothing more."""
    m = TM.TypeModel.from_dataset(_bnna(), multi_element=["B", "N"])

    def fake_mu(step, sigma, parts=None):
        s1, s2 = (float(v) for v in sigma)
        if s2 >= 0.3:
            v = 40 * (s1 - 0.75) ** 2 + 0.5 + 10 * (s2 - 0.5) ** 2
        else:
            v = 40 * (s1 - 0.75) ** 2 + 0.2 + 3000 * (s1 - 0.75) ** 2
        if parts is not None:
            parts.update(mu_E=v, mu_F=0.0, penalty=4.0, diffs=np.zeros(2))
        return v

    real, calls = L.minimize, []

    def first_stops_in_the_well(fun, x0, **kw):
        calls.append([float(v) for v in x0])
        if len(calls) == 1:
            x = np.array([0.75, 0.5])
            return SimpleNamespace(x=x, fun=fun(x))
        return real(fun, x0, **kw)

    monkeypatch.setattr(LG, "mu", fake_mu)
    monkeypatch.setattr(L, "minimize", first_stops_in_the_well)
    monkeypatch.setattr(LG.StepData, "from_files",
                        classmethod(_no_files_step))
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(optimization_option="dataset", log_file=str(tmp_path / "log.out"))
    loss = L.Loss(args_input=args, dft_input=_bnna(), types=m)
    svec, mu = loss.Repulsion_optimization_types([3.0, 4.0, 2.5, 3.6])
    assert mu == pytest.approx(0.2, abs=1e-9) and svec[0] == pytest.approx(0.75) and svec[1] < 0.3
    assert len(calls) == 2 and calls[1][1] < 0.3           # round 2 polished the plateau point
    assert fake_mu(None, svec) == mu


# ---- the gradient with a sigma vector (synthetic losses) -------------------------------------
class ModelN:
    """mu(x, sigma) = sum_j w_j |a_j + B_j . sigma + q_j |sigma - c_j|^2 + G_j . x|."""

    def __init__(self, a, b, q, c, g, w):
        self.a, self.q, self.w = (np.asarray(v, float) for v in (a, q, w))
        self.b, self.c, self.g = (np.asarray(v, float) for v in (b, c, g))
        self.x = np.zeros(self.g.shape[1])

    def r(self, s):
        s = np.asarray(s, float)
        return self.a + self.b @ s + self.q * np.sum((s - self.c) ** 2, axis=1) + self.g @ self.x

    def drds(self, s):
        return self.b + 2 * self.q[:, None] * (np.asarray(s, float) - self.c)

    def mu(self, s):
        return float(np.sum(self.w * np.abs(self.r(s))))

    def terms(self, s):
        n = len(self.a)
        return LG.TermsN(np.asarray(s, float), 1.0, self.w, self.r(s), self.drds(s), ["E"] * n,
                         np.eye(n), np.zeros((n, 0)), [], [None] * n, [1] * n)

    def mu_star(self, lo=0.1, hi=0.8):
        """Exact for linear residuals (q = 0): the minimum of a convex piecewise-linear
        function is at a vertex of its pieces or of the box."""
        lines = [(self.b[j], -(self.a[j] + self.g[j] @ self.x)) for j in range(len(self.a))]
        lines += [(np.array([1.0, 0.0]), lo), (np.array([1.0, 0.0]), hi),
                  (np.array([0.0, 1.0]), lo), (np.array([0.0, 1.0]), hi)]
        best = math.inf
        for i in range(len(lines)):
            for k in range(i + 1, len(lines)):
                m = np.array([lines[i][0], lines[k][0]])
                if abs(np.linalg.det(m)) < 1e-12:
                    continue
                s = np.linalg.solve(m, [lines[i][1], lines[k][1]])
                if np.all(s >= lo - 1e-12) and np.all(s <= hi + 1e-12):
                    best = min(best, self.mu(np.clip(s, lo, hi)))
        return best


def _fake_typed_step(m):
    model = SimpleNamespace(dimension_names=lambda: ["x1", "x2"], dlnp=lambda x: [0.0, 0.0])
    return SimpleNamespace(model=model, x=[0.0, 0.0])


@pytest.mark.parametrize("case, model, sigma_hat", [
    ("kink", ModelN([-0.3, 0.5, 0.2, 0.05], [[1, 0.2], [-0.4, -1], [0.5, 0.5], [0.1, -0.3]],
                    [0, 0, 0, 0], [[0, 0]] * 4, [[0.7, 0.1], [0.2, -0.5], [-0.4, 0.3], [0.1, 0.2]],
                    [1, 1, 0.2, 0.1]), [0.33, 0.47]),
    ("bound", ModelN([1.0, 2.0], [[1.0, 0.2], [0.5, 1.0]], [0, 0], [[0, 0]] * 2,
                     [[0.2, 0.1], [0.1, 0.3]], [1, 1]), [0.12, 0.13]),
])
def test_inner_minimum_and_gradient_of_a_sigma_vector(monkeypatch, case, model, sigma_hat):
    monkeypatch.setattr(LG, "mu", lambda step, s, parts=None: model.mu(s))
    monkeypatch.setattr(LG, "terms_typed", lambda step, s: model.terms(s))
    step = _fake_typed_step(model)
    inner = LG.refine_sigma_typed(step, sigma_hat)
    assert inner.case == case, inner
    assert inner.mu <= model.mu(sigma_hat) + 1e-12
    assert inner.mu == pytest.approx(model.mu_star(), abs=1e-9)
    g = LG.assemble_typed(step, inner, {"x1": model.g[:, 0], "x2": model.g[:, 1]})
    h = 1e-5
    for k, name in enumerate(("x1", "x2")):
        vals = []
        for s in (h, -h):
            model.x = np.zeros(2)
            model.x[k] = s
            vals.append(model.mu_star())
        model.x = np.zeros(2)
        fd = (vals[0] - vals[1]) / (2 * h)
        assert g[name] == pytest.approx(fd, abs=1e-6 * max(abs(fd), 1.0)), (name, g, fd)


def test_smooth_inner_minimum_of_a_sigma_vector(monkeypatch):
    """One residual that stays positive: a smooth interior minimum, envelope theorem."""
    model = ModelN([1.0], [[0.3, -0.2]], [2.0], [[0.4, 0.5]], [[0.6, -0.3]], [1.0])
    monkeypatch.setattr(LG, "mu", lambda step, s, parts=None: model.mu(s))
    monkeypatch.setattr(LG, "terms_typed", lambda step, s: model.terms(s))
    step = _fake_typed_step(model)
    inner = LG.refine_sigma_typed(step, [0.36, 0.53])
    exact = np.array([0.4, 0.5]) - np.array([0.3, -0.2]) / (2 * 2.0)
    assert inner.case == "smooth" and np.allclose(inner.sigma, exact, atol=1e-6)
    g = LG.assemble_typed(step, inner, {"x1": model.g[:, 0], "x2": model.g[:, 1]})
    assert g["x1"] == pytest.approx(0.6, abs=1e-9) and g["x2"] == pytest.approx(-0.3, abs=1e-9)


def test_one_sigma_kink_multiplier_is_m4c_correction(monkeypatch):
    """With one sigma the multiplier formula is M4c's kink correction."""
    model = LG.Terms(0.3, 1.0, np.array([1.0, 1.0, 0.2]), np.array([0.0, 0.2, 0.35]),
                     np.array([1.0, -1.0, 0.5]), ["E"] * 3, np.eye(3), np.zeros((3, 0)), [],
                     [None] * 3, [1] * 3)
    tn = LG.TermsN(np.array([0.3]), 1.0, model.w, model.r, model.drds[:, None], model.kind,
                   model.a_e, model.a_e0, [], model.force, model.natoms)
    inner1 = LG.InnerMinimum(0.3, "kink", model, (0,), 0.0)
    innern = LG.InnerMinimumN(np.array([0.3]), "kink", tn, (0,), (), 0.0)
    g = np.array([0.7, 0.2, -0.4])
    one = LG.assemble(SimpleNamespace(loss=SimpleNamespace(cov_radius=1.0), r0_w=2.0, r0_d=4.0),
                      inner1, {"r0_w": g})["r0_w"]
    step = SimpleNamespace(model=SimpleNamespace(dimension_names=lambda: ["r0_w"],
                                                 dlnp=lambda x: [0.0]), x=[0.0])
    vec = LG.assemble_typed(step, innern, {"r0_w": g})["r0_w"]
    assert vec == pytest.approx(one, rel=1e-14)


# ---- CLI, YAML, TUI ---------------------------------------------------------------------
def test_optimize_translates_the_type_options(tmp_path, monkeypatch):
    from unittest.mock import patch
    dataset = tmp_path / "d.xyz"
    dataset.write_text("0\n\n")
    monkeypatch.chdir(tmp_path)
    captured = {}
    with patch("cli.cmd_optimize.runpy.run_path",
               side_effect=lambda p, run_name=None: captured.setdefault("argv", list(sys.argv))):
        from cli.main import dispatch
        assert dispatch(["optimize", str(dataset), "--output", str(tmp_path / "run"),
                         "--mode", "energygeometry", "--engine", "pydftb",
                         "--switch", "Cu", "Cu2,Cu", "11.0", "1.0",
                         "--type-start", "Cu2", "3.0", "12.0", "--shared-parameters"]) == 0
    argv = captured["argv"]
    assert "--shared_parameters" in argv
    decl = json.loads(argv[argv.index("--types") + 1])
    assert decl == [{"element": "Cu", "labels": ["Cu2", "Cu"], "centre": [11.0], "width": [1.0]}]
    i = argv.index("--type_start")
    assert argv[i + 1:i + 4] == ["Cu2", "3.0", "12.0"]
    import yaml
    saved = yaml.safe_load((tmp_path / "run" / "ptbp_run.yaml").read_text())
    assert saved["shared_parameters"] is True and saved["types"] == decl


def test_default_run_writes_no_new_yaml_keys(tmp_path, monkeypatch):
    from unittest.mock import patch
    dataset = tmp_path / "d.xyz"
    dataset.write_text("0\n\n")
    monkeypatch.chdir(tmp_path)
    with patch("cli.cmd_optimize.runpy.run_path"):
        from cli.main import dispatch
        assert dispatch(["optimize", str(dataset), "--output", str(tmp_path / "run"),
                         "--mode", "dataset"]) == 0
    text = (tmp_path / "run" / "ptbp_run.yaml").read_text()
    assert "shared_parameters" not in text and "types" not in text and "type_start" not in text


def test_plan_counts_the_tables_of_the_types():
    import subprocess

    def plan(*extra):
        out = subprocess.run([sys.executable, str(ROOT / "parameterize.py"), "plan", "dft/dft.xyz",
                              "--mode", "energygeometry", "--engine", "pydftb", "--ncores", "4",
                              "--json", *extra], cwd=ROOT, capture_output=True, text=True,
                             timeout=600)
        assert out.returncode == 0, out.stderr
        report = json.loads(out.stdout)
        return report["workload"]["hotcent_tables"], report["estimates"]["hotcent"]
    one, two = plan(), plan("--switch", "Cu", "Cu2,Cu", "11.0", "1.0")
    assert one[0] == 1 and two[0] == 3                  # Cu2-Cu2, Cu-Cu2, Cu-Cu
    assert two[1] > 2 * one[1]          # the cost model knows the labels' basis (two atoms)


def test_tui_state_writes_the_type_options():
    from cli.tui.state import SetupState, assemble_optimize_yaml
    s = SetupState()
    s.dataset_path = Path("/data/x.xyz")
    out = assemble_optimize_yaml(s)
    assert "shared_parameters" not in out and "types" not in out
    s.shared_parameters, s.switch_types = True, "Cu:Cu2,Cu:11.0:1.0"
    out = assemble_optimize_yaml(s)
    assert out["shared_parameters"] is True
    assert out["types"] == [{"element": "Cu", "labels": ["Cu2", "Cu"], "centre": [11.0],
                             "width": [1.0]}]
    with pytest.raises(ValueError):
        TM.parse_switch_text("Cu:Cu2,Cu:11.0")


# ---- the Loss of parameters per type (pydftb, hotcent replaced by stored files) --------------
def _copy_skf(dst: Path, src: Path, names=None):
    for p in sorted(src.glob("*.skf.gz")):
        a, b = p.name[:-7].split("-")
        if names is not None and (a not in names or b not in names):
            continue
        with gzip.open(p, "rb") as fi, open(dst / p.name[:-3], "wb") as fo:
            shutil.copyfileobj(fi, fo)


def _ho_dataset():
    """Small H/O molecules in boxes with made-up reference energies and forces."""
    rng = np.random.default_rng(5)
    base = [(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
            (["O", "H"], [[0, 0, 0], [0.98, 0, 0]]),
            (["O", "O"], [[0, 0, 0], [1.21, 0, 0]]),
            (["H", "H"], [[0, 0, 0], [0.74, 0, 0]])]
    out = []
    for k in range(6):
        sym, pos = base[k % 4]
        pos = np.array(pos, float) + rng.normal(scale=0.03, size=(len(sym), 3)) + 4.0
        a = _box(sym, pos, a=8.0)
        a.info["energy"] = -14.0 * sym.count("H") / 2 - 430.0 * sym.count("O") + rng.normal(0.5)
        a.set_array("forces", rng.normal(scale=0.5, size=(len(sym), 3)))
        out.append(a)
    return out


def _ho_args(tmp_path, mode="dataset"):
    return argparse.Namespace(eos_points=1, kpt_density=5.0, multi_element=None,
                              optimization_option=mode, log_file=str(tmp_path / "log.out"),
                              para_file=str(tmp_path / "par.out"), E0s={1: -13.6, 8: -430.0},
                              xc="GGA_X_PBE+GGA_C_PBE", superposition="density",
                              results_dir="results")


@linux
def test_loss_per_element_is_the_pure_loss_and_writes_the_step(tmp_path, monkeypatch):
    """Two elements, a parameter set each: the loss is utils.loss_grad.mu at the sigma it
    found (bit for bit), the files of the step and a par.out line per type."""
    eng.configure(engine="pydftb")
    monkeypatch.setattr(L.hotpy, "full_hotcent",
                        lambda self, superposition, opt_elem=(), label="ptbp":
                        _copy_skf(Path("."), SKF / "ptbp_hcocu", {"H", "O"}))
    dft = _ho_dataset()
    model = TM.TypeModel.from_dataset(dft)
    assert model.labels == ["H", "O"] and not model.legacy
    loss = L.Loss(args_input=_ho_args(tmp_path), dft_input=dft, bolt_f11=[0.05] * len(dft),
                  opt_elem=[], path=tmp_path, types=model)
    x = [2.2, 4.0, 2.6, 5.4]
    mu = float(loss.Loss_EnergyGeometry(x))
    step_dir = tmp_path / "results" / model.directory_name(x)
    assert step_dir.is_dir()
    for name in ("dft_final.xyz", "dftb_band.xyz", "dftb_full.xyz", "H-O.skf", "O-H.skf"):
        assert (step_dir / name).exists(), name
    assert "Spline" in (step_dir / "H-O.skf").read_text()
    svec = [float(v) for v in (tmp_path / "par.out").read_text().split(",")[2:6:3]]
    assert len(svec) == 2 and all(0.1 <= s <= 0.8 for s in svec)
    monkeypatch.chdir(step_dir)
    step = LG.StepData.from_files(loss, x=x)
    assert float(LG.mu(step, loss.last_sigma)).hex() == mu.hex()      # the pure loss
    assert svec == pytest.approx(loss.last_sigma, abs=1e-8)
    lines = (tmp_path / "log.out").read_text().splitlines()
    assert len(lines) > 30 and all(len(v.split(",")) == 2 * 3 + 5 for v in lines)
    full = ase.io.read(step_dir / "dftb_full.xyz", ":")
    assert [a.get_chemical_symbols() for a in full] == [a.get_chemical_symbols() for a in dft]
    assert loss.last_failed is False and loss.worst_success == mu      # not a failure


@linux
def test_loss_per_element_when_a_type_has_no_tables(tmp_path, monkeypatch):
    """hotcent made no tables for O (as for Na at extreme radii in the reaction example): the
    structures with O fail in the DFTB stage.  Since M5b-2 the point is a failure (M5b scored it
    on the other structures, a loss biased low): before any success its loss is 1000 per
    structure in dataset mode, the step stops after the DFTB stage, par.out has the failure line."""
    eng.configure(engine="pydftb")
    monkeypatch.setattr(L.hotpy, "full_hotcent",
                        lambda self, superposition, opt_elem=(), label="ptbp":
                        _copy_skf(Path("."), SKF / "ptbp_hcocu", {"H"}))
    dft = _ho_dataset()
    model = TM.TypeModel.from_dataset(dft)
    loss = L.Loss(args_input=_ho_args(tmp_path), dft_input=dft, bolt_f11=[0.05] * len(dft),
                  opt_elem=[], path=tmp_path, types=model)
    x = [2.2, 4.0, 2.6, 5.4]
    mu = float(loss.Loss_EnergyGeometry(x))
    d = tmp_path / "results" / model.directory_name(x)
    assert (d / "search_index.txt").read_text().strip() == "3"         # the H2 structure
    assert mu == L.NO_SUCCESS_FAILURE * len(dft) == 6000.0
    assert loss.last_failed is True and loss.is_failure(x) and loss.worst_success is None
    assert not (d / "dftb_full.xyz").exists()                          # no sigma search
    par = [float(v) for v in (tmp_path / "par.out").read_text().split(",")]
    assert par[2] == par[5] == 0.4321 and par[-1] == mu                # the failure line


@linux
@pytest.mark.skipif(shutil.which("dftb+") is None, reason="dftb+ binary not available")
def test_loss_per_element_on_dftbplus_with_tables_without_repulsive(tmp_path, monkeypatch):
    """DFTB+ engine, dataset mode: hotcent's files have no repulsive spline, and DFTB+ refuses
    them for the isolated atoms' energies (poly_rep No) that the sigma search needs, so the
    search writes a repulsive first; at the end the files hold sigma*'s."""
    eng.configure(engine="dftbplus")

    def tables_without_repulsive(self, superposition, opt_elem=(), label="ptbp"):
        _copy_skf(Path("."), SKF / "ptbp_hcocu", {"H", "O"})
        for p in Path(".").glob("*.skf"):
            lines = p.read_text().splitlines(keepends=True)
            if "Spline \n" in lines:
                p.write_text("".join(lines[:lines.index("Spline \n")]))

    monkeypatch.setattr(L.hotpy, "full_hotcent", tables_without_repulsive)
    dft = _ho_dataset()
    model = TM.TypeModel.from_dataset(dft)
    loss = L.Loss(args_input=_ho_args(tmp_path), dft_input=dft, bolt_f11=[0.05] * len(dft),
                  opt_elem=[], path=tmp_path, types=model)
    x = [2.2, 4.0, 2.6, 5.4]
    mu = float(loss.Loss_EnergyGeometry(x))
    assert loss.last_failed is False and loss.worst_success == mu      # not a failure
    d = tmp_path / "results" / model.directory_name(x)
    assert all("Spline" in (d / f"{a}-{b}.skf").read_text() for a in "HO" for b in "HO")
    monkeypatch.chdir(d)
    step = LG.StepData.from_files(loss, x=x)
    assert float(LG.mu(step, loss.last_sigma)).hex() == mu.hex()


@linux
def test_loss_with_switched_types(tmp_path, monkeypatch, images):
    """Cu in two types (graphite CN 3 -> Cu2, diamond CN 4 -> Cu): the mixed repulsive, a
    sigma per type, the type weights in dftb_full.xyz, label-named SKF files."""
    eng.configure(engine="pydftb")
    decl = {"element": "Cu", "labels": ["Cu2", "Cu"], "centre": 3.5, "width": 0.5}
    sub = images[11:33]
    model = TM.TypeModel.from_dataset(sub, declarations=[decl])
    eng.set_type_switching(model.switches)
    monkeypatch.setattr(L.hotpy, "full_hotcent",
                        lambda self, superposition, opt_elem=(), label="ptbp":
                        _copy_skf(Path("."), SKF / "cu_types"))
    _, bolt = data.get_bolt_factor(sub[5::11], "boltzmann", 3000)
    args = _ho_args(tmp_path, "energygeometry")
    args.eos_points = 11
    loss = L.Loss(args_input=args, dft_input=sub, bolt_f11=bolt, opt_elem=[], path=tmp_path,
                  types=model)
    x = [2.749, 12.036, 2.749, 12.036]
    mu = float(loss.Loss_EnergyGeometry(x))
    d = tmp_path / "results" / model.directory_name(x)
    names = sorted(p.name for p in d.glob("*.skf"))
    assert names == ["Cu-Cu.skf", "Cu-Cu2.skf", "Cu2-Cu.skf", "Cu2-Cu2.skf"]
    full = ase.io.read(d / "dftb_full.xyz", ":")
    w2 = np.array([a.get_array("type_weight_Cu2")[0] for a in full])
    assert np.all(w2[:11] == 1.0) and np.all(w2[11:] == 0.0)
    par = [float(v) for v in (tmp_path / "par.out").read_text().split(",")]
    assert par[-1] == pytest.approx(mu) and 0.1 <= par[2] <= 0.8 and 0.1 <= par[5] <= 0.8
    monkeypatch.chdir(d)
    step = LG.StepData.from_files(loss, x=x)
    parts = {}
    assert float(LG.mu(step, loss.last_sigma, parts)).hex() == mu.hex()
    # the EOS files at sigma* (each group overwrites them, as before M5b: the last group's)
    eos_b = json.loads((d / "dftb.json").read_text())["eos_Cu_0"]
    eos_d = json.loads((d / "dft.json").read_text())["eos_Cu_0"]
    assert (eos_b["dftb_volume"], eos_d["dft_volume"]) == parts["v0"][-1]
    assert "1000.00" not in (d / "volume.csv").read_text()
    assert len(ase.io.read(d / "eos_Cu_0.traj", ":")) == 11


@linux
def test_typed_gradient_request(tmp_path, monkeypatch):
    """GradRequest with parameters per element (synthetic tangents instead of hotcent): the
    same loss as without, a gradient for every outer parameter whose r0_d parts are the
    penalty's."""
    pytest.importorskip("torch")
    from pydftb.skf import SKFile

    from utils import hotcent_tangent as HT
    from utils.hotcent_tangent import Tangent
    eng.configure(engine="pydftb")
    monkeypatch.setattr(L.hotpy, "full_hotcent",
                        lambda self, superposition, opt_elem=(), label="ptbp":
                        _copy_skf(Path("."), SKF / "ptbp_hcocu", {"H", "O"}))

    def fake_tangents(model, x, hp, base_dir, **kw):
        out = {}
        for name in model.dimension_names():
            lab = name.split("_")[-1]
            if name.startswith("r0_w"):
                t = SKFile.read(str(Path(base_dir) / f"{lab}-{lab}.skf"), homo=True).table
                out[name] = Tangent(name, 0.04, tables={(lab, lab): 0.01 * t})
            else:
                out[name] = Tangent(name, 0.0, zero=True, probed=True)
        return out

    monkeypatch.setattr(HT, "compute_types", fake_tangents)
    dft = _ho_dataset()
    model = TM.TypeModel.from_dataset(dft)
    x = [2.2, 4.0, 2.6, 5.4]
    out = {}
    for g in (False, True):
        wd = tmp_path / f"run_{g}"
        wd.mkdir()
        monkeypatch.chdir(wd)
        loss = L.Loss(args_input=_ho_args(wd), dft_input=dft, bolt_f11=[0.05] * len(dft),
                      opt_elem=[], path=wd, types=model)
        req = LG.GradRequest() if g else None
        loss.grad_request = req
        out[g] = (float(loss.Loss_EnergyGeometry(x)), req)
    assert out[False][0].hex() == out[True][0].hex()
    res = out[True][1].result
    assert res.grad is not None, res.reason
    assert set(res.grad) == set(model.dimension_names())
    assert all(np.isfinite(v) for v in res.grad.values())
    mu_star = res.mu - res.gap
    for name, dl in zip(model.dimension_names(), model.dlnp(x)):
        if name.startswith("r0_d"):
            assert res.parts[name]["energies"] == 0.0 and res.parts[name]["forces"] == 0.0
            assert res.grad[name] == pytest.approx(mu_star * dl + res.parts[name]["kink"],
                                                   abs=1e-12 * res.mu)
    assert len(res.line().split(",")) == len(LG.grad_header(model.dimension_names()).split(","))
