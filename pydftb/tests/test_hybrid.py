"""Range-separated hybrid functionals (milestone M6h-1): LC and CAM for molecules; with
non-collinear spin (M6h-1b).  Periodic systems at the Gamma point (M6h-3a): test_hybrid_periodic.py.

Comparisons with DFTB+ are the frozen references *_lc*, *_cam*, c2h4_global and the restart
cases *_lc*_read_charges* in test_reference.py, and h2o_lc_opt, h2o_lc_md, ch3_lc_spin_md and
ch3_lc_noncolinear_md in test_md_opt.py; here the building blocks, self-consistency, limits and
errors (no dftb+).
"""
import dataclasses
import gzip
import os

import numpy as np
import pytest
from ase import Atoms, units
from scipy.integrate import quad

from pydftb import Hybrid
from pydftb import hybrid as hy
from pydftb.api import SCC
from pydftb.ase_calc import PyDFTB
from pydftb.bands import eigenstates
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.dftbplus_io import read_charges, write_charges
from pydftb.params import ParamSet
from pydftb.skf import SKFile
from pydftb.system import System
from pydftb.validation import refgen
from pydftb.validation.cases import ONS, SKF_SETS, TILT, U_O, W_HCO
from pydftb.validation.make_hybrid_sets import with_range_sep

from .conftest import SKF_HCOCU

LC_SET = SKF_SETS["ptbp_hcocu_lc"]
MAXL = {"H": 0, "C": 1, "O": 1, "Cu": 2}
WATER = (["O", "H", "H"], np.array([[0.0, 0.0, 0.119], [0.0, 0.763, -0.477],
                                    [0.0, -0.763, -0.477]]) / BOHR)


def _sk(path, labels, **kw):
    return ParamSet(path, max_l={lab: MAXL[lab] for lab in set(labels)},
                    **kw).skset(sorted(set(labels)))


def _case(name, **options):
    """(result, engine, labels, positions in Bohr) of a reference case; exact SK derivatives
    (no fd_delta) unless ``options`` say otherwise."""
    case = refgen.CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    res, calc = refgen.run_pydftb(case, inp, engine=True, **options)
    return res, calc, inp["labels"], np.array(inp["positions_angstrom"]) / BOHR


# ---- gamma (DFTB+ rshgamma.F90) -------------------------------------------------------
def _gamma_integral(ua, ub, r, omega=None):
    """gamma of the densities tau^3/(8 pi) exp(-tau r), tau = 3.2 U, in reciprocal space: the
    Coulomb kernel, or with ``omega`` the long-range Yukawa kernel (1 - exp(-omega r)) / r."""
    ta, tb = 3.2 * ua, 3.2 * ub

    def f(k):
        fa, fb = ta ** 4 / (ta ** 2 + k ** 2) ** 2, tb ** 4 / (tb ** 2 + k ** 2) ** 2
        kern = 1.0 if omega is None else omega ** 2 / (k ** 2 + omega ** 2)
        return fa * fb * kern * (1.0 if r == 0 else np.sinc(k * r / np.pi))
    edges = [0.0, 1.0, 5.0, 20.0, 100.0, np.inf]
    return 2 / np.pi * sum(quad(f, a, b, limit=500, epsabs=1e-14, epsrel=1e-13)[0]
                           for a, b in zip(edges[:-1], edges[1:]))


@pytest.mark.parametrize("ua, ub", [(0.4, 0.4), (0.4, 0.6), (0.3, 0.55), (0.4, 0.4001),
                                    (0.4, 0.400001)])
@pytest.mark.parametrize("r", [0.7, 2.5, 6.0])
def test_gamma_closed_forms_are_the_integrals(ua, ub, r):
    """Every branch (equal, nearly equal and unequal U) of gamma_HF and gamma_LR against the
    integral; DFTB+'s closed form for unequal U is off by 1e-7 and 2 at the last two pairs."""
    assert hy.gamma_hf(ua, ub, r) == pytest.approx(_gamma_integral(ua, ub, r), abs=1e-13)
    assert hy.gamma_lr(ua, ub, 0.3, r) == pytest.approx(_gamma_integral(ua, ub, r, 0.3),
                                                        abs=1e-13)
    assert hy.gamma_hf(ub, ua, r) == pytest.approx(hy.gamma_hf(ua, ub, r), abs=1e-15)


def test_gamma_limits():
    u = 0.4
    assert hy.gamma_hf(u, u, 0.0) == pytest.approx(u, abs=1e-15)          # on site: U
    assert hy.gamma_lr(u, u, 0.3, 0.0) == pytest.approx(_gamma_integral(u, u, 0.0, 0.3),
                                                        abs=1e-13)
    for ub in (u, 0.6):                                       # omega = 0: no long-range part
        assert np.abs(hy.gamma_lr(u, ub, 0.0, np.array([0.0, 1.0, 3.0]))).max() < 1e-14
    r = 100.0                                                 # far apart: 1/r
    assert hy.gamma_hf(u, 0.6, r) == pytest.approx(1 / r, abs=1e-14)
    assert hy.gamma_lr(u, 0.6, 0.3, r) == pytest.approx(1 / r, abs=1e-13)
    # CAM: alpha gamma_HF + beta gamma_LR, a coefficient below 100 eps left out (as DFTB+)
    d = np.array([0.0, 1.5, 4.0])
    assert np.abs(hy.gamma_cam(u, 0.6, d, 0.3, 1.0, 0.0) - hy.gamma_hf(u, 0.6, d)).max() == 0.0
    assert np.abs(hy.gamma_cam(u, 0.6, d, 0.3, 0.0, 1.0)
                  - hy.gamma_lr(u, 0.6, 0.3, d)).max() == 0.0
    assert np.all(hy.gamma_cam(u, 0.6, d, 0.3, 1e-15, 1e-15) == 0.0)
    # tau within MinHubDiff: the equal-U formula with the mean tau (DFTB+'s switch)
    du = 0.5 * hy.MIN_HUB_DIFF / 3.2
    assert hy.gamma_hf(u, u + du, 2.0) == pytest.approx(
        hy.gamma_hf(u + du / 2, u + du / 2, 2.0), abs=1e-15)


@pytest.mark.parametrize("ua, ub", [(0.4, 0.4), (0.4, 0.6), (0.4, 0.4001)])
def test_gamma_derivatives(ua, ub):
    r, h = np.array([0.5, 2.0, 6.0]), 1e-5
    pairs = ((lambda x: hy.gamma_hf(ua, ub, x), lambda x: hy.dgamma_hf(ua, ub, x)),
             (lambda x: hy.gamma_lr(ua, ub, 0.3, x), lambda x: hy.dgamma_lr(ua, ub, 0.3, x)),
             (lambda x: hy.gamma_cam(ua, ub, x, 0.3, 0.19, 0.46),
              lambda x: hy.dgamma_cam(ua, ub, x, 0.3, 0.19, 0.46)))
    for g, dg in pairs:
        assert np.abs((g(r + h) - g(r - h)) / (2 * h) - dg(r)).max() < 1e-9
        assert dg(np.array([0.0]))[0] == 0.0                   # on site


# ---- RangeSep sections ----------------------------------------------------------------
def _write_set(path, lines: dict, elements=("H", "O")) -> str:
    """Copies of ptbp_hcocu's files of ``elements`` in ``path``, with the RangeSep line of
    ``lines[(a, b)]`` (or ``lines["all"]``; none if neither)."""
    os.makedirs(path, exist_ok=True)
    for a in elements:
        for b in elements:
            with gzip.open(os.path.join(SKF_HCOCU, f"{a}-{b}.skf.gz"), "rt") as fh:
                text = fh.read()
            line = lines.get((a, b), lines.get("all"))
            if line is not None:
                text = with_range_sep(text, line)
            with open(os.path.join(path, f"{a}-{b}.skf"), "w") as fh:
                fh.write(text)
    return str(path)


@pytest.mark.parametrize("line, expected", [
    ("LC 0.3", ("lc", 0.3, 0.0, 1.0)),
    ("CAM 0.3 0.19 0.46", ("cam", 0.3, 0.19, 0.46)),
    ("CAM 0.4, 0.0, 1.0", ("lc", 0.4, 0.0, 1.0)),          # DFTB+ reclassifies CAM
    ("CAM 0.0 0.2 0.0", ("global", 0.0, 0.2, 0.0)),
    ("CAM 0.3 0.0 0.0", None),
])
def test_range_sep_section(tmp_path, line, expected):
    path = _write_set(tmp_path, {"all": line})
    assert SKFile.read(os.path.join(path, "O-O.skf")).range_sep == expected


@pytest.mark.parametrize("line, match", [
    ("XC 0.3", "Unknown hybrid xc-functional method"),
    ("LC", "Error in reading hybrid xc-functional parameters"),
    ("CAM 0.3 0.2", "Error in reading hybrid xc-functional parameters"),
    ("LC -0.1", "Range-separation parameter is negative"),
])
def test_range_sep_errors(tmp_path, line, match):
    path = _write_set(tmp_path, {"all": line})
    with pytest.raises(ValueError, match=match):
        SKFile.read(os.path.join(path, "O-O.skf"))


def test_set_needs_every_pair_and_one_value(tmp_path):
    lc = "LC 0.3"
    one_way = _write_set(tmp_path / "a", {("H", "H"): lc, ("O", "O"): lc, ("H", "O"): lc})
    assert _sk(one_way, ["H", "O"]).range_sep() == ("lc", 0.3, 0.0, 1.0)   # O-H without: fine
    no_oo = _write_set(tmp_path / "b", {("H", "H"): lc, ("H", "O"): lc, ("O", "H"): lc})
    with pytest.raises(ValueError, match="O-O.skf.*no RangeSep section"):
        _sk(no_oo, ["H", "O"]).range_sep()
    mixed = _write_set(tmp_path / "c", {"all": lc, ("H", "H"): "LC 0.35"})
    with pytest.raises(ValueError, match="disagree"):
        _sk(mixed, ["H", "O"]).range_sep()


# ---- options and refusals ---------------------------------------------------------------
def test_hybrid_spec():
    assert Hybrid.make(None) is None
    assert Hybrid.make("LC") == Hybrid()
    assert (Hybrid.make({"type": "CAM", "screening": "Neighbour", "cutoff_reduction": 2.0})
            == Hybrid("cam", "neighbour", cutoff_reduction=2.0))
    for bad, match in (("global", "no 'Global'"), ({"screening": "sparse"}, "screening"),
                       ({"screening": "neighbour", "cutoff_reduction": -1.0}, "zero or positive"),
                       ({"cutoff_reduction": 1.0}, "matrix screening"),
                       ({"type": "lc", "omega": 0.3}, "unknown keys"),
                       ({"screening": "threshold", "threshold": -1.0}, "negative")):
        with pytest.raises(ValueError, match=match):
            Hybrid.make(bad)
    with pytest.raises(TypeError):
        Hybrid.make(0.3)


def test_sk_files_and_the_option():
    labels, pos = WATER
    lc = _sk(LC_SET, labels)
    with pytest.raises(ValueError, match="hybrid parametrisation.*hybrid=False"):
        DFTB(lc)
    plain = DFTB(_sk(SKF_HCOCU, labels), scc_tol=1e-10).calculate(labels, pos)
    ignored = DFTB(lc, scc_tol=1e-10, hybrid=False).calculate(labels, pos)
    assert ignored.energy == pytest.approx(plain.energy, abs=1e-12) and ignored.e_fock == 0.0
    with pytest.raises(ValueError, match="no RangeSep section"):
        DFTB(_sk(SKF_HCOCU, labels), hybrid="lc")
    with pytest.raises(ValueError, match="type conflict"):
        DFTB(_sk(SKF_SETS["ptbp_hcocu_cam"], labels), hybrid="lc")
    assert DFTB(lc, hybrid="cam").hybrid.kind == "cam"           # CAM takes LC files


@pytest.mark.parametrize("opts, match", [
    ({"scc": False}, "require SCC"),
    ({"spin_orbit": {"O": [0.0, 0.01]}, "spin": "noncolinear", "spin_constants": W_HCO},
     "spin orbit"),
    ({"third_order": "onsite", "hubbard_derivs": {"H": -0.19, "O": -0.16}}, "3rd-order"),
    ({"hybrid": {"type": "lc", "screening": "threshold"}, "spin": True,
      "spin_constants": W_HCO}, "thresholded"),
    ({"analysis": {"atom_energies": True}}, "Atomic resolved energies"),
    ({"analysis": {"coord_derivatives": True}}, "Coordinate derivative"),
    ({"spin": "noncolinear", "spin_constants": W_HCO,
      "hybrid": {"type": "lc", "screening": "neighbour"}}, "matrix alogorithm"),
    ({"casida": {"n_excitations": 3, "energy_window": 0.1}}, "window options"),
    ({"casida": {"n_excitations": 3, "oscillator_window": 0.01}}, "window options"),
    ({"casida": {"n_excitations": 3}, "onsite_corrections": ONS}, "Onsite corrections not imp"),
    ({"casida": {"n_excitations": 3, "solver": "arpack"}}, "Hermitian Omega"),
    ({"analysis": {"polarisability": {"frequencies": [0.1]}}}, "Finite frequency hybrid"),
    ({"analysis": {"response_kernel": {"frequencies": [0.1]}}}, "Finite frequency hybrid"),
    ({"spin": "noncolinear", "spin_constants": W_HCO, "analysis": {"polarisability": True}},
     "two-component response analyses are milestone M6h-4b"),
    ({"truncate_sk_range": 10.0}, "truncate_sk_range"),
    ({"force_evaluation": "dynamics"}, "traditional"),
])
@pytest.mark.filterwarnings("ignore:Artificially truncating")
def test_refusals(opts, match):
    labels, _ = WATER
    with pytest.raises(ValueError, match=match):
        DFTB(_sk(LC_SET, labels), **{"hybrid": "lc", **opts})


def test_refusals_of_the_setup():
    labels, pos = WATER
    sk = _sk(LC_SET, labels)
    with pytest.raises(ValueError, match="shell-resolved"):
        DFTB(_sk(LC_SET, labels, shell_resolved=True), hybrid="lc")
    with pytest.raises(ValueError, match="Monkhorst-Pack-like sampling"):    # M6h-3b
        DFTB(sk, hybrid="lc", kpts=(np.array([[0.0, 0.0, 0.25]]), np.ones(1))).calculate(
            labels, pos, np.eye(3) * 20.0, np.ones(3, bool))
    with pytest.raises(ValueError, match="too short"):
        DFTB(sk, hybrid={"screening": "neighbour", "cutoff_reduction": 100.0}).calculate(
            labels, pos)
    with pytest.raises(ValueError, match="not currently implemented for XLBOMD"):
        DFTB(sk, hybrid="lc")._check_xl()


def test_torch_and_autodiff_refuse():
    pytest.importorskip("torch")
    from pydftb.autodiff import evaluate
    labels, pos = WATER
    sk = _sk(LC_SET, labels)
    with pytest.raises(ValueError, match="NumPy backend only"):
        DFTB(sk, backend="torch", hybrid="lc")
    with pytest.raises(NotImplementedError, match="hybrid"):
        evaluate(DFTB(sk, hybrid="lc"), labels, pos)


# ---- the exchange matrix ----------------------------------------------------------------
def test_fock_matrix_is_the_energy_gradient():
    """H_x is linear, symmetric and self-adjoint, E = 1/2 <H_x[D], D> has the gradient H_x,
    and the four matrix products are DFTB+'s sum over orbital pairs."""
    _, calc, labels, pos = _case("h2co_cam")
    t = calc._setup(labels, pos, None, None, False, {}).hybrid
    rng = np.random.default_rng(3)
    d1, d2, x = (m + m.T for m in rng.normal(size=(3, t.n, t.n)))
    h1, h2 = t.fock([d1])[0], t.fock([d2])[0]
    assert np.abs(h1 - h1.T).max() < 1e-13
    assert np.sum(h1 * d2) == pytest.approx(np.sum(d1 * h2), rel=1e-12)
    assert np.abs(t.fock([d1 + 2 * d2])[0] - (h1 + 2 * h2)).max() < 1e-12

    def e(d):
        return t.energy(t.fock([d]), [d])
    eps = 1e-6
    assert (e(d1 + eps * x) - e(d1 - eps * x)) / (2 * eps) == pytest.approx(np.sum(h1 * x),
                                                                              rel=1e-7)
    s, g = t.S, t.G
    ref = np.zeros((t.n, t.n))
    for m in range(t.n):
        for nu in range(t.n):
            w = g[m, nu] + g[m, :][None, :] + g + g[:, nu][:, None]   # over (a, b)
            ref[m, nu] = -0.125 * np.sum(s[m][:, None] * d1 * s[:, nu][None, :] * w)
    assert np.abs(h1 - ref).max() < 1e-12
    # spin: -1/4 per channel
    assert np.abs(t.fock([d1, d2])[0] - 2 * h1).max() < 1e-12


def test_packing_and_reference():
    _, calc, labels, pos = _case("h2o_lc")
    t = calc._setup(labels, pos, None, None, False, {}).hybrid
    rng = np.random.default_rng(1)
    d = [m + m.T for m in rng.normal(size=(2, t.n, t.n))]
    x = t.pack(d)
    assert len(x) == 2 * t.packed_size == t.n * (t.n + 1)
    back = t.unpack(x, 2)
    assert all(np.array_equal(a, b) for a, b in zip(back, d))
    with pytest.raises(ValueError, match="packed DeltaP"):
        t.unpack(x, 1)
    # P0: the reference populations on the diagonal (S_mumu = 1), half per spin channel
    assert np.trace(t.reference(1)) == pytest.approx(8.0)
    assert np.array_equal(t.reference(2), 0.5 * t.reference(1))


# ---- screenings -------------------------------------------------------------------------
def test_screenings():
    """NeighbourBased without a reduction and Thresholded with threshold 0 are the matrix
    algorithm; a threshold of 1e-6 changes the energy by less than 1e-5 Ha."""
    res, calc, labels, pos = _case("c6h6_lc", scc_tol=1e-12)
    sk = calc.sk
    nb0 = DFTB(sk, scc_tol=1e-12, hybrid={"screening": "neighbour"}).calculate(
        labels, pos, forces=True)
    assert nb0.energy == pytest.approx(res.energy, abs=1e-11)
    exact = DFTB(sk, scc_tol=1e-12, hybrid="lc").calculate(labels, pos, forces=True)
    assert np.abs(nb0.forces - exact.forces).max() < 1e-10
    thr0 = DFTB(sk, scc_tol=1e-12, hybrid={"screening": "threshold", "threshold": 0.0}
                ).calculate(labels, pos)
    assert thr0.energy == pytest.approx(res.energy, abs=1e-10)
    thr = DFTB(sk, scc_tol=1e-8, hybrid={"screening": "threshold"}).calculate(labels, pos)
    assert thr.energy == pytest.approx(res.energy, abs=1e-5)
    # a large reduction leaves pairs out: another energy
    red = DFTB(sk, scc_tol=1e-10, hybrid={"screening": "neighbour", "cutoff_reduction": 10.0}
               ).calculate(labels, pos)
    assert abs(red.energy - res.energy) > 1e-4


# ---- results ----------------------------------------------------------------------------
def test_energy_terms():
    res, *_ = _case("ch3oh_lc")
    assert res.e_fock < 0.0 and res.terms["Fock"] == res.e_fock
    total = sum(v for k, v in res.terms.items() if k != "-TS")
    assert total == pytest.approx(res.energy, abs=1e-12)
    assert res.free_energy == pytest.approx(res.energy - res.ts, abs=1e-14)


def test_closed_shell_spin_equals_restricted():
    res, *_ = _case("h2o_lc", scc_tol=1e-12)
    spin, *_ = _case("h2o_lc_spin_closed", scc_tol=1e-12)
    assert spin.energy == pytest.approx(res.energy, abs=1e-10)
    assert spin.e_fock == pytest.approx(res.e_fock, abs=1e-10)
    assert np.abs(spin.charges - res.charges).max() < 1e-9


def _fd_forces(calc, labels, pos, atoms, h=1e-6):
    out = []
    for a in atoms:
        for c in range(3):
            e = []
            for sgn in (1, -1):
                p = pos.copy()
                p[a, c] += sgn * h
                e.append(calc.calculate(labels, p, forces=False).free_energy)
            out.append(-(e[0] - e[1]) / (2 * h))
    return np.array(out)


@pytest.mark.parametrize("name", ["h2o_lc", "h2co_cam", "c2h4_global", "ch3_lc_spin",
                                  "h2co_lc_plusu", "cuco_lc_300K", "h2o_lc_onsite",
                                  "ch3oh_lc_field_pc", "hco_lc_noncolinear",
                                  "h3_lc_noncolinear_120"])
def test_forces_match_energy_differences(name):
    """Central differences with h = 1e-6 Bohr (below the distance of water's H-H pair to an SK
    grid node); their noise reaches 2.2e-7 Ha/Bohr for the 300 K CuCO (minimal environment)."""
    res, calc, labels, pos = _case(name, scc_tol=1e-12)
    res = calc.calculate(labels, pos, forces=True)
    atoms = [0, len(labels) - 1]
    fd = _fd_forces(calc, labels, pos, atoms)
    assert np.abs(res.forces[atoms].ravel() - fd).max() < 5e-7


def test_threshold_and_reduced_neighbour_forces_are_exact_ones():
    """Like DFTB+, the screened algorithms take the exact force expression with DeltaP_out:
    the matrix algorithm's forces at their density, not their energy's derivative."""
    _, calc, labels, pos = _case("c6h6_lc_neighbour_10", scc_tol=1e-12)
    res = calc.calculate(labels, pos, forces=True)
    st = calc._setup(labels, pos, None, None, True, {})
    assert st.hybrid.spec.cutoff_reduction == 10.0
    fd = _fd_forces(calc, labels, pos, [0])
    assert np.abs(res.forces[0] - fd).max() > 1e-6


# ---- restarts, charge files, eigenstates -------------------------------------------------
@pytest.mark.parametrize("name", ["h2o_lc", "ch3_lc_spin", "h2co_lc_plusu"])
def test_restart_from_the_mixed_vector(name):
    res, calc, labels, pos = _case(name, scc_tol=1e-10)
    nspin = 2 if res.q_spin is not None or calc.spin else 1
    n = calc._setup(labels, pos, None, None, False, {}).basis.norb
    npop = len(res.q_in) - nspin * n * (n + 1) // 2
    again = calc.calculate(labels, pos, q_init=res.q_in)
    assert again.n_scc <= 2 and again.energy == pytest.approx(res.energy, abs=1e-10)
    pops = calc.calculate(labels, pos, q_init=res.q_in[:npop])     # DeltaP_in = 0
    assert pops.energy == pytest.approx(res.energy, abs=1e-9) and pops.n_scc > again.n_scc
    with pytest.raises(ValueError, match="q_init of a hybrid calculation"):
        calc.calculate(labels, pos, q_init=res.q_in[:-1])


@pytest.mark.parametrize("name, text", [("ch3_lc_spin", False), ("h2co_lc_plusu", True)])
def test_charge_file_round_trip(tmp_path, name, text):
    res, calc, labels, pos = _case(name, scc_tol=1e-10)
    data = calc.charge_data(labels, res)
    n = data.q.shape[1]
    assert data.density_matrix and data.rho.shape == (2 if calc.spin else 1, n, n)
    assert np.all(np.triu(data.rho[0], 1) == 0.0)                 # DFTB+'s lower triangle
    path = tmp_path / ("charges.dat" if text else "charges.bin")
    write_charges(path, data)
    back = read_charges(path, data.norb_atom)
    assert back.density_matrix and np.abs(back.rho - data.rho).max() == 0.0
    assert np.abs(calc.populations_from_file(labels, back) - res.q_in).max() < 1e-14
    # without the density matrix: the populations only (DeltaP_in = 0, as DFTB+)
    back.rho, back.density_matrix = None, False
    q = calc.populations_from_file(labels, back)
    assert np.abs(q - res.q_in[:len(q)]).max() < 1e-14              # (q, m) -> (up, down)
    assert len(q) == len(res.q_in) - data.rho.shape[0] * n * (n + 1) // 2


def test_plain_calculation_ignores_a_density_matrix(tmp_path):
    res, calc, labels, pos = _case("h2o_lc", scc_tol=1e-10)
    write_charges(tmp_path / "charges.bin", calc.charge_data(labels, res))
    plain = DFTB(_sk(SKF_HCOCU, labels), scc_tol=1e-10)
    q = plain.populations_from_file(labels, str(tmp_path / "charges.bin"))
    assert np.array_equal(q, res.q_in[:len(q)]) and len(q) == len(labels)


@pytest.mark.parametrize("name, tol", [("h2o_lc", 1e-12), ("ch3_lc_spin", 1e-12),
                                       ("h2co_lc_plusu", 1e-12), ("hco_lc_noncolinear", 1e-12),
                                       ("c6h6_lc_threshold", 1e-5)])
def test_eigenstates_reproduce_the_scc_levels(name, tol):
    """The SCC's last Hamiltonian from q_in: H_x[DeltaP_in] included (the threshold screening:
    a fresh build, not the SCC's accumulated one)."""
    res, calc, labels, pos = _case(name)
    st = eigenstates(calc, labels, pos, None, None, res.q_in)
    ev = np.asarray(res.eigenvalues).reshape(st.eigenvalues.shape)
    assert np.abs(st.eigenvalues - ev).max() < tol
    plain = eigenstates(calc, labels, pos)                         # reference populations
    assert np.abs(plain.eigenvalues - ev).max() > 1e-3


def test_api_and_ase_calculator():
    labels, pos = WATER
    params = ParamSet(LC_SET, max_l={"H": 0, "O": 1})
    system = System(labels, pos)
    scc = SCC(system, params, scc_tol=1e-10, hybrid="lc")
    e = scc.kernel()
    direct = DFTB(_sk(LC_SET, labels), scc_tol=1e-10, hybrid="lc").calculate(labels, pos)
    assert e == pytest.approx(direct.energy, abs=1e-12)
    scc.forces()                                                   # restarts from q_in
    assert scc.result.n_scc <= 2
    assert scc.states().eigenvalues.shape[-1] == 6
    atoms = Atoms(labels, positions=pos * BOHR)
    red = 6.0                                                      # Angstrom
    for spec in ({"type": "lc", "screening": "neighbour", "cutoff_reduction": red},
                 Hybrid("lc", "neighbour", cutoff_reduction=red)):
        atoms.calc = PyDFTB(LC_SET, max_l={"H": 0, "O": 1}, scc_tol=1e-10, hybrid=spec)
        ref = DFTB(_sk(LC_SET, labels), scc_tol=1e-10,
                   hybrid={"screening": "neighbour", "cutoff_reduction": red / BOHR}
                   ).calculate(labels, pos)
        assert atoms.get_potential_energy() == pytest.approx(ref.energy * units.Hartree,
                                                             rel=1e-12)


# ---- non-collinear spin (M6h-1b) ---------------------------------------------------------
def _hermitian(rng, n):
    m = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return m + m.conj().T


def _spinor_terms(name="hco_lc_noncolinear"):
    _, calc, labels, pos = _case(name)
    return calc._setup(labels, pos, None, None, False, {}).hybrid


def test_spinor_fock_is_the_block_operator():
    """The two-component H_x is the operator on S x 1 and G on all four blocks with c = -1/4,
    Hermitian; E = 1/2 Re <H_x[D], D> has the gradient H_x; a block-diagonal real DeltaP gives
    the collinear channels' H_x."""
    t = _spinor_terms()
    n = t.n
    rng = np.random.default_rng(5)
    d, x = _hermitian(rng, 2 * n), _hermitian(rng, 2 * n)
    h = t.fock([d])[0]
    s2 = np.zeros((2 * n, 2 * n))
    s2[:n, :n] = s2[n:, n:] = t.S
    g2 = np.tile(t.G, (2, 2))
    ref = -0.25 * ((s2 @ d @ s2) * g2 + ((s2 @ d) * g2) @ s2 + s2 @ (d * g2) @ s2
                   + s2 @ ((d @ s2) * g2))
    assert np.abs(h - ref).max() < 1e-12 and np.abs(h - h.conj().T).max() < 1e-12

    def e(m):
        return t.energy(t.fock([m]), [m])
    eps = 1e-6
    assert (e(d + eps * x) - e(d - eps * x)) / (2 * eps) == pytest.approx(
        np.sum(h * x.conj()).real, rel=1e-7)
    up, dn = (m + m.T for m in rng.normal(size=(2, n, n)))
    blk = np.zeros((2 * n, 2 * n), complex)
    blk[:n, :n], blk[n:, n:] = up, dn
    h = t.fock([blk])[0]
    collinear = [-0.25 * t._four_terms(t.S, m) for m in (up, dn)]
    assert np.abs(h[:n, :n] - collinear[0]).max() < 1e-12
    assert np.abs(h[n:, n:] - collinear[1]).max() < 1e-12 and np.abs(h[n:, :n]).max() == 0.0


def test_spinor_packing_and_reference():
    t = _spinor_terms()
    n2 = 2 * t.n
    d = _hermitian(np.random.default_rng(2), n2)
    x = t.pack([d])
    assert len(x) == t.packed_size == n2 * (n2 + 1)
    assert np.abs(t.unpack(x, 1)[0] - d).max() == 0.0
    d2 = d + 1e-3 * (1 + 1j)
    k = len(x) // 2
    assert t.error(t.pack([d2]), x) == pytest.approx(np.abs(d2 - d)[np.tril_indices(n2)].max())
    assert t.error(t.pack([d2]), x) == pytest.approx(np.sqrt(2) * 1e-3, rel=1e-9)
    assert len(x[:k]) == len(x[k:])
    with pytest.raises(ValueError, match="two-component"):
        t.unpack(x[:-1], 1)
    p0 = t.reference(1)                    # half the reference populations on both blocks
    assert np.trace(p0).real == pytest.approx(11.0) and np.abs(p0.imag).max() == 0.0


def test_spinor_pair_coefficient_is_the_overlap_derivative():
    """dE_x/dS at fixed DeltaP for a DeltaP with complex spin-diagonal blocks and a lower left
    block that is not symmetric (what a non-collinear SCC passes through)."""
    t = _spinor_terms()
    n = t.n
    d = _hermitian(np.random.default_rng(7), 2 * n)

    class _All:
        rows, cols = np.indices((n, n)).reshape(2, -1)
    x = t.pair_coefficient([d], _All).reshape(n, n)
    s0 = t.S.copy()
    for i, j in [(0, 3), (2, 5), (1, 1)]:
        e = []
        for sgn in (1, -1):
            t.S = s0.copy()
            t.S[i, j] += sgn * 1e-6
            if i != j:
                t.S[j, i] += sgn * 1e-6
            e.append(t.energy(t.fock([d]), [d]))
        t.S = s0
        assert (e[0] - e[1]) / 2e-6 == pytest.approx(x[i, j] + (x[j, i] if i != j else 0.0),
                                                      rel=1e-6)


def _noncollinear(name, **changes):
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], **changes)
    inp = refgen._inputs(case)
    return refgen.run_pydftb(case, inp, engine=True, scc_tol=1e-11)


def test_noncollinear_along_z_and_tilted_equal_collinear():
    """Without spin-orbit coupling the spin direction is free: along z or tilted, the energy,
    E_Fock, forces and charges are the collinear calculation's."""
    nc = refgen.CASES_BY_NAME["hco_lc_noncolinear"].noncolinear
    z = {**nc, "initial_spins": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]}
    col, _ = _noncollinear("hco_lc_spin_neighbour", hybrid="lc")
    for spins in (nc, z):
        res, _ = _noncollinear("hco_lc_noncolinear", noncolinear=spins)
        assert res.energy == pytest.approx(col.energy, abs=1e-9)
        assert res.e_fock == pytest.approx(col.e_fock, abs=1e-9)
        assert np.abs(res.forces - col.forces).max() < 1e-8
        assert np.abs(res.charges - col.charges).max() < 1e-8
        m = np.asarray(res.magmoms)                       # (n_atom, 3), along the start
        direction = np.asarray(spins["initial_spins"][1])
        assert np.abs(m - np.outer(col.magmoms, direction)).max() < 1e-7


@pytest.mark.parametrize("collinear, block", [("hco_lc_plusu_spin", {"plus_u": U_O}),
                                               ("hco_lc_onsite_spin", {"onsite": ONS})])
def test_noncollinear_blocks_come_from_deltap_in(collinear, block):
    """+U and onsite corrections: the input blocks after mixing are those of DeltaP_in + P0, as
    with collinear spin (DFTB+'s two-component path keeps the first output's blocks: 2.5e-3 and
    1.3e-3 Ha above, M6h-1b report): the collinear result (a DFTB+ reference)."""
    col, _ = _noncollinear(collinear)
    res, _ = _noncollinear("hco_lc_noncolinear", **block)
    assert res.energy == pytest.approx(col.energy, abs=1e-9)
    assert res.e_plus_u == pytest.approx(col.e_plus_u, abs=1e-9)
    assert res.e_onsite == pytest.approx(col.e_onsite, abs=1e-9)
    assert np.abs(res.forces - col.forces).max() < 1e-8


def test_noncollinear_restart_and_charge_files(tmp_path):
    res, calc = _noncollinear("hco_lc_noncolinear")
    inp = refgen._inputs(refgen.CASES_BY_NAME["hco_lc_noncolinear"])
    labels, pos = inp["labels"], np.array(inp["positions_angstrom"]) / BOHR
    n2 = 2 * calc._charge_layout(labels)[0].norb
    npop = len(res.q_in) - n2 * (n2 + 1)
    again = calc.calculate(labels, pos, q_init=res.q_in)
    assert again.n_scc <= 2 and again.energy == pytest.approx(res.energy, abs=1e-10)
    pops = calc.calculate(labels, pos, q_init=res.q_in[:npop])     # DeltaP_in = 0
    assert pops.energy == pytest.approx(res.energy, abs=1e-9) and pops.n_scc > again.n_scc
    # DFTB+ writes no charge file for a non-collinear hybrid, and reads one without a density
    with pytest.raises(ValueError, match="writes no charge file"):
        calc.charge_data(labels, res)
    plain = DFTB(_sk(SKF_HCOCU, labels), scc_tol=1e-10, spin="noncolinear",
                 spin_constants={k: W_HCO[k] for k in set(labels)},
                 initial_spins=[[0.0, 0.0, 0.0], TILT, [0.0, 0.0, 0.0]])
    data = plain.charge_data(labels, plain.calculate(labels, pos))
    write_charges(tmp_path / "charges.bin", data)
    q = calc.populations_from_file(labels, str(tmp_path / "charges.bin"))
    assert len(q) == npop
    data.rho, data.density_matrix = np.zeros((1, n2 // 2, n2 // 2)), True
    with pytest.raises(ValueError, match="non-collinear hybrid"):
        calc.populations_from_file(labels, data)

