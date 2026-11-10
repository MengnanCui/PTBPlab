"""P2: the DFTB+ engine (pydftb.dftbplus_calc), one set of options for both engines
(pydftb.engines), ``pydftb run`` and the dual checks (pydftb.validation.dual).

Without dftb+ the input writer, the reading of stored DFTB+ outputs, the option layer and the
pydftb engine are tested; the tests marked ``dftbplus`` run DFTB+."""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule

from pydftb import dftbplus_calc as dc
from pydftb import dftbplus_io as io
from pydftb.ase_calc import PyDFTB, kpoints_of
from pydftb.constants import BOHR, HARTREE
from pydftb.engines import (
    OPTIONS,
    EngineOptionError,
    check_options,
    common_results,
    compare_results,
    make_calculator,
)
from pydftb.kpoints import monkhorst_pack, reduce_time_reversal
from pydftb.validation import dftbplus as dp
from pydftb.validation import dual
from pydftb.validation.cases import CASES_BY_NAME, REF_DIR, SKF_SETS

from .conftest import DATA, SKF_HCOCU

OUTPUTS = os.path.join(DATA, "dftbplus_outputs")


def plain_water():
    return molecule("H2O")


def copper():
    return bulk("Cu", "fcc", a=3.61)


def hsd(atoms, tmp_path, properties=("energy",), skdir=SKF_HCOCU, **options):
    return dc.write_input(atoms, skdir, options, properties, cache=str(tmp_path / "skf")).hsd


# ---- every frozen input (V1) -------------------------------------------------------------------
def test_every_frozen_input_is_what_the_engine_writes():
    report = dual.check_inputs(verbose=False)
    assert len(report) == 590          # 519 single points, 13 band runs, 24 Hessians, 34 drivers
    assert {k: v for k, v in report.items() if v} == {}


def test_hsd_diff_reads_as_dftbplus_does():
    a = "A {\n  X = 1.0\n  Y = Yes\n}\nB = 2\n"
    assert dual.hsd_diff(a, "B = 2.0000000000000004\nA {\n  Y = yes\n  X = 1.0\n}\n") == []
    assert dual.hsd_diff(a, "A {\n  X = 1.1\n  Y = Yes\n}\nB = 2\n") == ["/a/x: '1.0' != '1.1'"]
    assert dual.hsd_diff(a, "A {\n  X = 1.0\n}\nB = 2\n") == ["/a/y: only in the first"]


# ---- the input writer --------------------------------------------------------------------------
def test_defaults_are_pydftbs(tmp_path):
    text = hsd(plain_water(), tmp_path)
    assert "SCCTolerance = 1e-06" in text and "MaxSCCIterations = 200" in text
    assert "Temperature [K] = 0.0" in text and "PrintForces = No" in text
    assert "Mixer" not in text and "EwaldTolerance" not in text and "Differentiation" not in text
    assert "PrintForces = Yes" in hsd(plain_water(), tmp_path, ("energy", "forces"))


def test_periodic_kpoints_and_ewald(tmp_path):
    text = hsd(copper(), tmp_path, kpts=(3, 3, 3))
    assert "EwaldTolerance = 1e-12" in text
    assert "EwaldTolerance" not in hsd(copper(), tmp_path, kpts=(3, 3, 3), ewald_tolerance=None)
    block = text.split("KPointsAndWeights = {")[1].split("}")[0]
    rows = np.array([[float(x) for x in ln.split()] for ln in block.strip().splitlines()])
    k, w = PyDFTB(SKF_HCOCU, kpts=(3, 3, 3))._kpoints(copper())
    assert np.array_equal(rows[:, :3], k) and np.array_equal(rows[:, 3], w)


def test_periodic_hybrid_uses_dftbplus_foldings(tmp_path):
    c2 = bulk("C", "diamond", a=3.567)
    lc = SKF_SETS["ptbp_hcocu_lc"]
    text = hsd(c2, tmp_path, skdir=lc, hybrid="lc", kpts={"size": (2, 2, 2), "gamma": True})
    assert "SupercellFolding {\n    2 0 0\n    0 2 0\n    0 0 2" in text
    assert "Threshold" not in text                    # matrix screening: no threshold
    gamma = hsd(c2, tmp_path, skdir=lc, hybrid="lc")
    assert "SupercellFolding {\n    1 0 0\n    0 1 0\n    0 0 1" in gamma


@pytest.mark.parametrize("options", [
    {"type_switching": {"element": "Cu", "types": ["Cu", "Cu2"]}},
    {"backend": "torch"}, {"dtype": "float32"}, {"repulsive": "none"}, {"solver": "foe"},
    {"xlbomd": {}}, {"analysis": {"cm5": {"alpha": 1.0}}},
])
def test_pydftb_only_options_are_refused(options, tmp_path):
    with pytest.raises(EngineOptionError):
        hsd(plain_water(), tmp_path, **options)


def test_solvation_model_object_is_refused_spec_taken(tmp_path):
    from pydftb.solvation import SASA
    with pytest.raises(EngineOptionError, match="spec dict"):
        hsd(plain_water(), tmp_path, solvation=SASA(probe_radius=1.0, surface_tension={}))
    text = hsd(plain_water(), tmp_path, solvation={"model": "sasa", "probe_radius": 1.0,
                                                   "surface_tension": {"H": 0.1, "O": 0.2}})
    assert "Solvation = SASA {" in text and "ProbeRadius = 1.0" in text


def test_solvation_spec_errors(tmp_path, monkeypatch):
    """A wrong solvation spec is the same ValueError on both engines; what DFTB+'s input cannot
    say is refused by the DFTB+ engine only; a relative parameter file reaches DFTB+ absolute."""
    from pydftb.solvation import make_solvation
    from pydftb.validation.cases import GB_WATER
    for spec, what in (({"model": "gbsa", "solvent": "water"}, "model must be 'gb'"),
                       ({"model": "gb", "solvent": "water"}, "needs free_energy_shift"),
                       ({**GB_WATER, "born_scael": 1.5}, r"unknown keys \['born_scael'\]"),
                       ({"model": "sasa", "probe_radius": 1.0}, "needs surface_tension"),
                       ({**GB_WATER, "solvent": {"eps": 80.0}}, r"unknown keys \['eps'\]"),
                       ({**GB_WATER, "cm5": {"alfa": 1.0}}, r"unknown keys \['alfa'\]")):
        with pytest.raises(ValueError, match=what):
            make_solvation(spec)
        with pytest.raises(ValueError, match=what):
            hsd(plain_water(), tmp_path, solvation=spec)
    sasa = {"probe_radius": 1.0, "surface_tension": {"H": 0.1, "O": 0.2}}
    for spec, what in (({**GB_WATER, "cm5": {}}, "counts every atom pair"),
                       ({"model": "sasa", **sasa, "radii_table": "bondi"}, "radii_table"),
                       ({**GB_WATER, "sasa": {**sasa, "radii": {"O": 3.0}}}, "radii")):
        make_solvation(spec)                                    # the pydftb engine takes it
        with pytest.raises(EngineOptionError, match=what):
            hsd(plain_water(), tmp_path, solvation=spec)
        with pytest.raises(EngineOptionError, match=r"pydftb=\{\.\.\.\}"):
            make_calculator(SKF_HCOCU, "dftbplus", solvation=spec)
    hsd(plain_water(), tmp_path, solvation={**GB_WATER, "cm5": {"double_count": True}})
    spec = CASES_BY_NAME["ch3oh_gbsa_paramfile"].solvation
    data, skdir = os.path.abspath(DATA), os.path.abspath(SKF_HCOCU)
    monkeypatch.chdir(data)
    text = hsd(plain_water(), tmp_path, skdir=skdir, solvation=spec)
    path = re.search(r'ParamFile = "([^"]+)"', text).group(1)
    assert os.path.isabs(path) and os.path.samefile(path, os.path.join(data, spec["param_file"]))


def test_unknown_option_is_a_type_error(tmp_path):
    with pytest.raises(TypeError, match="unknown options"):
        hsd(plain_water(), tmp_path, scc_tolerance=1e-8)


def test_finite_difference_step(tmp_path):
    assert "Differentiation" not in hsd(plain_water(), tmp_path, fd_delta=2.0 ** -13)
    assert "Delta [Bohr] = 0.0001" in hsd(plain_water(), tmp_path, fd_delta=1e-4)


def test_units_are_pydftbs(tmp_path):
    text = hsd(copper(), tmp_path, kpts=(2, 2, 2), smearing=("fermi", 300.0),
               fixed_fermi_level=-5.0, truncate_sk_range=(6.0, True))
    assert f"FixedFermiLevel [Hartree] = {{ {dp._f(-5.0 / HARTREE)} }}" in text
    assert f"SKMaxDistance [Bohr] = {dp._f(6.0 / BOHR)}" in text
    text = hsd(plain_water(), tmp_path, electric_field=(0.0, 0.0, 0.5))
    assert "Strength [V/AA] = 0.5" in text and "Direction = 0.0 0.0 1.0" in text
    assert text == hsd(plain_water(), tmp_path,
                       electric_field={"strength": 0.5, "direction": [0.0, 0.0, 1.0]})


def test_read_charges_drops_initial_spins(tmp_path):
    o2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.21]])
    opts = {"spin": True, "spin_constants": {"O": -0.028}, "initial_spins": [1.0, 1.0]}
    assert "InitialSpins" in hsd(o2, tmp_path, **opts)
    i = dc.write_input(o2, SKF_HCOCU, opts, cache=str(tmp_path / "skf"),
                       charges=("charges.bin", b"\x00"))
    assert "InitialSpins" not in i.hsd and "ReadInitialCharges = Yes" in i.hsd
    assert i.files == {"charges.bin": b"\x00"}


def test_casida_of_a_spin_polarised_ground_state(tmp_path):
    o2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.21]])
    text = hsd(o2, tmp_path, ("energy", "forces"), spin=True, spin_constants={"O": -0.028},
               unpaired_electrons=2.0, casida={"n_excitations": 3, "state_of_interest": 1})
    assert "NrOfExcitations = 3" in text and "Symmetry" not in text
    assert "ExcitedStateForces = Yes" in text
    singlet = hsd(plain_water(), tmp_path, casida={"n_excitations": 3, "solver": "arpack"})
    assert "Symmetry = singlet" in singlet and "Diagonaliser = Arpack {}" in singlet


# ---- DFTB+'s outputs, read without dftb+ -------------------------------------------------------
def _stored(name, tmp_path):
    with open(os.path.join(REF_DIR, f"{name}.json")) as fh:
        doc = json.load(fh)
    inp = dual.case_inputs(CASES_BY_NAME[name], doc, str(tmp_path))[0][1]
    d = os.path.join(OUTPUTS, name)
    with open(os.path.join(d, "stdout.txt")) as fh:
        stdout = fh.read()
    out = dp.parse_run(d, stdout, True, inp.analysis, inp.make_hsd["autotest"], False, False)
    plain = dc.plain_sk_dir(dual.sk_dir(CASES_BY_NAME[name], str(tmp_path)),
                            str(tmp_path / "plain"))
    q0 = dc.neutral_populations(plain, inp.labels, inp.make_hsd["max_l"], {})
    return doc, inp, out, dc.result_of(out, inp, q0), q0


@pytest.mark.parametrize("name", ["h2o", "cu4co_300K_fixed_ef", "h2o_gbsa", "h2o_ntpoly"])
def test_stored_outputs_as_pydftb_results(name, tmp_path):
    doc, _, _, r, q0 = _stored(name, tmp_path)
    ref = doc["reference"]
    assert abs(r.energy - ref["energy"]) < 5.1e-11           # full precision vs 10 decimals
    assert abs(r.free_energy - ref.get("free_energy", ref["energy"])) < 5.1e-11
    assert np.abs(r.forces - np.array(ref["forces"])).max() < 1e-12
    assert np.abs((q0 - r.charges) - np.array(ref["charges_net"])).max() < 1e-12
    assert r.n_scc == ref["n_scc"]


def test_fixed_fermi_level_tag_is_not_the_mermin_energy(tmp_path):
    _, _, out, r, _ = _stored("cu4co_300K_fixed_ef", tmp_path)
    assert abs(out["mermin_tag"] - out["free_energy"]) > 1.0        # DFTB+'s results.tag
    assert r.free_energy == out["free_energy"]                      # detailed.out's, then
    assert r.e_force_related == pytest.approx(out["e_force_related"], rel=1e-14)


def test_elsi_build_writes_only_the_total_energy(tmp_path):
    _, _, out, r, _ = _stored("h2o_ntpoly", tmp_path)
    assert out["mermin_tag"] is None and out["total_tag"] is not None
    assert r.energy == out["total_tag"] and r.eigenvalues is None


def test_solvent_dipole_is_the_unscaled_one(tmp_path):
    from pydftb.validation.cases import solvation_model
    doc, _, _, r, _ = _stored("h2o_gbsa", tmp_path)
    scale = solvation_model(CASES_BY_NAME["h2o_gbsa"].solvation).field_scaling()
    assert np.allclose(r.dipole * scale, doc["reference"]["dipole"], atol=1e-8)


def test_energies_take_the_tag_only_when_it_is_the_same_quantity():
    base = {"energy": -1.0000000001, "free_energy": -1.0000000002, "e_zero": -1.00000000015}
    full = dict(base, mermin_tag=-1.000000000213, extrapolated0_tag=-1.000000000157)
    tot, mer, ez = dc.energies(full)
    assert (mer, ez) == (full["mermin_tag"], full["extrapolated0_tag"])
    assert tot == 2 * full["extrapolated0_tag"] - full["mermin_tag"]
    tot, mer, ez = dc.energies(dict(base, mermin_tag=-0.3, extrapolated0_tag=-1.000000000157))
    assert (tot, mer) == (base["energy"], base["free_energy"])


# ---- the option layer --------------------------------------------------------------------------
def test_options_table_covers_both_engines():
    import inspect
    names = set(inspect.signature(PyDFTB.__init__).parameters) - {"self", "skdir", "kwargs"}
    assert names <= set(OPTIONS)
    assert all(OPTIONS[k]["dftbplus"] and not OPTIONS[k]["pydftb"] for k in dc.DFTBPLUS_DEFAULTS)
    assert not OPTIONS["type_switching"]["dftbplus"] and OPTIONS["k_threads"]["dftbplus"]


def test_check_options_says_where_an_option_belongs():
    with pytest.raises(EngineOptionError, match=r"dftbplus=\{\.\.\.\}"):
        check_options("pydftb", {"binary": "dftb+"})
    with pytest.raises(EngineOptionError, match=r"pydftb=\{\.\.\.\}"):
        check_options("dftbplus", {"backend": "torch"})
    with pytest.raises(ValueError, match="engine must be"):
        check_options("xtb", {})
    check_options("dftb+", {"scc_tol": 1e-8, "binary": "x"})


def test_make_calculator_and_sections():
    py = make_calculator(SKF_HCOCU, "pydftb", scc_tol=1e-8, pydftb={"k_threads": 2},
                         dftbplus={"binary": "nowhere"})
    assert isinstance(py, PyDFTB) and py.engine_opts["scc_tol"] == 1e-8
    d = make_calculator(SKF_HCOCU, "dftbplus", scc_tol=1e-8, pydftb={"k_threads": 2},
                        dftbplus={"binary": "nowhere"})
    assert isinstance(d, dc.DFTBPlus) and d.options == {"scc_tol": 1e-8, "binary": "nowhere"}
    with pytest.raises(FileNotFoundError, match="DFTB\\+ not found"):
        d.get_potential_energy(plain_water())


def test_common_results_of_the_pydftb_engine():
    calc = make_calculator(SKF_HCOCU, "pydftb", kpts=(2, 2, 2))
    res = common_results(calc, copper(), properties=("energy", "forces", "stress"))
    assert res["engine"] == "pydftb" and len(res["stress"]) == 6
    assert np.shape(res["eigenvalues"]) == (1, len(res["kweights"]), 9)
    assert compare_results(res, res)["energy"] == 0.0


# ---- the engine-neutral forms on PyDFTB (bit for bit as the old ones) ---------------------
def _energy(**opts):
    a = plain_water()
    a.calc = PyDFTB(SKF_HCOCU, scc_tol=1e-10, **opts)
    return a.get_potential_energy(), a.get_forces()


def test_field_as_strength_and_direction():
    d = np.array([1.0, 2.0, 2.0])
    e1, f1 = _energy(electric_field={"strength": 0.3, "direction": d})
    e2, f2 = _energy(electric_field=0.3 * d / np.linalg.norm(d))
    assert e1 == e2 and np.array_equal(f1, f2)


def test_solvation_spec_dict():
    from pydftb.validation.cases import solvation_model
    spec = CASES_BY_NAME["h2o_gbsa"].solvation
    full = dict(spec, param_file=os.path.join(DATA, spec["param_file"])) \
        if "param_file" in spec else spec
    e1, f1 = _energy(solvation=full)
    e2, f2 = _energy(solvation=solvation_model(spec))
    assert e1 == e2 and np.array_equal(f1, f2)


def test_ntpoly_solver_dict():
    from pydftb.purification import NTPoly
    e1, _ = _energy(solver={"name": "ntpoly", "method": "trs2", "tolerance": 1e-9})
    e2, _ = _energy(solver=NTPoly(method="trs2", tolerance=1e-9))
    assert e1 == e2


def _old_kpoints(atoms, spec, time_reversal, two_component):
    """PyDFTB._kpoints before P2 (grids and explicit points)."""
    from ase.calculators.calculator import kptdensity2monkhorstpack
    if isinstance(spec, dict) and ("size" in spec or "density" in spec):
        grid = spec.get("size") or kptdensity2monkhorstpack(atoms, spec["density"],
                                                            even=spec.get("even"))
        k, w = monkhorst_pack(tuple(int(n) for n in grid),
                              gamma_centered=bool(spec.get("gamma", False)))
    elif isinstance(spec, dict):
        k, w = np.atleast_2d(spec["kpts"]), np.asarray(spec.get("weights",
                                                                np.ones(len(spec["kpts"]))))
    else:
        grid = kptdensity2monkhorstpack(atoms, spec, even=False) if np.isscalar(spec) else spec
        k, w = monkhorst_pack(tuple(int(n) for n in grid))
    reduce = not two_component if time_reversal is None else time_reversal
    if reduce:
        k, w = reduce_time_reversal(k, w)
    return k, w / w.sum()


@pytest.mark.parametrize("spec", [(3, 3, 3), (2, 3, 4), 2.5, {"size": (4, 4, 4), "gamma": True},
                                  {"density": 3.0}, {"kpts": [[0, 0, 0], [0.5, 0, 0]],
                                                     "weights": [1, 3]}])
@pytest.mark.parametrize("tr,tc", [(None, False), (None, True), (False, False)])
def test_kpoints_of_is_the_old_rule(spec, tr, tc):
    a = copper()
    k1, w1 = kpoints_of(a, spec, tr, tc)
    k2, w2 = _old_kpoints(a, spec, tr, tc)
    assert np.array_equal(k1, k2) and np.array_equal(w1, w2)


def test_klines_kpoints():
    from pydftb.bands import klines
    lines = [(1, (0, 0, 0)), (4, (0.5, 0, 0.5))]
    k, w = kpoints_of(copper(), {"klines": lines})
    assert np.array_equal(k, klines(lines)[0]) and np.allclose(w, 1.0 / len(k))


def test_states_regions_and_second_derivatives_of_pydftb():
    from pydftb.vibrations import second_derivatives
    a = plain_water()
    calc = PyDFTB(SKF_HCOCU, scc_tol=1e-10)
    a.calc = calc
    regions = [{"label": "O", "atoms": [0], "shell_resolved": True}, {"label": "H",
                                                                      "atoms": [1, 2]}]
    st = calc.states(regions=regions, atoms=a)
    assert set(st.region_weights) == {"O.1.out", "O.2.out", "H.out"}
    assert np.array_equal(st.region_weights["H.out"], st.weights([1, 2]))
    sd = calc.second_derivatives(a)
    engine, labels, _ = calc._engine(a)
    ref = second_derivatives(engine, labels, a.positions / BOHR)
    assert np.array_equal(sd.hessian, ref.hessian)
    assert np.allclose(calc.masses(atoms=a), [15.999, 1.008, 1.008], atol=1e-2)


# ---- pydftb run --------------------------------------------------------------------------------
def _run(args, tmp_path):
    from pydftb import run
    out = tmp_path / "res.json"
    assert run.main([*args, "--json", str(out)]) == 0
    with open(out) as fh:
        return json.load(fh)


def test_run_energy_relax_and_options_file(tmp_path):
    from ase.io import write
    write(tmp_path / "h2o.xyz", plain_water())
    opts = tmp_path / "opts.json"
    opts.write_text(json.dumps({"scc_tol": 1e-9, "pydftb": {"k_threads": 1},
                                "dftbplus": {"binary": "nowhere"}}))
    res = _run([str(tmp_path / "h2o.xyz"), "--skdir", SKF_HCOCU, "--options", str(opts)],
               tmp_path)
    assert res["engine"] == "pydftb" and res["options"]["scc_tol"] == 1e-9
    assert res["results"]["energy"] < -100.0 and len(res["results"]["forces"]) == 3
    res = _run([str(tmp_path / "h2o.xyz"), "--skdir", SKF_HCOCU, "--task", "relax",
                "--fmax", "1e-3", "--set", "scc_tol=1e-9", "--output",
                str(tmp_path / "relaxed.xyz")], tmp_path)
    assert res["converged"] and np.abs(res["results"]["forces"]).max() < 1e-3
    assert os.path.exists(tmp_path / "relaxed.xyz")


def test_run_tasks_on_pydftb(tmp_path):
    from ase.io import write
    write(tmp_path / "cu.cif", copper())
    base = [str(tmp_path / "cu.cif"), "--skdir", SKF_HCOCU, "--kpts", "3", "3", "3",
            "--temperature", "1000"]
    bands = _run(base + ["--task", "bands", "--path", "GXL", "--npoints", "12"], tmp_path)
    assert np.shape(bands["energies"]) == (1, 12, 9)
    dos = _run(base + ["--task", "dos"], tmp_path)
    assert len(dos["energies"]) == np.shape(dos["dos"])[1]
    write(tmp_path / "h2o.xyz", plain_water())
    md = _run([str(tmp_path / "h2o.xyz"), "--skdir", SKF_HCOCU, "--task", "md", "--steps", "5"],
              tmp_path)
    assert len(md["potential"]) == 6 and md["drift"] < 1e-2
    hess = _run([str(tmp_path / "h2o.xyz"), "--skdir", SKF_HCOCU, "--task", "hessian",
                 "--project"], tmp_path)
    assert hess["options"]["scc_tol"] == 1e-10 and len(hess["wavenumbers_cm"]) == 9


def test_run_refuses_a_pydftb_option_on_dftbplus(tmp_path, capsys):
    from ase.io import write

    from pydftb import run
    write(tmp_path / "h2o.xyz", plain_water())
    rc = run.main([str(tmp_path / "h2o.xyz"), "--skdir", SKF_HCOCU, "--engine", "dftbplus",
                   "--set", "backend=torch"])
    assert rc == 2 and "pydftb={...}" in capsys.readouterr().err


# ---- with DFTB+ ---------------------------------------------------------------------------------
@pytest.mark.dftbplus
def test_engines_agree_on_water_and_copper():
    for atoms, opts, props in ((plain_water(), {}, ("energy", "forces")),
                               (copper(), {"kpts": (4, 4, 4), "smearing": ("fermi", 1000.0)},
                                ("energy", "forces", "stress"))):
        res = {e: common_results(make_calculator(SKF_HCOCU, e, scc_tol=1e-10, fd_delta=2 ** -13,
                                                 **opts), atoms.copy(), properties=props)
               for e in ("pydftb", "dftbplus")}
        d = compare_results(res["pydftb"], res["dftbplus"])
        assert d["energy"] < 1e-9 and d["forces"] < 1e-8 and d["charges"] < 1e-10
        assert d.get("stress", 0.0) < 1e-10 and d["n_scc"] == 0


@pytest.mark.dftbplus
def test_reused_charges_start_from_the_last_run():
    a = plain_water()
    calc = dc.DFTBPlus(SKF_HCOCU, scc_tol=1e-10, reuse_charges=True)
    a.calc = calc
    a.get_potential_energy()
    first = calc.raw.n_scc
    for _ in range(3):              # every step from the one before (DFTB+ rewrites the file)
        a.positions[0, 2] += 0.01
        a.get_potential_energy()
        assert calc.raw.n_scc < first and "ReadInitialCharges = Yes" in calc.input.hsd
        assert calc._charges is not None


@pytest.mark.dftbplus
@pytest.mark.filterwarnings("ignore:SCC not converged")
def test_capped_scc_with_reused_charges_follows_dftbplus():
    """Three SCC iterations per MD step, unconverged (ConvergentSCCOnly = No), charges reused:
    on both engines every step starts from the input of the step before's last Hamiltonian
    (DFTB+ writes it to charges.bin, it does not mix after the last iteration), so the
    trajectories agree step by step.  (With the next mixed vector instead, the forces were up
    to 6e-3 eV/A apart from the second step on.)"""
    from ase import units
    from ase.md.verlet import VelocityVerlet
    steps = {}
    for e in ("pydftb", "dftbplus"):
        a = plain_water()
        a.rattle(0.05, seed=3)
        a.set_velocities(np.zeros((3, 3)))
        a.calc = make_calculator(SKF_HCOCU, e, scc_tol=1e-10, max_scc=3, reuse_charges=True,
                                 fd_delta=2 ** -13, dftbplus={"convergent_scc_only": False})
        rec = steps[e] = []
        md = VelocityVerlet(a, timestep=0.5 * units.fs, logfile=None)
        md.attach(lambda a=a, rec=rec: rec.append((a.get_potential_energy(),
                                                   a.get_forces().copy(), a.calc.raw.n_scc)))
        md.run(steps=4)
    assert len(steps["pydftb"]) == len(steps["dftbplus"]) == 5
    for (e1, f1, n1), (e2, f2, n2) in zip(steps["pydftb"], steps["dftbplus"]):
        assert n1 == n2 == 3
        assert abs(e1 - e2) < 1e-9 and np.abs(f1 - f2).max() < 1e-8


@pytest.mark.dftbplus
def test_states_bands_and_hessian_of_dftbplus():
    cu = copper()
    calcs = {e: make_calculator(SKF_HCOCU, e, kpts=(3, 3, 3), smearing=("fermi", 1000.0),
                                scc_tol=1e-10) for e in ("pydftb", "dftbplus")}
    bs = {}
    for e, c in calcs.items():
        cu.calc = c
        bs[e] = c.band_structure(path="GXL", npoints=10, atoms=cu).energies
    assert np.abs(bs["pydftb"] - bs["dftbplus"]).max() < 1e-7
    regions = [{"label": "Cu", "atoms": [0], "shell_resolved": True}]
    w = {e: c.states(regions=regions, atoms=cu).region_weights for e, c in calcs.items()}
    assert np.abs(w["pydftb"]["Cu.3.out"] - w["dftbplus"]["Cu.3.out"]).max() < 1e-6
    a = plain_water()
    sd = {e: make_calculator(SKF_HCOCU, e, scc_tol=1e-10, fd_delta=2 ** -13)
          .second_derivatives(a) for e in ("pydftb", "dftbplus")}
    assert np.abs(sd["pydftb"].hessian - sd["dftbplus"].hessian).max() < 1e-6


@pytest.mark.dftbplus
def test_native_drivers_and_failures(tmp_path):
    a = plain_water()
    calc = dc.DFTBPlus(SKF_HCOCU, scc_tol=1e-10)
    new, out = calc.relax(a, fmax=1e-3)
    assert out["converged_geometry"] and out["geometry_steps"] >= 1
    a.set_velocities(np.zeros((3, 3)))
    md = calc.md(new, steps=3, time_step=0.5)
    assert len(md["md"]["potential"]) == 4
    bad = dc.DFTBPlus(SKF_HCOCU, max_scc=1, scc_tol=1e-12)
    from ase.calculators.calculator import CalculationFailed
    with pytest.raises(CalculationFailed, match="SCC is NOT converged"):
        bad.get_potential_energy(plain_water())
    calc.get_potential_energy(plain_water())
    calc.write_charges(tmp_path / "charges.bin")
    again = dc.DFTBPlus(SKF_HCOCU, scc_tol=1e-10, read_charges=str(tmp_path / "charges.bin"))
    again.get_potential_energy(plain_water())
    assert again.raw.n_scc <= 2 and "ReadInitialCharges = Yes" in again.input.hsd


@pytest.mark.dftbplus
def test_stopped_scc_writes_the_last_input(tmp_path):
    """An SCC stopped at MaxSCCIterations: DFTB+ writes the input of its last iteration to
    charges.bin, pydftb's q_in (M6f-1b; DFTB+ writes no file when MaxSCCIterations = 1)."""
    for n in (2, 3):
        d = dc.DFTBPlus(SKF_HCOCU, max_scc=n, scc_tol=1e-12, convergent_scc_only=False)
        d.get_potential_energy(plain_water())
        d.write_charges(tmp_path / f"charges_{n}.bin")
        p = PyDFTB(SKF_HCOCU, max_scc=n, scc_tol=1e-12, warn_unconverged=False)
        a = plain_water()
        p.get_potential_energy(a)
        engine, labels, _ = p._engine(a)
        data = io.read_charges(tmp_path / f"charges_{n}.bin",
                               engine._charge_layout(labels)[0].norb_atom)
        assert p.raw.n_scc == n and not p.raw.converged
        assert np.abs(engine.populations_from_file(labels, data) - p.raw.q_in).max() < 1e-12


@pytest.mark.dftbplus
def test_dual_live_and_drivers_on_a_few_cases():
    recs = dual.check_live(["h2o", "cu_fcc_k6", "o2_triplet_wmatrix", "h2co_exc_forces"],
                           verbose=False)
    for r in recs:
        assert "error" not in r, r.get("error")
        assert max(r["dev"].values()) == 0.0
        assert all(v <= t for v, t in r["cross"].values()), r["cross"]
    recs = dual.check_drivers(["h2o_opt", "water_dimer_xl5_md"], verbose=False)
    assert all(r["ok"] for r in recs)


@pytest.mark.dftbplus
def test_pydftb_run_on_dftbplus(tmp_path):
    from ase.io import write
    write(tmp_path / "h2o.xyz", plain_water())
    res = {e: _run([str(tmp_path / "h2o.xyz"), "--skdir", SKF_HCOCU, "--engine", e,
                    "--scc-tol", "1e-10", "--set", f"fd_delta={2 ** -13}"], tmp_path)
           for e in ("pydftb", "dftbplus")}
    assert res["dftbplus"]["dftbplus_runs"] == 1
    assert abs(res["pydftb"]["results"]["energy"] - res["dftbplus"]["results"]["energy"]) < 1e-9
    out = subprocess.run([sys.executable, "-m", "pydftb", "run", "--help"], capture_output=True,
                         text=True, check=False)
    assert out.returncode == 0 and "--engine" in out.stdout
