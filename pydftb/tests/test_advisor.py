"""DFTB+ configuration advisor (pydftb.advisor, M3d): HSD reader, screen output, rules, CLI."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pydftb.advisor import advise, hsd
from pydftb.advisor.__main__ import main as cli
from pydftb.advisor.inputs import parse_stdout
from pydftb.advisor.rules import PTBP_LEGACY, RULES

DATA = Path(__file__).resolve().parent / "data" / "advisor"


def rules_of(findings):
    return sorted((f.rule, f.severity) for f in findings)


# ---------------------------------------------------------------------------------------
# HSD reader

def test_hsd_syntax():
    text = """# comment
Geometry = GenFormat {
  2 C
  Cu O   # species
  1 1 0.0 0.0 0.0
  2 2 0.0 0.0 1.8
}
Hamiltonian = DFTB {
  SCC = Yes ; SCCTolerance = 1e-6
  Filling = Fermi {
    Temperature [K] = 300
  }
  MaxAngularMomentum = {
    Cu = "d"
    O = 'p'
  }
  PolynomialRepulsive = SetForAll {Yes}
  KPointsAndWeights = SupercellFolding {
    2 0 0
    0 2 0
    0 0 2
    0.5 0.5 0.5
  }
  Mixer = Broyden {}
}
Options { WriteResultsTag = Yes }
"""
    root = hsd.parse(text, "dftb_in.hsd")
    assert [c.name for c in root.children] == ["Geometry", "Hamiltonian", "Options"]
    dftb = root.find("hamiltonian").method
    assert dftb.name == "DFTB" and dftb.is_method
    assert root.flag("Hamiltonian/DFTB/SCC") is True
    assert root.number("Hamiltonian/DFTB/SCCTolerance") == 1e-6
    t = root.find("Hamiltonian/DFTB/Filling/Fermi/Temperature")
    assert (t.modifier, t.value, t.where) == ("K", "300", "dftb_in.hsd:11")
    assert root.text("Hamiltonian/DFTB/MaxAngularMomentum/O") == "p"
    assert root.flag("Hamiltonian/DFTB/PolynomialRepulsive") is True
    kp = root.find("Hamiltonian/DFTB/KPointsAndWeights")
    assert kp.method.name == "SupercellFolding" and len(kp.method.data) == 4
    assert root.find("Hamiltonian/DFTB/Mixer").method.name == "Broyden"
    assert root.flag("Options/WriteResultsTag") is True
    geo = root.find("Geometry").method
    assert geo.data[1] == (4, ["Cu", "O"])                  # comment removed, line kept
    assert root.find("Hamiltonian").where == "dftb_in.hsd:8-25"
    assert root.find("Hamiltonian/DFTB/Filling").path == "Hamiltonian/DFTB/Filling"


def test_hsd_includes(tmp_path):
    (tmp_path / "geo.gen").write_text("1 C\nH\n1 1 0 0 0\n")
    (tmp_path / "ham.hsd").write_text("SCC = Yes\nMaxSCCIterations = 7\n")
    (tmp_path / "dftb_in.hsd").write_text(
        'Geometry = GenFormat {\n  <<< "geo.gen"\n}\nHamiltonian = DFTB {\n  <<+ "ham.hsd"\n'
        '  <<+ "missing.hsd"\n}\n')
    root = hsd.read(tmp_path / "dftb_in.hsd")
    geo = root.find("Geometry").method
    assert geo.raw_file.endswith("geo.gen") and [t for _, t in geo.data][1] == ["H"]
    dftb = root.find("Hamiltonian/DFTB")
    assert dftb.number("MaxSCCIterations") == 7 and dftb.missing_includes == ["missing.hsd"]
    assert dftb.find("SCC").file.endswith("ham.hsd")


def test_hsd_errors_and_dftbplus_value_rule():
    with pytest.raises(hsd.HSDError):
        hsd.parse("A {\n  B = 1\n")
    with pytest.raises(hsd.HSDError):
        hsd.parse("A = 1\n}\n")
    # two assignments on a line: the rest of the line is the value (DFTB+: "Superfluous data")
    assert hsd.parse("Order = 2  Temperature [K] = 1000\n").text("Order") == "2 Temperature [K] = 1000"


def test_hsd_reads_what_ase_writes(tmp_path):
    from ase.build import bulk
    from ase.calculators.dftb import Dftb
    atoms = bulk("Cu", "fcc", a=3.61)
    calc = Dftb(atoms=atoms, label="Cu", directory=str(tmp_path), kpts=(4, 4, 4),
                Hamiltonian_SCC="Yes", Hamiltonian_MaxAngularMomentum_="",
                Hamiltonian_MaxAngularMomentum_Cu='"d"',
                Hamiltonian_Filling="Fermi {Temperature [Kelvin] = 300}",
                Hamiltonian_PolynomialRepulsive="SetForAll {Yes}")
    calc.write_input(atoms)
    root = hsd.read(tmp_path / "dftb_in.hsd")
    assert root.number("ParserOptions/ParserVersion") == 1
    assert root.flag("ParserOptions/IgnoreUnprocessedNodes") is True
    assert root.find("Hamiltonian/DFTB/Filling").method.name == "Fermi"
    assert root.find("Hamiltonian/DFTB/KPointsAndWeights").method.name == "SupercellFolding"
    assert root.text("Hamiltonian/DFTB/MaxAngularMomentum/Cu") == "d"


def test_dftbplus_pin_and_screen_output():
    pin = hsd.read(DATA / "ptbp_legacy" / "dftb_pin.hsd")
    dftb = pin.find("Hamiltonian/DFTB")
    assert dftb.flag("ShellResolvedSCC") is True
    assert dftb.number("Differentiation/FiniteDiff/Delta") == 0.01
    assert dftb.find("Filling").method.number("Temperature") == 0.0
    out = parse_stdout((DATA / "ptbp_legacy" / "dftb.out").read_text())
    assert (out.converted_from, out.parser_version, out.omp_threads) == (1, 14, 1)
    assert [(m.path, m.lines) for m in out.ignored] == [
        ("Hamiltonian/DFTB/Fermi", "10-12"), ("Hamiltonian/DFTB/MethfesselPaxton", "24-26")]
    assert any("legacy step size" in m.text for m in out.warnings)
    err = parse_stdout((DATA / "no_spline" / "dftb.out").read_text())
    assert [m.text for m in err.errors] == ["No spline repulsive found in file '././Gd-Gd.skf'"]
    assert parse_stdout((DATA / "scc_unconverged" / "dftb.out").read_text()).scc_not_converged
    mpi = parse_stdout("MPI processes:               4 (split into 2 groups)\nOpenMP threads:  2\n"
                       "Current stacksize:           unlimited\n")
    assert (mpi.mpi_procs, mpi.mpi_groups, mpi.omp_threads, mpi.stack_unlimited) == (4, 2, 2, True)


# ---------------------------------------------------------------------------------------
# rules on frozen DFTB+ 25.1 runs (pydftb/validation/advisor_cases.py)

EXPECTED = {
    "clean": [("T04", "info")],
    "ptbp_legacy": [("E01", "info"), ("P01", "warning"), ("P03", "warning"), ("P03", "warning"),
                    ("T04", "info")],
    "scc_unconverged": [("E01", "info"), ("S01", "warning"), ("T04", "info")],
    "no_spline": [("E01", "info"), ("K01", "error"), ("X01", "error")],
    "klines": [("B01", "error"), ("E01", "info"), ("T04", "info"), ("X01", "error")],
    "mp_order": [("E01", "info"), ("P02", "warning"), ("T04", "info")],
    "temp_no_unit": [("E01", "info"), ("F01", "error"), ("T04", "info")],
    "groups": [("E01", "info"), ("T02", "warning"), ("T04", "info")],
    "omp_small": [("E01", "info"), ("T03", "warning"), ("T04", "info")],
    "d4_periodic": [("D02", "warning"), ("D03", "warning"), ("E01", "info"), ("T04", "info")],
    "lattice_iso": [("E01", "info"), ("G01", "warning"), ("G02", "info"), ("T04", "info")],
}


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_frozen_runs(case):
    meta = json.loads((DATA / case / "case.json").read_text())
    run, findings = advise(DATA / case, environ={}, cores=4)
    assert rules_of(findings) == sorted(EXPECTED[case])
    # a run that DFTB+ stopped is reported as an error, and the other way round
    assert (meta["returncode"] != 0) == any(f.rule == "X01" for f in findings)


def test_ptbp_legacy_details_and_context():
    _, findings = advise(DATA / "ptbp_legacy", environ={}, cores=4)
    p01 = next(f for f in findings if f.rule == "P01")
    assert "ShellResolvedSCC" in p01.message and "0.01" in p01.message and p01.where == "dftb_in.hsd:46"
    p03 = [f for f in findings if f.rule == "P03"]
    assert [f.where for f in p03] == ["dftb_in.hsd:10-12", "dftb_in.hsd:24-26"]
    assert all("T = 0 Ha" in f.message for f in p03)
    assert not any(f.expected for f in findings)
    _, ptbp = advise(DATA / "ptbp_legacy", environ={}, cores=4, context=PTBP_LEGACY)
    assert {f.rule for f in ptbp if f.expected} == {"P01", "P03"}


def _write_run(tmp_path, body, skfs=None, geometry=None, stdout=None, parser=14):
    geometry = geometry or "1 C\nC\n1 1 0.0 0.0 0.0\n"
    (tmp_path / "geo.gen").write_text(geometry)
    (tmp_path / "dftb_in.hsd").write_text(
        f'Geometry = GenFormat {{\n  <<< "geo.gen"\n}}\n{body}\n'
        f'ParserOptions {{\n  ParserVersion = {parser}\n}}\n')
    for name, text in (skfs or {}).items():
        (tmp_path / name).write_text(text)
    if stdout is not None:
        (tmp_path / "dftb.out").write_text("|  DFTB+ release 25.1\n" + stdout)
    return tmp_path


def _ham(lines, max_l='C = "p"'):
    return (f'Hamiltonian = DFTB {{\n  SCC = Yes\n  MaxAngularMomentum {{ {max_l} }}\n'
            f'  SlaterKosterFiles = Type2FileNames {{\n    Prefix = ./\n    Separator = "-"\n'
            f'    Suffix = ".skf"\n  }}\n{lines}\n}}')


def _skf(us, up, spline=True):
    return (f"0.02, 10\n0.0 -0.19 -0.50 0.0 0.0 {up} {us} 0.0 2.0 2.0\n"
            + ("Spline\n1 1.0\n1.0 0.0 0.0\n0.0 1.0 0.0 0.0 0.0 0.0\n" if spline else ""))


@pytest.mark.parametrize("dU, extra, expect", [
    (1e-6, "", None), (1e-5, "", "error"), (2e-4, "", "warning"), (2e-3, "", None),
    (5e-4, "  ThirdOrderFull = Yes", "error")])
def test_h01_nearly_equal_hubbard(tmp_path, dU, extra, expect):
    """Shell-resolved: the s and p shells of carbon interact with each other."""
    run_dir = _write_run(tmp_path, _ham("  ShellResolvedSCC = Yes\n" + extra),
                         {"C-C.skf": _skf(0.4457, 0.4457 + dU)})
    _, findings = advise(run_dir, environ={}, cores=1, only=["H01"])
    assert [f.severity for f in findings] == ([expect] if expect else [])


def test_h01_uses_the_s_shell_without_shell_resolution(tmp_path):
    run_dir = _write_run(tmp_path, _ham(""), {"C-C.skf": _skf(0.4457, 0.4457 + 1e-5)})
    assert advise(run_dir, environ={}, only=["H01"])[1] == []          # one U per atom
    # ParserVersion < 3 with SCC = Yes: DFTB+ switches shell resolution on (P01)
    _write_run(tmp_path, _ham(""), parser=1)
    assert [f.rule for f in advise(run_dir, environ={}, only=["H01"])[1]] == ["H01"]


def test_k01_and_polynomial_repulsive(tmp_path):
    run_dir = _write_run(tmp_path, _ham(""), {"C-C.skf": _skf(0.44, 0.35, spline=False)})
    assert [f.rule for f in advise(run_dir, environ={}, only=["K01"])[1]] == ["K01"]
    _write_run(tmp_path, _ham("  PolynomialRepulsive = SetForAll { Yes }"))
    assert advise(run_dir, environ={}, only=["K01"])[1] == []
    _write_run(tmp_path, _ham("  PolynomialRepulsive {\n    C-C = Yes\n  }"))
    assert advise(run_dir, environ={}, only=["K01"])[1] == []


def test_threads_and_processes(tmp_path):
    run_dir = _write_run(tmp_path, _ham(""), {"C-C.skf": _skf(0.44, 0.35)},
                         geometry="40 C\nC\n" + "".join(f"{i + 1} 1 {i * 1.5} 0 0\n" for i in range(40)))
    t01 = lambda **kw: [f.rule for f in advise(run_dir, only=["T01"], **kw)[1]]
    assert t01(environ={}, cores=4, concurrent=4) == ["T01"]           # all cores x 4 runs
    assert t01(environ={"OMP_NUM_THREADS": "1"}, cores=4, concurrent=4) == []
    assert t01(environ={"OMP_NUM_THREADS": "4"}, cores=4, mpi_procs=2) == ["T01"]
    t03 = lambda **kw: [f.rule for f in advise(run_dir, only=["T03"], **kw)[1]]
    assert t03(environ={"OMP_NUM_THREADS": "4"}, cores=4) == ["T03"]   # 160 orbitals
    assert t03(environ={"OMP_NUM_THREADS": "1"}, cores=4) == []
    # MPI build: Groups must divide the processes
    _write_run(tmp_path, _ham("") + "\nParallel {\n  Groups = 2\n}", stdout="MPI processes: 3\n")
    t02 = [(f.rule, f.severity) for f in advise(run_dir, environ={}, only=["T02"])[1]]
    assert t02 == [("T02", "error")]


def test_rules_without_screen_output(tmp_path):
    # smearing outside Filling, without IgnoreUnprocessedNodes: DFTB+ stops
    run_dir = _write_run(tmp_path, _ham("  Fermi {\n    Temperature = 0.001\n  }"),
                         {"C-C.skf": _skf(0.44, 0.35)})
    p03 = advise(run_dir, environ={}, only=["P03"])[1]
    assert [(f.rule, f.severity) for f in p03] == [("P03", "error")]
    _write_run(tmp_path, _ham("  Filling = Fermi {\n    Temperature = 0.001\n  }"))
    f01 = advise(run_dir, environ={}, only=["F01"])[1]
    assert [(f.rule, f.severity) for f in f01] == [("F01", "info")] and "316 K" in f01[0].message
    # orthogonal cell: no G01; D4 with ChargeScale = 0 and s9 = 0: no D02/D03
    cubic = "1 S\nC\n1 1 0 0 0\n0 0 0\n3 0 0\n0 3 0\n0 0 3\n"
    _write_run(tmp_path, _ham("  Dispersion = DftD4 {\n    s9 = 0\n    ChargeScale = 0\n  }")
               + "\nDriver = GeometryOptimisation {\n  LatticeOpt = Yes\n  Isotropic = Yes\n}",
               geometry=cubic)
    assert advise(run_dir, environ={}, only=["G01", "D02", "D03"])[1] == []


def test_a_failing_rule_does_not_stop_the_others(monkeypatch):
    import pydftb.advisor.rules as R

    def broken(run):
        raise KeyError("x")

    monkeypatch.setitem(R.RULES, "P01", R.Rule("P01", "warning", "t", "s", "e", broken))
    _, findings = advise(DATA / "ptbp_legacy", environ={}, cores=4)
    p01 = [f for f in findings if f.rule == "P01"]
    assert len(p01) == 1 and p01[0].severity == "info" and "KeyError" in p01[0].message
    assert any(f.rule == "P03" for f in findings)


# ---------------------------------------------------------------------------------------
# command line

def test_cli_exit_codes_and_json(tmp_path, capsys):
    assert cli([str(DATA / "clean")]) == 0
    assert cli([str(DATA / "temp_no_unit")]) == 1                     # an error
    assert cli([str(DATA / "ptbp_legacy")]) == 0                      # warnings only
    assert cli([str(DATA / "ptbp_legacy"), "--fail-on", "warning"]) == 1
    assert cli([str(DATA / "temp_no_unit"), "--fail-on", "never"]) == 0
    assert cli([str(tmp_path)]) == 2                                   # no dftb_in.hsd
    (tmp_path / "dftb_in.hsd").write_text("Hamiltonian = DFTB {\n")
    assert cli([str(tmp_path)]) == 3                                   # unreadable
    capsys.readouterr()
    assert cli([str(DATA / "ptbp_legacy"), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["summary"] == {"error": 0, "warning": 3, "info": 2}
    assert {f["rule"] for f in data["findings"]} == {"P01", "P03", "E01", "T04"}
    assert cli(["--list-rules"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == len(RULES) == 18
    assert cli(["--explain", "p03"]) == 0 and "IgnoreUnprocessedNodes" in capsys.readouterr().out
    assert cli(["--explain", "Z99"]) == 2


def test_module_entry_point():
    out = subprocess.run([sys.executable, "-m", "pydftb.advisor", str(DATA / "klines")],
                         capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
    assert out.returncode == 1
    assert "[B01] error" in out.stdout and "Klines" in out.stdout
