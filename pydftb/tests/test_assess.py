"""Assessment against frozen DFT sets (A series): energies by role and family, offsets, EOS,
dimers, reactions with the balance check, tiers and the Markdown report."""
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator

from pydftb.assess import (energy_metrics, eos_metrics, records_from_arrays, summarize,
                           write_markdown)

E0 = {"Li": -0.1, "O": -0.5}


def _frame(atoms, e, name, family, role="train", f=None):
    a = atoms.copy()
    a.calc = SinglePointCalculator(a, energy=e, forces=np.zeros((len(a), 3)) if f is None else f)
    a.info.update(name=name, family=family, role=role)
    return a


def _eos(name, sym, a0, e_min, b=0.5, n=9):
    out = []
    base = bulk(sym, "bcc", a=a0)
    v0 = base.get_volume()
    for k, s in enumerate(np.linspace(0.9, 1.1, n)):
        at = base.copy()
        at.set_cell(base.cell * s ** (1 / 3), scale_atoms=True)
        v = at.get_volume()
        e = e_min + 0.5 * b * (v - v0) ** 2 / v0
        out.append(_frame(at, e, f"{name}_{k:02d}", name))
    return out


def test_energy_metrics_relative_and_offsets():
    fr = [_frame(Atoms("Li2", positions=[[0, 0, 0], [0, 0, 2.6]]), -2.0, "a", "g1"),
          _frame(Atoms("Li2", positions=[[0, 0, 0], [0, 0, 2.8]]), -1.8, "b", "g1"),
          _frame(Atoms("LiO", positions=[[0, 0, 0], [0, 0, 1.7]]), -4.0, "c", "g2", "test")]
    # predictions: g1 shifted by +0.1 eV/atom (relative energies exact), g2 by +0.3
    pred = [-2.0 + 0.2, -1.8 + 0.2, -4.0 + 0.6]
    rec = records_from_arrays(fr, pred, [np.zeros((2, 3))] * 3)
    m = energy_metrics(rec, E0, E0, offsets="per_element")
    assert m["by_role"]["train"]["atomization_mae"] == pytest.approx(0.1)
    assert m["by_role"]["train"]["relative_mae"] == pytest.approx(0.0, abs=1e-12)
    assert m["offsets"]["Li"] == pytest.approx(0.1)
    assert m["by_role"]["test"]["offset_mae"] == pytest.approx(0.3 - 0.05, abs=1e-12)  # (0.1 + mu_O)/2
    assert m["by_role"]["train"]["force_mae"] == 0.0


def test_eos_dimers_reactions_tier_and_report(tmp_path):
    fr = _eos("eos_li_bcc", "Li", 3.44, -1.9) + _eos("eos_x_bcc", "Li", 3.44, -1.7)
    for k, r in enumerate([1.2, 1.6, 2.0, 2.7, 3.5]):
        e = -1.0 + (r - 2.7) ** 2 * 0.8 - 0.0 * k
        fr.append(_frame(Atoms("Li2", positions=[[0, 0, 0], [0, 0, r]], cell=[12] * 3),
                         e, f"dimer_Li2_{r:.2f}", "dimer_Li2", "test"))
    pred = [a.get_potential_energy() + 0.002 * len(a) for a in fr]   # 2 meV/atom shift
    rec = records_from_arrays(fr, pred, [np.zeros((len(a), 3)) for a in fr])
    eos = eos_metrics(rec)
    assert eos["eos_li_bcc"]["v0_rel_err"] == pytest.approx(0.0, abs=1e-6)
    rx = [{"name": "x -> li", "reactants": {"family:eos_x_bcc": 1}, "products": {"family:eos_li_bcc": 1},
           "decisive": True},
          {"name": "unbalanced", "reactants": {"family:eos_x_bcc": 1}, "products": {}}]
    s = summarize(rec, {"Li": 0.0}, {"Li": 0.0}, reactions=rx)
    good, bad = s["reactions"]
    assert good["de_ref"] == pytest.approx(-0.2, abs=1e-6) and good["sign_ok"]
    assert "not balanced" in bad["error_text"]
    assert s["dimers"]["dimer_Li2"]["r_min_ref"] == pytest.approx(2.7)
    assert s["tier"] == "A"
    path = write_markdown(s, str(tmp_path / "a.md"))
    text = open(path).read()
    assert "tier: **A**" in text and "eos_li_bcc" in text and "dimer_Li2" in text


def test_failed_prediction_blocks_the_tier():
    fr = [_frame(Atoms("Li"), -0.1, "a", "g", "test")]
    rec = records_from_arrays(fr, [np.nan], [None], ["SCCNotConvergedError: no"])
    s = summarize(rec, {"Li": -0.1}, {"Li": -0.1})
    assert s["failed"] and s["energies"]["by_role"]["test"]["failed"] == 1
    assert s["tier"] is None


def test_a_single_structure_family_has_no_relative_energy():
    fr = [_frame(Atoms("Li"), -0.1, "a", "g", "test")]
    rec = records_from_arrays(fr, [-0.05], [np.zeros((1, 3))])
    s = summarize(rec, {"Li": -0.1}, {"Li": -0.1})
    b = s["energies"]["by_role"]["test"]
    assert np.isnan(b["relative_mae"]) and b["relative_n"] == 0
    assert s["tier"] is None              # nothing shows the relative energies are right
