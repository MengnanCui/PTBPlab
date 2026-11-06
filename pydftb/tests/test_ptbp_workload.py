"""PTBP's Cu DFTB stage without the dataset (milestone M7e): the structures, k-point grids
and calculator settings of ``pydftb.validation.ptbp_workload``."""
import os

import numpy as np

from pydftb.validation import ptbp_workload as W

# k-point grids (n, n, n) of EOS points 0..10, PTBP's rule on dft/dft.xyz (M7e design, P129)
GRIDS = {"dimer": [1] * 11, "AA-graphite": [1] * 11, "diamond": [1] * 11,
         "beta-Sn": [15, 15, 14, 14, 14, 14, 14, 13, 13, 13, 13],
         "bcc": [16, 16, 16, 15, 15, 15, 15, 14, 14, 14, 14],
         "fcc": [16, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14]}


def test_the_stage_structures():
    ss = W.structures()
    assert len(ss) == 66
    assert [s.name for s in ss[:12]] == [f"dimer {i:02d}" for i in range(11)] + ["AA-graphite 00"]
    for s in ss:
        assert s.kpts == (GRIDS[s.phase][s.eos],) * 3
        assert s.gamma == (s.phase in W.GAMMA_ONLY)
        assert s.norb == 9 * len(s.atoms) == (9 if s.phase in ("bcc", "fcc") else 18)
        assert s.atoms.pbc.all() and set(s.atoms.get_chemical_symbols()) == {"Cu"}
        if not s.gamma:                         # the rule itself: int(5 * 2 pi |b_i|)
            assert W.kpoint_grid(s.atoms) == s.kpts


def test_volumes_and_fractional_positions():
    by_name = {s.name: s for s in W.structures()}
    for phase, (cell, frac) in W.PHASES.items():
        centre = by_name[f"{phase} 05"].atoms
        assert np.abs(centre.cell.array - np.array(cell)).max() < 1e-14
        for i in range(11):
            a = by_name[f"{phase} {i:02d}"].atoms
            assert abs(a.get_volume() / centre.get_volume() - W.volume_ratio(i)) < 1e-12
            assert np.abs(a.get_scaled_positions(wrap=False) - np.array(frac)).max() < 1e-12


def test_quick_subset_and_phases():
    quick = W.structures(quick=True)
    assert [s.eos for s in quick] == [0, 5, 10] * 6
    assert {s.phase for s in W.structures(phases=["bcc"])} == {"bcc"}


def test_the_calculator_has_ptbps_settings():
    s = W.structures(phases=["fcc"])[0]
    calc = W.calculator(s)
    assert os.path.isdir(W.skf_dir())
    assert calc.kpts_spec == s.kpts and calc.time_reversal is True
    assert calc.threads == 1 and calc.k_threads.n == 1
    o = calc.engine_opts
    assert (o["scc"], o["scc_tol"], o["max_scc"], o["repulsive"], o["fd_delta"]) == \
        (True, 1e-5, 1000, "polynomial", 0.01)
    assert calc.params.shell_resolved
    assert W.calculator(s, k_threads=2).k_threads.n == 2
