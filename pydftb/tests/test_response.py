"""Polarisability and response kernel (milestone M6g-1, DFTB+ ``Polarisability`` and
``ResponseKernel``).

The DFTB+ references are the ``*_pol*`` and ``*_kernel*`` cases of
:mod:`pydftb.validation.refgen`.  Here: the static polarisability against finite differences of
the dipole in an external field, dE_F/dE of a metal against the Fermi level's, the symmetry and
charge conservation of the kernel, the zero-frequency limit, rotations inside degenerate runs,
the options that do not change the response, the text outputs and the refused combinations.
"""
import numpy as np
import pytest

from pydftb import DFTB, ParamSet
from pydftb import dftbplus_io as io
from pydftb import response as resp
from pydftb.constants import BOHR
from pydftb.validation import cases as C

SKF = C.SKF_SETS["ptbp_hcocu"]
DFTB3 = {"third_order": "full", "hubbard_derivs": C.UD_3OB, "h_damping": 4.0}
O2_SPIN = {"spin": True, "unpaired_electrons": 2.0, "spin_constants": {"O": [-0.028]}}


def _calc(atoms, analysis=None, shell_resolved=False, **kw):
    labels = atoms.get_chemical_symbols()
    p = ParamSet(SKF, max_l={e: C.MAXL[e] for e in set(labels)}, shell_resolved=shell_resolved)
    return DFTB(p.skset(labels), scc_tol=1e-12, max_scc=500, analysis=analysis, **kw), labels


def _run(atoms, analysis=None, **kw):
    calc, labels = _calc(atoms, analysis, **kw)
    return calc.calculate(labels, atoms.positions / BOHR)


@pytest.mark.parametrize("atoms, kw", [
    (C._water(), {}),
    (C._water(), {"shell_resolved": True}),
    (C._meoh(), DFTB3),
    (C._o2(), O2_SPIN),
    (C._water(), {"charge": -1.0, "smearing": ("fermi", 300.0)}),
    (C._cu4co(), {"smearing": ("fermi", 3000.0)}),
], ids=["water", "water_shell", "ch3oh_dftb3", "o2_triplet", "water_anion_300K", "cu4co_3000K"])
def test_polarisability_against_dipole_differences(atoms, kw):
    """alpha_ji = d mu_j / d E_i: central differences of the dipole in a field (pydftb's
    electric_field, DFTB+'s sign: the electrons' energy + E.r)."""
    r = _run(atoms, {"polarisability": True}, **kw)
    alpha = r.analysis.polarisability.tensor[0]
    h = 1e-4
    for i in range(3):
        mu = []
        for s in (h, -h):
            e = np.zeros(3)
            e[i] = s
            mu.append(_run(atoms, electric_field=e, **kw).dipole)
        fd = (mu[0] - mu[1]) / (2 * h)
        assert np.abs(fd - alpha[:, i]).max() < 2e-5 * max(1.0, np.abs(alpha).max())
    assert np.abs(alpha - alpha.T).max() < 1e-9 * np.abs(alpha).max()   # static: symmetric


def test_fermi_derivative_of_a_metal():
    """dE_F/dE of a metal (the physical value; DFTB+ prints it halved) against differences of
    the Fermi level in a field."""
    atoms = C._cu4co()
    kw = {"smearing": ("fermi", 3000.0)}
    r = _run(atoms, {"polarisability": True}, **kw)
    pol = r.analysis.polarisability
    assert pol.metallic and pol.dos_fermi[0] > 0
    h = 1e-4
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        fd = (_run(atoms, electric_field=e, **kw).fermi
              - _run(atoms, electric_field=-e, **kw).fermi) / (2 * h)
        assert abs(fd - pol.fermi_derivatives[0, i, 0]) < 1e-6


@pytest.mark.parametrize("atoms, kw", [
    (C._water(), {}),
    (C._cu4co(), {"smearing": ("fermi", 3000.0)}),
    (C._o2(), O2_SPIN),
], ids=["water", "cu4co_3000K", "o2_triplet"])
def test_kernel_symmetric_and_conserving(atoms, kw):
    """dN_B/dV_A = dN_A/dV_B (static) and every column sums to zero (metals through the move
    of the Fermi level); the bare (RPA) kernel too; finite frequency keeps the zero sums."""
    for spec in (True, {"rpa": True}, {"static": False, "frequencies": [0.05]}):
        k = _run(atoms, {"response_kernel": spec}, **kw).analysis.response_kernel
        q = k.atom[0, :, 0, :]
        assert np.abs(q.sum(axis=1)).max() < 1e-10
        if spec is not True and "frequencies" in spec:
            continue
        assert np.abs(q - q.T).max() < 1e-10
        # Mulliken and on-site populations: the same total change
        assert abs(k.net[0].sum()) < 1e-10


def test_zero_frequency_limit():
    """alpha(omega) of an insulator tends to the static value (as omega^2)."""
    pol = _run(C._water(), {"polarisability": {"frequencies": [1e-4, 2e-4]}}
               ).analysis.polarisability
    d1 = np.abs(pol.tensor[1] - pol.tensor[0]).max()
    d2 = np.abs(pol.tensor[2] - pol.tensor[0]).max()
    assert d1 < 1e-6 and abs(d2 / d1 - 4.0) < 0.05


def test_dispersion_does_not_change_the_response():
    """Dispersion does not depend on the density (DFTB+ refuses it, pydftb allows it)."""
    atoms = C._water()
    a = _run(atoms, {"polarisability": True, "response_kernel": True}).analysis
    b = _run(atoms, {"polarisability": True, "response_kernel": True},
             dispersion={"model": "lennardjones"}).analysis
    assert np.array_equal(a.polarisability.tensor, b.polarisability.tensor)
    assert np.array_equal(a.response_kernel.orbital, b.response_kernel.orbital)


def test_degenerate_runs_rotation_free():
    """A partly filled degenerate pair (O2 without spin polarisation, 300 K: the pi* pair
    holds two electrons): chi0 does not depend on the eigenvectors' rotation inside the run."""
    atoms = C._o2()
    calc, labels = _calc(atoms, {"polarisability": True}, smearing=("fermi", 300.0))
    st = calc._setup(labels, atoms.positions / BOHR, None, None, False, {})
    sc = calc._scc(st, None, False, {})
    gs = resp.GroundState(calc, st, sc)
    x0 = resp.chi0(gs, [0.0])[0].orb
    rng = np.random.default_rng(3)
    rotated = 0
    for blk in gs.blocks:
        for gid in np.unique(blk.group):
            idx = np.flatnonzero(blk.group == gid)
            if idx.size == 2 and blk.first_empty <= idx[0] < blk.n_filled:
                t = rng.uniform(0, np.pi)
                u = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                blk.c[:, idx] = blk.c[:, idx] @ u
                blk.sc[:, idx] = blk.sc[:, idx] @ u
                rotated += 1
    assert rotated >= 1
    x1 = resp.chi0(gs, [0.0])[0].orb
    assert np.abs(x1 - x0).max() < 1e-9 * np.abs(x0).max()


def test_kpoints_reduced_equals_full():
    """The finite-frequency response with a time-reversal reduced k-point set equals the full
    set's (the eta term of complex eigenvectors cancels between k and -k; DFTB+ keeps it for
    reduced sets, ``DFTB._response_eta_compat``)."""
    from pydftb.api import make_kpoints
    atoms = C._cuo_chain()
    labels = atoms.get_chemical_symbols()
    p = ParamSet(SKF, max_l={"Cu": 2, "O": 1})
    cell = atoms.cell.array / BOHR
    out = []
    for tr in (True, False):
        calc = DFTB(p.skset(labels), scc_tol=1e-12, max_scc=500, smearing=("fermi", 1000.0),
                    kpts=make_kpoints((4, 2, 2), time_reversal=tr),
                    analysis={"polarisability": {"frequencies": [0.0367]}})
        r = calc.calculate(labels, atoms.positions / BOHR, cell, np.ones(3, bool))
        out.append(r.analysis.polarisability.charge_derivatives)
    assert np.abs(out[0] - out[1]).max() < 1e-10
    assert r.analysis.polarisability.tensor is None              # periodic: no tensor


def test_frequency_list_and_options():
    w = resp.frequencies({"frequencies": [0.1], "frequency_range": (0.25, 0.75, 0.25)},
                         "polarisability")
    np.testing.assert_allclose(w, [0.0, 0.1, 0.25, 0.5, 0.75])
    # DFTB+'s count int((stop - start) / step) + 1 may drop the end point by rounding
    w = resp.frequencies({"static": False, "frequency_range": (0.2, 0.5, 0.1)}, "polarisability")
    np.testing.assert_allclose(w, [0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="nothing to calculate"):
        DFTB(ParamSet(SKF).skset(["O", "H"]), analysis={"polarisability": {"static": False}})
    with pytest.raises(ValueError, match="unknown keys"):
        DFTB(ParamSet(SKF).skset(["O", "H"]), analysis={"polarisability": {"freq": [0.1]}})
    with pytest.raises(ValueError, match="perturbation"):
        DFTB(ParamSet(SKF).skset(["O", "H"]),
             analysis={"polarisability": True, "perturbation": {"tol": 1e-9}})


def test_text_outputs(tmp_path):
    r = _run(C._water(), {"polarisability": {"frequencies": [0.1]}, "response_kernel": True})
    text = io.response_tables(r)
    assert " Electric field polarisability (a.u.)" in text and " Static polarisability:" in text
    assert "Polarisability at omega = :" in text and "Derivatives wrt. a potential at atom 3" in text
    io.write_band_derivatives(tmp_path / "dE_band.out", r)
    io.write_band_derivatives(tmp_path / "dV_band.out", r, "potential")
    assert (tmp_path / "dE_band.out").read_text().startswith(" DIR x KPT ")
    assert (tmp_path / "dV_band.out").read_text().startswith("ATOM 1  KPT ")


def test_default_path_unchanged():
    """Without the response analyses nothing changes (another analysis in the same run)."""
    atoms = C._water()
    a = _run(atoms, {"cm5": True})
    b = _run(atoms, {"cm5": True, "polarisability": True})
    assert a.energy == b.energy and np.array_equal(a.charges, b.charges)
    assert np.array_equal(a.analysis.cm5_charges, b.analysis.cm5_charges)


@pytest.mark.parametrize("kw, what", [
    ({"smearing": ("mp", 300.0, 1)}, "Fermi filling"),
    ({"solvation": {"model": "gbsa", "solvent": "water"}}, "solvation"),
    ({"third_order": "onsite", "hubbard_derivs": {"O": [-0.1575], "H": [-0.1857]}},
     "full third order"),
    ({"spin_orbit": {"O": [0.0, 0.01]}, "temperature": 300.0}, "M6g-3b"),
])
def test_refused(kw, what):
    with pytest.raises(ValueError, match=what):
        _calc(C._water(), {"polarisability": True}, **kw)


def test_refused_xlbomd_and_torch():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="analysis"):
        _calc(C._water(), {"polarisability": True}, backend="torch")
