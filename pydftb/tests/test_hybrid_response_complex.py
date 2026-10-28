"""Responses with a hybrid functional at k-points and with two-component spin (milestone
M6h-4b): DFTB+ 25.1 crashes there (B24, B25), so the checks are pydftb's own: the supercell's
Gamma point, finite differences of site potentials, fields and point charges, M6g's and M6g-3b's
responses without the exchange, the full k-point grid and the collinear limit.
"""
import copy
import dataclasses

import numpy as np
import pytest

from pydftb import response as R
from pydftb import response_hybrid as H
from pydftb import response_pauli as RP
from pydftb.calculator import DFTB
from pydftb.constants import BOHR, HARTREE
from pydftb.kpoints import monkhorst_pack, supercell_folding
from pydftb.tests import test_hybrid_kpoints as TK
from pydftb.tests import test_hybrid_response as T
from pydftb.validation import refgen
from pydftb.validation.cases import ONS, U_O

KERNEL = {"response_kernel": True}
POL = {"polarisability": True}
CO_BOX_HYBRID = {"type": "lc", "coulomb_cutoff": 7.0, "gsum_cutoff": 14.0}


class _Grabbed(Exception):
    pass


def _ground(name, what="response_kernel", **over):
    """(calc, st, sc, opts) of a reference case's ground state: the analysis' arguments."""
    def grab(calc, st, sc, spec, opts):
        raise _Grabbed(calc, st, sc, opts)
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], analysis={what: True}, **over)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(R, what, grab)
        try:
            refgen.run_pydftb(case, refgen._inputs(case))
        except _Grabbed as g:
            return g.args
    raise AssertionError("the analysis was not called")


def _run(name, analysis, **over):
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], analysis=analysis, **over)
    return refgen.run_pydftb(case, refgen._inputs(case))


def _site_fd(name, atoms, h=1e-4, **over):
    """dN / dV_A by 4-point central differences of gross site potentials (pydftb's SCC)."""
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], scc_tol=1e-13, analysis=None, **over)
    inp = refgen._inputs(case)
    out = {}
    for a in atoms:
        q = {}
        for s in (2, 1, -1, -2):
            c = dataclasses.replace(case, site_potentials={"gross": {a: s * h * HARTREE}})
            q[s] = np.asarray(refgen.run_pydftb(c, inp).charges)
        out[a] = (8.0 * (q[1] - q[-1]) - (q[2] - q[-2])) / (12.0 * h)
    return out


def _zero_exchange(m, st):
    m.setattr(type(st.hybrid), "response_fock", lambda self, ps: [np.zeros_like(p) for p in ps])


# ---- k-points --------------------------------------------------------------------------------
def _o2_in_co_box():
    """Triplet O2 in the CO box: its ground state, both spin channels closed.  (Triplet CO, an
    excited configuration, lets the supercell's copies trade charge under the fixed total moment:
    its SCC need not converge.)"""
    return ["O", "O"], np.array([[0.3, 0.2, 0.1], [1.50, 0.29, 0.19]]) / BOHR, TK._co_box()[2]


@pytest.mark.filterwarnings("error:SCC not converged")
@pytest.mark.parametrize("hybrid, spin", [
    (CO_BOX_HYBRID, None),
    ({**CO_BOX_HYBRID, "screening": "neighbour"}, None),
    (CO_BOX_HYBRID, {"spin": True, "unpaired_electrons": 2.0, "spin_constants": {"O": -0.028}})])
def test_kpoint_kernel_is_the_supercells_gamma_point(hybrid, spin):
    """A molecule in a box without self-image overlaps (CO; with spin triplet O2): the k-point
    kernel of the cell is the supercell's Gamma-point kernel (M6h-4, against DFTB+) with the
    images summed."""
    labels, pos, cell = _o2_in_co_box() if spin else TK._co_box()
    n = (2, 1, 1)
    sk = TK._sk(labels)
    kw = dict(hybrid=hybrid, scc_tol=1e-12, analysis=KERNEL, **(spin or {}))
    r = DFTB(sk, kpts=supercell_folding(n), **kw).calculate(labels, pos, cell, TK.PBC,
                                                            forces=False)
    if spin:
        kw["unpaired_electrons"] = 2 * spin["unpaired_electrons"]    # per cell copy
    sc_pos, sc_cell, _ = TK._supercell(pos, cell, n)
    r_sc = DFTB(sk, **kw).calculate(labels * 2, sc_pos, sc_cell, TK.PBC, forces=False)
    ker = r.analysis.response_kernel.atom[0, :, 0, :]
    ksc = r_sc.analysis.response_kernel.atom[0, :, 0, :].reshape(2, 2, 2, 2).sum(0)[:, 0, :]
    assert np.abs(ker - ksc).max() <= 1e-11


@pytest.mark.parametrize("name, over", [
    ("diamond2_lc_k222", {}),
    ("diamond2_lc_k222", {"filling": ("fermi", 30000.0)}),          # metallic: E_F moves
    ("o2_box_lc_spin_k222", {}),
    ("co2_crystal_lc_k211_plusu_b20", {"stale_blocks_compat": False}),
    ("diamond2_lc_k312_mic", {})])
def test_kpoint_kernel_is_the_site_potential_one(name, over):
    """dN_B / dV_A against differences of a gross site potential on atom A (+U without
    DFTB+ 25.1's stale k-point blocks, B20, whose SCC has no block response)."""
    ker = _run(name, KERNEL, scc_tol=1e-13, **over).analysis.response_kernel.atom[0, :, 0, :]
    fd = _site_fd(name, [0], **over)
    assert np.abs(ker[0] - fd[0]).max() <= 2e-9


@pytest.mark.parametrize("name, over", [
    ("co2_crystal_lc_k211_plusu_b20", {"stale_blocks_compat": False}),
    ("diamond2_lc_k222", {"filling": ("fermi", 30000.0)}),
    ("o2_box_lc_spin_k222", {})])
def test_kpoints_without_exchange_are_m6g(name, over):
    """With the exchange left out, the kernel and the charge response are M6g's k-point
    response on the same ground state (the block channels, the spin constants, the Fermi
    level's move)."""
    calc, st, sc, opts = _ground(name, **over)
    gs_v = np.asarray(st.basis.shell_atom)
    with pytest.MonkeyPatch.context() as m:
        _zero_exchange(m, st)
        hr = H.KPointHybridResponse(calc, st, sc, opts)
        orb = hr.solve(np.eye(len(st.species))[gs_v])[0]
        pol = H.polarisability(calc, st, sc, True, opts)
    plain = dataclasses.replace(st, hybrid=None)
    ref = R.response_kernel(calc, plain, sc, True, opts)
    assert np.abs(orb - ref.orbital[0]).max() <= 1e-12
    ref_pol = R.polarisability(calc, plain, sc, True, opts)
    assert np.abs(pol.charge_derivatives[0] - ref_pol.charge_derivatives[0]).max() <= 1e-12


def test_reduced_and_full_kpoint_grids_agree():
    """DFTB+'s reduced supercell folding and the whole Gamma-centred grid give one kernel."""
    labels, pos, cell = TK._diamond()
    sk = TK._sk(labels)
    out = []
    for kp in (supercell_folding((3, 1, 2)), monkhorst_pack((3, 1, 2), gamma_centered=True)):
        r = DFTB(sk, kpts=kp, hybrid="lc", scc_tol=1e-12, analysis=KERNEL).calculate(
            labels, pos, cell, TK.PBC, forces=False)
        out.append(r.analysis.response_kernel.atom[0, :, 0, :])
    assert np.abs(out[0] - out[1]).max() <= 1e-11


def test_periodic_charge_response_is_the_kernel_times_positions():
    """The periodic polarisability is the charge response to the atoms' coordinates (as M6g and
    DFTB+): dN / dE_i = sum_A x_A,i dN / dV_A."""
    st = _ground("co2_crystal_lc_k211")[1]
    pol = _run("co2_crystal_lc_k211", POL).analysis.polarisability
    assert pol.tensor is None
    ker = _run("co2_crystal_lc_k211", KERNEL).analysis.response_kernel
    expect = np.einsum("Ai,AB->iB", np.asarray(st.pos, float), ker.atom[0, :, 0, :])
    assert np.abs(pol.charge_derivatives[0][:, 0, :] - expect).max() <= 1e-10


def test_kpoint_exchange_batches_and_takes_the_thresholds_limit():
    """KPointHybridTerms.response_fock of several right-hand sides is the unscreened fock of
    each (with neighbour_threshold: the screening's limit) and leaves the ground state's
    screening history alone."""
    st = _ground("diamond2_lc_k312_neighbour_t6")[1]
    hyb = st.hybrid
    assert hyb.threshold is not None
    rng = np.random.default_rng(5)
    d = rng.normal(size=(2, hyb.layout.nk, hyb.n, hyb.n)) * (1.0 + 0.5j)
    d = d + np.conj(np.swapaxes(d, -1, -2))
    history = [None if x is None else x.copy() for x in (hyb._prev, hyb._hprev)]
    out = hyb.response_fock([d])[0]
    unscreened = copy.copy(hyb)
    unscreened.threshold = None
    ref = np.stack([unscreened.fock([d[r]])[0] for r in range(2)])
    assert np.abs(out - ref).max() <= 1e-13 * np.abs(ref).max()
    for before, now in zip(history, (hyb._prev, hyb._hprev)):
        assert (before is None and now is None) or np.array_equal(before, now)


def test_full_kernel_saddle_point_warns():
    """The untruncated kernel on diamond's 2x2x2 grid: the symmetric SCC solution is a saddle
    point (charge transfer between the two atoms lowers the energy); the response of this
    stationary state comes with the warning."""
    with pytest.warns(RuntimeWarning, match="saddle point of the SCC"):
        _run("diamond2_lc_k222_full", KERNEL)


def test_minimum_image_neighbour_responses_are_refused():
    with pytest.raises(ValueError, match="not self-adjoint"):
        _run("diamond2_lc_k222_mic_neighbour", KERNEL)


# ---- two-component spin ----------------------------------------------------------------------
@pytest.mark.parametrize("name, over", [("hco_lc_noncolinear", {}),
                                        ("o2_cam_noncolinear_90", {}),
                                        ("h3_lc_noncolinear_120", {}),
                                        ("hco_lc_noncolinear", {"plus_u": U_O}),
                                        ("hco_lc_noncolinear", {"onsite": ONS})])
def test_two_component_polarisability_is_the_finite_field_one(name, over):
    """Against pydftb's finite fields (h = 1e-4: H3's response is strongly nonlinear); the
    +U and onsite blocks in the four Pauli components."""
    a = _run(name, POL, scc_tol=1e-13, **over).analysis.polarisability.tensor[0]
    fd = T._finite_field(name, h=1e-4, **over)
    assert np.abs(a - fd).max() <= 2e-8 * np.abs(fd).max()


def test_two_component_kernel_is_the_site_potential_one():
    ker = _run("hco_lc_noncolinear", KERNEL, scc_tol=1e-13).analysis.response_kernel
    assert ker.atom.shape[2] == 4
    fd = _site_fd("hco_lc_noncolinear", [0, 1])
    for a, f in fd.items():
        assert np.abs(ker.atom[0, a, 0, :] - f).max() <= 2e-9


def test_two_component_without_exchange_is_m6g3b():
    """H3 with moments at 120 degrees: without the exchange, M6g-3b's two-component response
    on the same ground state, all four components."""
    calc, st, sc, opts = _ground("h3_lc_noncolinear_120", what="polarisability")
    with pytest.MonkeyPatch.context() as m:
        _zero_exchange(m, st)
        mine = H.polarisability(calc, st, sc, True, opts)
    ref = RP.polarisability(calc, dataclasses.replace(st, hybrid=None), sc, True, opts)
    assert np.abs(mine.tensor[0] - ref.tensor[0]).max() <= 1e-12
    assert np.abs(mine.charge_derivatives[0] - ref.charge_derivatives[0]).max() <= 1e-12


def test_two_component_collinear_moments_are_the_collinear_response():
    """Moments along z: the charge response is the collinear hybrid's (M6h-4, against DFTB+),
    the z magnetisation's its magnetisation response, the transverse ones zero."""
    nc = refgen.CASES_BY_NAME["hco_lc_noncolinear"].noncolinear
    z = {**nc, "initial_spins": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]}
    col = _run("hco_lc_spin_neighbour", POL, hybrid="lc", scc_tol=1e-12).analysis.polarisability
    res = _run("hco_lc_noncolinear", POL, noncolinear=z, scc_tol=1e-12).analysis.polarisability
    assert np.abs(res.tensor[0] - col.tensor[0]).max() <= 1e-8
    dq, dq_col = res.charge_derivatives[0], col.charge_derivatives[0]
    assert np.abs(dq[:, 0] - dq_col[:, 0]).max() <= 1e-8
    assert np.abs(dq[:, 3] - dq_col[:, 1]).max() <= 1e-8
    assert np.abs(dq[:, 1:3]).max() <= 1e-8


def test_spin_rotations_are_projected_out():
    """Without spin-orbit coupling the global spin rotations are zero modes: three for H3 at 120
    degrees, two for moments along one axis.  The solution has no component along them, and
    where MINRES also converges without the projection the response is the same."""
    calc, st, sc, opts = _ground("h3_lc_noncolinear_120", what="polarisability")
    hr = H.PauliHybridResponse(calc, st, sc, opts)
    assert len(hr.rotations) == 3
    v = np.asarray(st.pos, float)[hr.gs.shell_atom]
    b = hr._project(hr.t_adjoint(hr._external(v[hr.gs.orb_shell].T)))
    z = hr._minres(b)[0]
    for q in hr.rotations:
        assert np.abs(np.einsum("ij,rij->r", q.conj(), z).real).max() <= 1e-12 * np.abs(z).max()
    calc, st, sc, opts = _ground("hco_lc_noncolinear", what="polarisability")
    hr = H.PauliHybridResponse(calc, st, sc, opts)
    assert len(hr.rotations) == 2
    v = np.asarray(st.pos, float)[hr.gs.shell_atom]
    with_projection = hr.solve(v)[0]
    hr.rotations = []
    without = hr.solve(v)[0]
    assert np.abs(with_projection - without).max() <= 1e-10


@pytest.mark.parametrize("name", ["hco_lc_noncolinear_field_pc", "co2_crystal_lc_k211_field_pc"])
def test_point_charge_derivatives_are_the_finite_difference_ones(name):
    """dN / dR_j of the point charges (with a field), two-component and at k-points, against
    moving them."""
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], scc_tol=1e-13)
    inp = refgen._inputs(case)
    ext = _run(name, {"external_charge_derivatives": True},
               scc_tol=1e-13).analysis.external_charge_derivatives.atom[:, :, 0, :]
    h = 1e-3
    pos = np.array(case.point_charges["positions"], float)
    for j, c in ((0, 0), (len(pos) - 1, 2)):
        q = {}
        for s in (2, 1, -1, -2):
            p = pos.copy()
            p[j, c] += s * h * BOHR                              # Angstrom
            cc = dataclasses.replace(case, analysis=None,
                                     point_charges={**case.point_charges, "positions": p.tolist()})
            q[s] = np.asarray(refgen.run_pydftb(cc, inp).charges)
        fd = (8.0 * (q[1] - q[-1]) - (q[2] - q[-2])) / (12.0 * h)
        assert np.abs(ext[j, c] - fd).max() <= 1e-7
