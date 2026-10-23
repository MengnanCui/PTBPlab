"""DFTB+'s charge mixers (milestone M6f-1).

The Anderson and DIIS mixers are checked against line-by-line transcriptions of DFTB+'s
``mixer/andersonmixer.F90`` and ``mixer/diismixer.F90`` (2-D storage, 1-based index arrays,
gesv); every mixer must reach the same SCC solution.  The SCC iterations against DFTB+ itself
are the frozen references *_mixer_* in test_reference.py.
"""
import numpy as np
import pytest

from pydftb.calculator import DFTB
from pydftb.mixer import (AndersonMixer, BroydenMixer, DIISMixer, PulayMixer, SimpleMixer,
                          make_mixer, mixer_spec)

from .test_spin import _system


class FortranAnderson:
    """andersonmixer.F90 as written (TAndersonMixer_mix, calcAndersonAverages, storeVectors);
    prevQInput(:, i) is row i - 1 here (contiguous, as the Fortran columns)."""

    def __init__(self, n, mix_param=0.05, generations=4, init_mix=0.01, conv=(), omega0=0.01):
        self.m = generations - 1
        self.prev_q = np.zeros((self.m, n))
        self.prev_f = np.zeros((self.m, n))
        self.indx = np.array([self.m + 1 - ii for ii in range(1, self.m + 1)])   # 1-based
        self.n_prev = -1
        self.mix_param, self.init_mix = mix_param, init_mix
        self.conv = np.array(conv, float).reshape(-1, 2).T          # convMixParam(2, nConv)
        self.break_sym = omega0 > 0
        self.omega02 = omega0 ** 2 if self.break_sym else 0.0

    def store(self, q, f):
        tmp = self.indx[self.m - 1]
        self.indx[1:] = self.indx[:-1].copy()
        self.indx[0] = tmp
        self.prev_q[self.indx[0] - 1] = q
        self.prev_f[self.indx[0] - 1] = f

    def mix(self, q, f):
        if self.n_prev < self.m:
            self.n_prev += 1
            mix_param = self.init_mix
        else:
            mix_param = self.mix_param
        r = np.sqrt(np.sum(np.abs(f) ** 2))
        for ii in range(self.conv.shape[1] - 1, -1, -1):
            if r < self.conv[0, ii]:
                mix_param = self.conv[1, ii]
                break
        if self.n_prev == 0:
            self.store(q, f)
            return q + self.init_mix * f
        n = self.n_prev
        aa, bb = np.zeros((n, n)), np.zeros((n, 1))
        for ii in range(n):
            tmp1 = f - self.prev_f[self.indx[ii] - 1]
            tmp2 = np.dot(f, tmp1)
            bb[ii, 0] = tmp2
            for jj in range(n):
                aa[ii, jj] = tmp2 - np.dot(self.prev_f[self.indx[jj] - 1], tmp1)
        if self.break_sym:
            for ii in range(n):
                aa[ii, ii] = (1.0 + self.omega02) * aa[ii, ii]
        bb = np.linalg.solve(aa, bb)
        f_mid = np.zeros_like(f)
        for ii in range(n):
            f_mid = f_mid + bb[ii, 0] * self.prev_f[self.indx[ii] - 1]
        f_mid = f_mid + (1.0 - np.sum(bb[:, 0])) * f
        q_mid = np.zeros_like(q)
        for ii in range(n):
            q_mid = q_mid + bb[ii, 0] * self.prev_q[self.indx[ii] - 1]
        q_mid = q_mid + (1.0 - np.sum(bb[:, 0])) * q
        self.store(q, f)
        return q_mid + mix_param * f_mid


class FortranDIIS:
    """diismixer.F90 as written (TDiisMixer_mix without AddIntrpGradient, storeVectors);
    storage as in :class:`FortranAnderson`."""

    def __init__(self, n, init_mix=0.2, generations=6, from_start=True):
        self.m = generations
        self.prev_q = np.zeros((self.m, n))
        self.prev_f = np.zeros((self.m, n))
        self.i_prev, self.indx = 0, 0
        self.init_mix, self.from_start = init_mix, from_start

    def mix(self, q, f):
        if self.i_prev < self.m:
            self.i_prev += 1
        self.indx = self.indx % self.m + 1
        self.prev_q[self.indx - 1] = q
        self.prev_f[self.indx - 1] = f
        out = q.copy()
        if self.from_start or self.i_prev == self.m:
            n = self.i_prev
            aa, bb = np.zeros((n + 1, n + 1)), np.zeros((n + 1, 1))
            for ii in range(n):
                for jj in range(n):
                    aa[ii, jj] = np.dot(self.prev_f[ii], self.prev_f[jj])
            aa[n, :n] = -1.0
            aa[:n, n] = -1.0
            bb[n, 0] = -1.0
            bb = np.linalg.solve(aa, bb)
            out = np.zeros_like(q)
            for ii in range(n):
                out = out + bb[ii, 0] * (self.prev_q[ii] + self.prev_f[ii])
        if self.i_prev < self.m:
            out = out + self.init_mix * f
        return out


def _drive(mixer, fortran, steps=14, n=9, seed=3):
    """Feed both mixers the same outputs of a contracting linear map, from the pydftb mixer's
    own sequence of inputs; return the largest relative difference of their next inputs."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    a = 0.9 * a / np.abs(np.linalg.eigvals(a)).max()
    b = rng.normal(size=n)
    q, dev = rng.normal(size=n), 0.0
    for _ in range(steps):
        q_out = a @ q + b
        new, ref = mixer.mix(q.copy(), q_out), fortran.mix(q.copy(), q_out - q)
        dev = max(dev, float(np.abs(new - ref).max() / np.abs(ref).max()))
        q = new
    return dev


@pytest.mark.parametrize("opts", [
    {}, {"generations": 2}, {"generations": 6, "init_mixing": 0.1},
    {"dynamic_mixing": [(1e-1, 0.3), (1e-3, 0.6)]}, {"diagonal_rescaling": 0.0},
    {"generations": 5, "diagonal_rescaling": 0.2, "dynamic_mixing": [(2.0, 0.2)]}])
def test_anderson_is_dftbplus_anderson(opts):
    mixer = make_mixer({"kind": "anderson", **opts}, 0.15)
    fortran = FortranAnderson(9, mix_param=0.15, generations=opts.get("generations", 4),
                              init_mix=opts.get("init_mixing", 0.01),
                              conv=opts.get("dynamic_mixing", ()),
                              omega0=opts.get("diagonal_rescaling", 0.01))
    assert isinstance(mixer, AndersonMixer)
    assert _drive(mixer, fortran) < 1e-12


@pytest.mark.parametrize("opts", [{}, {"use_from_start": False}, {"generations": 3},
                                  {"generations": 4, "init_mixing": 0.4, "use_from_start": False}])
def test_diis_is_dftbplus_diis(opts):
    mixer = make_mixer({"kind": "diis", **opts})
    fortran = FortranDIIS(9, init_mix=opts.get("init_mixing", 0.2),
                          generations=opts.get("generations", 6),
                          from_start=opts.get("use_from_start", True))
    assert isinstance(mixer, DIISMixer)
    assert _drive(mixer, fortran, steps=8) < 1e-12


def test_simple_mixer_and_first_steps():
    q, q_out = np.array([1.0, 2.0, 3.0]), np.array([1.5, 1.0, 3.5])
    assert np.array_equal(SimpleMixer(0.3).mix(q, q_out), q + 0.3 * (q_out - q))
    # Anderson's first call mixes with InitMixingParameter, also under DynMixingParameters
    m = AndersonMixer(alpha=0.5, init_alpha=0.02, dynamic=[(10.0, 0.9)])
    assert np.array_equal(m.mix(q, q_out), q + 0.02 * (q_out - q))
    # DIIS from the start: one vector, c = 1, so q + F + init F
    assert np.allclose(DIISMixer(init_alpha=0.2).mix(q, q_out), q_out + 0.2 * (q_out - q))
    assert np.array_equal(DIISMixer(from_start=False).mix(q, q_out), q + 0.2 * (q_out - q))


def test_mixer_options():
    assert mixer_spec() == ("broyden", {})
    assert mixer_spec("Anderson", 0.1) == ("anderson", {"alpha": 0.1})
    assert mixer_spec({"kind": "broyden", "mixing": 0.3, "inverse_jacobi_weight": 0.02,
                       "minimal_weight": 2.0, "maximal_weight": 1e4, "weight_factor": 0.05}) == (
        "broyden", {"alpha": 0.3, "omega0": 0.02, "min_weight": 2.0, "max_weight": 1e4,
                    "weight_factor": 0.05})
    assert mixer_spec({"kind": "diis", "init_mixing": 0.3, "generations": 4,
                       "use_from_start": False}) == (
        "diis", {"init_alpha": 0.3, "generations": 4, "from_start": False})
    assert mixer_spec("pulay", 0.3) == ("pulay", {"beta": 0.3})
    assert isinstance(make_mixer("pulay"), PulayMixer)
    assert make_mixer("simple").alpha == 0.05 and make_mixer("anderson").alpha == 0.05
    assert make_mixer().alpha == 0.2
    with pytest.raises(ValueError, match="unknown mixer"):
        make_mixer("kerker")
    with pytest.raises(ValueError, match="unknown options"):
        make_mixer({"kind": "anderson", "history": 3})
    with pytest.raises(ValueError, match="no mixing parameter"):
        make_mixer("diis", 0.1)
    with pytest.raises(ValueError, match="generations >= 2"):
        make_mixer({"kind": "anderson", "generations": 1})
    _, sk, _, _, _, _, _ = _system("h2o")
    with pytest.raises(ValueError, match="unknown options"):
        DFTB(sk, mixer={"kind": "simple", "generations": 3})     # checked at construction


def test_default_mixing_is_broyden_02():
    """mixing=None takes the mixer's DFTB+ default: Broyden 0.2, the value before M6f."""
    _, sk, labels, pos, _, _, _ = _system("ch3oh")
    a = DFTB(sk).calculate(labels, pos)
    b = DFTB(sk, mixer="broyden", mixing=0.2).calculate(labels, pos)
    assert a.n_scc == b.n_scc and a.energy == b.energy
    assert np.array_equal(a.scc_errors, b.scc_errors)
    assert len(a.scc_errors) == a.n_scc and a.scc_errors[-1] < 1e-8


@pytest.mark.parametrize("mixer", ["simple", "anderson", "diis", "pulay",
                                   {"kind": "broyden", "inverse_jacobi_weight": 0.02,
                                    "minimal_weight": 2.0, "weight_factor": 0.05}])
def test_every_mixer_reaches_the_broyden_solution(mixer):
    # shell-resolved: 16 SCC sites, more than DIIS's 6 vectors (fewer sites make its matrix
    # singular, as in DFTB+)
    case, sk, labels, pos, _, _, _ = _system("cu4co_shell_resolved")
    ref = DFTB(sk, smearing=tuple(case.filling), scc_tol=1e-12).calculate(labels, pos)
    r = DFTB(sk, smearing=tuple(case.filling), scc_tol=1e-12, max_scc=1000,
             mixer=mixer, mixing=0.1 if mixer == "simple" else None).calculate(labels, pos)
    assert r.energy == pytest.approx(ref.energy, abs=1e-10)
    assert np.allclose(r.charges, ref.charges, atol=1e-9)
    assert np.allclose(r.forces, ref.forces, atol=1e-8)


def test_torch_and_autodiff_refuse():
    pytest.importorskip("torch")
    from pydftb.autodiff import evaluate
    _, sk, labels, pos, _, _, _ = _system("h2o")
    with pytest.raises(ValueError, match="the anderson mixer.*NumPy backend only"):
        DFTB(sk, mixer="anderson", backend="torch")
    with pytest.raises(ValueError, match="the broyden mixer.*NumPy backend only"):
        DFTB(sk, mixer={"kind": "broyden", "weight_factor": 0.1}, backend="torch")
    DFTB(sk, mixer="broyden", mixing=0.3, backend="torch")          # as before M6f
    DFTB(sk, mixer="pulay", backend="torch")
    with pytest.raises(NotImplementedError, match="the diis mixer"):
        evaluate(DFTB(sk, mixer="diis"), labels, pos)


def test_broyden_mixer_unchanged():
    """BroydenMixer itself is untouched by M6f: a fixed sequence of its outputs."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=(5, 5))
    a = 0.8 * a / np.abs(np.linalg.eigvals(a)).max()
    b = rng.normal(size=5)
    m, q = BroydenMixer(), np.zeros(5)
    for _ in range(30):
        q = m.mix(q, a @ q + b)
    assert np.allclose(q, np.linalg.solve(np.eye(5) - a, b), atol=1e-10)
