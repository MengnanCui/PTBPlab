"""Self-consistent-charge DFTB driver (DFTB2, DFTB3, collinear and non-collinear spin,
spin-orbit coupling, dispersion): energies, forces, stress.

Reference implementation: clarity first, dense linear algebra, NumPy/SciPy.

A calculation runs in four stages, each a separate method so that new
physics (DFTB3, spin, stress, ...) has an obvious place to plug in:

1. ``_setup``     basis, neighbour pairs, H0/S (real space and folded), Coulomb kernel,
                  dispersion (it depends on the geometry only)
2. ``_scc``       self-consistent charge loop (density matrices from a solver)
3. ``_energies``  energy terms, populations, DFTB+-style energy bookkeeping
4. ``_gradient``  analytic forces (band + SCC + repulsive + dispersion)

The matching DFTB+ routines are listed in ``pydftb/docs/dftbplus_map.md``.
``pydftb.bands`` reuses stage 1 and the potentials of stage 2 to diagonalise H(k) at
frozen charges (band structures, DOS).

Two-component calculations (non-collinear spin, spin-orbit coupling; milestone M6a) take
separate branches (``st.pauli``): 2N x 2N spinor matrices, four Pauli components
(q, m_x, m_y, m_z) per shell in the mixed vector, one Fermi level; see
:mod:`pydftb.spinorbit`.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from .backend import ops
from .backend.spec import make_backend
from .coulomb import Coulomb
from .dispersion import DispersionTerms, make_dispersion
from .hamiltonian import Basis, PairBlocks, build_pairs, fold, fold_all, onsite_energies, shift_index
from .mixer import make_mixer
from .occupations import Smearing
from .neighbors import neighbor_list
from .parallel import as_kthreads
from .solvers import SOLVERS, _density
from .skf import SKSet
from . import spinorbit as so
from .thirdorder import ThirdOrder


@dataclass
class DFTBResult:
    """Results in atomic units (Hartree, Bohr).

    Energy names follow DFTB+'s ``detailed.out``:
    ``energy`` = "Total energy" (E_H0 + E_SCC + E_3rd + E_spin + E_LS + E_rep + E_disp),
    ``free_energy`` = "Total Mermin free energy" (Total - TS),
    ``e_zero`` = "Extrapolated to 0" (Total - TS/2).
    """
    energy: float                 # total energy E
    free_energy: float            # Mermin free energy E - TS
    e_band: float                 # Tr(rho H0)
    e_scc: float
    e_rep: float
    ts: float
    fermi: float
    charges: np.ndarray           # Mulliken electron populations per atom
    eigenvalues: list | None
    forces: np.ndarray | None = None
    n_scc: int = 0
    converged: bool = True
    timings: dict = field(default_factory=dict)
    q_shell: np.ndarray | None = None    # Mulliken populations per shell
    q_orbital: np.ndarray | None = None  # Mulliken populations per orbital
    occupations: list | None = None      # per k-point, incl. spin factor 2
    scc_error: float = 0.0               # max |q_out - q_in| of the last iteration
    e_3rd: float = 0.0                   # third-order (DFTB3) energy
    stress: np.ndarray | None = None     # (3, 3) Hartree/Bohr^3, DFTB+ sign convention
    kpoints: np.ndarray | None = None    # fractional k-points used
    kweights: np.ndarray | None = None
    # collinear spin: eigenvalues/occupations are stacked [up k-points..., down k-points...];
    # two-component (M6a): 2N levels per k-point, one Fermi level, fermi_spin None
    e_spin: float = 0.0
    magmoms: np.ndarray | None = None    # (n_atom,) N_up - N_down per atom; two-component:
                                         # (n_atom, 3) magnetisation vectors
    q_spin: np.ndarray | None = None     # (2, n_shell) up / down populations per shell;
                                         # two-component: (4, n_shell) q, m_x, m_y, m_z
    fermi_spin: tuple | None = None      # (mu_up, mu_down); equal with relax_total_spin
    e_disp: float = 0.0                  # dispersion energy
    q_in: np.ndarray | None = None       # input populations of the last SCC iteration (the mixed
                                         # vector; what DFTB+ writes to charges.bin)
    e_ls: float = 0.0                    # spin-orbit energy (DFTB+ "Energy L.S")
    orbital_moments: np.ndarray | None = None   # (n_shell, 3) <L> per shell (spin-orbit)

    @property
    def e_total(self) -> float:
        return self.energy

    @property
    def e_mermin(self) -> float:
        return self.free_energy

    @property
    def e_zero(self) -> float:
        return self.energy - 0.5 * self.ts

    @property
    def terms(self) -> dict:
        return {"H0": self.e_band, "SCC": self.e_scc, "3rd": self.e_3rd, "spin": self.e_spin,
                "LS": self.e_ls, "repulsive": self.e_rep, "dispersion": self.e_disp,
                "-TS": -self.ts}


@dataclass
class _Setup:
    """Everything that depends only on the geometry (fixed during SCC)."""
    species: list
    pos: np.ndarray
    cell: np.ndarray | None
    pbc: np.ndarray | None
    basis: Basis
    pairs: PairBlocks
    kpts: np.ndarray            # (nk, 3) fractional
    kw: np.ndarray              # (nk,) weights, sum 1
    gamma_only: bool
    h0: np.ndarray              # (nk, N, N)
    smat: np.ndarray            # (nk, N, N)
    site_of_orb: np.ndarray
    site_atom: np.ndarray
    q0: np.ndarray              # reference (neutral atom) populations per site
    nelec: float
    coulomb: Coulomb | None
    third: ThirdOrder | None = None
    spin: "_Spin | None" = None
    disp: DispersionTerms | None = None
    eps0: np.ndarray | None = None   # on-site energies per orbital
    dev: "_Device | None" = None     # torch backend: device copies (h0, smat above are on it)
    pauli: "_Pauli | None" = None    # two-component calculation (M6a)

    @property
    def kw_dev(self):
        """k-point weights next to the density matrices (the device with torch)."""
        return self.kw if self.dev is None else self.dev.kw

    @property
    def smat_precise(self):
        """The overlap stack in double precision (for the Cholesky factor)."""
        return self.smat if self.dev is None else self.dev.smat_precise


@dataclass
class _Device:
    """Torch backend: what the SCC loop needs on the device besides h0 and smat."""
    kw: object                  # k-point weights (working precision)
    smat_precise: object        # overlap stack in double precision


@dataclass
class _Spin:
    """Collinear spin: W over all shells (block-diagonal per atom) and electron counts."""
    w: np.ndarray               # (n_shell, n_shell)
    n_up: float
    n_down: float
    shared_ef: bool             # one Fermi level for both channels (DFTB+ RelaxTotalSpin)


@dataclass
class _Pauli:
    """Two-component calculation (M6a): non-collinear spin and/or spin-orbit coupling."""
    w: np.ndarray | None        # (n_shell, n_shell) spin constants; None: no spin polarisation
    spins: np.ndarray           # (n_atom, 3) initial magnetisation per atom
    lam: np.ndarray | None      # (3, N, N) Lambda^a = Im(xi/2 L_a); None: no spin-orbit coupling
    dual: bool                  # dual (DFTB+ default) or onsite spin-orbit model
    hso: np.ndarray | None      # H_SO(k), (nk or 1, 2N, 2N), constant during the SCC
    s2: np.ndarray | None       # (nk, 2N, 2N) overlap S x 1


@dataclass
class _SCCState:
    """Converged (or last) state of the SCC loop.

    ``q_in``/``q_out`` are the mixed vectors: site populations, or with spin, as in
    DFTB+, [up populations per shell, down populations per shell].  With spin the
    density matrices are stacked [up k-points, down k-points].
    """
    q_in: np.ndarray
    q_out: np.ndarray
    q_orb: np.ndarray           # total (up + down) population per orbital
    v_orb: np.ndarray           # charge (SCC + 3rd order) potential per orbital
    h: np.ndarray               # (n_spin * nk, N, N) final Hamiltonian
    rho: np.ndarray             # (n_spin * nk, N, N)
    wdm: np.ndarray | None      # energy-weighted density
    evals: np.ndarray | None
    occ: list | None
    mu: float | tuple
    ts: float
    n_iter: int
    converged: bool
    error: float
    vs_orb: np.ndarray | None = None   # spin potential per orbital (+ for up, - for down)


class DFTB:
    """SCC-DFTB calculator.

    Parameters
    ----------
    sk : SKSet            Slater-Koster data.
    scc : bool            Self-consistent charges.
    kpts : (nk,3), (nk,)  k-points in fractional reciprocal coords and weights
                          (weights normalised to 1).  ``None`` => Gamma.
    temperature : float   Electronic temperature (K) for Fermi smearing
                          (shortcut for ``smearing=("fermi", T)``).
    smearing : spec       ("fermi", T) | ("gaussian", T) | ("mp", T, order) or
                          an :class:`~pydftb.occupations.Smearing`.  T < 1e-8 Ha
                          is raised to 1e-8 Ha like DFTB+.
    mixer : str           "broyden" (DFTB+ default) or "pulay".
    solver : str | object "eigh" (reference), "batched" or "sp2" (see solvers.py).
    repulsive : str       "skf" (spline, else polynomial), "polynomial"
                          (DFTB+ ``PolynomialRepulsive = Yes``) or "none".
    warn_unconverged : bool  warn (but still return results, like DFTB+
                          ``ConvergentSCCOnly = No``) if the SCC does not converge.
    fd_delta : float      None (default): exact H0/S derivatives.  A step (Bohr)
                          reproduces DFTB+'s finite-difference derivatives
                          (current DFTB+ default 1.22e-4; ParserVersion < 5,
                          e.g. via ASE: 1e-2).
    third_order : str     None (DFTB2), "full" (DFTB3, DFTB+ ``ThirdOrderFull``) or
                          "onsite" (DFTB+ ``ThirdOrder``: R = 0 terms only).
    hubbard_derivs : dict type label -> Hubbard derivative Ud (Hartree/e), or a
                          list with one value per shell for orbital-resolved SCC
                          (DFTB+ ``HubbardDerivs``).
    h_damping : float     exponent zeta of the H-X damping of gamma (and Gamma),
                          DFTB+ ``HCorrection = Damping { Exponent = zeta }``;
                          damped are atoms lighter than 3.5 amu, as in DFTB+.
    spin : bool | str     collinear spin polarisation (True or "colinear", DFTB+
                          ``SpinPolarisation = Colinear``) or "noncolinear" (two-component,
                          DFTB+ ``NonColinear``; milestone M6a).
    unpaired_electrons : float  N_up - N_down (DFTB+ ``UnpairedElectrons``); fixed,
                          with one Fermi level per spin, unless ``relax_total_spin``.
    relax_total_spin : bool  one Fermi level for both spins (DFTB+ ``RelaxTotalSpin``).
    spin_constants : dict type label -> W (Hartree): one number for all shell pairs,
                          or a (n_shell, n_shell) matrix (DFTB+ ``SpinConstants``).
    dispersion : dict | Dispersion  dispersion correction (DFTB+ ``Dispersion``), e.g.
                          ``{"model": "dftd3", "damping": "beckejohnson", "a1": 0.5719,
                          "a2": 3.6017, "s6": 1.0, "s8": 0.5883}``; see
                          :mod:`pydftb.dispersion`.
    k_threads : int | "auto" | KThreads  threads over chunks of k-points for the
                          eigensolver and density matrices (:mod:`pydftb.parallel`);
                          bitwise identical results for any number.  With more than one,
                          BLAS runs on one thread during the calculation.
    backend : str | Backend  "numpy" (default) or "torch": the folding to k-points, the
                          eigensolver, density matrices, populations, energy traces and the
                          real-space densities of the forces run on a torch ``device``
                          (:mod:`pydftb.backend.spec`); the rest stays NumPy.
    device : str          torch device ("cpu", "cuda", ...); torch backend only.
    dtype : str           "float64" (default) or "float32" (torch backend only: H, the
                          eigensolver and the densities in single precision).
    initial_spins : (n_atom, 3)  initial magnetisation vector of every atom with
                          ``spin="noncolinear"`` (DFTB+ ``InitialSpins``); default zero.
    spin_orbit : dict     type label -> spin-orbit constant xi (Hartree) of every shell, zero
                          for s shells (DFTB+ ``SpinOrbit``); types with only s shells may be
                          left out.  Two-component: with ``spin="noncolinear"`` or without
                          spin polarisation; with a k-point set that contains -k for every k.
    spin_orbit_dual : bool  the dual model (default, DFTB+ ``Dual = Yes``), where the term
                          acts through the overlap like a block potential, or the onsite model.

    Two-component calculations are NumPy only (not ``backend="torch"``, the SP2 solver or
    :mod:`pydftb.autodiff`).
    """

    def __init__(self, sk: SKSet, scc: bool = True, kpts=None, temperature: float = 0.0,
                 scc_tol: float = 1e-8, max_scc: int = 200, mixing: float = 0.2,
                 charge: float = 0.0, solver: str = "eigh", verbose: bool = False,
                 repulsive: str = "skf", smearing=None, mixer: str = "broyden",
                 warn_unconverged: bool = True, fd_delta: float | None = None,
                 third_order: str | None = None, hubbard_derivs: dict | None = None,
                 h_damping: float | None = None, spin: bool = False,
                 unpaired_electrons: float = 0.0, relax_total_spin: bool = False,
                 spin_constants: dict | None = None, dispersion=None, k_threads=1,
                 backend=None, device: str | None = None, dtype: str | None = None,
                 initial_spins=None, spin_orbit: dict | None = None,
                 spin_orbit_dual: bool = True):
        noncolinear = isinstance(spin, str) and spin.lower() in ("noncolinear", "noncollinear")
        if isinstance(spin, str) and not noncolinear:
            if spin.lower() not in ("colinear", "collinear"):
                raise ValueError(f"spin must be False, True, 'colinear' or 'noncolinear', "
                                 f"not {spin!r}")
            spin = True
        if noncolinear:
            spin = False
            self._check_noncolinear(scc, spin_constants, solver, unpaired_electrons,
                                    relax_total_spin)
        elif initial_spins is not None:
            raise ValueError("initial_spins needs spin='noncolinear'")
        if spin_orbit is not None and spin:
            raise ValueError("spin-orbit coupling needs spin='noncolinear' or no spin "
                             "polarisation (as in DFTB+)")
        if spin_orbit is not None and solver == "sp2":
            raise ValueError("the SP2 solver does not support spin-orbit coupling")
        if spin and not scc:
            raise ValueError("spin polarisation needs SCC")
        if spin and spin_constants is None:
            raise ValueError("spin polarisation needs spin_constants")
        if spin and solver == "sp2":
            raise ValueError("the SP2 solver does not support spin polarisation")
        if third_order not in (None, "full", "onsite"):
            raise ValueError(f"third_order must be None, 'full' or 'onsite', not {third_order!r}")
        if third_order and not scc:
            raise ValueError("third_order needs SCC")
        if third_order and hubbard_derivs is None:
            raise ValueError("third_order needs hubbard_derivs")
        if third_order == "onsite" and sk.shell_resolved:
            raise ValueError("third_order='onsite' needs atom-resolved SCC (as in DFTB+)")
        self.sk = sk
        self.scc = scc
        self.kpts = kpts
        self.smearing = Smearing.from_spec(smearing if smearing is not None else ("fermi", temperature))
        self.scc_tol = scc_tol
        self.max_scc = max_scc
        self.mixing = mixing
        self.mixer = mixer
        self.charge = charge
        self.solver = solver
        self.verbose = verbose
        self.repulsive = repulsive
        self.warn_unconverged = warn_unconverged
        self.fd_delta = fd_delta
        self.third_order = third_order
        self.hubbard_derivs = hubbard_derivs
        self.h_damping = h_damping
        self.spin = spin
        self.unpaired_electrons = unpaired_electrons
        self.relax_total_spin = relax_total_spin
        self.spin_constants = spin_constants
        self.dispersion = make_dispersion(dispersion)
        self.k_threads = as_kthreads(k_threads)
        self.noncolinear = noncolinear
        self.initial_spins = initial_spins
        self.spin_orbit = None if spin_orbit is None else dict(spin_orbit)
        self.spin_orbit_dual = bool(spin_orbit_dual)
        if self.spin_orbit is not None:      # the types of this SK set (the rest at setup)
            so.check_xi(self.spin_orbit, sk, [lab for lab in self.spin_orbit if lab in sk.onsite])
        self.backend = make_backend(backend, device, dtype)
        if self.backend.is_torch and self.pauli:
            raise ValueError("non-collinear spin and spin-orbit coupling are implemented for "
                             "the NumPy backend only (milestone M6a)")
        if self.backend.is_torch:
            if solver not in ("eigh", "batched"):
                raise ValueError("the torch backend has its own eigensolver: use solver='eigh' "
                                 f"or 'batched', not {solver!r}")
            if self.k_threads.n > 1:
                warnings.warn("k-point threads are not used with the torch backend",
                              RuntimeWarning, stacklevel=2)
                self.k_threads = as_kthreads(1)
            if self.backend.dtype == "float32" and scc and scc_tol < 1e-6:
                warnings.warn(f"scc_tol = {scc_tol:g} is below what float32 can resolve "
                              "(about 1e-6); the SCC may not converge", RuntimeWarning,
                              stacklevel=2)

    @staticmethod
    def _check_noncolinear(scc, spin_constants, solver, unpaired_electrons, relax_total_spin):
        if not scc:
            raise ValueError("spin polarisation needs SCC")
        if spin_constants is None:
            raise ValueError("spin polarisation needs spin_constants")
        if solver == "sp2":
            raise ValueError("the SP2 solver does not support spin polarisation")
        if unpaired_electrons or relax_total_spin:
            raise ValueError("unpaired_electrons and relax_total_spin are for collinear spin: "
                             "a non-collinear calculation has one Fermi level and a free "
                             "total magnetisation (set initial_spins instead)")

    @property
    def pauli(self) -> bool:
        """Whether the calculation is two-component (non-collinear spin or spin-orbit)."""
        return self.noncolinear or self.spin_orbit is not None

    @property
    def kt(self) -> float:
        return self.smearing.kt

    # ------------------------------------------------------------------
    def calculate(self, species, positions, cell=None, pbc=None, forces=True,
                  q_init=None, stress: bool | None = None) -> DFTBResult:
        """Run a calculation. Positions/cell in Bohr.

        ``stress`` (default: with forces for periodic systems) adds the stress
        tensor sigma = -(1/V) dE/d(strain), the sign convention of DFTB+.
        """
        if self.backend.is_torch:                # CUDA: deterministic algorithms (M4d)
            from .backend.devices import deterministic
            with deterministic(self.backend):
                return self._calculate(species, positions, cell, pbc, forces, q_init, stress)
        if self.k_threads.n == 1:
            return self._calculate(species, positions, cell, pbc, forces, q_init, stress)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1, user_api="blas"):    # the k threads take the cores
            return self._calculate(species, positions, cell, pbc, forces, q_init, stress)

    def _calculate(self, species, positions, cell, pbc, forces, q_init, stress) -> DFTBResult:
        return self._run(species, positions, cell, pbc, forces, q_init, stress)[0]

    def _run(self, species, positions, cell, pbc, forces, q_init, stress):
        """The calculation of :meth:`calculate`, also returning its setup and SCC state
        (``pydftb.autodiff`` builds on them)."""
        t0 = time.perf_counter()
        timings = {}
        periodic = cell is not None and (pbc is None or np.any(pbc))
        if stress is None:
            stress = forces and periodic
        if stress and not periodic:
            raise ValueError("stress needs a periodic system")
        need_derivs = forces or stress
        setup = self._setup(species, positions, cell, pbc, need_derivs, timings)
        state = self._scc(setup, q_init, need_derivs, timings)
        res = self._energies(setup, state)
        if need_derivs:
            t1 = time.perf_counter()
            grad, virial = self._gradient(setup, state, virial=stress)
            if forces:
                res.forces = -grad
            if stress:
                vol = abs(np.linalg.det(setup.cell))
                res.stress = -virial / vol
            timings["forces"] = time.perf_counter() - t1
        timings["total"] = time.perf_counter() - t0
        res.timings = timings
        return res, setup, state

    # ------------------------------------------------------------------
    def _sites(self, basis: Basis):
        """Charge sites: atoms, or shells when orbital-resolved."""
        sk = self.sk
        if sk.shell_resolved:
            site_of_orb = basis.orb_shell
            site_atom = basis.shell_atom
            u = np.array([sk.hubbard[basis.species[a]][l]
                          for a, l in zip(basis.shell_atom, basis.shell_l)])
            q0 = np.array([sk.occupation[basis.species[a]][l]
                           for a, l in zip(basis.shell_atom, basis.shell_l)])
        else:
            site_of_orb = basis.orb_atom
            site_atom = np.arange(len(basis.species))
            u = np.array([sk.hubbard[s][0] for s in basis.species])
            q0 = np.array([sk.occupation[s].sum() for s in basis.species])
        return site_of_orb, site_atom, u, q0

    def _setup(self, species, positions, cell, pbc, forces, timings, matrices=True) -> _Setup:
        """``matrices=False`` skips the folded H0/S on the k-point set and the dispersion
        (``bands.eigenstates`` folds at its own k-points)."""
        t0 = time.perf_counter()
        sk = self.sk
        pos = np.asarray(positions, float)
        if cell is not None:
            cell = np.asarray(cell, float)
            pbc = np.ones(3, bool) if pbc is None else np.asarray(pbc, bool)
        basis = Basis.build(species, sk)
        norb = basis.norb
        pb = build_pairs(species, pos, cell, pbc, sk, basis, derivatives=forces,
                         fd_delta=self.fd_delta)
        eps0 = onsite_energies(basis, sk)
        site_of_orb, site_atom, hub, q0 = self._sites(basis)
        periodic = cell is not None and np.any(pbc)
        if periodic and self.kpts is not None:
            kpts, kw = self.kpts
            kpts = np.atleast_2d(np.asarray(kpts, float))
            kw = np.asarray(kw, float) / np.sum(kw)
        else:
            kpts, kw = np.zeros((1, 3)), np.ones(1)
        gamma_only = len(kpts) == 1 and not np.any(kpts[0])
        b = self.backend
        dev = None
        if not matrices:
            h0 = smat = None
        elif gamma_only:
            h0 = fold(pb, pb.h, norb, None, eps0)[None]
            smat = fold(pb, pb.s, norb, None, np.ones(norb))[None]
        else:
            h0 = fold_all(pb, pb.h, norb, kpts, eps0, backend=b)
            smat = fold_all(pb, pb.s, norb, kpts, np.ones(norb), backend=b)
        if b.is_torch and matrices:
            s_precise = b.asarray(smat, precise=True)
            dev = _Device(kw=b.asarray(kw), smat_precise=s_precise)
            h0, smat = b.cast(b.asarray(h0, precise=True)), b.cast(s_precise)
        timings["build"] = time.perf_counter() - t0
        coul = third = spin = None
        if self.scc:
            t1 = time.perf_counter()
            damped = None
            if self.h_damping is not None:
                damped = np.array([sk.mass[lab] < 3.5 for lab in species])   # H, as in DFTB+
            coul = Coulomb(pos, site_atom, hub, cell if periodic else None,
                           pbc if periodic else None, damped=damped,
                           damp_exp=self.h_damping or 0.0)
            if self.third_order:
                third = ThirdOrder(coul, self._site_hubbard_derivs(basis),
                                   onsite_only=self.third_order == "onsite")
            timings["gamma"] = time.perf_counter() - t1
        nelec = q0.sum() - self.charge
        if self.spin:
            spin = _Spin(self._spin_matrix(basis), 0.5 * (nelec + self.unpaired_electrons),
                         0.5 * (nelec - self.unpaired_electrons), self.relax_total_spin)
        pauli = None
        if self.pauli:
            pauli = self._pauli_setup(basis, smat, kpts, kw, periodic)
        disp = None
        if self.dispersion is not None and matrices:
            t1 = time.perf_counter()
            disp = self.dispersion.evaluate(species, pos, cell if periodic else None,
                                            pbc if periodic else None, grad=forces,
                                            charge=self.charge)
            timings["dispersion"] = time.perf_counter() - t1
        return _Setup(list(species), pos, cell, pbc, basis, pb, kpts, kw, gamma_only, h0, smat,
                      site_of_orb, site_atom, q0, nelec, coul, third, spin, disp, eps0, dev,
                      pauli=pauli)

    def _pauli_setup(self, basis: Basis, smat, kpts, kw, periodic) -> _Pauli:
        """Spin constants, initial spins, the spin-orbit Lambda^a and H_SO(k) (M6a)."""
        natom = len(basis.species)
        spins = (np.zeros((natom, 3)) if self.initial_spins is None
                 else np.array(self.initial_spins, float))
        if spins.shape != (natom, 3):
            raise ValueError(f"initial_spins needs one (x, y, z) vector per atom, "
                             f"shape ({natom}, 3), not {spins.shape}")
        w = self._spin_matrix(basis) if self.noncolinear else None
        lam = hso = s2 = None
        if self.spin_orbit is not None:
            if periodic and not so.closed_under_inversion(kpts, kw):
                raise ValueError("spin-orbit coupling needs a k-point set with -k for every k "
                                 "(H(-k) is not H(k)*): do not reduce the grid by time reversal")
            lam = so.lambda_matrices(basis, so.check_xi(self.spin_orbit, self.sk,
                                                        basis.species))
        if smat is not None:
            s2 = so.overlap2(smat)
            if lam is not None:
                hso = so.soc_hamiltonian(lam, smat if self.spin_orbit_dual else None)
        return _Pauli(w, spins, lam, self.spin_orbit_dual, hso, s2)

    def _site_hubbard_derivs(self, basis: Basis) -> np.ndarray:
        """Ud of every charge site: one value per type, or per shell (orbital-resolved SCC)."""
        missing = sorted(set(basis.species) - set(self.hubbard_derivs))
        if missing:
            raise ValueError(f"hubbard_derivs has no entry for {missing}")
        ud = {lab: np.atleast_1d(np.asarray(v, float)) for lab, v in self.hubbard_derivs.items()}
        if not self.sk.shell_resolved:
            return np.array([ud[lab][0] for lab in basis.species])
        out = []
        for a, l in zip(basis.shell_atom, basis.shell_l):
            v = ud[basis.species[a]]
            out.append(v[0] if len(v) == 1 else v[self.sk.shells(basis.species[a]).index(l)])
        return np.array(out)

    def _spin_matrix(self, basis: Basis) -> np.ndarray:
        """Spin constants over all shells: block-diagonal, one (n_shell, n_shell) block per atom."""
        missing = sorted(set(basis.species) - set(self.spin_constants))
        if missing:
            raise ValueError(f"spin_constants has no entry for {missing}")
        nsh = len(basis.shell_l)
        w = np.zeros((nsh, nsh))
        for a, lab in enumerate(basis.species):
            idx = np.nonzero(basis.shell_atom == a)[0]
            v = np.asarray(self.spin_constants[lab], float)
            if v.size == 1:                       # DFTB+ ShellResolvedSpin = No
                blk = np.full((len(idx), len(idx)), float(v.ravel()[0]))
            elif v.size == len(idx) ** 2:
                blk = v.reshape(len(idx), len(idx))
            else:
                raise ValueError(f"spin_constants[{lab!r}] needs 1 or {len(idx)}x{len(idx)} values")
            w[np.ix_(idx, idx)] = blk
        return w

    # ---- the mixed vector: site populations, or [up, down] per shell with spin ----
    def _initial_populations(self, st: _Setup) -> np.ndarray:
        """DFTB+ start: reference populations rescaled to the electron count; with spin
        every shell also gets m = q * (N_up - N_down) / N (DFTB+ initializeCharges)."""
        scale = st.nelec / st.q0.sum() if st.q0.sum() > 0 else 1.0
        if st.pauli is not None:
            # two-component: [q, m_x, m_y, m_z] per shell; an atom's initial spin vector is
            # shared out in proportion to its shell populations (DFTB+ initializeCharges)
            b = st.basis
            q = np.array([self.sk.occupation[b.species[a]][l]
                          for a, l in zip(b.shell_atom, b.shell_l)]) * scale
            q_atom = ops.segment_sum(q, b.shell_atom, len(b.species))[b.shell_atom]
            share = np.divide(q, q_atom, out=np.zeros_like(q), where=q_atom != 0)
            m = st.pauli.spins[b.shell_atom].T * share
            return np.concatenate([q, m.ravel()])
        if st.spin is None:
            return st.q0 * scale
        b = st.basis
        q = np.array([self.sk.occupation[b.species[a]][l]
                      for a, l in zip(b.shell_atom, b.shell_l)]) * scale
        m = q * (st.spin.n_up - st.spin.n_down) / st.nelec
        return np.concatenate([0.5 * (q + m), 0.5 * (q - m)])

    def _charge_and_magnetisation(self, st: _Setup, qvec):
        """Site population deviations dq and shell magnetisations m (None without spin;
        (3, n_shell) for two-component calculations)."""
        if st.pauli is not None:
            nsh = len(st.basis.shell_l)
            q = qvec[:nsh]
            if not self.sk.shell_resolved:
                q = ops.segment_sum(q, st.basis.shell_atom, len(st.species))
            return q - st.q0, qvec[nsh:].reshape(3, nsh)
        if st.spin is None:
            return qvec - st.q0, None
        nsh = len(st.basis.shell_l)
        up, down = qvec[:nsh], qvec[nsh:]
        q = up + down
        if not self.sk.shell_resolved:
            q = ops.segment_sum(q, st.basis.shell_atom, len(st.species))
        return q - st.q0, up - down

    def _potentials(self, st: _Setup, qvec):
        """Charge potential (SCC + third order) and spin potential, per orbital."""
        if not self.scc:
            return np.zeros(st.basis.norb), None
        dq, m = self._charge_and_magnetisation(st, qvec)
        _, v_site = st.coulomb.energy_shift(dq)
        if st.third is not None:
            v_site = v_site + st.third.energy_shift(dq)[1]
        if st.pauli is not None:        # (3, N): W m_a for a = x, y, z
            w = st.pauli.w
            return v_site[st.site_of_orb], None if w is None else (w @ m.T).T[:, st.basis.orb_shell]
        vs_orb = None if m is None else (st.spin.w @ m)[st.basis.orb_shell]
        return v_site[st.site_of_orb], vs_orb

    @staticmethod
    def _spinor_hamiltonian(h0, smat, v_orb, vs_orb, hso):
        """The two-component H: H_q x 1 + sum_a H_a sigma_a + H_SO, with H_q = H0 +
        S (v_mu + v_nu)/2 and H_a = S (v^a_mu + v^a_nu)/2 (DFTB+ unpackHPauli)."""
        hq = h0 + 0.5 * smat * (v_orb[:, None] + v_orb[None, :])
        if vs_orb is None:
            h = so.spinor(hq, 0.0, 0.0, 0.0)
        else:
            h = so.spinor(hq, *(0.5 * smat * (v[:, None] + v[None, :]) for v in vs_orb))
        return h if hso is None else h + hso

    def _solve(self, solver, st: _Setup, h, need_w):
        """Densities for H, stacked [up k-points, down k-points] with spin; one Fermi
        level per spin channel, or a shared one (DFTB+ RelaxTotalSpin).  Two-component:
        one Fermi level, at most one electron per spinor level."""
        if st.pauli is not None:
            e, c = solver.eig(h)
            rho, wdm, evals, mu, ts, occ = _density(e, c, st.kw, st.nelec, self.smearing, need_w,
                                                    spin_factor=1.0,
                                                    threads=getattr(solver, "threads", None))
            return rho, wdm, evals, mu, ts, list(occ)
        if st.spin is None:
            rho, wdm, evals, mu, ts = solver.solve(h, st.kw, st.nelec, self.smearing, need_w=need_w)
            return rho, wdm, evals, mu, ts, getattr(solver, "last_occupations", None)
        e, c = solver.eig(h)
        nk = len(st.kw)
        threads = getattr(solver, "threads", None)
        b = self.backend
        if st.spin.shared_ef:
            rho, wdm, evals, mu, ts, occ = _density(e, c, np.tile(st.kw, 2), st.nelec,
                                                    self.smearing, need_w, spin_factor=1.0,
                                                    threads=threads, backend=b)
            return rho, wdm, evals, (mu, mu), ts, list(occ)
        parts = [_density(e[s], c[s], st.kw, n, self.smearing, need_w, spin_factor=1.0,
                          threads=threads, backend=b)
                 for s, n in ((slice(0, nk), st.spin.n_up), (slice(nk, 2 * nk), st.spin.n_down))]
        rho = b.xp.concatenate([p[0] for p in parts])
        wdm = b.xp.concatenate([p[1] for p in parts]) if need_w else None
        evals = np.concatenate([p[2] for p in parts])
        occ = list(parts[0][5]) + list(parts[1][5])
        return rho, wdm, evals, (parts[0][3], parts[1][3]), parts[0][4] + parts[1][4], occ

    def _populations(self, st: _Setup, rho):
        """Mulliken populations per orbital (up + down) and the vector to mix."""
        nk = len(st.kw)
        if st.pauli is not None:        # [q, m_x, m_y, m_z] per shell
            q = [self._mulliken(st, c) for c in so.components(rho)]
            shells, nsh = st.basis.orb_shell, len(st.basis.shell_l)
            return q[0], np.concatenate([ops.segment_sum(x, shells, nsh) for x in q])
        # q_mu = sum_k w_k Re (rho_k S_k)_{mu mu}
        if st.spin is None:
            q_orb = self._mulliken(st, rho)
            return q_orb, ops.segment_sum(q_orb, st.site_of_orb, len(st.q0))
        q_up = self._mulliken(st, rho[:nk])
        q_dn = self._mulliken(st, rho[nk:])
        shells, nsh = st.basis.orb_shell, len(st.basis.shell_l)
        return q_up + q_dn, np.concatenate([ops.segment_sum(q_up, shells, nsh),
                                            ops.segment_sum(q_dn, shells, nsh)])

    def _mulliken(self, st: _Setup, rho) -> np.ndarray:
        """sum_k w_k Re (rho_k S_k)_{mu mu} per orbital, as a host array."""
        if not self.backend.is_torch:
            return np.einsum("k,kij,kji->i", st.kw, rho, st.smat).real
        b = self.backend
        prod = b.widen(rho * st.smat.mT)                 # sums in double precision
        return b.to_host((b.widen(st.kw_dev)[:, None, None] * prod).sum(dim=(0, 2)).real)

    def _trace(self, st: _Setup, rho, m) -> float:
        """sum_k w_k Re Tr(rho_k M_k)."""
        if not self.backend.is_torch:
            return float(np.einsum("k,kij,kji->", st.kw, rho, m).real)
        b = self.backend
        return float((b.widen(st.kw_dev)[:, None, None] * b.widen(rho * m.mT)).sum().real)

    def _scc(self, st: _Setup, q_init, forces, timings) -> _SCCState:
        b = self.backend
        if b.is_torch:
            from .backend.torch_solver import TorchSolver
            solver = TorchSolver(b)
        else:
            solver = (self.solver if not isinstance(self.solver, str)
                      else SOLVERS[self.solver](threads=self.k_threads if self.k_threads.n > 1 else None))
        t1 = time.perf_counter()
        s = st.smat_precise
        if st.pauli is not None:
            solver.prepare(st.pauli.s2)
        else:
            solver.prepare(s if st.spin is None else b.xp.concatenate([s, s]))
        t_diag = time.perf_counter() - t1
        mixer = make_mixer(self.mixer, self.mixing)
        q_in = (np.asarray(q_init, float).copy() if q_init is not None
                else self._initial_populations(st))
        converged = not self.scc
        nit = 0
        diff = 0.0
        # DFTB+: the first input guess of a spin calculation is not mixed in, and a
        # two-component SCC needs at least two iterations (unless charges are given)
        first_out = (st.spin is not None or st.pauli is not None) and q_init is None
        min_iter = 2 if st.pauli is not None and q_init is None else 1
        for nit in range(1, (self.max_scc if self.scc else 1) + 1):
            v_orb, vs_orb = self._potentials(st, q_in)
            if st.pauli is not None:
                h = self._spinor_hamiltonian(st.h0, st.smat, v_orb, vs_orb, st.pauli.hso)
            else:
                v = b.asarray(v_orb)
                h = st.h0 + 0.5 * st.smat * (v[:, None] + v[None, :])
                if vs_orb is not None:
                    vs = b.asarray(vs_orb)
                    hs = 0.5 * st.smat * (vs[:, None] + vs[None, :])
                    h = b.xp.concatenate([h + hs, h - hs])          # up, down
            t1 = time.perf_counter()
            rho, wdm, evals, mu, ts, occ = self._solve(solver, st, h, need_w=False)
            t_diag += time.perf_counter() - t1
            q_orb, q_out = self._populations(st, rho)
            if not self.scc:
                break
            diff = np.abs(q_out - q_in).max()
            if self.verbose:
                print(f"SCC {nit:3d}  max dq = {diff:.3e}")
            if diff < self.scc_tol and nit >= min_iter:
                converged = True
                break
            if nit == 1 and first_out:
                q_in = q_out            # DFTB+: the guessed spin is not mixed in (main.F90)
            else:
                q_in = mixer.mix(q_in, q_out)
        if forces:
            # W is only needed once, for the final density
            t1 = time.perf_counter()
            rho, wdm, evals, mu, ts, occ = self._solve(solver, st, h, need_w=True)
            t_diag += time.perf_counter() - t1
        timings["diag"] = t_diag
        timings["scc_iter"] = nit
        if self.scc and not converged and self.warn_unconverged:
            warnings.warn(f"SCC not converged after {nit} iterations "
                          f"(max |dq| = {diff:.2e} > {self.scc_tol:.1e}); returning the last "
                          "iteration's results", RuntimeWarning, stacklevel=3)
        return _SCCState(q_in, q_out, q_orb, v_orb, h, rho, wdm, evals, occ, mu, ts, nit,
                         converged, diff, vs_orb)

    def _energies_pauli(self, st: _Setup, sc: _SCCState) -> DFTBResult:
        """Energies and populations of a two-component calculation (from the last output,
        as DFTB+): E_H0 = Tr(n H0), E_spin = 1/2 sum_a m_a W m_a, E_LS = Tr(rho H_SO)."""
        p = st.pauli
        comp = so.components(sc.rho)
        e_band = self._trace(st, comp[0], st.h0)
        dq, m = self._charge_and_magnetisation(st, sc.q_out)
        e_scc = st.coulomb.energy_shift(dq)[0] if self.scc else 0.0
        e_3rd = st.third.energy_shift(dq)[0] if st.third is not None else 0.0
        e_spin = 0.5 * float(np.einsum("as,st,at->", m, p.w, m)) if p.w is not None else 0.0
        e_ls, lmom = 0.0, None
        if p.hso is not None:
            e_ls = float(np.einsum("k,kij,kji->", st.kw, sc.rho,
                                   np.broadcast_to(p.hso, sc.rho.shape)).real)
            lmom = self._orbital_moments(st, comp)
        e_rep = self._repulsive_energy(st)
        e_disp = st.disp.energy if st.disp is not None else 0.0
        energy = e_band + e_scc + e_3rd + e_spin + e_ls + e_rep + e_disp
        b, natom = st.basis, len(st.species)
        q_atom = ops.segment_sum(sc.q_orb, b.orb_atom, natom)
        q_shell = ops.segment_sum(sc.q_orb, b.orb_shell, len(b.shell_l))
        magmoms = np.stack([ops.segment_sum(m[a], b.shell_atom, natom) for a in range(3)], axis=1)
        occ = None if sc.occ is None else [np.asarray(o) for o in sc.occ]
        return DFTBResult(energy, energy - sc.ts, e_band, e_scc, e_rep, sc.ts, sc.mu, q_atom,
                          None if sc.evals is None else list(sc.evals), n_scc=sc.n_iter,
                          converged=sc.converged, q_shell=q_shell, q_orbital=sc.q_orb,
                          occupations=occ, scc_error=sc.error, kpoints=st.kpts,
                          kweights=st.kw, e_3rd=e_3rd, e_disp=e_disp, q_in=sc.q_in,
                          e_spin=e_spin, magmoms=magmoms, q_spin=sc.q_out.reshape(4, -1),
                          e_ls=e_ls, orbital_moments=lmom)

    def _orbital_moments(self, st: _Setup, comp) -> np.ndarray:
        """<L> per shell from the imaginary part of the charge density's atom blocks (onsite
        model, DFTB+ getLOnsite) or of its dual populations (dual model, getLDual)."""
        b = st.basis
        if st.pauli.dual:
            x = [blk[0].imag for blk in so.block_populations(comp[:1], st.smat, st.kw, b)]
        else:
            n0 = np.einsum("k,kij->ij", st.kw, comp[0])
            x = [n0[o:o + n, o:o + n].imag for o, n in zip(b.offsets, b.norb_atom)]
        return so.orbital_moments(so.l_imag(b), x, b)

    def _energies(self, st: _Setup, sc: _SCCState) -> DFTBResult:
        if st.pauli is not None:
            return self._energies_pauli(st, sc)
        nk = len(st.kw)
        e_band = sum(self._trace(st, sc.rho[s:s + nk], st.h0)
                     for s in range(0, len(sc.rho), nk))       # Tr(rho H0), summed over spins
        dq, m = self._charge_and_magnetisation(st, sc.q_out)
        e_scc = st.coulomb.energy_shift(dq)[0] if self.scc else 0.0
        e_3rd = st.third.energy_shift(dq)[0] if st.third is not None else 0.0
        e_spin = 0.5 * m @ st.spin.w @ m if m is not None else 0.0
        e_rep = self._repulsive_energy(st)
        e_disp = st.disp.energy if st.disp is not None else 0.0
        energy = e_band + e_scc + e_3rd + e_spin + e_rep + e_disp
        q_atom = ops.segment_sum(sc.q_orb, st.basis.orb_atom, len(st.species))
        q_shell = ops.segment_sum(sc.q_orb, st.basis.orb_shell, len(st.basis.shell_l))
        occ = None if sc.occ is None else [np.asarray(o) for o in sc.occ]
        mu, extra = sc.mu, {}
        if st.spin is not None:
            mu = sc.mu[0]
            extra = dict(e_spin=e_spin, q_spin=sc.q_out.reshape(2, -1), fermi_spin=tuple(sc.mu),
                         magmoms=ops.segment_sum(m, st.basis.shell_atom, len(st.species)))
        return DFTBResult(energy, energy - sc.ts, e_band, e_scc, e_rep, sc.ts, mu, q_atom,
                          None if sc.evals is None else list(sc.evals), n_scc=sc.n_iter,
                          converged=sc.converged, q_shell=q_shell, q_orbital=sc.q_orb,
                          occupations=occ, scc_error=sc.error, kpoints=st.kpts,
                          kweights=st.kw, e_3rd=e_3rd, e_disp=e_disp, q_in=sc.q_in, **extra)

    def _gradient(self, st: _Setup, sc: _SCCState, virial: bool = False):
        """dE/dR (n_atom, 3) and, if requested, the virial dE/d(strain) (3, 3).

        Every pair term contributes (dE/d r_pair) (x) r_pair to the virial.  As in DFTB+
        (``RecomputeAfterDensity = Yes``, its default) the charge terms use the output
        populations of the last iteration and their potentials: with a loose SCC
        tolerance (PTBP: 1e-5) this keeps the forces equal to DFTB+'s to ~1e-11 Ha/Bohr,
        where the input populations differ by ~|q_out - q_in| (M2f report).
        """
        pb = st.pairs
        v_orb, vs_orb = self._potentials(st, sc.q_out)
        grad, w_rep = self._repulsive_gradient(st)
        if st.pauli is not None:
            p_e, w_e, p_m, c_soc = self._pauli_pair_densities(st, sc)
        else:
            p_e, w_e, p_m = self._pair_densities(st, sc)
        vsum = 0.5 * (v_orb[pb.rows] + v_orb[pb.cols])
        coef_s = p_e * vsum - w_e
        if st.pauli is not None:
            # spin potentials of the three directions; the dual spin-orbit term's dS part
            if p_m is not None:
                for a in range(3):
                    coef_s = coef_s + p_m[a] * 0.5 * (vs_orb[a][pb.rows] + vs_orb[a][pb.cols])
            if c_soc is not None:
                coef_s = coef_s + c_soc
        elif p_m is not None:
            coef_s = coef_s + p_m * 0.5 * (vs_orb[pb.rows] + vs_orb[pb.cols])
        ge = p_e[:, None] * pb.dh + coef_s[:, None] * pb.ds   # dE/dvec per element
        gp = ops.segment_sum(ge, pb.pair_of, len(pb.i))          # per pair
        natom = len(st.pos)
        grad = grad + ops.segment_sum(gp, pb.j, natom) - ops.segment_sum(gp, pb.i, natom)
        dq, _ = self._charge_and_magnetisation(st, sc.q_out)
        if self.scc:
            g_scc, w_scc = st.coulomb.gradient(dq, virial=virial)
            grad = grad + g_scc                                    # already per atom
        if st.third is not None:
            g_3rd, w_3rd = st.third.gradient(dq, virial=virial)
            grad = grad + g_3rd
        if st.disp is not None:
            grad = grad + st.disp.gradient
        if not virial:
            return grad, None
        w = gp.T @ pb.vec + w_rep                                  # band + SCC shift + repulsive
        if self.scc:
            w = w + w_scc
        if st.third is not None:
            w = w + w_3rd
        if st.disp is not None:
            w = w + st.disp.virial
        return grad, 0.5 * (w + w.T)

    def _pair_densities(self, st: _Setup, sc: _SCCState):
        """Real-space density, energy-weighted density and (with spin) magnetisation
        density at every pair element, summed over spins and k (host arrays)."""
        pb = st.pairs
        nk = len(st.kw)
        nspin = len(sc.rho) // nk
        kw = np.tile(st.kw, nspin)                        # weights of the stacked channels
        sgn = np.repeat([1.0, -1.0][:nspin], nk)          # up - down: magnetisation density
        # real-space (energy-weighted) density at the pair elements, summed over spins:
        # P_e = Re sum_k w_k rho_k[nu, mu] exp(2 pi i k.T)
        # with the torch backend the densities are on the device: only the pair elements
        # come back to the host (b.* are identities for NumPy)
        b = self.backend
        ci, ri = b.index(pb.cols), b.index(pb.rows)
        p_m = None
        if st.gamma_only:
            kwd = b.asarray(kw)
            rho_e = sc.rho[:, ci, ri].real
            p_e = b.to_host(kwd @ rho_e)
            w_e = b.to_host(kwd @ sc.wdm[:, ci, ri].real)
            if nspin == 2:
                p_m = b.to_host(b.asarray(kw * sgn) @ rho_e)
        else:
            # group by translation T: R_T = sum_k w_k e^{2 pi i k.T} rho_k  (nT GEMM)
            shifts, t_of = shift_index(pb)
            ph = (np.exp(2j * np.pi * (np.tile(st.kpts, (nspin, 1)) @ shifts.T)) * kw[:, None]).T
            norb = st.basis.norb
            n2 = norb * norb
            ti = b.index(t_of)
            phd = b.asarray(ph)
            rho2 = sc.rho.reshape(len(kw), n2)
            r_t = (phd @ rho2).reshape(-1, norb, norb)
            w_t = (phd @ sc.wdm.reshape(len(kw), n2)).reshape(-1, norb, norb)
            p_e = b.to_host(r_t[ti, ci, ri].real)
            w_e = b.to_host(w_t[ti, ci, ri].real)
            if nspin == 2:
                p_m = b.to_host((b.asarray(ph * sgn[None, :]) @ rho2)
                                .reshape(-1, norb, norb)[ti, ci, ri].real)
        return p_e, w_e, p_m

    def _pauli_pair_densities(self, st: _Setup, sc: _SCCState):
        """Two-component densities at every pair element (host arrays): the charge density
        p_e, the energy-weighted density w_e (Tr over spin), the magnetisation densities p_m
        (3, n_elem) (None without spin polarisation) and, for the dual spin-orbit model, the
        coefficient of dS: -1/2 sum_a (Lambda^a I^a(T) + I^a(T) Lambda^a)[nu, mu], where
        I^a(T) = Im sum_k w_k e^{2 pi i k.T} m_a(k) (DFTB+ derivative_iBlock)."""
        pb, p = st.pairs, st.pauli
        comp = so.components(sc.rho)
        need_m = p.w is not None
        need_i = p.hso is not None and p.dual
        mats = [comp[0], so.components(sc.wdm)[0]]
        if need_m or need_i:
            mats += [comp[1], comp[2], comp[3]]
        ci, ri = pb.cols, pb.rows
        norb = st.basis.norb
        if st.gamma_only:
            real_space = [np.einsum("k,kij->ij", st.kw, x)[None] for x in mats]
            ti = np.zeros(len(ci), int)
        else:
            shifts, ti = shift_index(pb)
            ph = (np.exp(2j * np.pi * (st.kpts @ shifts.T)) * st.kw[:, None]).T
            real_space = [(ph @ x.reshape(len(st.kw), norb * norb)).reshape(-1, norb, norb)
                          for x in mats]
        p_e = real_space[0][ti, ci, ri].real
        w_e = real_space[1][ti, ci, ri].real
        p_m = np.stack([real_space[2 + a][ti, ci, ri].real for a in range(3)]) if need_m else None
        c_soc = None
        if need_i:
            c_soc = np.zeros(len(ci))
            for a in range(3):
                im = real_space[2 + a].imag
                c_soc -= 0.5 * (p.lam[a] @ im + im @ p.lam[a])[ti, ci, ri]
        return p_e, w_e, p_m, c_soc

    # ------------------------------------------------------------------
    def _repulsive_pairs(self, st: _Setup):
        """Neighbour list and the (mask, Repulsive) of every label pair present."""
        labels = sorted(set(st.species))
        rc = max((r.cutoff for a in labels for b in labels
                  if (r := self.sk.repulsive(a, b, self.repulsive)) is not None), default=0.0)
        if rc == 0.0:
            return None, []
        nl = neighbor_list(st.pos, st.cell, st.pbc, rc)
        i, j = nl[0], nl[1]
        spec = np.array(st.species)
        out = []
        for a in labels:
            for b in labels:
                rep = self.sk.repulsive(a, b, self.repulsive)
                m = (spec[i] == a) & (spec[j] == b)
                if rep is not None and m.any():
                    out.append((m, rep))
        return nl, out

    def _repulsive_energy(self, st: _Setup) -> float:
        nl, terms = self._repulsive_pairs(st)
        e_tot = 0.0
        for m, rep in terms:
            e, _ = rep(nl[4][m])
            e_tot += 0.5 * e.sum()       # ordered pairs: every bond counted twice
        return e_tot

    def _repulsive_gradient(self, st: _Setup):
        """Repulsive dE/dR (n_atom, 3) and virial (3, 3)."""
        grad = np.zeros_like(st.pos)
        w = np.zeros((3, 3))
        nl, terms = self._repulsive_pairs(st)
        for m, rep in terms:
            i, j, _, vec, d = (x[m] for x in nl)
            _, de = rep(d)
            g = (0.5 * de / d)[:, None] * vec          # dE/d(vec) of each ordered pair
            grad = ops.scatter_add(grad, i, -g)
            grad = ops.scatter_add(grad, j, g)
            w += g.T @ vec
        return grad, w
