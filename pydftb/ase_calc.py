"""ASE calculator: drop-in replacement for ``ase.calculators.dftb.Dftb``
in workflows that need energies, forces, charges and eigenvalues.

Unit handling mirrors what ASE + DFTB+ do, so results can be compared
directly: positions go Angstrom -> Bohr with DFTB+'s constant; energies and
forces come back with ASE's constants (``ase.units``), exactly as
``ase.calculators.dftb.Dftb`` converts DFTB+ output.
"""
from __future__ import annotations

import contextlib
from dataclasses import replace

import numpy as np
from ase import units
from ase.calculators.calculator import (CalculationFailed, Calculator, all_changes,
                                        kptdensity2monkhorstpack)

from .analysis import Analysis
from .bands import States, eigenstates
from .calculator import DFTB, SCCControl
from .casida import Casida
from .constants import BOHR, HARTREE, MIN_TEMP, VOLT_PER_ANGSTROM_AU
from .dftbplus_io import write_charges
from .dispersion import make_dispersion
from .external import PointCharges
from .hybrid import Hybrid
from .kpoints import monkhorst_pack, reduce_time_reversal
from .occupations import Smearing
from .parallel import as_kthreads
from .params import ParamSet
from .typeswitch import check_switches, internal_species
from .xlbomd import XLBOMD, Integrator


def _fermi_au(value):
    """``fixed_fermi_level`` from eV to Hartree (one value or (up, down))."""
    if value is None:
        return None
    v = np.asarray(value, float) / HARTREE
    return float(v) if v.ndim == 0 else tuple(float(x) for x in v)


def _hybrid_au(spec):
    """``hybrid`` with ``cutoff_reduction`` from Angstrom to Bohr (M6h-1)."""
    if isinstance(spec, dict) and "cutoff_reduction" in spec:
        return {**spec, "cutoff_reduction": float(spec["cutoff_reduction"]) / BOHR}
    if isinstance(spec, Hybrid):
        return replace(spec, cutoff_reduction=spec.cutoff_reduction / BOHR)
    return spec


def _truncation_au(spec):
    """``truncate_sk_range`` from Angstrom to Bohr: a distance, (r, hard) or
    {"max_distance": r, "hard_cutoff": hard}."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        return {**spec, "max_distance": float(spec["max_distance"]) / BOHR}
    if np.isscalar(spec):
        return float(spec) / BOHR
    r, hard = spec
    return float(r) / BOHR, bool(hard)


def _dipole_correction_au(spec):
    """``dipole_correction`` with the layer position from Angstrom to Bohr."""
    if isinstance(spec, dict) and spec.get("layer_position") is not None:
        return {**spec, "layer_position": float(spec["layer_position"]) / BOHR}
    return spec


def _analysis_au(spec):
    """``analysis`` with the electrostatic potential's points, grid and softening from
    Angstrom to Bohr (fractional coordinates stay)."""
    a = Analysis.make(spec)
    if a is None or a.esp is None:
        return a
    esp = dict(a.esp)
    frac = esp.get("fractional", False)
    if "softening" in esp:
        esp["softening"] = float(esp["softening"]) / BOHR
    if "points" in esp and not frac:
        esp["points"] = np.asarray(esp["points"], float) / BOHR
    if "grid" in esp and not frac:
        grid = dict(esp["grid"])
        grid["spacing"] = np.asarray(grid["spacing"], float) / BOHR
        grid["origin"] = np.asarray(grid["origin"], float) / BOHR
        esp["grid"] = grid
    return replace(a, esp=esp)


@contextlib.contextmanager
def _as_calculation_failed():
    """A linear-algebra failure of the engine as ASE's ``CalculationFailed``, the error
    ``ase.calculators.dftb.Dftb`` raises when DFTB+ stops, e.g. on a non-positive-definite
    overlap (atoms too close); the original stays as the cause."""
    try:
        yield
    except np.linalg.LinAlgError as err:
        raise CalculationFailed(f"pydftb: {err}") from err


class PyDFTB(Calculator):
    """
    Parameters
    ----------
    skdir : directory with ``A-B.skf`` (or ``.skf.gz``) files.
    max_l : optional {label: l}; default is the ASE/PTBP rule (highest occupied
            shell in the homonuclear SKF).
    kpts  : None (Gamma), (n1, n2, n3) Monkhorst-Pack grid, a k-point density
            (float, ASE ``kptdensity2monkhorstpack``), explicit
            ``{"kpts": (nk,3) fractional, "weights": (nk,)}``, or a band path (ASE
            ``BandPath`` or ``{"path": "GXWLGK", "npoints": 100}``: the path points in
            order, equal weights, as ASE + DFTB+; for a band structure after an SCC on a
            grid use :meth:`band_structure` instead).
    time_reversal : reduce k and -k (DFTB+ does this for SupercellFolding); never for paths.
            Default (None): yes, except for two-component calculations (non-collinear spin,
            spin-orbit coupling), which need the full grid.
    smearing : ("fermi", T) | ("gaussian", T) | ("mp", T, order); T in K.
               Default: Fermi at T = 0 (i.e. 1e-8 Ha, DFTB+ behaviour).
    repulsive : "skf" | "polynomial" | "none" (see :class:`pydftb.DFTB`).
    threads : limit BLAS/OpenMP threads during the calculation (None = leave as is); with
            the torch backend also torch's intra-op threads.
    k_threads : threads over chunks of k-points (int or "auto"; see :mod:`pydftb.parallel`);
            bitwise identical results for any number, BLAS on one thread while they run.
    backend, device, dtype : "numpy" (default) or "torch" on a torch device, in "float64"
            or "float32" (see :class:`pydftb.DFTB` and :mod:`pydftb.backend.spec`).
    reuse_charges : start each SCC from the previous calculation's input populations
            (same type labels), as DFTB+'s own drivers do between geometry steps; far
            fewer SCC iterations in optimisations and MD.  Default off: every
            calculation starts from neutral atoms, as ASE + DFTB+ (a new process each time).
    labels : optional type labels per atom (several parameter sets per element);
             default: chemical symbols.
    Other keywords (``scc``, ``scc_tol``, ``max_scc``, ``mixer``, ``mixing``,
    ``solver``, ``shell_resolved``, ``fd_delta``, for DFTB3 ``third_order``,
    ``hubbard_derivs``, ``h_damping``, for collinear spin ``spin``,
    ``unpaired_electrons``, ``relax_total_spin``, ``spin_constants``, for two-component
    calculations ``spin="noncolinear"``, ``initial_spins``, ``spin_orbit``,
    ``spin_orbit_dual``, for DFTB+U and onsite corrections ``plus_u``,
    ``plus_u_functional``, ``onsite_corrections``, implicit solvation ``solvation``,
    ``rescale_solvated_fields``, Casida excitations ``casida``, and ``dispersion``) are passed
    to :class:`pydftb.DFTB`.  The excitations of the last calculation are in
    :attr:`excitations` (atomic units).  With a Casida state of interest (and excited-state
    forces, the default) energy and forces are the excited state's, E_0 + Omega and its
    gradient, as for DFTB+'s own drivers, so ASE optimisers and MD run in that state (M6e);
    charges and dipole stay the ground state's (the excited ones are in ``raw``).
    The environment of milestone M6c in ASE units: ``electric_field`` (E_x, E_y, E_z) in
    V/Angstrom, ``point_charges`` (positions in Angstrom, charges in e[, Gaussian blur width
    in Angstrom]), ``site_potentials`` {"gross"/"net": {atom: V}} in eV; the solvation
    model's parameters stay in atomic units like the other model parameters.  For ASE's
    QM/MM (``EIQMMM``) the calculator offers :meth:`embed`; ``dipole`` (e Angstrom) is
    available for molecules.
    Dispersion parameters stay in atomic units (Bohr, Hartree) as in DFTB+; the elements
    come from ``atoms``.  With ``spin="noncolinear"`` and no ``initial_spins`` the atoms'
    initial magnetic moments are used ((N, 3) vectors, or (N,) along z); ``magmoms`` are
    then (N, 3) vectors and ``magmom`` the length of their sum.  With collinear spin, non-zero
    initial magnetic moments are the initial spins (DFTB+ ``InitialSpins``, M6f).
    Milestone M6f: ``mixer``/``mixing`` take DFTB+'s mixers and their options,
    ``initial_charges`` (e per atom) starts the SCC from those charges, ``read_charges`` (a
    DFTB+ ``charges.bin``/``charges.dat``) starts the first calculation from that file, and
    :meth:`write_charges` writes one.
    Milestone M6f-2: ``fixed_fermi_level`` (eV; (up, down) for collinear spin) fills at a fixed
    Fermi level (DFTB+ ``FixedFermiLevel``), and ``free_energy`` is then the force-consistent
    F - sum_s N_s E_f,s (DFTB+'s "Force related energy"; ``energy`` stays the total energy, as
    ASE + DFTB+ read it); ``independent_k_filling``, ``hubbard`` (Hartree, like the other model
    parameters), ``reference_occupations`` (electrons; ``charges`` are then relative to them)
    and ``truncate_sk_range`` (Angstrom, or {"max_distance", "hard_cutoff"}) as in
    :class:`pydftb.DFTB`.
    Milestone M6f-3: ``h5`` (the H5 hydrogen-bond correction), ``halogen_correction`` (DFTB3)
    and ``dipole_correction`` (slabs; {"layer_position": z0} in Angstrom) as in
    :class:`pydftb.DFTB`.
    Milestone M6f-4a: ``analysis`` as in :class:`pydftb.DFTB` (the electrostatic potential's
    points, grid and softening in Angstrom); with atom-resolved energies ``energies`` (eV,
    ``atoms.get_potential_energies()``) are the total energy per atom, summing to ``energy``;
    the rest is in ``calc.raw.analysis`` (atomic units).
    Milestone M5a: ``type_switching`` (:class:`pydftb.typeswitch.TypeSwitch` objects or dicts;
    cut-offs in Bohr, like the other model parameters): the atoms of a switched element mix its
    type labels by their coordination numbers, as in :class:`pydftb.DFTB`; the SK set holds all
    the types; the weights of the last calculation are ``raw.type_weights``.
    Milestone M6f-5: ``selected_shells`` (DFTB+ ``SelectedShells``), a label's shells taken from
    several SK files, as in :class:`pydftb.ParamSet`.
    Milestone M6h-1: ``hybrid`` ("lc", "cam", a dict or a :class:`pydftb.Hybrid`, as in
    :class:`pydftb.DFTB`; ``cutoff_reduction`` in Angstrom): a range-separated hybrid functional
    for molecules, with
    SK files that have a ``RangeSep`` section; ``reuse_charges`` then carries the density matrix
    between geometry steps, as DFTB+ does.
    """
    implemented_properties = ["energy", "free_energy", "forces", "charges", "stress",
                              "magmom", "magmoms", "dipole", "energies"]

    def __init__(self, skdir, max_l=None, kpts=None, time_reversal: bool | None = None,
                 smearing=None, temperature: float | None = None, scc: bool = True,
                 scc_tol: float = 1e-6, max_scc: int = 200, mixer: str = "broyden",
                 mixing: float | None = None, repulsive: str = "skf", solver: str = "batched",
                 shell_resolved: bool = False, threads: int | None = None, labels=None,
                 fd_delta: float | None = None, third_order: str | None = None,
                 hubbard_derivs: dict | None = None, h_damping: float | None = None,
                 spin: bool | str = False, unpaired_electrons: float = 0.0,
                 relax_total_spin: bool = False, spin_constants: dict | None = None,
                 dispersion=None, reuse_charges: bool = False, k_threads=1,
                 backend=None, device: str | None = None, dtype: str | None = None,
                 initial_spins=None, spin_orbit: dict | None = None,
                 spin_orbit_dual: bool = True, plus_u: dict | None = None,
                 plus_u_functional: str = "fll", onsite_corrections: dict | None = None,
                 electric_field=None, point_charges=None, site_potentials: dict | None = None,
                 solvation=None, rescale_solvated_fields: bool = True, casida=None,
                 xlbomd=None, force_evaluation: str | None = None, initial_charges=None,
                 read_charges=None, fixed_fermi_level=None, independent_k_filling: bool = False,
                 hubbard: dict | None = None, reference_occupations=None,
                 truncate_sk_range=None, h5=None, halogen_correction: bool = False,
                 dipole_correction=None, analysis=None, type_switching=None,
                 selected_shells=None, hybrid=None, **kwargs):
        super().__init__(**kwargs)
        self.params = ParamSet(skdir, max_l=max_l, shell_resolved=shell_resolved,
                               selected_shells=selected_shells)
        self.kpts_spec = kpts
        self.time_reversal = time_reversal
        if smearing is None:
            smearing = ("fermi", 0.0 if temperature is None else temperature)
        self.xlbomd = XLBOMD.make(xlbomd)             # M6e
        if force_evaluation is None:                  # DFTB+ needs a dynamics type for XL
            force_evaluation = "traditional"
            if self.xlbomd is not None:
                hot = Smearing.from_spec(smearing).kt > MIN_TEMP
                force_evaluation = "dynamics" if hot else "dynamics_t0"
        self._xl = None                               # the XL-BOMD state (Integrator)
        self.engine_opts = dict(scc=scc, scc_tol=scc_tol, max_scc=max_scc, mixer=mixer,
                                mixing=mixing, repulsive=repulsive, solver=solver,
                                smearing=smearing, fd_delta=fd_delta, third_order=third_order,
                                hubbard_derivs=hubbard_derivs, h_damping=h_damping, spin=spin,
                                unpaired_electrons=unpaired_electrons,
                                relax_total_spin=relax_total_spin, spin_constants=spin_constants,
                                backend=backend, device=device, dtype=dtype,
                                initial_spins=initial_spins, spin_orbit=spin_orbit,
                                spin_orbit_dual=spin_orbit_dual, plus_u=plus_u,
                                plus_u_functional=plus_u_functional,
                                onsite_corrections=onsite_corrections, solvation=solvation,
                                rescale_solvated_fields=rescale_solvated_fields,
                                casida=casida, force_evaluation=force_evaluation,
                                initial_charges=initial_charges,
                                fixed_fermi_level=_fermi_au(fixed_fermi_level),
                                independent_k_filling=independent_k_filling, hubbard=hubbard,
                                reference_occupations=reference_occupations,
                                truncate_sk_range=_truncation_au(truncate_sk_range), h5=h5,
                                halogen_correction=halogen_correction,
                                dipole_correction=_dipole_correction_au(dipole_correction),
                                analysis=_analysis_au(analysis), hybrid=_hybrid_au(hybrid))
        self.type_switching = None                    # M5a
        if type_switching is not None:
            self.type_switching = check_switches(
                type_switching if isinstance(type_switching, (list, tuple)) else [type_switching])
            self.engine_opts["type_switching"] = self.type_switching
        self.read_charges = read_charges              # DFTB+ charge file for the first run (M6f)
        self.electric_field = electric_field          # V/Angstrom (M6c)
        self.point_charges = point_charges            # Angstrom, e
        self.site_potentials = site_potentials        # eV
        self.pcpot = None
        self.dispersion = make_dispersion(dispersion)
        self.threads = threads
        self.k_threads = as_kthreads(k_threads)       # one pool for all calculations
        self.labels = None if labels is None else list(labels)
        self.reuse_charges = reuse_charges
        self.raw = None          # last pydftb.DFTBResult (atomic units)
        self._raw_labels = None  # type labels of that result

    # ------------------------------------------------------------------
    @property
    def excitations(self):
        """Casida excitations of the last calculation (:class:`pydftb.casida.Excitations`,
        atomic units; [triplets, singlets] for symmetry "both"), None without ``casida``."""
        return None if self.raw is None else self.raw.excitations

    @property
    def two_component(self) -> bool:
        """Non-collinear spin or spin-orbit coupling (M6a)."""
        o = self.engine_opts
        return (isinstance(o["spin"], str) and o["spin"].lower() in ("noncolinear", "noncollinear")
                or o["spin_orbit"] is not None)

    @property
    def _torch(self) -> bool:
        b = self.engine_opts["backend"]
        return getattr(b, "is_torch", False) or str(b).lower() == "torch"

    def _kpoints(self, atoms, spec=None):
        spec = self.kpts_spec if spec is None else spec
        if not atoms.pbc.any() or spec is None:
            return None
        if hasattr(spec, "special_points") or (isinstance(spec, dict) and "path" in spec):
            path = spec if hasattr(spec, "special_points") else atoms.cell.bandpath(
                spec["path"], npoints=spec.get("npoints"), density=spec.get("density"))
            k = np.asarray(path.kpts, float)
            return k, np.full(len(k), 1.0 / len(k))
        if isinstance(spec, dict):
            k, w = np.atleast_2d(spec["kpts"]), np.asarray(spec.get("weights", np.ones(len(spec["kpts"]))))
        else:
            grid = kptdensity2monkhorstpack(atoms, spec, even=False) if np.isscalar(spec) else spec
            k, w = monkhorst_pack(tuple(int(n) for n in grid))
        reduce = not self.two_component if self.time_reversal is None else self.time_reversal
        if reduce:
            k, w = reduce_time_reversal(k, w)
        return k, w / w.sum()

    def _engine(self, atoms):
        """pydftb.DFTB for these atoms, its type labels and SK set."""
        labels = self.labels or atoms.get_chemical_symbols()
        if self.type_switching is not None:           # M5a: the engine's labels, every type
            sk = self.params.skset(internal_species(self.type_switching, labels)
                                   + [t for sw in self.type_switching for t in sw.types])
        else:
            sk = self.params.skset(labels)
        disp = self.dispersion
        if disp is not None:
            disp = disp.with_elements({lab: int(z) for lab, z in zip(labels, atoms.numbers)})
        opts = dict(self.engine_opts)
        spin = opts["spin"]
        if (isinstance(spin, str) and spin.lower() in ("noncolinear", "noncollinear")
                and opts["initial_spins"] is None):
            m = np.asarray(atoms.get_initial_magnetic_moments(), float)
            opts["initial_spins"] = m if m.ndim == 2 else np.outer(m, [0.0, 0.0, 1.0])
        elif spin and opts["initial_spins"] is None and not self._torch:
            # collinear: the moments, if any, start the SCC (M6f; NumPy only, so the torch
            # backend ignores them as before)
            m = np.asarray(atoms.get_initial_magnetic_moments(), float)
            if m.ndim == 1 and np.any(m != 0.0):
                opts["initial_spins"] = m
        calc = DFTB(sk, kpts=self._kpoints(atoms), charge=float(atoms.get_initial_charges().sum()),
                    dispersion=disp, k_threads=self.k_threads, **opts, **self._environment())
        return calc, labels, sk

    def _environment(self) -> dict:
        """Field, point charges and site potentials in atomic units (M6c)."""
        out = {}
        if self.electric_field is not None:
            out["electric_field"] = np.asarray(self.electric_field, float) * VOLT_PER_ANGSTROM_AU
        if self.point_charges is not None:
            pc = PointCharges.make(self.point_charges)
            out["point_charges"] = PointCharges(pc.positions / BOHR, pc.charges, pc.blur / BOHR)
        if self.site_potentials is not None:
            out["site_potentials"] = {
                kind: ({a: v / HARTREE for a, v in val.items()} if isinstance(val, dict)
                       else np.asarray(val, float) / HARTREE)
                for kind, val in self.site_potentials.items()}
        return out

    def embed(self, charges, blur: float = 0.0) -> "PointChargePotential":
        """Point-charge embedding for ASE's QM/MM (``ase.calculators.qmmm.EIQMMM``): MM
        charges (e) whose positions ASE sets (Angstrom); optional Gaussian blur (Angstrom)."""
        self.pcpot = PointChargePotential(self, charges, blur)
        return self.pcpot

    def reset(self):
        """Clear the last calculation; with XL-BOMD the integration starts again."""
        super().reset()
        self._xl = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        periodic = bool(a.pbc.any())
        calc, labels, sk = self._engine(a)
        # As in DFTB+, periodic force runs also give the stress (about 1 % extra
        # time), so ASE cell filters, which ask for forces and then stress, need one run.
        want_forces = "forces" in properties or (periodic and "stress" in properties)
        q_init = None
        if self.reuse_charges and self.raw is not None and self._raw_labels == list(labels):
            q_init = self.raw.q_in        # DFTB+ keeps qInput between geometry steps
        elif self.read_charges is not None and self.raw is None:
            q_init = calc.populations_from_file(labels, self.read_charges)    # M6f
        xl = {}
        if self.xlbomd is not None:       # one calculation = one MD step (M6e)
            if self._xl is None or self._raw_labels != list(labels):
                self._xl = Integrator(self.xlbomd)
            mn, mx, tol = self._xl.scc_parameters(calc.max_scc, calc.scc_tol)
            xl = dict(scc_control=SCCControl(mn, mx, tol), xl=True)
            q_init, want_forces = self._xl.next, True
        run = lambda: calc.calculate(labels, a.positions / BOHR,       # noqa: E731
                                     a.cell.array / BOHR if periodic else None,
                                     a.pbc if periodic else None,
                                     forces=want_forces, q_init=q_init, **xl)
        with _as_calculation_failed():
            if self.threads is None:
                r = run()
            else:
                from threadpoolctl import threadpool_limits
                # torch's count first: it is read before threadpoolctl changes OpenMP's
                with calc.backend.threads(self.threads), threadpool_limits(self.threads):
                    r = run()
        if xl:
            self._xl.advance(r.q_out)
        self._store(r, sk, labels)

    def write_charges(self, path, text: bool | None = None):
        """Write the last calculation's input populations as DFTB+'s ``charges.bin`` (or
        ``charges.dat`` with ``text``, default by the extension), which DFTB+'s
        ``ReadInitialCharges`` and ``read_charges`` here read (M6f)."""
        if self.raw is None:
            raise RuntimeError("no calculation to write the charges of")
        calc, labels, _ = self._engine(self.atoms)
        write_charges(path, calc.charge_data(labels, self.raw), text)

    def evaluate(self, atoms, params=None, requires_grad=(), mode: str = "hf",
                 forces: bool = False, stress: bool = False, q_init=None):
        """:func:`pydftb.autodiff.evaluate` for ``atoms`` (milestone M4c): the engine run of
        :meth:`calculate` (forces, and the stress of periodic systems), whose results are
        stored exactly as :meth:`calculate` stores them, plus the derivatives.  ``params``:
        a :class:`pydftb.autodiff.Params` of this calculator's SK set for these atoms
        (default: its values, with gradients for the kinds in ``requires_grad``);
        ``forces``/``stress``: whether they carry a graph.  With ``threads`` the BLAS and
        torch threads are limited during the calculation.  Returns (Result, Params)."""
        from .autodiff import GradientUnavailable, Params, evaluate
        Calculator.calculate(self, atoms, ["energy", "forces"], all_changes)
        a = self.atoms
        periodic = bool(a.pbc.any())
        calc, labels, sk = self._engine(a)
        if params is None:
            params = Params.from_skset(sk, calc.repulsive, calc.hubbard_derivs,
                                       calc.spin_constants, requires_grad=requires_grad)
        limits = contextlib.ExitStack()
        if self.threads is not None:
            from threadpoolctl import threadpool_limits
            from .backend.spec import _torch_threads
            limits.enter_context(_torch_threads(self.threads))     # before threadpoolctl
            limits.enter_context(threadpool_limits(self.threads))
        with limits, _as_calculation_failed():
            try:
                res = evaluate(calc, labels, a.positions / BOHR,
                               a.cell.array / BOHR if periodic else None,
                               a.pbc if periodic else None, params, mode=mode, forces=forces,
                               stress=stress and periodic, q_init=q_init)
            except GradientUnavailable as err:
                if getattr(err, "raw", None) is not None:   # e.g. an unconverged SCC: the
                    self._store(err.raw, sk, labels)        # results of calculate, no gradient
                raise
        self._store(res.raw, sk, labels)
        return res, params

    @property
    def excited(self) -> bool:
        """Whether energy and forces are those of an excited state (M6e): a Casida state of
        interest with excited-state forces (the default)."""
        cas = Casida.make(self.engine_opts["casida"])
        return cas is not None and cas.state_of_interest != 0 and cas.excited_forces

    def _store(self, r, sk, labels):
        """``self.results`` (ASE units) from a :class:`pydftb.DFTBResult`."""
        self.raw = r
        self._raw_labels = list(labels)
        if self.excited:        # E_0 + Omega, consistent with the forces (as DFTB+'s drivers)
            self.results["energy"] = r.excited_energy * units.Hartree
            self.results["free_energy"] = r.excited_free_energy * units.Hartree
        else:
            self.results["energy"] = r.energy * units.Hartree
            # force-consistent: F - sum_s N_s E_f,s with a fixed Fermi level (M6f), else F
            self.results["free_energy"] = r.e_force_related * units.Hartree
        if self.type_switching is not None:               # M5a: the engine's labels
            labels = internal_species(self.type_switching, labels)
        q0 = (r.q0_atom if r.q0_atom is not None          # custom reference occupations (M6f)
              else np.array([sk.occupation[lab].sum() for lab in labels]))
        self.results["charges"] = q0 - r.charges
        if r.forces is not None:
            self.results["forces"] = r.forces * units.Hartree / units.Bohr
        if r.stress is not None:
            # same conversion as ase.calculators.dftb.Dftb: -sigma_DFTB+, Voigt xx yy zz yz xz xy
            self.results["stress"] = (-r.stress * units.Hartree / units.Bohr ** 3).flat[[0, 4, 8, 5, 2, 1]]
        if r.dipole is not None:
            self.results["dipole"] = r.dipole * units.Bohr
        if (r.analysis is not None and r.analysis.atom_energies is not None
                and not self.excited):                                   # M6f-4a
            self.results["energies"] = r.analysis.atom_energies["total"] * units.Hartree
        magmoms = np.zeros(len(labels)) if r.magmoms is None else r.magmoms
        self.results["magmoms"] = magmoms
        self.results["magmom"] = (float(magmoms.sum()) if magmoms.ndim == 1
                                  else float(np.linalg.norm(magmoms.sum(axis=0))))

    # ---- ASE accessors used by band-structure / DOS tools -------------
    def _raw(self):
        if self.raw is None:
            self.get_potential_energy()
        return self.raw

    def get_number_of_spins(self):
        return 1 if self._raw().fermi_spin is None else 2

    def _index(self, kpt, spin):
        """Position of (k-point, spin) in the eigenvalue list ([up k..., down k...])."""
        return kpt + spin * len(self._raw().kweights)

    def get_ibz_k_points(self):
        r = self._raw()
        return np.array(r.kpoints)

    def get_k_point_weights(self):
        return np.array(self._raw().kweights)

    def get_eigenvalues(self, kpt=0, spin=0):
        """Eigenvalues (eV) at k-point index ``kpt`` (and ``spin`` 0 = up, 1 = down)."""
        r = self._raw()
        if r.eigenvalues is None:
            raise RuntimeError("eigenvalues are not available with this solver")
        return np.asarray(r.eigenvalues[self._index(kpt, spin)]) * units.Hartree

    def get_occupation_numbers(self, kpt=0, spin=0):
        return np.asarray(self._raw().occupations[self._index(kpt, spin)])

    def get_fermi_level(self):
        """Fermi level (eV); with two spin Fermi levels, that of spin up."""
        return self._raw().fermi * units.Hartree

    def get_fermi_levels(self):
        r = self._raw()
        return np.array(r.fermi_spin if r.fermi_spin is not None else [r.fermi]) * units.Hartree

    # ---- band structure, DOS, PDOS at the SCC charges (pydftb.bands) -------
    def states(self, kpts=None, projections: bool = False, atoms=None,
               chunk: int | None = None) -> States:
        """Eigenstates at the charges of the SCC run for ``atoms`` (run first if needed), as
        DFTB+'s band-structure run (``ReadInitialCharges``, one iteration).

        ``kpts``: None (the SCC k-points) or any form the constructor accepts (grid, density,
        explicit dict, ASE ``BandPath`` or path dict).  Returns :class:`pydftb.bands.States`
        (Hartree; ``dos()``/``pdos()`` in eV).
        """
        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise ValueError("no atoms: attach the calculator or pass atoms")
        self.get_potential_energy(atoms)
        r = self.raw
        a = self.atoms
        periodic = bool(a.pbc.any())
        if kpts is not None and not periodic:
            raise ValueError("k-points need a periodic system")
        calc, labels, _ = self._engine(a)
        run = lambda: eigenstates(                                     # noqa: E731
            calc, labels, a.positions / BOHR, a.cell.array / BOHR if periodic else None,
            a.pbc if periodic else None, r.q_in,
            kpts=None if kpts is None else self._kpoints(a, kpts), projections=projections,
            scc_fermi=r.fermi_spin if r.fermi_spin is not None else r.fermi, chunk=chunk)
        if self.threads is None:
            return run()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(self.threads):
            return run()

    def band_structure(self, path=None, npoints=None, atoms=None):
        """ASE ``BandStructure`` (eV, zero = SCC Fermi level of spin up).

        Without arguments: ASE's default, i.e. the calculator's own k-points (useful with
        ``kpts=BandPath``).  With ``path`` (ASE ``BandPath`` or special points such as
        "GXWLGK") and/or ``npoints``: the states along that path at the SCC charges.
        """
        if path is None and npoints is None:
            return super().band_structure()
        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise ValueError("no atoms: attach the calculator or pass atoms")
        if not hasattr(path, "special_points"):
            path = atoms.cell.bandpath(path, npoints=npoints)
        st = self.states(kpts=path, atoms=atoms)
        return st.band_structure(path, reference=self.get_fermi_level(), ev=units.Hartree)


class PointChargePotential:
    """The object :meth:`PyDFTB.embed` returns (ASE's point-charge embedding protocol):
    ASE sets the MM charges' positions (Angstrom) before each QM calculation, which then
    starts afresh; :meth:`get_forces` gives the forces on the charges (eV/Angstrom)."""

    def __init__(self, calc: PyDFTB, charges, blur: float = 0.0):
        self.calc = calc
        self.charges = np.asarray(charges, float)
        self.blur = blur
        self.positions = None

    def _update(self):
        if self.positions is not None:
            self.calc.point_charges = (self.positions, self.charges, self.blur)
            self.calc.reset()

    def set_positions(self, positions, com_pv=None):
        self.positions = np.array(positions, float)
        self._update()

    def set_charges(self, charges):
        self.charges = np.asarray(charges, float)
        self._update()

    def get_forces(self, calc=None, get_forces: bool = True) -> np.ndarray:
        if not get_forces:
            return np.zeros_like(self.positions)
        r = self.calc.raw
        if r is None or r.point_charge_forces is None:
            self.calc.get_forces()
            r = self.calc.raw
        return r.point_charge_forces * units.Hartree / units.Bohr
