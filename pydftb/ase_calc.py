"""ASE calculator: drop-in replacement for ``ase.calculators.dftb.Dftb``
in workflows that need energies, forces, charges and eigenvalues.

Unit handling mirrors what ASE + DFTB+ do, so results can be compared
directly: positions go Angstrom -> Bohr with DFTB+'s constant; energies and
forces come back with ASE's constants (``ase.units``), exactly as
``ase.calculators.dftb.Dftb`` converts DFTB+ output.
"""
from __future__ import annotations

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes, kptdensity2monkhorstpack

from .calculator import DFTB
from .constants import BOHR
from .dispersion import make_dispersion
from .kpoints import monkhorst_pack, reduce_time_reversal
from .params import ParamSet


class PyDFTB(Calculator):
    """
    Parameters
    ----------
    skdir : directory with ``A-B.skf`` (or ``.skf.gz``) files.
    max_l : optional {label: l}; default is the ASE/PTBP rule (highest occupied
            shell in the homonuclear SKF).
    kpts  : None (Gamma), (n1, n2, n3) Monkhorst-Pack grid, a k-point density
            (float, ASE ``kptdensity2monkhorstpack``), or explicit
            ``{"kpts": (nk,3) fractional, "weights": (nk,)}``.
    time_reversal : reduce k and -k (DFTB+ does this for SupercellFolding).
    smearing : ("fermi", T) | ("gaussian", T) | ("mp", T, order); T in K.
               Default: Fermi at T = 0 (i.e. 1e-8 Ha, DFTB+ behaviour).
    repulsive : "skf" | "polynomial" | "none" (see :class:`pydftb.DFTB`).
    threads : limit BLAS/OpenMP threads during the calculation (None = leave as is).
    labels : optional type labels per atom (several parameter sets per element);
             default: chemical symbols.
    Other keywords (``scc``, ``scc_tol``, ``max_scc``, ``mixer``, ``mixing``,
    ``solver``, ``shell_resolved``, ``fd_delta``, for DFTB3 ``third_order``,
    ``hubbard_derivs``, ``h_damping``, for collinear spin ``spin``,
    ``unpaired_electrons``, ``relax_total_spin``, ``spin_constants``, and
    ``dispersion``) are passed to :class:`pydftb.DFTB`.  Dispersion parameters stay
    in atomic units (Bohr, Hartree) as in DFTB+; the elements come from ``atoms``.
    """
    implemented_properties = ["energy", "free_energy", "forces", "charges", "stress",
                              "magmom", "magmoms"]

    def __init__(self, skdir, max_l=None, kpts=None, time_reversal: bool = True,
                 smearing=None, temperature: float | None = None, scc: bool = True,
                 scc_tol: float = 1e-6, max_scc: int = 200, mixer: str = "broyden",
                 mixing: float = 0.2, repulsive: str = "skf", solver: str = "batched",
                 shell_resolved: bool = False, threads: int | None = None, labels=None,
                 fd_delta: float | None = None, third_order: str | None = None,
                 hubbard_derivs: dict | None = None, h_damping: float | None = None,
                 spin: bool = False, unpaired_electrons: float = 0.0,
                 relax_total_spin: bool = False, spin_constants: dict | None = None,
                 dispersion=None, **kwargs):
        super().__init__(**kwargs)
        self.params = ParamSet(skdir, max_l=max_l, shell_resolved=shell_resolved)
        self.kpts_spec = kpts
        self.time_reversal = time_reversal
        if smearing is None:
            smearing = ("fermi", 0.0 if temperature is None else temperature)
        self.engine_opts = dict(scc=scc, scc_tol=scc_tol, max_scc=max_scc, mixer=mixer,
                                mixing=mixing, repulsive=repulsive, solver=solver,
                                smearing=smearing, fd_delta=fd_delta, third_order=third_order,
                                hubbard_derivs=hubbard_derivs, h_damping=h_damping, spin=spin,
                                unpaired_electrons=unpaired_electrons,
                                relax_total_spin=relax_total_spin, spin_constants=spin_constants)
        self.dispersion = make_dispersion(dispersion)
        self.threads = threads
        self.labels = None if labels is None else list(labels)
        self.raw = None          # last pydftb.DFTBResult (atomic units)

    # ------------------------------------------------------------------
    def _kpoints(self, atoms):
        spec = self.kpts_spec
        if not atoms.pbc.any() or spec is None:
            return None
        if isinstance(spec, dict):
            k, w = np.atleast_2d(spec["kpts"]), np.asarray(spec.get("weights", np.ones(len(spec["kpts"]))))
        else:
            grid = kptdensity2monkhorstpack(atoms, spec, even=False) if np.isscalar(spec) else spec
            k, w = monkhorst_pack(tuple(int(n) for n in grid))
        if self.time_reversal:
            k, w = reduce_time_reversal(k, w)
        return k, w / w.sum()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        labels = self.labels or a.get_chemical_symbols()
        periodic = bool(a.pbc.any())
        kpts = self._kpoints(a)
        sk = self.params.skset(labels)
        disp = self.dispersion
        if disp is not None:
            disp = disp.with_elements({lab: int(z) for lab, z in zip(labels, a.numbers)})
        calc = DFTB(sk, kpts=kpts, charge=float(a.get_initial_charges().sum()), dispersion=disp,
                    **self.engine_opts)
        # As in DFTB+, periodic force runs also give the stress (about 1 % extra
        # time), so ASE cell filters, which ask for forces and then stress, need one run.
        want_forces = "forces" in properties or (periodic and "stress" in properties)
        run = lambda: calc.calculate(labels, a.positions / BOHR,       # noqa: E731
                                     a.cell.array / BOHR if periodic else None,
                                     a.pbc if periodic else None,
                                     forces=want_forces)
        if self.threads is None:
            r = run()
        else:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(self.threads):
                r = run()
        self.raw = r
        self.results["energy"] = r.energy * units.Hartree
        self.results["free_energy"] = r.free_energy * units.Hartree
        q0 = np.array([sk.occupation[lab].sum() for lab in labels])
        self.results["charges"] = q0 - r.charges
        if r.forces is not None:
            self.results["forces"] = r.forces * units.Hartree / units.Bohr
        if r.stress is not None:
            # same conversion as ase.calculators.dftb.Dftb: -sigma_DFTB+, Voigt xx yy zz yz xz xy
            self.results["stress"] = (-r.stress * units.Hartree / units.Bohr ** 3).flat[[0, 4, 8, 5, 2, 1]]
        magmoms = np.zeros(len(a)) if r.magmoms is None else r.magmoms
        self.results["magmoms"] = magmoms
        self.results["magmom"] = float(magmoms.sum())

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
