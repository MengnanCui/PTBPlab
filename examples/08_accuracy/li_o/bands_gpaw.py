"""D1 (A series, A0 section 2): band and orbital references for Li and Li-O, with GPAW.

The structures are:

* crystals: the EOS points of ``examples/07_battery_li_li2o/data/train.xyz`` at 0.952 / 1.000 /
  1.048 V0 (indices 03, 05, 07) of bcc Li, fcc Li, Li2O and Li2O2, and hcp Li built at the bcc
  volume per atom (ideal c/a 1.633) at the same three volumes;
* dimers: Li2, LiO and O2 at 5 distances each from ``data/short_range.xyz``, in their box at
  the Gamma point (kind "molecule").

For every structure GPAW does a PBE SCF in plane waves (ECUT 800 eV as the VASP data).  The
k-grid follows VASP's KSPACING rule N_i = max(1, ceil(2 pi |b_i| / 0.15)), Gamma-centred, and
the smearing is Methfessel-Paxton of order 0, i.e. Gaussian with width 0.05 eV, like ISMEAR 0
and SIGMA 0.05.  Crystals then get a non-self-consistent run at fixed density on the k-points
of ``pydftb.bandref.special_kpoints``: special points, points at 5 % and 10 % of every
adjoining segment (level 1), and 30 points on every segment (level 2).

Each structure is written as a :class:`pydftb.bandref.BandRef` to ``OUT/bands/<name>.json``:

* occupations are recomputed from the SCF Fermi level with the same Gaussian, so path runs
  have meaningful ones;
* ``n_semicore`` = (GPAW valence electrons - DFTB valence electrons) / 2, with DFTB's valence
  from PTBP's valence_shell: Li 1 (2s1), O 6 (2s2 2p4);
* ``meta`` holds code, version, ecut, kspacing, group, source, scaling and the SCF energy.

``OUT/bands/energies.json`` collects the SCF energies, E(sigma->0) and the free energy.  They
check the GPAW relative energies along each EOS against VASP's; they are never mixed into a
fit (one DFT code per fit, A0 section 1).

    # local test of what would run (no GPAW import)
    python bands_gpaw.py --out ../runs/d1_li_o --dry-run
    # the calculations (conda env gpaw); MPI: only rank 0 writes files
    mpirun -np 8 gpaw python bands_gpaw.py --out ../runs/d1_li_o [--only li_bcc,dimer_O2]

A finished structure (its JSON exists) is skipped; ``--force`` recomputes it.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import EX07_DATA, REPO, setup  # noqa: E402

setup()

EOS_GROUPS = ("eos_li_bcc", "eos_li_fcc", "eos_li2o", "eos_li2o2")
EOS_INDICES = (3, 5, 7)                         # 0.952, 1.000, 1.048 V0 (np.linspace(0.88, 1.12, 11))
DIMER_TARGETS = {"dimer_Li2": (1.6, 2.0, 2.4, 2.7, 3.5),     # DFT minimum near 2.67 A
                 "dimer_LiO": (1.4, 1.6, 1.8, 2.2, 3.0),
                 "dimer_O2": (1.1, 1.2, 1.3, 1.6, 2.1)}
DFTB_VALENCE = {"Li": 1.0, "O": 6.0}            # PTBP valence_shell: Li 2s1 2p0, O 2s2 2p4
KSPACING = 0.15                                 # 1/A, as the VASP training data
SIGMA = 0.05                                    # eV, Gaussian (ISMEAR 0)
HCP_COA = 1.633


def eos_scaling(index: int) -> float:
    return float(np.linspace(0.88, 1.12, 11)[index])


def kgrid(atoms, kspacing: float = KSPACING) -> list:
    """VASP's KSPACING rule, Gamma-centred: N_i = max(1, ceil(2 pi |b_i| / kspacing))."""
    rec = atoms.cell.reciprocal()
    return [max(1, int(math.ceil(2 * math.pi * np.linalg.norm(b) / kspacing - 1e-9))) for b in rec]


def reference_structures(data_dir: str = EX07_DATA, dimers: bool = True) -> list:
    """[{name, group, kind, atoms, scaling, source}] of D1 (and of vasp_bands.py)."""
    import ase.io
    from ase.build import bulk
    train = {a.info["structure_name"]: a for a in ase.io.read(os.path.join(data_dir, "train.xyz"), ":")}
    out = []
    for g in EOS_GROUPS:
        for k in EOS_INDICES:
            name = f"{g}_{k:02d}"
            at = train[name].copy()
            at.calc = None
            f = eos_scaling(k)
            out.append({"name": f"{g[4:]}_v{f:.3f}", "group": g[4:], "kind": "path", "atoms": at,
                        "scaling": f, "source": f"examples/07_battery_li_li2o/data/train.xyz:{name}"})
    v_atom = train["eos_li_bcc_05"].get_volume() / len(train["eos_li_bcc_05"])
    a = (4.0 * v_atom / (math.sqrt(3.0) * HCP_COA)) ** (1.0 / 3.0)
    hcp0 = bulk("Li", "hcp", a=a, c=HCP_COA * a)
    for k in EOS_INDICES:
        f = eos_scaling(k)
        at = hcp0.copy()
        at.set_cell(hcp0.cell * f ** (1.0 / 3.0), scale_atoms=True)
        out.append({"name": f"li_hcp_v{f:.3f}", "group": "li_hcp", "kind": "path", "atoms": at,
                    "scaling": f, "source": f"built: hcp Li at the bcc V0 per atom, c/a {HCP_COA}"})
    if dimers:
        sr = ase.io.read(os.path.join(data_dir, "short_range.xyz"), ":")
        for fam, targets in DIMER_TARGETS.items():
            members = [x for x in sr if x.info.get("group") == fam]
            if not members:
                raise ValueError(f"no {fam} in short_range.xyz")
            r = np.array([x.get_distance(0, 1, mic=True) for x in members])
            for t in targets:
                j = int(np.argmin(np.abs(r - t)))
                at = members[j].copy()
                at.calc = None
                out.append({"name": f"{fam}_{r[j]:.2f}", "group": fam, "kind": "molecule",
                            "atoms": at, "scaling": None,
                            "source": f"examples/07_battery_li_li2o/data/short_range.xyz:"
                                      f"{members[j].info['structure_name']}"})
    names = [s["name"] for s in out]
    assert len(names) == len(set(names)), names
    return out


def band_kpoints(atoms, points_per_segment: int = 30):
    from pydftb.bandref import special_kpoints
    return special_kpoints(atoms, fractions=(0.05, 0.10), points_per_segment=points_per_segment)


def dftb_valence(atoms) -> float:
    return float(sum(DFTB_VALENCE[s] for s in atoms.get_chemical_symbols()))


def empty_bands(atoms) -> int:
    """Empty bands to compute: enough for CBM + 2 eV in these small cells."""
    return max(10, 4 * len(atoms))


def gaussian_occupations(eps, fermi, sigma=SIGMA):
    from scipy.special import erfc
    return 0.5 * erfc((np.asarray(eps) - fermi) / sigma)


def _rank() -> int:
    try:
        from gpaw.mpi import world
        return world.rank
    except Exception:                            # noqa: BLE001  (no GPAW: one process)
        return 0


def compute(item: dict, out_dir: str, ecut: float, points: int, parallel: dict | None,
            log=print) -> dict:
    """GPAW SCF (+ fixed-density path run) of one structure -> BandRef JSON (rank 0 writes)."""
    from gpaw import GPAW, PW
    from pydftb.bandref import BandRef
    import gpaw
    at = item["atoms"].copy()
    name = item["name"]
    txt_dir = os.path.join(out_dir, "gpaw_txt")
    if _rank() == 0:
        os.makedirs(txt_dir, exist_ok=True)
    n = kgrid(at) if item["kind"] == "path" else [1, 1, 1]
    kw = dict(mode=PW(ecut), xc="PBE", kpts={"size": n, "gamma": True},
              occupations={"name": "methfessel-paxton", "width": SIGMA, "order": 0},
              convergence={"energy": 1e-6, "density": 1e-6}, spinpol=False,
              txt=os.path.join(txt_dir, f"{name}_scf.txt"))
    if item["kind"] == "molecule":
        kw["nbands"] = -8
        kw["symmetry"] = "off"
    if parallel:
        kw["parallel"] = parallel
    t0 = time.perf_counter()
    calc = GPAW(**kw)
    at.calc = calc
    e0 = float(at.get_potential_energy())                    # extrapolated to zero width
    try:
        efree = float(at.get_potential_energy(force_consistent=True))
    except Exception:                            # noqa: BLE001  (older interfaces)
        efree = None
    fermi = float(calc.get_fermi_level())
    nel = float(calc.get_number_of_electrons())
    t_scf = time.perf_counter() - t0
    if item["kind"] == "path":
        k, labels, level, segments = band_kpoints(at, points)
        nb = -empty_bands(at)
        bc = calc.fixed_density(kpts=k, symmetry="off", nbands=nb, convergence={"bands": -4},
                                txt=os.path.join(txt_dir, f"{name}_bands.txt"))
        ibz = np.asarray(bc.get_ibz_k_points(), float)
        if ibz.shape != k.shape or not np.allclose(ibz, k, atol=1e-8):
            raise RuntimeError(f"{name}: GPAW changed the band k-points (symmetry must be off)")
        eps = np.array([[bc.get_eigenvalues(kpt=i, spin=0) for i in range(len(k))]])
        kweights = np.ones(len(k))
    else:
        k = np.zeros((1, 3))
        labels, level, segments = ["G"], np.zeros(1, int), []
        eps = np.array([[calc.get_eigenvalues(kpt=0, spin=0)]])
        kweights = np.ones(1)
    t_band = time.perf_counter() - t0 - t_scf
    extra = nel - dftb_valence(at)
    if extra < -1e-6 or abs(extra / 2 - round(extra / 2)) > 1e-6:
        raise RuntimeError(f"{name}: GPAW has {nel} valence electrons, DFTB {dftb_valence(at)}")
    meta = {"code": "GPAW", "version": gpaw.__version__, "xc": "PBE", "ecut_eV": ecut,
            "kspacing": KSPACING if item["kind"] == "path" else "gamma", "scf_kgrid": n,
            "smearing": f"gaussian {SIGMA} eV (MP order 0)", "group": item["group"],
            "source": item["source"], "scaling": item["scaling"], "energy_sigma0_eV": e0,
            "free_energy_eV": efree, "n_valence_gpaw": nel, "seconds_scf": round(t_scf, 1),
            "seconds_bands": round(t_band, 1)}
    ref = BandRef(name=name, kind=item["kind"], kpts=k, kweights=kweights, energies=eps,
                  occupations=gaussian_occupations(eps, fermi), fermi=fermi,
                  n_semicore=int(round(extra / 2)), labels=list(labels), level=level,
                  segments=segments, meta=meta).set_atoms(item["atoms"])
    if _rank() == 0:
        ref.to_json(os.path.join(out_dir, "bands", f"{name}.json"))
    log(f"[d1] {name}: E {e0:.6f} eV, E_F {fermi:.3f} eV, {len(k)} k, "
        f"{eps.shape[2]} bands, n_semicore {ref.n_semicore}, scf {t_scf:.0f} s, bands {t_band:.0f} s")
    return {"name": name, "group": item["group"], "kind": item["kind"], "natoms": len(at),
            "volume_per_atom": float(at.get_volume() / len(at)) if item["kind"] == "path" else None,
            "scaling": item["scaling"], "energy_sigma0_eV": e0, "free_energy_eV": efree,
            "fermi_eV": fermi, "source": item["source"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="output directory (bands/ and logs/ inside)")
    ap.add_argument("--only", default=None, help="comma-separated names or groups")
    ap.add_argument("--ecut", type=float, default=800.0, help="plane-wave cut-off (eV)")
    ap.add_argument("--points", type=int, default=30, help="path points per segment (level 2)")
    ap.add_argument("--no-dimers", action="store_true")
    ap.add_argument("--parallel", default=None,
                    help='GPAW parallel dict as JSON, e.g. \'{"kpt": 4}\' (default: GPAW\'s choice)')
    ap.add_argument("--force", action="store_true", help="recompute existing JSON files")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the structures, k-grids and band-path sizes; no GPAW import")
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    items = reference_structures(dimers=not args.no_dimers)
    if args.only:
        want = {w.strip() for w in args.only.split(",")}
        items = [s for s in items if s["name"] in want or s["group"] in want]
    if args.dry_run:
        rows = []
        for s in items:
            at = s["atoms"]
            nk = len(band_kpoints(at, args.points)[0]) if s["kind"] == "path" else 1
            rows.append({"name": s["name"], "group": s["group"], "kind": s["kind"],
                         "natoms": len(at), "scf_kgrid": kgrid(at) if s["kind"] == "path" else [1, 1, 1],
                         "band_kpoints": nk, "dftb_valence": dftb_valence(at),
                         "empty_bands": empty_bands(at), "source": s["source"]})
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "d1_dry_run.json"), "w") as fh:
            json.dump(rows, fh, indent=1)
        for r in rows:
            print(f"{r['name']:22s} {r['kind']:8s} {r['natoms']:3d} atoms  scf k {r['scf_kgrid']}  "
                  f"band k {r['band_kpoints']}")
        print(f"{len(rows)} structures -> {os.path.join(out, 'd1_dry_run.json')}")
        return 0
    parallel = json.loads(args.parallel) if args.parallel else None
    root = _rank() == 0
    from utils.runlog import session
    ctx = session("d1_bands_gpaw", vars(args), root=os.path.join(out, "logs")) if root \
        else contextlib.nullcontext()
    with ctx as log:
        energies_path = os.path.join(out, "bands", "energies.json")
        energies = {}
        if os.path.exists(energies_path):
            with open(energies_path) as fh:
                energies = json.load(fh)
        for s in items:
            target = os.path.join(out, "bands", f"{s['name']}.json")
            if os.path.exists(target) and not args.force:
                print(f"[d1] skip {s['name']} (done)")
                continue
            try:
                rec = compute(s, out, args.ecut, args.points, parallel)
                energies[s["name"]] = rec
                if log is not None:
                    log.event("structure", name=s["name"], energy=rec["energy_sigma0_eV"])
            except Exception as exc:                 # noqa: BLE001  (next structure)
                print(f"[d1] {s['name']} FAILED: {type(exc).__name__}: {exc}")
                if log is not None:
                    log.event("failed", name=s["name"], error=f"{type(exc).__name__}: {exc}")
            if root:
                os.makedirs(os.path.dirname(energies_path), exist_ok=True)
                with open(energies_path, "w") as fh:
                    json.dump(energies, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
