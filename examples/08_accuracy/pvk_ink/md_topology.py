"""MD topology check for the precursor ink (A0 plan, A1.3 and the acceptance line: stable MD, no
spurious bond breaking).  M409 with DFTB(PTBP) showed a spurious FA+ -> I- proton transfer
within 10 fs; this script runs NVT MD of ink structures with any model directory and watches
the bond topology.

* Bonds: d_ij < scale x (R_cov,i + R_cov,j) (ASE's covalent radii); scale 1.20 for covalent
  pairs, 1.30 for Pb/Sn with their ligands (O, S, N, I), Cs-I 1.0, other Cs pairs not counted.
* A change counts when it persists for ``--persist`` consecutive samples (one sample every
  ``--every`` steps), so vibrations do not flicker.
* Reported per run: covalent bonds broken and formed per element pair (any one fails the run),
  proton transfers (an H whose heavy-atom partner changes, e.g. N -> I, N -> O; any one fails the
  run), metal-ligand coordination of every Pb/Sn (initial, final, range; reported, not failing:
  ligand exchange can be real), the shortest distance per element pair, and a failed SCC.
* Dispersion: PBE-D3(BJ) (simple-dftd3) is on by default, as the models are used in the ink;
  ``--no-d3`` switches it off.  The model's own ``dispersion`` spec is used if it has one.

    python md_topology.py --model ../li_o/models/X --config scan_ink.yaml --out runs/md_topo \\
        [--names ion_fa_i ion_fa_i_dmso ion_pbi4_2fa cplx_PbI2_DMSO2_o0] [--temps 300] [--ps 20]

Writes ``<out>/<name>_<T>K.json`` per run (a finished run is skipped on a rerun; ``--force``
redoes it), the trajectory every ``--every`` steps (``<name>_<T>K.traj``),
``md_topology_results.json`` and ``md_topology_summary.md``; logs in ``<out>/logs/md_topology/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import setup  # noqa: E402

setup()

DEFAULT_NAMES = ["ion_fa_i", "ion_fa_i_dmso", "ion_fa_i_dmf", "ion_pbi3_fa", "ion_sni3_fa",
                 "ion_pbi4_2fa", "ion_sni4_2fa", "cplx_PbI2_DMSO2_o0", "cplx_SnI2_DMF2_o0"]
METALS = {"Pb", "Sn"}
LIGAND = {"O", "S", "N", "I"}
COVALENT_SCALE, METAL_SCALE, CSI_SCALE = 1.20, 1.30, 1.00


def cutoffs(symbols, covalent=COVALENT_SCALE, metal=METAL_SCALE, csi=CSI_SCALE):
    """(n, n) bond cut-offs (A; 0 = never a bond) and the (n, n) kind: 1 covalent, 2 metal-ligand."""
    from ase.data import atomic_numbers, covalent_radii
    s = list(symbols)
    r = np.array([covalent_radii[atomic_numbers[x]] for x in s])
    n = len(s)
    cut = np.zeros((n, n))
    kind = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = s[i], s[j]
            pair = {a, b}
            if "Cs" in pair:
                if pair == {"Cs", "I"}:
                    cut[i, j], kind[i, j] = csi * (r[i] + r[j]), 2
                continue
            if pair & METALS and (pair - METALS) & LIGAND:
                cut[i, j], kind[i, j] = metal * (r[i] + r[j]), 2
            elif pair & METALS:
                continue                    # metal-metal, metal-C/H: not bonds here
            else:
                cut[i, j], kind[i, j] = covalent * (r[i] + r[j]), 1
    return cut + cut.T, kind + kind.T


class Topology:
    """Persistent bond topology of one trajectory (see the module docstring)."""

    def __init__(self, atoms, persist, scales):
        self.sym = np.array(atoms.get_chemical_symbols())
        self.cut, self.kind = cutoffs(self.sym, *scales)
        self.persist = int(persist)
        self.pbc = bool(np.any(atoms.pbc))
        d = atoms.get_all_distances(mic=self.pbc)
        self.bonded = (d < self.cut) & (self.cut > 0)
        np.fill_diagonal(self.bonded, False)
        self.initial = self.bonded.copy()
        self.count = np.zeros_like(self.bonded, int)
        self.events = []
        heavy = self.sym != "H"
        self.heavy = heavy
        self.owner = {int(h): self._owner(int(h), d) for h in np.flatnonzero(~heavy)}
        self.transfers = []
        self.metals = {int(m): self._coord(int(m)) for m in np.flatnonzero(np.isin(self.sym, list(METALS)))}
        self.metal_track = {m: [c] for m, c in self.metals.items()}
        self.shortest = defaultdict(lambda: np.inf)

    def _owner(self, h, d):
        partners = [j for j in np.flatnonzero(self.bonded[h]) if self.heavy[j]]
        if not partners:
            return None
        return int(min(partners, key=lambda j: d[h, j]))

    def _coord(self, m):
        return int(sum(1 for j in np.flatnonzero(self.bonded[m]) if self.kind[m, j] == 2))

    def update(self, atoms, t_fs):
        d = atoms.get_all_distances(mic=self.pbc)
        iu = np.triu_indices(len(self.sym), 1)
        for x, y, dist in zip(self.sym[iu[0]], self.sym[iu[1]], d[iu]):
            key = "-".join(sorted((x, y)))
            self.shortest[key] = min(self.shortest[key], float(dist))
        now = (d < self.cut) & (self.cut > 0)
        np.fill_diagonal(now, False)
        diff = now != self.bonded
        self.count = np.where(diff, self.count + 1, 0)
        flip = np.triu(self.count >= self.persist, 1)
        for i, j in zip(*np.nonzero(flip)):
            self.bonded[i, j] = self.bonded[j, i] = now[i, j]
            self.count[i, j] = self.count[j, i] = 0
            self.events.append({"t_fs": t_fs, "i": int(i), "j": int(j),
                                "pair": "-".join(sorted((self.sym[i], self.sym[j]))),
                                "kind": "covalent" if self.kind[i, j] == 1 else "metal-ligand",
                                "change": "formed" if now[i, j] else "broken"})
        if np.any(flip):
            for h in list(self.owner):
                o = self._owner(h, d)
                if o != self.owner[h] and o is not None and self.owner[h] is not None:
                    self.transfers.append({"t_fs": t_fs, "H": h, "from": f"{self.sym[self.owner[h]]}{self.owner[h]}",
                                           "to": f"{self.sym[o]}{o}",
                                           "pair": f"{self.sym[self.owner[h]]}->{self.sym[o]}"})
                if o is not None:
                    self.owner[h] = o
            for m in self.metals:
                self.metal_track[m].append(self._coord(m))

    def summary(self) -> dict:
        cov = [e for e in self.events if e["kind"] == "covalent"]
        met = [e for e in self.events if e["kind"] == "metal-ligand"]
        return {"n_bonds_initial": int(np.triu(self.initial, 1).sum()),
                "covalent_changes": dict(Counter(f"{e['pair']} {e['change']}" for e in cov)),
                "metal_ligand_changes": dict(Counter(f"{e['pair']} {e['change']}" for e in met)),
                "proton_transfers": dict(Counter(t["pair"] for t in self.transfers)),
                "metal_coordination": {f"{self.sym[m]}{m}": {"initial": tr[0], "final": tr[-1],
                                                              "min": min(tr), "max": max(tr)}
                                       for m, tr in self.metal_track.items()},
                "events": (cov + met)[:200], "transfers": self.transfers[:100],
                "shortest": {k: v for k, v in self.shortest.items()},
                "topology_ok": not cov and not self.transfers}


def run_one(model, atoms, temperature, args, traj_path, dispersion):
    from ase import units
    from ase.io.trajectory import Trajectory
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from common.models import converging_calculator
    a = atoms.copy()
    try:
        a.calc, attempt = converging_calculator(model, a, "free", dispersion, reuse_charges=True)
    except Exception as exc:                  # noqa: BLE001  (no SCC setting converges at t = 0)
        return {"temperature": temperature, "ps": args.ps, "t_final_fs": 0.0, "ok": False,
                "failed": f"{type(exc).__name__}: {exc}", "topology_ok": None, "dispersion": dispersion,
                "covalent_changes": {}, "metal_ligand_changes": {}, "proton_transfers": {},
                "metal_coordination": {}, "scc_attempt": None}
    MaxwellBoltzmannDistribution(a, temperature_K=temperature,
                                 rng=np.random.default_rng(args.seed))
    Stationary(a)
    if not np.any(a.pbc):
        ZeroRotation(a)
    topo = Topology(a, args.persist, (args.scale, args.scale_metal, CSI_SCALE))
    dyn = Langevin(a, args.dt * units.fs, temperature_K=temperature,
                   friction=args.friction / units.fs, rng=np.random.default_rng(args.seed + 1))
    traj = Trajectory(traj_path, "w", a)
    temps = []
    state = {"t": 0.0}

    def observe():
        t = dyn.nsteps * args.dt
        state["t"] = t
        topo.update(a, t)
        temps.append(float(a.get_temperature()))
        traj.write(a)

    dyn.attach(observe, interval=args.every)
    n_steps = int(round(args.ps * 1000.0 / args.dt))
    failed = None
    try:
        dyn.run(n_steps)
    except Exception as exc:                  # noqa: BLE001  (an SCC failure ends the run, recorded)
        failed = f"{type(exc).__name__}: {exc}"
    traj.close()
    out = topo.summary()
    out.update(temperature=temperature, ps=args.ps, dt_fs=args.dt, steps=n_steps, scc_attempt=attempt,
               t_final_fs=state["t"], failed=failed, dispersion=dispersion,
               temperature_mean=float(np.mean(temps)) if temps else None,
               temperature_trace=temps[:: max(1, len(temps) // 50)])
    out["ok"] = bool(out["topology_ok"] and failed is None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="a model directory (common/models.py)")
    ap.add_argument("--config", default=os.path.join(HERE, "scan_ink.yaml"),
                    help="scan config whose dftb section the model uses")
    ap.add_argument("--out", required=True)
    ap.add_argument("--structures", default=os.path.join(HERE, "data", "structures.xyz"))
    ap.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    ap.add_argument("--temps", nargs="+", type=float, default=[300.0])
    ap.add_argument("--ps", type=float, default=20.0, help="MD length (ps)")
    ap.add_argument("--dt", type=float, default=0.5, help="time step (fs)")
    ap.add_argument("--friction", type=float, default=0.005, help="Langevin friction (1/fs)")
    ap.add_argument("--every", type=int, default=20, help="steps between topology samples")
    ap.add_argument("--persist", type=int, default=10, help="samples a change must last")
    ap.add_argument("--scale", type=float, default=COVALENT_SCALE)
    ap.add_argument("--scale-metal", type=float, default=METAL_SCALE)
    ap.add_argument("--no-d3", action="store_true", help="switch PBE-D3(BJ) off")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    import ase.io
    from common import models as M
    from utils import accuracy as acc
    from utils.runlog import session
    cfg = acc.load_config(args.config)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    with session("md_topology", vars(args), root=os.path.join(out, "logs")) as log:
        model = M.load_model(args.model, dftb=cfg["dftb"])
        dispersion = not args.no_d3
        if dispersion:
            model.d3()                        # fails early, with the install hint, if dftd3 is missing
        frames = {a.info.get("name"): a for a in ase.io.read(args.structures, ":")}
        missing = [n for n in args.names if n not in frames]
        if missing:
            raise SystemExit(f"not in {args.structures}: {missing}")
        for name in args.names:
            for t in args.temps:
                key = f"{name}_{int(t)}K"
                path = os.path.join(out, f"{key}.json")
                if os.path.exists(path) and not args.force:
                    log.event("skip", run=key)
                    continue
                with log.step("md", run=key, atoms=len(frames[name])):
                    r = run_one(model, frames[name], t, args, os.path.join(out, f"{key}.traj"),
                                dispersion)
                with open(path + ".tmp", "w") as fh:
                    json.dump(M._clean(r), fh, indent=1)
                os.replace(path + ".tmp", path)
                log.event("md_result", run=key, ok=r["ok"], covalent=r["covalent_changes"],
                          transfers=r["proton_transfers"], failed=r["failed"])
        results = {}
        for name in args.names:
            for t in args.temps:
                p = os.path.join(out, f"{name}_{int(t)}K.json")
                if os.path.exists(p):
                    results[f"{name}_{int(t)}K"] = json.load(open(p))
        with open(os.path.join(out, "md_topology_results.json"), "w") as fh:
            json.dump({"model": model.path, "dispersion": dispersion, "runs": results}, fh, indent=1)
        L = [f"# MD topology of {model.path}", "",
             f"NVT {args.ps} ps, dt {args.dt} fs, D3 {'on' if dispersion else 'off'}; a run fails on "
             "a covalent bond change, a proton transfer or a failed SCC", "",
             "| run | ok | covalent changes | proton transfers | metal coordination (init -> final) | t (fs) | failed |",
             "|---|---|---|---|---|---|---|"]
        for k, r in results.items():
            mc = "; ".join(f"{m} {v['initial']}->{v['final']}" for m, v in r["metal_coordination"].items())
            L.append(f"| {k} | {r['ok']} | {r['covalent_changes'] or ''} | {r['proton_transfers'] or ''} | "
                     f"{mc} | {r['t_final_fs']:.0f} | {r['failed'] or ''} |")
        with open(os.path.join(out, "md_topology_summary.md"), "w") as fh:
            fh.write("\n".join(L) + "\n")
        print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
