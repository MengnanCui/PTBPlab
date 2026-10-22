"""Layered checks of the gradient of PTBP's loss (milestone M4c, design section 9).

* L1 -- pydftb on PTBP structures, along the real hotcent tangent t: the torch hf
  Edot = <dE/dtheta, t> against the same contraction in NumPy (P.H0[t] + (P v - W).S[t])
  and against central differences E(theta +- h t) (SCC 1e-12); the implicit
  <d(w.F)/dtheta, t> (DFTB+'s 0.01 Bohr derivatives, as PTBP) against differences.
* L2 -- the hotcent step study: the tangent and Edot at h = 0.005 ... 0.32 Bohr.
* L3 -- at fixed sigma*, dmu/dr0 against [mu(r0 + h, sigma*) - mu(r0 - h, sigma*)] / 2h,
  with the whole BO step (hotcent, DFTB) at r0 +- h and the same h as the tangent.
* L4 -- the same differences of PTBP's own loss (with its sigma search): reported only.

Example (conda environment with hotcent; about 10 minutes on 4 cores)::

    python tools/grad_check.py dft/dft.xyz --levels 1 2 3 4 --json grad_check.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POINT = (2.749, 12.036)            # the published Cu parameters
PICK = (0, 5, 33, 60)              # Gamma-only and k-point structures of dft/dft.xyz


def _hotcent(r0_w, r0_d, outdir, workers=4):
    from utils.hotcent_tangent import run_hotcent
    hp = {"xc": "GGA_X_PBE+GGA_C_PBE", "superposition": "density", "opt_elem": [], "label": "ptbp"}
    run_hotcent(hp, ({"Cu": [r0_w, r0_d]}, {"Cu": 2.0}), outdir, workers)
    return outdir


def _tight_engine(atoms, skdir, kgrid, scc_tol=1e-12, temperature=0.0):
    """PTBP's pydftb calculator for ``atoms`` (utils.engines) with a tight SCC (and, for
    ``temperature`` > 0, Fermi smearing at that temperature in K)."""
    import utils.engines as eng
    eng.configure(engine="pydftb")
    calc = eng.make_calculator(atoms, {"kpts": kgrid, "calc_type": "static", "poly_rep": "Yes",
                                       "SCC": "Yes", "slater_p": os.path.join(skdir, "")})
    calc.engine_opts["scc_tol"] = scc_tol
    if temperature > 0:
        calc.engine_opts["smearing"] = ("fermi", float(temperature))
    return calc


def numpy_tangent(eng, species, pos, cell, pbc, tables):
    """<dF/dtheta, t> of the tables in NumPy: sum P h[t] + (P (v_mu + v_nu)/2 - W) s[t] with
    the engine's converged densities (H0 and S are linear in the tables)."""
    from pydftb.hamiltonian import pair_values
    res, st, sc = eng._run(species, pos, cell, pbc, True, None, None)
    p_e, w_e, p_m = eng._pair_densities(st, sc)
    v, vs = eng._potentials(st, sc.q_out)
    pb = st.pairs
    h_t, s_t = pair_values(pb, eng.sk, pb.vec, tables=tables)
    coef = p_e * 0.5 * (v[pb.rows] + v[pb.cols])
    if w_e is not None:
        coef = coef - w_e
    if p_m is not None:
        coef = coef + p_m * 0.5 * (vs[pb.rows] + vs[pb.cols])
    return float(np.sum(p_e * h_t) + np.sum(coef * s_t)), res


def level1(images, bolt, kdens, skdir, tangent, rows, temperature=0.0):
    """L1 on the structures PICK (with smearing at ``temperature`` K: the free energy F,
    whose hf derivative is exact at any T, and the implicit forces)."""
    import torch

    from ase.units import Bohr, Hartree
    from pydftb.autodiff import Params, evaluate
    from pydftb.autodiff.api import _engine_for
    from pydftb.validation.gradcheck import _richardson
    from utils.dftb_stage import kpoint_grid
    torch.set_num_threads(1)
    t_cu = tangent.tables
    for i in PICK:
        a = images[i]
        calc = _tight_engine(a, skdir, kpoint_grid(a, kdens, bolt[i]), temperature=temperature)
        eng, labels, sk = calc._engine(a)
        pos, cell = a.positions / Bohr, a.cell.array / Bohr
        pbc = a.pbc
        quantity = "energy" if temperature == 0 else "free_energy"
        params = Params.from_skset(sk, eng.repulsive, requires_grad=("tables",))
        res = evaluate(eng, labels, pos, cell, pbc, params, mode="hf", forces=False)
        (g,) = torch.autograd.grad(getattr(res, quantity), [params.tables[("Cu", "Cu")]])
        e_torch = float((g * torch.as_tensor(t_cu[("Cu", "Cu")])).sum())
        e_np, _ = numpy_tangent(eng, labels, pos, cell, pbc, t_cu)

        def energy(h):
            p = Params.from_skset(sk, eng.repulsive)
            with torch.no_grad():
                p.tables[("Cu", "Cu")] += h * torch.as_tensor(t_cu[("Cu", "Cu")])
            return {"e": getattr(_engine_for(eng, p).calculate(labels, pos, cell, pbc,
                                                               forces=False), quantity)}
        rows.append({"level": 1, "structure": i, "quantity": "Edot (Ha/Bohr)", "ad": e_torch,
                     "ref": e_np, "rel": abs(e_torch - e_np) / max(abs(e_np), 1e-300),
                     "against": "NumPy contraction", "temperature": temperature})
        try:
            fd, spread = _richardson(energy, 1e-3)
        except RuntimeError as err:      # T = 0 metal: a level at the Fermi level (DFTB+ rule)
            rows.append({"level": 1, "structure": i, "quantity": "Edot (Ha/Bohr)", "ad": e_torch,
                         "ref": None, "rel": None, "against": "differences",
                         "error": f"{type(err).__name__}: {err}",
                         "temperature": temperature})
        else:
            rows.append({"level": 1, "structure": i, "quantity": "Edot (Ha/Bohr)", "ad": e_torch,
                         "ref": fd["e"], "rel": abs(e_torch - fd["e"]) / max(abs(fd["e"]), 1e-300),
                         "noise": spread["e"] / max(abs(fd["e"]), 1e-300),
                         "against": "differences", "temperature": temperature})
        if len(a) > 1:
            w = np.random.default_rng(i).choice([-1.0, 1.0], size=(len(a), 3))
            p2 = Params.from_skset(sk, eng.repulsive, requires_grad=("tables",))
            r2 = evaluate(eng, labels, pos, cell, pbc, p2, mode="implicit", forces=True)
            (g2,) = torch.autograd.grad((torch.as_tensor(w) * r2.forces).sum(),
                                        [p2.tables[("Cu", "Cu")]])
            f_ad = float((g2 * torch.as_tensor(t_cu[("Cu", "Cu")])).sum())

            def forces(h):
                p = Params.from_skset(sk, eng.repulsive)
                with torch.no_grad():
                    p.tables[("Cu", "Cu")] += h * torch.as_tensor(t_cu[("Cu", "Cu")])
                return {"f": float(np.sum(w * _engine_for(eng, p).calculate(labels, pos, cell,
                                                                           pbc).forces))}
            try:
                fd, spread = _richardson(forces, 1e-3)
            except RuntimeError as err:
                rows.append({"level": 1, "structure": i, "quantity": "w.Fdot (Ha/Bohr^2)",
                             "ad": f_ad, "ref": None, "rel": None, "against": "differences",
                             "error": f"{type(err).__name__}: {err}",
                         "temperature": temperature})
            else:
                rows.append({"level": 1, "structure": i, "quantity": "w.Fdot (Ha/Bohr^2)",
                             "ad": f_ad, "ref": fd["f"],
                             "rel": abs(f_ad - fd["f"]) / max(abs(fd["f"]), 1e-300),
                             "noise": spread["f"] / max(abs(fd["f"]), 1e-300),
                             "against": "differences", "temperature": temperature})
        print(f"[L1] structure {i}: done", flush=True)
    return rows


def level2(images, bolt, kdens, workdir, rows, h0=0.04):
    """L2: tangents at h0 2^k, k = -3..3, and Edot of the structures PICK."""
    import torch

    from ase.units import Bohr, Hartree
    from pydftb.autodiff import Params, evaluate
    from utils.dftb_stage import kpoint_grid
    from utils.hotcent_tangent import skf_parameters
    torch.set_num_threads(1)
    base = _hotcent(*POINT, os.path.join(workdir, "l2_base"))
    grads = {}
    for i in PICK:
        a = images[i]
        calc = _tight_engine(a, base, kpoint_grid(a, kdens, bolt[i]), scc_tol=1e-10)
        eng, labels, sk = calc._engine(a)
        params = Params.from_skset(sk, eng.repulsive, requires_grad=("tables",))
        res = evaluate(eng, labels, a.positions / Bohr, a.cell.array / Bohr, a.pbc, params,
                       forces=False)
        grads[i] = torch.autograd.grad(res.energy, [params.tables[("Cu", "Cu")]])[0].numpy()
    steps = [h0 * 2.0 ** k for k in range(-3, 4)]
    tans = {}
    for h in steps:
        plus = _hotcent(POINT[0] + h, POINT[1], os.path.join(workdir, f"l2_p{h}"))
        minus = _hotcent(POINT[0] - h, POINT[1], os.path.join(workdir, f"l2_m{h}"))
        tans[h] = ((skf_parameters(plus, ["Cu"])["tables"][("Cu", "Cu")]
                    - skf_parameters(minus, ["Cu"])["tables"][("Cu", "Cu")]) / (2 * h))
        shutil.rmtree(plus)
        shutil.rmtree(minus)
    for i in PICK:
        vals = [float(np.sum(grads[i] * tans[h])) * Hartree for h in steps]
        ref = vals[steps.index(h0)]
        rows.append({"level": 2, "structure": i, "quantity": "dE/dr0_w (eV/Bohr)",
                     "steps": steps, "values": vals,
                     "spread": (max(vals[1:-1]) - min(vals[1:-1])) / max(abs(ref), 1e-300)})
    scale = np.abs(tans[h0]).max(0) + 1e-300
    for a, b in zip(steps[:-1], steps[1:]):
        rel = np.abs(tans[a] - tans[b]) / scale
        rows.append({"level": 2, "quantity": "tangent", "steps": [a, b],
                     "rms": float(np.sqrt((rel ** 2).mean())), "median": float(np.median(rel)),
                     "max": float(rel.max())})
    shutil.rmtree(base)
    return rows


def level34(mode, dataset, workdir, rows, h=0.04, hd=0.2):
    """L3 and L4 at POINT: one evaluation with the gradient, then r0_w +- h and
    r0_d +- hd without (plain BO steps), with mu at fixed sigma* from their files."""
    import utils.engines as eng
    from tools.engine_parity import build_loss
    from utils import loss_grad as LG
    from utils.parallel import detect_cpu, set_plan
    from utils.recommender import make_plan, workload_from_dataset
    eng.configure(engine="pydftb")
    wd = Path(workdir) / f"l34_{mode}"
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    loss, _ = build_loss(dataset, mode, wd, 11)
    work = workload_from_dataset(loss.dft_input, loss.bolt_f11,
                                 kpts_density=loss.args_input.kpt_density,
                                 eos_points=loss.args_input.eos_points, engine="pydftb", mode=mode)
    set_plan(make_plan(detect_cpu("auto"), work, n_calls=1))
    cwd = os.getcwd()
    req = LG.GradRequest((h, hd))
    loss.grad_request = req
    t0 = time.perf_counter()
    mu0 = float(loss.Loss_EnergyGeometry(list(POINT)))
    t_grad = time.perf_counter() - t0
    loss.grad_request = None
    os.chdir(cwd)
    res = req.result
    if res.grad is None:
        raise RuntimeError(f"no gradient at {POINT}: {res.reason}")
    sigma = res.sigma_star
    mu_star = mu0 - res.gap                          # mu at sigma*, r0
    for name, step, delta in (("r0_w", h, (h, 0.0)), ("r0_d", hd, (0.0, hd))):
        mus, fixed = [], []
        for sgn in (1, -1):
            x = [POINT[0] + sgn * delta[0], POINT[1] + sgn * delta[1]]
            mus.append(float(loss.Loss_EnergyGeometry(x)))
            os.chdir(cwd)
            d = wd / loss.args_input.results_dir / f"opt_{x[0]:.3f}_{x[1]:.3f}"
            step_data = LG.StepData.from_files(loss, x[0], x[1], str(d))
            with contextlib.chdir(d):
                fixed.append(float(LG.mu(step_data, sigma)))
        part = res.parts[name]
        partial = part["energies"] + part["forces"] + part["penalty"]
        for level, ad, (plus, minus), centre in ((3, partial, fixed, mu_star),
                                                 (4, res.grad[name], mus, mu0)):
            fd = {"central": (plus - minus) / (2 * step), "left": (centre - minus) / step,
                  "right": (plus - centre) / step}
            # a jump of the loss (an EOS fit gone astray, ...) shows as one-sided
            # differences that disagree: compare with the closer one then
            jump = abs(fd["left"] - fd["right"]) > 0.25 * max(abs(fd["left"]), abs(fd["right"]))
            ref = min((fd["left"], fd["right"]), key=lambda v: abs(v - ad)) if jump else fd["central"]
            rows.append({"level": level, "mode": mode,
                         "quantity": (f"dmu/d{name} at sigma*" if level == 3 else f"dmu*/d{name}"),
                         "ad": ad, "ref": ref, "rel": abs(ad - ref) / max(abs(ref), 1e-300),
                         "h": step, "fd": fd, "jump": jump, "mu": [minus, centre, plus],
                         "kink": part["kink"]})
    rows.append({"level": 0, "mode": mode, "mu": mu0, "case": res.case, "sigma_hat": res.sigma_hat,
                 "sigma_star": res.sigma_star, "gap": res.gap, "grad": res.grad,
                 "parts": res.parts, "timings": res.timings, "wall_with_gradient": t_grad})
    set_plan(None)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--modes", nargs="+", default=["energygeometry", "dataset"])
    ap.add_argument("--step", type=float, default=0.04, help="r0_w step (Bohr)")
    ap.add_argument("--l34-steps", type=float, nargs="+", default=None,
                    help="r0_w steps of L3/L4 (default: --step)")
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--l1-temperature", type=float, nargs="+", default=[0.0],
                    help="L1 with Fermi smearing at these temperatures (K; 0: PTBP's T = 0)")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--confinement", default="selfconsistent",
                    choices=["selfconsistent", "perturbative"],
                    help="hotcent's confinement scheme (M4e; perturbative: as before M4e)")
    a = ap.parse_args(argv)
    from utils.hotpy import set_confinement
    set_confinement(a.confinement)
    import ase.io

    import utils.data as data
    from utils.hotcent_tangent import Tangent, skf_parameters
    dataset = a.dataset.resolve()
    workdir = Path(a.workdir or tempfile.mkdtemp(prefix="grad_check_")).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    images = ase.io.read(dataset, ":")
    _, bolt = data.get_bolt_factor(images[5::11], "boltzmann", 3000)
    rows = []
    if 1 in a.levels:
        base = _hotcent(*POINT, str(workdir / "l1_base"))
        plus = _hotcent(POINT[0] + a.step, POINT[1], str(workdir / "l1_plus"))
        minus = _hotcent(POINT[0] - a.step, POINT[1], str(workdir / "l1_minus"))
        t = Tangent("r0_w", a.step, tables={
            k: (v - skf_parameters(minus, ["Cu"])["tables"][k]) / (2 * a.step)
            for k, v in skf_parameters(plus, ["Cu"])["tables"].items()})
        for temp in a.l1_temperature:
            level1(images, bolt, 5.0, base, t, rows, temperature=temp)
    if 2 in a.levels:
        level2(images, bolt, 5.0, str(workdir), rows, a.step)
    for mode in a.modes:
        if 3 in a.levels or 4 in a.levels:
            for h in a.l34_steps or [a.step]:
                level34(mode, dataset, str(workdir / f"h{h}"), rows, h)
    for r in rows:
        print(json.dumps(r, default=float))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
