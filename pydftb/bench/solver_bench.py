"""Diagonalisation vs SP2 purification (dense, FP64 / FP32+refinement)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import scipy.linalg as sla
from pydftb.skf import SKSet
from pydftb.hamiltonian import Basis, build_pairs, fold, onsite_energies
from pydftb.solvers import sp2
from pydftb.constants import BOHR
from systems import water_box

SK = os.environ.get("SKDIR", "/tmp/claude-0/skf2")
sk = SKSet(SK, ["H", "O"], {"H": 0, "O": 1})


def hs(n):
    at = water_box(n)
    sym = at.get_chemical_symbols()
    b = Basis.build(sym, sk)
    pb = build_pairs(sym, at.positions / BOHR, at.cell.array / BOHR, at.pbc, sk, b)
    h = fold(pb, pb.h, b.norb, None, onsite_energies(b, sk))
    s = fold(pb, pb.s, b.norb, None, np.ones(b.norb))
    nelec = sum(sk.occupation[x].sum() for x in sym)
    return len(at), h, s, nelec


rows = []
for n in [int(x) for x in sys.argv[1:]] or [6, 8, 10]:
    natom, h, s, nelec = hs(n)
    N = h.shape[0]
    nocc = int(round(nelec / 2))
    r = {"natom": natom, "N": N}
    t = time.perf_counter(); e, c = sla.eigh(h, s, driver="gvd"); r["t_eigh_gen"] = time.perf_counter() - t
    rho_ref = 2 * c[:, :nocc] @ c[:, :nocc].T
    E_ref = np.sum(rho_ref * h)
    r["gap_eV"] = (e[nocc] - e[nocc - 1]) * 27.2114
    t = time.perf_counter()
    w, v = np.linalg.eigh(s); z = (v / np.sqrt(w)) @ v.T
    r["t_loewdin"] = time.perf_counter() - t
    hp = z @ h @ z
    a = np.random.rand(N, N); t = time.perf_counter(); a @ a; tg = time.perf_counter() - t
    r["gemm_gflops_fp64"] = 2 * N ** 3 / tg / 1e9
    a32 = a.astype(np.float32); t = time.perf_counter(); a32 @ a32; tg32 = time.perf_counter() - t
    r["gemm_gflops_fp32"] = 2 * N ** 3 / tg32 / 1e9
    for name, dt, ref in [("sp2_fp64", np.float64, 0), ("sp2_fp32+2xMcWeeny64", np.float32, 2)]:
        t = time.perf_counter()
        x, nit = sp2(hp, nocc, dt, ref)
        rho = 2 * z @ x @ z
        r[f"t_{name}"] = time.perf_counter() - t
        r[f"iters_{name}"] = nit
        r[f"dE_per_atom_meV_{name}"] = (np.sum(rho * h) - E_ref) / natom * 27211.4
        r[f"max_drho_{name}"] = float(np.abs(rho - rho_ref).max())
        r[f"idem_{name}"] = float(np.abs(x @ x - x).max())
    # sparsity of the density matrix in the orthogonal basis
    xr = (z @ rho_ref @ z) / 2  # not exactly X but close; use projector from eig instead
    xo = sla.sqrtm(s).real @ (rho_ref / 2) @ sla.sqrtm(s).real if N <= 3500 else None
    if xo is not None:
        for th in (1e-3, 1e-5):
            r[f"nnz_row_X_{th}"] = float((np.abs(xo) > th).sum() / N)
        r["nnz_row_H"] = float((np.abs(hp) > 1e-5).sum() / N)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    rows.append(r)
json.dump(rows, open(os.path.join(os.path.dirname(__file__), "results_solver.json"), "w"), indent=1)
