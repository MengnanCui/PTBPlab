"""O(N) sparse SP2 on large water boxes; accuracy vs dense where affordable."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import scipy.linalg as sla
from pydftb.skf import SKSet
from pydftb.hamiltonian import Basis, build_pairs, onsite_energies
from pydftb.sparse import to_csr, inv_sqrt_newton_schulz, sp2_sparse, _drop
from pydftb.constants import BOHR
from systems import water_box

SK = os.environ.get("SKDIR", "/tmp/claude-0/skf2")
sk = SKSet(SK, ["H", "O"], {"H": 0, "O": 1})
TH = float(os.environ.get("SP2_THRESH", "1e-5"))
rows = []
for n in [int(x) for x in sys.argv[1:]]:
    at = water_box(n)
    sym = at.get_chemical_symbols()
    r = {"natom": len(at)}
    t = time.perf_counter()
    b = Basis.build(sym, sk)
    pb = build_pairs(sym, at.positions / BOHR, at.cell.array / BOHR, at.pbc, sk, b)
    h = to_csr(pb, pb.h, b.norb, onsite_energies(b, sk))
    s = to_csr(pb, pb.s, b.norb, np.ones(b.norb))
    r["t_build"] = time.perf_counter() - t
    r["N"] = b.norb
    nocc = sum(sk.occupation[x].sum() for x in sym) / 2
    t = time.perf_counter(); z, nns = inv_sqrt_newton_schulz(s, thresh=TH * 0.1); r["t_inv_sqrt"] = time.perf_counter() - t
    t = time.perf_counter()
    hp = _drop(z @ h @ z, TH * 0.1)
    x, nit = sp2_sparse(hp, nocc, TH)
    rho = (2 * z @ x @ z).tocsr()
    r["t_sp2"] = time.perf_counter() - t
    r["sp2_iter"], r["ns_iter"] = nit, nns
    r["nnz_row_S"] = s.nnz / b.norb
    r["nnz_row_Z"] = z.nnz / b.norb
    r["nnz_row_X"] = x.nnz / b.norb
    e = (rho.multiply(h.T)).sum()
    r["E"] = float(e)
    if b.norb <= 6000:
        hd, sd = h.toarray(), s.toarray()
        t = time.perf_counter(); ev, c = sla.eigh(hd, sd, driver="gvd"); r["t_eigh_dense"] = time.perf_counter() - t
        no = int(round(nocc))
        e_ref = np.sum((2 * c[:, :no] @ c[:, :no].T) * hd)
        r["dE_per_atom_meV"] = (e - e_ref) / len(at) * 27211.4
        r["dN_elec"] = float((rho.multiply(s.T)).sum() - 2 * nocc)
    print(json.dumps({k: (float('%.4g' % v) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    rows.append(r)
json.dump(rows, open(os.path.join(os.path.dirname(__file__), f"results_sparse_{TH:g}.json"), "w"), indent=1)
