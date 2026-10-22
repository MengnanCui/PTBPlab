"""Slater-Koster file (SKF) reader and DFTB+-compatible interpolation.

Format reference: https://dftb.org/fileadmin/DFTB/public/misc/slakoformat.pdf

Integrals are numbered as in DFTB+ (``type/oldskdata.F90``): 20 per shell pair table,
    ff0 ff1 ff2 ff3 df0 df1 df2 dd0 dd1 dd2 pf0 pf1 pd0 pd1 pp0 pp1 sf0 sd0 sp0 ss0
(the digit is |m|), H in columns 0-19 and S in columns 20-39 of :meth:`SKFile.evaluate`.
The extended format (first line starting with ``@``, for f electrons) stores all 40; the
simple format stores the 10 spd integrals of H and of S, which go to the same places as in
DFTB+ (``iSKInterOld``), the f integrals being zero.
Row ``i`` (1-based) of the table is the integral at distance ``i * dr``;
like DFTB+, only the first ``ngrid - 1`` rows of the header's count are used.

Interpolation reproduces DFTB+'s default ``SlakoEqGrid`` (new scheme):
  * inside the grid: 8-point polynomial interpolation, 4 points to the right
  * beyond the last grid point: a 5th-order polynomial smoothly decaying to
    zero within ``DIST_FUDGE`` = 1 Bohr.
To make evaluation fast and give exact analytical derivatives, the degree-7
interpolating polynomial for every possible stencil is pre-computed once.
"""
from __future__ import annotations

import gzip
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .backend import ops

_READ_CACHE: dict = {}

N_INTER = 8          # points used for polynomial interpolation
N_RIGHT = 4          # points right of the evaluation point
DIST_FUDGE = 1.0     # tail length (Bohr) beyond the last grid point
DELTA_R = 1e-5       # displacement used by DFTB+ for tail derivatives

# monomial -> Lagrange weights of the 8-point stencil, nodes u = -3.5 .. 3.5
_VINV = np.linalg.inv(np.vander(np.arange(N_INTER) - 3.5, N_INTER, increasing=True))

N_INTEGRALS = 20     # H (or S) integrals per table; the S columns start here
# (l1, l2, |m|) -> column of the H integral, l1 <= l2 (DFTB+ parser.F90:getFullTable, skMap)
SK_COLUMN = {
    (3, 3, 0): 0, (3, 3, 1): 1, (3, 3, 2): 2, (3, 3, 3): 3,
    (2, 3, 0): 4, (2, 3, 1): 5, (2, 3, 2): 6,
    (2, 2, 0): 7, (2, 2, 1): 8, (2, 2, 2): 9,
    (1, 3, 0): 10, (1, 3, 1): 11,
    (1, 2, 0): 12, (1, 2, 1): 13,
    (1, 1, 0): 14, (1, 1, 1): 15,
    (0, 3, 0): 16, (0, 2, 0): 17, (0, 1, 0): 18, (0, 0, 0): 19,
}
# where the 10 integrals of the simple format go (dd0-2 pd0-1 pp0-1 sd0 sp0 ss0; iSKInterOld)
SIMPLE_COLUMNS = np.array([7, 8, 9, 12, 13, 14, 15, 17, 18, 19])


def _open_text(path: str):
    """Open a (possibly gzip-compressed, ``.gz``) text file."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _pair_from_filename(path: str) -> tuple[str, str]:
    """``.../A-B.skf[.gz]`` -> ("A", "B")."""
    name = os.path.basename(path)
    for ext in (".gz", ".skf"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    a, b = name.split("-")[:2]
    return a, b


def max_l_from_skf(path: str) -> int | None:
    """Highest angular momentum with non-zero occupation in a homonuclear SKF.

    Same rule as ASE's ``read_max_angular_momentum`` (which PTBP relies on
    through ``ase.calculators.dftb``): occupations fd, fp, fs are fields 7-9
    of line 2 (simple format).  Returns None if all occupations are zero.
    """
    with _open_text(path) as fh:
        first = fh.readline()
        if first.lstrip().startswith("@"):
            fh.readline()
            lmax, pos = 3, 9
        else:
            lmax, pos = 2, 7
        occ = _expand_fortran(fh.readline())[pos:pos + lmax + 1]
    for f in occ:
        if f > 0.0:
            return lmax
        lmax -= 1
    return None


def write_typed_copies(src_dir: str, dst_dir: str, base_of: dict[str, str],
                       template: str = "{a}-{b}.skf") -> list[str]:
    """Create SKF files for type labels that reuse existing parameters.

    ``base_of`` maps every label to the label whose files it copies, e.g.
    ``{"O": "O", "O2": "O", "H": "H"}``.  Files for all label pairs are
    written to ``dst_dir``.  A pair of *different* labels built from the same
    homonuclear file (``O-O2``) must be in heteronuclear format, i.e. without
    the on-site line 2 (DFTB+ decides the format from the species, not the
    element).  Returns the written paths.
    """
    os.makedirs(dst_dir, exist_ok=True)
    written = []
    for a, base_a in base_of.items():
        for b, base_b in base_of.items():
            src = os.path.join(src_dir, template.format(a=base_a, b=base_b))
            with _open_text(src) as fh:
                lines = fh.read().splitlines(keepends=True)
            if base_a == base_b and a != b:
                # homonuclear -> heteronuclear: drop the on-site line (after "@" if extended)
                del lines[2 if lines[0].lstrip().startswith("@") else 1]
            dst = os.path.join(dst_dir, template.format(a=a, b=b).removesuffix(".gz"))
            with open(dst, "w") as fh:
                fh.writelines(lines)
            written.append(dst)
    return written


def _expand_fortran(line: str) -> list[float]:
    """Parse a line that may contain Fortran repeat syntax (``20*0.0``)."""
    out: list[float] = []
    for tok in re.split(r"[,\s]+", line.strip()):
        if not tok:
            continue
        if "*" in tok:
            n, val = tok.split("*")
            out.extend([float(val)] * int(n))
        else:
            out.append(float(tok))
    return out


@dataclass
class Repulsive:
    """Pair repulsive energy (spline or polynomial), in Hartree / Bohr."""
    cutoff: float
    # spline part
    exp_coeffs: np.ndarray | None = None      # a1, a2, a3
    knots: np.ndarray | None = None           # (n+1,) start of each interval
    coeffs: np.ndarray | None = None          # (n, 6) c0..c5 (last row may use c4,c5)
    # polynomial part
    poly: np.ndarray | None = None            # c2..c9

    def __call__(self, r: np.ndarray):
        """Return (E(r), dE/dr) for an array of distances."""
        r = np.asarray(r, dtype=float)
        e = np.zeros_like(r)
        de = np.zeros_like(r)
        if self.poly is not None:
            mask = r < self.cutoff
            x = self.cutoff - r[mask]
            powers = np.arange(2, 10)
            e[mask] = (self.poly[None, :] * x[:, None] ** powers).sum(1)
            de[mask] = -(self.poly[None, :] * powers * x[:, None] ** (powers - 1)).sum(1)
            return e, de
        if self.knots is None:
            return e, de
        a1, a2, a3 = self.exp_coeffs
        first = self.knots[0]
        m_exp = r < first
        ex = np.exp(-a1 * r[m_exp] + a2)
        e[m_exp] = ex + a3
        de[m_exp] = -a1 * ex
        m_spl = (r >= first) & (r < self.cutoff)
        rs = r[m_spl]
        idx = np.clip(np.searchsorted(self.knots, rs, side="right") - 1,
                      0, len(self.coeffs) - 1)
        c = self.coeffs[idx]
        x = rs - self.knots[idx]
        e[m_spl] = c[:, 0] + x * (c[:, 1] + x * (c[:, 2] + x * (c[:, 3] + x * (c[:, 4] + x * c[:, 5]))))
        de[m_spl] = c[:, 1] + x * (2 * c[:, 2] + x * (3 * c[:, 3] + x * (4 * c[:, 4] + x * 5 * c[:, 5])))
        return e, de

    def energy(self, r, poly=None, exp_coeffs=None, coeffs=None):
        """E(r) for any array type (milestone M4b): ``r`` may be a torch tensor, and the
        coefficients may be given as (differentiable) arrays in place of the stored ones;
        the interval of every distance is chosen from its value.  Derivatives come from
        autograd; :meth:`__call__` is the NumPy evaluation with dE/dr."""
        poly = ops.asarray(self.poly if poly is None else poly, like=r)
        exp_coeffs = ops.asarray(self.exp_coeffs if exp_coeffs is None else exp_coeffs, like=r)
        coeffs = ops.asarray(self.coeffs if coeffs is None else coeffs, like=r)
        rh = np.asarray(ops.to_host(r), dtype=float)
        e = ops.zeros(rh.shape, like=r)
        if self.poly is not None:
            mask = rh < self.cutoff
            x = (self.cutoff - r[mask])[:, None]
            powers = ops.asarray(np.arange(2, 10), like=r)
            e[mask] = (poly[None, :] * x ** powers).sum(1)
            return e
        if self.knots is None:
            return e
        xp = ops.namespace(r)
        first = self.knots[0]
        m_exp = rh < first
        e[m_exp] = xp.exp(-exp_coeffs[0] * r[m_exp] + exp_coeffs[1]) + exp_coeffs[2]
        m_spl = (rh >= first) & (rh < self.cutoff)
        idx = np.clip(np.searchsorted(self.knots, rh[m_spl], side="right") - 1,
                      0, len(self.coeffs) - 1)
        c = coeffs[idx]
        x = r[m_spl] - ops.asarray(self.knots[idx], like=r)
        e[m_spl] = c[:, 0] + x * (c[:, 1] + x * (c[:, 2] + x * (c[:, 3] + x * (c[:, 4] + x * c[:, 5]))))
        return e


@dataclass
class SKFile:
    """Content of one ``A-B.skf`` file (atomic units)."""
    dr: float
    table: np.ndarray                  # (ngrid, ncol): the H and S integrals the file stores
    homo: bool
    columns: np.ndarray | None = None  # (ncol,) column of each stored integral in evaluate()
    onsite: np.ndarray | None = None   # Es, Ep, Ed, Ef (Ef = 0 in the simple format)
    hubbard: np.ndarray | None = None  # Us, Up, Ud, Uf
    occupation: np.ndarray | None = None  # fs, fp, fd, ff
    mass: float = 0.0
    repulsive: Repulsive | None = None       # spline if present, else polynomial
    poly_repulsive: Repulsive | None = None  # polynomial part only (DFTB+ PolynomialRepulsive)
    _coef: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def read_cached(cls, path: str, homo: bool | None = None) -> "SKFile":
        """Like :meth:`read`, but re-uses a previous parse of an unchanged file.

        The key includes modification time and size, so files that are
        rewritten (e.g. by PTBP on every optimisation step) are re-read.
        """
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_mtime_ns, st.st_size, homo)
        hit = _READ_CACHE.get(key)
        if hit is None:
            hit = cls.read(path, homo)
            if len(_READ_CACHE) > 512:
                _READ_CACHE.clear()
            _READ_CACHE[key] = hit
        return hit

    @classmethod
    def read(cls, path: str, homo: bool | None = None) -> "SKFile":
        with _open_text(path) as fh:
            lines = fh.read().splitlines()
        extended = lines[0].lstrip().startswith("@")
        if extended:
            lines = lines[1:]
        if homo is None:
            a, b = _pair_from_filename(path)
            homo = a == b
        head = _expand_fortran(lines[0])
        dr, ngrid = head[0], int(head[1])
        pos = 1
        onsite = hubbard = occ = None
        if homo:
            # E (f) d p s, SPE, U (f) d p s, occupations (f) d p s
            vals = _expand_fortran(lines[1])
            ns = 4 if extended else 3
            onsite, hubbard, occ = np.zeros(4), np.zeros(4), np.zeros(4)
            onsite[:ns] = vals[0:ns][::-1]
            hubbard[:ns] = vals[ns + 1:2 * ns + 1][::-1]
            occ[:ns] = vals[2 * ns + 1:3 * ns + 1][::-1]
            pos = 2
        # mass, c2..c9, rcut, d1..d10 (d's unused)
        mass_line = _expand_fortran(lines[pos])
        mass = mass_line[0]
        poly = np.array(mass_line[1:9])
        rcut_poly = mass_line[9] if len(mass_line) > 9 else 0.0
        pos += 1
        # DFTB+ (type/oldskdata.F90) reads only the first ngrid - 1 rows; using the
        # last row too would move the end of the table and its tail by one step.
        rows = [_expand_fortran(lines[pos + k]) for k in range(ngrid - 1)]
        pos += ngrid - 1
        if extended:
            columns = np.arange(2 * N_INTEGRALS)
        else:
            columns = np.concatenate([SIMPLE_COLUMNS, SIMPLE_COLUMNS + N_INTEGRALS])
        table = np.array(rows, dtype=float)[:, :len(columns)]
        rep = None
        rest = lines[pos:]
        for i, ln in enumerate(rest):
            if ln.strip().startswith("Spline"):
                rep = _parse_spline(rest[i + 1:])
                break
        poly_rep = Repulsive(cutoff=rcut_poly, poly=poly) if rcut_poly > 0.0 else None
        if rep is None:
            rep = poly_rep
        return cls(dr=dr, table=table, homo=homo, columns=columns, onsite=onsite,
                   hubbard=hubbard, occupation=occ, mass=mass, repulsive=rep,
                   poly_repulsive=poly_rep)

    # ------------------------------------------------------------------
    @property
    def ngrid(self) -> int:
        return self.table.shape[0]

    @property
    def cutoff(self) -> float:
        return self.ngrid * self.dr + DIST_FUDGE

    def _coefficients(self) -> np.ndarray:
        """Monomial coefficients (in local variable u) for every stencil.

        For a point with ``ind = floor(r/dr)``, DFTB+ interpolates through
        grid rows ``iLast-7 .. iLast`` (1-based) with
        ``iLast = max(min(ngrid, ind + 4), 8)``.  We store for each ``ind``
        the 8 coefficients of p(u) = sum_k c_k u^k, u = r/dr - (iLast - 3.5).
        """
        if self._coef is not None:
            return self._coef
        n = self.ngrid
        u_nodes = np.arange(N_INTER) - 3.5
        vinv = np.linalg.inv(np.vander(u_nodes, N_INTER, increasing=True))
        coef = np.empty((n, N_INTER, self.table.shape[1]))
        for ind in range(n):
            ilast = max(min(n, ind + N_RIGHT), N_INTER)
            y = self.table[ilast - N_INTER: ilast]          # rows (0-based)
            coef[ind] = vinv @ y
        self._coef = coef
        return coef

    @property
    def colmap(self) -> np.ndarray:
        """For each of the 40 integrals (H then S, DFTB+'s order) its stored column, or -1."""
        cols = np.arange(self.table.shape[1]) if self.columns is None else self.columns
        cmap = np.full(2 * N_INTEGRALS, -1)
        cmap[cols] = np.arange(len(cols))
        return cmap

    def evaluate(self, r: np.ndarray, deriv: bool = False, full: bool = True):
        """Interpolated integrals (and d/dr).

        ``full``: shape (len(r), 40), H then S in DFTB+'s order, zero for the integrals the
        file does not store (f integrals of the simple format).  Otherwise only the stored
        columns, located with :attr:`colmap` (faster: nothing is copied).
        """
        r = np.atleast_1d(np.asarray(r))
        coef = self._coefficients()
        n = self.ngrid
        out = np.zeros(r.shape + (self.table.shape[1],), dtype=r.dtype)
        dout = np.zeros_like(out) if deriv else None
        ind = np.floor(r.real / self.dr).astype(int)
        inside = ind < n
        if inside.any():
            ii = np.maximum(ind[inside], 0)
            ilast = np.maximum(np.minimum(n, ii + N_RIGHT), N_INTER)
            u = r[inside] / self.dr - (ilast - 3.5)
            c = coef[ii]                                    # (m, 8, 20)
            val = c[:, -1]
            dval = np.zeros_like(val)
            for k in range(N_INTER - 2, -1, -1):
                if deriv:
                    dval = dval * u[:, None] + val
                val = val * u[:, None] + c[:, k]
            out[inside] = val
            if deriv:
                dout[inside] = dval / self.dr
        rmax = n * self.dr + DIST_FUDGE
        tail = (~inside) & (r.real < rmax)
        if tail.any():
            y1, y1p, y1pp = self._tail_data()
            dr_ = r[tail] - rmax
            dx = -DIST_FUDGE
            dx1 = y1p * dx
            dx2 = y1pp * dx * dx
            dd = 10.0 * y1 - 4.0 * dx1 + 0.5 * dx2
            ee = -15.0 * y1 + 7.0 * dx1 - 1.0 * dx2
            ff = 6.0 * y1 - 3.0 * dx1 + 0.5 * dx2
            xr = (dr_ / dx)[:, None]
            out[tail] = ((ff * xr + ee) * xr + dd) * xr ** 3
            if deriv:
                dout[tail] = ((5 * ff * xr + 4 * ee) * xr + 3 * dd) * xr ** 2 / dx
        if full and self.columns is not None:
            out = self._full_layout(out)
            dout = self._full_layout(dout) if deriv else None
        return (out, dout) if deriv else out

    def evaluate_table(self, r, table):
        """Interpolated integrals at ``r`` from ``table`` (same shape as :attr:`table`, the
        stored columns) -- for torch tensors, where the table values are differentiable
        parameters (milestone M4b).  Same scheme as :meth:`evaluate` (DFTB+'s 8-point
        stencils and 5th-order tail), written as weights times table rows, both linear in
        the table: the Lagrange weights of the stencil at u, and fixed weights for the tail.
        Returns the stored-column layout (:meth:`evaluate` with ``full=False``)."""
        n = self.ngrid
        rh = np.atleast_1d(np.asarray(ops.to_host(r))).real
        out = ops.zeros(rh.shape + (table.shape[1],), like=table)
        ind = np.floor(rh / self.dr).astype(int)
        inside = ind < n
        if inside.any():
            ii = np.maximum(ind[inside], 0)
            ilast = np.maximum(np.minimum(n, ii + N_RIGHT), N_INTER)
            u = r[inside] / self.dr - ops.asarray(ilast - 3.5, like=r)
            w = (u[:, None] ** ops.asarray(np.arange(N_INTER), like=r)) @ ops.asarray(_VINV, like=r)
            rows = table[(ilast - N_INTER)[:, None] + np.arange(N_INTER)[None, :]]   # (m, 8, nc)
            out[inside] = (w[:, :, None] * rows).sum(1)
        rmax = n * self.dr + DIST_FUDGE
        tail = (~inside) & (rh < rmax)
        if tail.any():
            w0, w2 = self._tail_weights()
            y = table[n - N_INTER: n]
            y0, y2 = ops.asarray(w0, like=r) @ y, ops.asarray(w2, like=r) @ y
            y1 = table[n - 1]
            y1p = (y2 - y0) / (2.0 * DELTA_R)
            y1pp = (y2 + y0 - 2.0 * y1) / (DELTA_R * DELTA_R)
            dx = -DIST_FUDGE
            dx1 = y1p * dx
            dx2 = y1pp * dx * dx
            dd = 10.0 * y1 - 4.0 * dx1 + 0.5 * dx2
            ee = -15.0 * y1 + 7.0 * dx1 - 1.0 * dx2
            ff = 6.0 * y1 - 3.0 * dx1 + 0.5 * dx2
            xr = ((r[tail] - rmax) / dx)[:, None]
            out[tail] = ((ff * xr + ee) * xr + dd) * xr ** 3
        return out

    def _tail_weights(self):
        """Weights of the table's last 8 rows that give, like :meth:`_tail_data`, the
        Lagrange polynomial at the last grid point -/+ DELTA_R."""
        n = self.ngrid
        x = np.arange(n - N_INTER + 1, n + 1) * self.dr

        def weights(xx):
            w = np.ones(N_INTER)
            for k in range(N_INTER):
                for j in range(N_INTER):
                    if j != k:
                        w[k] *= (xx - x[j]) / (x[k] - x[j])
            return w
        return weights(x[-1] - DELTA_R), weights(x[-1] + DELTA_R)

    def _full_layout(self, a: np.ndarray) -> np.ndarray:
        full = np.zeros(a.shape[:-1] + (2 * N_INTEGRALS,), dtype=a.dtype)
        full[..., self.columns] = a
        return full

    def _tail_data(self):
        n = self.ngrid
        x = np.arange(n - N_INTER + 1, n + 1) * self.dr
        y = self.table[n - N_INTER: n]
        def poly(xx):
            # Lagrange interpolation through (x, y)
            w = np.ones(N_INTER)
            for k in range(N_INTER):
                for j in range(N_INTER):
                    if j != k:
                        w[k] *= (xx - x[j]) / (x[k] - x[j])
            return w @ y
        y0 = poly(x[-1] - DELTA_R)
        y2 = poly(x[-1] + DELTA_R)
        y1 = self.table[n - 1]
        y1p = (y2 - y0) / (2.0 * DELTA_R)
        y1pp = (y2 + y0 - 2.0 * y1) / (DELTA_R * DELTA_R)
        return y1, y1p, y1pp


def _parse_spline(lines: list[str]) -> Repulsive:
    n_int, cutoff = lines[0].split()[:2]
    n_int, cutoff = int(n_int), float(cutoff)
    a = np.array(_expand_fortran(lines[1])[:3])
    knots, coeffs = [], []
    for i in range(n_int):
        v = _expand_fortran(lines[2 + i])
        r0, r1 = v[0], v[1]
        c = np.zeros(6)
        c[: len(v) - 2] = v[2:8]
        knots.append(r0)
        coeffs.append(c)
    knots.append(r1)
    return Repulsive(cutoff=cutoff, exp_coeffs=a, knots=np.array(knots),
                     coeffs=np.array(coeffs))


@dataclass
class SKPair:
    """Ordered pair (A, B) with integrals from both A-B and B-A files."""
    ab: SKFile
    ba: SKFile


class SKSet:
    """All Slater-Koster data needed for a set of type labels.

    ``elements`` are the type labels used in file names (normally element
    symbols, but e.g. ``Li@el`` is allowed: then ``Li@el-O.skf`` is read).
    ``max_l`` maps label -> highest angular momentum; labels missing from it
    are filled with the ASE/PTBP rule (:func:`max_l_from_skf`).
    ``template`` may end in ``.gz`` for compressed files.
    """

    def __init__(self, path: str, elements: list[str], max_l: dict[str, int] | None = None,
                 template: str = "{a}-{b}.skf", shell_resolved: bool = False,
                 hubbard_from: str = "s"):
        self.path = path
        self.template = template
        self.elements = list(dict.fromkeys(elements))
        self.max_l = dict(max_l or {})
        for a in self.elements:
            if a not in self.max_l:
                lmax = max_l_from_skf(self.file_name(a, a))
                if lmax is None:
                    raise ValueError(f"no occupied shell in {self.file_name(a, a)}; pass max_l")
                self.max_l[a] = lmax
        self.files: dict[tuple[str, str], SKFile] = {}
        for a in self.elements:
            for b in self.elements:
                self.files[(a, b)] = SKFile.read_cached(self.file_name(a, b), homo=(a == b))
        self.shell_resolved = shell_resolved
        self.onsite: dict[str, np.ndarray] = {}
        self.hubbard: dict[str, np.ndarray] = {}
        self.occupation: dict[str, np.ndarray] = {}
        self.mass: dict[str, float] = {}
        for a in self.elements:
            f = self.files[(a, a)]
            lmax = self.max_l[a]
            # shells ordered s, p, d, f
            self.onsite[a] = f.onsite[: lmax + 1].copy()
            u = f.hubbard[: lmax + 1].copy()
            if not shell_resolved:
                # DFTB+ (non orbital-resolved SCC) uses one U per atom
                u[:] = u[0] if hubbard_from == "s" else u[lmax]
            self.hubbard[a] = u
            self.occupation[a] = f.occupation[: lmax + 1].copy()
            self.mass[a] = f.mass

    def file_name(self, a: str, b: str) -> str:
        return os.path.join(self.path, self.template.format(a=a, b=b))

    def shells(self, el: str) -> list[int]:
        return list(range(self.max_l[el] + 1))

    def norb(self, el: str) -> int:
        return (self.max_l[el] + 1) ** 2

    @property
    def cutoff(self) -> float:
        return max(f.cutoff for f in self.files.values())

    def repulsive(self, a: str, b: str, mode: str = "skf") -> Repulsive | None:
        """Pair repulsive for labels (a, b).

        mode ``"skf"``: spline if the file has one, else the polynomial (default);
        ``"polynomial"``: the polynomial only, like DFTB+ ``PolynomialRepulsive = Yes``;
        ``"none"``: no repulsive energy.
        """
        if mode == "skf":
            return self.files[(a, b)].repulsive
        if mode == "polynomial":
            return self.files[(a, b)].poly_repulsive
        if mode == "none":
            return None
        raise ValueError(f"unknown repulsive mode {mode!r}")
