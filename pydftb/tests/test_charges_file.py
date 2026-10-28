"""DFTB+'s charge restart files charges.bin / charges.dat (milestone M6f-1).

Byte-for-byte round trips of the files DFTB+ 25.1 wrote for the frozen references
*_read_charges* (binary and text, every layout; with a hybrid functional also the density
matrix, M6h-1, and for the Gamma point the k-point folding before it, M6h-3a), pydftb's own files
read back, DFTB+'s checks on reading, and the older formats
4-7.  Restarting from DFTB+'s files against DFTB+ itself is
in test_reference.py.
"""
import base64
import struct

import numpy as np
import pytest

from pydftb import dftbplus_io as io
from pydftb.calculator import DFTB
from pydftb.validation import refgen

from .test_spin import _system

READ_CASES = sorted(c.name for c in refgen.CASES if c.read_charges is not None)


def _engine(name):
    """The case's DFTB engine, labels and positions (Bohr)."""
    case, sk, labels, pos, cell, kpts, opts = _system(name)
    opts = {**opts, **refgen.pauli_options(case, labels), **refgen.block_options(case, labels),
            **({"hybrid": case.hybrid} if case.hybrid is not None else {})}       # M6h-1
    calc = DFTB(sk, smearing=tuple(case.filling), kpts=kpts, scc_tol=1e-10, max_scc=500, **opts)
    return case, calc, labels, pos, cell


@pytest.mark.parametrize("name", READ_CASES)
def test_dftbplus_files_round_trip_byte_for_byte(name, tmp_path):
    doc = refgen.load(name)
    inp = doc["input"]
    blob = base64.b64decode(inp["charges_file"])
    text = inp["charges_text"]
    _, calc, labels, _, _ = _engine(doc["case"])
    norb = calc._charge_layout(labels)[0].norb_atom
    src = tmp_path / ("in.dat" if text else "in.bin")
    src.write_bytes(blob)
    data = io.read_charges(src, norb)
    gamma = inp["cell_angstrom"] is not None and data.rho is not None    # M6h-3a: the folding
    assert data.version == 8 and data.multipolar and data.kpoint_info == gamma
    assert (data.folding is not None) == gamma
    assert np.array_equal(data.checksum, data.sums())          # DFTB+'s own summation order
    out = tmp_path / ("out.dat" if text else "out.bin")
    io.write_charges(out, data)
    assert out.read_bytes() == blob


@pytest.mark.parametrize("name", ["h2o", "cu4co_shell_resolved", "o2_triplet",
                                  "oh_noncolinear_tilted", "cu4co_plusu",
                                  "oh_noncolinear_soc_plusu", "cuo_chain_spin_plusu_k422"])
@pytest.mark.parametrize("text", [False, True])
def test_pydftb_reads_what_it_writes(name, text, tmp_path):
    """The input populations of the last iteration survive the file (17 digits in text) and
    restart the SCC where it ended."""
    case, calc, labels, pos, cell = _engine(name)
    pbc = None if cell is None else np.ones(3, bool)
    r = calc.calculate(labels, pos, cell, pbc, forces=False)
    path = tmp_path / ("charges.dat" if text else "charges.bin")
    io.write_charges(path, calc.charge_data(labels, r))
    q = calc.populations_from_file(labels, path)
    assert np.allclose(q, r.q_in, rtol=0, atol=1e-14)
    again = calc.calculate(labels, pos, cell, pbc, forces=False,
                           q_init=io.read_charges(path, calc._charge_layout(labels)[0].norb_atom))
    assert again.n_scc <= 2
    assert again.energy == pytest.approx(r.energy, abs=1e-9)


def test_dftbplus_checks_on_reading(tmp_path):
    _, calc, labels, pos, _ = _engine("o2_triplet")
    r = calc.calculate(labels, pos, forces=False)
    path = tmp_path / "charges.bin"
    io.write_charges(path, calc.charge_data(labels, r))
    norb = calc._charge_layout(labels)[0].norb_atom
    _, sk, _, _, _, _, opts = _system("o2_triplet")
    with pytest.raises(ValueError, match="magnetisation of 2.000000 instead of 0.000000"):
        DFTB(sk, **{**opts, "unpaired_electrons": 0.0}).populations_from_file(labels, path)
    with pytest.raises(ValueError, match="12.000000 electrons instead of 11.000000"):
        DFTB(sk, charge=1.0, **opts).populations_from_file(labels, path)
    DFTB(sk, charge=1.0, **opts).populations_from_file(labels, path, skip_charge_test=True)
    with pytest.raises(ValueError, match="2 spin channels, this calculation 1"):
        DFTB(sk).populations_from_file(labels, path)
    bad = io.read_charges(path, norb)
    bad.checksum = bad.checksum + 1e-6
    with pytest.raises(ValueError, match="checksums do not match"):
        calc.populations_from_file(labels, bad)
    with pytest.raises(ValueError, match="more atoms than the system"):
        io.read_charges(path, norb[:1])
    with pytest.raises(ValueError, match="ends too early"):
        io.read_charges(path, norb + 1)
    (tmp_path / "v9.bin").write_bytes(struct.pack("<i", 9) + path.read_bytes()[4:])
    with pytest.raises(ValueError, match="unknown charge file format 9"):
        io.read_charges(tmp_path / "v9.bin", norb)


def test_torch_refuses_charge_files(tmp_path):
    pytest.importorskip("torch")
    _, calc, labels, pos, _ = _engine("h2o")
    r = calc.calculate(labels, pos, forces=False)
    _, sk, _, _, _, _, _ = _system("h2o")
    with pytest.raises(ValueError, match="charge files are read with the NumPy backend only"):
        DFTB(sk, backend="torch").calculate(labels, pos, q_init=calc.charge_data(labels, r))


@pytest.mark.parametrize("version", [4, 5, 6, 7])
def test_older_formats(version, tmp_path):
    """Formats 4-7 differ in the header only (and 4-5 have no multipole record)."""
    _, calc, labels, pos, _ = _engine("h2o")
    r = calc.calculate(labels, pos, forces=False)
    data = calc.charge_data(labels, r)
    q = data.q[0]
    flags = {4: [0, 0, 0], 5: [0, 0, 0], 6: [0, 0, 0, 1], 7: [0, 0, 0, 0, 1]}[version]
    natom = [] if version == 4 else [3]
    head = struct.pack(f"<i{len(flags)}i", version, *flags) + struct.pack(
        f"<{len(natom) + 1}i", *natom, 1) + struct.pack("<d", data.sums()[0])
    body = struct.pack(f"<{len(q)}d", *q) + (b"" if version < 6 else struct.pack("<2i", 0, 0))
    (tmp_path / "old.bin").write_bytes(head + body)
    old = io.read_charges(tmp_path / "old.bin", data.norb_atom)
    assert old.version == version and np.array_equal(old.q, data.q)
    assert np.allclose(calc.populations_from_file(labels, old), r.q_in, rtol=0, atol=1e-14)


def test_list_directed_format():
    """gfortran's list-directed items, as in DFTB+'s charges.dat files."""
    assert io.ld_integer(8) == "           8"
    assert io.ld_logical(True) + io.ld_logical(False) == " T F"
    for token in ["   8.0000000000000018     ", "  0.77308034137419235     ",
                  "   0.0000000000000000     ", "   12.000000000000000     ",
                  "   5.1994542654409948E-004", "  -3.6354425614791554E-002",
                  "  0.49734087187448095     ", "  -2.0422581609358944E-002"]:
        assert io.ld_real(float(token)) == token
    assert io.ld_real(-123.456) == "  -123.45600000000000     "
    assert io.ld_real(1e20) == "   1.0000000000000000E+020"
    # list-directed reading also takes commas and D exponents
    rd = io._Records("1.5D-3, 2.0\n 3 ,4\n")
    rd.start()
    assert np.allclose(rd.reals(4), [1.5e-3, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("fname", ["charges.bin", "charges.dat"])
def test_ase_calculator_writes_and_reads(fname, tmp_path):
    from ase.build import molecule

    from pydftb.ase_calc import PyDFTB

    from .conftest import SKF_HCOCU
    atoms = molecule("CH3OH")
    atoms.rattle(0.05, seed=2)
    atoms.calc = PyDFTB(SKF_HCOCU, scc_tol=1e-10)
    e = atoms.get_potential_energy()
    atoms.calc.write_charges(tmp_path / fname)
    restart = atoms.copy()
    restart.calc = PyDFTB(SKF_HCOCU, scc_tol=1e-10, read_charges=tmp_path / fname)
    assert restart.get_potential_energy() == pytest.approx(e, abs=1e-8)
    assert restart.calc.raw.n_scc <= 2 < atoms.calc.raw.n_scc
    with pytest.raises(RuntimeError, match="no calculation"):
        PyDFTB(SKF_HCOCU).write_charges(tmp_path / "none.bin")
