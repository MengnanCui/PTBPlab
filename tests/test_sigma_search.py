"""The sigma_rep search of a BO step (M3b): equation-of-state fits in memory with the same
files, the step's files parsed once, the repulsion evaluated without a process pool."""
import argparse
import gzip
import hashlib
import shutil
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest

import utils.data as data
import utils.engines as eng
import utils.loss as L
from utils import dftb_stage

ROOT = Path(__file__).resolve().parents[1]
CU_SKF_GZ = ROOT / "pydftb" / "tests" / "data" / "skf" / "ptbp_hcocu" / "Cu-Cu.skf.gz"


def _digests(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}


@pytest.fixture(scope="module")
def images():
    return ase.io.read(ROOT / "dft" / "dft.xyz", ":22")             # two EOS phases


def test_eos_fits_in_memory_write_the_same_files(tmp_path, monkeypatch, images):
    loss = L.Loss()
    (tmp_path / "files").mkdir()
    (tmp_path / "memory").mkdir()
    monkeypatch.chdir(tmp_path / "files")
    ref = loss.eos_atoms(images, 11, "dft")
    monkeypatch.chdir(tmp_path / "memory")
    files = {}
    out = loss.eos_atoms(images, 11, "dft", files=files)
    assert not any((tmp_path / "memory").iterdir())                 # recorded, not written
    loss.write_eos_files(files)
    assert repr(out) == repr(ref)
    assert _digests(tmp_path / "memory") == _digests(tmp_path / "files")


def test_a_failed_fit_gives_the_fallback_every_time(monkeypatch, images):
    class Broken:
        def __init__(self, *a, **k):
            pass

        def fit(self, warn=False):
            raise RuntimeError("no minimum")

    monkeypatch.setattr(L, "EOS", Broken)
    monkeypatch.setattr(L, "_EOS_FITS", type(L._EOS_FITS)())
    for _ in range(2):
        with pytest.raises(RuntimeError, match="no minimum"):
            L._eos_fit(images[:11])
        out = L.Loss().eos_atoms(images[:11], 11, "dftb", files={})
        assert out["eos_Cu_0"]["dftb_volume"] == 1000


def test_step_files_are_parsed_once_per_content(tmp_path, images):
    path = tmp_path / "step.xyz"
    ase.io.write(path, images[:3])
    first = L._read_step_file(path)
    assert L._read_step_file(path)[0] is first[0]                   # cached
    ase.io.write(path, images[3:5])
    again = L._read_step_file(path)
    assert len(again) == 2 and again[0] is not first[0]             # new content, new parse


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork backend")
def test_repulsion_without_pool_and_bitwise_like_the_reference(tmp_path, monkeypatch, images):
    """A two-group step: the loss of the vectorised evaluate_pbc equals the original loop's."""
    with gzip.open(CU_SKF_GZ, "rb") as fi, open(tmp_path / "Cu-Cu.skf", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    monkeypatch.chdir(tmp_path)
    saved = eng.settings()
    eng.configure(engine="pydftb")
    try:
        _, bolt = data.get_bolt_factor(images[5::11], "boltzmann", 3000)
        args = argparse.Namespace(eos_points=11, kpt_density=5.0, multi_element=None,
                                  optimization_option="energygeometry",
                                  log_file=str(tmp_path / "log.out"))
        loss = L.Loss(args_input=args, dft_input=images, bolt_f11=bolt, opt_elem=[])
        dftb_stage.write_outputs(*dftb_stage.run_dataset(images, bolt, 11, 5.0))
    finally:
        eng._SETTINGS.update(saved)

    def no_pool(*a, **k):
        raise AssertionError("parallel_repulsion must not start a process pool")

    monkeypatch.setattr(L.mp, "Pool", no_pool)
    mu = loss.parallel_repulsion(np.array([0.33]), 2.749, 12.036, True)
    skf = (tmp_path / "Cu-Cu.skf").read_bytes()
    monkeypatch.setenv("PTBP_PHYSREP_REFERENCE", "1")
    assert loss.parallel_repulsion(np.array([0.33]), 2.749, 12.036, True).hex() == mu.hex()
    assert (tmp_path / "Cu-Cu.skf").read_bytes() == skf
    lines = (tmp_path / "log.out").read_text().splitlines()
    assert len(lines) == 2 and lines[0] == lines[1]
