"""Test systems for benchmarks."""
import numpy as np
from ase import Atoms
from ase.build import molecule


def water_box(nside: int, seed: int = 0, spacing: float = 3.104) -> Atoms:
    """nside^3 water molecules on a jittered cubic grid (~1 g/cm^3)."""
    rng = np.random.default_rng(seed)
    w = molecule("H2O")
    w.translate(-w.get_center_of_mass())
    box = Atoms(cell=np.eye(3) * nside * spacing, pbc=True)
    for ijk in np.ndindex(nside, nside, nside):
        m = w.copy()
        m.rotate(rng.uniform(0, 360), rng.normal(size=3), center=(0, 0, 0))
        m.translate((np.array(ijk) + 0.5) * spacing + rng.normal(scale=0.15, size=3))
        box += m
    return box
