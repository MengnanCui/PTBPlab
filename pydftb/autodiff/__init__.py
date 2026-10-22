"""Derivatives of pydftb results with PyTorch (milestone M4b; needs ``torch``).

* :class:`Params` -- SK tables, on-site energies, Hubbard U and derivatives, spin
  constants and repulsive coefficients as torch tensors;
* :func:`evaluate` -- energy, free energy, forces and stress with gradients with respect to
  the parameters and the geometry.

See ``pydftb/docs/design/M4b-autodiff.html``.
"""
from .api import GradientUnavailable, Result, evaluate
from .params import KINDS, Params

__all__ = ["GradientUnavailable", "KINDS", "Params", "Result", "evaluate"]
