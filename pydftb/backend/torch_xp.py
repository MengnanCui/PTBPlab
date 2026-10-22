"""NumPy names over torch, for the code that runs on both backends (milestone M4).

Only what that code uses; each function takes the same arguments as its NumPy namesake.
Imported only for torch inputs (``Backend.xp``, ``ops.namespace``), never by
``import pydftb``.
"""
from __future__ import annotations

import torch

conj = torch.conj
swapaxes = torch.swapaxes


def concatenate(arrays, axis: int = 0):
    return torch.cat(list(arrays), dim=axis)
