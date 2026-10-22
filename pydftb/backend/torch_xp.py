"""NumPy names over torch, for the code that runs on both backends (milestone M4).

Only what that code uses; each function takes the same arguments as its NumPy namesake.
Imported only for torch inputs (``Backend.xp``, ``ops.namespace``), never by
``import pydftb``.
"""
from __future__ import annotations

import torch

abs = torch.abs                       # noqa: A001
conj = torch.conj
cos = torch.cos
exp = torch.exp
sin = torch.sin
sqrt = torch.sqrt
swapaxes = torch.swapaxes
where = torch.where
zeros_like = torch.zeros_like
special_erfc = torch.special.erfc


def concatenate(arrays, axis: int = 0):
    return torch.cat(list(arrays), dim=axis)


def stack(arrays, axis: int = 0):
    return torch.stack(list(arrays), dim=axis)


def cross(a, b):
    return torch.linalg.cross(a, b, dim=-1)


def broadcast_arrays(*arrays):
    return torch.broadcast_tensors(*arrays)
