"""Geometric midpoint (Theorem 2, Eq 4).

    mid_kappa({x_i, v_i}) = (1/sqrt(|kappa|)) * sum_i v_i x_i / || sum_j v_j x_j ||_kappa

is the closed-form Frechet mean under the curvature-aware distance. It's the
aggregation primitive used across cross-geometry attention and bundle
convolution.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..manifolds.ccs import ConstantCurvatureSpace

_EPS = 1e-15


def geometric_midpoint(
    manifold: ConstantCurvatureSpace,
    x: Tensor,          # [..., N, d + 1]
    weights: Tensor,    # [..., N]
) -> Tensor:
    """Weighted geometric midpoint on the manifold.

    Broadcasts over any leading batch dimensions. N is the aggregation axis.
    """
    # Weighted sum in ambient space: [..., d + 1]
    weighted = (weights.unsqueeze(-1) * x).sum(dim=-2)

    # Curvature-aware norm of the weighted sum.
    inner = manifold.inner(weighted, weighted, keepdim=True)  # [..., 1]
    # For Lorentz (kappa<0) sum weighted*x may lie inside the light cone; the
    # inner is positive (time-like) when weights are compatible. Clamp for
    # safety.
    norm_k = inner.abs().clamp_min(_EPS).sqrt()

    result = weighted / (math.sqrt(manifold.abs_kappa) * norm_k)
    return manifold.project(result)
