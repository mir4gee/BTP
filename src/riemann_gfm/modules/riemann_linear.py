"""Manifold-preserving Riemannian linear operation (Theorem 1, Eq 3).

For x = [x_t; x_s] on L^{d_1}_kappa and W in R^{d_1 x d_2},

    f_W(x) = [x_t; alpha * W x_s],  alpha = sqrt((1/kappa - sgn(kappa) x_t^2) / ||W x_s||^2)

is on L^{d_2}_kappa. This is the "no tangent-space detour" version of the
hyperbolic linear layer used everywhere in RiemannGFM.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..manifolds.ccs import ConstantCurvatureSpace

_EPS = 1e-15


class RiemannLinear(nn.Module):
    """Riemannian linear layer that maps L^{d_in}_kappa -> L^{d_out}_kappa.

    Only the space-like block is transformed; the time-like coord is rescaled
    afterwards to snap the output back on-manifold.
    """

    def __init__(self, manifold: ConstantCurvatureSpace, d_in: int, d_out: int, bias: bool = True) -> None:
        super().__init__()
        self.manifold = manifold
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
        # Small init keeps ||W x_s|| bounded on the first forward.
        nn.init.xavier_uniform_(self.weight, gain=1.0 / math.sqrt(2))

    def forward(self, x: Tensor) -> Tensor:
        # Split time / space coords.
        x_t = x[..., :1]
        x_s = x[..., 1:]

        # Apply linear op to the space-like part.
        Wx = torch.nn.functional.linear(x_s, self.weight, self.bias)  # [..., d_out]

        # Numerator of alpha^2 depends on curvature sign:
        #   Lorentz  (kappa < 0):  1/|kappa| + x_t^2    (since sgn(kappa) = -1)
        #   Spherical(kappa > 0):  1/kappa   - x_t^2
        k = self.manifold.kappa
        sign_k = self.manifold.sign
        numer = 1.0 / k - sign_k * (x_t * x_t)
        # If numer < 0 due to numerical drift on the sphere, clamp.
        numer = numer.clamp_min(_EPS)

        denom = (Wx * Wx).sum(dim=-1, keepdim=True).clamp_min(_EPS)
        alpha = torch.sqrt(numer / denom)

        # New time-like coord is preserved from x_t; the paper's rescaling
        # keeps alpha only on the space-like block.
        x_new = torch.cat([x_t, alpha * Wx], dim=-1)
        return self.manifold.project(x_new)
