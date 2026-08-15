"""Bundle convolution — global learning module (Eq 7).

For each node i in a substructure Lambda, update its tangent-space encoding
z_i in T_{p_i} M by parallel-transporting neighbours' encodings z_l in T_{p_l} M
to T_{p_i} M and aggregating with attentional weights alpha_il (from Eq 6).

    BC_{p_i}({p_l, z_l}) = sum_l ( alpha_il z_l - kappa alpha_il <z_i, p_l> / (1 + kappa <p_i, p_l>) * (p_i + p_l) )

The (p_i + p_l) term is the closed-form parallel-transport correction on
CCSs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..manifolds.ccs import ConstantCurvatureSpace

_EPS = 1e-15


class BundleConvolution(nn.Module):
    """One bundle-convolution step over the whole substructure.

    Only touches z (tangent encoding); p is left untouched (updated by
    CrossGeometryAttention).
    """

    def __init__(self, manifold: ConstantCurvatureSpace) -> None:
        super().__init__()
        self.manifold = manifold

    def forward(
        self,
        p: Tensor,               # [B, N, d + 1]  node coordinates on manifold
        z: Tensor,               # [B, N, d + 1]  node encodings in tangent
        alpha: Tensor,           # [B, N, N]      attentional weights over j
    ) -> Tensor:
        """Return updated encodings z_new with same shape as z."""
        k = self.manifold.kappa
        B, N, D = p.shape

        # Curvature-aware pairwise inner products.
        sign_time = self.manifold.sign
        p_time = p[..., :1]                                        # [B, N, 1]
        p_space = p[..., 1:]                                       # [B, N, d]
        # <p_i, p_l>_kappa   -> [B, N, N]
        inner_pp = sign_time * (p_time * p_time.transpose(1, 2)) + p_space @ p_space.transpose(1, 2)

        # <z_i, p_l>_kappa   -> [B, N, N]
        z_time = z[..., :1]
        z_space = z[..., 1:]
        inner_zp = sign_time * (z_time * p_time.transpose(1, 2)) + z_space @ p_space.transpose(1, 2)

        # Denominator (1 + kappa <p_i, p_l>) with sign-safe epsilon.
        denom = 1.0 + k * inner_pp                                 # [B, N, N]
        denom = torch.where(denom.abs() < _EPS, torch.full_like(denom, _EPS), denom)

        # First term: sum_l alpha_il z_l    -> [B, N, d + 1]
        term_a = alpha @ z                                          # matmul over j-axis

        # Second term: sum_l alpha_il * kappa <z_i, p_l> / (1 + kappa <p_i, p_l>) * (p_i + p_l)
        # coefficient c[b, i, j] = alpha_il * kappa * inner_zp / denom
        coeff = alpha * (k * inner_zp / denom)                      # [B, N, N]
        # (p_i + p_l): broadcast to [B, N (i), N (j), D]
        p_i = p.unsqueeze(2).expand(B, N, N, D)
        p_l = p.unsqueeze(1).expand(B, N, N, D)
        p_sum = p_i + p_l
        term_b = (coeff.unsqueeze(-1) * p_sum).sum(dim=2)           # sum over j-axis

        z_new = term_a - term_b
        # Project back into the tangent space at p_i to counter drift.
        z_new = self.manifold.project_tangent(p, z_new)
        return z_new
