"""Cross-geometry attention — vocabulary learning module (Eq 5-6, Algo 2).

For each substructure (a tree in H or a cycle in S) we update node coordinates
via attentional aggregation. The distinctive feature of RiemannGFM is that the
attention *query* on geometry A is drawn from the counterpart node's
coordinate on geometry B — hence "cross-geometry".

Direction handling:
    - tree substructures: bottom-up (children -> parent).
    - cycle substructures: symmetric.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from ..manifolds.ccs import ConstantCurvatureSpace
from .midpoint import geometric_midpoint
from .riemann_linear import RiemannLinear


class ScalarMap(nn.Module):
    """phi: L x L -> R.

    Two-layer MLP with hidden dim 256 (Sec 4.1.3). Input is the concatenation
    of two curvature-aware inner products, following the paper's phi([q_i | k_j])
    interpretation as a scalar map.
    """

    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * (dim + 1), hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, q: Tensor, k: Tensor) -> Tensor:
        return self.net(torch.cat([q, k], dim=-1)).squeeze(-1)


class CrossGeometryAttention(nn.Module):
    """Cross-geometry attention on one factor of the product bundle.

    Parameters
    ----------
    manifold_self:
        The factor we are updating (H or S).
    manifold_other:
        The counterpart factor from which the query is sourced.
    dim_self, dim_other:
        Space-like dims of the two factors.
    direction:
        "tree" for bottom-up on trees, "cycle" for symmetric on cycles.
    """

    def __init__(
        self,
        manifold_self: ConstantCurvatureSpace,
        manifold_other: ConstantCurvatureSpace,
        dim_self: int,
        dim_other: int,
        direction: Literal["tree", "cycle"] = "tree",
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.manifold_self = manifold_self
        self.manifold_other = manifold_other
        self.direction = direction

        # Key / value derived from p_self, query from p_other (cross-geometry).
        self.f_K = RiemannLinear(manifold_self, dim_self, dim_self)
        self.f_V = RiemannLinear(manifold_self, dim_self, dim_self)
        self.f_Q = RiemannLinear(manifold_other, dim_other, dim_other)

        # The scalar map operates on the *self* geometry's embedding, so its
        # dim is the self dim. We first bring the query into the self manifold
        # by a light Riemannian projection (dim_other -> dim_self) via a
        # dedicated linear head. If dims differ we need a lift; when they
        # match we can share weights with f_K.
        if dim_other != dim_self:
            self.q_lift = RiemannLinear(manifold_other, dim_other, dim_self)
        else:
            self.q_lift = None

        self.phi = ScalarMap(dim_self, hidden=hidden)

    def forward(
        self,
        p_self: Tensor,          # [B, N, d_self + 1]
        p_other: Tensor,         # [B, N, d_other + 1]
        adjacency: Tensor,       # [B, N, N] bool mask of substructure edges
    ) -> Tensor:
        """Update p_self by aggregating over adjacent nodes in the substructure.

        adjacency[b, i, j] is True iff node j should send a message to node i.
        For tree (bottom-up): j is a descendant of i.
        For cycle (symmetric): j is a neighbour of i in the cycle.
        """
        # Compute keys, values, queries — all still on their respective manifolds.
        k = self.f_K(p_self)          # [B, N, d_self + 1]
        v = self.f_V(p_self)          # [B, N, d_self + 1]
        q_other = self.f_Q(p_other)   # [B, N, d_other + 1]

        if self.q_lift is not None:
            # Cross-manifold lift is somewhat ill-defined (different signatures).
            # We approximate by mapping q_other to the self manifold via q_lift,
            # which is a Riemannian linear op on the other manifold followed by
            # a fresh time-coord to move to the self manifold. See notes: this
            # matches the practical treatment in the official code.
            q = self.q_lift(q_other)
        else:
            # Same dim: reinterpret the coordinate on the self manifold by
            # rescaling the time component. This is what the paper calls
            # "leveraging compensatory information".
            q = self.manifold_self.project(q_other)

        # Score s[b, i, j] = phi(q_i, k_j)
        B, N, _ = q.shape
        q_expand = q.unsqueeze(2).expand(B, N, N, q.size(-1))
        k_expand = k.unsqueeze(1).expand(B, N, N, k.size(-1))
        scores = self.phi(q_expand, k_expand)         # [B, N, N]

        # Mask by adjacency and softmax over j.
        very_negative = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~adjacency, very_negative)
        alpha = torch.softmax(scores, dim=-1)         # [B, N, N]

        # Weighted midpoint over j for each i.
        # v has shape [B, N, d+1]; broadcast to [B, N (i), N (j), d+1].
        v_expand = v.unsqueeze(1).expand(B, N, N, v.size(-1))
        # geometric_midpoint over the j-axis (axis = -2 after flattening).
        p_new = geometric_midpoint(self.manifold_self, v_expand, alpha)

        return p_new
