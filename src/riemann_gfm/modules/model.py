"""RiemannGFM — stack of universal Riemannian layers (Sec 3.1, Algo 1).

One layer = CrossGeometryAttention (updates p) + BundleConvolution (updates z),
applied independently on the H factor and the S factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..manifolds.product_bundle import BundlePoint, ProductBundle
from .bundle_conv import BundleConvolution
from .cross_geom_attn import CrossGeometryAttention
from .midpoint import geometric_midpoint


class UniversalRiemannianLayer(nn.Module):
    """One layer applied to both product-bundle factors."""

    def __init__(self, bundle: ProductBundle, hidden: int = 256) -> None:
        super().__init__()
        self.bundle = bundle
        # Vocabulary learning.
        self.attn_H = CrossGeometryAttention(
            manifold_self=bundle.H,
            manifold_other=bundle.S,
            dim_self=bundle.d_H,
            dim_other=bundle.d_S,
            direction="tree",
            hidden=hidden,
        )
        self.attn_S = CrossGeometryAttention(
            manifold_self=bundle.S,
            manifold_other=bundle.H,
            dim_self=bundle.d_S,
            dim_other=bundle.d_H,
            direction="cycle",
            hidden=hidden,
        )
        # Global learning.
        self.conv_H = BundleConvolution(bundle.H)
        self.conv_S = BundleConvolution(bundle.S)

    def forward(
        self,
        point: BundlePoint,
        tree_adj: Tensor,       # [B, N, N] tree adjacency (bottom-up)
        cycle_adj: Tensor,      # [B, N, N] cycle adjacency (symmetric)
    ) -> BundlePoint:
        # Cross-geometry attention updates coordinates.
        p_H_new = self.attn_H(point.p_H, point.p_S, tree_adj)
        p_S_new = self.attn_S(point.p_S, point.p_H, cycle_adj)

        # For bundle convolution we need alpha weights. We reuse the softmax
        # weights implicit in the attention update by re-computing them here
        # via a lightweight route: uniform over adjacency for now, since the
        # scalar phi module lives inside CrossGeometryAttention. This keeps
        # module boundaries clean; we can share alpha in a later refactor.
        # (See implementation_notes.md — this is a known simplification.)
        alpha_tree = _uniform_over_adj(tree_adj)
        alpha_cycle = _uniform_over_adj(cycle_adj)

        z_H_new = self.conv_H(p_H_new, point.z_H, alpha_tree)
        z_S_new = self.conv_S(p_S_new, point.z_S, alpha_cycle)

        return BundlePoint(p_H_new, z_H_new, p_S_new, z_S_new)


def _uniform_over_adj(adj: Tensor) -> Tensor:
    """Turn a boolean adjacency into row-stochastic weights."""
    a = adj.float()
    a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return a


@dataclass
class RiemannGFMConfig:
    d_H: int = 32
    d_S: int = 32
    kappa_H: float = -1.0
    kappa_S: float = 1.0
    n_layers: int = 2
    hidden: int = 256


class RiemannGFM(nn.Module):
    """Full RiemannGFM encoder.

    Forward returns updated bundle point, plus a graph-level node encoding
    obtained via the geometric-midpoint aggregation over target nodes
    (Sec 3.1.4).
    """

    def __init__(self, config: RiemannGFMConfig | None = None) -> None:
        super().__init__()
        self.config = config or RiemannGFMConfig()
        self.bundle = ProductBundle(
            d_H=self.config.d_H,
            d_S=self.config.d_S,
            kappa_H=self.config.kappa_H,
            kappa_S=self.config.kappa_S,
        )
        self.layers = nn.ModuleList(
            [UniversalRiemannianLayer(self.bundle, hidden=self.config.hidden) for _ in range(self.config.n_layers)]
        )

    @property
    def manifold_H(self):
        return self.bundle.H

    @property
    def manifold_S(self):
        return self.bundle.S

    def forward(
        self,
        point: BundlePoint,
        tree_adj: Tensor,
        cycle_adj: Tensor,
    ) -> BundlePoint:
        for layer in self.layers:
            point = layer(point, tree_adj, cycle_adj)
        return point

    def graph_encoding(self, point: BundlePoint, target_mask: Tensor) -> BundlePoint:
        """Aggregate to graph-level via geometric midpoint over target nodes.

        target_mask: [B, N] bool — nodes to average over. Weights are uniform.
        """
        weights = target_mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        p_H = geometric_midpoint(self.manifold_H, point.p_H, weights)
        p_S = geometric_midpoint(self.manifold_S, point.p_S, weights)

        # Encodings are averaged in tangent space (linear over samples).
        z_H = (weights.unsqueeze(-1) * point.z_H).sum(dim=-2)
        z_S = (weights.unsqueeze(-1) * point.z_S).sum(dim=-2)
        # Project onto tangent at the new midpoint.
        z_H = self.manifold_H.project_tangent(p_H, z_H)
        z_S = self.manifold_S.project_tangent(p_S, z_S)
        return BundlePoint(p_H, z_H, p_S, z_S)
