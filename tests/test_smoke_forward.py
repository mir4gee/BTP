"""Smoke test: run a forward pass through the whole RiemannGFM stack.

Uses random adjacency tensors — verifies the module wiring, shapes, and
that gradients flow through the contrastive loss.
"""

from __future__ import annotations

import torch

from riemann_gfm.manifolds.product_bundle import BundlePoint
from riemann_gfm.modules.contrastive import geometric_contrastive_loss
from riemann_gfm.modules.model import RiemannGFM, RiemannGFMConfig


def _make_bundle(model: RiemannGFM, B: int, N: int) -> BundlePoint:
    d = model.config.d_H
    space = torch.randn(B, N, d) * 0.2
    origin_time = torch.zeros(B, N, 1)
    p_H = model.manifold_H.project(torch.cat([origin_time, space], dim=-1))
    p_S = model.manifold_S.project(torch.cat([origin_time, space], dim=-1))
    z_H = model.manifold_H.project_tangent(p_H, torch.cat([origin_time, space], dim=-1))
    z_S = model.manifold_S.project_tangent(p_S, torch.cat([origin_time, space], dim=-1))
    return BundlePoint(p_H, z_H, p_S, z_S)


def test_forward_and_loss() -> None:
    torch.manual_seed(0)
    cfg = RiemannGFMConfig(d_H=8, d_S=8, hidden=32, n_layers=2)
    model = RiemannGFM(cfg)

    B, N = 2, 6
    point = _make_bundle(model, B, N)
    tree_adj = torch.ones(B, N, N, dtype=torch.bool)
    cycle_adj = torch.ones(B, N, N, dtype=torch.bool)
    anchor_mask = torch.zeros(B, N, dtype=torch.bool)
    anchor_mask[:, 0] = True

    updated = model(point, tree_adj, cycle_adj)
    graph = model.graph_encoding(updated, anchor_mask)

    loss = geometric_contrastive_loss(
        model.manifold_H, model.manifold_S,
        graph.p_H, graph.z_H, graph.p_S, graph.z_S,
    )
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert not torch.isnan(p.grad).any(), name
