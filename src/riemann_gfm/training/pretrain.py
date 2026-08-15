"""Pre-training loop — Algorithm 1.

Simplified single-graph variant. For multi-dataset pretraining (Table 1) the
caller loops over datasets and re-invokes pretrain(); state carries via the
model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import torch
from torch import nn
from torch_geometric.data import Data

from ..data.init_encoding import laplacian_positional_encoding
from ..data.vocab_sampler import batch_substructures, sample_substructures
from ..manifolds.product_bundle import BundlePoint
from ..modules.contrastive import geometric_contrastive_loss
from ..modules.model import RiemannGFM


@dataclass
class PretrainConfig:
    dim: int = 32
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.1
    lr: float = 0.01
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 3
    iters_per_dataset: int = 3
    tree_depth: int = 2
    cycle_max_len: int = 6
    seed: int = 42
    device: str = "cpu"


def _init_bundle(model: RiemannGFM, feats: torch.Tensor) -> BundlePoint:
    """Initialise a BundlePoint from Laplacian-eigvec features.

    feats: [B, N, K] with K matching model.config.d_H == d_S.
    """
    d_H = model.config.d_H
    d_S = model.config.d_S
    assert feats.size(-1) == d_H == d_S, "feature dim must match both factors"

    # Place features into the space-like block; time coord chosen so the
    # point is on-manifold. This is Exp_o([0 | z^T]) as per Sec E.3.2.
    B, N, K = feats.shape
    # H factor
    p_H_space = feats
    p_H = torch.cat([torch.zeros(B, N, 1, device=feats.device, dtype=feats.dtype), p_H_space], dim=-1)
    p_H = model.manifold_H.project(p_H)
    # S factor
    p_S_space = feats
    p_S = torch.cat([torch.zeros(B, N, 1, device=feats.device, dtype=feats.dtype), p_S_space], dim=-1)
    p_S = model.manifold_S.project(p_S)

    # z on each factor initialised to the projection of feats onto T_p M.
    z_H_amb = torch.cat([torch.zeros(B, N, 1, device=feats.device, dtype=feats.dtype), feats], dim=-1)
    z_H = model.manifold_H.project_tangent(p_H, z_H_amb)
    z_S_amb = torch.cat([torch.zeros(B, N, 1, device=feats.device, dtype=feats.dtype), feats], dim=-1)
    z_S = model.manifold_S.project_tangent(p_S, z_S_amb)

    return BundlePoint(p_H=p_H, z_H=z_H, p_S=p_S, z_S=z_S)


def pretrain(model: RiemannGFM, datasets: List[Data], cfg: PretrainConfig) -> None:
    """Pretrain model in-place on a list of graphs.

    Each iteration samples cfg.batch_size anchor substructures per graph,
    runs the encoder, aggregates to graph-level via geometric midpoint over
    the anchor, and computes the geometric contrastive loss.
    """
    device = torch.device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Pre-compute Laplacian features per graph so we only do the eigen-decomp
    # once per dataset per run.
    all_feats: List[torch.Tensor] = []
    for data in datasets:
        feats = laplacian_positional_encoding(
            num_nodes=data.num_nodes, edge_index=data.edge_index, k=cfg.dim
        ).to(device)
        all_feats.append(feats)

    model.train()
    for epoch in range(cfg.epochs):
        for it in range(cfg.iters_per_dataset):
            for data, feats in zip(datasets, all_feats):
                # Sample cfg.batch_size anchors uniformly.
                idx = torch.randperm(data.num_nodes)[: cfg.batch_size].tolist()
                subs = sample_substructures(
                    edge_index=data.edge_index,
                    num_nodes=data.num_nodes,
                    anchors=idx,
                    tree_depth=cfg.tree_depth,
                    cycle_max_len=cfg.cycle_max_len,
                )
                feats_b, tree_adj, cycle_adj, anchor_mask = batch_substructures(subs, feats)

                point = _init_bundle(model, feats_b)
                point = model(point, tree_adj, cycle_adj)
                graph_point = model.graph_encoding(point, anchor_mask)

                # Squeeze the singleton N dim so contrastive sees [B, d + 1].
                # graph_encoding returned shape [B, d + 1] since target_mask reduced N.
                loss = geometric_contrastive_loss(
                    model.manifold_H, model.manifold_S,
                    graph_point.p_H, graph_point.z_H,
                    graph_point.p_S, graph_point.z_S,
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()

                print(f"epoch {epoch} iter {it} loss {loss.item():.4f}")
