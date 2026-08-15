"""Geometric contrastive loss (Eq 8).

Positive pair: the same node's hyperbolic and hyperspherical tangent-space
encodings, parallel-transported to the shared north-pole tangent space of
each factor.
Negative pairs: other nodes in the batch.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from ..manifolds.ccs import ConstantCurvatureSpace


def _transport_to_origin(manifold: ConstantCurvatureSpace, p: Tensor, z: Tensor) -> Tensor:
    """PT_{p -> o}(z) using the CCS parallel_transport helper."""
    o = manifold.origin(*p.shape[:-1], device=p.device, dtype=p.dtype)
    return manifold.parallel_transport(p, o, z)


def geometric_contrastive_loss(
    manifold_H: ConstantCurvatureSpace,
    manifold_S: ConstantCurvatureSpace,
    p_H: Tensor, z_H: Tensor,
    p_S: Tensor, z_S: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Symmetric geometric contrastive loss J(H, S) + J(S, H).

    Input shapes: [N, d + 1]. Non-batched — pretraining calls this once per
    mini-batch of nodes.
    """
    # Transport to shared tangent space at the north pole of each factor.
    z_H_o = _transport_to_origin(manifold_H, p_H, z_H)      # [N, d_H + 1]
    z_S_o = _transport_to_origin(manifold_S, p_S, z_S)      # [N, d_S + 1]

    # Sim matrix. The two tangent-spaces are at different geometric origins
    # but both are Euclidean vector spaces around the north pole. We take the
    # standard inner product over a shared coordinate range — this matches
    # the paper's definition where <PT(z^H), PT(z^S)> is a scalar comparison.
    # If d_H != d_S the paper implicitly assumes they match (both = 32). We
    # enforce that here.
    if z_H_o.shape[-1] != z_S_o.shape[-1]:
        raise ValueError("H and S factor dimensions must match for the contrastive loss.")

    sim = z_H_o @ z_S_o.transpose(-1, -2) / temperature     # [N, N]

    labels = torch.arange(sim.size(0), device=sim.device)
    # J(H -> S): row-softmax over S negatives, positive = diagonal.
    loss_hs = torch.nn.functional.cross_entropy(sim, labels)
    # J(S -> H): column-softmax over H negatives.
    loss_sh = torch.nn.functional.cross_entropy(sim.t(), labels)
    return loss_hs + loss_sh
