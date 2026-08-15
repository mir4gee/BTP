"""Link-prediction downstream head.

Scorer: dot product on tangent-space encodings. Reports AUC + AP as per Table 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch_geometric.utils import negative_sampling


@dataclass
class LPReport:
    auc: float
    ap: float


def evaluate_lp(
    embeddings: Tensor,       # [N, D]
    edge_index: Tensor,       # [2, E] positive test edges
    num_nodes: int,
    negatives_per_pos: int = 1,
    seed: int = 0,
) -> LPReport:
    torch.manual_seed(seed)
    pos = edge_index
    neg = negative_sampling(
        edge_index=pos,
        num_nodes=num_nodes,
        num_neg_samples=pos.size(1) * negatives_per_pos,
    )

    def _score(edges: Tensor) -> Tensor:
        u = embeddings[edges[0]]
        v = embeddings[edges[1]]
        return (u * v).sum(dim=-1)

    pos_scores = _score(pos).detach().cpu().numpy()
    neg_scores = _score(neg).detach().cpu().numpy()

    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = np.concatenate([pos_scores, neg_scores])
    return LPReport(
        auc=float(roc_auc_score(y_true, y_score)),
        ap=float(average_precision_score(y_true, y_score)),
    )
