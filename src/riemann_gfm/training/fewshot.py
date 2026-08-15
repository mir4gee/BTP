"""Few-shot node-classification protocol (Sec E.3.1, Xia et al.).

Retain k labelled instances per class from the training split; evaluate on
the standard test split. All other machinery mirrors nc.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import Tensor

from .nc import NCReport, linear_probe


@dataclass
class FewShotConfig:
    k: int = 1                 # 1-shot / 5-shot
    n_trials: int = 5          # average over multiple random supports
    seed: int = 0


def _kshot_train_mask(labels: Tensor, train_mask: Tensor, k: int, rng: np.random.Generator) -> Tensor:
    """Return a boolean mask over the same universe as labels, selecting k
    random labelled nodes per class from the current train_mask."""
    y = labels.cpu().numpy()
    tr = train_mask.cpu().numpy().astype(bool)
    keep = np.zeros_like(tr)
    for cls in np.unique(y[tr]):
        cand = np.where(tr & (y == cls))[0]
        rng.shuffle(cand)
        keep[cand[:k]] = True
    return torch.from_numpy(keep)


def evaluate_fewshot(
    embeddings: Tensor,
    labels: Tensor,
    train_mask: Tensor,
    test_mask: Tensor,
    cfg: FewShotConfig,
) -> List[NCReport]:
    rng = np.random.default_rng(cfg.seed)
    reports: List[NCReport] = []
    for _ in range(cfg.n_trials):
        km = _kshot_train_mask(labels, train_mask, cfg.k, rng)
        reports.append(linear_probe(embeddings, labels, km, test_mask))
    return reports
