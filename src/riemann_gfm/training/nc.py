"""Node-classification downstream head — linear probe on frozen encodings.

Reports accuracy and weighted F1 following Table 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor


@dataclass
class NCReport:
    accuracy: float
    weighted_f1: float


def linear_probe(
    embeddings: Tensor,       # [N, D]
    labels: Tensor,           # [N]
    train_mask: Tensor,       # [N] bool
    test_mask: Tensor,        # [N] bool
    max_iter: int = 2000,
) -> NCReport:
    X = embeddings.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    train_idx = train_mask.cpu().numpy().astype(bool)
    test_idx = test_mask.cpu().numpy().astype(bool)

    clf = LogisticRegression(max_iter=max_iter, n_jobs=-1)
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])

    return NCReport(
        accuracy=float(accuracy_score(y[test_idx], pred)),
        weighted_f1=float(f1_score(y[test_idx], pred, average="weighted")),
    )
