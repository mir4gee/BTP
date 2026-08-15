"""Geometric midpoint should lie on the manifold and reduce to identity
when only one weight is non-zero (Theorem 2 sanity)."""

from __future__ import annotations

import pytest
import torch

from riemann_gfm.manifolds.ccs import ConstantCurvatureSpace
from riemann_gfm.modules.midpoint import geometric_midpoint


@pytest.mark.parametrize("kappa", [-1.0, -0.5, 0.5, 1.0])
def test_midpoint_on_manifold(kappa: float) -> None:
    m = ConstantCurvatureSpace(kappa=kappa, dim=8)
    x = m.project(torch.randn(4, 5, 9) * 0.3)     # [B=4, N=5, D=9]
    w = torch.softmax(torch.randn(4, 5), dim=-1)
    mid = geometric_midpoint(m, x, w)
    inner = m.inner(mid, mid, keepdim=False)
    expected = 1.0 / kappa
    assert torch.allclose(inner, torch.full_like(inner, expected), atol=1e-3)


@pytest.mark.parametrize("kappa", [-1.0, 1.0])
def test_midpoint_identity_when_single_weight(kappa: float) -> None:
    m = ConstantCurvatureSpace(kappa=kappa, dim=8)
    x = m.project(torch.randn(1, 3, 9) * 0.3)
    w = torch.tensor([[0.0, 1.0, 0.0]])
    mid = geometric_midpoint(m, x, w)
    assert torch.allclose(mid, x[:, 1], atol=1e-3)
