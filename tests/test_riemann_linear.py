"""RiemannLinear should map manifold -> manifold (Theorem 1)."""

from __future__ import annotations

import pytest
import torch

from riemann_gfm.manifolds.ccs import ConstantCurvatureSpace
from riemann_gfm.modules.riemann_linear import RiemannLinear


@pytest.mark.parametrize("kappa", [-1.0, -0.3, 0.3, 1.0])
def test_manifold_preserving(kappa: float) -> None:
    torch.manual_seed(0)
    m = ConstantCurvatureSpace(kappa=kappa, dim=8)
    layer = RiemannLinear(m, d_in=8, d_out=8, bias=False)
    x = m.project(torch.randn(32, 9) * 0.3)
    y = layer(x)
    inner = m.inner(y, y, keepdim=False)
    expected = 1.0 / kappa
    assert torch.allclose(inner, torch.full_like(inner, expected), atol=1e-3), inner


@pytest.mark.parametrize("kappa", [-1.0, 1.0])
def test_gradients_flow(kappa: float) -> None:
    torch.manual_seed(0)
    m = ConstantCurvatureSpace(kappa=kappa, dim=8)
    layer = RiemannLinear(m, d_in=8, d_out=8)
    x = m.project(torch.randn(4, 9) * 0.2)
    y = layer(x)
    loss = y.pow(2).sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert not torch.isnan(layer.weight.grad).any()
