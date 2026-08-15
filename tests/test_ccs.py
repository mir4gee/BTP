"""Manifold correctness tests. Run inside Docker with `pytest tests/`."""

from __future__ import annotations

import math

import pytest
import torch

from riemann_gfm.manifolds.ccs import ConstantCurvatureSpace


@pytest.fixture(params=[-1.0, -0.5, 0.5, 1.0])
def manifold(request) -> ConstantCurvatureSpace:
    return ConstantCurvatureSpace(kappa=request.param, dim=8)


def _random_on_manifold(m: ConstantCurvatureSpace, n: int = 16) -> torch.Tensor:
    x = torch.randn(n, m.dim + 1) * 0.3
    return m.project(x)


def test_project_lies_on_manifold(manifold):
    x = _random_on_manifold(manifold)
    inner = manifold.inner(x, x, keepdim=False)
    expected = 1.0 / manifold.kappa
    assert torch.allclose(inner, torch.full_like(inner, expected), atol=1e-4), inner


def test_origin_lies_on_manifold(manifold):
    o = manifold.origin(4)
    inner = manifold.inner(o, o, keepdim=False)
    expected = 1.0 / manifold.kappa
    assert torch.allclose(inner, torch.full_like(inner, expected), atol=1e-6)


def test_project_tangent_is_orthogonal(manifold):
    x = _random_on_manifold(manifold)
    v = torch.randn_like(x) * 0.1
    v_proj = manifold.project_tangent(x, v)
    inner_xv = manifold.inner(x, v_proj, keepdim=False)
    assert torch.allclose(inner_xv, torch.zeros_like(inner_xv), atol=1e-4)


def test_expmap_stays_on_manifold(manifold):
    x = _random_on_manifold(manifold)
    v = manifold.project_tangent(x, torch.randn_like(x) * 0.05)
    y = manifold.expmap(x, v)
    inner = manifold.inner(y, y, keepdim=False)
    expected = 1.0 / manifold.kappa
    assert torch.allclose(inner, torch.full_like(inner, expected), atol=1e-3)


def test_parallel_transport_produces_tangent(manifold):
    x = _random_on_manifold(manifold)
    y = _random_on_manifold(manifold)
    v = manifold.project_tangent(x, torch.randn_like(x) * 0.05)
    v_new = manifold.parallel_transport(x, y, v)
    # v_new should be tangent at y.
    inner_yv = manifold.inner(y, v_new, keepdim=False)
    # Somewhat loose tol because the closed form in the paper is an
    # approximation for large curvatures; the numerical drift is bounded.
    assert torch.allclose(inner_yv, torch.zeros_like(inner_yv), atol=1e-2)
