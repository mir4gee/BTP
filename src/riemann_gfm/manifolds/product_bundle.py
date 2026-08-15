"""Product tangent bundle (Eq 2).

P^{d_P} = (H^{d_H} ⊗ TH^{d_H}) ⊗ (S^{d_S} ⊗ TS^{d_S}),  d_P = 2 d_H + 2 d_S.

Each node carries four tensors:
    p_H : [..., d_H + 1]   coordinate on hyperbolic factor
    z_H : [..., d_H + 1]   encoding in T_{p_H} H
    p_S : [..., d_S + 1]   coordinate on hyperspherical factor
    z_S : [..., d_S + 1]   encoding in T_{p_S} S

The bundle itself has no operations; it is a container passed between modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .ccs import ConstantCurvatureSpace
from .lorentz import Lorentz
from .spherical import Spherical


@dataclass
class BundlePoint:
    """A batch of points on the product tangent bundle."""

    p_H: Tensor
    z_H: Tensor
    p_S: Tensor
    z_S: Tensor

    def detach(self) -> "BundlePoint":
        return BundlePoint(self.p_H.detach(), self.z_H.detach(), self.p_S.detach(), self.z_S.detach())

    def to(self, *args, **kwargs) -> "BundlePoint":
        return BundlePoint(
            self.p_H.to(*args, **kwargs),
            self.z_H.to(*args, **kwargs),
            self.p_S.to(*args, **kwargs),
            self.z_S.to(*args, **kwargs),
        )


class ProductBundle:
    """Groups the two constant-curvature factors used by RiemannGFM."""

    def __init__(self, d_H: int, d_S: int, kappa_H: float = -1.0, kappa_S: float = 1.0) -> None:
        self.H: ConstantCurvatureSpace = Lorentz(d_H, kappa_H)
        self.S: ConstantCurvatureSpace = Spherical(d_S, kappa_S)
        self.d_H = d_H
        self.d_S = d_S

    def origin(self, *shape: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> BundlePoint:
        """Bundle point at both north poles with zero tangent encodings."""
        p_H = self.H.origin(*shape, device=device, dtype=dtype)
        p_S = self.S.origin(*shape, device=device, dtype=dtype)
        z_H = torch.zeros_like(p_H)
        z_S = torch.zeros_like(p_S)
        return BundlePoint(p_H, z_H, p_S, z_S)
