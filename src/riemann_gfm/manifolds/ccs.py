"""Unified Constant Curvature Space (CCS) — Lorentz/Spherical model.

Follows Eq 1, App D of RiemannGFM (WWW '25). One class handles both

    kappa < 0  →  Lorentz model of hyperbolic space,
    kappa > 0  →  Spherical model of hyperspherical space,

by switching the sign of the time-like component in the inner product.

Convention. Points live in R^{d+1}. Index 0 is the time-like coordinate
x_t; the remaining d entries are the space-like x_s. This matches Sec 2 of
the paper: x = [x_t; x_s].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

_EPS = 1e-15
_CLAMP = 1.0 - 1e-6


@dataclass
class ConstantCurvatureSpace:
    """Constant curvature manifold with unified formalism for kappa != 0.

    Attributes
    ----------
    kappa: float
        Signed curvature. Negative -> hyperbolic (Lorentz), positive ->
        hyperspherical (Spherical). Zero is not supported by this formalism;
        Euclidean space is a special case handled in a separate module if we
        ever need it (Sec 4.2.2 baseline).
    dim: int
        Space-like dimension d. Points have shape [..., d + 1].
    """

    kappa: float
    dim: int

    def __post_init__(self) -> None:
        if self.kappa == 0.0:
            raise ValueError("kappa must be non-zero — Euclidean space is not part of this formalism.")

    @property
    def sign(self) -> float:
        return math.copysign(1.0, self.kappa)

    @property
    def abs_kappa(self) -> float:
        return abs(self.kappa)

    def origin(self, *shape: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
        """North pole o = [1/sqrt(|kappa|), 0, ..., 0]."""
        o = torch.zeros(*shape, self.dim + 1, device=device, dtype=dtype)
        o[..., 0] = 1.0 / math.sqrt(self.abs_kappa)
        return o

    # ------------------------------------------------------------------
    # Core inner products and norms
    # ------------------------------------------------------------------
    def inner(self, x: Tensor, y: Tensor, keepdim: bool = True) -> Tensor:
        """Curvature-aware inner product (Eq 1).

        <x, y>_kappa = sgn(kappa) x_t y_t + x_s^T y_s
        """
        prod = x * y
        s_time = self.sign * prod[..., :1]
        s_space = prod[..., 1:].sum(dim=-1, keepdim=True)
        result = s_time + s_space
        return result if keepdim else result.squeeze(-1)

    def norm(self, x: Tensor, keepdim: bool = True) -> Tensor:
        """|| x ||_kappa = sqrt(|<x, x>_kappa|), guarded by EPS.

        The magnitude is taken so both Lorentz (where <x, x> < 0 for on-manifold x)
        and Spherical (where <x, x> > 0) yield a well-defined real number.
        """
        sq = self.inner(x, x, keepdim=keepdim).abs().clamp_min(_EPS)
        return sq.sqrt()

    # ------------------------------------------------------------------
    # Projection helpers — snap floating-point drift back to the manifold.
    # ------------------------------------------------------------------
    def project(self, x: Tensor) -> Tensor:
        """Project x onto the manifold by fixing x_t to satisfy the constraint.

        For Lorentz (kappa < 0): x_t = sqrt(1/|kappa| + ||x_s||^2)
        For Spherical (kappa > 0): rescale x to lie on the sphere of radius 1/sqrt(kappa).
        """
        if self.kappa < 0:
            x_s = x[..., 1:]
            x_t = torch.sqrt(1.0 / self.abs_kappa + (x_s * x_s).sum(dim=-1, keepdim=True))
            return torch.cat([x_t, x_s], dim=-1)
        # kappa > 0: Spherical. Rescale to the sphere.
        r = math.sqrt(1.0 / self.abs_kappa)
        n = x.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        return x * (r / n)

    def project_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """Project v onto T_x M.

        A vector v is in T_x M iff <x, v>_kappa = 0. We subtract the component
        along x with respect to the curvature-aware inner product.

        Uses the closed-form <x, x> = 1/kappa (a known constant on the
        manifold) rather than the numerical inner product, which avoids the
        sign-flip trap when clamping negative Lorentz inners.
        """
        proj = self.inner(x, v) * self.kappa * x        # 1/(1/kappa) = kappa
        return v - proj

    # ------------------------------------------------------------------
    # Exp / Log — closed forms with curvature-aware cos/sin/arccos (App D).
    # ------------------------------------------------------------------
    def _cos_k(self, t: Tensor) -> Tensor:
        """cos_kappa: cos for kappa>0, cosh for kappa<0."""
        if self.kappa > 0:
            return torch.cos(t)
        return torch.cosh(t)

    def _sin_k(self, t: Tensor) -> Tensor:
        if self.kappa > 0:
            return torch.sin(t)
        return torch.sinh(t)

    def _acos_k(self, t: Tensor) -> Tensor:
        """Inverse of cos_kappa. clamped for numerical safety."""
        if self.kappa > 0:
            return torch.acos(t.clamp(-_CLAMP, _CLAMP))
        # arccosh domain is [1, ∞)
        return torch.acosh(t.clamp_min(1.0 + 1e-7))

    def expmap(self, x: Tensor, v: Tensor) -> Tensor:
        """Exponential map at x applied to tangent vector v.

        Exp_x(v) = cos_k(sqrt(|k|) ||v||) x + sin_k(sqrt(|k|) ||v||) v / (sqrt(|k|) ||v||)
        (App D Eq 22, generalised for both curvatures.)
        """
        sqrt_k = math.sqrt(self.abs_kappa)
        # ||v||_kappa: v is tangent, so <v, v>_kappa >= 0 in both cases.
        # Use magnitude for numerical safety.
        v_norm = self.inner(v, v).abs().clamp_min(_EPS).sqrt()
        theta = sqrt_k * v_norm
        cos = self._cos_k(theta)
        sin = self._sin_k(theta)
        result = cos * x + sin * v / (sqrt_k * v_norm).clamp_min(_EPS)
        return self.project(result)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map — inverse of expmap.

        Log_x(y) = arccos_k(-kappa <x, y>_kappa)   * (y - kappa <x, y>_kappa * x)
                   -------------------------------------  ---------------------
                   sqrt(kappa^2 <x, y>_kappa^2 - 1)      (a projection to T_x M)
        """
        # Following App D Eq 23 which is written for the Lorentz case; the
        # formula generalises to both signs when we use curvature-aware inner
        # product throughout.
        k = self.kappa
        inner_xy = self.inner(x, y)
        # Argument of arccos_k
        arg = -k * inner_xy
        # For hyperbolic (k<0), arg = |k| <x,y> is >= 1 (points are on upper hyperboloid).
        # For spherical (k>0), arg = -k <x,y> lies in [-1, 1].
        acos_arg = self._acos_k(arg)
        denom = (k * k * inner_xy * inner_xy - 1.0).clamp_min(_EPS).sqrt()
        direction = y + k * inner_xy * x  # = y - (-k <x,y>) x = projection direction
        return acos_arg / denom * direction

    # ------------------------------------------------------------------
    # Parallel transport (App D Eq 20)
    # ------------------------------------------------------------------
    def parallel_transport(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        """Transport tangent vector v in T_x M to T_y M along the geodesic.

        PT_{x -> y}(v) = v - kappa <Log_x(y), v>_kappa / d_L(x, y)^2 * (Log_x(y) + Log_y(x))

        In the unified formalism this is equivalent to the simpler form (Eq 20):
            PT_{p_i -> p_l}(z_i) = z_i - kappa <z_i, p_l> / (1 + kappa <p_i, p_l>) * (p_i + p_l)
        which is the form used in the bundle convolution (Eq 7). We implement
        that variant directly since it is what the paper's downstream ops need.
        """
        k = self.kappa
        # <v, y>_kappa
        inner_vy = self.inner(v, y)
        # 1 + k <x, y>_kappa; guarded against zero.
        denom = 1.0 + k * self.inner(x, y)
        # Add EPS with the correct sign so we don't accidentally flip signs.
        denom = torch.where(denom.abs() < _EPS, torch.full_like(denom, _EPS), denom)
        return v - (k * inner_vy / denom) * (x + y)
