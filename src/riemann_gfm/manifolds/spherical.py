"""Spherical model of hyperspherical space — thin wrapper around ConstantCurvatureSpace."""

from __future__ import annotations

from .ccs import ConstantCurvatureSpace


def Spherical(dim: int, kappa: float = 1.0) -> ConstantCurvatureSpace:
    if kappa <= 0:
        raise ValueError("Spherical model requires kappa > 0")
    return ConstantCurvatureSpace(kappa=kappa, dim=dim)
