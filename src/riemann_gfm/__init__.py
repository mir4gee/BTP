"""RiemannGFM — from-scratch reimplementation.

Submodules are not eagerly imported so partial environments (e.g., torch
without PyG) can still load `riemann_gfm.manifolds` for exploration.
"""

__version__ = "0.1.0"

__all__ = ["data", "manifolds", "modules", "training", "utils"]
