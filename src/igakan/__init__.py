"""Local ridge post-process on the unit square."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from igakan.model import (
    accept_patch,
    bernstein,
    bernstein_deriv,
    blend,
    blend_with_fallback,
    circumradius,
    evaluate_local,
    fit_local,
    hat_functions,
    nodal_interpolant,
    physics_block_residual,
    support_box,
    tensor_gauss_points,
    unit_directions,
)

__all__ = [
    "accept_patch",
    "bernstein",
    "bernstein_deriv",
    "blend",
    "blend_with_fallback",
    "circumradius",
    "evaluate_local",
    "fit_local",
    "hat_functions",
    "nodal_interpolant",
    "physics_block_residual",
    "support_box",
    "tensor_gauss_points",
    "unit_directions",
]
