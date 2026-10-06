"""1D isogeometric Kolmogorov-Arnold network toy fit."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

from igakan.model import (
    accept_if_improved,
    bernstein,
    hat_functions,
    local_predict,
    safeguarded_fit,
)

__all__ = [
    "accept_if_improved",
    "bernstein",
    "hat_functions",
    "local_predict",
    "safeguarded_fit",
]
