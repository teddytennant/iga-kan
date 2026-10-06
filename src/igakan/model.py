"""1D Bernstein hats and a safeguarded local ridge fit.

Degree p=2 only in the experiments. No PDE residual and no tensor-product grid.
"""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_RIDGE = 1e-8


def _as_f64(values):
    return jnp.asarray(values, dtype=jnp.float64)


def knot_vector(n_elements):
    n_elements = int(n_elements)
    return jnp.linspace(0.0, 1.0, n_elements + 1, dtype=jnp.float64)


def hat_supports(n_elements):
    """Left and right ends of each linear hat support on [0, 1]."""
    n_elements = int(n_elements)
    h = 1.0 / n_elements
    i = jnp.arange(n_elements + 1, dtype=jnp.float64)
    left = jnp.where(i == 0.0, 0.0, (i - 1.0) * h)
    right = jnp.where(i == float(n_elements), 1.0, (i + 1.0) * h)
    return left, right


def hat_functions(x, n_elements):
    """Linear B-spline hats on uniform knots of [0, 1].

    n_elements interior elements give n_elements + 1 hats. Hat i is 1 at its
    knot, linear down to the neighboring knots, and 0 outside that support.
    The hats sum to 1 on [0, 1].
    """
    n_elements = int(n_elements)
    x = _as_f64(x).reshape(-1)
    h = 1.0 / n_elements
    n_hats = n_elements + 1
    i = jnp.arange(n_hats, dtype=jnp.float64)[None, :]
    xx = x[:, None]
    left = (i - 1.0) * h
    center = i * h
    right = (i + 1.0) * h
    rise = (xx - left) / h
    fall = (right - xx) / h
    phi = jnp.where((i > 0.0) & (xx >= left) & (xx <= center), rise, 0.0)
    phi = jnp.where((i < float(n_elements)) & (xx >= center) & (xx <= right), fall, phi)
    return phi


def _pow_int(base, exp):
    """Integer power with 0**0 = 1."""
    return jnp.where(exp == 0, 1.0, base ** exp)


def bernstein(xi, p):
    """Bernstein basis B_{j,p}(xi) = binom(p, j) * xi^j * (1-xi)^(p-j)."""
    p = int(p)
    xi = _as_f64(xi)
    j = jnp.arange(p + 1, dtype=jnp.float64)
    coeffs = jnp.array([float(math.comb(p, k)) for k in range(p + 1)], dtype=jnp.float64)
    xi_b = xi[..., None]
    return coeffs * _pow_int(xi_b, j) * _pow_int(1.0 - xi_b, p - j)


def local_xi(x, n_elements):
    """Map each hat support onto [0, 1]. Points outside the support are not used."""
    x = _as_f64(x).reshape(-1)
    left, right = hat_supports(n_elements)
    width = right - left
    return (x[:, None] - left[None, :]) / width[None, :]


def design_features(x, n_elements, p):
    """phi_i(x) * B_j(xi_i(x)). Inactive hats (phi=0) contribute 0."""
    phi = hat_functions(x, n_elements)
    basis = bernstein(local_xi(x, n_elements), p)
    return phi[:, :, None] * basis


def local_predict(x, knots, beta):
    """f(x) = sum_i phi_i(x) * sum_j beta[i, j] * B_{j,p}(xi_i(x))."""
    beta = _as_f64(beta)
    knots = _as_f64(knots).reshape(-1)
    n_elements = int(knots.shape[0] - 1)
    p = int(beta.shape[1] - 1)
    feats = design_features(x, n_elements, p)
    return jnp.sum(feats * beta[None, :, :], axis=(1, 2))


def _residual_norm(pred, y):
    return float(jnp.linalg.norm(_as_f64(pred) - _as_f64(y)))


def accept_if_improved(beta_prev, beta_proposed, x, y, knots):
    """Keep a proposal only when it strictly cuts the residual on the samples."""
    beta_prev = _as_f64(beta_prev)
    beta_proposed = _as_f64(beta_proposed)
    y = _as_f64(y).reshape(-1)
    r_prev = _residual_norm(local_predict(x, knots, beta_prev), y)
    r_new = _residual_norm(local_predict(x, knots, beta_proposed), y)
    if r_new < r_prev:
        return beta_proposed
    return beta_prev


def _ridge_solve(design, target):
    gram = design.T @ design
    n_cols = int(gram.shape[0])
    gram = gram + _RIDGE * jnp.eye(n_cols, dtype=jnp.float64)
    return jnp.linalg.solve(gram, design.T @ target)


def _project_to_ball(beta_ls, beta_prev, delta):
    step = beta_ls - beta_prev
    step_norm = float(jnp.linalg.norm(step))
    if step_norm > delta and step_norm > 0.0:
        return beta_prev + (delta / step_norm) * step
    return beta_ls


def _neighborhood_bounds(element, n_elements):
    h = 1.0 / n_elements
    e_left = max(0, element - 1)
    e_right = min(n_elements - 1, element + 1)
    return e_left * h, (e_right + 1) * h


def safeguarded_fit(x, y, n_elements=4, p=2, delta0=1.0, passes=4):
    """Block least squares with a trust ball and strict residual acceptance.

    beta starts at 0 with shape (n_hats, p+1). Each pass walks elements. The
    sample set for an element is that element plus its two neighbors. Hats with
    phi=0 on every selected sample keep their coefficients exactly. The active
    block is the ridge solution of min ||A beta - y||^2, pulled back onto the
    ball of radius delta around the previous active coefficients when needed.
    A proposal is kept only if the residual on those samples is strictly
    smaller. Otherwise delta is halved and the solve is retried, up to 6
    shrinks. Between elements delta is doubled back toward delta0.
    """
    x = _as_f64(x).reshape(-1)
    y = _as_f64(y).reshape(-1)
    n_elements = int(n_elements)
    p = int(p)
    n_hats = n_elements + 1
    beta = jnp.zeros((n_hats, p + 1), dtype=jnp.float64)
    feats = design_features(x, n_elements, p)
    phi = hat_functions(x, n_elements)
    delta = float(delta0)
    delta0 = float(delta0)
    x_np = np.asarray(x)

    for _pass in range(int(passes)):
        for element in range(n_elements):
            left, right = _neighborhood_bounds(element, n_elements)
            mask = (x_np >= left - 1e-14) & (x_np <= right + 1e-14)
            sample_idx = np.flatnonzero(mask)
            if sample_idx.size == 0:
                delta = min(delta0, delta * 2.0)
                continue

            active = np.asarray(jnp.any(phi[sample_idx] > 0.0, axis=0))
            active_idx = np.flatnonzero(active)
            if active_idx.size == 0:
                delta = min(delta0, delta * 2.0)
                continue

            design = feats[sample_idx][:, active_idx, :].reshape(sample_idx.size, -1)
            target = y[sample_idx]
            beta_prev = beta[active_idx, :].reshape(-1)
            r_prev = float(jnp.linalg.norm(design @ beta_prev - target))

            delta_try = delta
            accepted = False
            for shrink in range(7):
                beta_ls = _ridge_solve(design, target)
                beta_prop = _project_to_ball(beta_ls, beta_prev, delta_try)
                r_new = float(jnp.linalg.norm(design @ beta_prop - target))
                if r_new < r_prev:
                    beta = beta.at[active_idx, :].set(beta_prop.reshape(active_idx.size, p + 1))
                    delta = delta_try
                    accepted = True
                    break
                if shrink < 6:
                    delta_try *= 0.5
            if not accepted:
                delta = delta_try
            delta = min(delta0, delta * 2.0)

    return beta
