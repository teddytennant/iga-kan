"""Local ridge post-process on the unit square.

JAX implementation of the vertex-wise ridge fit from arXiv:2610.06348.
Identity geometry, degree-1 tensor-product hats, fixed unit directions.
The post-process is the paper's. Nodal interpolation stands in for the
Galerkin solve. The paper uses dloc=9 and Q=12.
"""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# s = (z + SHIFT) / SCALE maps the ridge coordinate into the Bernstein interval.
SHIFT = 1.02
SCALE = 2.04

# Collocation weights on the stacked least squares. Not tuned here.
DATA_WEIGHT = 0.1
DIRICHLET_WEIGHT = 100.0


def _as_f64(values):
    return jnp.asarray(values, dtype=jnp.float64)


def _integer_powers(base, degree):
    """base**0 .. base**degree along a new last axis. base**0 is 1."""
    degree = int(degree)
    base = _as_f64(base)
    acc = jnp.ones_like(base)
    cols = [acc]
    for _ in range(degree):
        acc = acc * base
        cols.append(acc)
    return jnp.stack(cols, axis=-1)


def _bernstein(s, degree):
    degree = int(degree)
    s = _as_f64(s)
    binom = jnp.array(
        [float(math.comb(degree, k)) for k in range(degree + 1)],
        dtype=jnp.float64,
    )
    powers = _integer_powers(s, degree)
    powers_1ms = _integer_powers(1.0 - s, degree)
    return binom * powers * jnp.flip(powers_1ms, axis=-1)


def _bernstein_deriv(s, degree, order):
    """Degree-lowering formula, order times.

    d/ds B_{k,p} = p * (B_{k-1,p-1} - B_{k,p-1}), with the out-of-range
    basis functions taken to be zero.
    """
    degree = int(degree)
    order = int(order)
    s = _as_f64(s)
    if order == 0:
        return _bernstein(s, degree)
    if order > degree or degree < 0:
        return jnp.zeros(s.shape + (degree + 1,), dtype=jnp.float64)
    lower = _bernstein_deriv(s, degree - 1, order - 1)
    left = jnp.pad(lower, [(0, 0)] * (lower.ndim - 1) + [(1, 0)])
    right = jnp.pad(lower, [(0, 0)] * (lower.ndim - 1) + [(0, 1)])
    return degree * (left - right)


def bernstein(s, degree):
    """Bernstein basis B_{k,degree}(s), last axis over k = 0..degree."""
    return np.asarray(_bernstein(s, degree), dtype=np.float64)


def bernstein_deriv(s, degree, order):
    """Derivative of order `order` of each Bernstein basis function in s."""
    return np.asarray(_bernstein_deriv(s, degree, order), dtype=np.float64)


def unit_directions(Q):
    """Fixed unit directions a_q = (cos(q pi / Q), sin(q pi / Q)), q = 0..Q-1.

    Not learned. The paper uses Q=12.
    """
    Q = int(Q)
    theta = np.arange(Q, dtype=np.float64) * np.pi / Q
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def knots_1d(n):
    """Uniform knots of n elements on [0, 1]."""
    return np.linspace(0.0, 1.0, int(n) + 1, dtype=np.float64)


def _hats_1d(coord, n):
    """Degree-1 hats on the uniform knots of [0, 1].

    Points are assumed to lie in [0, 1]. The hats are non-negative and sum to 1.
    Hat i is supported on the elements that touch knot i.
    """
    n = int(n)
    coord = np.asarray(coord, dtype=np.float64).reshape(-1)
    knots = knots_1d(n)
    h = float(knots[1] - knots[0])
    scaled = coord / h
    element = np.floor(scaled).astype(np.int64)
    element = np.clip(element, 0, n - 1)
    xi = (coord - element * h) / h
    values = np.zeros((coord.shape[0], n + 1), dtype=np.float64)
    rows = np.arange(coord.shape[0])
    values[rows, element] = 1.0 - xi
    values[rows, element + 1] = xi
    return values


def hat_functions(points, n):
    """Degree-1 tensor-product hats, one per knot vertex.

    Returns an array of shape (n_points, n+1, n+1). Values are non-negative
    and sum to 1 on [0, 1]^2. The support of N_v is the elements that touch v.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    hx = _hats_1d(points[:, 0], n)
    hy = _hats_1d(points[:, 1], n)
    return hx[:, :, None] * hy[:, None, :]


def nodal_interpolant(points, nodal, n):
    """Degree-1 tensor-product interpolant of nodal values on the uniform mesh."""
    weights = hat_functions(points, n)
    nodal = np.asarray(nodal, dtype=np.float64)
    return np.tensordot(weights, nodal, axes=([1, 2], [0, 1]))


def support_box(i, j, n):
    """Axis-aligned box of the elements that touch vertex (i, j)."""
    n = int(n)
    knots = knots_1d(n)
    i = int(i)
    j = int(j)
    i0 = max(i - 1, 0)
    i1 = min(i + 1, n)
    j0 = max(j - 1, 0)
    j1 = min(j + 1, n)
    return (float(knots[i0]), float(knots[i1]), float(knots[j0]), float(knots[j1]))


def circumradius(xv, box):
    """Radius of the smallest circle centered at xv that covers the box corners.

    For an interior vertex this is the circumradius of the 2-by-2 element patch.
    """
    xlo, xhi, ylo, yhi = box
    corners = np.array(
        [[xlo, ylo], [xlo, yhi], [xhi, ylo], [xhi, yhi]],
        dtype=np.float64,
    )
    xv = np.asarray(xv, dtype=np.float64).reshape(2)
    return float(np.max(np.linalg.norm(corners - xv, axis=1)))


def _leggauss(n_1d):
    xi, weights = np.polynomial.legendre.leggauss(int(n_1d))
    return xi.astype(np.float64, copy=False), weights.astype(np.float64, copy=False)


def _map_interval(xi, left, right):
    left = float(left)
    right = float(right)
    return 0.5 * (left + right) + 0.5 * (right - left) * np.asarray(xi, dtype=np.float64)


def tensor_gauss_points(box, n_1d):
    """Tensor-product Gauss points of the patch box, n_1d per direction."""
    xi, _ = _leggauss(n_1d)
    xlo, xhi, ylo, yhi = box
    xs = _map_interval(xi, xlo, xhi)
    ys = _map_interval(xi, ylo, yhi)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def dirichlet_points(box, n_1d):
    """Gauss points on the Dirichlet sides of the unit square that the patch touches."""
    xlo, xhi, ylo, yhi = box
    xi, _ = _leggauss(n_1d)
    blocks = []
    if abs(xlo - 0.0) <= 1e-12:
        blocks.append(np.stack([np.zeros(xi.shape[0], dtype=np.float64), _map_interval(xi, ylo, yhi)], axis=1))
    if abs(xhi - 1.0) <= 1e-12:
        blocks.append(np.stack([np.ones(xi.shape[0], dtype=np.float64), _map_interval(xi, ylo, yhi)], axis=1))
    if abs(ylo - 0.0) <= 1e-12:
        blocks.append(np.stack([_map_interval(xi, xlo, xhi), np.zeros(xi.shape[0], dtype=np.float64)], axis=1))
    if abs(yhi - 1.0) <= 1e-12:
        blocks.append(np.stack([_map_interval(xi, xlo, xhi), np.ones(xi.shape[0], dtype=np.float64)], axis=1))
    if not blocks:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(blocks, axis=0)


def ridge_basis(points, xv, rho, directions, dloc):
    """Bernstein values and exact z-derivatives of each ridge basis function.

    z_q = a_q · (x - x_v) / rho
    s = (z + 1.02) / 2.04
    dPhi/dz = dPhi/ds * (1/2.04)
    d2Phi/dz2 = d2Phi/ds2 * (1/2.04)^2

    Returns B, dB/dz, d2B/dz2, each of shape (n_points, Q, dloc+1).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    xv = np.asarray(xv, dtype=np.float64).reshape(2)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 2)
    rho = float(rho)
    z = ((points - xv) @ directions.T) / rho
    s = (z + SHIFT) / SCALE
    values = bernstein(s, dloc)
    d_ds = bernstein_deriv(s, dloc, 1)
    d2_ds2 = bernstein_deriv(s, dloc, 2)
    return values, d_ds / SCALE, d2_ds2 / (SCALE ** 2)


def evaluate_local(points, xv, rho, directions, coeffs):
    """Value, gradient, and Laplacian of K_v = sum_q Phi_q(z_q).

    grad K = (1/rho) sum_q Phi'(z_q) a_q
    lap K = (1/rho^2) sum_q Phi''(z_q)
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    dloc = int(coeffs.shape[1] - 1)
    rho = float(rho)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 2)
    values, d_dz, d2_dz2 = ridge_basis(points, xv, rho, directions, dloc)
    value = np.sum(values * coeffs, axis=(-2, -1))
    phi_z = np.sum(d_dz * coeffs, axis=-1)
    phi_zz = np.sum(d2_dz2 * coeffs, axis=-1)
    grad = (phi_z @ directions) / rho
    lap = np.sum(phi_zz, axis=-1) / (rho ** 2)
    return value, grad, lap


def _minimum_norm_svd(matrix, rhs):
    """Minimum-norm least squares from the thin SVD. Not an optimizer."""
    matrix = np.asarray(matrix, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or singular[0] == 0.0:
        return np.zeros(matrix.shape[1], dtype=np.float64)
    cutoff = np.finfo(np.float64).eps * max(matrix.shape) * float(singular[0])
    inv = np.zeros_like(singular)
    keep = singular > cutoff
    inv[keep] = 1.0 / singular[keep]
    return (vt.T * inv) @ (u.T @ rhs)


def fit_local(points, f_vals, uh_vals, bc_points, g_vals, xv, rho, directions, dloc):
    """One minimum-norm SVD least squares for the coefficients of K_v.

    The paper uses dloc=9 and Q=12. Three blocks are stacked at the given
    sample points:

    physics: row -Phi''_q,k(z), right hand side rho^2 * f.
    With L = -lap this is the collocation of rho^2 * (L K - f).
    data: DATA_WEIGHT * (K - u_h)
    boundary: DIRICHLET_WEIGHT * (K - g) on Dirichlet sides the patch touches
    """
    dloc = int(dloc)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 2)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    f_vals = np.asarray(f_vals, dtype=np.float64).reshape(-1)
    uh_vals = np.asarray(uh_vals, dtype=np.float64).reshape(-1)
    bc_points = np.asarray(bc_points, dtype=np.float64).reshape(-1, 2)
    g_vals = np.asarray(g_vals, dtype=np.float64).reshape(-1)
    rho = float(rho)
    n_dir = directions.shape[0]
    n_basis = dloc + 1
    values, _, d2_dz2 = ridge_basis(points, xv, rho, directions, dloc)
    # -Phi''(z) per coefficient. Phi'' already includes (1/2.04)^2.
    physics = (-d2_dz2).reshape(points.shape[0], n_dir * n_basis)
    physics_rhs = (rho ** 2) * f_vals
    data = DATA_WEIGHT * values.reshape(points.shape[0], n_dir * n_basis)
    data_rhs = DATA_WEIGHT * uh_vals
    blocks = [physics, data]
    rhs_blocks = [physics_rhs, data_rhs]
    if bc_points.shape[0] > 0:
        bc_values, _, _ = ridge_basis(bc_points, xv, rho, directions, dloc)
        blocks.append(DIRICHLET_WEIGHT * bc_values.reshape(bc_points.shape[0], n_dir * n_basis))
        rhs_blocks.append(DIRICHLET_WEIGHT * g_vals)
    matrix = np.concatenate(blocks, axis=0)
    rhs = np.concatenate(rhs_blocks, axis=0)
    coeffs = _minimum_norm_svd(matrix, rhs)
    return coeffs.reshape(n_dir, n_basis)


def physics_block_residual(points, xv, rho, directions, coeffs, f_vals):
    """RMS of -sum_q Phi''(z_q) - rho^2 * f on the sample points.

    That is the physics block of the least squares. L = -lap, so the same
    quantity is rho^2 * (L K - f). A zero coefficient vector leaves rms(rho^2 * f).
    """
    rho = float(rho)
    _, _, lap = evaluate_local(points, xv, rho, directions, coeffs)
    f_vals = np.asarray(f_vals, dtype=np.float64).reshape(-1)
    mismatch = -(rho ** 2) * lap - (rho ** 2) * f_vals
    return float(np.sqrt(np.mean(mismatch ** 2)))


def blend(hats, local_values):
    """u_hyb(x) = sum_v N_v(x) K_v(x).

    hats and local_values have shape (n_points, n_x, n_y).
    """
    hats = np.asarray(hats, dtype=np.float64)
    local_values = np.asarray(local_values, dtype=np.float64)
    return np.sum(hats * local_values, axis=(-2, -1))


def accept_patch(s, delta, r, rms_uh):
    """Pure safeguard. Not used by the recovery fit.

    Accept if s < 0.5 * delta and delta < 0.3 * rms(u_h) and r < delta.
    """
    s = float(s)
    delta = float(delta)
    r = float(r)
    rms_uh = float(rms_uh)
    return bool((s < 0.5 * delta) and (delta < 0.3 * rms_uh) and (r < delta))


def blend_with_fallback(hats, local_values, uh, accepted):
    """Blend local models, using u_h on every rejected vertex.

    accepted has shape (n_x, n_y). uh is the nodal interpolant at the same
    points as hats, shape (n_points,).
    """
    hats = np.asarray(hats, dtype=np.float64)
    local_values = np.asarray(local_values, dtype=np.float64)
    uh = np.asarray(uh, dtype=np.float64).reshape(-1)
    accepted = np.asarray(accepted, dtype=bool)
    mixed = np.where(accepted[None, :, :], local_values, uh[:, None, None])
    return blend(hats, mixed)
