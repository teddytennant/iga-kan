"""Checks for the local ridge post-process on the unit square."""

import numpy as np

from igakan.model import (
    accept_patch,
    blend,
    blend_with_fallback,
    circumradius,
    dirichlet_points,
    evaluate_local,
    fit_local,
    hat_functions,
    knots_1d,
    nodal_interpolant,
    physics_block_residual,
    ridge_basis,
    support_box,
    tensor_gauss_points,
    unit_directions,
)


def test_directions_are_fixed_unit_vectors():
    Q = 4
    got = unit_directions(Q)
    theta = np.arange(Q, dtype=np.float64) * np.pi / Q
    expected = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    np.testing.assert_allclose(got, expected, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-15)


def test_hats_nonnegative_partition_of_unity_and_support():
    n = 6
    h = 1.0 / n
    xs = np.linspace(0.0, 1.0, 31, dtype=np.float64)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    weights = hat_functions(points, n)
    assert np.all(weights >= -1e-15)
    np.testing.assert_allclose(weights.sum(axis=(1, 2)), 1.0, atol=1e-12)
    i, j = 3, 2
    outside = (
        (points[:, 0] < (i - 1) * h - 1e-12)
        | (points[:, 0] > (i + 1) * h + 1e-12)
        | (points[:, 1] < (j - 1) * h - 1e-12)
        | (points[:, 1] > (j + 1) * h + 1e-12)
    )
    assert np.all(np.abs(weights[outside, i, j]) <= 1e-12)
    vertex = hat_functions(np.array([[i * h, j * h]], dtype=np.float64), n)
    np.testing.assert_allclose(vertex[0, i, j], 1.0, atol=1e-12)


def test_interior_circumradius_of_two_by_two_patch():
    n = 6
    knots = knots_1d(n)
    i, j = 3, 4
    box = support_box(i, j, n)
    xv = np.array([knots[i], knots[j]], dtype=np.float64)
    h = float(knots[1] - knots[0])
    np.testing.assert_allclose(
        box,
        (knots[i - 1], knots[i + 1], knots[j - 1], knots[j + 1]),
        atol=1e-15,
    )
    np.testing.assert_allclose(circumradius(xv, box), h * np.sqrt(2.0), rtol=0.0, atol=1e-12)


def test_quadratic_ridge_grad_and_lap_match_hand_expansion():
    # One quadratic ridge. Hand derivatives of the degree-2 Bernstein basis,
    # then the chain rule through s = (z + 1.02) / 2.04.
    alpha = 1.02
    beta = 2.04
    rho = 1.5
    xv = np.array([0.2, 0.1], dtype=np.float64)
    direction = np.array([0.6, 0.8], dtype=np.float64)
    coeffs = np.array([[1.0, -0.5, 0.25]], dtype=np.float64)
    points = np.array(
        [[0.5, 0.7], [0.2, 0.1], [0.9, 0.4]],
        dtype=np.float64,
    )
    value, grad, lap = evaluate_local(points, xv, rho, direction.reshape(1, 2), coeffs)

    z = ((points - xv) @ direction) / rho
    s = (z + alpha) / beta
    b0 = (1.0 - s) ** 2
    b1 = 2.0 * s * (1.0 - s)
    b2 = s ** 2
    c0, c1, c2 = 1.0, -0.5, 0.25
    phi = c0 * b0 + c1 * b1 + c2 * b2
    dphi_ds = c0 * (-2.0 * (1.0 - s)) + c1 * (2.0 - 4.0 * s) + c2 * (2.0 * s)
    dphi_dz = dphi_ds / beta
    d2phi_dz2 = (c0 * 2.0 + c1 * (-4.0) + c2 * 2.0) / (beta ** 2)
    grad_hand = (dphi_dz / rho)[:, None] * direction[None, :]
    lap_hand = np.full(points.shape[0], d2phi_dz2 / (rho ** 2), dtype=np.float64)

    np.testing.assert_allclose(value, phi, atol=1e-12)
    np.testing.assert_allclose(grad, grad_hand, atol=1e-12)
    np.testing.assert_allclose(lap, lap_hand, atol=1e-12)


def test_physics_rows_match_scaled_laplacian():
    rng = np.random.default_rng(2)
    dloc = 3
    directions = unit_directions(4)
    coeffs = rng.normal(size=(4, dloc + 1))
    xv = np.array([0.45, 0.55], dtype=np.float64)
    rho = 0.35
    points = rng.uniform(0.25, 0.75, size=(9, 2))
    _, _, d2_dz2 = ridge_basis(points, xv, rho, directions, dloc)
    assembled = -np.sum(d2_dz2 * coeffs, axis=(-2, -1))
    _, _, lap = evaluate_local(points, xv, rho, directions, coeffs)
    np.testing.assert_allclose(assembled, -(rho ** 2) * lap, atol=1e-12)


def test_consistent_ridge_is_reproduced():
    # If physics, data, and no boundary are all satisfied by a known ridge,
    # the minimum-norm solution must reproduce that ridge. A wrong right-hand
    # side (f instead of rho^2 * f) does not.
    directions = np.array([[1.0, 0.0]], dtype=np.float64)
    coeffs_true = np.array([[0.2, -0.1, 0.05]], dtype=np.float64)
    xv = np.array([0.5, 0.5], dtype=np.float64)
    rho = 0.4
    xi = np.linspace(-0.15, 0.15, 6, dtype=np.float64)
    xx, yy = np.meshgrid(xi, xi, indexing="ij")
    points = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], axis=1)
    value, _, lap = evaluate_local(points, xv, rho, directions, coeffs_true)
    f_vals = -lap
    coeffs = fit_local(
        points,
        f_vals,
        value,
        np.zeros((0, 2), dtype=np.float64),
        np.zeros(0, dtype=np.float64),
        xv,
        rho,
        directions,
        2,
    )
    value_fit, _, lap_fit = evaluate_local(points, xv, rho, directions, coeffs)
    np.testing.assert_allclose(value_fit, value, atol=1e-9)
    np.testing.assert_allclose(lap_fit, lap, atol=1e-9)
    np.testing.assert_allclose(coeffs, coeffs_true, atol=1e-8)


def test_blend_equals_uh_when_every_local_model_returns_uh():
    n = 5
    rng = np.random.default_rng(1)
    points = np.vstack(
        [
            rng.random((24, 2)),
            np.array([[0.0, 0.0], [1.0, 1.0], [0.4, 0.2], [1.0, 0.0]], dtype=np.float64),
        ]
    )
    nodal = rng.normal(size=(n + 1, n + 1))
    hats = hat_functions(points, n)
    uh = nodal_interpolant(points, nodal, n)
    local_values = np.broadcast_to(uh[:, None, None], hats.shape).copy()
    np.testing.assert_allclose(blend(hats, local_values), uh, atol=1e-12)


def test_safeguard_on_hand_built_tuples():
    # Meets s < 0.5 * delta, delta < 0.3 * rms(u_h), and r < delta.
    assert accept_patch(0.04, 0.10, 0.09, 1.0) is True
    # Same s, delta, and rms, but r is not strictly below delta.
    assert accept_patch(0.04, 0.10, 0.10, 1.0) is False


def test_rejected_patch_falls_back_to_uh():
    n = 2
    rng = np.random.default_rng(0)
    points = rng.random((8, 2))
    nodal = rng.normal(size=(n + 1, n + 1))
    hats = hat_functions(points, n)
    uh = nodal_interpolant(points, nodal, n)
    local_values = rng.normal(size=hats.shape)
    accepted = np.zeros((n + 1, n + 1), dtype=bool)
    accepted[1, 0] = True
    mixed = blend_with_fallback(hats, local_values, uh, accepted)
    expected_values = np.broadcast_to(uh[:, None, None], hats.shape).copy()
    expected_values[:, 1, 0] = local_values[:, 1, 0]
    np.testing.assert_allclose(mixed, np.sum(hats * expected_values, axis=(1, 2)), atol=1e-12)
    none = np.zeros_like(accepted)
    np.testing.assert_allclose(blend_with_fallback(hats, local_values, uh, none), uh, atol=1e-12)
    all_ok = np.ones_like(accepted)
    np.testing.assert_allclose(
        blend_with_fallback(hats, local_values, uh, all_ok),
        blend(hats, local_values),
        atol=1e-12,
    )


def test_poisson_recovery_beats_nodal_interpolant():
    # Manufactured Poisson problem. The safeguard is not applied.
    # u_h is the degree-1 nodal interpolant, in place of a Galerkin solve.
    n = 6
    dloc = 3
    Q = 4
    n_gauss = 5
    directions = unit_directions(Q)
    knots = knots_1d(n)
    nodal = np.sin(np.pi * knots)[:, None] * np.sin(np.pi * knots)[None, :]
    fitted = {}
    for i in range(n + 1):
        for j in range(n + 1):
            box = support_box(i, j, n)
            xv = np.array([knots[i], knots[j]], dtype=np.float64)
            rho = circumradius(xv, box)
            points = tensor_gauss_points(box, n_gauss)
            uh = nodal_interpolant(points, nodal, n)
            f_vals = (2.0 * np.pi ** 2) * np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])
            bc = dirichlet_points(box, n_gauss)
            g_vals = np.zeros(bc.shape[0], dtype=np.float64)
            coeffs = fit_local(points, f_vals, uh, bc, g_vals, xv, rho, directions, dloc)
            zero = np.zeros_like(coeffs)
            residual = physics_block_residual(points, xv, rho, directions, coeffs, f_vals)
            residual_zero = physics_block_residual(points, xv, rho, directions, zero, f_vals)
            assert residual < residual_zero
            fitted[(i, j)] = (xv, rho, coeffs)

    xs = (np.arange(20, dtype=np.float64) + 0.5) / 20.0
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    hats = hat_functions(grid, n)
    uh_grid = nodal_interpolant(grid, nodal, n)
    local_values = np.zeros((grid.shape[0], n + 1, n + 1), dtype=np.float64)
    for i in range(n + 1):
        for j in range(n + 1):
            xv, rho, coeffs = fitted[(i, j)]
            local_values[:, i, j], _, _ = evaluate_local(grid, xv, rho, directions, coeffs)
    hybrid = blend(hats, local_values)
    true = np.sin(np.pi * grid[:, 0]) * np.sin(np.pi * grid[:, 1])
    err_uh = float(np.sqrt(np.mean((uh_grid - true) ** 2)))
    err_hybrid = float(np.sqrt(np.mean((hybrid - true) ** 2)))
    assert err_hybrid < err_uh
