"""Seeded checks for the 1D Bernstein hat model and safeguarded fit."""

import math

import numpy as np
import pytest

np.random.seed(0)

from igakan.model import (
    accept_if_improved,
    bernstein,
    hat_functions,
    local_predict,
    safeguarded_fit,
)


def _l2(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def test_bernstein_p2_partition_and_binomial():
    xi = np.linspace(0.0, 1.0, 21)
    basis = np.asarray(bernstein(xi, 2))
    np.testing.assert_allclose(basis.sum(axis=-1), 1.0, atol=1e-12)
    assert basis[0, 0] == 1.0
    assert basis[-1, 2] == 1.0
    xi0 = 0.3
    got = np.asarray(bernstein(xi0, 2))
    expected = np.array(
        [
            math.comb(2, j) * (xi0 ** j) * ((1.0 - xi0) ** (2 - j))
            for j in range(3)
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_hats_sum_to_one_and_vanish_outside_support():
    n_elements = 6
    x = np.linspace(0.0, 1.0, 61)
    phi = np.asarray(hat_functions(x, n_elements))
    interior = (x > 0.0) & (x < 1.0)
    np.testing.assert_allclose(phi[interior].sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(phi.sum(axis=1), 1.0, atol=1e-12)
    i = 3
    h = 1.0 / n_elements
    left = (i - 1) * h
    right = (i + 1) * h
    outside = (x < left) | (x > right)
    assert np.all(phi[outside, i] == 0.0)
    assert phi[np.argmin(np.abs(x - i * h)), i] == pytest.approx(1.0)


def test_sine_held_out_l2_below_threshold():
    np.random.seed(0)
    x_train = np.linspace(0.0, 1.0, 40)
    y_train = np.sin(2.0 * np.pi * x_train)
    beta = safeguarded_fit(x_train, y_train, n_elements=8)
    x_test = np.linspace(0.0, 1.0, 100)
    knots = np.linspace(0.0, 1.0, 9)
    pred = local_predict(x_test, knots, beta)
    err = _l2(pred, np.sin(2.0 * np.pi * x_test))
    assert err < 0.05


def test_quadratic_in_span_is_recovered():
    np.random.seed(0)
    x_train = np.linspace(0.0, 1.0, 40)
    y_train = x_train ** 2
    beta = safeguarded_fit(x_train, y_train, n_elements=1)
    x_test = np.linspace(0.0, 1.0, 100)
    knots = np.linspace(0.0, 1.0, 2)
    pred = local_predict(x_test, knots, beta)
    err = _l2(pred, x_test ** 2)
    assert err < 1e-6


def test_worse_proposal_is_rejected_and_fit_beats_zero():
    np.random.seed(0)
    x = np.linspace(0.0, 1.0, 30)
    y = np.sin(2.0 * np.pi * x)
    knots = np.linspace(0.0, 1.0, 5)
    beta_prev = np.zeros((5, 3), dtype=np.float64)
    beta_bad = np.full((5, 3), 5.0, dtype=np.float64)
    r_prev = np.linalg.norm(np.asarray(local_predict(x, knots, beta_prev)) - y)
    r_bad = np.linalg.norm(np.asarray(local_predict(x, knots, beta_bad)) - y)
    assert r_bad > r_prev
    chosen = accept_if_improved(beta_prev, beta_bad, x, y, knots)
    assert np.array_equal(np.asarray(chosen), beta_prev)
    beta = safeguarded_fit(x, y, n_elements=4, p=2, delta0=1.0, passes=4)
    r_fit = np.linalg.norm(np.asarray(local_predict(x, knots, beta)) - y)
    assert r_fit <= r_prev


def test_inactive_hats_stay_exactly_zero():
    np.random.seed(0)
    n_elements = 8
    h = 1.0 / n_elements
    x = np.linspace(0.0, h, 12)
    y = np.sin(2.0 * np.pi * x)
    phi = np.asarray(hat_functions(x, n_elements))
    inactive = np.all(phi == 0.0, axis=0)
    assert np.any(inactive)
    beta = np.asarray(safeguarded_fit(x, y, n_elements=n_elements, p=2))
    assert beta.shape == (n_elements + 1, 3)
    assert np.all(beta[inactive] == 0.0)
