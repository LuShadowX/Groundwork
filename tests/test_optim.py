import numpy as np
import pytest

from groundwork import optim


def quadratic_grad(p):
    return 2 * (p - np.array([3.0, -2.0]))  # minimum at (3, -2)


@pytest.mark.parametrize("opt", [optim.GD(0.1), optim.Momentum(0.05), optim.RMSProp(0.05), optim.Adam(0.1)])
def test_every_optimizer_finds_the_minimum_of_a_bowl(opt):
    path = optim.minimize(quadratic_grad, [0.0, 0.0], opt, steps=500)
    assert np.allclose(path[-1], [3, -2], atol=1e-2)


def test_gd_diverges_when_the_learning_rate_is_too_big():
    path = optim.minimize(quadratic_grad, [0.0, 0.0], optim.GD(1.1), steps=50)
    assert np.linalg.norm(path[-1]) > 1e3


def test_rosenbrock_gradient_matches_finite_differences():
    from groundwork.stats import numerical_grad
    p = np.array([-1.2, 1.0])
    assert np.allclose(optim.rosenbrock_grad(p), numerical_grad(lambda q: optim.rosenbrock(q), p), rtol=1e-5)


def test_adam_gets_closer_to_rosenbrock_minimum_than_plain_gd():
    gd = optim.minimize(optim.rosenbrock_grad, [-1.5, 2.0], optim.GD(1e-4), steps=3000)[-1]
    adam = optim.minimize(optim.rosenbrock_grad, [-1.5, 2.0], optim.Adam(0.02), steps=3000)[-1]
    assert np.linalg.norm(adam - 1) < np.linalg.norm(gd - 1)


def test_softmax_gradient_is_correct():
    from groundwork.stats import numerical_grad
    rng = np.random.default_rng(0)
    X = np.hstack([np.ones((20, 1)), rng.normal(size=(20, 3))])
    Y = np.eye(3)[rng.integers(0, 3, 20)]
    W = rng.normal(size=(4, 3))
    _, g = optim.cross_entropy(W, X, Y)
    assert np.allclose(g, numerical_grad(lambda w: optim.cross_entropy(w, X, Y)[0], W), atol=1e-6)


def test_softmax_regression_learns_separable_blobs():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(c, 0.3, size=(50, 2)) for c in [(0, 0), (3, 0), (0, 3)]])
    y = np.repeat([0, 1, 2], 50)
    W, hist = optim.train_softmax(X, y, optim.Adam(0.1), epochs=100)
    assert hist[-1] < hist[0]
    assert (optim.predict_softmax(W, X) == y).mean() > 0.97
