"""Small statistics helpers used across the notebooks."""

import numpy as np


def numerical_grad(f, x, h=1e-5):
    """Central differences. Used to check hand-derived gradients."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e.flat[i] = h
        g.flat[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def bootstrap_ci(values, stat=np.mean, n_boot=2000, level=0.95, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boots = np.array([stat(rng.choice(values, len(values))) for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return stat(values), lo, hi


def mle_normal(x):
    """Closed-form MLE for a normal distribution. Note the 1/n variance, not 1/(n-1)."""
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    return mu, np.sqrt(np.mean((x - mu) ** 2))


def normal_nll(params, x):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    return 0.5 * len(x) * np.log(2 * np.pi * sigma**2) + np.sum((x - mu) ** 2) / (2 * sigma**2)


# ---------- loss functions on a residual or margin ----------

def mse(r):
    return r**2


def mae(r):
    return np.abs(r)


def huber(r, delta=1.0):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))


def log_loss(margin):
    """Binary cross-entropy written as a function of the margin y*f(x)."""
    return np.log1p(np.exp(-margin))


def hinge(margin):
    return np.maximum(0, 1 - margin)
