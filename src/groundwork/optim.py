"""Gradient descent and its variants, written with NumPy only.

Every optimizer has the same interface: step(params, grad) -> new params.
They are kept deliberately small so the update rule is easy to read next to
the formula in the notes.
"""

import numpy as np


class GD:
    name = "Gradient descent"

    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, w, g):
        return w - self.lr * g


class Momentum:
    name = "Momentum"

    def __init__(self, lr=0.1, beta=0.9):
        self.lr, self.beta = lr, beta
        self.v = None

    def step(self, w, g):
        self.v = np.zeros_like(w) if self.v is None else self.v
        self.v = self.beta * self.v + g
        return w - self.lr * self.v


class RMSProp:
    name = "RMSProp"

    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        self.lr, self.beta, self.eps = lr, beta, eps
        self.s = None

    def step(self, w, g):
        self.s = np.zeros_like(w) if self.s is None else self.s
        self.s = self.beta * self.s + (1 - self.beta) * g**2
        return w - self.lr * g / (np.sqrt(self.s) + self.eps)


class Adam:
    name = "Adam"

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, w, g):
        if self.m is None:
            self.m, self.v = np.zeros_like(w), np.zeros_like(w)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g**2
        m_hat = self.m / (1 - self.beta1**self.t)  # bias correction: m and v start at 0
        v_hat = self.v / (1 - self.beta2**self.t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def minimize(grad, x0, optimizer, steps=200):
    """Run an optimizer on a function given by its gradient; returns the path taken."""
    x = np.asarray(x0, dtype=float)
    path = [x.copy()]
    for _ in range(steps):
        x = optimizer.step(x, grad(x))
        path.append(x.copy())
    return np.array(path)


# ---------- test functions ----------

def rosenbrock(p, a=1.0, b=100.0):
    x, y = p[..., 0], p[..., 1]
    return (a - x) ** 2 + b * (y - x**2) ** 2


def rosenbrock_grad(p, a=1.0, b=100.0):
    x, y = p
    return np.array([-2 * (a - x) - 4 * b * x * (y - x**2), 2 * b * (y - x**2)])


def saddle(p):
    return p[..., 0] ** 2 - p[..., 1] ** 2


def saddle_grad(p):
    return np.array([2 * p[0], -2 * p[1]])


# ---------- softmax regression trained with any optimizer ----------

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # stable: exp of large numbers would overflow
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(W, X, Y):
    """Mean cross-entropy loss and its gradient for softmax regression.
    X has a bias column, Y is one-hot."""
    P = softmax(X @ W)
    loss = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))
    grad = X.T @ (P - Y) / len(X)
    return loss, grad


def train_softmax(X, y, optimizer, epochs=30, batch_size=None, seed=0):
    """batch_size=None is full-batch GD, 1 is SGD, anything else is mini-batch."""
    rng = np.random.default_rng(seed)
    X = np.hstack([np.ones((len(X), 1)), X])
    k = int(y.max()) + 1
    Y = np.eye(k)[y]
    W = np.zeros((X.shape[1], k))
    bs = len(X) if batch_size is None else batch_size
    history = [cross_entropy(W, X, Y)[0]]
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), bs):
            idx = order[start:start + bs]
            _, g = cross_entropy(W, X[idx], Y[idx])
            W = optimizer.step(W, g)
        history.append(cross_entropy(W, X, Y)[0])
    return W, np.array(history)


def predict_softmax(W, X):
    X = np.hstack([np.ones((len(X), 1)), X])
    return softmax(X @ W).argmax(axis=1)
