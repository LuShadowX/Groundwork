"""Statistical learning theory, made computable.

PAC sample-complexity bounds, a shattering test for VC dimension and empirical
risk minimisation over a small hypothesis class (decision stumps).
"""

from itertools import product

import numpy as np
from scipy.optimize import linprog


def pac_sample_size(h_size, eps, delta, realizable=True):
    """Training examples needed so that, with probability >= 1 - delta, the learned
    hypothesis has true error <= eps (realizable) or within eps of the best (agnostic)."""
    if realizable:
        return int(np.ceil((np.log(h_size) + np.log(1 / delta)) / eps))
    return int(np.ceil((np.log(h_size) + np.log(2 / delta)) / (2 * eps**2)))


def vc_sample_size(d, eps, delta):
    """Blumer et al. (1989) bound for a class of VC dimension d, realizable case."""
    return int(np.ceil(max((4 / eps) * np.log2(2 / delta), (8 * d / eps) * np.log2(13 / eps))))


def linearly_separable(X, y):
    """Is there w, b with y_i (w.x_i + b) >= 1 for all i? Solved as an LP feasibility problem."""
    X = np.asarray(X, dtype=float)
    s = np.where(np.asarray(y) > 0, 1.0, -1.0)
    A = -s[:, None] * np.hstack([X, np.ones((len(X), 1))])
    res = linprog(np.zeros(X.shape[1] + 1), A_ub=A, b_ub=-np.ones(len(X)), bounds=[(None, None)] * (X.shape[1] + 1))
    return res.status == 0


def shattered(X):
    """Can lines in the plane realise every one of the 2^n labelings of these points?"""
    X = np.asarray(X, dtype=float)
    failed = [labels for labels in product([0, 1], repeat=len(X))
              if len(set(labels)) > 1 and not linearly_separable(X, labels)]
    return len(failed) == 0, failed


# ---------- ERM over decision stumps ----------

def stump_predict(x, threshold, sign):
    return (sign * (x - threshold) > 0).astype(int)


def erm_stump(x, y, thresholds=None):
    """Empirical risk minimisation: pick the stump with the lowest training error."""
    thresholds = np.unique(x) if thresholds is None else thresholds
    best = (np.inf, None, None)
    for t in thresholds:
        for sign in (1, -1):
            risk = np.mean(stump_predict(x, t, sign) != y)
            if risk < best[0]:
                best = (risk, t, sign)
    return best
