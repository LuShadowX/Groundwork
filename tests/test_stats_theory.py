import numpy as np

from groundwork import stats, theory


def test_mle_normal_matches_numpy_population_formulas():
    x = np.random.default_rng(0).normal(5, 2, 10_000)
    mu, sigma = stats.mle_normal(x)
    assert np.isclose(mu, x.mean()) and np.isclose(sigma, x.std(ddof=0))


def test_mle_minimises_the_negative_log_likelihood():
    x = np.random.default_rng(1).normal(-1, 0.5, 500)
    best = stats.normal_nll(stats.mle_normal(x), x)
    for mu, s in [(-0.9, 0.5), (-1.0, 0.6), (-1.1, 0.45)]:
        assert stats.normal_nll((mu, s), x) > best


def test_bootstrap_interval_contains_the_true_mean_most_of_the_time():
    rng = np.random.default_rng(2)
    hits = 0
    for seed in range(60):
        sample = rng.normal(10, 3, 200)
        _, lo, hi = stats.bootstrap_ci(sample, n_boot=400, seed=seed)
        hits += lo <= 10 <= hi
    assert hits >= 50  # 95% nominal, allow some slack for 60 trials


def test_huber_is_quadratic_inside_and_linear_outside():
    assert np.isclose(stats.huber(0.5), 0.125)
    assert np.isclose(stats.huber(3.0), 2.5)


def test_pac_bounds_grow_with_hypotheses_and_shrink_with_eps():
    assert theory.pac_sample_size(1000, 0.05, 0.05) > theory.pac_sample_size(10, 0.05, 0.05)
    assert theory.pac_sample_size(100, 0.01, 0.05) > theory.pac_sample_size(100, 0.1, 0.05)
    assert theory.pac_sample_size(100, 0.05, 0.05, realizable=False) > theory.pac_sample_size(100, 0.05, 0.05)


def test_lines_shatter_three_points_but_not_four():
    ok3, _ = theory.shattered([[0, 0], [1, 0], [0, 1]])
    ok4, failed = theory.shattered([[0, 0], [1, 0], [0, 1], [1, 1]])
    assert ok3 and not ok4
    assert (0, 1, 1, 0) in failed  # the XOR labelling


def test_erm_stump_finds_a_perfect_threshold():
    x = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([0, 0, 0, 1, 1, 1])
    risk, t, sign = theory.erm_stump(x, y)
    assert risk == 0 and 3 <= t < 4 and sign == 1
