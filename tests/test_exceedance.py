"""Monte Carlo: with exchangeable (i.i.d.) calibration and test scores, the CP threshold exceeds the alert budget
alpha with probability <= delta (paper Sec. 4.3). Fast (a few seconds)."""
import numpy as np
import pytest
from scipy import stats

from cdr.threshold import cp_threshold, empirical_threshold


def mc_exceedance(m, alpha, delta, reps, rng, dist='uniform'):
    """Fraction of deployments whose TRUE FPR P(S > thr) exceeds alpha."""
    fails = 0
    for _ in range(reps):
        if dist == 'uniform':
            cal = rng.random(m); thr, ok = cp_threshold(cal, alpha, delta); fpr = 1 - min(thr, 1.0)
        else:  # exponential scores
            cal = rng.exponential(size=m); thr, ok = cp_threshold(cal, alpha, delta); fpr = float(np.exp(-thr))
        fails += fpr > alpha
    return fails / reps


@pytest.mark.parametrize('m,alpha', [(45, 0.05), (100, 0.05), (100, 0.10), (500, 0.05)])
@pytest.mark.parametrize('dist', ['uniform', 'exponential'])
def test_cp_exceedance_below_delta(m, alpha, dist):
    rng = np.random.default_rng(m + int(100 * alpha)); reps = 4000; delta = 0.1
    rate = mc_exceedance(m, alpha, delta, reps, rng, dist)
    # one-sided binomial test at level 1e-4 against the guarantee delta
    assert stats.binomtest(int(rate * reps), reps, delta, alternative='greater').pvalue > 1e-4, rate


def test_plain_quantile_does_not_hold_the_budget():
    """Sanity check of the test itself: the empirical (1-alpha) quantile exceeds alpha in about half the runs."""
    rng = np.random.default_rng(1); fails = 0
    for _ in range(2000):
        fails += (1 - empirical_threshold(rng.random(100), 0.05)) > 0.05
    assert fails / 2000 > 0.3


def test_ties_are_conservative():
    """Discrete scores: alarm iff score > thr keeps the realised FPR at or below the continuous case."""
    rng = np.random.default_rng(2); fails = 0; reps = 3000
    for _ in range(reps):
        cal = rng.integers(0, 20, 200); thr, _ = cp_threshold(cal, 0.10, 0.1)
        fails += (19 - thr) / 20 > 0.10                      # true P(S > thr) for S uniform on {0..19}
    assert fails / reps <= 0.1 + 3 * np.sqrt(0.09 / reps)
