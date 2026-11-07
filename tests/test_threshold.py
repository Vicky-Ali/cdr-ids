"""Clopper-Pearson threshold and m_min against closed-form values (paper Sec. 4.3)."""
import math

import numpy as np
import pytest

from cdr.threshold import conformal_threshold, cp_rank, cp_threshold, cp_upper, effective_level, empirical_threshold, m_min


def binom_tail(m, p, j):
    """P(Binom(m, p) >= j), exact."""
    return sum(math.comb(m, i) * p ** i * (1 - p) ** (m - i) for i in range(j, m + 1))


@pytest.mark.parametrize('alpha,delta,expected', [(0.01, 0.1, 230), (0.05, 0.1, 45), (0.10, 0.1, 22), (0.05, 0.05, 59),
                                                  (0.10, 0.01, 44)])
def test_m_min_closed_form(alpha, delta, expected):
    assert m_min(alpha, delta) == expected == math.ceil(math.log(delta) / math.log(1 - alpha))
    # (1 - alpha)^m <= delta exactly at m_min and not before
    assert (1 - alpha) ** expected <= delta < (1 - alpha) ** (expected - 1)


@pytest.mark.parametrize('alpha,delta', [(0.01, 0.1), (0.05, 0.1), (0.10, 0.1)])
def test_feasibility_boundary(alpha, delta):
    mm = m_min(alpha, delta)
    assert cp_rank(mm, alpha, delta) == mm              # only the maximum is a valid threshold at m_min
    assert cp_rank(mm - 1, alpha, delta) is None        # infeasible below m_min
    assert cp_threshold(np.arange(mm - 1.0), alpha, delta) == (float('inf'), False)


@pytest.mark.parametrize('m', [22, 45, 50, 100, 250, 500])
@pytest.mark.parametrize('alpha', [0.05, 0.10])
def test_cp_rank_is_smallest_valid_rank(m, alpha):
    delta = 0.1; j = cp_rank(m, alpha, delta)
    if j is None:
        assert binom_tail(m, 1 - alpha, m) > delta; return
    assert binom_tail(m, 1 - alpha, j) <= delta + 1e-12
    if j > 1: assert binom_tail(m, 1 - alpha, j - 1) > delta
    assert effective_level(m, alpha, delta) == j / m >= 1 - alpha


def test_cp_rank_equals_cp_upper_bound_rule():
    """smallest threshold whose one-sided (1-delta) CP upper bound of the CALIB exceedance rate is <= alpha."""
    for m in (50, 100, 300):
        for a in (0.05, 0.1):
            j = cp_rank(m, a, 0.1)
            assert cp_upper(m - j, m, 0.1) <= a + 1e-12
            assert cp_upper(m - j + 1, m, 0.1) > a


def test_cp_upper_closed_form_zero_successes():
    for m in (10, 45, 200):
        assert cp_upper(0, m, 0.1) == pytest.approx(1 - 0.1 ** (1 / m), rel=1e-10)
    assert cp_upper(5, 5, 0.1) == 1.0


def test_cp_threshold_is_order_statistic():
    rng = np.random.default_rng(0); cal = rng.normal(size=100)
    thr, ok = cp_threshold(cal, 0.05, 0.1)
    assert ok and thr == np.sort(cal)[cp_rank(100, 0.05, 0.1) - 1]


def test_conformal_and_empirical_thresholds():
    cal = np.arange(1.0, 101.0)                          # 1..100
    assert conformal_threshold(cal, 0.05) == 96.0        # ceil(101 * 0.95) = 96
    assert conformal_threshold(cal[:10], 0.05) == float('inf')
    assert empirical_threshold(cal, 0.05) == 96.0        # 'higher' quantile at 0.95
