"""INSERT/CALIB split rules (paper Sec. 4.4, 6.5) and the nested draws of the protocol."""
import numpy as np
import pytest

from cdr.protocol import draw
from cdr.split_rules import MIN_INSERT, TAU, m_grid, rule_m, shift_aware_m
from cdr.threshold import effective_level, m_min


def test_rule_eps001_closed_values():
    # documented in the paper: eps = 0.01 needs m >= 873 (alpha 0.05) and 1,577 (alpha 0.10) at delta = 0.1
    assert rule_m(1000, 0.05, 0.01) == (873, True)
    assert rule_m(2000, 0.10, 0.01) == (1577, True)
    assert rule_m(1000, 0.10, 0.01) == (1000 - MIN_INSERT, False)     # criterion not met -> cap n - 10


def test_rule_is_smallest_m_meeting_eps():
    for n, a, eps in [(500, 0.05, 0.025), (300, 0.10, 0.05), (1000, 0.05, 0.02)]:
        m, met = rule_m(n, a, eps)
        assert met and effective_level(m, a, 0.1) - (1 - a) <= eps + 1e-12
        assert all(effective_level(x, a, 0.1) - (1 - a) > eps + 1e-12 for x in range(m_min(a, 0.1), m))


def test_rule_impossible_when_n_too_small():
    assert rule_m(m_min(0.05, 0.1) + MIN_INSERT - 1, 0.05, 0.01) == (None, False)


def test_shift_aware_rule():
    assert shift_aware_m(1000, 0.05, TAU) == (873, 'rule-eps0.01')
    assert shift_aware_m(1000, 0.05, TAU + 1e-6) == (500, 'half')


class _Dom:
    groups = {'pool': np.arange(1000, 2000)}


def test_draws_are_nested_across_m():
    """For fixed (seed, n): same new sample and test set for every m; CALIB(m) and INSERT(m) are nested."""
    c1, i1, t1 = draw(_Dom, 3, 200, 50, 'x'); c2, i2, t2 = draw(_Dom, 3, 200, 120, 'x')
    assert np.array_equal(t1, t2) and set(c1) | set(i1) == set(c2) | set(i2)
    assert set(c1) <= set(c2) and set(i2) <= set(i1) and not set(c1) & set(i1)
