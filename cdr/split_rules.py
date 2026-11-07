"""INSERT/CALIB split rules (paper Sec. 4.4 and 6.5; Fig. 3, Tables S2-S7).

n new benign units are split into m CALIB units (CP threshold) and n - m INSERT units (memory insertion).
  RULE(eps)  data-free: the smallest m >= m_min whose CP effective quantile level j(m)/m is within eps of 1 - alpha,
             capped at n - MIN_INSERT. Variants: eps = 0.01 (specified), 0.02 and alpha/2 (pre-registered).
  RULE-SA    shift-aware (pre-registered): KS distance D between the source-memory scores of the n new units and the
             source-calibration scores; D <= TAU -> RULE(eps = 0.01), else the 50/50 split.
  m_grid     the CALIB sizes evaluated in the split experiment for a given n.
"""
from __future__ import annotations

from .threshold import effective_level, m_min

DELTA = 0.10
SPLIT_ALPHAS = (0.05, 0.10)
MIN_INSERT = 10
TAU = 0.2
RULES = {'rule-eps0.01': lambda a: 0.01, 'rule-eps0.02': lambda a: 0.02, 'rule-eps-alpha/2': lambda a: a / 2}


def rule_m(n: int, alpha: float, eps: float, delta: float = DELTA) -> tuple[int | None, bool]:
    """Smallest m >= m_min whose effective quantile level j(m)/m is within eps of 1-alpha; capped at n - MIN_INSERT.
    Returns (m, met) where met = whether the eps criterion is met within the cap; None if n - MIN_INSERT < m_min."""
    mm = m_min(alpha, delta); cap = n - MIN_INSERT
    if cap < mm: return None, False
    for m in range(mm, cap + 1):
        if effective_level(m, alpha, delta) - (1 - alpha) <= eps + 1e-12: return m, True
    return cap, False


def m_grid(n: int) -> list[int]:
    """CALIB sizes of the split experiment: {m_min(0.05), m_min(0.10), n/4, n/2, 3n/4, n-10} + every RULE m."""
    ms = {m_min(0.05, DELTA), m_min(0.10, DELTA), int(round(.25 * n)), int(round(.5 * n)), int(round(.75 * n)), n - MIN_INSERT}
    for a in SPLIT_ALPHAS:
        for f in RULES.values():
            m, _ = rule_m(n, a, f(a))
            if m: ms.add(m)
    return sorted(m for m in ms if 1 <= m <= n - MIN_INSERT)


def shift_aware_m(n: int, alpha: float, ks: float) -> tuple[int, str]:
    """RULE-SA: (m, choice) with choice 'rule-eps0.01' if ks <= TAU else 'half'."""
    if ks <= TAU:
        return rule_m(n, alpha, 0.01)[0], 'rule-eps0.01'
    return int(round(.5 * n)), 'half'
