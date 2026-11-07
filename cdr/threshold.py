"""Threshold rules (paper Sec. 4.3: CP recalibration; Sec. 5.2: baselines M0, M1, M5, M6). Convention: alarm iff score > threshold.

Clopper-Pearson (CP) threshold. Calibration scores s_(1) <= ... <= s_(m) of target benign data. Using t = s_(j),
the true false-alarm rate is FPR(t) = P(S > s_(j)). FPR(t) > alpha  <=>  s_(j) < q_{1-alpha}  <=>  at least j of the m
calibration scores fall below q_{1-alpha}, a Binomial(m, 1 - alpha) event (continuous scores). Hence
    P(FPR(s_(j)) > alpha) = P(Binom(m, 1 - alpha) >= j)
and we take the smallest j with P(Binom(m, 1 - alpha) >= j) <= delta. This equals "the smallest threshold whose
one-sided (1 - delta) Clopper-Pearson upper bound of the calibration exceedance rate is <= alpha" (x = m - j exceedances).
Feasible iff j <= m, i.e. (1 - alpha)^m <= delta, i.e. m >= m_min = ceil(ln delta / ln(1 - alpha)).
With ties (discrete scores) alarm-iff-> keeps the bound valid (conservative).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def m_min(alpha: float, delta: float) -> int:
    """Smallest calibration size for which a CP threshold exists: ceil(ln delta / ln(1 - alpha))."""
    return int(math.ceil(math.log(delta) / math.log(1 - alpha) - 1e-12))


def cp_rank(m: int, alpha: float, delta: float) -> int | None:
    """Smallest 1-based rank j with P(Binom(m, 1-alpha) >= j) <= delta; None if infeasible."""
    if m <= 0:
        return None
    j = np.arange(1, m + 1)
    tail = stats.binom.sf(j - 1, m, 1 - alpha)  # P(X >= j)
    ok = np.flatnonzero(tail <= delta)
    return int(j[ok[0]]) if len(ok) else None


def cp_upper(x: int, m: int, delta: float) -> float:
    """One-sided (1-delta) Clopper-Pearson upper bound of a binomial rate with x successes out of m."""
    return 1.0 if x >= m else float(stats.beta.ppf(1 - delta, x + 1, m - x))


def effective_level(m: int, alpha: float, delta: float) -> float | None:
    """Empirical quantile level used by the CP threshold, j/m (None if infeasible)."""
    j = cp_rank(m, alpha, delta)
    return None if j is None else j / m


def cp_threshold(cal, alpha: float, delta: float) -> tuple[float, bool]:
    """CP threshold on calibration scores `cal`: (threshold, feasible). Infeasible -> (inf, False): no alarms."""
    cal = np.sort(np.asarray(cal, float)); j = cp_rank(len(cal), alpha, delta)
    return (float('inf'), False) if j is None else (float(cal[j - 1]), True)


def conformal_threshold(cal, alpha: float) -> float:
    """Marginal split-conformal: the ceil((n+1)(1-alpha))-th smallest calibration score (inf if > n)."""
    cal = np.sort(np.asarray(cal, float)); n = len(cal); k = int(math.ceil((n + 1) * (1 - alpha)))
    return float('inf') if k > n else float(cal[k - 1])


def empirical_threshold(cal, alpha: float) -> float:
    """Plain empirical (1-alpha) quantile ('higher' interpolation)."""
    return float(np.quantile(np.asarray(cal, float), 1 - alpha, method='higher'))


def weighted_conformal_thresholds(cal, w_cal, w_test, alpha: float) -> np.ndarray:
    """Tibshirani et al. (2019) weighted split conformal: per test point, the (1-alpha) quantile of
    sum_i p_i delta_{s_i} + p_test delta_{+inf}, p proportional to the density-ratio weights."""
    cal = np.asarray(cal, float); w_cal = np.asarray(w_cal, float); o = np.argsort(cal)
    s, cw = cal[o], np.cumsum(w_cal[o]); S = cw[-1]
    need = (1 - alpha) * (S + np.asarray(w_test, float))
    idx = np.searchsorted(cw, need - 1e-12 * (S + 1), side='left')
    thr = np.full(len(need), np.inf); ok = idx < len(s); thr[ok] = s[idx[ok]]
    return thr
