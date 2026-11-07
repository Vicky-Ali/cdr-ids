"""Evaluation metrics (paper Sec. 5.4): FPR / TPR / precision / F1 at a threshold, AUROC, exceedance
P(test FPR > alpha) with its Clopper-Pearson CI, and the finite-test-set noise floor of the exceedance."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def evaluate(sb, sa, thr) -> tuple[float, float, float, float]:
    """(FPR, TPR, precision, F1) of the rule 'alarm iff score > thr' on benign scores sb and attack scores sa."""
    pb = sb > thr; pa = sa > thr
    fp, tp = int(pb.sum()), int(pa.sum()); fpr = fp / len(sb); tpr = tp / len(sa)
    prec = tp / (tp + fp) if tp + fp else 0.0; f1 = 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0
    return fpr, tpr, prec, f1


def auroc(sb, sa) -> float:
    """AUROC of attack (positive) vs benign scores."""
    return float(roc_auc_score(np.r_[np.zeros(len(sb)), np.ones(len(sa))], np.r_[sb, sa]))


def exceeds(fpr: float, alpha: float) -> int:
    """Budget exceedance flag of one deployment: 1 iff the realised test FPR is above alpha."""
    return int(fpr > alpha)


def binom_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Two-sided Clopper-Pearson CI of a proportion."""
    if n == 0:
        return (float('nan'), float('nan'))
    lo = 0.0 if k == 0 else stats.beta.ppf((1 - conf) / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - (1 - conf) / 2, k + 1, n - k)
    return float(lo), float(hi)


def noise_floor(n_test: int, alpha: float) -> float:
    """P(empirical test FPR > alpha) for a threshold whose TRUE FPR is exactly alpha (finite-test-set noise)."""
    return float(stats.binom.sf(math.floor(alpha * n_test), n_test, alpha))
