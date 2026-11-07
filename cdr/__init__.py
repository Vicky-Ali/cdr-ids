"""cdr -- Calibrate, Don't Retrain: training-free adaptation of a kNN-memory intrusion detector to a new benign
environment by memory insertion + Clopper-Pearson recalibration, with a per-deployment alert-budget guarantee
P(FPR > alpha) <= delta (paper Sec. 4).

Quick use (see README.md):
    from cdr import KNNMemoryDetector, cp_threshold, m_min
    det = KNNMemoryDetector(k=10, metric='l2').fit(X_source).insert(X_insert)
    thr, feasible = cp_threshold(det.score(X_calib), alpha=0.05, delta=0.10)
    alarms = det.score(X_new) > thr
"""
from cdr.memory import Domain, KNNMemoryDetector
from cdr.metrics import auroc, binom_ci, evaluate, noise_floor
from cdr.threshold import (conformal_threshold, cp_rank, cp_threshold, cp_upper, effective_level, empirical_threshold,
                           m_min, weighted_conformal_thresholds)

__version__ = '1.0.0'
__all__ = ['Domain', 'KNNMemoryDetector', 'auroc', 'binom_ci', 'evaluate', 'noise_floor', 'conformal_threshold', 'cp_rank',
           'cp_threshold', 'cp_upper', 'effective_level', 'empirical_threshold', 'm_min', 'weighted_conformal_thresholds']
