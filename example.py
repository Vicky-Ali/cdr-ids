"""Minimal example: adapt a kNN-memory intrusion detector to a new deployment without retraining.

1. fit the detector on benign data of the source deployment (its memory of normal behaviour);
2. take a small benign sample of the new deployment and split it into INSERT and CALIB;
3. insert INSERT into the memory (training-free adaptation);
4. set the alarm threshold on CALIB with the one-sided Clopper-Pearson bound, so that
   P(false-positive rate > alpha) <= delta for this deployment.
Synthetic data are used here; replace them with your own feature vectors.
"""
import numpy as np

from cdr import KNNMemoryDetector, cp_threshold, m_min

rng = np.random.default_rng(0)
X_source = rng.normal(0.0, 1.0, (5000, 16))      # benign data of the source deployment (the memory)
X_new = rng.normal(0.5, 1.2, (200, 16))          # small benign sample of the new deployment (normality shift)
X_insert, X_calib = X_new[:100], X_new[100:]      # 50/50 INSERT / CALIB split

alpha, delta = 0.05, 0.10                         # alert budget and allowed failure probability
print('minimum CALIB size m_min =', m_min(alpha, delta))   # 45 for alpha = 0.05, delta = 0.10
assert len(X_calib) >= m_min(alpha, delta)

det = KNNMemoryDetector(k=10, metric='l2').fit(X_source).insert(X_insert)   # memory insertion, no training
thr, feasible = cp_threshold(det.score(X_calib), alpha, delta)              # Clopper-Pearson recalibration

X_test_benign = rng.normal(0.5, 1.2, (2000, 16))  # unseen benign traffic of the new deployment
X_test_attack = rng.normal(3.0, 1.0, (500, 16))   # synthetic "attacks" (far from normal behaviour)
fpr = float((det.score(X_test_benign) > thr).mean())
tpr = float((det.score(X_test_attack) > thr).mean())
print(f'threshold = {thr:.4f}  feasible = {feasible}  FPR = {fpr:.3f} (budget {alpha})  TPR = {tpr:.3f}')
