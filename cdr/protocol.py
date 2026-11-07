"""One deployment run and the compared methods M0-M6 (+ the REF oracle threshold) (paper Sec. 5.2-5.3).

A run = (domain, seed, n_new, m_cal): draw the new-environment benign sample, split it into CALIB (m) and INSERT
(n - m), score, threshold every method at every alpha, evaluate on the remaining target benign + all attacks.

Methods:
  M0 frozen               source memory, empirical (1-alpha) quantile of source calibration scores
  M1 src-conformal        source memory, split-conformal threshold on source calibration scores
  M2 recal-CP             source memory, Clopper-Pearson threshold on target CALIB        (recalibration only)
  M3 insert               memory + INSERT, empirical quantile of source calibration scores (insertion only)
  M4 insert+CP            memory + INSERT, Clopper-Pearson threshold on target CALIB      (ours)
  M5 insert+conformal     memory + INSERT, marginal split-conformal threshold on target CALIB
  M6 weighted-conformal   source memory, weighted conformal with a source-vs-target logistic density ratio
  REF oracle-thr          memory + INSERT, (1-alpha) quantile of the TEST benign scores (reference only)
"""
from __future__ import annotations

import zlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .metrics import auroc, evaluate
from .threshold import conformal_threshold, cp_threshold, empirical_threshold, weighted_conformal_thresholds

ALPHAS = (0.01, 0.05, 0.10)
DELTA = 0.10
METHODS = ['M0 frozen', 'M1 src-conformal', 'M2 recal-CP', 'M3 insert', 'M4 insert+CP', 'M5 insert+conformal',
           'M6 weighted-conformal']
REF = 'REF oracle-thr'   # threshold = (1-alpha) quantile of the TEST benign scores under M4's memory; reference only
INSERT_METHODS = ('M3 insert', 'M4 insert+CP', 'M5 insert+conformal', REF)


def draw(dom, seed: int, n: int, m: int, tag: str):
    """Seeded draw of the new benign sample (n units of the target pool) -> (calib, insert, test benign).
    For fixed (seed, n) the new sample and the test set do not depend on m, and CALIB/INSERT are nested across m."""
    pool = dom.groups['pool']
    perm = np.random.default_rng([seed, zlib.crc32(tag.encode()), 7]).permutation(pool)
    new, test_b = perm[:n], perm[n:]
    p2 = np.random.default_rng([seed, n, zlib.crc32(tag.encode()), 11]).permutation(new)
    return p2[:m], p2[m:], test_b      # calib, insert, test benign (calib/insert nested across m for fixed seed,n)


def m6_weights(dom, new, units, seed: int) -> np.ndarray:
    """Density-ratio weights p/(1-p) of a source-vs-new logistic classifier (M6)."""
    X = np.r_[dom.unit_feat[dom.groups['src_aux']], dom.unit_feat[new]]
    y = np.r_[np.zeros(len(dom.groups['src_aux'])), np.ones(len(new))]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=seed).fit(sc.transform(X), y)
    p = clf.predict_proba(sc.transform(dom.unit_feat[units]))[:, 1]
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return p / (1 - p)


def run_one(dom, seed, n, m, tag, alphas=ALPHAS, delta=DELTA, methods=METHODS, with_ref=True) -> list[dict]:
    """All methods x alphas for one (seed, n_new, m_cal) draw; one result dict per (alpha, method)."""
    cal, ins, test_b = draw(dom, seed, n, m, tag)
    att = dom.groups['att']; scal = dom.groups['src_cal']
    s_src = dom.src_score; s_ins = dom.scores_with_insert(ins)
    au_src, au_ins = auroc(s_src[test_b], s_src[att]), auroc(s_ins[test_b], s_ins[att])
    rows = []
    w = None
    if 'M6 weighted-conformal' in methods:
        w = m6_weights(dom, np.r_[cal, ins], np.r_[scal, test_b, att], seed)
        w_cal, w_tb, w_att = w[:len(scal)], w[len(scal):len(scal) + len(test_b)], w[len(scal) + len(test_b):]
    for a in alphas:
        for meth in methods + ([REF] if with_ref else []):
            feas = 1; S = s_src
            if meth == 'M0 frozen': thr = empirical_threshold(s_src[scal], a)
            elif meth == 'M1 src-conformal': thr = conformal_threshold(s_src[scal], a)
            elif meth == 'M2 recal-CP': thr, feas = cp_threshold(s_src[cal], a, delta)
            elif meth == 'M3 insert': S = s_ins; thr = empirical_threshold(s_ins[scal], a)
            elif meth == 'M4 insert+CP': S = s_ins; thr, feas = cp_threshold(s_ins[cal], a, delta)
            elif meth == 'M5 insert+conformal': S = s_ins; thr = conformal_threshold(s_ins[cal], a); feas = int(np.isfinite(thr))
            elif meth == REF: S = s_ins; thr = float(np.quantile(s_ins[test_b], 1 - a, method='higher'))
            if meth == 'M6 weighted-conformal':
                tb = weighted_conformal_thresholds(s_src[scal], w_cal, w_tb, a)
                ta = weighted_conformal_thresholds(s_src[scal], w_cal, w_att, a)
                fpr, tpr, prec, f1 = evaluate(s_src[test_b] - tb, s_src[att] - ta, 0.0)
                thr = float(np.median(tb)); feas = float(np.isfinite(tb).mean())
            else:
                fpr, tpr, prec, f1 = evaluate(S[test_b], S[att], thr)
            rows.append(dict(seed=seed, n_new=n, m_cal=m, n_insert=n - m, alpha=a, method=meth, thr=thr, feasible=float(feas),
                             fpr=fpr, exceed=int(fpr > a), tpr=tpr, precision=prec, f1=f1,
                             auroc=au_ins if S is s_ins else au_src, n_test_benign=len(test_b), n_att=len(att)))
    return rows
