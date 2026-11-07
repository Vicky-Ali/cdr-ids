# Calibrate, Don't Retrain (cdr-ids)

Core implementation of the method in the paper **"Calibrate, Don't Retrain: Training-Free Adaptation with Guaranteed
Alert Budgets for Intrusion Detection under Normality Shift"** (Journal of Information Security and Applications,
under review).

When an anomaly-based intrusion detector moves to a new deployment, benign behaviour changes and the alert budget
(the share of benign traffic that raises alerts) silently breaks. For nearest-neighbour detectors we adapt without
any training. A small benign sample of the new deployment is split in two:

- the **INSERT** part is added to the detector's memory of normal behaviour (memory insertion);
- the **CALIB** part sets the alarm threshold through a one-sided **Clopper-Pearson** bound.

Then, per deployment, P(false-positive rate > alpha) <= delta, with no attack data needed. A threshold exists only if
the calibration sample has at least m_min = ceil(ln delta / ln(1 - alpha)) units (45 for alpha = 0.05, delta = 0.1).

## Files

| File | Content |
|---|---|
| `cdr/memory.py` | training-free kNN memory detector (k = 10) and memory insertion |
| `cdr/threshold.py` | Clopper-Pearson threshold, `m_min`, and the conformal / empirical / weighted-conformal baselines |
| `cdr/split_rules.py` | rules for splitting the new benign sample into INSERT and CALIB |
| `cdr/protocol.py` | one deployment run with the compared methods M0-M6 (M4 = the proposed method) |
| `cdr/metrics.py` | FPR, TPR, F1, AUROC and exceedance with Clopper-Pearson confidence intervals |
| `example.py` | a minimal, runnable example on synthetic data |
| `tests/` | unit tests (pytest) |

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.12 (tested: 3.12.3)
pip install -r requirements.txt
python example.py       # prints m_min, the threshold, the realised FPR and TPR
python -m pytest        # 47 tests, a few seconds
```

A GPU is optional (the kNN search uses PyTorch and falls back to the CPU).

## Usage

```python
from cdr import KNNMemoryDetector, cp_threshold, m_min

det = KNNMemoryDetector(k=10, metric='l2').fit(X_source).insert(X_insert)   # memory insertion, no training
thr, feasible = cp_threshold(det.score(X_calib), alpha=0.05, delta=0.10)    # Clopper-Pearson recalibration
alarms = det.score(X_new) > thr                                              # P(FPR > alpha) <= delta
```

`cp_threshold` works on the scores of any detector (higher = more anomalous), so the recalibration step can also be
used with other scorers; memory insertion is specific to memory-based (kNN) detectors.

## Datasets used in the paper

No data are included. The paper evaluates the method on 5G-NIDD (base-station shift,
<https://ieee-dataport.org/node/10203>), CSE-CIC-IDS2018 (corrected release,
<https://intrusion-detection.distrinet-research.be/CNS2022/>) and CAShift (<https://github.com/fish98/CAShift>).

## Citation

```bibtex
@article{ali2026calibrate,
  title   = {Calibrate, Don't Retrain: Training-Free Adaptation with Guaranteed Alert Budgets for Intrusion
             Detection under Normality Shift},
  author  = {Ali, Syed Waqas and others},
  journal = {Journal of Information Security and Applications},
  note    = {Under review},
  year    = {2026}
}
```

## Licence and contact

MIT licence (`LICENSE`). Corresponding author: Syed Waqas Ali (University of Azad Jammu and Kashmir),
e-mail: waqas.ali@uajk.edu.pk.
