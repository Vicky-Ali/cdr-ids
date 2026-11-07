"""One deployment run on a synthetic shifted domain (CPU): all methods and alphas, feasibility, budget of M4."""
import numpy as np

from cdr.memory import Domain
from cdr.protocol import ALPHAS, METHODS, REF, run_one


def synthetic_domain(seed=0):
    rng = np.random.default_rng(seed); d = 4
    mem = rng.normal(size=(400, d)); src = rng.normal(size=(200, d))           # source benign: memory, cal + aux
    tgt = rng.normal(0.8, 1.0, size=(800, d)); att = rng.normal(3.0, 1.0, size=(100, d))
    E = np.r_[src, tgt, att].astype(np.float32)
    groups = {'src_cal': np.arange(100), 'src_aux': np.arange(100, 200), 'pool': np.arange(200, 1000), 'att': np.arange(1000, 1100)}
    return Domain('synthetic', 'l2', 'none', E, np.arange(len(E)), groups, mem.astype(np.float32), E, device='cpu')


def test_run_one_rows_and_feasibility():
    dom = synthetic_domain(); rows = run_one(dom, 0, 200, 100, 'syn')
    assert len(rows) == len(ALPHAS) * (len(METHODS) + 1)
    assert {r['method'] for r in rows} == set(METHODS) | {REF}
    m4 = {r['alpha']: r for r in rows if r['method'] == 'M4 insert+CP'}
    assert m4[0.01]['feasible'] == 0.0 and m4[0.01]['tpr'] == 0.0     # m = 100 < m_min(0.01) = 230: no alarms
    assert m4[0.05]['feasible'] == 1.0 and m4[0.10]['feasible'] == 1.0
    for r in rows: assert r['exceed'] == int(r['fpr'] > r['alpha'])


def test_m4_holds_budget_and_m0_breaks_under_shift():
    dom = synthetic_domain(1); ex = {'M0 frozen': [], 'M4 insert+CP': []}
    for s in range(20):
        for r in run_one(dom, s, 200, 100, 'syn', alphas=(0.05,), methods=['M0 frozen', 'M4 insert+CP'], with_ref=False):
            ex[r['method']].append(r['exceed'])
    assert np.mean(ex['M0 frozen']) > 0.5            # the frozen source threshold breaks under the benign shift
    assert np.mean(ex['M4 insert+CP']) <= 0.3         # guarantee 0.1 + finite-test-set noise
