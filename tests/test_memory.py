"""kNN memory detector and memory insertion (paper Sec. 4.1-4.2), on CPU."""
import numpy as np
import pytest

from cdr.memory import K, Domain, KNNMemoryDetector


def brute_kth(Q, M, k, metric):
    if metric == 'cos':
        Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True); Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
        D = 1 - Qn @ Mn.T
    else:
        D = np.sqrt(((Q[:, None, :] - M[None]) ** 2).sum(-1))
    return np.sort(D, axis=1)[:, k - 1]


@pytest.mark.parametrize('metric', ['l2', 'cos'])
def test_kth_neighbour_distance(metric):
    rng = np.random.default_rng(0); M = rng.normal(size=(300, 8)); Q = rng.normal(size=(50, 8))
    s = KNNMemoryDetector(k=10, metric=metric, device='cpu').fit(M).score(Q)
    assert np.allclose(s, brute_kth(Q, M, 10, metric), atol=1e-4)


def test_insertion_equals_refit_on_concatenated_memory():
    rng = np.random.default_rng(1); S = rng.normal(size=(200, 5)); I = rng.normal(2.0, 1.0, size=(40, 5)); Q = rng.normal(1.0, 1.0, size=(60, 5))
    a = KNNMemoryDetector(device='cpu').fit(S).insert(I).score(Q)
    b = KNNMemoryDetector(device='cpu').fit(np.r_[S, I]).score(Q)
    assert np.allclose(a, b)


def test_insertion_never_increases_scores():
    """Adding memory units can only shrink the k-th neighbour distance."""
    rng = np.random.default_rng(2); S = rng.normal(size=(200, 5)); I = rng.normal(3.0, 1.0, size=(50, 5)); Q = rng.normal(3.0, 1.0, size=(80, 5))
    det = KNNMemoryDetector(device='cpu').fit(S); before = det.score(Q); after = det.insert(I).score(Q)
    assert (after <= before + 1e-9).all() and after.mean() < before.mean()


def test_domain_insert_matches_concatenated_memory():
    """The experiment engine (cached source distances merged with INSERT distances) equals scoring against the
    concatenated memory, for every unit not inserted."""
    rng = np.random.default_rng(3); mem = rng.normal(size=(300, 6)).astype(np.float32)
    E = np.r_[rng.normal(size=(100, 6)), rng.normal(1.5, 1.0, size=(150, 6))].astype(np.float32)
    groups = {'src_cal': np.arange(50), 'src_aux': np.arange(50, 100), 'pool': np.arange(100, 250), 'att': np.arange(0)}
    dom = Domain('t', 'l2', 'none', E, np.arange(len(E)), groups, mem, E, device='cpu')
    ins = np.arange(100, 140)
    got = dom.scores_with_insert(ins)
    want = KNNMemoryDetector(device='cpu').fit(np.r_[mem, E[ins]]).score(E)
    rest = np.setdiff1d(np.arange(len(E)), ins)
    assert np.allclose(got[rest], want[rest], atol=1e-4)
    assert np.allclose(dom.src_score, KNNMemoryDetector(device='cpu').fit(mem).score(E), atol=1e-4)


def test_domain_top3_trace_aggregation():
    """Trace score = mean of the three largest window scores (all windows if fewer than three)."""
    rng = np.random.default_rng(4); d = 16
    mem = rng.normal(size=(200, d)); mem /= np.linalg.norm(mem, axis=1, keepdims=True)
    E = rng.normal(size=(9, d)); E /= np.linalg.norm(E, axis=1, keepdims=True)
    uw = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2])                          # traces with 5, 2, 2 windows
    dom = Domain('t', 'cos', 'top3', E.astype(np.float16), uw, {'pool': [0, 1, 2]}, mem.astype(np.float16), None, device='cpu')
    w = brute_kth(E.astype(np.float16).astype(np.float64), mem.astype(np.float16).astype(np.float64), K, 'cos')
    want = [np.sort(w[uw == u])[-3:].mean() for u in range(3)]
    assert np.allclose(dom.src_score, want, atol=2e-3)
