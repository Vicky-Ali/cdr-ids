"""kNN memory detector and memory insertion (paper Sec. 4.1-4.2).

Score of a unit = distance to its k-th nearest neighbour (k = 10) in the benign memory; for trace data (CAShift) a
trace is a sequence of windows and its score is the mean of its three largest window scores ('top3').
Insertion = appending new benign target units to the memory; no parameter is trained.

Two classes:
  KNNMemoryDetector  small numpy-in / numpy-out API (fit, insert, score) for any feature vectors.
  Domain             the experiment engine used for every reported number: it caches the top-k source-memory
                     distances of all query windows once, so that scoring with memory = source + INSERT only needs
                     the distances to the INSERT windows (exactly equal to scoring against the concatenated memory).
"""
from __future__ import annotations

import numpy as np
import torch

K = 10


def default_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class KNNMemoryDetector:
    """k-th-nearest-neighbour distance to a benign memory.

    metric 'l2' (Euclidean; used for flow features) or 'cos' (1 - cosine similarity; rows are L2-normalised here,
    as the window embeddings used in the paper already are).
    """

    def __init__(self, k: int = K, metric: str = 'l2', device: str | None = None):
        self.k, self.metric, self.device = k, metric, device or default_device()
        self.M = None

    def _t(self, X) -> torch.Tensor:
        X = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=self.device)
        return torch.nn.functional.normalize(X, dim=1) if self.metric == 'cos' else X

    def fit(self, X_source) -> 'KNNMemoryDetector':
        """Memory = the source benign units."""
        self.M = self._t(X_source); return self

    def insert(self, X_new) -> 'KNNMemoryDetector':
        """Memory insertion: append new benign target units (no retraining)."""
        self.M = torch.cat([self.M, self._t(X_new)]); return self

    def score(self, X, bs: int = 8192) -> np.ndarray:
        """Distance of every row of X to its k-th nearest memory unit (higher = more anomalous)."""
        Q, out, kk = self._t(X), [], min(self.k, self.M.shape[0])
        for s in range(0, len(Q), bs):
            q = Q[s:s + bs]
            if self.metric == 'cos':
                out.append(1 - torch.topk(q @ self.M.T, kk, dim=1).values[:, kk - 1])
            else:
                d2 = (q * q).sum(1, keepdim=True) + (self.M * self.M).sum(1)[None] - 2 * q @ self.M.T
                out.append(torch.topk(d2.clamp_min(0), kk, dim=1, largest=False).values[:, kk - 1].sqrt())
        return torch.cat(out).cpu().numpy().astype(np.float64)


class Domain:
    """Everything a run needs for one (dataset, view, mode, target):
    windows (embedding rows) grouped into units (D1: traces, D2/D3: flows), source-memory top-k distances per window,
    and a scorer that merges an INSERT memory into the kNN and aggregates windows to unit scores.
    Groups: 'src_cal' (source calibration), 'src_aux' (M6 classifier source sample), 'pool' (target benign), 'att'."""

    def __init__(self, name, metric, agg, E, unit_of_win, groups, memory, unit_feat, device: str | None = None):
        """E: (n_win, d) float16/32 numpy of query windows; unit_of_win: unit id per window (units numbered 0..U-1,
        windows of a unit contiguous); groups: name -> array of unit ids; memory: source memory windows (numpy);
        unit_feat: (U, f) numpy features for the M6 density-ratio classifier; agg: 'none' (unit = window) or 'top3'."""
        self.name, self.metric, self.agg = name, metric, agg
        self.device = dev = device or default_device()
        dt = torch.float16 if metric == 'cos' else torch.float32
        self.E = torch.from_numpy(np.ascontiguousarray(E)).to(dev, dt)
        self.uw = np.asarray(unit_of_win); self.U = int(self.uw.max()) + 1
        self.groups = {k: np.asarray(v) for k, v in groups.items()}
        self.unit_feat = unit_feat
        starts = np.r_[0, np.flatnonzero(np.diff(self.uw)) + 1]; ends = np.r_[starts[1:], len(self.uw)]
        assert len(starts) == self.U and (self.uw[starts] == np.arange(self.U)).all()
        self.ws, self.we = starts, ends
        cnt = ends - starts; self.cnt = torch.tensor(cnt, device=dev)
        if agg == 'top3':
            maxw = int(cnt.max()); col = np.arange(len(self.uw)) - starts[self.uw]
            self.pad_r = torch.tensor(self.uw, device=dev); self.pad_c = torch.tensor(col, device=dev); self.maxw = maxw
        M = torch.from_numpy(np.ascontiguousarray(memory)).to(dev, dt)
        self.Dsrc = self._topk(self.E, M)             # (n_win, K) ascending
        self.src_score = self._agg(self.Dsrc[:, K - 1])
        del M
        if dev == 'cuda': torch.cuda.empty_cache()

    def _topk(self, Q, M, k=K, bs=8192):
        out = []; kk = min(k, M.shape[0])
        if self.metric == 'cos':
            for s in range(0, len(Q), bs):
                out.append(1 - torch.topk((Q[s:s + bs] @ M.T).float(), kk, dim=1).values)
        else:
            m2 = (M * M).sum(1)
            for s in range(0, len(Q), bs):
                q = Q[s:s + bs]; d2 = (q * q).sum(1, keepdim=True) + m2[None] - 2 * q @ M.T
                out.append(torch.topk(d2.clamp_min(0), kk, dim=1, largest=False).values.sqrt())
        return torch.cat(out).float()

    def _agg(self, wscore):
        if self.agg == 'none':
            return wscore.cpu().numpy().astype(np.float64)
        P = torch.full((self.U, self.maxw), -1e9, device=self.device)
        P[self.pad_r, self.pad_c] = wscore
        v = torch.topk(P, min(3, self.maxw), dim=1).values
        v = torch.where(v > -1e8, v, torch.zeros_like(v))
        return (v.sum(1) / self.cnt.clamp(max=3)).cpu().numpy().astype(np.float64)

    def win_idx(self, units):
        units = np.asarray(units)
        if len(units) == 0: return np.zeros(0, int)
        return np.concatenate([np.arange(self.ws[u], self.we[u]) for u in units])

    def scores_with_insert(self, insert_units):
        """unit scores for ALL units with memory = source memory + windows of insert_units.
        (Scores of the inserted units themselves are meaningless and must not be used.)"""
        if len(insert_units) == 0: return self.src_score.copy()
        I = self.E[torch.from_numpy(self.win_idx(insert_units)).to(self.device)]
        d = self._topk(self.E, I)
        merged = torch.topk(torch.cat([self.Dsrc, d], 1), K, dim=1, largest=False).values[:, K - 1]
        return self._agg(merged)
