"""Exact quantiles of data that is read in pieces and never held in memory all at once.

np.quantile needs every value. For a dataset of tens of millions of frames that is gigabytes, so
quantiles are computed in two passes instead:

1. While the data streams past, keep a random sample of rows (about SAMPLE_ROWS of them). The
   sample tells, for each wanted rank, a narrow band of values the exact answer must lie in.
2. Stream the data again, counting the values below each band and keeping only the values inside
   it. The exact order statistics are then picked from those few values.

The result is identical to np.quantile (same order statistics, same interpolation, computed by
numpy itself), not an approximation. If a band turns out too narrow, which is very unlikely, it is
widened and the second pass repeated. When the data is small (at most EXACT_ROWS rows), the first
pass simply keeps everything and np.quantile is called directly.
"""

from __future__ import annotations

import math

import numpy as np

EXACT_ROWS = 250_000  # up to this many rows, values are kept and np.quantile used directly
SAMPLE_ROWS = 100_000  # rows sampled for locating the bands on larger data
SEED = 20261119


def lerp_quantile(lo: np.ndarray, hi: np.ndarray, gamma: float) -> np.ndarray:
    """np.quantile's linear interpolation between neighbouring order statistics, done by numpy
    itself so the floating-point result is bit-identical."""
    return np.quantile(np.stack([lo, hi]), gamma, axis=0)


class QuantileStream:
    """Exact quantiles of the rows of a stream of (n, d) arrays, per column.

    Usage: call add() for every piece, then needs_second_pass(). If True, call add_second() for
    every piece again, in any order, then result(); repeat the second pass while result() returns
    None (a band was too narrow and has been widened).
    """

    def __init__(self, qs: list[float], max_rows: int, *, exact_rows: int | None = None, seed: int = SEED) -> None:
        self.qs = [float(q) for q in qs]
        limit = EXACT_ROWS if exact_rows is None else exact_rows
        self.exact = max_rows <= limit
        self.p = 1.0 if self.exact else min(1.0, SAMPLE_ROWS / max(1, max_rows))
        self._rng = np.random.default_rng(seed)
        self._chunks: list[np.ndarray] = []
        self.n = 0
        self.d: int | None = None
        self._widen = 1.0
        self._bands: np.ndarray | None = None  # (len(qs), 2, d): lower and upper edge per quantile and column
        self._below: np.ndarray | None = None
        self._inside: list[list[list[np.ndarray]]] = []
        self._result: np.ndarray | None = None

    # -- first pass ---------------------------------------------------------------------------

    def add(self, x: np.ndarray) -> None:
        if self.d is None:
            self.d = x.shape[1]
        self.n += len(x)
        if not len(x):
            return
        if self.exact:
            self._chunks.append(x)
            return
        keep = self._rng.random(len(x)) < self.p
        if keep.any():
            self._chunks.append(x[keep])

    def needs_second_pass(self) -> bool:
        if self.n == 0:
            return False
        if self.exact:
            allv = np.concatenate(self._chunks)
            self._result = np.stack([np.quantile(allv, q, axis=0) for q in self.qs])
            self._chunks = []
            return False
        self._sample = np.sort(np.concatenate(self._chunks), axis=0) if self._chunks else None
        self._chunks = []
        self._start_second()
        return True

    # -- second pass --------------------------------------------------------------------------

    def _ranks(self, q: float) -> tuple[int, int, float]:
        h = (self.n - 1) * q  # the same expression np.quantile uses for its linear method
        lo = int(math.floor(h))
        return lo, min(lo + 1, self.n - 1), h - math.floor(h)

    def _start_second(self) -> None:
        d = self.d or 0
        bands = np.empty((len(self.qs), 2, d))
        s = self._sample
        for i, q in enumerate(self.qs):
            lo_r, hi_r, _ = self._ranks(q)
            if s is None or len(s) == 0 or self._widen > 64:
                bands[i, 0], bands[i, 1] = -np.inf, np.inf
                continue
            m = len(s)
            margin = self._widen * (4.0 * math.sqrt(m * max(q * (1 - q), 1.0 / m)) + 0.002 * m + 2.0)
            i_lo = math.floor(lo_r * m / self.n - margin)
            i_hi = math.ceil((hi_r + 1) * m / self.n + margin)
            bands[i, 0] = s[i_lo] if i_lo > 0 else -np.inf
            bands[i, 1] = s[i_hi] if i_hi < m - 1 else np.inf
        self._bands = bands
        self._below = np.zeros((len(self.qs), d), np.int64)
        self._inside = [[[] for _ in range(d)] for _ in self.qs]

    def add_second(self, x: np.ndarray) -> None:
        if not len(x) or self._bands is None or self._below is None:
            return
        for i in range(len(self.qs)):
            lo, hi = self._bands[i, 0], self._bands[i, 1]
            self._below[i] += np.count_nonzero(x < lo, axis=0)
            inside = (x >= lo) & (x <= hi)
            for j in np.flatnonzero(inside.any(axis=0)):
                self._inside[i][j].append(x[inside[:, j], j])

    def result(self) -> np.ndarray | None:
        """(len(qs), d) quantiles; None if the second pass must be run again."""
        if self._result is not None or self.n == 0:
            return self._result
        assert self._bands is not None and self._below is not None and self.d is not None
        out = np.empty((len(self.qs), self.d))
        ok = True
        for i, q in enumerate(self.qs):
            lo_r, hi_r, gamma = self._ranks(q)
            lo_v, hi_v = np.empty(self.d), np.empty(self.d)
            for j in range(self.d):
                below = int(self._below[i, j])
                vals = np.sort(np.concatenate(self._inside[i][j])) if self._inside[i][j] else np.zeros(0)
                if below > lo_r or below + len(vals) < hi_r + 1:
                    ok = False
                    break
                lo_v[j], hi_v[j] = vals[lo_r - below], vals[hi_r - below]
            if not ok:
                break
            out[i] = lerp_quantile(lo_v, hi_v, gamma)
        if not ok:
            self._widen *= 4.0
            self._start_second()
            return None
        self._result = out
        self._inside = []
        return out


class Moments:
    """Running count, means, variances and covariance of paired columns, merged chunk by chunk
    (Chan et al.), so correlations over a whole dataset need no concatenated copy of it."""

    def __init__(self) -> None:
        self.n = 0
        self.mx: np.ndarray | float = 0.0
        self.my: np.ndarray | float = 0.0
        self.sxx: np.ndarray | float = 0.0
        self.syy: np.ndarray | float = 0.0
        self.sxy: np.ndarray | float = 0.0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        nb = len(x)
        if nb == 0:
            return
        mxb, myb = x.mean(axis=0), y.mean(axis=0)
        dxb, dyb = x - mxb, y - myb
        sxxb, syyb, sxyb = (dxb * dxb).sum(axis=0), (dyb * dyb).sum(axis=0), (dxb * dyb).sum(axis=0)
        if self.n == 0:
            self.n, self.mx, self.my, self.sxx, self.syy, self.sxy = nb, mxb, myb, sxxb, syyb, sxyb
            return
        n = self.n + nb
        dx, dy = mxb - self.mx, myb - self.my
        f = self.n * nb / n
        self.sxx = self.sxx + sxxb + dx * dx * f
        self.syy = self.syy + syyb + dy * dy * f
        self.sxy = self.sxy + sxyb + dx * dy * f
        self.mx = self.mx + dx * nb / n
        self.my = self.my + dy * nb / n
        self.n = n

    def std_x(self):
        return np.sqrt(self.sxx / self.n) if self.n else 0.0

    def std_y(self):
        return np.sqrt(self.syy / self.n) if self.n else 0.0

    def corr(self):
        """Pearson correlation, clipped to [-1, 1] like np.corrcoef."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.clip(self.sxy / np.sqrt(self.sxx * self.syy), -1.0, 1.0)
