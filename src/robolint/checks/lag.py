"""Action-to-state lag: how many frames the measured follower state trails the commanded action.

A follower arm reaches each commanded position a little later than it is commanded, so the
state stream is a delayed copy of the action stream. That delay is set by the hardware and
the recording pipeline and should be about the same in every episode. An episode whose delay
is far from the dataset's typical one had its state (or action) stream buffered, stalled or
stamped late, which teaches a policy the wrong cause and effect.

The lag is estimated by cross-correlating smoothed action and state velocities over a range
of shifts, per matching joint, and taking the shift where they line up best.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..dataset import Dataset
from ..findings import Finding, Severity
from . import Config
from .signals import DimStats, _as2d, dataset_stats

W, I = Severity.WARNING, Severity.INFO


@dataclass
class LagEstimate:
    lag: float  # frames; positive = the second signal trails the first
    peak: float  # match quality at the best shift (correlation, -1..1)
    edge: bool  # best shift sits at the edge of the search window: the true lag may be larger


@dataclass
class LagCurve:
    """Match quality between two signals for every shift from -max_lag to +max_lag frames."""

    values: np.ndarray
    max_lag: int

    def peak(self) -> LagEstimate | None:
        return _peak(self.values, self.max_lag)

    def at(self, lag: float) -> float:
        """Match quality at a (fractional) shift, linearly interpolated."""
        x = np.clip(lag + self.max_lag, 0, len(self.values) - 1)
        i = int(np.floor(x))
        j = min(i + 1, len(self.values) - 1)
        v = self.values
        return float(v[i] + (x - i) * (v[j] - v[i]))


def smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Centered moving average along axis 0 (edges use the samples available)."""
    if width <= 1 or len(x) < width:
        return x
    k = np.ones(width) / width
    norm = np.convolve(np.ones(len(x)), k, mode="same")
    if x.ndim == 1:
        return np.convolve(x, k, mode="same") / norm
    return np.stack([np.convolve(x[:, j], k, mode="same") / norm for j in range(x.shape[1])], axis=1)


def _overlap(x: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    return (x[: n - k], y[k:]) if k >= 0 else (x[-k:], y[: n + k])


def _peak(curve: np.ndarray, max_lag: int) -> LagEstimate | None:
    if not np.isfinite(curve).any():
        return None
    i = int(np.nanargmax(curve))
    peak = float(curve[i])
    lag = float(i - max_lag)
    edge = i == 0 or i == len(curve) - 1
    if not edge and np.isfinite(curve[i - 1]) and np.isfinite(curve[i + 1]):
        a, b, c = curve[i - 1], curve[i], curve[i + 1]
        den = a - 2 * b + c
        if den < 0:  # a real maximum: refine to sub-frame precision
            lag += float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return LagEstimate(lag=lag, peak=peak, edge=edge)


def xcorr_curve(x: np.ndarray, y: np.ndarray, max_lag: int, weights: np.ndarray | None = None) -> LagCurve | None:
    """Correlation of `y` shifted against `x`, per shift. Positive shifts mean y[t + lag] follows x[t].

    x, y: (n,) or (n, d) series on the same clock. With several columns, the per-column Pearson
    correlations are averaged with `weights` (default equal). None if too short or flat to judge.
    """
    x = x.reshape(len(x), -1).astype(np.float64)
    y = y.reshape(len(y), -1).astype(np.float64)
    n, d = x.shape
    max_lag = int(min(max_lag, n // 3))
    if max_lag < 1 or n < 8:
        return None
    w = np.ones(d) if weights is None else np.asarray(weights, dtype=np.float64)
    curve = np.full(2 * max_lag + 1, np.nan)
    for i, k in enumerate(range(-max_lag, max_lag + 1)):
        xa, yb = _overlap(x, y, k)
        xa = xa - xa.mean(axis=0)
        yb = yb - yb.mean(axis=0)
        den = np.sqrt((xa**2).sum(axis=0) * (yb**2).sum(axis=0))
        ok = den > 1e-12
        if not ok.any() or w[ok].sum() <= 0:
            continue
        r = (xa[:, ok] * yb[:, ok]).sum(axis=0) / den[ok]
        curve[i] = float(np.sum(w[ok] * r) / w[ok].sum())
    return LagCurve(curve, max_lag) if np.isfinite(curve).any() else None


def _nonneg_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares prediction of y from the columns of X plus a constant, with every column's weight
    kept at or above zero (more motion can only mean more image change): drop the most negative column
    and refit until none is negative."""
    keep = np.ones(X.shape[1], bool)
    while keep.any():
        A = np.column_stack([X[:, keep], np.ones(len(X))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        if np.all(coef[:-1] >= 0):
            return A @ coef
        idx = np.flatnonzero(keep)
        keep[idx[int(np.argmin(coef[:-1]))]] = False
    return np.full(len(y), y.mean())


def fit_curve(X: np.ndarray, y: np.ndarray, max_lag: int) -> LagCurve | None:
    """How well the columns of X (e.g. per-joint speeds) explain y (e.g. image change) for every shift
    of y, as the square root of the fit's R^2. Positive shifts mean y[t + lag] follows X[t]."""
    X = X.reshape(len(X), -1).astype(np.float64)
    y = y.astype(np.float64)
    n = len(y)
    max_lag = int(min(max_lag, n // 3))
    if max_lag < 1 or n < 8:
        return None
    curve = np.full(2 * max_lag + 1, np.nan)
    for i, k in enumerate(range(-max_lag, max_lag + 1)):
        xa, yb = _overlap(X, y, k)
        var = float(np.var(yb))
        if var <= 1e-12:
            continue
        r2 = 1.0 - float(np.mean((yb - _nonneg_fit(xa, yb)) ** 2)) / var
        curve[i] = float(np.sqrt(max(r2, 0.0)))
    return LagCurve(curve, max_lag) if np.isfinite(curve).any() else None


@dataclass
class LagOutlier:
    episode: int
    estimate: LagEstimate
    margin: float  # how much better the episode's own shift fits than the dataset's typical one


@dataclass
class LagSummary:
    typical: float  # frames
    spread: float  # robust standard deviation of per-episode lags, frames
    episodes: int  # episodes with a usable estimate
    outliers: list[LagOutlier]


def judge(
    curves: dict[int, LagCurve],
    fps: float,
    *,
    min_corr: float,
    flag_corr: float,
    min_episodes: int,
    tol_seconds: float,
    tol_frames: float,
    mad_factor: float,
    margin: float,
) -> LagSummary | None:
    """Typical lag across a dataset, and the episodes whose own lag is clearly different.

    The typical lag is the best shift of the median curve, which is far steadier than any one
    episode. An episode is an outlier only if its best shift is far from the typical one, the
    match there is strong, and it fits clearly better than the typical shift does: motion with a
    repeating rhythm can line up almost as well one cycle over, and that must not count.
    """
    good = {ep: (c, est) for ep, c in curves.items() if (est := c.peak()) is not None}
    good = {ep: (c, est) for ep, (c, est) in good.items() if not est.edge and est.peak >= min_corr}
    if len(good) < min_episodes:
        return None
    max_lag = min(c.max_lag for c, _ in good.values())
    pooled = np.nanmedian([c.values[c.max_lag - max_lag : c.max_lag + max_lag + 1] for c, _ in good.values()], axis=0)
    overall = _peak(pooled, max_lag)
    if overall is None or overall.edge:
        return None
    typical = overall.lag
    lags = np.array([est.lag for _, est in good.values()])
    spread = robust_spread(lags)
    tol = max(tol_seconds * fps, tol_frames, mad_factor * spread)
    outliers = []
    for ep, (c, est) in sorted(good.items()):
        gain = est.peak - c.at(typical)
        if abs(est.lag - typical) > tol and est.peak >= flag_corr and gain >= margin:
            outliers.append(LagOutlier(ep, est, gain))
    return LagSummary(typical=typical, spread=spread, episodes=len(good), outliers=outliers)


def robust_spread(values: np.ndarray) -> float:
    """Scaled median absolute deviation (matches the standard deviation for normal data)."""
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def matched_dims(a_st: DimStats, s_st: DimStats) -> list[tuple[int, int]]:
    """(action dim, state dim) pairs for joints that appear under the same name in both, excluding
    grippers (an object stops them short of the command) and on/off channels."""
    a_names = a_st.names
    s_index = {n: i for i, n in enumerate(s_st.names)}
    pairs = []
    for ja, name in enumerate(a_names):
        js = s_index.get(name)
        if js is None or "gripper" in name.lower():
            continue
        if not (a_st.active[ja] and s_st.active[js]) or a_st.discrete[ja] or s_st.discrete[js]:
            continue
        pairs.append((ja, js))
    if not pairs and len(a_names) == len(s_st.names) and a_names != s_st.names:
        # No shared names (e.g. undeclared names): pair by position; the tracking test below keeps
        # only columns where the state really follows the command.
        pairs = [
            (j, j)
            for j in range(len(a_names))
            if a_st.active[j] and s_st.active[j] and not a_st.discrete[j] and not s_st.discrete[j]
        ]
    return pairs


def _usable(key: str, d: int, ep) -> np.ndarray | None:
    arr = ep.frames.get(key)
    if arr is None:
        return None
    arr = _as2d(arr).astype(np.float64)
    if arr.shape[1] != d or not np.all(np.isfinite(arr)):
        return None
    return arr


@dataclass
class PositionJoints:
    """Joints whose commanded action is a position that the measured state follows."""

    action_dims: np.ndarray  # indices into the action vector
    state_dims: np.ndarray  # matching indices into the state vector
    names: list[str]
    rng: np.ndarray  # robust range of each action dim
    episodes: list[tuple[int, np.ndarray, np.ndarray]]  # (episode, action, state) for usable episodes


def position_joints(ds: Dataset, cfg: Config) -> PositionJoints | None:
    """Joints that appear in both action and state, where the state really follows the command
    (position control). Velocity or delta commands, torque commands, or a state that measures
    something else have no lag to estimate, and their actions do not describe visible motion as
    positions. None if there are none."""
    cache = ds.__dict__.setdefault("_position_joints_cache", {})
    key = (cfg.lag_tracking_corr, cfg.spike_quantile)
    if key not in cache:
        cache[key] = _position_joints(ds, cfg)
    return cache[key]


def _position_joints(ds: Dataset, cfg: Config) -> PositionJoints | None:
    a_st = dataset_stats(ds, "action", cfg)
    s_st = dataset_stats(ds, "observation.state", cfg)
    if a_st is None or s_st is None:
        return None
    pairs = matched_dims(a_st, s_st)
    if not pairs:
        return None
    ia = np.array([p[0] for p in pairs])
    i_s = np.array([p[1] for p in pairs])

    episodes = []
    for ep in ds.episodes:
        fi = ep.frames.get("frame_index")
        if fi is not None and len(np.unique(fi)) < len(fi):
            continue  # merged/duplicated frames: already an error, and the signals are interleaved
        a = _usable("action", len(a_st.rng), ep)
        s = _usable("observation.state", len(s_st.rng), ep)
        if a is None or s is None or len(a) != len(s):
            continue
        episodes.append((ep.index, a[:, ia], s[:, i_s]))
    if not episodes:
        return None

    a_all = np.concatenate([a for _, a, _ in episodes])
    s_all = np.concatenate([s for _, _, s in episodes])
    tracks = np.array(
        [
            np.std(a_all[:, j]) > 1e-12
            and np.std(s_all[:, j]) > 1e-12
            and np.corrcoef(a_all[:, j], s_all[:, j])[0, 1] >= cfg.lag_tracking_corr
            for j in range(len(pairs))
        ]
    )
    if not tracks.any():
        return None
    return PositionJoints(
        action_dims=ia[tracks],
        state_dims=i_s[tracks],
        names=[a_st.names[int(j)] for j in ia[tracks]],
        rng=a_st.rng[ia[tracks]],
        episodes=[(ep, a[:, tracks], s[:, tracks]) for ep, a, s in episodes],
    )


def check(ds: Dataset, cfg: Config) -> Iterator[Finding]:
    pj = position_joints(ds, cfg)
    if pj is None:
        return
    fps = ds.fps or 30.0
    rng = pj.rng
    max_lag = max(3, int(round(cfg.lag_max_seconds * fps)))
    width = max(1, int(round(cfg.lag_smooth_seconds * fps)))
    min_len = max(int(cfg.lag_min_seconds * fps), 4 * width, 12)
    curves: dict[int, LagCurve] = {}
    for ep_idx, a, s in pj.episodes:
        if len(a) < min_len:
            continue
        # Static episodes (or ones where only noise moves) have nothing to line up.
        travel = (a.max(axis=0) - a.min(axis=0)) / rng
        moving = travel >= cfg.lag_min_travel
        if not moving.any():
            continue
        va = smooth(np.diff(a[:, moving], axis=0), width) / rng[moving]
        vs = smooth(np.diff(s[:, moving], axis=0), width) / rng[moving]
        curve = xcorr_curve(va, vs, max_lag, travel[moving] ** 2)  # joints that move more count more
        if curve is not None:
            curves[ep_idx] = curve

    summary = judge(
        curves,
        fps,
        min_corr=cfg.lag_min_corr,
        flag_corr=cfg.lag_flag_corr,
        min_episodes=cfg.lag_min_episodes,
        tol_seconds=cfg.lag_tol_seconds,
        tol_frames=cfg.lag_tol_frames,
        mad_factor=cfg.lag_mad_factor,
        margin=cfg.lag_margin,
    )
    if summary is None:
        return
    typical = summary.typical
    ms = typical / fps * 1000
    if typical >= 0:
        text = f"the measured state typically trails the commanded action by {typical:.1f} frames ({ms:.0f} ms)"
    else:
        text = (
            f"the measured state typically moves {-typical:.1f} frames ({-ms:.0f} ms) before the commanded action: "
            "actions may be recorded late relative to observations"
        )
    yield Finding(
        "STATE_LAG",
        I,
        f"{text}, estimated over {summary.episodes} episodes",
        None,
        {
            "typical_lag_frames": round(typical, 2),
            "typical_lag_ms": round(ms, 1),
            "spread_frames": round(summary.spread, 2),
            "episodes_estimated": summary.episodes,
            "joints": pj.names,
        },
    )
    for o in summary.outliers:
        off = o.estimate.lag - typical
        side = "later" if off > 0 else "earlier"
        yield Finding(
            "STATE_LAG_OUTLIER",
            W,
            f"the measured state follows the commanded action {abs(off):.1f} frames ({abs(off) / fps * 1000:.0f} ms) "
            f"{side} than in the rest of the dataset (lag {o.estimate.lag:.1f} vs typical {typical:.1f} frames): "
            "the state and action streams may be out of step",
            o.episode,
            {
                "lag_frames": round(o.estimate.lag, 2),
                "typical_frames": round(typical, 2),
                "offset_frames": round(off, 2),
                "confidence": round(o.estimate.peak, 3),
                "fit_gain": round(o.margin, 3),
            },
        )
