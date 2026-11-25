"""Robot-agnostic per-episode features, measured relative to the episode's own dataset.

Every feature is a ratio or fraction against dataset-level statistics, so one model works across
arms, grippers, humanoids and control rates without knowing the robot.
"""

from __future__ import annotations

import numpy as np

N_RESAMPLE = 32
FEATURES = [
    "len_rel",
    "idle_frac",
    "start_idle",
    "end_idle",
    "mid_idle_max",
    "max_jump",
    "n_jumps",
    "jerk_p99",
    "hf_energy",
    "repeat_frac",
    "state_frozen_max",
    "track_err",
    "track_corr",
    "traj_dist",
    "start_dist",
    "end_dist",
    "path_rel",
    "range_rel",
    "return_to_start",
]


def _resample(x: np.ndarray, n: int = N_RESAMPLE) -> np.ndarray:
    t = np.linspace(0, len(x) - 1, n)
    i = np.floor(t).astype(int)
    j = np.minimum(i + 1, len(x) - 1)
    w = (t - i)[:, None]
    return x[i] * (1 - w) + x[j] * w


def _runs(mask: np.ndarray) -> tuple[int, int, int]:
    """(leading run, trailing run, longest run) of True in mask."""
    if not mask.any():
        return 0, 0, 0
    if mask.all():
        return len(mask), len(mask), len(mask)
    lead = int(np.argmin(mask))
    trail = int(np.argmin(mask[::-1]))
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    best = int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())
    return lead, trail, best


class DatasetContext:
    """Robust statistics of one dataset, computed from all its episodes."""

    def __init__(self, episodes: list[dict]):
        acts = [e["action"] for e in episodes if len(e["action"]) >= 4]
        allx = np.concatenate(acts)
        self.center = np.median(allx, axis=0)
        iqr = np.percentile(allx, 75, axis=0) - np.percentile(allx, 25, axis=0)
        span = allx.max(axis=0) - allx.min(axis=0)
        scale = np.where(iqr > 1e-6, iqr, np.where(span > 1e-6, span, 1.0))
        self.scale = scale
        self.active = span > 1e-6  # dimensions that ever move
        self.med_len = float(np.median([len(a) for a in acts]))
        sp = [self._speed(self.norm(a)) for a in acts]
        allsp = np.concatenate(sp)
        self.speed_med = float(np.median(allsp[allsp > 0])) if (allsp > 0).any() else 1e-3
        self.speed_p99 = float(np.percentile(allsp, 99)) + 1e-9
        jk = np.concatenate([np.abs(np.diff(self.norm(a), n=2, axis=0)).max(axis=1) for a in acts if len(a) > 3])
        self.jerk_p99 = float(np.percentile(jk, 99)) + 1e-9
        rs = np.stack([_resample(self.norm(a)) for a in acts])
        self.mean_traj = np.median(rs, axis=0)
        d = np.sqrt(((rs - self.mean_traj) ** 2).mean(axis=(1, 2)))
        self.traj_med = float(np.median(d)) + 1e-9
        self.start = np.median(rs[:, 0], axis=0)
        self.end = np.median(rs[:, -1], axis=0)
        sd = np.linalg.norm(rs[:, 0] - self.start, axis=1)
        ed = np.linalg.norm(rs[:, -1] - self.end, axis=1)
        self.start_med = float(np.median(sd)) + 0.05
        self.end_med = float(np.median(ed)) + 0.05
        self.path_med = float(np.median([s.sum() for s in sp])) + 1e-9
        rg = np.stack([self.norm(a).max(0) - self.norm(a).min(0) for a in acts])
        self.range_med = np.median(rg, axis=0) + 1e-6
        errs = [self._track(e) for e in episodes if e.get("state") is not None]
        errs = [x for x in errs if x is not None]
        self.track_med = float(np.median(errs)) + 1e-6 if errs else 1.0

    def norm(self, a: np.ndarray) -> np.ndarray:
        return ((a - self.center) / self.scale)[:, self.active]

    @staticmethod
    def _speed(x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.diff(x, axis=0), axis=1) if len(x) > 1 else np.zeros(1)

    def _track(self, e: dict) -> float | None:
        s, a = e.get("state"), e["action"]
        if s is None or s.shape != a.shape or len(a) < 4:
            return None
        return float(np.abs(self.norm(a) - self.norm(s)).mean())


def episode_features(e: dict, ctx: DatasetContext) -> np.ndarray:
    a = ctx.norm(e["action"])
    T = len(a)
    if T < 4:
        a = np.concatenate([a, np.repeat(a[-1:], 4 - T, axis=0)])
    sp = ctx._speed(a)
    idle = sp < 0.05 * ctx.speed_med
    lead, trail, _ = _runs(idle)
    mid = idle[lead : len(idle) - trail] if trail else idle[lead:]
    _, _, mid_best = _runs(mid) if len(mid) else (0, 0, 0)
    jumps = sp / ctx.speed_p99
    jerk = np.abs(np.diff(a, n=2, axis=0)).max(axis=1) / ctx.jerk_p99
    # high-frequency energy: share of motion that is frame-to-frame reversal
    d1 = np.diff(a, axis=0)
    rev = (d1[1:] * d1[:-1] < 0).mean() if len(d1) > 1 else 0.0
    repeat = float((sp == 0).mean())
    state_frozen = 0.0
    track_err, track_corr = 1.0, 1.0
    s = e.get("state")
    if s is not None and s.shape == e["action"].shape and len(s) >= 4:
        sn = ctx.norm(s)
        ss = ctx._speed(sn)
        _, _, sbest = _runs(ss < 1e-7)
        state_frozen = sbest / len(ss)
        track_err = float(np.abs(a[: len(sn)] - sn).mean()) / ctx.track_med
        da, ds_ = np.diff(a, axis=0).ravel(), np.diff(sn, axis=0).ravel()
        if da.std() > 1e-9 and ds_.std() > 1e-9:
            track_corr = float(np.corrcoef(da, ds_)[0, 1])
        else:
            track_corr = 0.0
    r = _resample(a)
    traj = np.sqrt(((r - ctx.mean_traj) ** 2).mean()) / ctx.traj_med
    start_d = np.linalg.norm(r[0] - ctx.start) / ctx.start_med
    end_d = np.linalg.norm(r[-1] - ctx.end) / ctx.end_med
    # stacked recordings: the arm comes back to the start pose in the middle, then moves again
    dist0 = np.linalg.norm(a - a[0], axis=1)
    far = dist0.max() + 1e-9
    m = slice(T // 5, max(T // 5 + 1, 4 * T // 5))
    back = dist0[m] / far
    ret = 0.0
    if len(back) and far > 0.5:
        k = int(np.argmin(back))
        later = dist0[m][k:].max() / far if k < len(back) else 0.0
        ret = float((1 - back[k]) * later)
    rng = (a.max(0) - a.min(0)) / ctx.range_med
    return np.array(
        [
            np.log(T / ctx.med_len),
            idle.mean(),
            lead / len(idle),
            trail / len(idle),
            mid_best / len(idle),
            np.log1p(jumps.max()),
            float((jumps > 3).sum()),
            np.log1p(np.percentile(jerk, 99)),
            rev,
            repeat,
            state_frozen,
            np.log1p(track_err),
            track_corr,
            np.log1p(traj),
            np.log1p(start_d),
            np.log1p(end_d),
            np.log(sp.sum() / ctx.path_med + 1e-6),
            float(np.median(rng)),
            ret,
        ],
        dtype=np.float32,
    )
