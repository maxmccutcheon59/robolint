"""Action and state signal quality: NaNs, dead or frozen episodes, glitches, idle padding."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..dataset import Dataset
from ..findings import Finding, Severity
from . import Config

E, W, I = Severity.ERROR, Severity.WARNING, Severity.INFO
KEYS = ("action", "observation.state")


@dataclass
class DimStats:
    lo: np.ndarray  # true min per dimension
    hi: np.ndarray  # true max per dimension
    rng: np.ndarray  # robust range (0.5th-99.5th percentile) per dimension
    step_scale: np.ndarray  # high quantile of |value - rolling median| per dimension
    names: list[str]
    discrete: np.ndarray  # dims that only take a handful of values (e.g. a binary gripper command)

    @property
    def active(self) -> np.ndarray:
        return self.rng > 1e-9

    def gripper_mask(self) -> np.ndarray:
        return (
            np.array(["gripper" in n.lower() for n in self.names], dtype=bool)
            if self.names
            else np.zeros(len(self.rng), bool)
        )


def _as2d(a: np.ndarray) -> np.ndarray:
    return a.reshape(-1, 1) if a.ndim == 1 else a


SPIKE_WINDOW = 5  # frames; a glitch sticks out from the median of its neighbours


def outlier_residual(a: np.ndarray) -> np.ndarray:
    """|value - rolling median|. Large for one- or two-frame spikes that snap back; zero for
    steps, ramps and sample-and-hold signals, which are normal motion rather than glitches."""
    n = len(a)
    if n < SPIKE_WINDOW:
        return np.zeros_like(a)
    h = SPIKE_WINDOW // 2
    padded = np.concatenate([np.repeat(a[:1], h, axis=0), a, np.repeat(a[-1:], h, axis=0)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, SPIKE_WINDOW, axis=0)
    return np.abs(a - np.partition(windows, h, axis=-1)[..., h])


def dataset_stats(ds: Dataset, key: str, cfg: Config) -> DimStats | None:
    arrays = [_as2d(ep.frames[key]) for ep in ds.episodes if key in ep.frames and ep.length]
    if not arrays:
        return None
    d = arrays[0].shape[1]
    arrays = [a for a in arrays if a.shape[1] == d]
    allv = np.concatenate(arrays).astype(np.float64)
    finite_rows = np.all(np.isfinite(allv), axis=1)
    allv = allv[finite_rows]
    if len(allv) == 0:
        return None
    resid = [outlier_residual(a[np.all(np.isfinite(a), axis=1)].astype(np.float64)) for a in arrays]
    resid = [r for r in resid if len(r)]
    if resid:
        # Quantile over every frame: in held/stepped signals almost all residuals are zero, and a
        # quantile over only the non-zero ones would let a lone glitch set its own threshold.
        scale = np.quantile(np.concatenate(resid), cfg.spike_quantile, axis=0)
    else:
        scale = np.zeros(d)
    names = ds.feature_names(key)
    if len(names) != d:
        names = [f"{key}[{i}]" for i in range(d)]
    sample = allv[:: max(1, len(allv) // 200_000)]
    discrete = np.array([len(np.unique(sample[:, j])) <= 4 for j in range(d)], dtype=bool)
    return DimStats(
        lo=allv.min(axis=0),
        hi=allv.max(axis=0),
        rng=np.quantile(allv, 0.995, axis=0) - np.quantile(allv, 0.005, axis=0),
        step_scale=scale,
        names=names,
        discrete=discrete,
    )


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[::2]).max())


def check(ds: Dataset, cfg: Config) -> Iterator[Finding]:
    stats = {k: dataset_stats(ds, k, cfg) for k in KEYS}
    fps = ds.fps or 30.0
    a_st, s_st = stats["action"], stats["observation.state"]
    grasp_eps: list[tuple[int, np.ndarray, np.ndarray]] = []  # healthy episodes for the grasp check

    for ep in ds.episodes:
        if ep.length < 2:
            continue
        # Episodes whose frames are duplicated (merged recordings) are already errors; their
        # interleaved signals would only produce misleading motion findings.
        fi = ep.frames.get("frame_index")
        corrupt = fi is not None and len(np.unique(fi)) < len(fi)
        clean: dict[str, bool] = {}
        for key in KEYS:
            if key not in ep.frames:
                continue
            arr = _as2d(ep.frames[key])
            bad = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
            clean[key] = bad == 0
            if bad:
                yield Finding(
                    "NONFINITE_VALUES",
                    E,
                    f"'{key}' contains {bad} NaN/inf values",
                    ep.index,
                    {"key": key, "count": bad},
                )

        action_dead = False
        if a_st is not None and clean.get("action"):
            a = _as2d(ep.frames["action"]).astype(np.float64)
            if a.shape[1] == len(a_st.rng):
                act = a_st.active
                ep_rng = a.max(axis=0) - a.min(axis=0)
                tol_dead = cfg.dead_tol * a_st.rng
                if act.any() and np.all(ep_rng[act] < tol_dead[act]):
                    action_dead = True
                    yield Finding(
                        "DEAD_EPISODE",
                        E,
                        "the commanded actions never meaningfully change: the robot does not move",
                        ep.index,
                    )
                elif not corrupt:
                    yield from _gripper(ep.index, ep_rng, a_st, cfg)
                    if len(a) >= 2 * fps:  # too short to judge padding meaningfully
                        yield from _idle(ep.index, a, a_st, cfg)
                    yield from _spikes(ep.index, a, a_st, cfg)

        if s_st is not None and clean.get("observation.state"):
            s = _as2d(ep.frames["observation.state"]).astype(np.float64)
            if s.shape[1] == len(s_st.rng):
                act = s_st.active
                s_rng = s.max(axis=0) - s.min(axis=0)
                state_frozen = act.any() and np.all(s_rng[act] < cfg.dead_tol * s_st.rng[act])
                if state_frozen and not action_dead and a_st is not None and clean.get("action"):
                    yield Finding(
                        "STATE_FROZEN",
                        E,
                        "actions change but the measured state never does: the follower robot was likely disconnected or not recording",
                        ep.index,
                    )
                elif not state_frozen and not corrupt and a_st is not None and clean.get("action"):
                    a = _as2d(ep.frames["action"]).astype(np.float64)
                    if not action_dead and a.shape[1] == len(a_st.rng):
                        grasp_eps.append((ep.index, a, s))
                    if a_st.names == s_st.names:
                        lead, tail = idle_bounds(a, a_st, cfg)
                        # Resting on a hard stop before/after a take is normal; judge only the active part.
                        if lead < len(a) - tail:
                            yield from _limits(
                                ep.index, s[lead : len(a) - tail], a[lead : len(a) - tail], s_st, fps, cfg
                            )

    if a_st is not None and s_st is not None:
        yield from _grasps(grasp_eps, a_st, s_st, fps, cfg)


def _gripper(ep: int, ep_rng: np.ndarray, st: DimStats, cfg: Config) -> Iterator[Finding]:
    g = st.gripper_mask() & st.active
    if g.any() and np.all(ep_rng[g] < cfg.gripper_tol * st.rng[g]):
        yield Finding("GRIPPER_NEVER_ACTUATES", W, "the gripper never opens or closes in this episode", ep)


def idle_bounds(a: np.ndarray, st: DimStats, cfg: Config) -> tuple[int, int]:
    """Frames of idle lead-in and tail (command not yet / no longer moving)."""
    tol = np.where(st.active, cfg.move_tol * st.rng, np.inf)
    n = len(a)
    from_start = np.any(np.abs(a - a[0]) > tol, axis=1)
    from_end = np.any(np.abs(a - a[-1]) > tol, axis=1)
    if not from_start.any():
        return n, 0
    lead = int(np.argmax(from_start))
    tail = n - 1 - int(np.flatnonzero(from_end)[-1]) if from_end.any() else 0
    return lead, tail


def _idle(ep: int, a: np.ndarray, st: DimStats, cfg: Config) -> Iterator[Finding]:
    n = len(a)
    lead, tail = idle_bounds(a, st, cfg)
    if lead >= n:
        return
    frac = (lead + tail) / n
    if frac >= cfg.idle_info:
        sev = W if frac >= cfg.idle_warn else I
        yield Finding(
            "IDLE_PADDING",
            sev,
            f"{frac * 100:.0f}% of the episode is idle ({lead} frames before motion starts, {tail} after it ends)",
            ep,
            {"idle_fraction": round(frac, 3), "lead_frames": lead, "tail_frames": tail},
        )


def _spikes(ep: int, a: np.ndarray, st: DimStats, cfg: Config) -> Iterator[Finding]:
    thr = np.maximum(cfg.spike_multiplier * st.step_scale, cfg.spike_min_range * st.rng)
    thr = np.where(st.active & ~st.discrete & (thr > 0), thr, np.inf)  # on/off channels toggle by design
    jumps = outlier_residual(a) > thr
    rows = np.flatnonzero(jumps.any(axis=1))
    if not len(rows):
        return
    # neighbouring outlier frames belong to the same glitch
    events = [int(rows[0])]
    for r in rows[1:]:
        if r - events[-1] > 3:
            events.append(int(r))
    dims = sorted({st.names[j] for j in np.flatnonzero(jumps.any(axis=0))})
    n = len(events)
    yield Finding(
        "ACTION_SPIKE",
        W,
        f"{n} sudden action {'jump' if n == 1 else 'jumps'} (likely teleoperation glitches) in {', '.join(dims)}",
        ep,
        {"count": n, "frames": events[:20], "dims": dims},
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _limits(ep: int, s: np.ndarray, a: np.ndarray, st: DimStats, fps: float, cfg: Config) -> Iterator[Finding]:
    """The arm sits at its extreme while the command is past it.

    Command moving inside the limit run -> the operator is pushing into a hard stop mid-task (warning).
    Command static -> leader and follower calibrations disagree at a rest pose (info)."""
    need = int(cfg.limit_seconds * fps)
    margin = 0.005 * st.rng
    push = 0.02 * st.rng
    pushed, offset = [], []
    skip = st.gripper_mask()
    for j in range(s.shape[1]):
        if not st.active[j] or skip[j]:
            continue
        low = (s[:, j] <= st.lo[j] + margin[j]) & (a[:, j] < s[:, j] - push[j])
        high = (s[:, j] >= st.hi[j] - margin[j]) & (a[:, j] > s[:, j] + push[j])
        long_runs = [(b, e) for b, e in _runs(low) + _runs(high) if e - b >= need]
        if not long_runs:
            continue
        moving = any(np.ptp(a[b:e, j]) > 2 * cfg.move_tol * st.rng[j] for b, e in long_runs)
        (pushed if moving else offset).append(st.names[j])
    if pushed:
        yield Finding(
            "JOINT_AT_LIMIT",
            W,
            f"pushed past its limit for {cfg.limit_seconds:g}s+ during the task while the arm stays pinned "
            f"(hard stop): {', '.join(pushed)}",
            ep,
            {"joints": pushed},
        )
    if offset:
        yield Finding(
            "CALIBRATION_OFFSET",
            I,
            f"at rest the command sits past what the arm can reach (leader/follower calibration mismatch): "
            f"{', '.join(offset)}",
            ep,
            {"joints": offset},
        )


def _gripper_pairs(a_st: DimStats, s_st: DimStats) -> list[tuple[int, int]]:
    """(action dim, state dim) for each gripper, matched in order (one pair per arm)."""
    ia = [i for i, n in enumerate(a_st.names) if "gripper" in n.lower()]
    i_s = [i for i, n in enumerate(s_st.names) if "gripper" in n.lower()]
    return list(zip(ia, i_s, strict=False)) if len(ia) == len(i_s) else []


def _grasps(
    eps: list[tuple[int, np.ndarray, np.ndarray]], a_st: DimStats, s_st: DimStats, fps: float, cfg: Config
) -> Iterator[Finding]:
    """Grasps that closed on nothing.

    When a gripper closes on an object, the object blocks it and the follower stops partway.
    When it closes on air, it travels to its fully closed position. For each side the gripper
    is commanded to and held at, collect where the follower settles in every episode: if it
    usually stops short of the farthest settling point (the dataset holds objects), a settle
    at that farthest point is a grasp on nothing. An episode whose last grasp missed is very
    likely a failed demonstration; one that missed and then grasped is a recovery.
    """
    pairs = _gripper_pairs(a_st, s_st)
    if not pairs or len(eps) < 5:
        return
    min_len = max(3, int(cfg.grasp_hold_seconds * fps))
    for ja, js in pairs:
        a_lo, a_hi, s_rng = a_st.lo[ja], a_st.hi[ja], s_st.rng[js]
        if a_hi - a_lo <= 1e-9 or s_rng <= 1e-9:
            continue
        # does the follower's reading rise or fall when the command rises?
        a_all = np.concatenate([a[:, ja] for _, a, _ in eps])
        s_all = np.concatenate([s[:, js] for _, _, s in eps])
        if np.std(a_all) <= 1e-12 or np.std(s_all) <= 1e-12:
            continue
        sign = 1.0 if np.corrcoef(a_all, s_all)[0, 1] >= 0 else -1.0
        # (episode, settle position, follower minus command) for each held segment, per side
        segs: dict[float, list[tuple[int, float, float]]] = {1.0: [], -1.0: []}
        for ep, a, s in eps:
            c = (a[:, ja] - a_lo) / (a_hi - a_lo)
            for side, held in ((1.0, c > 0.8), (-1.0, c < 0.2)):
                for b, e in _runs(held):
                    if e - b >= min_len:
                        tail = slice(b + (e - b) // 2, e)
                        segs[side].append(
                            (ep, float(np.median(s[tail, js])), float(np.median(s[tail, js] - a[tail, ja])))
                        )
        for side in (1.0, -1.0):
            if len(segs[side]) < 5:
                continue
            d = side * sign  # direction the follower travels when commanded to this side
            settle = np.array([v for _, v, _ in segs[side]])
            farthest = float(np.max(d * settle))
            shortfall = (farthest - d * settle) / s_rng
            if np.median(shortfall) < cfg.grasp_object_min:
                continue  # this side is free motion (opening, or closing on nothing by design)
            # Second test, so a smaller object is not mistaken for air: a gripper closing on nothing
            # reaches the operator's command, offset only by the leader/follower calibration seen
            # when the gripper moves freely to the other side.
            free = [g for _, _, g in segs[-side]]
            offset = float(np.median(free)) if free else 0.0
            by_ep: dict[int, list[bool]] = {}
            for (ep, _, gap), short in zip(segs[side], shortfall, strict=True):
                reached = abs(gap - offset) < cfg.grasp_air_tol * s_rng
                by_ep.setdefault(ep, []).append(bool(short < cfg.grasp_air_tol and reached))
            name = s_st.names[js]
            for ep, missed in sorted(by_ep.items()):
                if missed[-1]:
                    yield Finding(
                        "FAILED_GRASP",
                        E,
                        f"'{name}' closed fully on nothing in the last grasp, where other episodes stop on an "
                        "object: likely a failed demonstration",
                        ep,
                        {"gripper": name, "missed_grasps": sum(missed), "grasps": len(missed)},
                    )
                elif any(missed):
                    yield Finding(
                        "REGRASP",
                        I,
                        f"'{name}' missed {sum(missed)} grasp(s) before picking the object up: a recovery, "
                        "useful data if the retry is intended",
                        ep,
                        {"gripper": name, "missed_grasps": sum(missed), "grasps": len(missed)},
                    )
