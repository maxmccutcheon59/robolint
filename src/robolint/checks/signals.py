"""Action and state signal quality: NaNs, dead or frozen episodes, glitches, idle padding."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np

from ..dataset import Dataset, DatasetError
from ..findings import Finding, Severity
from ..stream import Moments, QuantileStream
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
    grippers: np.ndarray | None = None  # gripper dims named in the config; None: found by name

    @property
    def active(self) -> np.ndarray:
        return self.rng > 1e-9

    def gripper_mask(self) -> np.ndarray:
        if self.grippers is not None:
            return self.grippers
        return (
            np.array(["gripper" in n.lower() for n in self.names], dtype=bool)
            if self.names
            else np.zeros(len(self.rng), bool)
        )


def gripper_spec(cfg: Config, key: str, a_names: list[str], a_width: int, s_names: list[str], s_width: int):
    """The configured gripper dims of `key` ("action" or "observation.state"), as given or, for
    the state when only action grippers are given, carried over: the same name if the state has
    it, otherwise the same position when action and state have the same width."""

    def norm(spec, names):  # "5" means index 5 unless a dimension is called "5"
        return tuple(int(g) if isinstance(g, str) and g.isdigit() and g not in names else g for g in spec)

    if key == "action" or cfg.state_gripper_dims:
        return norm(cfg.gripper_dims, a_names) if key == "action" else norm(cfg.state_gripper_dims, s_names)
    out: list[str | int] = []
    for g in norm(cfg.gripper_dims, a_names):
        j = a_names.index(g) if isinstance(g, str) and g in a_names else g if isinstance(g, int) else None
        if isinstance(g, str) and g in s_names:
            out.append(g)
        elif j is not None and a_width == s_width:
            out.append(int(j))
        else:
            raise DatasetError(
                f"gripper {g!r} has no matching state dimension; name it with state_gripper_dims (--state-gripper)"
            )
    return tuple(out)


def gripper_mask_for(spec, names: list[str], width: int, key: str) -> np.ndarray:
    """Mask of the dims in `spec` (names or indices). Raises DatasetError for an unknown one."""
    mask = np.zeros(width, bool)
    for g in spec:
        if isinstance(g, bool) or not isinstance(g, (int, str)):
            raise DatasetError(f"gripper dims must be names or indices, got {g!r}")
        if isinstance(g, int):
            if not 0 <= g < width:
                raise DatasetError(f"gripper index {g} is outside '{key}', which has {width} dimensions")
            mask[g] = True
        elif g in names:
            mask[names.index(g)] = True
        else:
            known = ", ".join(names) if names else "none declared; use an index"
            raise DatasetError(f"gripper {g!r} is not a dimension of '{key}' (names: {known})")
    return mask


def configured_grippers(ds: Dataset, cfg: Config) -> dict[str, np.ndarray]:
    """Gripper masks named in the config, for the keys the dataset has. Raises DatasetError if a
    configured dim does not exist."""
    if not cfg.gripper_dims and not cfg.state_gripper_dims:
        return {}
    info = {k: (_names(ds, k), _width(ds, k)) for k in KEYS}
    (an, aw), (sn, sw) = info["action"], info["observation.state"]
    out = {}
    for key in KEYS:
        names, width = info[key]
        if width is None:
            continue
        spec = gripper_spec(cfg, key, an, aw or 0, sn, sw or 0)
        if spec:
            out[key] = gripper_mask_for(spec, names, width, key)
    return out


def _width(ds: Dataset, key: str) -> int | None:
    if ds.store is not None:
        sp = ds.store.spec.get(key)
        return None if sp is None else max(1, sp.width)
    for ep in ds.episodes:
        if key in ep.frames:
            return _as2d(ep.frames[key]).shape[1]
    return None


def _names(ds: Dataset, key: str) -> list[str]:
    names = ds.feature_names(key)
    w = _width(ds, key)
    return names if w is not None and len(names) == w else []


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
    """Per-dimension statistics of `key` across the dataset (computed once per dataset and config).

    Action and state statistics are computed together, in the same passes over the data.
    """
    cache = ds.__dict__.setdefault("_dim_stats_cache", {})
    ck = (key, cfg.spike_quantile)
    if ck not in cache:
        keys = [key] + [k for k in KEYS if k != key and (k, cfg.spike_quantile) not in cache]
        for k, st in _dataset_stats(ds, keys, cfg).items():
            cache[(k, cfg.spike_quantile)] = st
    st = cache[ck]
    if st is None or not (cfg.gripper_dims or cfg.state_gripper_dims):
        return st
    gk = (key, cfg.spike_quantile, cfg.gripper_dims, cfg.state_gripper_dims)
    if gk not in cache:
        mask = configured_grippers(ds, cfg).get(key)
        cache[gk] = st if mask is None or len(mask) != len(st.rng) else replace(st, grippers=mask)
    return cache[gk]


DISCRETE_SAMPLE = 200_000  # rows looked at to decide whether a dimension only takes a handful of values


class _StatsAcc:
    """Streaming statistics of one key's finite rows, fed one episode at a time."""

    def __init__(self, cfg: Config, max_rows: int, exact_rows: int | None) -> None:
        self.cfg, self.max_rows, self.exact_rows = cfg, max_rows, exact_rows
        self.d: int | None = None
        self.n = 0

    def start(self, d: int) -> None:
        self.d = d
        self.lo = np.full(d, np.inf)
        self.hi = np.full(d, -np.inf)
        self.values = QuantileStream([0.005, 0.995], self.max_rows, exact_rows=self.exact_rows)
        # Quantile over every frame: in held/stepped signals almost all residuals are zero, and a
        # quantile over only the non-zero ones would let a lone glitch set its own threshold.
        self.resid = QuantileStream([self.cfg.spike_quantile], self.max_rows, exact_rows=self.exact_rows)
        self.discrete: np.ndarray | None = None

    def add(self, fin: np.ndarray) -> None:
        self.n += len(fin)
        if len(fin):
            self.lo = np.minimum(self.lo, fin.min(axis=0))
            self.hi = np.maximum(self.hi, fin.max(axis=0))
        self.values.add(fin)
        self.resid.add(outlier_residual(fin))

    def end_first(self) -> bool:
        """True if a second pass over the data is needed."""
        if self.n == 0:
            return False
        assert self.d is not None
        if self.values.exact:
            allv = np.concatenate(self.values._chunks)
            sample = allv[:: max(1, len(allv) // DISCRETE_SAMPLE)]
            self.discrete = np.array([len(np.unique(sample[:, j])) <= 4 for j in range(self.d)], dtype=bool)
        else:
            # Same rows as above (every k-th finite row), gathered on the second pass.
            self._k = max(1, self.n // DISCRETE_SAMPLE)
            self._pos = 0
            self._uniq: list[set[float]] = [set() for _ in range(self.d)]
        self._v_more = self.values.needs_second_pass()
        self._r_more = self.resid.needs_second_pass()
        return self._v_more or self._r_more or self.discrete is None

    def add_second(self, fin: np.ndarray) -> None:
        if self.discrete is None:
            rows = fin[np.arange((-self._pos) % self._k, len(fin), self._k)]
            self._pos += len(fin)
            for j, u in enumerate(self._uniq):
                if len(u) <= 4 and len(rows):
                    u.update(np.unique(rows[:, j]).tolist())
        if self._v_more:
            self.values.add_second(fin)
        if self._r_more:
            self.resid.add_second(outlier_residual(fin))

    def end_second(self) -> bool:
        """True if done; False if the second pass must run again (a quantile band was too narrow)."""
        if self.discrete is None:
            self.discrete = np.array([len(u) <= 4 for u in self._uniq], dtype=bool)
        if self._v_more:
            self._v_more = self.values.result() is None
        if self._r_more:
            self._r_more = self.resid.result() is None
        return not (self._v_more or self._r_more)

    def stats(self, names: list[str]) -> DimStats | None:
        if self.n == 0 or self.d is None:
            return None
        q = self.values.result()
        r = self.resid.result()
        assert q is not None and r is not None and self.discrete is not None
        if len(names) != self.d:
            names = []
        return DimStats(lo=self.lo, hi=self.hi, rng=q[1] - q[0], step_scale=r[0], names=names, discrete=self.discrete)


def _finite_rows(ds: Dataset, keys: list[str], accs: dict[str, _StatsAcc]) -> Iterator[tuple[str, np.ndarray]]:
    for ep in ds.episodes:
        if not ep.length:
            continue
        for key in keys:
            if key not in ep.frames:
                continue
            a = _as2d(ep.frames[key])
            acc = accs[key]
            if acc.d is None:
                acc.start(a.shape[1])
            if a.shape[1] != acc.d:
                continue
            yield key, a[np.all(np.isfinite(a), axis=1)].astype(np.float64)


def _dataset_stats(
    ds: Dataset, keys: list[str], cfg: Config, exact_rows: int | None = None
) -> dict[str, DimStats | None]:
    """Statistics of several keys in shared passes over the data: one pass when the dataset is
    small, two (rarely three) when it is large, holding only a sample of it in memory."""
    max_rows = ds.total_frames
    accs = {k: _StatsAcc(cfg, max_rows, exact_rows) for k in keys}
    for key, fin in _finite_rows(ds, keys, accs):
        accs[key].add(fin)
    todo = [k for k in keys if accs[k].end_first()]
    while todo:
        for key, fin in _finite_rows(ds, todo, accs):
            accs[key].add_second(fin)
        todo = [k for k in todo if not accs[k].end_second()]
    out = {}
    for key, acc in accs.items():
        names = ds.feature_names(key)
        if acc.d is not None and len(names) != acc.d:
            names = [f"{key}[{i}]" for i in range(acc.d)]
        out[key] = acc.stats(names)
    return out


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
    grasp_eps: list[int] = []  # positions in ds.episodes of healthy episodes, for the grasp check

    for pos, ep in enumerate(ds.episodes):
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
                        grasp_eps.append(pos)
                    if a_st.names == s_st.names:
                        lead, tail = idle_bounds(a, a_st, cfg)
                        # Resting on a hard stop before/after a take is normal; judge only the active part.
                        if lead < len(a) - tail:
                            yield from _limits(
                                ep.index, s[lead : len(a) - tail], a[lead : len(a) - tail], s_st, fps, cfg
                            )

    if a_st is not None and s_st is not None:
        yield from _grasps(ds, grasp_eps, a_st, s_st, fps, cfg)


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
    ia = np.flatnonzero(a_st.gripper_mask()).tolist()
    i_s = np.flatnonzero(s_st.gripper_mask()).tolist()
    if len(ia) != len(i_s):
        return []
    if not a_st.names or not s_st.names:
        return list(zip(ia, i_s, strict=True))
    a_names, s_names = [a_st.names[i] for i in ia], [s_st.names[i] for i in i_s]
    if sorted(a_names) == sorted(s_names):  # same names on both sides: pair by name, not by position
        return [(ja, s_st.names.index(a_st.names[ja])) for ja in ia]
    return list(zip(ia, i_s, strict=True))


def _closed_and_reopened(cmd: np.ndarray, lo: float, hi: float, side: float, depth: float, min_len: int) -> bool:
    """True if the command moved toward `side` by at least `depth` of its range, stayed there for
    `min_len` frames, and moved back again (a close-and-release cycle, however gentle)."""
    if len(cmd) < min_len:
        return False
    q = (cmd - lo) / (hi - lo)
    q = q if side > 0 else 1.0 - q
    bump = np.minimum(q - np.minimum.accumulate(q), q - np.minimum.accumulate(q[::-1])[::-1])
    return any(e - b >= min_len for b, e in _runs(bump >= depth))


def _grasps(ds: Dataset, eps: list[int], a_st: DimStats, s_st: DimStats, fps: float, cfg: Config) -> Iterator[Finding]:
    """Grasps that closed on nothing.

    When a gripper closes on an object, the object blocks it and the follower stops partway.
    When it closes on air, it travels to its fully closed position. For each side the gripper
    is commanded to and held at, collect where the follower settles in every episode: if it
    usually stops short of the farthest settling point (the dataset holds objects), a settle
    at that farthest point is a grasp on nothing. An episode whose last grasp missed is very
    likely a failed demonstration; one that missed and then grasped is a recovery.

    `eps` are positions in ds.episodes. Episodes are read twice, gripper columns only, so the
    check needs no copy of the whole dataset in memory.
    """
    pairs = _gripper_pairs(a_st, s_st)
    if not pairs or len(eps) < 5:
        return
    min_len = max(3, int(cfg.grasp_hold_seconds * fps))
    pairs = [(ja, js) for ja, js in pairs if a_st.hi[ja] - a_st.lo[ja] > 1e-9 and s_st.rng[js] > 1e-9]
    if not pairs:
        return

    def load(pos: int) -> tuple[int, np.ndarray, np.ndarray]:
        ep = ds.episodes[pos]
        a = _as2d(ep.frames["action"]).astype(np.float64)
        s = _as2d(ep.frames["observation.state"]).astype(np.float64)
        return ep.index, a, s

    # Pass 1: does the follower's reading rise or fall when the command rises, and the held segments.
    # Held segments per side: (episode, start, end, settle position, follower minus command, lead?).
    # "lead" = held from the first frame: the rest pose before the gripper has moved, never a grasp.
    moments = [Moments() for _ in pairs]
    segs: list[dict[float, list[tuple[int, int, int, float, float, bool]]]] = [{1.0: [], -1.0: []} for _ in pairs]
    lengths: dict[int, int] = {}
    for pos in eps:
        ep, a, s = load(pos)
        lengths[ep] = len(a)
        for p, (ja, js) in enumerate(pairs):
            moments[p].add(a[:, ja], s[:, js])
            c = (a[:, ja] - a_st.lo[ja]) / (a_st.hi[ja] - a_st.lo[ja])
            for side, held in ((1.0, c > 0.8), (-1.0, c < 0.2)):
                for b, e in _runs(held):
                    if e - b >= min_len:
                        tail = slice(b + (e - b) // 2, e)
                        med_s = float(np.median(s[tail, js]))
                        segs[p][side].append((ep, b, e, med_s, float(np.median(s[tail, js] - a[tail, ja])), b == 0))

    # Which (gripper, side) combinations can hold grasps, and their calibration offset.
    plans: dict[tuple[int, float], tuple[float, float]] = {}  # -> (direction, offset)
    for p in range(len(pairs)):
        m = moments[p]
        if m.std_x() <= 1e-12 or m.std_y() <= 1e-12:
            continue
        sign = 1.0 if m.corr() >= 0 else -1.0
        for side in (1.0, -1.0):
            if sum(1 for g in segs[p][side] if not g[5]) < 5:
                continue
            free = [g[4] for g in segs[p][-side]]
            plans[(p, side)] = (side * sign, float(np.median(free)) if free else 0.0)
    if not plans:
        return

    # Pass 2: a hold that runs to the end of the episode on nothing is the gripper put back to rest
    # when the task is over, not a grasp attempt, if the episode already held an object or already
    # closed and reopened (a pick and release, even one too gentle to show as a held segment).
    rest: set[tuple[int, float, int, int]] = set()  # (pair, side, episode, segment start) to ignore
    ends_by_ep: dict[tuple[int, float], dict[int, list[tuple[int, int, int, float, float, bool]]]] = {}
    for (p, side), _ in plans.items():
        js = pairs[p][1]
        by_ep: dict[int, list] = {}
        for g in segs[p][side]:
            if (
                not g[5]
                and g[2] == lengths[g[0]]
                and abs(g[4] - plans[(p, side)][1]) < cfg.grasp_object_min * s_st.rng[js]
            ):
                by_ep.setdefault(g[0], []).append(g)
        ends_by_ep[(p, side)] = by_ep
    if any(ends_by_ep.values()):
        wanted = {ep for by_ep in ends_by_ep.values() for ep in by_ep}
        for pos in eps:
            if ds.episodes[pos].index not in wanted:
                continue
            ep, a, s = load(pos)
            for (p, side), (d, offset) in plans.items():
                cands = ends_by_ep[(p, side)].get(ep)
                if not cands:
                    continue
                ja, js = pairs[p]
                # Frames where an object holds the follower short of the command (calibration offset
                # removed), whatever the command level: operators often grasp without squeezing the
                # leader all the way.
                blocked = d * (offset - (s[:, js] - a[:, ja])) >= cfg.grasp_object_min * s_st.rng[js]
                held_obj = [(b, e) for b, e in _runs(blocked) if e - b >= min_len]
                for g in cands:
                    if any(e <= g[1] for _, e in held_obj) or _closed_and_reopened(
                        a[: g[1], ja], a_st.lo[ja], a_st.hi[ja], side, cfg.grasp_object_min, min_len
                    ):
                        rest.add((p, side, g[0], g[1]))

    for p, (_, js) in enumerate(pairs):
        s_rng = s_st.rng[js]
        for side in (1.0, -1.0):
            if (p, side) not in plans:
                continue
            d, offset = plans[(p, side)]
            grasps = [g for g in segs[p][side] if not g[5] and (p, side, g[0], g[1]) not in rest]
            if len(grasps) < 5:
                continue
            farthest = float(np.max([d * g[3] for g in grasps]))
            shortfall = (farthest - d * np.array([g[3] for g in grasps])) / s_rng
            if np.median(shortfall) < cfg.grasp_object_min:
                continue  # this side is free motion (opening, or closing on nothing by design)
            # A gripper closing on nothing reaches the operator's command, offset only by the
            # leader/follower calibration seen when it moves freely to the other side; an object
            # holds it short of the command. This also keeps a smaller object from being taken for air.
            if np.median([abs(g[4] - offset) for g in grasps]) / s_rng < cfg.grasp_object_min:
                # Settle points vary but the follower follows the command: the operator just opens (or
                # closes) by different amounts, and nothing is held. Not a grasping side.
                continue
            by_ep: dict[int, list[bool]] = {}
            for g, short in zip(grasps, shortfall, strict=True):
                by_ep.setdefault(g[0], []).append(
                    bool(short < cfg.grasp_air_tol and abs(g[4] - offset) < cfg.grasp_air_tol * s_rng)
                )
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
